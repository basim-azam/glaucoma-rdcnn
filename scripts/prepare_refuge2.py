"""Convert the REFUGE2 release to the repo's raw schema.

REFUGE2 (REtinal FUndus Glaucoma challenGE 2nd edition) was released for the
MICCAI 2020 Endless Glaucoma Challenge. It provides 1200 colour fundus images
(800 training, 400 validation, 400 test depending on the redistributed split)
with optic-disc and optic-cup masks plus glaucoma labels.

The official release ships several layouts depending on the year/mirror; we
support the common ones:

  Layout A:  REFUGE2/
               Training/      Images/  *.jpg
                              Disc_Cup_Masks/  *.bmp
               Validation/    Images/  *.jpg
                              Disc_Cup_Masks/  *.bmp
               Test/          Images/  *.jpg
                              Disc_Cup_Masks/  *.bmp

  Layout B:  REFUGE2/
               train/  images/  *.jpg
                       masks/   *.png    (two-class: bg=255, OD=128, OC=0)

  Layout C:  flat directory of  <stem>.jpg and <stem>_mask.png

The original REFUGE2 masks encode three classes: background=255, OD=128, OC=0
(in 8-bit grayscale). We binarise:
  od_mask = (mask <= 128)        # includes OD interior, which contains the cup
  oc_mask = (mask == 0)          # cup only

Output: data/refuge2/raw/<stem>{,_od,_oc}.png triples that
scripts/preprocess.py consumes.

Usage:
    python scripts/prepare_refuge2.py --source-dir <path/to/REFUGE2>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _binarise(mask: np.ndarray) -> np.ndarray:
    return ((mask > 127).astype(np.uint8)) * 255


def _decode_od_oc_from_three_class(mask_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Decode REFUGE2's three-class mask (bg=255, OD=128, OC=0) into OD/OC binary masks."""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    # OD = OD interior + OC interior = everything that's not background
    od = (m < 200).astype(np.uint8) * 255
    # OC = darkest region only
    oc = (m < 50).astype(np.uint8) * 255
    return od, oc


def _find_root(src: Path) -> tuple[Path, str]:
    """Auto-detect REFUGE2 layout. Walks recursively."""
    candidates: list[Path] = [src]
    candidates += [p for p in src.rglob("*") if p.is_dir()][:300]
    for c in candidates:
        if (c / "Training" / "Images").is_dir() and (c / "Training" / "Disc_Cup_Masks").is_dir():
            return c, "A"
        if (c / "train" / "images").is_dir() and (c / "train" / "masks").is_dir():
            return c, "B"
        # Layout C: any directory with both *.jpg and *_mask.png
        jpgs = list(c.glob("*.jpg")) + list(c.glob("*.png"))
        masks = list(c.glob("*_mask.png"))
        if jpgs and masks and len(jpgs) > 50:
            return c, "C"
    raise FileNotFoundError(f"Could not detect REFUGE2 layout under {src}")


def discover_layout_a(root: Path) -> list[tuple[str, Path, Path, Path]]:
    """Official challenge layout with Training/Validation/Test subdirs."""
    samples: list[tuple[str, Path, Path, Path]] = []
    for split_dir in ("Training", "Validation", "Test"):
        img_dir = root / split_dir / "Images"
        mask_dir = root / split_dir / "Disc_Cup_Masks"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            stem = img_path.stem
            # Mask file may have any extension; try common ones
            for ext in (".bmp", ".png", ".tif"):
                mp = mask_dir / f"{stem}{ext}"
                if mp.exists():
                    samples.append((f"{split_dir}_{stem}", img_path, mp, mp))
                    break
    return samples


def discover_layout_b(root: Path) -> list[tuple[str, Path, Path, Path]]:
    """train/val/test with images/ and masks/ subdirs, flat split."""
    samples: list[tuple[str, Path, Path, Path]] = []
    for split_dir in ("train", "val", "test", "training", "validation"):
        s = root / split_dir
        if not s.is_dir():
            continue
        img_dir = s / "images"
        mask_dir = s / "masks"
        if not (img_dir.is_dir() and mask_dir.is_dir()):
            continue
        for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            stem = img_path.stem
            for ext in (".png", ".bmp", ".tif"):
                mp = mask_dir / f"{stem}{ext}"
                if mp.exists():
                    samples.append((f"{split_dir}_{stem}", img_path, mp, mp))
                    break
    return samples


def discover_layout_c(root: Path) -> list[tuple[str, Path, Path, Path]]:
    """Flat layout: <stem>.jpg + <stem>_mask.png."""
    samples: list[tuple[str, Path, Path, Path]] = []
    for img_path in sorted(root.glob("*.jpg")) + sorted(root.glob("*.png")):
        if img_path.stem.endswith("_mask"):
            continue
        mp = root / f"{img_path.stem}_mask.png"
        if mp.exists():
            samples.append((img_path.stem, img_path, mp, mp))
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/refuge2/raw"))
    args = ap.parse_args()

    if not args.source_dir.is_dir():
        print(f"!! source not found: {args.source_dir}", file=sys.stderr)
        return 2

    try:
        root, layout = _find_root(args.source_dir)
    except FileNotFoundError as e:
        print(f"!! {e}", file=sys.stderr)
        return 3
    print(f"detected layout {layout} at {root}")

    discover_fn = {"A": discover_layout_a, "B": discover_layout_b, "C": discover_layout_c}[layout]
    samples = discover_fn(root)
    if not samples:
        print(f"!! no usable samples found under {root}", file=sys.stderr)
        return 4
    print(f"found {len(samples)} samples")

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for stem, img_path, mask_path, _ in samples:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  skip {stem}: failed to read image", file=sys.stderr)
            continue
        decoded = _decode_od_oc_from_three_class(mask_path)
        if decoded is None:
            print(f"  skip {stem}: failed to read mask", file=sys.stderr)
            continue
        od, oc = decoded
        cv2.imwrite(str(args.out / f"{stem}.png"), img)
        cv2.imwrite(str(args.out / f"{stem}_od.png"), _binarise(od))
        cv2.imwrite(str(args.out / f"{stem}_oc.png"), _binarise(oc))
        written += 1

    print(f"OK: wrote {written} triples to {args.out}")
    print("Next: python scripts/preprocess.py --config-name=data/refuge2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
