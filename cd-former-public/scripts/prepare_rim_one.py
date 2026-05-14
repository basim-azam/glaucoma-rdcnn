"""Convert the RIM-ONE v3 release to the repo's raw schema.

Three layouts supported (auto-detected):

  Layout A: original 'RIM-ONE r3' release (some mirrors)
    RIM-ONE_r3/
      Stereo Images/
      Expert1/Disc/, Expert1/Cup/

  Layout B: 'RIM-ONE DL' repackaged (MIAG-ULL GitHub, 2020 release)
    RIM-ONE_DL_images/
      partitioned_randomly/{training_set,test_set}/{images,reference_segmentations}/

  Layout C: official 'RIM-ONE r3' (2015) — organized by class:
    RIM-ONE r3/
      Healthy/{Stereo Images,Expert1_masks,Expert2_masks,Average_masks}/
      Glaucoma and suspects/{Stereo Images,Expert1_masks,...}/

Per the R-DCNN paper, Expert 1 annotations are used.

Output: ``data/rim_one_v3/raw/<stem>{,_od,_oc}.png`` triples that
``scripts/preprocess.py`` consumes.

Usage:
    python scripts/prepare_rim_one.py --source-dir <path-to-extracted-folder>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _binarize(mask: np.ndarray) -> np.ndarray:
    return ((mask > 127).astype(np.uint8)) * 255


def _find_root(src: Path) -> tuple[Path, str]:
    """Return (root, layout) — 'A', 'B', or 'C'.

    Walks up to ~500 dirs deep to handle WinRAR-style nested extraction.
    """
    candidates: list[Path] = [src]
    candidates += [p for p in src.rglob("*") if p.is_dir()][:500]
    for c in candidates:
        if (c / "Healthy").is_dir() and (c / "Glaucoma and suspects").is_dir():
            return c, "C"
        if (c / "Expert1").is_dir() and (c / "Stereo Images").is_dir():
            return c, "A"
        if (c / "partitioned_randomly").is_dir():
            return c, "B"
        if (c / "Expert1" / "Disc").is_dir():
            return c, "A"
    raise FileNotFoundError(f"Could not detect RIM-ONE layout under {src}")


def discover_layout_a(root: Path) -> list[tuple[str, Path, Path, Path]]:
    images_dir = root / "Stereo Images"
    od_dir = root / "Expert1" / "Disc"
    oc_dir = root / "Expert1" / "Cup"
    if not all(d.is_dir() for d in (images_dir, od_dir, oc_dir)):
        raise FileNotFoundError(f"Layout A missing one of: {images_dir}, {od_dir}, {oc_dir}")

    samples: list[tuple[str, Path, Path, Path]] = []
    for img_path in sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg")):
        stem = img_path.stem
        od = next(
            (
                od_dir / f"{stem}{s}"
                for s in (".png", "-Disc.png", "_OD.png")
                if (od_dir / f"{stem}{s}").exists()
            ),
            None,
        )
        oc = next(
            (
                oc_dir / f"{stem}{s}"
                for s in (".png", "-Cup.png", "_OC.png")
                if (oc_dir / f"{stem}{s}").exists()
            ),
            None,
        )
        if od is None or oc is None:
            print(f"  skip {stem}: missing Expert1 OD or OC mask", file=sys.stderr)
            continue
        samples.append((stem, img_path, od, oc))
    return samples


def discover_layout_b(root: Path) -> list[tuple[str, Path, Path, Path]]:
    """RIM-ONE DL repackaged: training + test partitions with reference_segmentations/."""
    samples: list[tuple[str, Path, Path, Path]] = []
    for split in ("training_set", "test_set"):
        s_dir = root / "partitioned_randomly" / split
        img_dir = s_dir / "images"
        ref_dir = s_dir / "reference_segmentations"
        if not img_dir.is_dir() or not ref_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            disc_mask = ref_dir / f"{stem}.png"
            if disc_mask.exists():
                # Two-class mask: bg=0, OD=128, OC=255
                m = cv2.imread(str(disc_mask), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                od_mask = ((m >= 128).astype(np.uint8)) * 255  # OD includes OC
                oc_mask = ((m >= 200).astype(np.uint8)) * 255  # OC only
                tmp_od = ref_dir / f"_{stem}_od_tmp.png"
                tmp_oc = ref_dir / f"_{stem}_oc_tmp.png"
                cv2.imwrite(str(tmp_od), od_mask)
                cv2.imwrite(str(tmp_oc), oc_mask)
                samples.append((stem, img_path, tmp_od, tmp_oc))
                continue
            od = ref_dir / f"{stem}_disc.png"
            oc = ref_dir / f"{stem}_cup.png"
            if od.exists() and oc.exists():
                samples.append((stem, img_path, od, oc))
    return samples


def discover_layout_c(root: Path) -> list[tuple[str, Path, Path, Path]]:
    """RIM-ONE r3 by-class layout. Uses Expert1 masks per the paper."""
    samples: list[tuple[str, Path, Path, Path]] = []
    for class_dir in ("Healthy", "Glaucoma and suspects"):
        c_dir = root / class_dir
        if not c_dir.is_dir():
            continue
        img_dir = c_dir / "Stereo Images"
        mask_dir = c_dir / "Expert1_masks"
        if not img_dir.is_dir() or not mask_dir.is_dir():
            print(f"  skip {class_dir}: no Stereo Images/ or Expert1_masks/", file=sys.stderr)
            continue
        for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            stem = img_path.stem
            od = mask_dir / f"{stem}-1-Disc-exp1.png"
            oc = mask_dir / f"{stem}-1-Cup-exp1.png"
            if not od.exists() or not oc.exists():
                alt_od = mask_dir / f"{stem}-Disc-exp1.png"
                alt_oc = mask_dir / f"{stem}-Cup-exp1.png"
                if alt_od.exists() and alt_oc.exists():
                    od, oc = alt_od, alt_oc
                else:
                    print(f"  skip {stem}: no Expert1 OD/OC pair", file=sys.stderr)
                    continue
            samples.append((stem, img_path, od, oc))
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Directory containing the extracted RIM-ONE release",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/rim_one_v3/raw"),
    )
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

    if layout == "A":
        samples = discover_layout_a(root)
    elif layout == "B":
        samples = discover_layout_b(root)
    else:  # "C"
        samples = discover_layout_c(root)

    if not samples:
        print(f"!! no usable samples found under {root}", file=sys.stderr)
        return 4

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for stem, img_path, od_path, oc_path in samples:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        od = cv2.imread(str(od_path), cv2.IMREAD_GRAYSCALE)
        oc = cv2.imread(str(oc_path), cv2.IMREAD_GRAYSCALE)
        if img is None or od is None or oc is None:
            print(f"  skip {stem}: failed to read one of the inputs", file=sys.stderr)
            continue
        cv2.imwrite(str(args.out / f"{stem}.png"), img)
        cv2.imwrite(str(args.out / f"{stem}_od.png"), _binarize(od))
        cv2.imwrite(str(args.out / f"{stem}_oc.png"), _binarize(oc))
        written += 1

    print(f"OK: wrote {written} triples to {args.out}")
    print("Next: python scripts/preprocess.py --config-name=data/rim_one_v3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
