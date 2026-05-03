"""Convert the RIM-ONE v3 release to the repo's raw schema.

RIM-ONE v3 (MIAG-ULL group) ships in a few common layouts depending on which
mirror you got it from. This adapter handles the two most common:

  Layout A (original release, 'RIM-ONE r3'):
    RIM-ONE_r3/
      Stereo Images/             # full stereo pair PNGs
      Expert1/
        Disc/                    # binary OD masks
        Cup/                     # binary OC masks
      Expert2/                   # often present, optional
        ...

  Layout B (repackaged 'RIM-ONE DL' release on the MIAG-ULL GitHub):
    RIM-ONE_DL_images/
      partitioned_randomly/
        training_set/{images,reference_segmentations}/
        test_set/{images,reference_segmentations}/

Per the R-DCNN paper, Expert 1 annotations are used. Layout A is detected by
the presence of Expert1/. Layout B is detected by partitioned_randomly/.

Output: ``data/rim_one_v3/raw/<stem>{,_od,_oc}.png`` triples that
``scripts/preprocess.py`` understands.

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
    """Return (root, layout) where layout is 'A' (original) or 'B' (DL release).

    Walks up to 3 levels deep to handle WinRAR-style nested extraction.
    """
    candidates: list[Path] = [src]
    candidates += [p for p in src.rglob("*") if p.is_dir()][:200]  # bound the search
    for c in candidates:
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
        # OD/OC masks may have suffix variants; try a few
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
            # DL release names refs as <stem>.png with the disc/cup encoded as
            # different intensity levels in a single mask, or as two separate files.
            # Try the two common patterns.
            disc_mask = ref_dir / f"{stem}.png"
            if disc_mask.exists():
                # Two-class mask: 0 = bg, 128 = OD-only, 255 = OC
                m = cv2.imread(str(disc_mask), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                od_mask = ((m >= 128).astype(np.uint8)) * 255  # OD includes OC
                oc_mask = ((m >= 200).astype(np.uint8)) * 255  # OC only
                # Write to temp files in raw/ later; here we just stash arrays
                # by writing directly in main(). Trick: encode in path tuple
                # using a sentinel that main() recognises.
                # Simpler: write the temp images here and return their paths.
                tmp_od = ref_dir / f"_{stem}_od_tmp.png"
                tmp_oc = ref_dir / f"_{stem}_oc_tmp.png"
                cv2.imwrite(str(tmp_od), od_mask)
                cv2.imwrite(str(tmp_oc), oc_mask)
                samples.append((stem, img_path, tmp_od, tmp_oc))
                continue
            # Two-file pattern
            od = ref_dir / f"{stem}_disc.png"
            oc = ref_dir / f"{stem}_cup.png"
            if od.exists() and oc.exists():
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
        help="Where to write the <stem>{,_od,_oc}.png triples",
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

    samples = discover_layout_a(root) if layout == "A" else discover_layout_b(root)

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
