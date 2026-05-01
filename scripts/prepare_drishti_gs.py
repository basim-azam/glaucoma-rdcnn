"""Convert the official DRISHTI-GS archive layout to the repo's raw schema.

Input: a directory containing the extracted ``Drishti-GS1_files/`` distribution.
Output: ``data/drishti_gs/raw/<stem>{,_od,_oc}.png`` triples that
``scripts/preprocess.py`` can consume.

Handles the typical IIIT-H release layout (Training/Test split with soft
probability maps under ``GT/<stem>/SoftMap/``) and falls back to a glob-based
search if the structure differs slightly.

Usage:
    python scripts/prepare_drishti_gs.py --source-dir <path-to-extracted-folder>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

OD_SUFFIXES = ("_ODsegSoftmap.png", "_OD.png", "-OD.png", "_disc.png")
OC_SUFFIXES = ("_cupsegSoftmap.png", "_Cup.png", "-Cup.png", "_cup.png")
SOFTMAP_THRESHOLD = 127  # binarize softmaps at midpoint


def _binarize(mask: np.ndarray) -> np.ndarray:
    """Soft probability map (0..255) -> {0, 255} binary."""
    return ((mask > SOFTMAP_THRESHOLD).astype(np.uint8)) * 255


def _find_first(stem_dir: Path, suffixes: tuple[str, ...]) -> Path | None:
    """Find the first existing GT file matching any of the suffixes."""
    for sub in (stem_dir, stem_dir / "SoftMap", stem_dir / "AvgBoundary"):
        if not sub.is_dir():
            continue
        for s in suffixes:
            for p in sub.rglob(f"*{s}"):
                return p
    return None


def discover_samples(root: Path) -> list[tuple[str, Path, Path, Path]]:
    """Walk the extracted DRISHTI dir and yield (stem, image, od_mask, oc_mask)."""
    samples: list[tuple[str, Path, Path, Path]] = []
    for split in ("Training", "Test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        images_dir = split_dir / "Images"
        gt_dir = split_dir / "GT"
        if not images_dir.is_dir() or not gt_dir.is_dir():
            continue
        for img_path in sorted(images_dir.glob("*.png")):
            stem = img_path.stem
            stem_gt_dir = gt_dir / stem
            if not stem_gt_dir.is_dir():
                # Some mirrors flatten GT/<stem>/ into GT/
                stem_gt_dir = gt_dir
            od = _find_first(stem_gt_dir, OD_SUFFIXES)
            oc = _find_first(stem_gt_dir, OC_SUFFIXES)
            if od is None or oc is None:
                print(f"  skip {stem}: missing OD or OC mask", file=sys.stderr)
                continue
            samples.append((stem, img_path, od, oc))
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Directory containing the extracted Drishti-GS1_files/ tree",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/drishti_gs/raw"),
        help="Where to write the <stem>{,_od,_oc}.png triples",
    )
    args = ap.parse_args()

    src = args.source_dir
    if not src.is_dir():
        print(f"!! source not found: {src}", file=sys.stderr)
        return 2

    # Some users extract one level too deep / too shallow — try to find the right root
    candidates = [src, src / "Drishti-GS1_files", src / "Drishti-GS1_files" / "Drishti-GS1_files"]
    root = next((c for c in candidates if (c / "Training").is_dir()), None)
    if root is None:
        print(
            f"!! could not locate Training/ subdir under {src}. Looked in: "
            + ", ".join(str(c) for c in candidates),
            file=sys.stderr,
        )
        return 3

    samples = discover_samples(root)
    if not samples:
        print(f"!! found no usable samples under {root}", file=sys.stderr)
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
    print("Next: python scripts/preprocess.py --config-name=data/drishti_gs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
