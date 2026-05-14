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


def _find_gt_dir(split_dir: Path) -> Path | None:
    """Locate the GT directory within a split dir.

    DRISHTI-GS uses inconsistent names: Training/GT/, Test/Test_GT/, sometimes
    Training/Train_GT/. Match any subdirectory whose name contains 'GT' case-
    insensitively.
    """
    if not split_dir.is_dir():
        return None
    for child in sorted(split_dir.iterdir()):
        if child.is_dir() and "gt" in child.name.lower():
            return child
    return None


def discover_samples(root: Path) -> list[tuple[str, Path, Path, Path, str]]:
    """Walk the extracted DRISHTI dir and yield (stem, image, od_mask, oc_mask, split)."""
    samples: list[tuple[str, Path, Path, Path, str]] = []
    for split in ("Training", "Test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        images_dir = split_dir / "Images"
        gt_dir = _find_gt_dir(split_dir)
        if not images_dir.is_dir() or gt_dir is None:
            print(f"  skip {split}: no Images/ or GT-like dir", file=sys.stderr)
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
            samples.append((stem, img_path, od, oc, split))
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

    # Locate the directory that directly contains Training/. WinRAR "Extract
    # Here" + a same-named root inside the archive can produce arbitrarily deep
    # nesting like Drishti-GS1_files/Drishti-GS1_files/Drishti-GS1_files/Training.
    # Walk the tree (capped depth) to find it.
    root = None
    for candidate in [src, *(p.parent for p in src.rglob("Training") if p.is_dir())]:
        if (candidate / "Training").is_dir():
            root = candidate
            break
    if root is None:
        print(
            f"!! could not locate a Training/ subdir anywhere under {src}",
            file=sys.stderr,
        )
        return 3
    print(f"using root: {root}")

    samples = discover_samples(root)
    if not samples:
        print(f"!! found no usable samples under {root}", file=sys.stderr)
        return 4

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    train_stems: list[str] = []
    test_stems: list[str] = []
    for stem, img_path, od_path, oc_path, split in samples:
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
        (train_stems if split == "Training" else test_stems).append(stem)

    # Write the official Training/Test split as a sidecar so preprocess.py can
    # use it instead of the hash-based 70/15/15 fallback. Held-out 10% of
    # Training as val so we always have a val split for checkpoint selection.
    if train_stems and test_stems:
        n_val = max(1, len(train_stems) // 10)
        # Use stable order: sort then take last N as val
        train_sorted = sorted(train_stems)
        official = {
            "train": train_sorted[:-n_val],
            "val": train_sorted[-n_val:],
            "test": sorted(test_stems),
        }
        official_path = args.out.parent / "splits" / "drishti_official.json"
        official_path.parent.mkdir(parents=True, exist_ok=True)
        import json as _json

        official_path.write_text(_json.dumps(official, indent=2))
        print(f"OK: wrote official split sidecar -> {official_path}")
        print(
            f"  train={len(official['train'])} val={len(official['val'])} "
            f"test={len(official['test'])}"
        )

    print(f"OK: wrote {written} triples to {args.out}")
    print("Next: python scripts/preprocess.py --config-name=data/drishti_gs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
