"""Convert the SMDG-19 release to the repo's raw schema.

SMDG-19 (Standardized Multi-Channel Dataset for Glaucoma, 19 sources) is a
Kaggle-distributed aggregate of 19 public fundus datasets standardised into a
unified file layout. It contains ~12,000 images with optic-disc and optic-cup
masks plus glaucoma classification labels.

The typical layout (Kaggle "deathtrooper/multichannel-glaucoma-benchmark-dataset"):

    SMDG-19/
      full-fundus/full-fundus/   <stem>.png           (RGB fundus)
      optic-disc/optic-disc/      <stem>.png          (OD mask, white=1)
      optic-cup/optic-cup/        <stem>.png          (OC mask, white=1)
      metadata.csv                                    (per-image source + glaucoma label)

Some mirrors flatten these to:

    SMDG-19/
      full-fundus/   <stem>.png
      optic-disc/    <stem>.png
      optic-cup/     <stem>.png

Output: data/smdg19/raw/<stem>{,_od,_oc}.png triples that
scripts/preprocess.py consumes.

Optionally filter by source dataset (e.g. --include drishti-gs,refuge2) so we
can exclude duplicates of datasets we evaluate separately.

Usage:
    python scripts/prepare_smdg19.py --source-dir <path/to/SMDG-19>
    python scripts/prepare_smdg19.py --source-dir <path/to/SMDG-19> --exclude drishti-gs,rim-one-r3
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def _binarise(mask: np.ndarray) -> np.ndarray:
    return ((mask > 127).astype(np.uint8)) * 255


def _find_root(src: Path) -> Path:
    """Find the SMDG-19 root containing the three standard subdirs."""
    candidates: list[Path] = [src]
    candidates += [p for p in src.rglob("*") if p.is_dir()][:300]
    for c in candidates:
        # Common patterns: nested duplicate or flat
        has_full = (c / "full-fundus").is_dir() or (c / "full-fundus" / "full-fundus").is_dir()
        has_od = (c / "optic-disc").is_dir() or (c / "optic-disc" / "optic-disc").is_dir()
        has_oc = (c / "optic-cup").is_dir() or (c / "optic-cup" / "optic-cup").is_dir()
        if has_full and has_od and has_oc:
            return c
    raise FileNotFoundError(f"Could not detect SMDG-19 layout under {src}")


def _resolve_dir(root: Path, name: str) -> Path:
    nested = root / name / name
    flat = root / name
    return nested if nested.is_dir() else flat


def _load_metadata(root: Path) -> dict[str, dict]:
    """Load metadata.csv if present. Returns {stem: row_dict}."""
    md_paths = [root / "metadata.csv", root / "metadata" / "metadata.csv"]
    md: dict[str, dict] = {}
    for p in md_paths:
        if p.exists():
            with open(p, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 'names' field varies; common: 'fundus' or 'file_name'
                    stem_key = row.get("file_name") or row.get("names") or row.get("fundus")
                    if stem_key:
                        stem = Path(stem_key).stem
                        md[stem] = row
            break
    return md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/smdg19/raw"))
    ap.add_argument("--include", type=str, default="",
                    help="Comma-separated list of source datasets to keep (matches metadata.names prefix). Empty = keep all.")
    ap.add_argument("--exclude", type=str, default="",
                    help="Comma-separated list of source datasets to drop (overrides --include).")
    args = ap.parse_args()

    if not args.source_dir.is_dir():
        print(f"!! source not found: {args.source_dir}", file=sys.stderr)
        return 2

    try:
        root = _find_root(args.source_dir)
    except FileNotFoundError as e:
        print(f"!! {e}", file=sys.stderr)
        return 3
    print(f"SMDG-19 root: {root}")

    img_dir = _resolve_dir(root, "full-fundus")
    od_dir = _resolve_dir(root, "optic-disc")
    oc_dir = _resolve_dir(root, "optic-cup")
    md = _load_metadata(root)
    print(f"image dir: {img_dir}")
    print(f"OD mask dir: {od_dir}")
    print(f"OC mask dir: {oc_dir}")
    print(f"metadata rows: {len(md)}")

    include_set = {s.strip().lower() for s in args.include.split(",") if s.strip()}
    exclude_set = {s.strip().lower() for s in args.exclude.split(",") if s.strip()}

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_filter = 0
    skipped_missing = 0

    for img_path in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        od_p = od_dir / f"{stem}.png"
        oc_p = oc_dir / f"{stem}.png"

        # Apply source-dataset filter via metadata.names
        if md and (include_set or exclude_set):
            row = md.get(stem, {})
            src_name = (row.get("names") or row.get("dataset") or "").lower()
            if include_set and not any(src_name.startswith(s) for s in include_set):
                skipped_filter += 1
                continue
            if exclude_set and any(src_name.startswith(s) for s in exclude_set):
                skipped_filter += 1
                continue

        if not (od_p.exists() and oc_p.exists()):
            skipped_missing += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        od = cv2.imread(str(od_p), cv2.IMREAD_GRAYSCALE)
        oc = cv2.imread(str(oc_p), cv2.IMREAD_GRAYSCALE)
        if img is None or od is None or oc is None:
            skipped_missing += 1
            continue

        cv2.imwrite(str(args.out / f"{stem}.png"), img)
        cv2.imwrite(str(args.out / f"{stem}_od.png"), _binarise(od))
        cv2.imwrite(str(args.out / f"{stem}_oc.png"), _binarise(oc))
        written += 1

    print(f"OK: wrote {written} triples to {args.out}")
    if skipped_filter:
        print(f"  filtered out by --include/--exclude: {skipped_filter}")
    if skipped_missing:
        print(f"  skipped (missing files): {skipped_missing}")
    print("Next: python scripts/preprocess.py --config-name=data/smdg19")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
