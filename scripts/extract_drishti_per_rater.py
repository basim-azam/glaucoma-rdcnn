"""Extract per-rater DRISHTI-GS optic-disc and optic-cup masks.

DRISHTI-GS provides four expert annotations per training image. The default
adapter (scripts/prepare_drishti_gs.py) extracts only the SoftMap average,
discarding the per-rater disagreement information.

This script walks the original release and emits per-rater binary masks to:

    data/drishti_gs/masks/od_rater{1,2,3,4}/<stem>.png
    data/drishti_gs/masks/oc_rater{1,2,3,4}/<stem>.png

so the heteroscedastic multi-rater trainer (src/glaucoma_rdcnn_v2/multi_rater.py)
can compute the per-pixel rater-agreement target.

Handles common DRISHTI archive layouts:

    Layout A (official):
      Drishti-GS1_files/
        Training/Drishti-GS1_files/Training/
          <stem>/<stem>_<rater_idx>_<task>.png
            where <task> in {cdrSegSoftmap, ODsegSoftmap, OCsegSoftmap}
            (per-rater files exist for some splits)

    Layout B (per-rater subdirectories):
      Drishti-GS1_files/
        Training/<stem>/Expert{1,2,3,4}/
          <stem>_*_disc.png
          <stem>_*_cup.png

    Layout C (alternate per-rater naming):
      Drishti-GS1_files/
        Training/<stem>/rater_{1,2,3,4}/
          OD.png  OC.png

Usage:
    python scripts/extract_drishti_per_rater.py --source-dir <path-to-archive>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np


def _binarise(m: np.ndarray) -> np.ndarray:
    return ((m > 127).astype(np.uint8)) * 255


def _find_root(src: Path) -> Path:
    """Find the directory containing the per-image stem folders."""
    candidates: list[Path] = [src]
    candidates += [p for p in src.rglob("*") if p.is_dir()][:500]
    for c in candidates:
        # Look for a directory that contains stem-named subdirs like drishtiGS_001
        stems = [p for p in c.iterdir() if p.is_dir() and re.match(r"^drishtiGS_\d+$", p.name, re.IGNORECASE)]
        if len(stems) >= 5:
            return c
    raise FileNotFoundError(f"Could not detect DRISHTI per-image layout under {src}")


def _try_layout_b_per_rater(stem_dir: Path, rater_idx: int) -> tuple[Path | None, Path | None]:
    """Layout B: stem_dir/Expert{i}/ with OD and OC mask files."""
    rater_dir = stem_dir / f"Expert{rater_idx}"
    if not rater_dir.is_dir():
        rater_dir = stem_dir / f"expert{rater_idx}"
    if not rater_dir.is_dir():
        return None, None

    od_candidates = (
        list(rater_dir.glob(f"*OD*.png")) +
        list(rater_dir.glob(f"*disc*.png")) +
        list(rater_dir.glob(f"*Disc*.png")) +
        list(rater_dir.glob(f"*ODsegSoftmap*.png"))
    )
    oc_candidates = (
        list(rater_dir.glob(f"*OC*.png")) +
        list(rater_dir.glob(f"*cup*.png")) +
        list(rater_dir.glob(f"*Cup*.png")) +
        list(rater_dir.glob(f"*OCsegSoftmap*.png"))
    )
    od = od_candidates[0] if od_candidates else None
    oc = oc_candidates[0] if oc_candidates else None
    return od, oc


def _try_layout_a_inline(stem_dir: Path, stem: str, rater_idx: int) -> tuple[Path | None, Path | None]:
    """Layout A: per-rater files in the stem directory itself, named with rater idx."""
    od_candidates = (
        list(stem_dir.rglob(f"*{stem}*{rater_idx}*OD*.png")) +
        list(stem_dir.rglob(f"*{stem}*expert{rater_idx}*disc*.png")) +
        list(stem_dir.rglob(f"*{stem}*rater{rater_idx}*disc*.png"))
    )
    oc_candidates = (
        list(stem_dir.rglob(f"*{stem}*{rater_idx}*OC*.png")) +
        list(stem_dir.rglob(f"*{stem}*expert{rater_idx}*cup*.png")) +
        list(stem_dir.rglob(f"*{stem}*rater{rater_idx}*cup*.png"))
    )
    od = od_candidates[0] if od_candidates else None
    oc = oc_candidates[0] if oc_candidates else None
    return od, oc


def _try_layout_c_per_rater(stem_dir: Path, rater_idx: int) -> tuple[Path | None, Path | None]:
    """Layout C: stem_dir/rater_{i}/OD.png + OC.png."""
    for prefix in (f"rater_{rater_idx}", f"rater{rater_idx}"):
        d = stem_dir / prefix
        if d.is_dir():
            od = next((d / "OD.png", d / "od.png", d / "disc.png")[i] for i in range(3) if (d / ["OD.png", "od.png", "disc.png"][i]).exists())
            oc = next((d / "OC.png", d / "oc.png", d / "cup.png")[i] for i in range(3) if (d / ["OC.png", "oc.png", "cup.png"][i]).exists())
            return od, oc
    return None, None


def find_rater_masks(stem_dir: Path, stem: str, rater_idx: int) -> tuple[Path | None, Path | None]:
    """Try each layout in turn."""
    od, oc = _try_layout_b_per_rater(stem_dir, rater_idx)
    if od and oc:
        return od, oc
    od, oc = _try_layout_a_inline(stem_dir, stem, rater_idx)
    if od and oc:
        return od, oc
    try:
        od, oc = _try_layout_c_per_rater(stem_dir, rater_idx)
        if od and oc:
            return od, oc
    except StopIteration:
        pass
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True, type=Path,
                    help="Path to the extracted Drishti-GS1_files directory")
    ap.add_argument("--out", type=Path, default=Path("data/drishti_gs/masks"))
    ap.add_argument("--target-size", type=int, default=800,
                    help="Mask side length (matches existing preprocess output)")
    args = ap.parse_args()

    try:
        root = _find_root(args.source_dir)
    except FileNotFoundError as e:
        print(f"!! {e}", file=sys.stderr)
        return 2

    print(f"Detected DRISHTI per-image layout at: {root}")

    # Make rater output dirs
    for k in ("od", "oc"):
        for r in range(1, 5):
            (args.out / f"{k}_rater{r}").mkdir(parents=True, exist_ok=True)

    stem_dirs = sorted(p for p in root.iterdir() if p.is_dir() and re.match(r"^drishtiGS_\d+$", p.name, re.IGNORECASE))
    print(f"Found {len(stem_dirs)} stem directories")

    written = {f"{k}_rater{r}": 0 for k in ("od", "oc") for r in range(1, 5)}
    missing_any: list[str] = []

    for stem_dir in stem_dirs:
        stem = stem_dir.name
        for r in range(1, 5):
            od, oc = find_rater_masks(stem_dir, stem, r)
            if od is None and oc is None:
                missing_any.append(f"{stem} rater{r}")
                continue
            if od is not None:
                m = cv2.imread(str(od), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    m = cv2.resize(m, (args.target_size, args.target_size), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(str(args.out / f"od_rater{r}" / f"{stem}.png"), _binarise(m))
                    written[f"od_rater{r}"] += 1
            if oc is not None:
                m = cv2.imread(str(oc), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    m = cv2.resize(m, (args.target_size, args.target_size), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(str(args.out / f"oc_rater{r}" / f"{stem}.png"), _binarise(m))
                    written[f"oc_rater{r}"] += 1

    print()
    print("=== summary ===")
    for k, v in written.items():
        print(f"  {k}: {v} masks written")
    if missing_any:
        print(f"\n  missing: {len(missing_any)} (stem, rater) pairs")
        print(f"  first few: {missing_any[:6]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
