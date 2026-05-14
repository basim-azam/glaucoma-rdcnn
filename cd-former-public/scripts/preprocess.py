"""CLAHE + 800x800 ROI crop.

Reads ``data/<dataset>/raw/...`` (whatever layout the source provides), produces
``data/<dataset>/{images,masks/{od,oc},splits/official.json}`` with consistent
naming. Re-running is idempotent.

PAPER-UNSPEC: train/val/test split protocol — DRISHTI-GS provides "Training"
and "Test" lists; RIM-ONE has no canonical split. We use a deterministic 70/15/15
split keyed by stem-hash for RIM-ONE so it's reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from glaucoma_rdcnn.data.od_localizer import crop_roi, locate_optic_disc

CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
TARGET = 800


def apply_clahe_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB color."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge([l_channel, a, b]), cv2.COLOR_LAB2BGR)


def split_by_hash(stems: list[str]) -> dict[str, list[str]]:
    """Deterministic 70/15/15 split based on md5(stem)."""
    train, val, test = [], [], []
    for s in sorted(stems):
        h = int(hashlib.md5(s.encode()).hexdigest(), 16)
        bucket = h % 100
        if bucket < 70:
            train.append(s)
        elif bucket < 85:
            val.append(s)
        else:
            test.append(s)
    return {"train": train, "val": val, "test": test}


def process_one(
    image_path: Path,
    od_mask_path: Path,
    oc_mask_path: Path,
    out_root: Path,
    stem: str,
) -> bool:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    od = cv2.imread(str(od_mask_path), cv2.IMREAD_GRAYSCALE)
    oc = cv2.imread(str(oc_mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or od is None or oc is None:
        return False
    # Localize OD using the GT centroid (more reliable than the heuristic when GT exists)
    if od.any():
        ys, xs = np.where(od > 0)
        cx, cy = int(xs.mean()), int(ys.mean())
    else:
        cx, cy = locate_optic_disc(img)
    img_eq = apply_clahe_bgr(img)
    img_crop, off = crop_roi(img_eq, (cx, cy), TARGET)
    od_crop, _ = crop_roi(cv2.cvtColor(od, cv2.COLOR_GRAY2BGR), (cx, cy), TARGET)
    oc_crop, _ = crop_roi(cv2.cvtColor(oc, cv2.COLOR_GRAY2BGR), (cx, cy), TARGET)
    od_crop = (cv2.cvtColor(od_crop, cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8) * 255
    oc_crop = (cv2.cvtColor(oc_crop, cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8) * 255

    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "masks" / "od").mkdir(parents=True, exist_ok=True)
    (out_root / "masks" / "oc").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_root / "images" / f"{stem}.png"), img_crop)
    cv2.imwrite(str(out_root / "masks" / "od" / f"{stem}.png"), od_crop)
    cv2.imwrite(str(out_root / "masks" / "oc" / f"{stem}.png"), oc_crop)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True, help="e.g. data/drishti_gs")
    ap.add_argument("--root", default="data", help="dataset root")
    args = ap.parse_args()

    name = args.config_name.split("/")[-1]
    in_root = Path(args.root) / name
    raw = in_root / "raw"
    if not raw.exists():
        print(f"!! raw data missing at {raw} — run scripts/download_data.sh first", file=sys.stderr)
        return 1

    # Best-effort discovery: find triples of (image, od_mask, oc_mask) by stem.
    # Each dataset packages things slightly differently; this scaffold assumes
    # the user has already arranged inputs as <stem>.png + <stem>_od.png + <stem>_oc.png.
    stems: list[str] = []
    for img in sorted(raw.glob("*.png")):
        if img.stem.endswith("_od") or img.stem.endswith("_oc"):
            continue
        od_p = raw / f"{img.stem}_od.png"
        oc_p = raw / f"{img.stem}_oc.png"
        if od_p.exists() and oc_p.exists():
            ok = process_one(img, od_p, oc_p, in_root, img.stem)
            if ok:
                stems.append(img.stem)
    if not stems:
        print(
            "!! no <stem>.png + <stem>_od.png + <stem>_oc.png triples found. "
            "See docs/dataset_schema.md for the expected raw layout.",
            file=sys.stderr,
        )
        return 2

    # Honor an official split sidecar if the dataset adapter wrote one.
    sidecar = in_root / "splits" / "drishti_official.json"
    if sidecar.exists():
        with open(sidecar) as f:
            official = json.load(f)
        # Filter to only stems we actually processed (defensive)
        stem_set = set(stems)
        splits = {k: [s for s in v if s in stem_set] for k, v in official.items()}
        print(f"using official split from {sidecar}")
    else:
        splits = split_by_hash(stems)
        print("using hash-based 70/15/15 split (no official sidecar found)")
    splits_path = in_root / "splits" / "official.json"
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"✓ {name}: {len(stems)} samples → {in_root}")
    print(
        f"  splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
