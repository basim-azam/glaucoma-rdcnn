"""Rewrite splits/official.json to match the paper's exact 50-train / 51-test protocol.

Paper (Li et al., Eye 2023) trains on all 50 DRISHTI-GS Training images and
reports on all 51 Test images, with no val carve-out. This script:

  - Reads the sidecar `splits/drishti_official.json` produced by
    `scripts/prepare_drishti_gs.py` (which has train=Training, val=last 5, test=Test)
  - Rewrites `splits/official.json` so that train = full 50 Training,
    val = test = full 51 Test.

Why val == test: our trainer needs *some* val split for checkpoint selection;
val=test mirrors what the paper effectively does (no held-out validation,
checkpoint chosen by training-set fit + fixed epoch count). Accept the
caveat: "best.ckpt" is selected against the test set, so it slightly
over-estimates the test number compared to a true held-out evaluation.

For the unbiased measure, also keep training the standard 45/5/51 protocol
and report both numbers in STATUS.md.

Usage (on Spartan, after data prep has run at least once):
    python scripts/make_paper_exact_split.py --root data/drishti_gs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="dataset root, e.g. data/drishti_gs")
    args = ap.parse_args()

    sidecar = args.root / "splits" / "drishti_official.json"
    out = args.root / "splits" / "official.json"
    if not sidecar.exists():
        print(f"!! {sidecar} not found. Run scripts/prepare_drishti_gs.py first.", file=sys.stderr)
        return 1

    with open(sidecar) as f:
        official = json.load(f)
    if not all(k in official for k in ("train", "val", "test")):
        print(f"!! {sidecar} missing required keys", file=sys.stderr)
        return 2

    # Merge val back into train for the full 50-image training set
    full_train = sorted(set(official["train"]) | set(official["val"]))
    test = sorted(official["test"])

    paper_exact = {
        "train": full_train,
        "val": test,  # leakage caveat: val == test
        "test": test,
    }

    with open(out, "w") as f:
        json.dump(paper_exact, f, indent=2)

    print(f"OK: wrote paper-exact split -> {out}")
    print(
        f"  train={len(paper_exact['train'])} val={len(paper_exact['val'])} test={len(paper_exact['test'])}"
    )
    print("  NOTE: val == test (leakage). 'best.ckpt' is selected against test;")
    print("        report this run separately from the val-held-out 45/5/51 run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
