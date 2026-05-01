"""Build the qualitative figure grid from a directory of inferred overlays.

Usage:
    python scripts/make_figures.py --input outputs/eval_overlays --out assets/qualitative.png --cols 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path, help="folder of overlay PNGs")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--cell", type=int, default=256)
    args = p.parse_args()

    files = sorted(args.input.glob("*.png"))
    if not files:
        raise SystemExit(f"no PNGs in {args.input}")
    images = [cv2.imread(str(f)) for f in files]
    images = [cv2.resize(im, (args.cell, args.cell)) for im in images if im is not None]
    rows = (len(images) + args.cols - 1) // args.cols
    canvas = np.zeros((rows * args.cell, args.cols * args.cell, 3), dtype=np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, args.cols)
        canvas[r * args.cell : (r + 1) * args.cell, c * args.cell : (c + 1) * args.cell] = im
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), canvas)
    print(f"✓ wrote {args.out} ({len(images)} cells, {rows}x{args.cols})")


if __name__ == "__main__":
    main()
