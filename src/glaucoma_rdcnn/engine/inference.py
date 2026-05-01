"""Single-image inference helper + CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from glaucoma_rdcnn.data.od_localizer import crop_roi, locate_optic_disc
from glaucoma_rdcnn.data.transforms import normalize_image
from glaucoma_rdcnn.geometry import boxes_to_masks, compute_cdr
from glaucoma_rdcnn.utils.checkpoint import load_checkpoint
from glaucoma_rdcnn.utils.logging import get_logger
from glaucoma_rdcnn.utils.viz import save_overlay

LOG = get_logger("inference")


def infer_image(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    *,
    image_size: int = 800,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run the full pipeline (CLAHE → ROI crop → R-DCNN → ellipse → CDR) on one image."""
    if image_bgr.ndim != 3:
        raise ValueError("expected an HxWx3 BGR image")
    cx, cy = locate_optic_disc(image_bgr)
    crop_bgr, _offset = crop_roi(image_bgr, (cx, cy), image_size)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    inp = normalize_image(rgb, mean, std).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs, _ = model(inp)
    out = outputs[0]  # type: ignore[index]
    od_mask, oc_mask = boxes_to_masks(out["od_boxes"], out["oc_boxes"], (image_size, image_size))
    cdr = float("nan")
    if out["od_boxes"].numel() and out["oc_boxes"].numel():
        cdr = compute_cdr(out["od_boxes"][0], out["oc_boxes"][0])
    return {
        "od_mask": od_mask,
        "oc_mask": oc_mask,
        "od_boxes": out["od_boxes"].cpu().numpy(),
        "oc_boxes": out["oc_boxes"].cpu().numpy(),
        "cdr": cdr,
        "rgb_crop": rgb,
    }


def cli_main() -> None:
    p = argparse.ArgumentParser(description="Run R-DCNN on a single fundus image.")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("infer_out.png"))
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    from omegaconf import OmegaConf

    from glaucoma_rdcnn.models import build_rdcnn

    # Use the default rdcnn config — no Hydra needed for a one-shot invocation
    cfg = OmegaConf.load(Path(__file__).resolve().parents[3] / "configs" / "model" / "rdcnn.yaml")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = build_rdcnn(cfg).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    out = infer_image(model, img, device=device)
    LOG.info(f"CDR = {out['cdr']:.3f}")
    save_overlay(args.out, out["rgb_crop"], out["od_mask"], out["oc_mask"])
    LOG.info(f"wrote overlay to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    cli_main()
