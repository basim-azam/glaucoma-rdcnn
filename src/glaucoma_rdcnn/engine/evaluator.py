"""Evaluator: predicts ellipses from boxes, computes Dice/JC/E/SE/SP, plus AUC if labels present."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from glaucoma_rdcnn.geometry import boxes_to_masks, compute_cdr
from glaucoma_rdcnn.metrics import (
    aggregate,
    dice,
    glaucoma_auc,
    jaccard,
    overlap_error,
    sensitivity,
    specificity,
)
from glaucoma_rdcnn.utils.logging import get_logger

LOG = get_logger("evaluator")


class Evaluator:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        *,
        device: str | torch.device = "cpu",
    ) -> dict[str, float]:
        model.eval()
        per_image: list[dict[str, float]] = []
        cdr_scores: list[float] = []
        labels: list[int] = []
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            outputs, _ = model(images)
            if outputs is None:
                continue
            od_masks_gt = batch["od_masks"].numpy()
            oc_masks_gt = batch["oc_masks"].numpy()
            H, W = od_masks_gt.shape[-2:]
            for i, out in enumerate(outputs):
                od_pred, oc_pred = boxes_to_masks(out["od_boxes"], out["oc_boxes"], (H, W))
                row = {
                    "od_dice": dice(od_pred, od_masks_gt[i]),
                    "od_jaccard": jaccard(od_pred, od_masks_gt[i]),
                    "od_overlap_err": overlap_error(od_pred, od_masks_gt[i]),
                    "od_sensitivity": sensitivity(od_pred, od_masks_gt[i]),
                    "od_specificity": specificity(od_pred, od_masks_gt[i]),
                    "oc_dice": dice(oc_pred, oc_masks_gt[i]),
                    "oc_jaccard": jaccard(oc_pred, oc_masks_gt[i]),
                    "oc_overlap_err": overlap_error(oc_pred, oc_masks_gt[i]),
                    "oc_sensitivity": sensitivity(oc_pred, oc_masks_gt[i]),
                    "oc_specificity": specificity(oc_pred, oc_masks_gt[i]),
                }
                per_image.append(row)
                if out["od_boxes"].numel() and out["oc_boxes"].numel():
                    cdr_scores.append(compute_cdr(out["od_boxes"][0], out["oc_boxes"][0]))
                    # Glaucoma label inferred from GT: GT-CDR > 0.5
                    gt_od_h = od_masks_gt[i].sum(axis=0).max() if od_masks_gt[i].any() else 0
                    gt_oc_h = oc_masks_gt[i].sum(axis=0).max() if oc_masks_gt[i].any() else 0
                    gt_cdr = (gt_oc_h / gt_od_h) if gt_od_h > 0 else 0.0
                    labels.append(int(gt_cdr > 0.5))
        agg = aggregate(per_image)
        if cdr_scores and len(set(labels)) == 2:
            agg["glaucoma_auc"] = glaucoma_auc(np.asarray(cdr_scores), np.asarray(labels))
        return agg


def cli_main() -> None:  # used as `rdcnn-eval` entry point
    import hydra

    from glaucoma_rdcnn.data import build_dataloader
    from glaucoma_rdcnn.models import build_rdcnn
    from glaucoma_rdcnn.utils.checkpoint import load_checkpoint

    @hydra.main(version_base=None, config_path="../../../configs", config_name="default")
    def _run(cfg: Any) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_rdcnn(cfg.model).to(device)
        ckpt = getattr(cfg, "checkpoint", None)
        if ckpt:
            load_checkpoint(ckpt, model=model, map_location=device)
        loader = build_dataloader(cfg.data, "test")
        ev = Evaluator(cfg)
        out = ev.evaluate(model, loader, device=device)
        LOG.info("test | " + " | ".join(f"{k}={v:.4f}" for k, v in out.items()))

    _run()  # noqa: B018


if __name__ == "__main__":  # pragma: no cover
    cli_main()
