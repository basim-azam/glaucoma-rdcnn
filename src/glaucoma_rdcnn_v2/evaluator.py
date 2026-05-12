"""Evaluation metrics for v2: OD/OC Dice, Jaccard, overlap-error, glaucoma AUC.

Matches the reporting format of the existing R-DCNN evaluator so the result
tables line up cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SegMetrics:
    od_dice: float
    od_jaccard: float
    od_overlap_err: float
    od_sensitivity: float
    od_specificity: float
    oc_dice: float
    oc_jaccard: float
    oc_overlap_err: float
    oc_sensitivity: float
    oc_specificity: float
    glaucoma_auc: float
    cdr_mae: float
    n_images: int

    def as_log_line(self, split: str = "test") -> str:
        return (
            f"{split} | "
            f"od_dice={self.od_dice:.4f} | od_jaccard={self.od_jaccard:.4f} | "
            f"od_overlap_err={self.od_overlap_err:.4f} | "
            f"od_sensitivity={self.od_sensitivity:.4f} | od_specificity={self.od_specificity:.4f} | "
            f"oc_dice={self.oc_dice:.4f} | oc_jaccard={self.oc_jaccard:.4f} | "
            f"oc_overlap_err={self.oc_overlap_err:.4f} | "
            f"oc_sensitivity={self.oc_sensitivity:.4f} | oc_specificity={self.oc_specificity:.4f} | "
            f"glaucoma_auc={self.glaucoma_auc:.4f} | "
            f"cdr_mae={self.cdr_mae:.4f} | n={self.n_images}"
        )


def _dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    inter = (pred & target).sum()
    union = pred.sum() + target.sum()
    return float((2 * inter + eps) / (union + eps))


def _jaccard(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    inter = (pred & target).sum()
    uni = (pred | target).sum()
    return float((inter + eps) / (uni + eps))


def _vertical_extent(mask: np.ndarray) -> float:
    """Vertical extent (top-bottom span) of a binary mask, normalized to image height."""
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return 0.0
    return float(rows[-1] - rows[0] + 1) / mask.shape[0]


@torch.no_grad()
def evaluate_model(
    model_forward,
    loader,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    threshold: float = 0.5,
) -> SegMetrics:
    """Run the full eval loop. model_forward is a callable: image_tensor -> mask_logits (B, 2, H, W)."""
    from sklearn.metrics import roc_auc_score

    od_dices, oc_dices = [], []
    od_jcs, oc_jcs = [], []
    od_sens, od_specs = [], []
    oc_sens, oc_specs = [], []
    cdr_preds, cdr_gts = [], []
    is_glaucoma = []  # heuristic: GT CDR > 0.5

    for batch in loader:
        images = batch["image"].to(device=device, dtype=dtype)
        targets = batch["target"]  # (B, 2, H, W) on CPU is fine

        with torch.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda" and dtype != torch.float32)):
            logits = model_forward(images)  # (B, 2, H, W)
        probs = torch.sigmoid(logits.float()).cpu().numpy()

        for b in range(probs.shape[0]):
            od_p = (probs[b, 0] > threshold).astype(bool)
            oc_p = (probs[b, 1] > threshold).astype(bool)
            od_t = (targets[b, 0].numpy() > 0.5).astype(bool)
            oc_t = (targets[b, 1].numpy() > 0.5).astype(bool)

            od_dices.append(_dice(od_p, od_t))
            oc_dices.append(_dice(oc_p, oc_t))
            od_jcs.append(_jaccard(od_p, od_t))
            oc_jcs.append(_jaccard(oc_p, oc_t))

            # Sensitivity / specificity, treating disc/cup as foreground
            for pred, tgt, sens_l, spec_l in [
                (od_p, od_t, od_sens, od_specs),
                (oc_p, oc_t, oc_sens, oc_specs),
            ]:
                tp = (pred & tgt).sum()
                fn = (~pred & tgt).sum()
                tn = (~pred & ~tgt).sum()
                fp = (pred & ~tgt).sum()
                sens_l.append(float(tp / (tp + fn + 1e-6)))
                spec_l.append(float(tn / (tn + fp + 1e-6)))

            # CDR from vertical extent
            cdr_pred = (
                _vertical_extent(oc_p) / _vertical_extent(od_p)
                if od_p.any()
                else 0.0
            )
            cdr_gt = (
                _vertical_extent(oc_t) / _vertical_extent(od_t)
                if od_t.any()
                else 0.0
            )
            cdr_preds.append(cdr_pred)
            cdr_gts.append(cdr_gt)
            is_glaucoma.append(int(cdr_gt > 0.5))

    cdr_preds_a = np.array(cdr_preds)
    cdr_gts_a = np.array(cdr_gts)
    cdr_mae = float(np.mean(np.abs(cdr_preds_a - cdr_gts_a)))

    # Glaucoma AUC: predicted CDR vs GT-derived glaucoma label
    is_g = np.array(is_glaucoma)
    if is_g.min() == is_g.max():
        auc = 0.5
    else:
        try:
            auc = float(roc_auc_score(is_g, cdr_preds_a))
        except Exception:
            auc = 0.5

    return SegMetrics(
        od_dice=float(np.mean(od_dices)),
        od_jaccard=float(np.mean(od_jcs)),
        od_overlap_err=1.0 - float(np.mean(od_jcs)),
        od_sensitivity=float(np.mean(od_sens)),
        od_specificity=float(np.mean(od_specs)),
        oc_dice=float(np.mean(oc_dices)),
        oc_jaccard=float(np.mean(oc_jcs)),
        oc_overlap_err=1.0 - float(np.mean(oc_jcs)),
        oc_sensitivity=float(np.mean(oc_sens)),
        oc_specificity=float(np.mean(oc_specs)),
        glaucoma_auc=auc,
        cdr_mae=cdr_mae,
        n_images=len(od_dices),
    )
