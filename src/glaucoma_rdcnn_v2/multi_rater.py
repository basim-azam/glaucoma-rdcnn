"""Heteroscedastic multi-rater training for DRISHTI-GS.

DRISHTI-GS publishes 4 expert annotations per image. Standard practice is
to train against the majority-vote (or soft-mean) mask, discarding the rater
disagreement information.

Following Kendall & Gal (2017), we model per-pixel aleatoric uncertainty
explicitly: the segmentation head predicts both a mean mask and a per-pixel
log-variance map. The training target is the soft-mean across raters, and
the loss is reweighted by the model's predicted uncertainty:

    L_hetero = 0.5 * exp(-log_var) * BCE(logits, soft_target) + 0.5 * log_var

This downweights regions where the model is uncertain (typically high-rater-
disagreement boundary pixels), preventing the optimizer from over-fitting to
noisy labels. Published gain on dense seg: +2-3 pp Dice on the harder class.

NOTE: This module assumes the data adapter has been updated to emit per-rater
masks (data/drishti_gs/masks/od_rater{1,2,3,4}/<stem>.png and equivalently
for oc_rater{1,2,3,4}). If only the merged masks exist, the soft target
falls back to that single mask and the heteroscedastic term becomes
informative but not multi-rater.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Data: load per-rater masks and produce soft target + per-pixel agreement
# -----------------------------------------------------------------------------
def load_multi_rater_target(
    root: Path | str,
    stem: str,
    target_size: int,
    num_raters: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all rater masks for one image and produce soft target + agreement.

    Returns:
        soft_target: (2, H, W) float32 in [0, 1] — fraction of raters who labelled each pixel as foreground.
        agreement:   (2, H, W) float32 in [0, 1] — 1 = all raters agree (label is 0 or 1 for every rater),
                     0 = maximum disagreement (half label 1, half label 0).

    Falls back to the merged mask (raters = 1) if per-rater masks are missing.
    """
    root = Path(root)
    out_soft = np.zeros((2, target_size, target_size), dtype=np.float32)
    out_agreement = np.ones((2, target_size, target_size), dtype=np.float32)

    for ch_idx, ch_name in enumerate(("od", "oc")):
        rater_masks: list[np.ndarray] = []
        for r in range(1, num_raters + 1):
            p = root / "masks" / f"{ch_name}_rater{r}" / f"{stem}.png"
            if p.exists():
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    m = cv2.resize(m, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                    rater_masks.append((m > 127).astype(np.float32))

        if not rater_masks:
            # Fallback to merged mask
            p = root / "masks" / ch_name / f"{stem}.png"
            if p.exists():
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                m = cv2.resize(m, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                rater_masks.append((m > 127).astype(np.float32))

        if rater_masks:
            stack = np.stack(rater_masks, axis=0)
            out_soft[ch_idx] = stack.mean(axis=0)
            # Agreement: 1 - 2 * std (so 0 raters or all raters = 1, half/half = 0)
            std = stack.std(axis=0)
            out_agreement[ch_idx] = (1.0 - 2.0 * std).clip(0.0, 1.0)

    return out_soft, out_agreement


# -----------------------------------------------------------------------------
# Loss: heteroscedastic + rater-disagreement weighted
# -----------------------------------------------------------------------------
def heteroscedastic_multi_rater_loss(
    logits: torch.Tensor,        # (B, 2, H, W)
    log_var: torch.Tensor,       # (B, 2, H, W)
    soft_target: torch.Tensor,   # (B, 2, H, W) in [0, 1]
    agreement: torch.Tensor,     # (B, 2, H, W) in [0, 1]
    agreement_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Heteroscedastic loss with rater-agreement weighting.

    Args:
        logits:      raw seg head output.
        log_var:     raw heteroscedastic head output (per-pixel log variance).
        soft_target: target in [0,1], 0 if no rater labelled foreground, 1 if all did.
        agreement:   per-pixel rater agreement weight in [0,1].
        agreement_weight: how strongly to gate the loss by rater agreement.
                          1.0 = downweight disagreement regions to zero;
                          0.0 = ignore agreement, pure heteroscedastic.

    Returns:
        (loss_scalar, breakdown_dict).
    """
    # Per-pixel BCE on soft target
    bce = F.binary_cross_entropy_with_logits(logits, soft_target, reduction="none")

    # Heteroscedastic reweighting (Kendall & Gal 2017)
    precision = torch.exp(-log_var)
    hetero = 0.5 * precision * bce + 0.5 * log_var

    # Rater-agreement gating: high agreement -> full weight, low agreement -> downweight
    gate = (1.0 - agreement_weight) + agreement_weight * agreement
    weighted = hetero * gate

    loss = weighted.mean()
    return loss, {
        "hetero_mean": float(weighted.mean()),
        "bce_mean": float(bce.mean()),
        "log_var_mean": float(log_var.mean()),
        "agreement_mean": float(agreement.mean()),
    }


class MultiRaterDriftLoss(nn.Module):
    """Wrapper that combines the heteroscedastic multi-rater loss with the
    standard CompoundSegLoss for stability during early training.

    Usage in trainer:
        loss_fn = MultiRaterDriftLoss(
            w_hetero=0.4, w_dice=0.4, w_focal_tversky=0.2,
            agreement_weight=0.5, hetero_warmup_epochs=10,
        )
        loss, _ = loss_fn(logits, log_var, soft_target, agreement, current_epoch)
    """

    def __init__(
        self,
        w_hetero: float = 0.4,
        w_dice: float = 0.4,
        w_focal_tversky: float = 0.2,
        agreement_weight: float = 0.5,
        hetero_warmup_epochs: int = 10,
    ) -> None:
        super().__init__()
        from .losses import CompoundSegLoss
        self.compound = CompoundSegLoss(
            w_dice=w_dice,
            w_focal_tversky=w_focal_tversky,
            w_boundary=0.0,
            w_ce=0.0,
            w_heteroscedastic=0.0,
        )
        self.w_hetero = w_hetero
        self.agreement_weight = agreement_weight
        self.hetero_warmup = hetero_warmup_epochs

    def forward(
        self,
        logits: torch.Tensor,
        log_var: torch.Tensor,
        soft_target: torch.Tensor,
        agreement: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Compound (Dice + FT) on hard target (threshold soft target at 0.5)
        hard_target = (soft_target > 0.5).float()
        compound_loss, breakdown = self.compound(logits, hard_target)

        # Heteroscedastic ramp: 0 for first hetero_warmup epochs, then full
        w = self.w_hetero * max(0.0, min(1.0, (epoch - self.hetero_warmup) / 5.0))
        hetero_loss, hetero_break = heteroscedastic_multi_rater_loss(
            logits, log_var, soft_target, agreement, self.agreement_weight
        )
        total = compound_loss + w * hetero_loss

        breakdown["hetero"] = hetero_break["hetero_mean"]
        breakdown["hetero_w"] = w
        breakdown["agreement_mean"] = hetero_break["agreement_mean"]
        breakdown["total"] = float(total.item())
        return total, breakdown
