"""Compound loss for OD/OC segmentation.

L = 0.4 * SoftDice + 0.3 * FocalTversky + 0.2 * BoundaryLoss + 0.1 * CE
    + (optional) heteroscedastic uncertainty regularizer

Each component is computed per-channel (OD, OC) and averaged.

BoundaryLoss uses the signed-distance-transform (SDT) of the GT mask;
SDTs are precomputed per-target (CPU side) and passed in as a tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Per-channel soft Dice loss.

    Args:
        logits: (B, C, H, W) raw logits.
        target: (B, C, H, W) {0, 1} float or in [0, 1] for soft labels.
    """
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 4.0 / 3.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Focal Tversky loss. alpha penalizes FN, beta penalizes FP.

    Tuned for under-segmentation (alpha > beta) — small cup target.
    """
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    tp = (probs * target).sum(dim=dims)
    fn = ((1 - probs) * target).sum(dim=dims)
    fp = (probs * (1 - target)).sum(dim=dims)
    tversky = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    focal = (1.0 - tversky) ** gamma
    return focal.mean()


def boundary_loss(
    logits: torch.Tensor, sdt_target: torch.Tensor
) -> torch.Tensor:
    """Boundary loss (Kervadec 2019) using precomputed signed distance transform.

    Args:
        logits: (B, C, H, W) raw logits.
        sdt_target: (B, C, H, W) signed distance transform of GT (negative inside).

    Loss = mean( sigmoid(logits) * sdt_target ).
    """
    probs = torch.sigmoid(logits)
    return (probs * sdt_target).mean()


def ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-pixel BCE-with-logits, averaged over batch/channels/pixels."""
    return F.binary_cross_entropy_with_logits(logits, target)


def heteroscedastic_loss(
    logits: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Kendall & Gal 2017 heteroscedastic dense-regression loss.

    Adapted for segmentation logits:
        L = 0.5 * exp(-log_var) * (logit - target)^2 + 0.5 * log_var

    Uses BCE-residual instead of raw squared error so the head is comparable
    with the rest of the seg losses.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    precision = torch.exp(-log_var)
    return (0.5 * precision * bce + 0.5 * log_var).mean()


class CompoundSegLoss(nn.Module):
    """Compound loss used by the v2 trainer.

    Weights match docs/v2_smoke_test_plan.md.
    """

    def __init__(
        self,
        w_dice: float = 0.4,
        w_focal_tversky: float = 0.3,
        w_boundary: float = 0.2,
        w_ce: float = 0.1,
        w_heteroscedastic: float = 0.0,  # off by default; trainer enables
        focal_tversky_alpha: float = 0.7,
        focal_tversky_beta: float = 0.3,
        focal_tversky_gamma: float = 4.0 / 3.0,
    ) -> None:
        super().__init__()
        self.w_dice = w_dice
        self.w_focal_tversky = w_focal_tversky
        self.w_boundary = w_boundary
        self.w_ce = w_ce
        self.w_heteroscedastic = w_heteroscedastic
        self.ft_alpha = focal_tversky_alpha
        self.ft_beta = focal_tversky_beta
        self.ft_gamma = focal_tversky_gamma

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        sdt_target: torch.Tensor | None = None,
        log_var: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute compound loss. Returns (loss, breakdown_dict)."""
        components: dict[str, torch.Tensor] = {}

        components["dice"] = soft_dice_loss(logits, target)
        components["focal_tversky"] = focal_tversky_loss(
            logits,
            target,
            alpha=self.ft_alpha,
            beta=self.ft_beta,
            gamma=self.ft_gamma,
        )
        components["ce"] = ce_loss(logits, target)

        if sdt_target is not None:
            components["boundary"] = boundary_loss(logits, sdt_target)
        else:
            components["boundary"] = torch.zeros((), device=logits.device)

        if log_var is not None and self.w_heteroscedastic > 0:
            components["heteroscedastic"] = heteroscedastic_loss(logits, log_var, target)
        else:
            components["heteroscedastic"] = torch.zeros((), device=logits.device)

        total = (
            self.w_dice * components["dice"]
            + self.w_focal_tversky * components["focal_tversky"]
            + self.w_boundary * components["boundary"]
            + self.w_ce * components["ce"]
            + self.w_heteroscedastic * components["heteroscedastic"]
        )
        breakdown = {k: v.item() for k, v in components.items()}
        breakdown["total"] = total.item()
        return total, breakdown


def precompute_sdt(mask: torch.Tensor) -> torch.Tensor:
    """Compute signed distance transform from a binary mask.

    Args:
        mask: (B, C, H, W) {0, 1} float tensor.

    Returns:
        (B, C, H, W) signed distance: negative inside the mask, positive outside.

    Uses scipy.ndimage on CPU then returns a tensor on the mask's device.
    """
    from scipy.ndimage import distance_transform_edt

    device = mask.device
    mask_np = mask.detach().cpu().numpy()
    out = torch.zeros_like(mask, device="cpu")
    for b in range(mask_np.shape[0]):
        for c in range(mask_np.shape[1]):
            m = mask_np[b, c] > 0.5
            if not m.any():
                continue
            inside = distance_transform_edt(m)
            outside = distance_transform_edt(~m)
            out[b, c] = torch.from_numpy(outside - inside).float()
    return out.to(device)
