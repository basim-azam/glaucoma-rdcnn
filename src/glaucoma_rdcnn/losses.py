"""Losses helper.

The torchvision RPN + ROI heads return their own losses dict (objectness,
bbox regression, classifier, etc.). We just sum them with optional weights.
"""

from __future__ import annotations

from typing import Any

import torch


def weighted_total(
    losses: dict[str, torch.Tensor],
    weights: dict[str, float] | Any,
) -> torch.Tensor:
    """Sum the loss values with per-key weights.

    Keys in ``losses`` look like ``"dpn/loss_objectness"`` etc. We match
    ``weights`` by suffix-substring (e.g. ``dpn_objectness``).
    """
    total = None
    for k, v in losses.items():
        w = 1.0
        for wk, wv in dict(weights).items():
            if wk.replace("_", "/") in k or wk in k.replace("/", "_"):
                w = float(wv)
                break
        contribution = w * v
        total = contribution if total is None else total + contribution
    if total is None:
        # Empty losses dict — return a zero on the right device
        return torch.zeros((), requires_grad=True)
    return total
