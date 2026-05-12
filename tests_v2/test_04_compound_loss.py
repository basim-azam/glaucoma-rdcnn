"""Smoke test 4: Compound loss is finite, scalar, backward-able, monotonic."""

from __future__ import annotations

import torch


def _make_random_pair(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    logits = torch.randn(2, 2, 64, 64, requires_grad=True)
    # Simple synthetic targets: a centered square
    target = torch.zeros(2, 2, 64, 64)
    target[:, :, 20:44, 20:44] = 1.0
    return logits, target


def test_loss_is_scalar_and_finite() -> None:
    from glaucoma_rdcnn_v2.losses import CompoundSegLoss, precompute_sdt

    loss_fn = CompoundSegLoss()
    logits, target = _make_random_pair()
    sdt = precompute_sdt(target)

    total, breakdown = loss_fn(logits, target, sdt_target=sdt)
    assert total.ndim == 0, "loss must be a scalar"
    assert torch.isfinite(total), "loss is not finite"
    assert all(k in breakdown for k in ("dice", "focal_tversky", "boundary", "ce", "total"))


def test_loss_backward_succeeds() -> None:
    from glaucoma_rdcnn_v2.losses import CompoundSegLoss, precompute_sdt

    loss_fn = CompoundSegLoss()
    logits, target = _make_random_pair()
    sdt = precompute_sdt(target)
    total, _ = loss_fn(logits, target, sdt_target=sdt)
    total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_loss_is_monotonic_in_prediction_quality() -> None:
    """Better predictions should give lower loss."""
    from glaucoma_rdcnn_v2.losses import CompoundSegLoss, precompute_sdt

    loss_fn = CompoundSegLoss()
    _, target = _make_random_pair()
    sdt = precompute_sdt(target)

    # Random predictions
    torch.manual_seed(42)
    rand_logits = torch.randn_like(target) * 0.5
    rand_loss, _ = loss_fn(rand_logits, target, sdt_target=sdt)

    # Near-perfect predictions: logits aligned with target
    perfect_logits = (target * 6.0 - 3.0)  # +3 inside, -3 outside
    perfect_loss, _ = loss_fn(perfect_logits, target, sdt_target=sdt)

    assert perfect_loss < rand_loss, \
        f"loss not monotonic: random={rand_loss.item():.4f} vs perfect={perfect_loss.item():.4f}"


def test_heteroscedastic_component() -> None:
    """When enabled, the heteroscedastic term is finite and contributes."""
    from glaucoma_rdcnn_v2.losses import CompoundSegLoss

    loss_fn = CompoundSegLoss(w_heteroscedastic=0.1)
    logits, target = _make_random_pair()
    log_var = torch.zeros_like(logits)
    total, breakdown = loss_fn(logits, target, log_var=log_var)
    assert breakdown["heteroscedastic"] > 0
    assert torch.isfinite(total)
