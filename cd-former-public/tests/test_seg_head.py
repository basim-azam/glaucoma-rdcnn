"""Smoke test 3: Two-query Mask2Former-style head produces correct-shape masks."""

from __future__ import annotations

import torch


def test_seg_head_output_shape_and_disentanglement() -> None:
    """Head outputs (B, 2, 512, 512), and the two channels are not identical."""
    from cd_former.models import build_seg_head

    head = build_seg_head(
        encoder_dim=1024, decoder_dim=128, num_decoder_layers=2, target_size=512
    )
    head.eval()

    # Fake ViT-L features at 14x14 patch grid (196 patches, 1024-d)
    B, N, C = 2, 14 * 14, 1024
    fake_patches = torch.randn(B, N, C) * 0.1

    out = head(fake_patches, grid_hw=(14, 14))
    assert out.mask_logits.shape == (B, 2, 512, 512), out.mask_logits.shape
    assert out.log_var.shape == (B, 2, 512, 512), out.log_var.shape

    # Disentanglement: the two channels should differ after random init
    diff = (out.mask_logits[:, 0] - out.mask_logits[:, 1]).abs().mean().item()
    assert diff > 1e-6, f"disc and cup channels are identical (diff={diff:.2e})"

    # Aux logits for deep supervision
    assert len(out.aux_logits) == 4, f"expected 4 aux levels, got {len(out.aux_logits)}"
    for aux in out.aux_logits:
        assert aux.shape[0] == B and aux.shape[1] == 2


def test_seg_head_gradients_flow() -> None:
    """loss.backward() produces gradients on both query embeddings."""
    from cd_former.models import build_seg_head

    head = build_seg_head(encoder_dim=1024, decoder_dim=128, num_decoder_layers=2)
    fake = torch.randn(1, 196, 1024, requires_grad=True)
    out = head(fake, grid_hw=(14, 14))
    fake_target = torch.randint(0, 2, (1, 2, 512, 512)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out.mask_logits, fake_target)
    loss.backward()

    assert head.query_disc.grad is not None
    assert head.query_cup.grad is not None
    assert head.query_disc.grad.abs().sum() > 0
    assert head.query_cup.grad.abs().sum() > 0
