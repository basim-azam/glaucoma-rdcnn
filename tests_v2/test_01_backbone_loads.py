"""Smoke test 1: RETFound backbone loads and produces features.

CPU-only. Slow (~30s) because of the 1.2GB checkpoint download on first run.
Marked @pytest.mark.slow so quick CI runs can skip it.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.slow
def test_retfound_loads_with_correct_feature_shape() -> None:
    """Load RETFound (or fall back), push a dummy image through, check shapes."""
    from glaucoma_rdcnn_v2.models import build_backbone

    backbone = build_backbone(pretrained=True, image_size=224)
    backbone.eval()

    print(f"\n[test_01] backbone source: {backbone.source}")
    print(f"[test_01] param count: {sum(p.numel() for p in backbone.parameters()):,}")

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = backbone(x)

    # ViT-L/16 at 224 input -> 14x14 = 196 patches
    assert out.patch_features.dim() == 3, f"patch_features dim should be 3, got {out.patch_features.shape}"
    assert out.patch_features.shape[0] == 1
    assert out.patch_features.shape[-1] == 1024, f"feature_dim should be 1024, got {out.patch_features.shape[-1]}"

    # Finiteness check
    assert torch.isfinite(out.patch_features).all(), "patch features contain NaN/Inf"

    # Grid HW reported correctly
    assert out.grid_hw[0] * out.grid_hw[1] == out.patch_features.shape[1], \
        f"grid {out.grid_hw} doesn't match N={out.patch_features.shape[1]}"


def test_backbone_falls_back_cleanly() -> None:
    """Force fallback to DINOv2 — must succeed even if RETFound is gated."""
    from glaucoma_rdcnn_v2.models import build_backbone

    backbone = build_backbone(pretrained=True, image_size=224, prefer_fallback=True)
    assert "DINOv2" in backbone.source, f"expected DINOv2 fallback, got {backbone.source}"

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = backbone(x)
    assert torch.isfinite(out.patch_features).all()


def test_freeze_unfreeze() -> None:
    """freeze() and unfreeze() correctly toggle requires_grad on all params."""
    from glaucoma_rdcnn_v2.models import build_backbone

    backbone = build_backbone(pretrained=False, prefer_fallback=True)  # skip download
    backbone.freeze()
    assert all(not p.requires_grad for p in backbone.vit.parameters())
    assert backbone.is_frozen

    backbone.unfreeze()
    assert all(p.requires_grad for p in backbone.vit.parameters())
    assert not backbone.is_frozen
