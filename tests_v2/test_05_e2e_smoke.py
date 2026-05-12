"""Smoke test 5: End-to-end forward+backward on a real GPU.

Run via slurm/009_smoke_v2.slurm — NOT in local pytest.
Skipped automatically if no CUDA device is present.
"""

from __future__ import annotations

import time

import pytest
import torch


pytestmark = pytest.mark.gpu

GPU_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="no CUDA")
def test_e2e_forward_backward_step() -> None:
    """One real DRISHTI batch through the full v2 stack on A100."""
    from glaucoma_rdcnn_v2.losses import CompoundSegLoss, precompute_sdt
    from glaucoma_rdcnn_v2.models import build_backbone, build_seg_head

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"\n[test_05] device={device} dtype={dtype}")
    print(f"[test_05] CUDA device name: {torch.cuda.get_device_name()}")

    # --- Build model components ---
    backbone = build_backbone(pretrained=True, image_size=224)
    backbone.freeze()
    head = build_seg_head(
        encoder_dim=backbone.feature_dim,
        decoder_dim=256,
        num_decoder_layers=3,
        target_size=512,
    )
    backbone.to(device=device, dtype=dtype)
    head.to(device=device, dtype=dtype)
    print(f"[test_05] backbone source: {backbone.source}")

    loss_fn = CompoundSegLoss(w_heteroscedastic=0.05)

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in head.parameters() if p.requires_grad], "lr": 5e-4},
        ],
        weight_decay=1e-4,
    )

    # --- Fake "DRISHTI" batch: 4 fundus-like 224x224 + 2-channel target masks ---
    # Real DRISHTI loading is wired up in the trainer; here we just need shape/finiteness.
    torch.manual_seed(0)
    B = 4
    images = torch.randn(B, 3, 224, 224, device=device, dtype=dtype)
    target_512 = torch.zeros(B, 2, 512, 512, device=device, dtype=dtype)
    target_512[:, 0, 100:400, 100:400] = 1.0  # disc
    target_512[:, 1, 150:350, 150:350] = 1.0  # cup

    # SDT on CPU then move
    sdt = precompute_sdt(target_512.float().cpu()).to(device=device, dtype=dtype)

    # --- One full step ---
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    optimizer.zero_grad()
    with torch.autocast("cuda", dtype=dtype):
        feats = backbone(images)
        out = head(feats.patch_features.to(dtype), grid_hw=feats.grid_hw)
        loss_before = loss_fn(
            out.mask_logits.float(),
            target_512.float(),
            sdt_target=sdt.float(),
            log_var=out.log_var.float(),
        )[0]
    loss_before.backward()
    optimizer.step()
    torch.cuda.synchronize()
    wall = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"[test_05] wall: {wall:.2f}s | peak VRAM: {peak_gb:.2f}GB | loss: {loss_before.item():.4f}")

    # --- Assertions ---
    assert torch.isfinite(loss_before), "loss is not finite"
    assert wall < 30.0, f"step took {wall:.2f}s (>30s threshold)"
    assert peak_gb < 40.0, f"peak VRAM {peak_gb:.2f}GB exceeds 40GB threshold"

    # Verify the model actually moved by running a second step
    with torch.autocast("cuda", dtype=dtype):
        feats2 = backbone(images)
        out2 = head(feats2.patch_features.to(dtype), grid_hw=feats2.grid_hw)
        loss_after = loss_fn(
            out2.mask_logits.float(),
            target_512.float(),
            sdt_target=sdt.float(),
            log_var=out2.log_var.float(),
        )[0]

    # Loss may not strictly decrease in one step on random data, but predictions must change
    pred_diff = (out.mask_logits.float() - out2.mask_logits.float()).abs().mean().item()
    assert pred_diff > 1e-5, f"predictions unchanged after step (diff={pred_diff:.2e})"

    # Gradients finite
    for p in head.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "non-finite gradient"
