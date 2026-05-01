"""Smoke tests for the R-DCNN model.

Runs on CPU with dummy tensors. We use a small input (256x256) to keep CI fast,
and we don't load ImageNet pretrained weights here (the CI environment doesn't
need them — torchvision will download them once for real training).
"""

import torch
from omegaconf import OmegaConf

from glaucoma_rdcnn.models import build_rdcnn
from glaucoma_rdcnn.models.attention import DiscAttention, boxes_to_soft_mask
from glaucoma_rdcnn.models.dac import DenseAtrousConv


def test_dac_forward_shape():
    block = DenseAtrousConv(in_channels=64, out_channels=32, rates=(1, 3, 6, 12))
    x = torch.randn(2, 64, 16, 16)
    y = block(x)
    assert y.shape == (2, 32, 16, 16)


def test_disc_attention_forward_shape():
    attn = DiscAttention(in_channels=32, hidden_channels=16, out_channels=32)
    feat = torch.randn(1, 32, 8, 8)
    box = torch.tensor([[1.0, 1.0, 7.0, 7.0]])
    soft = boxes_to_soft_mask(box, (8, 8), (8, 8))
    out = attn(feat, soft)
    assert out.shape == (1, 32, 8, 8)


def test_rdcnn_train_forward(model_cfg):
    """In training mode the model returns (None, losses)."""
    # Override to avoid downloading ImageNet weights in CI
    cfg = OmegaConf.merge(model_cfg, OmegaConf.create({"backbone": {"pretrained": False}}))
    model = build_rdcnn(cfg)
    model.train()
    x = torch.randn(1, 3, 256, 256)
    targets = [
        {
            "od_boxes": torch.tensor([[60.0, 60.0, 200.0, 200.0]]),
            "oc_boxes": torch.tensor([[100.0, 100.0, 160.0, 160.0]]),
        }
    ]
    out, losses = model(x, targets)
    assert out is None
    assert len(losses) > 0
    # All losses should be finite tensors
    for k, v in losses.items():
        assert torch.is_tensor(v), k
        assert torch.isfinite(v).all(), k


def test_rdcnn_eval_forward(model_cfg):
    cfg = OmegaConf.merge(model_cfg, OmegaConf.create({"backbone": {"pretrained": False}}))
    model = build_rdcnn(cfg)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        outputs, losses = model(x)
    assert outputs is not None
    assert len(outputs) == 1
    assert "od_boxes" in outputs[0]
    assert "oc_boxes" in outputs[0]
