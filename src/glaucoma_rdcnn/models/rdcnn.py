"""Full R-DCNN: backbone → DPN → DiscAttention → CPN.

Public API:

* ``RDCNN(cfg)`` — full module.
* ``build_rdcnn(cfg)`` — convenience constructor.

In training mode the model expects ``targets`` per image with keys:

    {
        "od_boxes": Tensor[N, 4],   # xyxy in image pixels
        "oc_boxes": Tensor[M, 4],
    }

In eval mode it returns a list of dicts:

    {
        "od_boxes": Tensor[K, 4],
        "od_scores": Tensor[K],
        "oc_boxes": Tensor[J, 4],
        "oc_scores": Tensor[J],
    }
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from glaucoma_rdcnn.models.attention import DiscAttention, boxes_to_soft_mask
from glaucoma_rdcnn.models.backbone import build_backbone
from glaucoma_rdcnn.models.cpn import build_cpn
from glaucoma_rdcnn.models.dpn import build_dpn


def _to_target_dict(boxes: torch.Tensor, label: int = 1) -> dict[str, torch.Tensor]:
    """Wrap raw boxes in the dict form expected by torchvision ROI heads."""
    if boxes.numel() == 0:
        return {
            "boxes": boxes.new_zeros((0, 4)),
            "labels": boxes.new_zeros((0,), dtype=torch.int64),
        }
    return {
        "boxes": boxes,
        "labels": torch.full((boxes.shape[0],), label, dtype=torch.int64, device=boxes.device),
    }


class RDCNN(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        in_c = self.backbone.out_channels
        self.dpn = build_dpn(cfg, in_channels=in_c)
        self.attention = DiscAttention(
            in_channels=in_c,
            hidden_channels=int(cfg.attention.hidden_channels),
            out_channels=int(cfg.attention.out_channels),
        )
        self.cpn = build_cpn(cfg, in_channels=self.attention.out_channels)

    @staticmethod
    def _image_sizes(images: torch.Tensor) -> list[tuple[int, int]]:
        return [(int(images.shape[-2]), int(images.shape[-1]))] * images.shape[0]

    def forward(
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]] | None, dict[str, torch.Tensor]]:
        feats = self.backbone(images)
        image_sizes = self._image_sizes(images)

        # --- Disc head ---
        dpn_targets = None
        if self.training:
            assert targets is not None, "targets required in training mode"
            dpn_targets = [_to_target_dict(t["od_boxes"]) for t in targets]
        dpn_dets, dpn_losses = self.dpn(feats, image_sizes, dpn_targets)

        # --- Attention: rasterize OD bbox into a soft mask (use GT in train) ---
        if self.training:
            assert targets is not None
            od_boxes_per_img = [t["od_boxes"] for t in targets]
        else:
            od_boxes_per_img = [
                d["boxes"][:1] if d["boxes"].numel() else d["boxes"] for d in dpn_dets
            ]

        feat_map = feats["0"]
        Hf, Wf = feat_map.shape[-2:]
        Hi, Wi = image_sizes[0]
        attended = []
        for i, boxes in enumerate(od_boxes_per_img):
            if boxes.numel() == 0:
                soft = feat_map.new_ones((1, 1, Hf, Wf))
            else:
                soft = boxes_to_soft_mask(boxes, (Hf, Wf), (Hi, Wi))
                soft = soft.amax(dim=0, keepdim=True)
            attended.append(self.attention(feat_map[i : i + 1], soft))
        attended_feat = torch.cat(attended, dim=0)
        attended_feats = {"0": attended_feat}

        # --- Cup head ---
        cpn_targets = None
        if self.training:
            assert targets is not None
            cpn_targets = [_to_target_dict(t["oc_boxes"]) for t in targets]
        cpn_dets, cpn_losses = self.cpn(attended_feats, image_sizes, cpn_targets)

        losses: dict[str, torch.Tensor] = {}
        for k, v in dpn_losses.items():
            losses[f"dpn/{k}"] = v
        for k, v in cpn_losses.items():
            losses[f"cpn/{k}"] = v

        if self.training:
            return None, losses

        # Combine for evaluation
        out: list[dict[str, torch.Tensor]] = []
        for d, c in zip(dpn_dets, cpn_dets, strict=False):
            out.append(
                {
                    "od_boxes": d["boxes"],
                    "od_scores": d["scores"],
                    "oc_boxes": c["boxes"],
                    "oc_scores": c["scores"],
                }
            )
        return out, losses


def build_rdcnn(cfg: Any) -> RDCNN:
    return RDCNN(cfg)
