"""Disc Proposal Network (DPN).

A thin wrapper around torchvision's RPN + ROI heads that takes a single-level
feature map (from ``ResNet34DAC``) and returns OD bounding boxes. Built on
``torchvision.models.detection`` so we don't reimplement the proposal machinery.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RegionProposalNetwork,
    RPNHead,
)
from torchvision.ops import MultiScaleRoIAlign


class DiscProposalNetwork(nn.Module):
    """RPN + ROI head for a single OD class (binary: disc vs background)."""

    def __init__(
        self,
        in_channels: int,
        anchor_sizes: tuple[tuple[int, ...], ...] = ((64, 96, 128, 160, 192),),
        aspect_ratios: tuple[tuple[float, ...], ...] = ((0.8, 1.0, 1.25),),
        roi_output_size: int = 7,
        representation_size: int = 1024,
        rpn_pre_nms_top_n_train: int = 2000,
        rpn_post_nms_top_n_train: int = 1000,
        rpn_pre_nms_top_n_test: int = 1000,
        rpn_post_nms_top_n_test: int = 100,
        rpn_nms_thresh: float = 0.7,
        rpn_fg_iou_thresh: float = 0.7,
        rpn_bg_iou_thresh: float = 0.3,
        box_fg_iou_thresh: float = 0.5,
        box_bg_iou_thresh: float = 0.5,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 10,
    ) -> None:
        super().__init__()

        self.anchor_generator = AnchorGenerator(
            sizes=tuple(tuple(s) for s in anchor_sizes),
            aspect_ratios=tuple(tuple(a) for a in aspect_ratios),
        )
        rpn_head = RPNHead(in_channels, self.anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms = {"training": rpn_pre_nms_top_n_train, "testing": rpn_pre_nms_top_n_test}
        rpn_post_nms = {"training": rpn_post_nms_top_n_train, "testing": rpn_post_nms_top_n_test}

        self.rpn = RegionProposalNetwork(
            anchor_generator=self.anchor_generator,
            head=rpn_head,
            fg_iou_thresh=rpn_fg_iou_thresh,
            bg_iou_thresh=rpn_bg_iou_thresh,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=rpn_pre_nms,
            post_nms_top_n=rpn_post_nms,
            nms_thresh=rpn_nms_thresh,
        )

        roi_pool = MultiScaleRoIAlign(
            featmap_names=["0"], output_size=roi_output_size, sampling_ratio=2
        )
        resolution = roi_pool.output_size[0]
        box_head = TwoMLPHead(in_channels * resolution * resolution, representation_size)
        box_predictor = FastRCNNPredictor(representation_size, num_classes=2)  # bg + disc

        self.roi_heads = RoIHeads(
            box_roi_pool=roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=box_fg_iou_thresh,
            bg_iou_thresh=box_bg_iou_thresh,
            batch_size_per_image=128,
            positive_fraction=0.25,
            bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
            score_thresh=box_score_thresh,
            nms_thresh=box_nms_thresh,
            detections_per_img=box_detections_per_img,
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        image_sizes: list[tuple[int, int]],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        """Run RPN + ROI heads.

        Returns:
            (detections, losses). ``detections`` is a list (one per image) of dicts
            with keys ``boxes`` ``scores`` ``labels``. ``losses`` is empty in eval.
        """
        from torchvision.models.detection.image_list import ImageList

        feats = OrderedDict(list(features.items()))
        # Build a fake "images" tensor of zeros sized for the image list — only
        # the .image_sizes attribute is used by the RPN.
        h_max = max(s[0] for s in image_sizes)
        w_max = max(s[1] for s in image_sizes)
        dummy = features["0"].new_zeros((len(image_sizes), 3, h_max, w_max))
        image_list = ImageList(dummy, image_sizes)

        proposals, proposal_losses = self.rpn(image_list, feats, targets)
        detections, detector_losses = self.roi_heads(
            feats, proposals, image_list.image_sizes, targets
        )
        losses: dict[str, torch.Tensor] = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return detections, losses


def build_dpn(cfg: Any, in_channels: int) -> DiscProposalNetwork:
    return DiscProposalNetwork(
        in_channels=in_channels,
        anchor_sizes=tuple(tuple(s) for s in cfg.dpn.anchor_sizes),
        aspect_ratios=tuple(tuple(a) for a in cfg.dpn.aspect_ratios),
        roi_output_size=int(cfg.dpn.roi_output_size),
        representation_size=int(cfg.dpn.representation_size),
        rpn_pre_nms_top_n_train=int(cfg.dpn.pre_nms_top_n_train),
        rpn_post_nms_top_n_train=int(cfg.dpn.post_nms_top_n_train),
        rpn_pre_nms_top_n_test=int(cfg.dpn.pre_nms_top_n_test),
        rpn_post_nms_top_n_test=int(cfg.dpn.post_nms_top_n_test),
        rpn_nms_thresh=float(cfg.dpn.nms_thresh),
        rpn_fg_iou_thresh=float(cfg.dpn.fg_iou_thresh),
        rpn_bg_iou_thresh=float(cfg.dpn.bg_iou_thresh),
    )
