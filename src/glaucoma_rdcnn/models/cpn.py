"""Cup Proposal Network (CPN).

Identical structure to ``DiscProposalNetwork``, but with cup-scale anchors and
operating on attended features (after ``DiscAttention``).
"""

from __future__ import annotations

from typing import Any

from glaucoma_rdcnn.models.dpn import DiscProposalNetwork


class CupProposalNetwork(DiscProposalNetwork):
    """Same machinery as DPN — kept as a separate class for clarity and so that
    parameters are tracked separately when computing per-head losses.
    """


def build_cpn(cfg: Any, in_channels: int) -> CupProposalNetwork:
    return CupProposalNetwork(
        in_channels=in_channels,
        anchor_sizes=tuple(tuple(s) for s in cfg.cpn.anchor_sizes),
        aspect_ratios=tuple(tuple(a) for a in cfg.cpn.aspect_ratios),
        roi_output_size=int(cfg.cpn.roi_output_size),
        representation_size=int(cfg.cpn.representation_size),
        rpn_pre_nms_top_n_train=int(cfg.cpn.pre_nms_top_n_train),
        rpn_post_nms_top_n_train=int(cfg.cpn.post_nms_top_n_train),
        rpn_pre_nms_top_n_test=int(cfg.cpn.pre_nms_top_n_test),
        rpn_post_nms_top_n_test=int(cfg.cpn.post_nms_top_n_test),
        rpn_nms_thresh=float(cfg.cpn.nms_thresh),
        rpn_fg_iou_thresh=float(cfg.cpn.fg_iou_thresh),
        rpn_bg_iou_thresh=float(cfg.cpn.bg_iou_thresh),
    )
