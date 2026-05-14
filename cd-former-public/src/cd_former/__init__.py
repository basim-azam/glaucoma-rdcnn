"""rdcnn-modern (v2): RETFound + Mask2Former-style head for OD/OC segmentation.

Replaces the R-DCNN anchor-detector + ellipse-fit pipeline with:
  - RETFound MAE ViT-L/16 backbone (fundus-pretrained, frozen-then-unfrozen)
  - Hough + tiny U-Net Stage-1 ROI localizer
  - Two-query Mask2Former-style head (disc + cup, cup-init-from-disc)
  - Compound loss: SoftDice + FocalTversky + BoundaryLoss + CE
  - Heteroscedastic multi-rater uncertainty head (optional)

See docs/modernization_research.md and docs/v2_smoke_test_plan.md.
"""

from .data import FundusSegDataset
from .evaluator import SegMetrics, evaluate_model
from .losses import CompoundSegLoss, precompute_sdt
from .trainer import TrainerConfig, V2Trainer

__version__ = "0.2.0-dev"

__all__ = [
    "FundusSegDataset",
    "SegMetrics",
    "evaluate_model",
    "CompoundSegLoss",
    "precompute_sdt",
    "TrainerConfig",
    "V2Trainer",
]
