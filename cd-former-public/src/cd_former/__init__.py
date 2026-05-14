"""CD-Former — joint optic disc and cup segmentation framework.

A unified four-component framework for joint OD/OC segmentation from colour
fundus photographs:

  - Domain-Adapted Visual Encoder (DAVE): fundus-pretrained ViT-L/16 backbone.
  - Multi-Scale Feature Lifting (MSFL): 1D-to-2D feature pyramid.
  - Cup-Conditioned Dual-Query Decoder (C2QD): query-based mask prediction.
  - Geometry-Consistent vCDR Refinement (GCvR): inference-time post-processing.

Quickstart:
    >>> from cd_former import load_pretrained
    >>> backbone, head, cfg = load_pretrained("drishti_seed43")
    >>> # ... run inference on a fundus image ...

See docs/MODEL_ZOO.md for available checkpoints and docs/TRAINING_RECIPE.md
for the full training recipe.
"""

from .data import FundusSegDataset
from .evaluator import SegMetrics, evaluate_model
from .losses import CompoundSegLoss, precompute_sdt
from .trainer import TrainerConfig, Trainer

__version__ = "0.2.0-dev"

__all__ = [
    "list_pretrained",
    "download_checkpoint",
    "load_pretrained",
    "FundusSegDataset",
    "SegMetrics",
    "evaluate_model",
    "CompoundSegLoss",
    "precompute_sdt",
    "TrainerConfig",
    "Trainer",
]

# Convenience HuggingFace-backed loader (see hub.py)
from .hub import list_pretrained, download_checkpoint, load_pretrained, HF_REPO_DEFAULT
