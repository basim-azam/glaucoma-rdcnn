"""Evaluation entry point.

Usage:
    python scripts/evaluate.py --config-name=experiment/repro_drishti checkpoint=outputs/best.ckpt
"""

from __future__ import annotations

from typing import Any

import hydra
import torch

from glaucoma_rdcnn.data import build_dataloader
from glaucoma_rdcnn.engine.evaluator import Evaluator
from glaucoma_rdcnn.models import build_rdcnn
from glaucoma_rdcnn.utils.checkpoint import load_checkpoint
from glaucoma_rdcnn.utils.logging import get_logger, setup_logging

setup_logging()
LOG = get_logger("evaluate")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: Any) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_rdcnn(cfg.model).to(device)
    ckpt_path = getattr(cfg, "checkpoint", None)
    if ckpt_path:
        load_checkpoint(ckpt_path, model=model, map_location=device)
        LOG.info(f"loaded checkpoint {ckpt_path}")
    loader = build_dataloader(cfg.data, "test")
    out = Evaluator(cfg).evaluate(model, loader, device=device)
    LOG.info("test | " + " | ".join(f"{k}={v:.4f}" for k, v in out.items()))


if __name__ == "__main__":
    main()
