"""Hydra-driven training entry point.

Usage:
    python scripts/train.py                                            # default
    python scripts/train.py --config-name=experiment/repro_drishti
    python scripts/train.py training=spartan trainer.devices=4
"""

from __future__ import annotations

from typing import Any

import hydra

from glaucoma_rdcnn.engine.trainer import Trainer


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: Any) -> None:
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
