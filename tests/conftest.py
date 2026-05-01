"""Shared fixtures."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def model_cfg(repo_root: Path):
    cfg_path = repo_root / "configs" / "model" / "rdcnn.yaml"
    return OmegaConf.load(cfg_path)
