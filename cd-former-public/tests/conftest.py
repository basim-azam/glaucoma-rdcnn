"""Shared fixtures for v2 smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_OUTPUTS = REPO_ROOT / "tests_v2" / "_smoke_outputs"
SMOKE_OUTPUTS.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def smoke_outputs() -> Path:
    return SMOKE_OUTPUTS


@pytest.fixture
def fake_fundus() -> np.ndarray:
    """Synthetic fundus-like image with a bright disc near the center.

    Useful when the real DRISHTI data is not on disk (e.g., local laptop).
    """
    rng = np.random.default_rng(0)
    img = (rng.integers(40, 90, size=(960, 960, 3), dtype=np.uint8))
    # Add a bright disc
    cy, cx, r = 480, 540, 95
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    disc_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    img[disc_mask] = [220, 220, 200]
    return img


@pytest.fixture
def real_fundus_paths() -> list[Path]:
    """List of up to 5 real DRISHTI fundus image paths if available, else []."""
    candidates = sorted((REPO_ROOT / "data" / "drishti_gs" / "images").glob("*.png"))[:5]
    return candidates
