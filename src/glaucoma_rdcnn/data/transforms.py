"""Joint image+mask+box transforms.

The augmentations from the paper are: horizontal flip, vertical flip, and
rotation by 90/180/270 degrees. We implement them so they apply consistently
to image, OD/OC masks, and the OD/OC bboxes derived from the masks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch


def _hflip(img: np.ndarray, mask_od: np.ndarray, mask_oc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return img[:, ::-1].copy(), mask_od[:, ::-1].copy(), mask_oc[:, ::-1].copy()


def _vflip(img: np.ndarray, mask_od: np.ndarray, mask_oc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return img[::-1].copy(), mask_od[::-1].copy(), mask_oc[::-1].copy()


def _rot(img: np.ndarray, mask_od: np.ndarray, mask_oc: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.rot90(img, k=k).copy(), np.rot90(mask_od, k=k).copy(), np.rot90(mask_oc, k=k).copy()


@dataclass
class AugmentConfig:
    hflip: float = 0.5
    vflip: float = 0.5
    rotation_choices: Sequence[int] = (0, 90, 180, 270)


def apply_augmentation(
    rng: np.random.Generator,
    img: np.ndarray,
    mask_od: np.ndarray,
    mask_oc: np.ndarray,
    cfg: AugmentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng.random() < cfg.hflip:
        img, mask_od, mask_oc = _hflip(img, mask_od, mask_oc)
    if rng.random() < cfg.vflip:
        img, mask_od, mask_oc = _vflip(img, mask_od, mask_oc)
    angle = int(rng.choice(np.asarray(cfg.rotation_choices, dtype=np.int64)))
    k = angle // 90
    if k:
        img, mask_od, mask_oc = _rot(img, mask_od, mask_oc, k)
    return img, mask_od, mask_oc


def mask_to_bbox(mask: np.ndarray) -> np.ndarray | None:
    """Compute the tight axis-aligned bbox of a binary mask, in xyxy format."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def normalize_image(img: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """RGB uint8 H,W,3 → float32 tensor 3,H,W normalized to ImageNet stats."""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    arr = img.astype(np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())
