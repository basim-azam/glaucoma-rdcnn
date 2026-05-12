"""DRISHTI-GS and RIM-ONE v3 datasets for v2 training.

Reuses the preprocessed outputs of the existing R-DCNN pipeline:
  data/<dataset>/images/<stem>.png       (CLAHE + 800x800 ROI-cropped fundus)
  data/<dataset>/masks/<stem>_od.png     (binary OD mask)
  data/<dataset>/masks/<stem>_oc.png     (binary OC mask)
  data/<dataset>/splits/official.json    {"train": [...], "val": [...], "test": [...]}

For v2 we resize the 800x800 inputs to:
  - 224x224 for the ViT encoder
  - 512x512 for the seg-head target masks
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_NORM_MEAN = (0.485, 0.456, 0.406)
_NORM_STD = (0.229, 0.224, 0.225)


def _imread_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _imread_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.float32)


def _augment(
    img: np.ndarray, od: np.ndarray, oc: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Light augmentation: hflip + rotation + scale + colour jitter."""
    h, w = img.shape[:2]
    # Random horizontal flip
    if rng.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
        od = np.ascontiguousarray(od[:, ::-1])
        oc = np.ascontiguousarray(oc[:, ::-1])
    # Random affine: rotation +/- 15 deg, scale 0.9 - 1.1
    angle = float(rng.uniform(-15, 15))
    scale = float(rng.uniform(0.9, 1.1))
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    od = cv2.warpAffine(od, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    oc = cv2.warpAffine(oc, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Colour jitter on the image only
    if rng.random() < 0.8:
        img = img.astype(np.float32)
        img *= rng.uniform(0.85, 1.15, size=3).reshape(1, 1, 3)
        img += rng.uniform(-10, 10, size=3).reshape(1, 1, 3)
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img, od, oc


class FundusSegDataset(Dataset):
    """Returns (image_224, target_512) where target is (2, H, W) {0,1}.

    Channel 0 = OD, Channel 1 = OC.
    """

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "val", "test"],
        encoder_size: int = 224,
        target_size: int = 512,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.encoder_size = encoder_size
        self.target_size = target_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        splits_path = self.root / "splits" / "official.json"
        with open(splits_path) as f:
            splits = json.load(f)
        if split not in splits:
            raise KeyError(f"split={split!r} not found; have {list(splits)}")
        self.stems: list[str] = list(splits[split])

        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"

        # Filter to entries that have all 3 files (defensive)
        keep: list[str] = []
        for stem in self.stems:
            if (
                (self.images_dir / f"{stem}.png").exists()
                and (self.masks_dir / f"{stem}_od.png").exists()
                and (self.masks_dir / f"{stem}_oc.png").exists()
            ):
                keep.append(stem)
        if len(keep) != len(self.stems):
            missing = set(self.stems) - set(keep)
            print(f"[data] {self.root.name}/{split}: dropping {len(missing)} stems with missing files")
        self.stems = keep

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        stem = self.stems[idx]
        img = _imread_rgb(self.images_dir / f"{stem}.png")
        od = _imread_mask(self.masks_dir / f"{stem}_od.png")
        oc = _imread_mask(self.masks_dir / f"{stem}_oc.png")

        if self.augment:
            img, od, oc = _augment(img, od, oc, self.rng)

        # Resize: encoder input is 224, target masks at 512
        img_enc = cv2.resize(img, (self.encoder_size, self.encoder_size), interpolation=cv2.INTER_LINEAR)
        od_tgt = cv2.resize(od, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        oc_tgt = cv2.resize(oc, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        # Normalize to ImageNet stats (works for both RETFound and DINOv2)
        img_enc = img_enc.astype(np.float32) / 255.0
        img_enc = (img_enc - np.array(_NORM_MEAN)) / np.array(_NORM_STD)
        img_t = torch.from_numpy(img_enc.transpose(2, 0, 1)).float()  # (3, H, W)

        target = np.stack([od_tgt, oc_tgt], axis=0).astype(np.float32)
        target_t = torch.from_numpy(target)  # (2, H, W)

        return {"image": img_t, "target": target_t, "stem": stem}
