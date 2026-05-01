"""Fundus OD/OC dataset that emits images + OD/OC bboxes (and masks for eval).

Layout assumed (after ``scripts/preprocess.py``):

    <root>/
        images/<stem>.png
        masks/od/<stem>.png      # 0/255 binary
        masks/oc/<stem>.png      # 0/255 binary
        splits/official.json     # {"train": [...stems...], "val": [...], "test": [...]}

If you have your own data, see ``docs/dataset_schema.md`` and edit
``configs/data/custom.yaml``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from glaucoma_rdcnn.data.transforms import (
    AugmentConfig,
    apply_augmentation,
    mask_to_bbox,
    normalize_image,
)


@dataclass
class Sample:
    image: torch.Tensor   # (3, H, W)
    od_mask: torch.Tensor   # (H, W) uint8
    oc_mask: torch.Tensor   # (H, W) uint8
    od_boxes: torch.Tensor   # (N=1, 4) xyxy
    oc_boxes: torch.Tensor   # (N=1, 4) xyxy
    stem: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "od_mask": self.od_mask,
            "oc_mask": self.oc_mask,
            "od_boxes": self.od_boxes,
            "oc_boxes": self.oc_boxes,
            "stem": self.stem,
        }


class FundusODOCDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        stems: Iterable[str],
        image_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        augment: AugmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.stems = list(stems)
        self.image_size = int(image_size)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.stems)

    def _read_one(self, stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        img_path = self.root / "images" / f"{stem}.png"
        od_path = self.root / "masks" / "od" / f"{stem}.png"
        oc_path = self.root / "masks" / "oc" / f"{stem}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        od = cv2.imread(str(od_path), cv2.IMREAD_GRAYSCALE)
        oc = cv2.imread(str(oc_path), cv2.IMREAD_GRAYSCALE)
        if od is None or oc is None:
            raise FileNotFoundError(f"missing mask for {stem}")
        return img, (od > 127).astype(np.uint8), (oc > 127).astype(np.uint8)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        stem = self.stems[idx]
        img, od, oc = self._read_one(stem)
        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            od = cv2.resize(od, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            oc = cv2.resize(oc, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.augment is not None:
            img, od, oc = apply_augmentation(self._rng, img, od, oc, self.augment)
        od_box = mask_to_bbox(od)
        oc_box = mask_to_bbox(oc)
        if od_box is None or oc_box is None:
            raise ValueError(f"{stem}: empty mask, cannot produce bbox")
        sample = Sample(
            image=normalize_image(img, self.mean, self.std),
            od_mask=torch.from_numpy(od),
            oc_mask=torch.from_numpy(oc),
            od_boxes=torch.from_numpy(od_box).unsqueeze(0),
            oc_boxes=torch.from_numpy(oc_box).unsqueeze(0),
            stem=stem,
        )
        return sample.to_dict()


def collate_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate that keeps per-image targets as a list (variable bbox count)."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = [{"od_boxes": b["od_boxes"], "oc_boxes": b["oc_boxes"]} for b in batch]
    od_masks = torch.stack([b["od_mask"] for b in batch], dim=0)
    oc_masks = torch.stack([b["oc_mask"] for b in batch], dim=0)
    stems = [b["stem"] for b in batch]
    return {
        "images": images,
        "targets": targets,
        "od_masks": od_masks,
        "oc_masks": oc_masks,
        "stems": stems,
    }


def _read_splits(path: Path) -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


def build_dataset(cfg: Any, split: str) -> FundusODOCDataset:
    splits_path = Path(cfg.root) / cfg.splits_file
    splits = _read_splits(splits_path)
    if split not in splits:
        raise KeyError(f"split '{split}' not in {splits_path} (keys={list(splits)})")
    augment = None
    if split == "train":
        augment = AugmentConfig(
            hflip=float(cfg.augment.hflip),
            vflip=float(cfg.augment.vflip),
            rotation_choices=tuple(cfg.augment.rotation_choices),
        )
    return FundusODOCDataset(
        root=cfg.root,
        stems=splits[split],
        image_size=int(cfg.image_size),
        mean=tuple(cfg.mean),
        std=tuple(cfg.std),
        augment=augment,
    )


def build_dataloader(cfg: Any, split: str, *, distributed: bool = False, **kwargs: Any) -> DataLoader:
    ds = build_dataset(cfg, split)
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(ds, shuffle=(split == "train"))
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=(split == "train" and sampler is None),
        num_workers=int(cfg.num_workers),
        collate_fn=collate_samples,
        sampler=sampler,
        pin_memory=True,
        drop_last=(split == "train"),
        **kwargs,
    )
