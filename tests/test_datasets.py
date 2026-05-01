"""Smoke tests for the dataset pipeline using a tiny on-the-fly synthetic dataset."""
import json
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from glaucoma_rdcnn.data import build_dataloader
from glaucoma_rdcnn.data.transforms import AugmentConfig, apply_augmentation, mask_to_bbox


def _make_synthetic_dataset(root: Path, n: int = 4, size: int = 128) -> None:
    img_dir = root / "images"
    od_dir = root / "masks" / "od"
    oc_dir = root / "masks" / "oc"
    splits_dir = root / "splits"
    for d in (img_dir, od_dir, oc_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)
    stems = []
    rng = np.random.default_rng(0)
    for i in range(n):
        stem = f"sample_{i:03d}"
        stems.append(stem)
        img = (rng.integers(0, 255, size=(size, size, 3))).astype(np.uint8)
        od = np.zeros((size, size), dtype=np.uint8)
        oc = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(od, (size // 2, size // 2), size // 4, 255, -1)
        cv2.circle(oc, (size // 2, size // 2), size // 8, 255, -1)
        cv2.imwrite(str(img_dir / f"{stem}.png"), img)
        cv2.imwrite(str(od_dir / f"{stem}.png"), od)
        cv2.imwrite(str(oc_dir / f"{stem}.png"), oc)
    with open(splits_dir / "official.json", "w") as f:
        json.dump({"train": stems[:2], "val": stems[2:3], "test": stems[3:]}, f)


def test_mask_to_bbox():
    m = np.zeros((20, 20), dtype=np.uint8)
    m[5:15, 7:17] = 1
    box = mask_to_bbox(m)
    assert box is not None
    assert tuple(box.tolist()) == (7.0, 5.0, 17.0, 15.0)


def test_augmentation_keeps_shape():
    rng = np.random.default_rng(0)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    od = np.zeros((32, 32), dtype=np.uint8)
    oc = np.zeros((32, 32), dtype=np.uint8)
    cfg = AugmentConfig()
    img2, od2, oc2 = apply_augmentation(rng, img, od, oc, cfg)
    assert img2.shape == img.shape
    assert od2.shape == od.shape
    assert oc2.shape == oc.shape


def test_dataloader_yields_expected_keys(tmp_path: Path):
    _make_synthetic_dataset(tmp_path, n=4, size=128)
    cfg = OmegaConf.create(
        {
            "name": "synthetic",
            "root": str(tmp_path),
            "image_dir": "images",
            "od_mask_dir": "masks/od",
            "oc_mask_dir": "masks/oc",
            "splits_file": "splits/official.json",
            "image_size": 128,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.25, 0.25, 0.25],
            "batch_size": 2,
            "num_workers": 0,
            "augment": {"hflip": 0.5, "vflip": 0.5, "rotation_choices": [0, 90, 180, 270]},
        }
    )
    loader = build_dataloader(cfg, "train")
    batch = next(iter(loader))
    assert "images" in batch
    assert "targets" in batch
    assert batch["images"].shape == (2, 3, 128, 128)
    assert len(batch["targets"]) == 2
    assert batch["targets"][0]["od_boxes"].shape == (1, 4)
