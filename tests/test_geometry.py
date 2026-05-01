"""Tests for ellipse fitting and CDR."""

import numpy as np
import torch

from glaucoma_rdcnn.geometry import (
    boxes_to_masks,
    compute_cdr,
    inscribed_ellipse_mask,
    vertical_diameter,
)


def test_inscribed_ellipse_area_within_box():
    box = np.array([10, 20, 50, 80], dtype=np.float32)  # 40x60
    mask = inscribed_ellipse_mask(box, (100, 100))
    # Ellipse area ≈ π * (20) * (30) = 1885; box area = 40*60 = 2400
    area = mask.sum()
    assert area > 1500
    assert area < 2200


def test_vertical_diameter_simple():
    box = np.array([0, 5, 10, 25])
    assert vertical_diameter(box) == 20.0


def test_compute_cdr():
    od = np.array([0, 0, 100, 100], dtype=np.float32)  # 100 vertical
    oc = np.array([20, 20, 80, 70], dtype=np.float32)  # 50 vertical
    assert abs(compute_cdr(od, oc) - 0.5) < 1e-6


def test_boxes_to_masks_shapes():
    od = torch.tensor([[10.0, 20.0, 90.0, 80.0]])
    oc = torch.tensor([[30.0, 35.0, 70.0, 65.0]])
    od_m, oc_m = boxes_to_masks(od, oc, (100, 100))
    assert od_m.shape == (100, 100)
    assert oc_m.shape == (100, 100)
    # cup mask area should be smaller than disc mask area
    assert oc_m.sum() < od_m.sum()
