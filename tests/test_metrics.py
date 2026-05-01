"""Sanity checks for the metric definitions."""

import numpy as np

from glaucoma_rdcnn.metrics import (
    aggregate,
    dice,
    glaucoma_auc,
    jaccard,
    overlap_error,
    sensitivity,
    specificity,
)


def test_dice_perfect_match():
    a = np.zeros((10, 10), dtype=np.uint8)
    a[2:7, 2:7] = 1
    assert dice(a, a) == 1.0
    assert jaccard(a, a) == 1.0
    assert overlap_error(a, a) == 0.0


def test_dice_no_overlap():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    a[0:3, 0:3] = 1
    b[7:10, 7:10] = 1
    assert dice(a, b) == 0.0
    assert jaccard(a, b) == 0.0


def test_sensitivity_specificity():
    pred = np.array([[1, 0], [1, 0]])
    gt = np.array([[1, 1], [0, 0]])
    # TP=1 FN=1 TN=1 FP=1
    assert sensitivity(pred, gt) == 0.5
    assert specificity(pred, gt) == 0.5


def test_glaucoma_auc_perfect():
    cdr = np.array([0.3, 0.4, 0.6, 0.7])
    labels = np.array([0, 0, 1, 1])
    assert glaucoma_auc(cdr, labels) == 1.0


def test_aggregate_mean():
    rows = [{"a": 1.0, "b": 0.5}, {"a": 0.0, "b": 0.5}]
    out = aggregate(rows)
    assert out["a"] == 0.5
    assert out["b"] == 0.5
