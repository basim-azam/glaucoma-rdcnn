"""Segmentation and classification metrics from the R-DCNN paper.

Definitions used:

* ``dice``      = 2|A ∩ B| / (|A| + |B|)
* ``jaccard``   = |A ∩ B| / |A ∪ B|
* ``overlap_error`` = 1 - jaccard
* ``sensitivity`` = TP / (TP + FN)
* ``specificity`` = TN / (TN + FP)
* glaucoma classification AUC: receiver operating characteristic of CDR
  thresholded vs. ground-truth glaucoma label.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _to_bool(arr: np.ndarray) -> np.ndarray:
    return arr.astype(bool)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = _to_bool(pred)
    g = _to_bool(gt)
    inter = np.logical_and(p, g).sum()
    s = p.sum() + g.sum()
    return float(2 * inter / s) if s > 0 else 1.0


def jaccard(pred: np.ndarray, gt: np.ndarray) -> float:
    p = _to_bool(pred)
    g = _to_bool(gt)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union > 0 else 1.0


def overlap_error(pred: np.ndarray, gt: np.ndarray) -> float:
    return 1.0 - jaccard(pred, gt)


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    p = _to_bool(pred)
    g = _to_bool(gt)
    tp = np.logical_and(p, g).sum()
    fn = np.logical_and(~p, g).sum()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0


def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    p = _to_bool(pred)
    g = _to_bool(gt)
    tn = np.logical_and(~p, ~g).sum()
    fp = np.logical_and(p, ~g).sum()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0


def glaucoma_auc(cdr_scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC for glaucoma classification from CDR.

    Args:
        cdr_scores: 1D array of predicted CDR values.
        labels:     1D 0/1 array (1 = glaucoma).
    """
    cdr_scores = np.asarray(cdr_scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.sum() == 0 or labels.sum() == labels.size:
        # AUC undefined with one class — return NaN so caller can decide
        return float("nan")
    return float(roc_auc_score(labels, cdr_scores))


def aggregate(per_image_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across a list of per-image dicts."""
    if not per_image_metrics:
        return {}
    keys = per_image_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in per_image_metrics])) for k in keys}
