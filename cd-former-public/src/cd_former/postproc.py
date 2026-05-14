"""Inference-time post-processing primitives for OD/OC dense masks.

All ops operate on probability maps or binary masks of shape (H, W). No training.

Recipes (composable):
    cc:    keep only the largest connected component per class
    cid:   cup-inside-disc — mask cup probability where disc probability is low
    morph: morphological close then open (fills small holes, removes spurs)
    tta:   handled at inference time (flip image, forward, flip back, average)
"""

from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage as ndi


def largest_connected_component(mask_bin: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    if not mask_bin.any():
        return mask_bin
    labeled, n_components = ndi.label(mask_bin)
    if n_components <= 1:
        return mask_bin
    sizes = ndi.sum(mask_bin, labeled, range(1, n_components + 1))
    largest = int(np.argmax(sizes)) + 1
    return labeled == largest


def cup_inside_disc(cup_prob: np.ndarray, disc_prob: np.ndarray, disc_threshold: float = 0.3) -> np.ndarray:
    """Mask cup probability where disc probability is below threshold.

    The cup is anatomically inside the optic disc. Network leakage outside
    the disc region is a common failure mode that this fixes for free at
    inference time.
    """
    inside = disc_prob > disc_threshold
    return cup_prob * inside.astype(cup_prob.dtype)


def morph_close_open(mask_bin: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Closing fills small holes; opening removes thin protrusions."""
    if not mask_bin.any():
        return mask_bin
    kernel = np.ones((ksize, ksize), np.uint8)
    m = mask_bin.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    return m.astype(bool)


def apply_recipe(
    disc_prob: np.ndarray,
    cup_prob: np.ndarray,
    recipe: str,
    threshold: float = 0.5,
    morph_ksize: int = 5,
    cid_disc_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a recipe of post-processing ops, return (disc_bin, cup_bin).

    recipe is a '+'-joined string of op names, e.g. 'cc+cid+morph'.
    Order is fixed:
      1. cid    — clip cup probability by disc region (operates on probs)
      2. threshold — sigmoid > threshold to get binary masks
      3. morph  — closing then opening
      4. cc     — largest connected component
    """
    ops = set(recipe.split("+")) if recipe else set()
    if "cid" in ops:
        cup_prob = cup_inside_disc(cup_prob, disc_prob, disc_threshold=cid_disc_threshold)
    disc_bin = disc_prob > threshold
    cup_bin = cup_prob > threshold
    if "morph" in ops:
        disc_bin = morph_close_open(disc_bin, ksize=morph_ksize)
        cup_bin = morph_close_open(cup_bin, ksize=morph_ksize)
    if "cc" in ops:
        disc_bin = largest_connected_component(disc_bin)
     