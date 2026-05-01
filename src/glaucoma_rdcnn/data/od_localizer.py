"""Lightweight OD localizer used for the 800x800 ROI crop.

The R-DCNN paper cites Zou et al. 2018 (ref [39]) for OD localization. We provide
a fast, dependency-light replacement that's good enough for ROI cropping:

1. Extract the red channel (where OD is brightest in fundus images).
2. Apply CLAHE to even out illumination.
3. Smooth and find the global intensity peak in a local-mean image.

This is intentionally simple — for hard cases the test pipeline can fall back to
the GT OD centroid (when masks are available) or to a learned localizer if you
want to swap one in.
"""

from __future__ import annotations

import cv2
import numpy as np


def locate_optic_disc(
    image_bgr: np.ndarray,
    smooth_sigma: float = 31.0,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> tuple[int, int]:
    """Return ``(x, y)`` of the OD center in pixel coordinates."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("expected a BGR image")
    red = image_bgr[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    eq = clahe.apply(red)
    blurred = cv2.GaussianBlur(eq, ksize=(0, 0), sigmaX=smooth_sigma)
    _min_val, _max_val, _min_loc, max_loc = cv2.minMaxLoc(blurred)
    return int(max_loc[0]), int(max_loc[1])


def crop_roi(
    image_bgr: np.ndarray,
    center_xy: tuple[int, int],
    size: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a ``size x size`` patch centered on ``center_xy``, with reflection padding.

    Returns the patch and the ``(top_left_x, top_left_y)`` offset of the crop in
    the original image coordinates (after padding-aware adjustment).
    """
    h, w = image_bgr.shape[:2]
    half = size // 2
    cx, cy = center_xy
    x0, y0 = cx - half, cy - half
    pad_left = max(-x0, 0)
    pad_top = max(-y0, 0)
    pad_right = max(x0 + size - w, 0)
    pad_bottom = max(y0 + size - h, 0)
    padded = cv2.copyMakeBorder(
        image_bgr, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REFLECT_101
    )
    x0p, y0p = x0 + pad_left, y0 + pad_top
    crop = padded[y0p : y0p + size, x0p : x0p + size]
    return crop, (x0 - pad_left, y0 - pad_top)
