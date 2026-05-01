"""Visualization helpers for inference + figure generation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def overlay_predictions(
    image_rgb: np.ndarray,
    od_mask: np.ndarray,
    oc_mask: np.ndarray,
    *,
    od_color: tuple[int, int, int] = (0, 255, 0),
    oc_color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    """Overlay OD and OC masks on the image. Returns an RGB uint8 array."""
    out = image_rgb.copy()
    overlay = out.copy()
    overlay[od_mask > 0] = od_color
    overlay[oc_mask > 0] = oc_color
    blended = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    # Re-overlay edges
    od_edge = cv2.Canny((od_mask * 255).astype(np.uint8), 50, 150)
    oc_edge = cv2.Canny((oc_mask * 255).astype(np.uint8), 50, 150)
    blended[od_edge > 0] = od_color
    blended[oc_edge > 0] = oc_color
    return blended


def save_overlay(
    path: str | Path, image_rgb: np.ndarray, od_mask: np.ndarray, oc_mask: np.ndarray
) -> None:
    img = overlay_predictions(image_rgb, od_mask, oc_mask)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
