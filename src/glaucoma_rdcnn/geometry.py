"""Inscribed-ellipse fitting and CDR computation.

The R-DCNN paper turns predicted axis-aligned bboxes into segmentation masks by
fitting the largest ellipse inscribed in each box. CDR is then the ratio of the
vertical OC diameter to the vertical OD diameter.
"""

from __future__ import annotations

import numpy as np
import torch


def inscribed_ellipse_mask(
    box_xyxy: torch.Tensor | np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    """Rasterize the ellipse inscribed in ``box_xyxy`` onto an image-sized mask.

    Args:
        box_xyxy: ``(x0, y0, x1, y1)`` in image pixels.
        image_size: ``(H, W)`` of the output mask.
    Returns:
        ``np.uint8`` array of shape ``(H, W)`` with values in {0, 1}.
    """
    if isinstance(box_xyxy, torch.Tensor):
        box_xyxy = box_xyxy.detach().cpu().numpy()
    x0, y0, x1, y1 = box_xyxy.astype(np.float64)
    H, W = image_size
    mask = np.zeros((H, W), dtype=np.uint8)
    if x1 <= x0 or y1 <= y0:
        return mask
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    rx = max((x1 - x0) / 2.0, 1e-6)
    ry = max((y1 - y0) / 2.0, 1e-6)
    yy, xx = np.ogrid[:H, :W]
    inside = ((xx - cx) ** 2) / (rx**2) + ((yy - cy) ** 2) / (ry**2) <= 1.0
    mask[inside] = 1
    return mask


def vertical_diameter(box_xyxy: torch.Tensor | np.ndarray) -> float:
    """Return the vertical diameter (y1 - y0) of an axis-aligned bbox."""
    if isinstance(box_xyxy, torch.Tensor):
        box_xyxy = box_xyxy.detach().cpu().numpy()
    return float(box_xyxy[3] - box_xyxy[1])


def compute_cdr(od_box: torch.Tensor | np.ndarray, oc_box: torch.Tensor | np.ndarray) -> float:
    """Compute Cup-to-Disc Ratio = vertical_OC / vertical_OD."""
    od = vertical_diameter(od_box)
    oc = vertical_diameter(oc_box)
    if od <= 0:
        return float("nan")
    return oc / od


def boxes_to_masks(
    od_boxes: torch.Tensor,
    oc_boxes: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 box per head → ellipse masks.

    Returns ``(od_mask, oc_mask)`` each of shape ``(H, W)`` uint8.
    """
    H, W = image_size
    od_mask = np.zeros((H, W), dtype=np.uint8)
    oc_mask = np.zeros((H, W), dtype=np.uint8)
    if od_boxes.numel() > 0:
        od_mask = inscribed_ellipse_mask(od_boxes[0], (H, W))
    if oc_boxes.numel() > 0:
        oc_mask = inscribed_ellipse_mask(oc_boxes[0], (H, W))
    return od_mask, oc_mask
