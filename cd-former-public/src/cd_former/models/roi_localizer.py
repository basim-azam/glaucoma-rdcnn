"""Stage-1 ROI localizer: Circular-Hough + tiny U-Net hybrid.

Workflow:
    1. CLAHE on the green channel (matches our existing preprocess).
    2. Circular-Hough gives an initial (cx, cy, r) guess.
    3. A 4-layer U-Net regression head refines the guess from the full image.
    4. Crop a 512x512 ROI centered on the refined center.

The U-Net is small (~2M params) and trained jointly with the seg head.
Hough provides the prior so the U-Net never starts from random.

If Hough fails (no circles), we fall back to the image center as the
U-Net's input prior and rely on the U-Net regression.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ROIPrediction:
    """Predicted disc center (cx, cy) and radius r in pixel coords of the input."""

    cx: float
    cy: float
    r: float
    source: str  # "hough", "unet", "fallback"


def hough_initial_guess(
    image: np.ndarray,
    *,
    apply_clahe: bool = True,
    min_radius_frac: float = 0.05,
    max_radius_frac: float = 0.25,
) -> ROIPrediction | None:
    """Run Circular-Hough on the green channel after CLAHE.

    Args:
        image: H x W x 3 uint8 fundus image.
        apply_clahe: enhance contrast before edge detection.
        min/max_radius_frac: bounds as a fraction of image min(H, W).

    Returns:
        ROIPrediction with source='hough', or None if no circles found.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image, got {image.shape}")

    h, w = image.shape[:2]
    green = image[:, :, 1]
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        green = clahe.apply(green)
    green = cv2.GaussianBlur(green, (9, 9), 2)

    min_r = int(min(h, w) * min_radius_frac)
    max_r = int(min(h, w) * max_radius_frac)

    circles = cv2.HoughCircles(
        green,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 2,
        param1=100,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None

    # circles shape: (1, N, 3) where last dim is (x, y, r)
    circles = np.round(circles[0]).astype(int)
    # Prefer circles roughly in the central 60% of the image (disc is rarely on edge)
    cx_lo, cx_hi = int(0.2 * w), int(0.8 * w)
    cy_lo, cy_hi = int(0.2 * h), int(0.8 * h)
    central = [
        c for c in circles if cx_lo <= c[0] <= cx_hi and cy_lo <= c[1] <= cy_hi
    ]
    chosen = central[0] if central else circles[0]
    return ROIPrediction(cx=float(chosen[0]), cy=float(chosen[1]), r=float(chosen[2]), source="hough")


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ROILocalizer(nn.Module):
    """Tiny 4-stage U-Net regression head that predicts (cx, cy, log_r).

    Input: downsampled fundus (default 256x256, 3 channels) + Hough prior (3 channels)
    Output: 3 scalars per image (cx, cy, log_r) in normalized [-1, 1] coords.
    """

    def __init__(self, in_channels: int = 6, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = _ConvBlock(in_channels, c)
        self.enc2 = _ConvBlock(c, c * 2)
        self.enc3 = _ConvBlock(c * 2, c * 4)
        self.enc4 = _ConvBlock(c * 4, c * 8)
        self.pool = nn.AvgPool2d(2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 8, c * 4),
            nn.GELU(),
            nn.Linear(c * 4, 3),  # cx, cy, log_r
        )

    @staticmethod
    def _make_prior_channels(
        h: int, w: int, hough: ROIPrediction | None, device: torch.device
    ) -> torch.Tensor:
        """Build 3 prior channels: dist-from-center, hough-cx, hough-cy as 2D maps."""
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        if hough is not None:
            d = torch.sqrt((xs - hough.cx) ** 2 + (ys - hough.cy) ** 2)
            d = (d / max(h, w)).clamp(0, 1)
            cx_map = torch.full((h, w), hough.cx / w, device=device)
            cy_map = torch.full((h, w), hough.cy / h, device=device)
        else:
            d = torch.zeros((h, w), device=device)
            cx_map = torch.full((h, w), 0.5, device=device)
            cy_map = torch.full((h, w), 0.5, device=device)
        return torch.stack([d, cx_map, cy_map], dim=0)  # (3, H, W)

    def forward(
        self,
        image: torch.Tensor,
        hough_priors: list[ROIPrediction | None],
    ) -> torch.Tensor:
        """image: (B, 3, H, W). hough_priors: length-B list. Returns (B, 3): (cx, cy, log_r)."""
        b, _, h, w = image.shape
        priors = torch.stack(
            [self._make_prior_channels(h, w, hough_priors[i], image.device) for i in range(b)]
        )
        x = torch.cat([image, priors], dim=1)
        x = self.enc1(x)
        x = self.enc2(self.pool(x))
        x = self.enc3(self.pool(x))
        x = self.enc4(self.pool(x))
        return self.head(x)  # raw (cx_norm, cy_norm, log_r_norm) — trainer will map to pixels


def crop_roi(
    image: np.ndarray,
    cx: float,
    cy: float,
    crop_size: int = 512,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a crop_size x crop_size box centered on (cx, cy), pad if needed.

    Returns:
        (cropped_image, (x0, y0, x1, y1)) — the absolute crop coordinates.
    """
    h, w = image.shape[:2]
    half = crop_size // 2
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    # Pad if the box goes off-image
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    if any((pad_l, pad_t, pad_r, pad_b)):
        image = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t

    return image[y0:y1, x0:x1], (x0 - pad_l, y0 - pad_t, x1 - pad_l, y1 - pad_t)
