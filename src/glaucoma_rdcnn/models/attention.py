"""Disc Attention Module.

The CPN sees features that have been re-attended through the predicted OD region
(soft mask). This is implemented as channel-spatial attention: the OD soft mask
(rasterized from the predicted bbox, blurred) gates the input feature map,
followed by a 1x1 conv that mixes channels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def boxes_to_soft_mask(
    boxes: torch.Tensor,
    feature_size: tuple[int, int],
    image_size: tuple[int, int],
    sigma: float = 4.0,
) -> torch.Tensor:
    """Rasterize boxes onto a feature-resolution soft mask, then Gaussian-blur.

    Args:
        boxes: ``(N, 4)`` tensor in ``xyxy`` image-pixel coordinates.
        feature_size: ``(Hf, Wf)`` of the target feature map.
        image_size: ``(Hi, Wi)`` of the original image.
        sigma: stddev of the Gaussian blur in feature-map pixels.
    Returns:
        Tensor of shape ``(N, 1, Hf, Wf)`` in ``[0, 1]``.
    """
    n = boxes.shape[0]
    Hf, Wf = feature_size
    Hi, Wi = image_size
    sx = Wf / Wi
    sy = Hf / Hi
    masks = boxes.new_zeros((n, 1, Hf, Wf))
    for i in range(n):
        x0, y0, x1, y1 = boxes[i].tolist()
        x0i = max(int(round(x0 * sx)), 0)
        y0i = max(int(round(y0 * sy)), 0)
        x1i = min(int(round(x1 * sx)), Wf)
        y1i = min(int(round(y1 * sy)), Hf)
        if x1i > x0i and y1i > y0i:
            masks[i, 0, y0i:y1i, x0i:x1i] = 1.0
    # Blur with a separable gaussian
    k = max(int(2 * round(2 * sigma) + 1), 3)
    coords = torch.arange(k, device=masks.device, dtype=masks.dtype) - (k // 2)
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).view(1, 1, 1, k)
    masks = F.conv2d(masks, g, padding=(0, k // 2))
    masks = F.conv2d(masks, g.transpose(2, 3), padding=(k // 2, 0))
    return masks.clamp_(0.0, 1.0)


class DiscAttention(nn.Module):
    """Spatial-and-channel attention conditioned on the predicted OD region."""

    def __init__(
        self, in_channels: int, hidden_channels: int = 256, out_channels: int = 256
    ) -> None:
        super().__init__()
        self.spatial_proj = nn.Conv2d(1, in_channels, kernel_size=1, bias=True)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(
        self,
        feat: torch.Tensor,
        od_soft_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention to ``feat`` using ``od_soft_mask``.

        Args:
            feat: ``(N, C, H, W)`` feature map.
            od_soft_mask: ``(N, 1, H, W)`` soft mask in [0, 1].
        Returns:
            Attended feature ``(N, out_channels, H, W)``.
        """
        spatial = torch.sigmoid(self.spatial_proj(od_soft_mask))
        channel = self.channel_gate(feat)
        attn = spatial * channel
        return self.fuse(feat * attn + feat)  # residual fusion
