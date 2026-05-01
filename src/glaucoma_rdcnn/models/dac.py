"""Dense Atrous Convolution (DAC) block.

Originally from CE-Net (Gu et al., TMI 2019). The R-DCNN paper appends a DAC block
to the deepest stage of the backbone to widen the receptive field while preserving
spatial resolution. Each branch is a 3x3 conv with progressively larger dilation;
the outputs are summed (densely, like the original DAC) and projected to
``out_channels``.

PAPER-UNSPEC: the paper references DAC but does not list dilation rates explicitly.
We use ``[1, 3, 6, 12]`` as the standard CE-Net schedule. Override via the model
config.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DenseAtrousConv(nn.Module):
    """A four-branch dense atrous convolution block.

    Args:
        in_channels:  number of input channels (e.g., 512 for ResNet-34 layer4).
        out_channels: number of output channels.
        rates:        dilation rates for the four branches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rates: Sequence[int] = (1, 3, 6, 12),
    ) -> None:
        super().__init__()
        if len(rates) != 4:
            raise ValueError(f"DAC expects 4 dilation rates, got {len(rates)}")

        # Branch 1: 1x1 (rate 1)
        self.b1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        # Branch 2: 1x1 → 3x3@r2
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Conv2d(
                out_channels, out_channels, 3, padding=rates[1], dilation=rates[1], bias=False
            ),
        )
        # Branch 3: 1x1 → 3x3@r2 → 3x3@r3
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Conv2d(
                out_channels, out_channels, 3, padding=rates[1], dilation=rates[1], bias=False
            ),
            nn.Conv2d(
                out_channels, out_channels, 3, padding=rates[2], dilation=rates[2], bias=False
            ),
        )
        # Branch 4: 1x1 → 3x3@r2 → 3x3@r3 → 3x3@r4
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Conv2d(
                out_channels, out_channels, 3, padding=rates[1], dilation=rates[1], bias=False
            ),
            nn.Conv2d(
                out_channels, out_channels, 3, padding=rates[2], dilation=rates[2], bias=False
            ),
            nn.Conv2d(
                out_channels, out_channels, 3, padding=rates[3], dilation=rates[3], bias=False
            ),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.b1(x) + self.b2(x) + self.b3(x) + self.b4(x)
        return self.act(self.bn(y))
