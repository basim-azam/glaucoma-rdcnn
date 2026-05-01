"""ResNet-34 + DAC backbone.

Returns a single-level feature map with reduced channel count, suitable for
plugging into a torchvision RPN. We strip ResNet's FC head and the final
average pool, then append the DAC block.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34

from glaucoma_rdcnn.models.dac import DenseAtrousConv


class ResNet34DAC(nn.Module):
    """ResNet-34 trunk with a DAC block at the end.

    The output is a single-level tensor of shape ``(N, out_channels, H/32, W/32)``.
    ``out_channels`` defaults to 256 (DAC output) regardless of the 512-channel
    layer4 output.
    """

    def __init__(
        self,
        pretrained: bool = True,
        out_channels: int = 256,
        dac_rates: tuple[int, int, int, int] = (1, 3, 6, 12),
    ) -> None:
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet34(weights=weights)

        # Keep stem + 4 stages
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        self.dac = DenseAtrousConv(
            in_channels=512,
            out_channels=out_channels,
            rates=dac_rates,
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dac(x)
        # torchvision detection heads expect a dict[level, tensor]
        return {"0": x}

    @torch.no_grad()
    def feature_shape(self, x: torch.Tensor) -> tuple[int, int, int, int]:
        out = self.forward(x)["0"]
        return tuple(out.shape)  # type: ignore[return-value]


def build_backbone(cfg: Any) -> ResNet34DAC:
    """Construct a backbone from a Hydra/OmegaConf node."""
    return ResNet34DAC(
        pretrained=bool(cfg.backbone.pretrained),
        out_channels=int(cfg.dac.out_channels),
        dac_rates=tuple(cfg.dac.rates),
    )
