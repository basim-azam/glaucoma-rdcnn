"""RETFound MAE ViT-L/16 backbone with graceful fallback ladder.

Loading strategy (in order):
    1. timm.create_model('vit_large_patch16_224') + load HF state_dict
       — RETFound is published as a state-dict-only checkpoint.
    2. transformers.ViTMAEModel.from_pretrained — works for some mirrors.
    3. DINOv2-Large via timm — documented fallback per Risk 1.

The frozen / unfrozen schedule lives in the trainer, not here. This module
just builds the encoder and exposes feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class BackboneOutput:
    """Patch features (B, N, C) and optional CLS token (B, C)."""

    patch_features: torch.Tensor
    cls_token: Optional[torch.Tensor]
    grid_hw: tuple[int, int]  # (H_p, W_p) so the seg head can reshape


class RETFoundBackbone(nn.Module):
    """RETFound MAE ViT-L/16, loaded via timm with the HF state_dict."""

    HF_REPO = "YukunZhou/RETFound_mae_natureCFP"
    FALLBACK_DINOV2 = "facebook/dinov2-large"
    FEATURE_DIM = 1024  # ViT-L

    def __init__(
        self,
        pretrained: bool = True,
        image_size: int = 224,
        prefer_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.source = "uninitialized"
        self.vit, self.feature_dim, self.patch_size = self._build_vit(
            pretrained=pretrained, prefer_fallback=prefer_fallback
        )
        self._frozen = False

    def _build_vit(self, pretrained: bool, prefer_fallback: bool):
        """Build the ViT, trying RETFound first then DINOv2."""
        import timm  # imported lazily so module is importable without timm

        if not prefer_fallback:
            try:
                vit = timm.create_model(
                    "vit_large_patch16_224",
                    pretrained=False,
                    num_classes=0,  # we want features, not logits
                )
                if pretrained:
                    self._load_retfound_weights(vit)
                self.source = f"RETFound ({self.HF_REPO})"
                return vit, 1024, 16
            except Exception as e:  # noqa: BLE001
                print(
                    f"[backbone] RETFound load failed ({type(e).__name__}: {e});"
                    " falling back to DINOv2-Large."
                )

        # Fallback: DINOv2-Large via timm
        vit = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=0,
            img_size=self.image_size,
        )
        self.source = f"DINOv2-Large ({self.FALLBACK_DINOV2})"
        return vit, 1024, 14

    def _load_retfound_weights(self, vit: nn.Module) -> None:
        """Download RETFound state_dict from HuggingFace and load into timm ViT-L."""
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            repo_id=self.HF_REPO,
            filename="RETFound_mae_natureCFP.pth",
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # RETFound wraps weights under "model" key in the .pth
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        # Strip mask-decoder weights that don't apply to encoder-only use
        encoder_only = {
            k: v for k, v in state.items() if not k.startswith(("decoder", "mask_token"))
        }
        missing, unexpected = vit.load_state_dict(encoder_only, strict=False)
        print(
            f"[backbone] RETFound loaded: {len(encoder_only)} keys,"
            f" {len(missing)} missing, {len(unexpected)} unexpected"
        )

    def freeze(self) -> None:
        """Freeze all parameters. Trainer will call unfreeze() at epoch 10."""
        for p in self.vit.parameters():
            p.requires_grad = False
        self._frozen = True

    def unfreeze(self) -> None:
        for p in self.vit.parameters():
            p.requires_grad = True
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """x: (B, 3, H, W). Returns BackboneOutput with patch grid features."""
        # timm ViT forward_features returns (B, 1 + N_patches, C) with CLS token first
        feats = self.vit.forward_features(x)
        if feats.dim() == 3 and feats.shape[1] >= 2:
            cls = feats[:, 0]
            patches = feats[:, 1:]
        else:
            cls = None
            patches = feats

        # Compute the patch grid (assumes square input)
        N = patches.shape[1]
        side = int(round(N**0.5))
        if side * side != N:
            # Some ViTs include register tokens; trim them
            patches = patches[:, -(side * side) :]
        return BackboneOutput(
            patch_features=patches,
            cls_token=cls,
            grid_hw=(side, side),
        )


def build_backbone(
    pretrained: bool = True, image_size: int = 224, prefer_fallback: bool = False
) -> RETFoundBackbone:
    """Public constructor — call this from the trainer / smoke tests."""
    return RETFoundBackbone(
        pretrained=pretrained,
        image_size=image_size,
        prefer_fallback=prefer_fallback,
    )
