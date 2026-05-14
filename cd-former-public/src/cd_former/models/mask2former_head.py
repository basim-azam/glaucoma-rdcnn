"""Two-query Mask2Former-style segmentation head with heteroscedastic uncertainty.

Architecture (one batch element):
    encoder_features: (N_patches, C)  — from RETFound ViT-L
              |
              v
    Fidelity-Aware Projection (Dino U-Net trick):
      lift the 1D patch sequence to a 2D feature pyramid via learned upsampling
      [H/16, H/8, H/4, H/2, H]
              |
              v
    Two learnable queries q_disc, q_cup
      q_cup_init = q_disc.detach().clone() at init time (cup-from-disc)
              |
              v
    3 transformer decoder layers with masked cross-attention:
      query -> attends to pixel embeddings -> mask logits (B, 2, H, W)
              |
              v
    Heteroscedastic head:
      For each of 2 channels, also predict log_var (B, 2, H, W).
      Trainer uses the heteroscedastic loss formulation.

This is the modern replacement for R-DCNN's bbox->ellipse pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SegHeadOutput:
    """Output of the two-query head."""

    mask_logits: torch.Tensor  # (B, 2, H, W) — channel 0 OD, channel 1 OC
    log_var: torch.Tensor  # (B, 2, H, W) — heteroscedastic uncertainty
    aux_logits: list[torch.Tensor]  # deep supervision at intermediate scales


class FidelityAwareProjection(nn.Module):
    """Lift (B, N, C) ViT patch features to a (B, C_out, H, W) feature map.

    Per Dino U-Net 2025 — a learnable upsampling that preserves the spatial
    arrangement implicit in the patch sequence.
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, patches: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        b, n, c = patches.shape
        h, w = grid_hw
        assert h * w == n, f"grid {h}x{w} != N={n}"
        x = self.norm(self.proj(patches))
        return x.transpose(1, 2).reshape(b, -1, h, w)


class _UpBlock(nn.Module):
    """2x upsample + conv block for the feature pyramid."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class _MaskedCrossAttn(nn.Module):
    """Single masked cross-attention layer (Mask2Former-style)."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.norm1(queries)
        out, _ = self.attn(q, keys, keys, attn_mask=attn_mask, need_weights=False)
        queries = queries + out
        queries = queries + self.ffn(self.norm2(queries))
        return queries


class Mask2FormerSegHead(nn.Module):
    """Two-query Mask2Former-style head with heteroscedastic uncertainty.

    Args:
        encoder_dim: feature dimension of the ViT (1024 for ViT-L).
        decoder_dim: internal dim for queries / attention (256 typical).
        num_decoder_layers: number of cross-attention decoder layers.
        target_size: output mask spatial size (matches ROI crop, default 512).
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        decoder_dim: int = 256,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        target_size: int = 512,
    ) -> None:
        super().__init__()
        self.target_size = target_size
        self.proj = FidelityAwareProjection(in_dim=encoder_dim, out_dim=decoder_dim)

        # Feature pyramid: [H/16, H/8, H/4, H/2, H]
        self.up_h8 = _UpBlock(decoder_dim, decoder_dim)
        self.up_h4 = _UpBlock(decoder_dim, decoder_dim)
        self.up_h2 = _UpBlock(decoder_dim, decoder_dim)
        self.up_h1 = _UpBlock(decoder_dim, decoder_dim)

        # Two learnable queries; cup is initialized from disc at construction time
        self.query_disc = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)
        # We initialize cup_query by cloning disc_query at first forward pass,
        # then they evolve independently. We need cup to be a real Parameter,
        # so we register it but copy at __init__:
        # Independent random init breaks the disc/cup symmetry. The original
        # "init cup from disc" was a bug: identical queries with a symmetric
        # loss had no inductive bias for channel assignment, causing seed-
        # dependent OD/OC confusion across runs.
        self.query_cup = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)

        self.decoder_layers = nn.ModuleList(
            [_MaskedCrossAttn(decoder_dim, num_heads=num_heads) for _ in range(num_decoder_layers)]
        )

        # Mask prediction MLP — turns query into a mask-embedding for dot-product with pixels
        self.mask_mlp = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )

        # Heteroscedastic head: predicts log_var per pixel for each of 2 channels.
        # Implemented as a small conv on the final pixel features.
        self.log_var_conv = nn.Conv2d(decoder_dim, 2, 1)

    def _decode_pyramid(self, pixel_h16: torch.Tensor) -> list[torch.Tensor]:
        """Build the feature pyramid from the lowest-res ViT-patch grid."""
        h8 = self.up_h8(pixel_h16)
        h4 = self.up_h4(h8)
        h2 = self.up_h2(h4)
        h1 = self.up_h1(h2)
        return [pixel_h16, h8, h4, h2, h1]

    def forward(self, patch_features: torch.Tensor, grid_hw: tuple[int, int]) -> SegHeadOutput:
        """patch_features: (B, N, C_enc). Returns SegHeadOutput."""
        b = patch_features.shape[0]

        # 1) Lift patches into a 2D feature map at H/16 resolution
        pixel_h16 = self.proj(patch_features, grid_hw)  # (B, C_dec, H_p, W_p)
        pyramid = self._decode_pyramid(pixel_h16)  # 5 scales

        # 2) Decode queries with masked cross-attention against the lowest-res feats
        # (we attend at the coarse scale for tractability; mask logits are computed
        # against finer pyramid levels)
        b_dec, c_dec, hp, wp = pixel_h16.shape
        keys = pixel_h16.flatten(2).transpose(1, 2)  # (B, N, C_dec)
        queries = torch.cat(
            [self.query_disc.expand(b, -1, -1), self.query_cup.expand(b, -1, -1)],
            dim=1,
        )  # (B, 2, C_dec)
        for layer in self.decoder_layers:
            queries = layer(queries, keys)

        # 3) Compute mask logits at each pyramid level via query·pixel dot-product
        mask_emb = self.mask_mlp(queries)  # (B, 2, C_dec)

        aux_logits: list[torch.Tensor] = []
        for level_feats in pyramid[:-1]:  # exclude finest for aux supervision
            # dot product over channels: (B, 2, C) · (B, C, H, W) -> (B, 2, H, W)
            mask_logits = torch.einsum("bqc,bchw->bqhw", mask_emb, level_feats)
            aux_logits.append(mask_logits)

        # Final-resolution prediction
        final_pixel_feats = pyramid[-1]
        final_logits = torch.einsum("bqc,bchw->bqhw", mask_emb, final_pixel_feats)
        # Upsample to target_size if not already there
        if final_logits.shape[-1] != self.target_size:
            final_logits = F.interpolate(
                final_logits,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )

        # 4) Heteroscedastic log-variance head
        log_var_feats = final_pixel_feats
        log_var = self.log_var_conv(log_var_feats)
        if log_var.shape[-1] != self.target_size:
            log_var = F.interpolate(
                log_var,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        # Clamp log_var to a sane range to avoid exp() blow-up early in training
        log_var = log_var.clamp(min=-10.0, max=4.0)

        return SegHeadOutput(
            mask_logits=final_logits,
            log_var=log_var,
            aux_logits=aux_logits,
        )


def build_seg_head(
    encoder_dim: int = 1024,
    decoder_dim: int = 256,
    num_decoder_layers: int = 3,
    target_size: int = 512,
) -> Mask2FormerSegHead:
    return Mask2FormerSegHead(
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        num_decoder_layers=num_decoder_layers,
        target_size=target_size,
    )
