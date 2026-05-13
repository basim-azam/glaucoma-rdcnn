"""Chromatic Preprocessing (CP) module.

In retinal fundus photography, different colour channels carry different
diagnostic information:
  * Red   channel: best contrasts the optic DISC — high reflectance against
                   the dark choroid background; vasculature visible.
  * Green channel: best contrasts the optic CUP — chromatic gradient between
                   cup (lighter) and rim (darker neuro-retinal tissue).
  * Blue  channel: dominated by absorption from the pigment epithelium;
                   contributes least to OD/OC discrimination.

The Chromatic Preprocessing module replaces the standard RGB triplet fed
to the encoder with a per-structure-informative arrangement:

    R  →  channel 0 (OD-emphasised)
    G  →  channel 1 (OC-emphasised)
    L  →  channel 2 (luminance fallback, (R+G+B)/3)

where L preserves general fundus appearance for downstream features that
benefit from the broader signal. The encoder receives this 3-channel
input as if it were RGB, then ImageNet-mean/std normalisation is applied
as in standard practice.

This is a 3-line preprocessing change with no architectural cost.
"""

from __future__ import annotations

import numpy as np


def chromatic_preprocess(rgb: np.ndarray, mode: str = "rg_lum") -> np.ndarray:
    """Reformat a (H, W, 3) RGB uint8 fundus image per chromatic-preprocessing rules.

    Args:
        rgb: input (H, W, 3) uint8 RGB image.
        mode: 'rg_lum' = [R, G, L] where L = (R+G+B)/3.  Default.
              'rg_zero' = [R, G, 0]  — drop blue completely.
              'rrr_gg' = [R, R, G] — emphasise red.
              'identity' = unchanged RGB.

    Returns:
        (H, W, 3) uint8 array.
    """
    if mode == "identity":
        return rgb
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) image, got {rgb.shape}")

    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    L = (r + g + b) / 3.0

    if mode == "rg_lum":
        out = np.stack([r, g, L], axis=-1)
    elif mode == "rg_zero":
        out = np.stack([r, g, np.zeros_like(r)], axis=-1)
    elif mode == "rrr_gg":
        out = np.stack([r, r, g], axis=-1)
    else:
        raise ValueError(f"unknown chromatic mode: {mode!r}")

    return out.clip(0, 255).astype(np.uint8)
