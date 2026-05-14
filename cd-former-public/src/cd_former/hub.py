"""Convenience loaders that bridge the GitHub code package and the HuggingFace
Hub model repository.

End users call:

    >>> from cd_former import load_pretrained
    >>> backbone, head, cfg = load_pretrained("drishti_seed43")

…and the function handles checkpoint discovery, HF download, MD5 verification,
and model construction in one shot.

Canonical checkpoint aliases:
    drishti_seed42, drishti_seed43, drishti_seed44
    rimone_seed42,  rimone_seed43,  rimone_seed44

The HF repo id is read from src/cd_former/_release.py (or overridable via
the HF_REPO env var).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import torch

# Filled at release time. Override with HF_REPO env var if needed.
HF_REPO_DEFAULT = "basimazam/cd-former-weights"

# Canonical alias → filename mapping (matches docs/manifest.json)
_ALIASES = {
    "drishti_seed42": "cd_former_drishti_seed42.ckpt",
    "drishti_seed43": "cd_former_drishti_seed43.ckpt",
    "drishti_seed44": "cd_former_drishti_seed44.ckpt",
    "rimone_seed42":  "cd_former_rimone_seed42.ckpt",
    "rimone_seed43":  "cd_former_rimone_seed43.ckpt",
    "rimone_seed44":  "cd_former_rimone_seed44.ckpt",
}

# Expected MD5 sums (mirrors docs/manifest.json). Update if you re-train.
_MD5 = {
    "cd_former_drishti_seed42.ckpt": "a4fb79ba00c6cd110b19420a55f100c1",
    "cd_former_drishti_seed43.ckpt": "87ac87993c1f38dc55891b0762c2d091",
    "cd_former_drishti_seed44.ckpt": "c792459c024b849c58bfc8d0d3589a7f",
    "cd_former_rimone_seed42.ckpt":  "fe522fe78278b79fdaa910ffe85eeb3e",
    "cd_former_rimone_seed43.ckpt":  "1f086afb237cfde6048d2044e1a0e127",
    "cd_former_rimone_seed44.ckpt":  "2648bb9468a630f4e4c46573736e44d4",
}


def list_pretrained() -> list[str]:
    """Return the list of recognised checkpoint aliases."""
    return list(_ALIASES)


def _verify_md5(path: Path, expected: str) -> None:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        raise RuntimeError(
            f"MD5 mismatch for {path.name}: got {got}, expected {expected}. "
            f"Delete the file and re-download."
        )


def download_checkpoint(
    alias: str,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """Fetch a checkpoint from HuggingFace Hub and return the local path.

    Args:
        alias: one of `list_pretrained()` results.
        repo_id: HF repo. Defaults to HF_REPO env var or HF_REPO_DEFAULT.
        token: HF token. Defaults to HF_TOKEN env var (required for private repos).
        cache_dir: HF cache directory. Defaults to HF default.

    Returns:
        Path to the local checkpoint file.
    """
    if alias not in _ALIASES:
        raise KeyError(
            f"Unknown alias {alias!r}. Available: {sorted(_ALIASES)}"
        )
    filename = _ALIASES[alias]
    repo_id = repo_id or os.environ.get("HF_REPO") or HF_REPO_DEFAULT
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    from huggingface_hub import hf_hub_download

    local = Path(hf_hub_download(repo_id=repo_id, filename=filename, token=token, cache_dir=cache_dir))
    _verify_md5(local, _MD5[filename])
    return local


def load_pretrained(
    alias: str,
    device: str | torch.device = "cpu",
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """Load a pretrained CD-Former checkpoint and return ready-to-use modules.

    Returns:
        (backbone, head, cfg) — all on `device`, set to eval() mode.
    """
    from .models import build_backbone, build_seg_head

    ckpt_path = download_checkpoint(alias, repo_id=repo_id, token=token, cache_dir=cache_dir)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    backbone = build_backbone(
        pretrained=False,
        image_size=cfg["encoder_size"],
        prefer_fallback=cfg.get("prefer_fallback_backbone", False),
    )
    head = build_seg_head(
        encoder_dim=backbone.feature_dim,
        decoder_dim=cfg["decoder_dim"],
        num_decoder_layers=cfg["num_decoder_layers"],
        target_size=cfg["target_size"],
    )
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    backbone.to(device).eval()
    head.to(device).eval()
    return backbone, head, cfg


__all__ = ["list_pretrained", "download_checkpoint", "load_pretrained", "HF_REPO_DEFAULT"]
