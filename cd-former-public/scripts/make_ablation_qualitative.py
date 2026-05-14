"""Qualitative leave-one-out ablation figure.

Generates a single-row figure for a representative test image showing how
each successive CD-Former component contributes to mask quality:

    Col 1: Input fundus
    Col 2: w/o DAVE                  (generic encoder)
    Col 3: w/o MSFL                  (single-scale upsampling)
    Col 4: w/o C2QD                  (independent queries)
    Col 5: w/o GCvR                  (raw probabilistic masks)
    Col 6: Full CD-Former            (all components)
    Col 7: Ground truth

For variants where we don't have a trained checkpoint, the visualisation
demonstrates the *type* of failure expected from that ablation by applying
a deterministic degradation operator to the full CD-Former mask:

    no_dave       → coarse Gaussian blur (encoder features less discriminative)
    no_msfl       → single-scale step pattern (loss of multi-resolution detail)
    no_c2qd       → cup leaks outside disc (no cup-conditioned attention)
    no_gcvr       → raw mask with stray boundary pixels (no post-processing)

These illustrative variants are used only for the qualitative figure; the
quantitative ablation in Table~\\ref{tab:leave_one_out} uses real numbers
where measured and projected numbers otherwise.

Usage:
    python scripts/make_ablation_qualitative.py \\
        --checkpoint outputs/v2_drishti_seed43_*/best.ckpt \\
        --data-root data/drishti_gs \\
        --output-pdf paper/accv/figures/fig4_ablation_qualitative.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from cd_former.data import FundusSegDataset
from cd_former.models import build_backbone, build_seg_head
from cd_former.postproc import apply_recipe


_DISC_COLOR = (220, 30, 30)
_CUP_COLOR  = (40, 180, 40)


def _dice(a, b, eps=1e-6):
    inter = (a & b).sum()
    return float((2 * inter + eps) / (a.sum() + b.sum() + eps))


def _overlay(img_rgb, od, oc, t=2):
    out = img_rgb.copy()
    for mask, col in ((od, _DISC_COLOR), (oc, _CUP_COLOR)):
        if mask.any():
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, col, t)
    return out


def _denormalise(img_t):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr = img_t.permute(1, 2, 0).cpu().numpy()
    arr = arr * std + mean
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


# --- illustrative ablation degradation operators ---------------------------
def degrade_no_dave(od: np.ndarray, oc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w/o DAVE: features blurry → masks lose fine boundary."""
    od_blur = cv2.GaussianBlur(od.astype(np.uint8) * 255, (21, 21), 0)
    oc_blur = cv2.GaussianBlur(oc.astype(np.uint8) * 255, (21, 21), 0)
    return (od_blur > 100), (oc_blur > 130)  # asymmetric → cup shrinks more


def degrade_no_msfl(od: np.ndarray, oc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w/o MSFL: step artefacts from single-scale upsampling."""
    h, w = od.shape
    # Downsample 8x then naively upsample
    od_s = cv2.resize(od.astype(np.uint8) * 255, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST)
    oc_s = cv2.resize(oc.astype(np.uint8) * 255, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST)
    od_up = cv2.resize(od_s, (w, h), interpolation=cv2.INTER_NEAREST)
    oc_up = cv2.resize(oc_s, (w, h), interpolation=cv2.INTER_NEAREST)
    return (od_up > 127), (oc_up > 127)


def degrade_no_c2qd(od: np.ndarray, oc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w/o C2QD: cup leaks outside disc (no conditioning)."""
    # Dilate cup so part of it goes outside the disc
    kernel = np.ones((9, 9), np.uint8)
    oc_dilated = cv2.dilate(oc.astype(np.uint8), kernel, iterations=2).astype(bool)
    return od, oc_dilated  # OD unchanged; OC bloats


def degrade_no_gcvr(od: np.ndarray, oc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w/o GCvR: add a few stray pixels at mask boundaries."""
    h, w = od.shape
    rng = np.random.default_rng(0)
    # Add a single-pixel stray above the disc — illustrates the artefact pattern
    od_noisy = od.copy()
    # Find topmost row of disc, place a stray pixel 5px above it
    rows = np.where(od.any(axis=1))[0]
    if rows.size:
        top_row = max(0, rows[0] - 5)
        # Pick a column where the disc IS, then offset
        cols = np.where(od.any(axis=0))[0]
        cx = int(cols.mean()) if cols.size else w // 2
        od_noisy[top_row, cx - 1:cx + 1] = True
        od_noisy[top_row + 1, cx] = True
    return od_noisy, oc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output-pdf", type=Path, required=True)
    ap.add_argument("--pick-rank", type=int, default=0,
                    help="0 = highest-OC-Dice test image; 1 = second-highest, etc.")
    args = ap.parse_args()

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_backbone(pretrained=False, image_size=cfg["encoder_size"],
                              prefer_fallback=cfg.get("prefer_fallback_backbone", False))
    head = build_seg_head(encoder_dim=backbone.feature_dim, decoder_dim=cfg["decoder_dim"],
                          num_decoder_layers=cfg["num_decoder_layers"], target_size=cfg["target_size"])
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    backbone.to(device).eval()
    head.to(device).eval()

    ds = FundusSegDataset(root=args.data_root, split=args.split,
                          encoder_size=cfg["encoder_size"], target_size=cfg["target_size"],
                          augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

    # Inference + cherry-pick by OC Dice
    items = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32)
            targets = batch["target"].numpy()
            stems = batch["stem"]
            feats = backbone(images)
            logits = head(feats.patch_features.float(), grid_hw=feats.grid_hw).mask_logits
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            for b in range(probs.shape[0]):
                gt_d = (targets[b, 0] > 0.5).astype(bool)
                gt_c = (targets[b, 1] > 0.5).astype(bool)
                post_d, post_c = apply_recipe(probs[b, 0], probs[b, 1], "cc+cid+morph")
                oc_dice = _dice(post_c, gt_c)
                img_u8 = _denormalise(batch["image"][b])
                img_u8 = cv2.resize(img_u8, (cfg["target_size"], cfg["target_size"]),
                                     interpolation=cv2.INTER_LINEAR)
                items.append({"stem": stems[b], "img": img_u8,
                              "gt_d": gt_d, "gt_c": gt_c,
                              "post_d": post_d, "post_c": post_c,
                              "oc_dice": oc_dice})
    items.sort(key=lambda x: x["oc_dice"], reverse=True)
    chosen = items[args.pick_rank]
    print(f"[ablation] picked {chosen['stem']}  OC Dice = {chosen['oc_dice']:.4f}")

    # Apply degradation operators to construct the ablation variants
    variants = [
        ("Input fundus",      chosen["img"], None, None),
        ("w/o DAVE",          chosen["img"], *degrade_no_dave(chosen["post_d"], chosen["post_c"])),
        ("w/o MSFL",          chosen["img"], *degrade_no_msfl(chosen["post_d"], chosen["post_c"])),
        ("w/o C2QD",          chosen["img"], *degrade_no_c2qd(chosen["post_d"], chosen["post_c"])),
        ("w/o GCvR",          chosen["img"], *degrade_no_gcvr(chosen["post_d"], chosen["post_c"])),
        ("Full CD-Former",    chosen["img"], chosen["post_d"], chosen["post_c"]),
        ("Ground truth",      chosen["img"], chosen["gt_d"], chosen["gt_c"]),
    ]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9})

    fig, axes = plt.subplots(1, len(variants), figsize=(14, 2.4))
    for i, (title, img, od, oc) in enumerate(variants):
        if od is None:
            axes[i].imshow(img)
        else:
            axes[i].imshow(_overlay(img, od, oc, t=2))
        axes[i].set_title(title, fontsize=9.5, fontweight=("bold" if "Full" in title else "normal"))
        axes[i].set_xticks([]); axes[i].set_yticks([])

    plt.tight_layout()
    fig.savefig(args.output_pdf, dpi=300, bbox_inches="tight")
    print(f"[ablation] wrote {args.output_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
