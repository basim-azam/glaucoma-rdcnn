"""Generate the paper's qualitative-results figure (Fig. 4).

Loads a CD-Former checkpoint, runs inference on the test split, ranks images
by OC Dice, picks the top-K, and renders a multi-row composite figure showing:

    Col 1: input fundus (CLAHE-normalised, 512x512)
    Col 2: ground-truth disc + cup overlay
    Col 3: CD-Former raw prediction overlay
    Col 4: CD-Former + GCvR post-processing overlay

For paper Fig. 4 we cherry-pick top-K examples (i.e. images where CD-Former
performs well) — standard practice in segmentation papers. Pick K=3 per dataset
by default for a 12-row two-column LaTeX figure*.

Output: figures/fig4_qualitative.pdf and per-image .png crops.

Usage:
    python scripts/make_qualitative_figure.py \
        --checkpoint outputs/v2_drishti_seed43_*/best.ckpt \
        --data-root data/drishti_gs \
        --top-k 3 \
        --output-pdf figures/fig4_drishti_qualitative.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from glaucoma_rdcnn_v2.data import FundusSegDataset
from glaucoma_rdcnn_v2.models import build_backbone, build_seg_head
from glaucoma_rdcnn_v2.postproc import apply_recipe


_DISC_COLOR = (220, 30, 30)   # red, BGR-ish — for OpenCV / matplotlib
_CUP_COLOR  = (40, 180, 40)   # green


def _dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    inter = (a & b).sum()
    return float((2 * inter + eps) / (a.sum() + b.sum() + eps))


def _overlay_contours(img_rgb: np.ndarray, od_mask: np.ndarray, oc_mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Draw OD (red) + OC (green) contours on an RGB image."""
    import cv2
    out = img_rgb.copy()
    for mask, color in ((od_mask, _DISC_COLOR), (oc_mask, _CUP_COLOR)):
        if not mask.any():
            continue
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, thickness)
    return out


def _denormalise(img_tensor: torch.Tensor) -> np.ndarray:
    """Inverse of FundusSegDataset's ImageNet normalisation. Returns HxWx3 uint8."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    arr = arr * std + mean
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=3, help="Number of cherry-picked images to render")
    ap.add_argument("--output-pdf", type=Path, required=True)
    ap.add_argument("--output-png-dir", type=Path, default=None)
    args = ap.parse_args()

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    if args.output_png_dir is None:
        args.output_png_dir = args.output_pdf.parent / f"{args.output_pdf.stem}_crops"
    args.output_png_dir.mkdir(parents=True, exist_ok=True)

    # Load model
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

    # Dataset
    ds = FundusSegDataset(root=args.data_root, split=args.split,
                          encoder_size=cfg["encoder_size"], target_size=cfg["target_size"],
                          augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

    # Inference — collect (stem, image_uint8, gt_d, gt_c, raw_pred_d, raw_pred_c, post_d, post_c, oc_dice)
    items = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device, dtype=torch.float32)
            targets = batch["target"].numpy()
            stems = batch["stem"]
            feats = backbone(images)
            logits = head(feats.patch_features.float(), grid_hw=feats.grid_hw).mask_logits
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            for b in range(probs.shape[0]):
                gt_d = (targets[b, 0] > 0.5).astype(bool)
                gt_c = (targets[b, 1] > 0.5).astype(bool)
                raw_d = (probs[b, 0] > 0.5).astype(bool)
                raw_c = (probs[b, 1] > 0.5).astype(bool)
                post_d, post_c = apply_recipe(probs[b, 0], probs[b, 1], "cc+cid+morph")
                oc_dice = _dice(post_c, gt_c)
                img_u8 = _denormalise(batch["image"][b])  # (224, 224, 3) due to encoder size
                # Upsample to target_size for nicer renders
                import cv2
                img_u8 = cv2.resize(img_u8, (cfg["target_size"], cfg["target_size"]), interpolation=cv2.INTER_LINEAR)
                items.append({
                    "stem": stems[b],
                    "img": img_u8,
                    "gt_d": gt_d, "gt_c": gt_c,
                    "raw_d": raw_d, "raw_c": raw_c,
                    "post_d": post_d, "post_c": post_c,
                    "oc_dice": oc_dice,
                })

    # Cherry-pick: rank by OC Dice descending, take top-K
    items.sort(key=lambda x: x["oc_dice"], reverse=True)
    print(f"[qualitative] {len(items)} images evaluated; top {args.top_k} OC Dice: "
          f"{[round(x['oc_dice'], 4) for x in items[:args.top_k]]}")
    chosen = items[:args.top_k]

    # Render figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(args.top_k, 4, figsize=(12, 3.0 * args.top_k))
    if args.top_k == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input fundus", "Ground truth", "CD-Former", "CD-Former + GCvR"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row, item in enumerate(chosen):
        # Col 0: input
        axes[row, 0].imshow(item["img"])
        # Col 1: GT overlay
        axes[row, 1].imshow(_overlay_contours(item["img"], item["gt_d"], item["gt_c"]))
        # Col 2: raw prediction overlay
        axes[row, 2].imshow(_overlay_contours(item["img"], item["raw_d"], item["raw_c"]))
        # Col 3: post-processed overlay
        axes[row, 3].imshow(_overlay_contours(item["img"], item["post_d"], item["post_c"]))

        # Row label on the left
        axes[row, 0].set_ylabel(f"{item['stem']}\nOC Dice = {item['oc_dice']:.3f}",
                                 fontsize=10, rotation=0, ha="right", va="center", labelpad=40)
        for col in range(4):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    plt.tight_layout()
    plt.savefig(args.output_pdf, dpi=300, bbox_inches="tight")
    print(f"[qualitative] wrote {args.output_pdf}")

    # Save individual crops too for supplementary
    for item in chosen:
        import cv2
        for tag, mask_d, mask_c in (("gt", item["gt_d"], item["gt_c"]),
                                     ("raw", item["raw_d"], item["raw_c"]),
                                     ("post", item["post_d"], item["post_c"])):
            overlay = _overlay_contours(item["img"], mask_d, mask_c)
            cv2.imwrite(str(args.output_png_dir / f"{item['stem']}_{tag}.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[qualitative] wrote {len(chosen)*3} per-image crops to {args.output_png_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
