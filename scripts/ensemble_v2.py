"""3-seed ensemble inference for CD-Former.

Loads all 3 RETFound checkpoints for a given dataset, averages sigmoid
probabilities across seeds before thresholding, then applies the
cc+cid+morph post-processing recipe. Expected +1-2 pp OC Dice over
best single seed.

Usage:
    python scripts/ensemble_v2.py \
        --checkpoints outputs/v2_drishti_seed42_*/best.ckpt \
                      outputs/v2_drishti_seed43_*/best.ckpt \
                      outputs/v2_drishti_seed44_*/best.ckpt \
        --data-root data/drishti_gs \
        --output-json outputs/v2_drishti_ensemble.json
"""

from __future__ import annotations

import argparse
import json
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


def _dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    inter = (a & b).sum()
    return float((2 * inter + eps) / (a.sum() + b.sum() + eps))


def _jaccard(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    return float(((a & b).sum() + eps) / ((a | b).sum() + eps))


def _vertical_extent(mask: np.ndarray) -> float:
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return 0.0
    return float(rows[-1] - rows[0] + 1) / mask.shape[0]


def load_model(checkpoint: Path, device: torch.device) -> tuple:
    """Build backbone+head, load weights, return as eval-mode tuple."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True, type=Path, nargs="+",
                    help="2+ checkpoints to ensemble")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--use-tta", action="store_true",
                    help="Also do horizontal-flip TTA per model")
    args = ap.parse_args()

    print(f"[ensemble] ensembling {len(args.checkpoints)} checkpoints:")
    for c in args.checkpoints:
        print(f"  - {c}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # fp32 inference matches postprocess_v2.py

    # Load all models
    models = []
    for ckpt_path in args.checkpoints:
        backbone, head, cfg = load_model(ckpt_path, device)
        models.append((backbone, head, cfg))

    # Use the first cfg for dataset settings (they should all match)
    cfg0 = models[0][2]
    ds = FundusSegDataset(
        root=args.data_root,
        split=args.split,
        encoder_size=cfg0["encoder_size"],
        target_size=cfg0["target_size"],
        augment=False,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
    print(f"[ensemble] {len(ds)} test images")

    # Collect ensemble predictions
    predictions = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=dtype)
            targets = batch["target"].numpy()

            # Average probs across all models
            probs_sum = None
            for backbone, head, _ in models:
                feats = backbone(images)
                logits = head(feats.patch_features.to(dtype), grid_hw=feats.grid_hw).mask_logits
                probs = torch.sigmoid(logits.float())
                if args.use_tta:
                    images_f = torch.flip(images, dims=[-1])
                    feats_f = backbone(images_f)
                    logits_f = head(feats_f.patch_features.to(dtype), grid_hw=feats_f.grid_hw).mask_logits
                    logits_f = torch.flip(logits_f, dims=[-1])
                    probs_f = torch.sigmoid(logits_f.float())
                    probs = (probs + probs_f) / 2.0
                probs_sum = probs if probs_sum is None else probs_sum + probs

            avg_probs = (probs_sum / len(models)).cpu().numpy()

            for b in range(avg_probs.shape[0]):
                predictions.append((
                    avg_probs[b, 0], avg_probs[b, 1],
                    targets[b, 0] > 0.5, targets[b, 1] > 0.5
                ))

    print(f"[ensemble] {len(predictions)} ensembled predictions")

    # Evaluate per recipe
    from sklearn.metrics import roc_auc_score

    def evaluate(recipe: str) -> dict:
        od_dices, oc_dices = [], []
        od_jcs, oc_jcs = [], []
        sens_d, spec_d, sens_c, spec_c = [], [], [], []
        cdr_preds, cdr_gts = [], []
        for pd_, pc_, gd, gc in predictions:
            pred_d, pred_c = apply_recipe(pd_, pc_, recipe)
            od_dices.append(_dice(pred_d, gd))
            oc_dices.append(_dice(pred_c, gc))
            od_jcs.append(_jaccard(pred_d, gd))
            oc_jcs.append(_jaccard(pred_c, gc))
            for p, t, sl, spl in (
                (pred_d, gd, sens_d, spec_d),
                (pred_c, gc, sens_c, spec_c),
            ):
                tp = int((p & t).sum()); fn = int((~p & t).sum())
                tn = int((~p & ~t).sum()); fp = int((p & ~t).sum())
                sl.append(float(tp / (tp + fn + 1e-6)))
                spl.append(float(tn / (tn + fp + 1e-6)))
            cdr_pred = _vertical_extent(pred_c) / _vertical_extent(pred_d) if pred_d.any() else 0.0
            cdr_gt = _vertical_extent(gc) / _vertical_extent(gd) if gd.any() else 0.0
            cdr_preds.append(cdr_pred); cdr_gts.append(cdr_gt)
        cdr_preds_a = np.array(cdr_preds); cdr_gts_a = np.array(cdr_gts)
        is_g = (cdr_gts_a > 0.5).astype(int)
        auc = 0.5 if is_g.min() == is_g.max() else float(roc_auc_score(is_g, cdr_preds_a))
        return {
            "od_dice": float(np.mean(od_dices)),
            "od_jaccard": float(np.mean(od_jcs)),
            "oc_dice": float(np.mean(oc_dices)),
            "oc_jaccard": float(np.mean(oc_jcs)),
            "od_sensitivity": float(np.mean(sens_d)),
            "od_specificity": float(np.mean(spec_d)),
            "oc_sensitivity": float(np.mean(sens_c)),
            "oc_specificity": float(np.mean(spec_c)),
            "glaucoma_auc": auc,
            "cdr_mae": float(np.mean(np.abs(cdr_preds_a - cdr_gts_a))),
            "n_images": len(od_dices),
        }

    print()
    print(f"{'recipe':22s} {'OD Dice':>9s} {'OC Dice':>9s} {'AUC':>7s} {'CDR MAE':>9s}")
    print("-" * 60)
    recipes = ["", "cc", "cc+cid", "cc+cid+morph"]
    results = {}
    for r in recipes:
        m = evaluate(r)
        name = r if r else "baseline"
        results[name] = m
        print(f"{name:22s} {m['od_dice']:9.4f} {m['oc_dice']:9.4f} "
              f"{m['glaucoma_auc']:7.4f} {m['cdr_mae']:9.4f}")

    best = max(results, key=lambda k: results[k]["oc_dice"])
    print()
    print(f"[ensemble] best recipe: {best}")
    print(f"           OD {results[best]['od_dice']:.4f} | OC {results[best]['oc_dice']:.4f} | "
          f"AUC {results[best]['glaucoma_auc']:.4f} | CDR MAE {results[best]['cdr_mae']:.4f}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({
            "checkpoints": [str(c) for c in args.checkpoints],
            "tta": args.use_tta,
            "results": results,
            "best_recipe": best,
        }, f, indent=2)
    print(f"[ensemble] wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
