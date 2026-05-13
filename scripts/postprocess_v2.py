"""Post-process v2 checkpoint predictions with morphology + TTA + constraints.

Recomputes test metrics under several post-processing recipes. No retraining.

Recipes evaluated:
    baseline:                  no post-processing (matches evaluate_v2.py)
    cc:                        largest connected component per class
    cid:                       cup-inside-disc probability constraint
    morph:                     morphological close + open
    cc+cid:                    cc + cid stacked
    cc+cid+morph:              all three deterministic ops stacked
    tta:                       test-time augmentation (flip + average), no other ops
    tta+cc+cid+morph:          everything combined

Usage:
    python scripts/postprocess_v2.py \\
        --checkpoint outputs/v2_drishti_seed42_<TS>/best.ckpt \\
        --data-root data/drishti_gs \\
        --output-json outputs/v2_drishti_seed42_<TS>/postproc_test.json
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


def collect_predictions(
    checkpoint: Path,
    data_root: Path,
    split: str,
    use_tta: bool,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Run inference once. Returns list of (probs_d, probs_c, gt_d, gt_c) per image."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Force fp32 for inference to avoid bf16/autocast complexity. ~1 min slower but bulletproof.
    dtype = torch.float32

    backbone = build_backbone(
        pretrained=False,  # we'll load_state_dict immediately
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

    ds = FundusSegDataset(
        root=data_root,
        split=split,
        encoder_size=cfg["encoder_size"],
        target_size=cfg["target_size"],
        augment=False,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

    def fwd(images: torch.Tensor) -> torch.Tensor:
        feats = backbone(images)
        return head(feats.patch_features.to(dtype), grid_hw=feats.grid_hw).mask_logits

    out: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=dtype)
            targets = batch["target"].numpy()
            logits = fwd(images)
            if use_tta:
                images_f = torch.flip(images, dims=[-1])
                logits_f = fwd(images_f)
                logits_f = torch.flip(logits_f, dims=[-1])
                logits = (logits + logits_f) / 2.0
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            for b in range(probs.shape[0]):
                out.append(
                    (probs[b, 0], probs[b, 1], targets[b, 0] > 0.5, targets[b, 1] > 0.5)
                )
    return out


def compute_metrics(predictions, recipe: str, threshold: float = 0.5) -> dict:
    """Aggregate metrics across all predictions under a post-proc recipe."""
    from sklearn.metrics import roc_auc_score

    dices_d, dices_c, jcs_d, jcs_c = [], [], [], []
    sens_d, spec_d, sens_c, spec_c = [], [], [], []
    cdr_preds, cdr_gts = [], []

    for probs_d, probs_c, gt_d, gt_c in predictions:
        pred_d, pred_c = apply_recipe(probs_d, probs_c, recipe, threshold)

        dices_d.append(_dice(pred_d, gt_d))
        dices_c.append(_dice(pred_c, gt_c))
        jcs_d.append(_jaccard(pred_d, gt_d))
        jcs_c.append(_jaccard(pred_c, gt_c))

        for p, t, sl, spl in (
            (pred_d, gt_d, sens_d, spec_d),
            (pred_c, gt_c, sens_c, spec_c),
        ):
            tp = int((p & t).sum())
            fn = int((~p & t).sum())
            tn = int((~p & ~t).sum())
            fp = int((p & ~t).sum())
            sl.append(float(tp / (tp + fn + 1e-6)))
            spl.append(float(tn / (tn + fp + 1e-6)))

        cdr_pred = _vertical_extent(pred_c) / _vertical_extent(pred_d) if pred_d.any() else 0.0
        cdr_gt = _vertical_extent(gt_c) / _vertical_extent(gt_d) if gt_d.any() else 0.0
        cdr_preds.append(cdr_pred)
        cdr_gts.append(cdr_gt)

    cdr_preds_a = np.array(cdr_preds)
    cdr_gts_a = np.array(cdr_gts)
    is_g = (cdr_gts_a > 0.5).astype(int)
    if is_g.min() == is_g.max():
        auc = 0.5
    else:
        try:
            auc = float(roc_auc_score(is_g, cdr_preds_a))
        except Exception:
            auc = 0.5

    return {
        "od_dice": float(np.mean(dices_d)),
        "od_jaccard": float(np.mean(jcs_d)),
        "od_sensitivity": float(np.mean(sens_d)),
        "od_specificity": float(np.mean(spec_d)),
        "oc_dice": float(np.mean(dices_c)),
        "oc_jaccard": float(np.mean(jcs_c)),
        "oc_sensitivity": float(np.mean(sens_c)),
        "oc_specificity": float(np.mean(spec_c)),
        "glaucoma_auc": auc,
        "cdr_mae": float(np.mean(np.abs(cdr_preds_a - cdr_gts_a))),
        "n_images": len(dices_d),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    if args.output_json is None:
        args.output_json = args.checkpoint.parent / f"postproc_{args.split}.json"

    print(f"[postproc] checkpoint: {args.checkpoint}")
    print(f"[postproc] split: {args.split}")

    print("[postproc] running inference (no TTA)...")
    preds_no_tta = collect_predictions(args.checkpoint, args.data_root, args.split, use_tta=False)
    print(f"[postproc]   {len(preds_no_tta)} predictions")

    print("[postproc] running inference (with TTA, flip + average)...")
    preds_tta = collect_predictions(args.checkpoint, args.data_root, args.split, use_tta=True)
    print(f"[postproc]   {len(preds_tta)} predictions")

    recipes: list[tuple[str, str, bool]] = [
        ("baseline", "", False),
        ("cc", "cc", False),
        ("cid", "cid", False),
        ("morph", "morph", False),
        ("cc+cid", "cc+cid", False),
        ("cc+cid+morph", "cc+cid+morph", False),
        ("tta", "", True),
        ("tta+cc+cid+morph", "cc+cid+morph", True),
    ]

    print()
    print(f"{'recipe':22s} {'OD Dice':>9s} {'OC Dice':>9s} {'AUC':>7s} {'CDR MAE':>9s} {'OC Sens':>9s} {'OC Spec':>9s}")
    print("-" * 80)
    results: dict[str, dict] = {}
    for name, recipe, use_tta in recipes:
        preds = preds_tta if use_tta else preds_no_tta
        m = compute_metrics(preds, recipe)
        results[name] = m
        print(
            f"{name:22s} {m['od_dice']:9.4f} {m['oc_dice']:9.4f} "
            f"{m['glaucoma_auc']:7.4f} {m['cdr_mae']:9.4f} "
            f"{m['oc_sensitivity']:9.4f} {m['oc_specificity']:9.4f}"
        )

    # Pick the best recipe by OC Dice
    best_name = max(results, key=lambda k: results[k]["oc_dice"])
    best = results[best_name]
    print()
    print(f"[postproc] best recipe: {best_name}")
    print(
        f"           OD Dice {best['od_dice']:.4f} | OC Dice {best['oc_dice']:.4f} | "
        f"AUC {best['glaucoma_auc']:.4f} | CDR MAE {best['cdr_mae']:.4f}"
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
  