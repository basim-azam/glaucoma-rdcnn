"""v2 evaluation entry point. Loads a best.ckpt and reports test metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    from glaucoma_rdcnn_v2.data import FundusSegDataset
    from glaucoma_rdcnn_v2.evaluator import evaluate_model
    from glaucoma_rdcnn_v2.models import build_backbone, build_seg_head
    from torch.utils.data import DataLoader

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.get("dtype_str", "bfloat16") == "bfloat16" else torch.float32

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

    ds = FundusSegDataset(
        root=args.data_root,
        split=args.split,
        encoder_size=cfg["encoder_size"],
        target_size=cfg["target_size"],
        augment=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    def fwd(images: torch.Tensor) -> torch.Tensor:
        images = images.to(device, dtype=dtype)
        feats = backbone(images)
        out = head(feats.patch_features.to(dtype), grid_hw=feats.grid_hw)
        return out.mask_logits

    metrics = evaluate_model(model_forward=fwd, loader=loader, device=device, dtype=dtype)
    print(metrics.as_log_line(args.split))

    # Write next to the checkpoint
    out_path = args.checkpoint.parent / f"eval_{args.split}.json"
    payload = {
        k: getattr(metrics, k)
        for k in (
            "od_dice", "od_jaccard", "od_overlap_err", "od_sensitivity", "od_specificity",
            "oc_dice", "oc_jaccard", "oc_overlap_err", "oc_sensitivity", "oc_specificity",
            "glaucoma_auc", "cdr_mae", "n_images",
        )
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[evaluate_v2] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
