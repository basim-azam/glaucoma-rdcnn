"""v2 (rdcnn-modern) training entry point.

Usage:
    python scripts/train_v2.py --data-root data/drishti_gs --output-dir outputs/v2_drishti_<TS>
    python scripts/train_v2.py --data-root data/rim_one_v3 --output-dir outputs/v2_rimone_<TS> --epochs 60
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure src/ is on path so this works both inside and outside a pip install
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="If omitted, picks outputs/v2_<dataset>_<timestamp>")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Override output dir name. Combined with --output-base if provided")
    ap.add_argument("--output-base", type=Path, default=Path("outputs"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--freeze-epochs", type=int, default=10)
    ap.add_argument("--base-lr", type=float, default=5e-4)
    ap.add_argument("--encoder-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--decoder-dim", type=int, default=256)
    ap.add_argument("--num-decoder-layers", type=int, default=3)
    ap.add_argument("--prefer-fallback-backbone", action="store_true",
                    help="Skip RETFound and use DINOv2-Large directly")
    ap.add_argument("--use-heteroscedastic", action="store_true",
                    help="Enable heteroscedastic loss term (w=0.05)")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--chromatic-mode", choices=["identity", "rg_lum", "rg_zero", "rrr_gg"], default="identity",
                    help="Chromatic preprocessing mode (CP module): identity uses RGB; rg_lum emphasises R for OD, G for OC, luminance fallback")
    args = ap.parse_args()

    # Resolve output dir
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = args.run_name or f"v2_{args.data_root.name}_{ts}"
        args.output_dir = args.output_base / name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from cd_former.trainer import TrainerConfig, Trainer

    cfg = TrainerConfig(
        data_root=str(args.data_root),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        freeze_epochs=args.freeze_epochs,
        base_lr=args.base_lr,
        encoder_lr_ratio=args.encoder_lr_ratio,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        prefer_fallback_backbone=args.prefer_fallback_backbone,
        use_heteroscedastic=args.use_heteroscedastic,
        decoder_dim=args.decoder_dim,
        num_decoder_layers=args.num_decoder_layers,
        dtype_str=args.dtype,
        chromatic_mode=args.chromatic_mode,
    )

    print(f"[train_v2] config: {json.dumps(cfg.__dict__, indent=2)}")
    print(f"[train_v2] output_dir: {args.output_dir}")

    trainer = Trainer(cfg)
    result = trainer.train()

    # Save final metrics
    test_metrics_dict = {
        k: getattr(result["test_metrics"], k)
        for k in (
            "od_dice", "od_jaccard", "od_overlap_err", "od_sensitivity", "od_specificity",
            "oc_dice", "oc_jaccard", "oc_overlap_err", "oc_sensitivity", "oc_specificity",
            "glaucoma_auc", "cdr_mae", "n_images",
        )
    }
    summary = {
        "config": cfg.__dict__,
        "best_oc_dice_val": result["best_oc_dice_val"],
        "best_ckpt_epoch": result["best_ckpt_epoch"],
        "test_metrics": test_metrics_dict,
        "history": result["history"],
    }
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train_v2] wrote {summary_path}")
    print(f"[train_v2] {result['test_metrics'].as_log_line('TEST')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
