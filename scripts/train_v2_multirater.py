"""v2 training with heteroscedastic multi-rater loss on DRISHTI-GS.

Reuses the v2 trainer infrastructure but swaps the loss for the
heteroscedastic multi-rater formulation. Requires per-rater masks in
data/drishti_gs/masks/{od,oc}_rater{1,2,3,4}/<stem>.png; falls back to
the merged mask if per-rater masks are unavailable (heteroscedastic
term still active, multi-rater term becomes trivial).

Usage:
    python scripts/train_v2_multirater.py \
        --data-root data/drishti_gs \
        --epochs 60 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--output-base", type=Path, default=Path("outputs"))
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--freeze-epochs", type=int, default=10)
    ap.add_argument("--hetero-warmup-epochs", type=int, default=10)
    ap.add_argument("--agreement-weight", type=float, default=0.5,
                    help="Weight given to rater-agreement gating (0=ignore, 1=hard gate)")
    ap.add_argument("--w-hetero", type=float, default=0.4)
    ap.add_argument("--base-lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    if args.run_name is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_name = f"v2_multirater_{args.data_root.name}_seed{args.seed}_{ts}"
    output_dir = args.output_base / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from glaucoma_rdcnn_v2.data import FundusSegDataset
    from glaucoma_rdcnn_v2.evaluator import evaluate_model
    from glaucoma_rdcnn_v2.models import build_backbone, build_seg_head
    from glaucoma_rdcnn_v2.multi_rater import (
        MultiRaterDriftLoss,
        load_multi_rater_target,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Dataset (we'll do custom collate to load multi-rater targets)
    train_ds = FundusSegDataset(
        root=args.data_root, split="train", encoder_size=224, target_size=512,
        augment=True, seed=args.seed,
    )
    val_ds = FundusSegDataset(
        root=args.data_root, split="val", encoder_size=224, target_size=512,
        augment=False,
    )
    test_ds = FundusSegDataset(
        root=args.data_root, split="test", encoder_size=224, target_size=512,
        augment=False,
    )

    def collate_multirater(items):
        # Each item: {image, target, stem}. Load multi-rater target via stem.
        images = torch.stack([it["image"] for it in items])
        soft_list, agree_list = [], []
        for it in items:
            soft, agree = load_multi_rater_target(
                args.data_root, it["stem"], target_size=512, num_raters=4
            )
            soft_list.append(torch.from_numpy(soft))
            agree_list.append(torch.from_numpy(agree))
        return {
            "image": images,
            "target": torch.stack(soft_list),
            "agreement": torch.stack(agree_list),
            "stem": [it["stem"] for it in items],
        }

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_multirater, pin_memory=True,
    )

    # Model
    backbone = build_backbone(pretrained=True, image_size=224)
    head = build_seg_head(encoder_dim=backbone.feature_dim, decoder_dim=256,
                          num_decoder_layers=3, target_size=512)
    backbone.freeze()
    backbone.to(device); head.to(device)
    print(f"[multirater] backbone source: {backbone.source}")

    loss_fn = MultiRaterDriftLoss(
        w_hetero=args.w_hetero,
        agreement_weight=args.agreement_weight,
        hetero_warmup_epochs=args.hetero_warmup_epochs,
    )

    # Optimizer (decoder-only initially)
    optimizer = torch.optim.AdamW(
        list(head.parameters()), lr=args.base_lr, weight_decay=1e-4
    )
    encoder_unfrozen = False

    history = []
    best_oc_dice = -1.0
    for epoch in range(args.epochs):
        # Unfreeze encoder at freeze_epochs
        if epoch == args.freeze_epochs and not encoder_unfrozen:
            backbone.unfreeze()
            optimizer = torch.optim.AdamW([
                {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": args.base_lr * 0.1},
                {"params": list(head.parameters()), "lr": args.base_lr},
            ], weight_decay=1e-4)
            encoder_unfrozen = True
            print(f"[multirater] epoch {epoch}: unfroze encoder")

        backbone.train(); head.train()
        t0 = time.time()
        epoch_losses = []
        for batch in train_loader:
            images = batch["image"].to(device, dtype=dtype, non_blocking=True)
            soft_target = batch["target"].to(device, dtype=torch.float32, non_blocking=True)
            agreement = batch["agreement"].to(device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=dtype):
                feats = backbone(images)
                out = head(feats.patch_features.to(dtype), grid_hw=feats.grid_hw)

            logits = out.mask_logits.float()
            log_var = out.log_var.float()
            loss, breakdown = loss_fn(logits, log_var, soft_target, agreement, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in head.parameters() if p.grad is not None]
                + [p for p in backbone.parameters() if p.grad is not None],
                max_norm=1.0,
            )
            optimizer.step()
            epoch_losses.append(breakdown["total"])

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))

        # Val using standard FundusSegDataset (single target)
        backbone.eval(); head.eval()
        from torch.utils.data import DataLoader as DL
        val_loader = DL(val_ds, batch_size=8, shuffle=False, num_workers=4)
        val_metrics = evaluate_model(
            model_forward=lambda x: head(
                backbone(x.to(device, dtype=dtype)).patch_features.to(dtype),
                grid_hw=backbone(x.to(device, dtype=dtype)).grid_hw,
            ).mask_logits,
            loader=val_loader, device=device, dtype=dtype,
        )

        wall = time.time() - t0
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_od_dice": val_metrics.od_dice, "val_oc_dice": val_metrics.oc_dice,
            "val_auc": val_metrics.glaucoma_auc, "wall": wall,
        })
        print(f"[ep {epoch:03d}] loss={train_loss:.4f} | val OD={val_metrics.od_dice:.4f} "
              f"OC={val_metrics.oc_dice:.4f} AUC={val_metrics.glaucoma_auc:.4f} | wall={wall:.1f}s")

        if val_metrics.oc_dice > best_oc_dice:
            best_oc_dice = val_metrics.oc_dice
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                        "cfg": vars(args), "epoch": epoch, "val_oc_dice": val_metrics.oc_dice},
                       output_dir / "best.ckpt")
            print(f"  -> new best val OC {best_oc_dice:.4f}, saved best.ckpt")

    # Final test eval
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)
    ckpt = torch.load(output_dir / "best.ckpt", map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone"]); head.load_state_dict(ckpt["head"])
    test_metrics = evaluate_model(
        model_forward=lambda x: head(
            backbone(x.to(device, dtype=dtype)).patch_features.to(dtype),
            grid_hw=backbone(x.to(device, dtype=dtype)).grid_hw,
        ).mask_logits,
        loader=test_loader, device=device, dtype=dtype,
    )
    print(f"[multirater] TEST | {test_metrics.as_log_line('test')}")

    summary = {
        "config": vars(args),
        "best_val_oc_dice": best_oc_dice,
        "best_ckpt_epoch": ckpt.get("epoch", -1),
        "test_metrics": {k: getattr(test_metrics, k) for k in (
            "od_dice", "od_jaccard", "oc_dice", "oc_jaccard",
            "od_sensitivity", "od_specificity", "oc_sensitivity", "oc_specificity",
            "glaucoma_auc", "cdr_mae", "n_images",
        )},
        "history": history,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[multirater] wrote {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
