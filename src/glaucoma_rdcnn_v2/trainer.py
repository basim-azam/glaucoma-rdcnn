"""v2 trainer: frozen-then-unfrozen ViT encoder + Mask2Former-style head.

Architecture decisions:
- Encoder frozen for the first `freeze_epochs` epochs (decoder warms up).
- After unfreeze: encoder at 10x lower LR than decoder.
- Cosine schedule across the full run with 5-epoch linear warmup.
- bf16 autocast on CUDA.
- Per-epoch val metrics; best.ckpt saved on best val OC Dice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import FundusSegDataset
from .evaluator import evaluate_model
from .losses import CompoundSegLoss, precompute_sdt
from .models import build_backbone, build_seg_head


@dataclass
class TrainerConfig:
    data_root: str
    output_dir: str
    epochs: int = 40
    batch_size: int = 16
    freeze_epochs: int = 10
    encoder_size: int = 224
    target_size: int = 512
    base_lr: float = 5e-4
    encoder_lr_ratio: float = 0.1  # encoder LR = base_lr * ratio after unfreeze
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    num_workers: int = 4
    seed: int = 42
    prefer_fallback_backbone: bool = False
    use_heteroscedastic: bool = False
    decoder_dim: int = 256
    num_decoder_layers: int = 3
    dtype_str: str = "bfloat16"  # "bfloat16" or "float32"


def _build_loaders(cfg: TrainerConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = FundusSegDataset(
        root=cfg.data_root,
        split="train",
        encoder_size=cfg.encoder_size,
        target_size=cfg.target_size,
        augment=True,
        seed=cfg.seed,
    )
    val_ds = FundusSegDataset(
        root=cfg.data_root,
        split="val",
        encoder_size=cfg.encoder_size,
        target_size=cfg.target_size,
        augment=False,
        seed=cfg.seed,
    )
    test_ds = FundusSegDataset(
        root=cfg.data_root,
        split="test",
        encoder_size=cfg.encoder_size,
        target_size=cfg.target_size,
        augment=False,
        seed=cfg.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def _lr_scale(epoch: int, total: int, warmup: int) -> float:
    """Linear warmup -> cosine decay."""
    import math

    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class V2Trainer:
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Build dataloaders
        self.train_loader, self.val_loader, self.test_loader = _build_loaders(cfg)
        print(f"[trainer] data: train={len(self.train_loader.dataset)} val={len(self.val_loader.dataset)} test={len(self.test_loader.dataset)}")

        # Build model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if cfg.dtype_str == "bfloat16" else torch.float32

        self.backbone = build_backbone(
            pretrained=True,
            image_size=cfg.encoder_size,
            prefer_fallback=cfg.prefer_fallback_backbone,
        )
        self.head = build_seg_head(
            encoder_dim=self.backbone.feature_dim,
            decoder_dim=cfg.decoder_dim,
            num_decoder_layers=cfg.num_decoder_layers,
            target_size=cfg.target_size,
        )

        self.backbone.freeze()
        self.backbone.to(self.device)
        self.head.to(self.device)
        print(f"[trainer] backbone source: {self.backbone.source}")

        self.loss_fn = CompoundSegLoss(w_heteroscedastic=0.05 if cfg.use_heteroscedastic else 0.0)
        self._build_optimizer(encoder_unfrozen=False)

        self.best_oc_dice = -1.0

    def _build_optimizer(self, encoder_unfrozen: bool) -> None:
        if encoder_unfrozen:
            param_groups = [
                {"params": [p for p in self.backbone.parameters() if p.requires_grad], "lr": self.cfg.base_lr * self.cfg.encoder_lr_ratio, "name": "encoder"},
                {"params": list(self.head.parameters()), "lr": self.cfg.base_lr, "name": "head"},
            ]
        else:
            param_groups = [
                {"params": list(self.head.parameters()), "lr": self.cfg.base_lr, "name": "head"},
            ]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.cfg.weight_decay)
        self.base_lrs = [g["lr"] for g in self.optimizer.param_groups]

    def _apply_lr_schedule(self, epoch: int) -> None:
        scale = _lr_scale(epoch, self.cfg.epochs, self.cfg.warmup_epochs)
        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=False):
            g["lr"] = base_lr * scale

    def _forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mask_logits, log_var). Used by both training and eval."""
        feats = self.backbone(images)
        out = self.head(feats.patch_features.to(images.dtype), grid_hw=feats.grid_hw)
        return out.mask_logits, out.log_var

    def _step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        images = batch["image"].to(self.device, dtype=self.dtype, non_blocking=True)
        targets = batch["target"].to(self.device, dtype=torch.float32, non_blocking=True)

        self.optimizer.zero_grad()
        with torch.autocast("cuda", dtype=self.dtype, enabled=(self.device.type == "cuda" and self.dtype != torch.float32)):
            mask_logits, log_var = self._forward(images)

        # SDT in float32 on CPU then move (deterministic per batch)
        sdt = precompute_sdt(targets.cpu()).to(self.device)

        loss, breakdown = self.loss_fn(
            mask_logits.float(),
            targets,
            sdt_target=sdt,
            log_var=log_var.float() if self.cfg.use_heteroscedastic else None,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.head.parameters() if p.grad is not None]
            + [p for p in self.backbone.parameters() if p.grad is not None],
            max_norm=1.0,
        )
        self.optimizer.step()
        return breakdown

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> "SegMetrics":
        self.backbone.eval()
        self.head.eval()
        metrics = evaluate_model(
            model_forward=lambda x: self._forward(x.to(self.device, dtype=self.dtype))[0],
            loader=loader,
            device=self.device,
            dtype=self.dtype,
        )
        self.backbone.train()
        self.head.train()
        return metrics

    def _save_ckpt(self, name: str, extra: dict | None = None) -> None:
        ckpt = {
            "backbone": self.backbone.state_dict(),
            "head": self.head.state_dict(),
            "cfg": self.cfg.__dict__,
        }
        if extra is not None:
            ckpt.update(extra)
        torch.save(ckpt, Path(self.cfg.output_dir) / name)

    def train(self) -> dict:
        """Run the full training loop. Returns final test metrics + history."""
        history: list[dict] = []
        for epoch in range(self.cfg.epochs):
            # Encoder unfreeze trigger
            if epoch == self.cfg.freeze_epochs and self.backbone.is_frozen:
                self.backbone.unfreeze()
                self._build_optimizer(encoder_unfrozen=True)
                print(f"[trainer] epoch {epoch}: unfroze encoder, rebuilt optimizer")

            self._apply_lr_schedule(epoch)

            t0 = time.time()
            self.backbone.train()
            self.head.train()
            epoch_losses: list[float] = []
            for batch in self.train_loader:
                breakdown = self._step(batch)
                epoch_losses.append(breakdown["total"])

            wall = time.time() - t0
            train_loss = sum(epoch_losses) / max(1, len(epoch_losses))

            # Validation
            val_metrics = self._evaluate(self.val_loader)
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_od_dice": val_metrics.od_dice,
                "val_oc_dice": val_metrics.oc_dice,
                "val_auc": val_metrics.glaucoma_auc,
                "lr_head": self.optimizer.param_groups[-1]["lr"],
                "wall": wall,
            })
            print(
                f"[ep {epoch:03d}] train_loss={train_loss:.4f} | "
                f"val_od_dice={val_metrics.od_dice:.4f} oc_dice={val_metrics.oc_dice:.4f} | "
                f"auc={val_metrics.glaucoma_auc:.4f} | wall={wall:.1f}s"
            )

            if val_metrics.oc_dice > self.best_oc_dice:
                self.best_oc_dice = val_metrics.oc_dice
                self._save_ckpt(
                    "best.ckpt",
                    extra={"epoch": epoch, "val_oc_dice": val_metrics.oc_dice},
                )
                print(f"  -> new best OC Dice {self.best_oc_dice:.4f}, saved best.ckpt")

        # Final test eval using best checkpoint
        ckpt = torch.load(Path(self.cfg.output_dir) / "best.ckpt", map_location=self.device, weights_only=False)
        self.backbone.load_state_dict(ckpt["backbone"])
        self.head.load_state_dict(ckpt["head"])
        test_metrics = self._evaluate(self.test_loader)
        print(f"[trainer] {test_metrics.as_log_line('test')}")

        return {
            "history": history,
            "test_metrics": test_metrics,
            "best_oc_dice_val": self.best_oc_dice,
            "best_ckpt_epoch": ckpt.get("epoch", -1),
        }
