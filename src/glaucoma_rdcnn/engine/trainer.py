"""Minimal DDP-aware trainer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data import DataLoader

from glaucoma_rdcnn.data import build_dataloader
from glaucoma_rdcnn.engine.evaluator import Evaluator
from glaucoma_rdcnn.losses import weighted_total
from glaucoma_rdcnn.models import build_rdcnn
from glaucoma_rdcnn.utils.checkpoint import save_checkpoint
from glaucoma_rdcnn.utils.logging import RichConsoleLogger, get_logger, setup_logging
from glaucoma_rdcnn.utils.seed import seed_everything

LOG = get_logger("trainer")


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = -float("inf")
    history: list[dict[str, float]] = field(default_factory=list)


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def maybe_init_distributed() -> tuple[bool, int, int]:
    """Init torch.distributed if torchrun env vars are set. Returns (ddp, rank, world_size)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, world_size
    return False, 0, 1


class Trainer:
    def __init__(self, cfg: Any) -> None:
        setup_logging()
        seed_everything(int(cfg.seed))
        self.cfg = cfg
        self.ddp, self.rank, self.world_size = maybe_init_distributed()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_rdcnn(cfg.model).to(self.device)
        if self.ddp:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=True)

        params = [p for p in self.model.parameters() if p.requires_grad]
        opt_name = str(cfg.model.optimizer.name).lower()
        if opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                params,
                lr=float(cfg.model.optimizer.lr),
                momentum=float(cfg.model.optimizer.momentum),
                weight_decay=float(cfg.model.optimizer.weight_decay),
            )
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=float(cfg.model.optimizer.lr),
                weight_decay=float(cfg.model.optimizer.weight_decay),
            )
        else:
            raise ValueError(f"unknown optimizer {opt_name}")

        self.scheduler = self._build_scheduler(cfg)
        self.scaler = GradScaler(enabled=str(cfg.trainer.precision).startswith("16"))
        self.train_loader: DataLoader = build_dataloader(cfg.data, "train", distributed=self.ddp)
        try:
            self.val_loader: DataLoader = build_dataloader(cfg.data, "val", distributed=False)
        except Exception as e:  # noqa: BLE001
            LOG.warning(f"no val split: {e}")
            self.val_loader = None  # type: ignore[assignment]

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = (
            RichConsoleLogger(
                out_dir,
                wandb_project=str(cfg.logging.wandb_project),
                wandb_run_name=str(cfg.run_name),
                wandb_mode=str(cfg.logging.wandb_mode),
                use_wandb=bool(cfg.logging.wandb),
            )
            if is_main_process()
            else None
        )
        self.evaluator = Evaluator(cfg) if is_main_process() else None
        self.state = TrainState()

    def _build_scheduler(self, cfg: Any) -> torch.optim.lr_scheduler.LRScheduler | None:
        s = cfg.model.scheduler
        if str(s.name).lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=int(cfg.trainer.max_epochs)
            )
        if str(s.name).lower() == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        return None

    def _train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        running: dict[str, float] = {}
        n_steps = 0
        for batch in self.train_loader:
            images = batch["images"].to(self.device, non_blocking=True)
            targets = [
                {k: v.to(self.device, non_blocking=True) for k, v in t.items()}
                for t in batch["targets"]
            ]
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=str(self.cfg.trainer.precision).startswith("16")):
                _, losses = self.model(images, targets)
                total = weighted_total(losses, self.cfg.model.loss_weights)
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float(self.cfg.trainer.gradient_clip_val)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach().item())
            running["total"] = running.get("total", 0.0) + float(total.detach().item())
            n_steps += 1
            self.state.global_step += 1
            if self.cfg.trainer.fast_dev_run:
                break
        if self.scheduler is not None:
            self.scheduler.step()
        return {k: v / max(n_steps, 1) for k, v in running.items()}

    def fit(self) -> None:
        max_epochs = int(self.cfg.trainer.max_epochs)
        for epoch in range(max_epochs):
            self.state.epoch = epoch
            train_metrics = self._train_one_epoch()
            if is_main_process():
                LOG.info(
                    f"epoch {epoch:03d} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
                )
                if self.logger is not None:
                    self.logger.log_scalars(
                        {f"train/{k}": v for k, v in train_metrics.items()}, epoch
                    )
            # Val
            if is_main_process() and self.val_loader is not None and self.evaluator is not None:
                core = self.model.module if self.ddp else self.model
                val_metrics = self.evaluator.evaluate(core, self.val_loader, device=self.device)
                LOG.info("val | " + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                if self.logger is not None:
                    self.logger.log_scalars({f"val/{k}": v for k, v in val_metrics.items()}, epoch)
                metric = float(val_metrics.get(str(self.cfg.ckpt.monitor).split("/")[-1], 0.0))
                if metric > self.state.best_metric:
                    self.state.best_metric = metric
                    save_checkpoint(
                        Path(self.cfg.output_dir) / "best.ckpt",
                        model=core,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=val_metrics,
                    )
            if self.cfg.trainer.fast_dev_run:
                break
        if self.logger is not None:
            self.logger.close()


def cli_main() -> None:  # used as `rdcnn-train` entry point
    import hydra

    @hydra.main(version_base=None, config_path="../../../configs", config_name="default")
    def _run(cfg: Any) -> None:
        Trainer(cfg).fit()

    _run()  # noqa: B018  (hydra decorator runs it)


if __name__ == "__main__":  # pragma: no cover
    cli_main()
