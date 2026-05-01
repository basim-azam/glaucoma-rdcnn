"""Console + TensorBoard + (optional) W&B logging in one place."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

_CONSOLE = Console()


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=_CONSOLE, rich_tracebacks=True, markup=False)],
        force=True,
    )


def get_logger(name: str = "glaucoma_rdcnn") -> logging.Logger:
    return logging.getLogger(name)


class RichConsoleLogger:
    """Wraps TensorBoard and W&B so the rest of the code only calls one API."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        wandb_mode: str = "offline",
        use_wandb: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tb = None
        self.wb = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(self.output_dir / "tb")
        except Exception as e:  # noqa: BLE001
            logging.warning(f"tensorboard unavailable: {e}")
        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    dir=str(self.output_dir),
                    mode=wandb_mode,
                )
                self.wb = wandb
            except Exception as e:  # noqa: BLE001
                logging.warning(f"wandb unavailable: {e}")

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            if self.tb is not None:
                self.tb.add_scalar(k, v, step)
            if self.wb is not None:
                self.wb.log({k: v, "step": step})

    def log_text(self, key: str, text: str, step: int) -> None:
        if self.tb is not None:
            self.tb.add_text(key, text, step)
        if self.wb is not None:
            self.wb.log({key: text, "step": step})

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        if self.wb is not None:
            self.wb.finish()


def log_event(logger: logging.Logger, event: str, **kw: Any) -> None:
    if kw:
        kv = " ".join(f"{k}={v}" for k, v in kw.items())
        logger.info(f"{event} | {kv}")
    else:
        logger.info(event)
