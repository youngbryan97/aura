"""LatticeTrainer — single-process training harness for LatticeLM.

Stable training step with:

  * AdamW + weight decay
  * gradient clipping
  * non-finite loss / non-finite grad-norm guards (raises rather than
    silently corrupting weights)
  * AMP autocast on CUDA
  * atomic checkpoints with full optimizer + step state

The trainer is intentionally minimal — no scheduler, no logging
backend.  Outer code (research_core, evaluators) drives the loop.
"""
from __future__ import annotations

import dataclasses
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader

from core.lattice.model import LatticeLM


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    amp: bool = True
    device: str = "auto"
    checkpoint_dir: str = "./checkpoints_lattice"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


def _atomic_torch_save(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class LatticeTrainer:
    def __init__(self, model: LatticeLM, cfg: TrainConfig):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.resolve_device())
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(cfg.amp and self.device.type == "cuda")
        )
        self.global_step = 0

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        batch = self._move_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)

        amp_enabled = self.scaler.is_enabled()
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            out = self.model(batch["input_ids"], labels=batch["labels"])
            loss = out["loss"]

        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {float(loss)}")

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite grad norm: {float(grad_norm)}")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.global_step += 1

        return {
            "loss": float(loss.detach()),
            "lm_loss": float(out["lm_loss"].detach()),
            "moe_aux": float(out["moe_aux"].detach()),
            "world_loss": float(out["world_loss"].detach()),
            "route_entropy": float(out["route_entropy"].detach()),
            "grad_norm": float(grad_norm.detach()),
            "step": float(self.global_step),
        }

    @torch.no_grad()
    def eval_loss(self, loader: DataLoader, max_batches: int = 20) -> float:
        self.model.eval()
        losses = []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = self._move_batch(batch)
            out = self.model(batch["input_ids"], labels=batch["labels"])
            losses.append(float(out["loss"].detach()))
        return statistics.mean(losses) if losses else float("inf")

    def save_checkpoint(
        self, name: str = "latest.pt", extra: Optional[Dict[str, Any]] = None
    ) -> Path:
        path = Path(self.cfg.checkpoint_dir) / name
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "model_config": dataclasses.asdict(self.model.cfg),
            "train_config": dataclasses.asdict(self.cfg),
            "extra": extra or {},
        }
        _atomic_torch_save(payload, path)
        return path

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        payload = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload.get("global_step", 0))
        return payload
