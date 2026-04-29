"""DistributedGradientSync — opt-in torch.distributed all-reduce wrapper.

Designed so existing trainers can call ``sync.sync_model_grads(model)``
unconditionally:

  * ``enabled=False``: no-op, returns immediately.
  * ``enabled=True``: requires that ``torch.distributed`` is available
    and either initialised by the caller (e.g. ``torchrun``) or
    initialised here with the supplied backend.

When the world has only one process the all-reduce is also a no-op
even if enabled — so the wrapper is single-machine-safe.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DistributedGradientSync:
    def __init__(
        self,
        *,
        enabled: bool = False,
        backend: Optional[str] = None,
    ):
        self.enabled = bool(enabled)
        self.backend = backend
        self.available = False
        if not self.enabled:
            return
        import torch.distributed as dist  # noqa: WPS433

        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        if not dist.is_initialized():
            backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
            dist.init_process_group(backend=backend)
        self.available = True

    @property
    def world_size(self) -> int:
        if not self.enabled:
            return 1
        import torch.distributed as dist  # noqa: WPS433

        return dist.get_world_size() if dist.is_initialized() else 1

    def sync_model_grads(self, model: nn.Module) -> int:
        """All-reduce all parameter gradients in-place; return synced count."""
        if not self.enabled:
            return 0
        import torch.distributed as dist  # noqa: WPS433

        world = self.world_size
        if world <= 1:
            return 0
        synced = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world)
            synced += 1
        return synced

    def barrier(self) -> None:
        if not self.enabled:
            return
        import torch.distributed as dist  # noqa: WPS433

        if dist.is_initialized():
            dist.barrier()
