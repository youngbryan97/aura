"""Distributed-training abstractions for the Lattice substrate.

The honest answer to "escape local hardware limits" is to amortize:
spread the same training step across multiple machines so total
compute scales while each machine runs within its own RAM/VRAM.

This module ships the contract pieces:

  * ``Int8Compressor`` — symmetric per-tensor quantization for
    bandwidth-constrained gradient sync.
  * ``DistributedGradientSync`` — opt-in ``torch.distributed``
    all-reduce wrapper with a graceful single-machine fallback so
    every other module can call it unconditionally.

Real multi-node deployments wire ``DistributedGradientSync(enabled=
True)`` and run under ``torchrun``; single-machine tests construct
the unenabled path and exercise compression round-trips.
"""
from core.distributed.compress import Int8Compressor
from core.distributed.grad_sync import DistributedGradientSync

__all__ = ["DistributedGradientSync", "Int8Compressor"]
