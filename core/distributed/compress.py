"""Int8Compressor — symmetric per-tensor quantisation.

Compress: scale = max(abs(t)) / 127, q = round(t / scale).
Decompress: q.float() * scale.

The error is bounded by ``scale / 2`` per element.  Use this when a
distributed training run is bandwidth-bound on consumer links;
high-bandwidth clusters should keep fp16/bf16 instead.

This module ships only the math.  ``DistributedGradientSync`` calls
it on the gradient buffers before all-reduce when configured.
"""
from __future__ import annotations

from typing import Tuple

import torch


class Int8Compressor:
    """Symmetric per-tensor int8 quantizer with float scale carrier."""

    @staticmethod
    def compress(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if t.numel() == 0:
            return t.to(torch.int8), torch.tensor(1.0, dtype=t.dtype, device=t.device)
        max_abs = t.detach().abs().max().clamp(min=1e-8)
        scale = max_abs / 127.0
        q = torch.clamp((t / scale).round(), -127, 127).to(torch.int8)
        return q, scale.detach()

    @staticmethod
    def decompress(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return q.float() * scale.float()

    @staticmethod
    def round_trip_error(t: torch.Tensor) -> float:
        q, s = Int8Compressor.compress(t)
        recovered = Int8Compressor.decompress(q, s)
        return float((recovered.float() - t.float()).abs().max())
