"""Tests for Int8Compressor + DistributedGradientSync."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from core.distributed.compress import Int8Compressor
from core.distributed.grad_sync import DistributedGradientSync


# ---------------------------------------------------------------------------
# Int8Compressor — round-trip fidelity
# ---------------------------------------------------------------------------
def test_compress_zero_tensor_returns_zero_with_unit_scale():
    t = torch.zeros(4)
    q, s = Int8Compressor.compress(t)
    assert q.dtype == torch.int8
    assert torch.equal(q, torch.zeros_like(q, dtype=torch.int8))
    # Scale clamps to 1e-8; recovered is still zeros.
    recovered = Int8Compressor.decompress(q, s)
    assert torch.allclose(recovered, torch.zeros_like(recovered), atol=1e-7)


def test_compress_empty_tensor_handled():
    t = torch.empty(0)
    q, s = Int8Compressor.compress(t)
    assert q.numel() == 0


def test_compress_round_trip_within_quantization_error():
    torch.manual_seed(0)
    t = torch.randn(64) * 5.0
    q, s = Int8Compressor.compress(t)
    recovered = Int8Compressor.decompress(q, s)
    err = (recovered.float() - t.float()).abs().max().item()
    # max abs / 127 / 2 is the per-element bound for round-half-up.
    expected_bound = (t.abs().max().item() / 127.0)
    assert err <= expected_bound + 1e-5


def test_compress_preserves_shape():
    t = torch.randn(3, 4, 5)
    q, s = Int8Compressor.compress(t)
    assert q.shape == t.shape


def test_compress_handles_extreme_magnitudes():
    t = torch.tensor([1e6, -1e6, 0.0, 0.5])
    q, s = Int8Compressor.compress(t)
    recovered = Int8Compressor.decompress(q, s)
    # Sign of large values must be preserved.
    assert recovered[0] > 0
    assert recovered[1] < 0


def test_round_trip_error_helper():
    t = torch.tensor([1.0, -1.0, 0.5])
    err = Int8Compressor.round_trip_error(t)
    assert err >= 0.0
    assert err < 0.02


# ---------------------------------------------------------------------------
# DistributedGradientSync — single-machine fallback
# ---------------------------------------------------------------------------
def test_disabled_sync_is_noop():
    sync = DistributedGradientSync(enabled=False)
    assert sync.world_size == 1
    model = nn.Linear(4, 4)
    # No grads exist yet — should still be safe.
    assert sync.sync_model_grads(model) == 0
    # After backward
    out = model(torch.randn(1, 4)).sum()
    out.backward()
    assert sync.sync_model_grads(model) == 0
    # No barrier raises
    sync.barrier()


def test_disabled_sync_world_size_is_one():
    sync = DistributedGradientSync(enabled=False)
    assert sync.world_size == 1


def test_disabled_sync_does_not_initialize_process_group():
    """Constructing with enabled=False must not touch torch.distributed."""
    import torch.distributed as dist

    pre_initialized = dist.is_initialized() if dist.is_available() else False
    DistributedGradientSync(enabled=False)
    post_initialized = dist.is_initialized() if dist.is_available() else False
    assert pre_initialized == post_initialized
