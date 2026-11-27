"""What MLX holds in a model worker, in use and kept for reuse, rides along with each result and is kept by its client
(core/brain/llm/mlx_worker.py ``_what_mlx_holds``, core/brain/llm/mlx_client_recording.py).

LIVE 26 November 2026 the cortex's worker stood at 25 GB, 19 GB of it compressed, beside a second model of 11 GB, and
nothing said how much of either was freed buffers kept for reuse, which can be given back, and how much was in use.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.llm.mlx_worker import _what_mlx_holds

pytestmark = pytest.mark.unit


def test_a_worker_says_what_its_mlx_holds_in_use_and_kept_for_reuse() -> None:
    mx = SimpleNamespace(get_active_memory=lambda: 18 * 2 ** 30, get_cache_memory=lambda: 3 * 2 ** 30)
    assert _what_mlx_holds(mx) == {"active_bytes": 18 * 2 ** 30, "cache_bytes": 3 * 2 ** 30}


def test_a_worker_without_mlx_or_its_readings_says_nothing_of_them() -> None:
    assert _what_mlx_holds(None) == {}
    assert _what_mlx_holds(SimpleNamespace()) == {}


def test_its_client_keeps_the_last_reading() -> None:
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient("Aura-32B-crsm-closeout-jul1-20260701-215118")
    client._record_surface_control_receipt_from_response(
        {"status": "ok", "prompt_cache_bytes": 1000, "tokens_used": 1,
         "mlx_memory": {"active_bytes": 5, "cache_bytes": 7}}
    )
    assert client._last_mlx_memory == {"active_bytes": 5, "cache_bytes": 7}
