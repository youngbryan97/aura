"""A generation inside a prefill step is working, and this clock could not see it.

Progress is emitted per token, so a generation that spends twenty seconds
inside one prefill step emits nothing and reads as stalled. The worker already
records the case: "a measured 755-token recurrent prefill occupied the
inference thread for roughly 52 seconds". It happens after the first token,
when a second pass re-reads the context — exactly when the token-stall clock is
watching.

LIVE 2026-08-29: asked to work out why turns were slow, the 27B produced
tokens, went quiet for 40 seconds, and was abandoned. "Token progress stalled
during generation (>40.0s)" beside "Cortex still sending heartbeats (2.2s
ago)". The person got the canned apology.
"""

from __future__ import annotations

import pytest

from core.brain.llm.mlx_client import MLXLocalClient

pytestmark = pytest.mark.unit


class _Lane:
    _prefill_progress_at = MLXLocalClient._prefill_progress_at

    def __init__(self, *, total: int, done: int, observed_at: float) -> None:
        self._current_prefill_tokens_total = total
        self._current_prefill_tokens_processed = done
        self._prefill_observed_at = observed_at


def test_a_prefill_in_flight_counts_as_progress() -> None:
    assert _Lane(total=800, done=300, observed_at=1234.5)._prefill_progress_at() == 1234.5


def test_a_finished_prefill_does_not_keep_a_stalled_generation_alive() -> None:
    """Work done minutes ago is not evidence that anything is happening now."""

    assert _Lane(total=800, done=800, observed_at=1234.5)._prefill_progress_at() == 0.0
    assert _Lane(total=800, done=900, observed_at=1234.5)._prefill_progress_at() == 0.0


def test_no_prefill_telemetry_claims_nothing() -> None:
    assert _Lane(total=0, done=0, observed_at=1234.5)._prefill_progress_at() == 0.0


def test_the_stall_clock_consults_it() -> None:
    from pathlib import Path

    source = Path("core/brain/llm/mlx_client.py").read_text(encoding="utf-8")
    assert "self._prefill_progress_at()," in source
    # And the first-token clock still keeps prefill separate on purpose: there
    # the question is whether reading has begun at all.
    assert "prefilling = (" in source


def test_a_genuinely_silent_generation_is_still_caught() -> None:
    """Nothing here waives the stall; it only stops calling reading silence."""

    from pathlib import Path

    source = Path("core/brain/llm/mlx_client.py").read_text(encoding="utf-8")
    assert "> token_stall_after" in source
    assert "Token progress stalled during generation" in source
