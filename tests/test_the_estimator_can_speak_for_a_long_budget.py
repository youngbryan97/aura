"""The deadline estimator was silent for exactly the turns it protects.

``seconds_to_decode`` counts only runs of at least half the wanted length,
because below that a measurement is dominated by the prompt. Sound in
principle, and measured live on 2026-08-28 it meant: forty readings held, and
ZERO of them comparable to a 1536-token budget, because this runtime's
generations are mostly nineteen to ninety-nine tokens.

So it returned 0.0 — "unmeasured, extend nothing" — for every budget worth
extending, and the long turns stayed rare partly because the deadline it could
not extend kept cancelling them.

The fallback is safe in the direction a deadline needs. A short run amortises
its prefill over fewer tokens, so its per-token rate is the worse one: using it
for a long budget over-estimates the time and asks for more deadline, never
less.
"""

from __future__ import annotations

import pytest

from core.brain.llm.thinking_reserve import (
    _LONG_ENOUGH_TO_TIME,
    forget,
    record_decode_rate,
    seconds_to_decode,
)


@pytest.fixture(autouse=True)
def _clean():
    forget()
    yield
    forget()


def test_a_long_budget_can_be_timed_from_shorter_runs() -> None:
    for _ in range(14):
        record_decode_rate(generated_tokens=99, elapsed_s=20.0)
    got = seconds_to_decode(1536)
    assert got > 0.0
    # ~5 tokens/s over 1536 tokens.
    assert 250.0 < got < 360.0, got


def test_comparable_runs_are_still_preferred() -> None:
    """When there are enough of the right size, they decide it."""

    for _ in range(14):
        record_decode_rate(generated_tokens=99, elapsed_s=40.0)   # slow, short
    for _ in range(14):
        record_decode_rate(generated_tokens=1000, elapsed_s=100.0)  # fast, long
    got = seconds_to_decode(1536)
    # The long runs are comparable and they are the faster ones, so the
    # estimate follows them rather than the short ones.
    assert got < 300.0, got


def test_runs_dominated_by_their_prompt_are_still_excluded() -> None:
    """A one-token generation says nothing about decoding."""

    for _ in range(30):
        record_decode_rate(generated_tokens=1, elapsed_s=5.0)
    assert seconds_to_decode(1536) == 0.0
    assert _LONG_ENOUGH_TO_TIME > 1


def test_it_is_still_silent_with_nothing_measured() -> None:
    assert seconds_to_decode(1536) == 0.0
