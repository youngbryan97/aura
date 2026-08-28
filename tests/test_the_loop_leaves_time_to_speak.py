"""A turn that spends everything on tools has nothing left to say.

LIVE, 2026-08-28: three files read correctly, and the answer generation was
then given a 36.2s first-token budget for a 6,298-char prompt that takes about
120 seconds to read at this worker's measured rate. It was cancelled mid-prefill
and the person got the one canned reply that must never be reachable. Everything
the loop found was right and none of it could be said.

The reserve was a constant. The thing it reserves for grows with what the tools
return: the more the loop finds, the longer the prompt the answer is read from,
and the number never moved.

The worker already computes this quantity to raise its own first-token ceiling.
The reserve is the same measurement, asked one layer up and before the time is
spent rather than after.
"""

from __future__ import annotations

from core.brain.inference_gate import (
    _ANSWER_RESERVE_FALLBACK_S,
    _TOOL_LOOP_FLOOR_S,
    _answer_reserve_seconds,
    _tool_loop_budget,
)


class _Worker:
    """A worker with a known prefill rate, so the arithmetic is checkable."""

    def _prefill_floor_seconds(self, chars: int) -> float:
        return chars / 4.0 / 12.0 * 1.3


def test_the_reserve_grows_with_what_the_answer_must_read() -> None:
    small = _answer_reserve_seconds(_Worker(), 400)
    large = _answer_reserve_seconds(_Worker(), 6300)
    assert large > small * 2, (small, large)


def test_the_loop_gets_less_when_the_answer_will_need_more() -> None:
    whole = 516.0
    short = _tool_loop_budget(whole, _answer_reserve_seconds(_Worker(), 400))
    long = _tool_loop_budget(whole, _answer_reserve_seconds(_Worker(), 6300))
    assert short > long
    assert long > _TOOL_LOOP_FLOOR_S


def test_a_worker_that_cannot_say_falls_back_rather_than_failing() -> None:
    """An unmeasured rate must not produce a zero reserve or an exception."""

    assert _answer_reserve_seconds(None, 6300) == _ANSWER_RESERVE_FALLBACK_S
    assert _answer_reserve_seconds(_Worker(), 0) == _ANSWER_RESERVE_FALLBACK_S

    class _Broken:
        def _prefill_floor_seconds(self, chars: int) -> float:
            raise ValueError("no rate measured yet")

    assert _answer_reserve_seconds(_Broken(), 6300) == _ANSWER_RESERVE_FALLBACK_S


def test_the_loop_always_gets_enough_for_one_call() -> None:
    """Squeezing below a single call trades one failure for another."""

    assert _tool_loop_budget(30.0, 10_000.0) == _TOOL_LOOP_FLOOR_S
    assert _tool_loop_budget(None, None) == _TOOL_LOOP_FLOOR_S
    assert _tool_loop_budget(516.0, -5) < 516.0


def test_the_reserve_is_taken_from_the_evidence_the_loop_carries() -> None:
    from pathlib import Path

    body = Path("core/brain/inference_gate.py").read_text()
    start = body.index("timeout=_tool_loop_budget(")
    window = body[start : start + 620]
    assert "_answer_reserve_seconds(" in window
    # The objective AND what the turn already read, because both come back
    # with the answer.
    assert "len(str(text or \"\"))" in window
    assert "evidence" in window
