"""Knowing a lane cannot finish, and dispatching to it anyway.

LIVE 2026-08-30: the gate calculated that the turn needed 1,200 seconds and had
480, clamped the deadline to 480, and dispatched. Three generations, 476
seconds each, every token discarded, and a refusal after a quarter of an hour.

The reserve alone decides it. If the reasoning a model does BEFORE it answers
does not fit the cap, no answer length fits and shortening the reply cannot
help.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from core.brain import inference_gate


def _clock_block() -> str:
    source = Path(inspect.getfile(inference_gate)).read_text(encoding="utf-8")
    at = source.index("[ANSWER CLOCK] %d tokens (%d asked")
    return source[at : at + 5200]


def test_the_reserve_alone_is_weighed_against_the_cap():
    block = _clock_block()
    assert "_reserve_alone" in block
    assert "_reserve_the_worker_adds" in block


def test_a_lane_that_cannot_fit_its_own_reasoning_is_not_dispatched_to():
    block = _clock_block()
    assert 'requested_tier = "tertiary"' in block, (
        "the gate must move off a lane it has just proved cannot finish"
    )


def test_only_the_primary_lane_is_demoted():
    """There is nowhere below the smallest model to go."""
    assert 'requested_tier == "primary" and _reserve_alone > _cap' in _clock_block()


def test_the_demotion_is_recorded_rather_than_silent():
    block = _clock_block()
    assert "answer_clock_demoted_from_primary" in block
    assert "primary_lane_cannot_fit_its_own_reasoning" in block


def test_the_numbers_that_decided_it_are_kept():
    block = _clock_block()
    for named in ("reserve_tokens", "reserve_seconds", "cap_seconds"):
        assert named in block
