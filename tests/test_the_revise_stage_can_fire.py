"""The belief-revision stage had one caller and it passed an empty list.

`IntentionLoop.revise` pushes into BeliefRevisionEngine, writes a
SELF_MODEL_UPDATE transition to the CognitiveLedger, and reports what changed.
All of it is guarded by `if rec.belief_updates`, and its only production caller
— `core/constitution.py`, after every governed tool execution — passed
`belief_updates=[]` hardcoded.

So the stage was correct, complete, and could not fire. Every live turn logged
"Intention completed [...]: 0 belief updates, 0 self-model updates", including
the one on 2026-09-07 where a web search returned the right answer to the
question that was asked.

What a cycle establishes is read off the record it already holds, and the
confidence is the measured success rate of that capability rather than a
number chosen here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.agency.intention_loop import IntentionLoop


@pytest.fixture
def loop(tmp_path, monkeypatch) -> IntentionLoop:
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path))
    return IntentionLoop()


def _cycle(loop: IntentionLoop, tool: str, *, success: bool, intention: str) -> str:
    identity = loop.intend(intention=intention, drive="curiosity")
    loop.record_action(
        identity, tool_name=tool, args={}, result="r", success=success, duration_ms=1.0
    )
    loop.observe(
        identity,
        observation="tool_succeeded" if success else "tool_failed",
        actual_outcome="done" if success else "error",
    )
    return identity


def test_a_cycle_that_used_a_tool_revises_a_belief(loop: IntentionLoop) -> None:
    identity = _cycle(loop, "web_search", success=True, intention="look something up")
    loop.revise(identity, success=True)
    record = next(r for r in loop._completed_intentions if r.id == identity)
    assert record.belief_updates, "the revise stage still cannot fire"
    assert "web_search" in record.belief_updates[0].belief


def test_an_explicit_empty_list_still_means_none(loop: IntentionLoop) -> None:
    """A caller that decided there is nothing to revise keeps saying so."""
    identity = _cycle(loop, "web_search", success=True, intention="look something up")
    loop.revise(identity, belief_updates=[], success=True)
    record = next(r for r in loop._completed_intentions if r.id == identity)
    assert record.belief_updates == []


def test_confidence_moves_with_what_actually_happened(loop: IntentionLoop) -> None:
    for index in range(4):
        identity = _cycle(loop, "flaky", success=True, intention=f"try {index}")
        loop.revise(identity, success=True)
    identity = _cycle(loop, "flaky", success=True, intention="try again")
    loop.revise(identity, success=True)
    record = next(r for r in loop._completed_intentions if r.id == identity)
    update = record.belief_updates[0]
    assert update.new_confidence > update.old_confidence

    failing = _cycle(loop, "flaky", success=False, intention="and it breaks")
    loop.revise(failing, success=False)
    record = next(r for r in loop._completed_intentions if r.id == failing)
    update = record.belief_updates[0]
    assert update.new_confidence < update.old_confidence


def test_an_unused_capability_starts_at_even_odds(loop: IntentionLoop) -> None:
    """Laplace, so one success is not certainty and nobody picked a prior."""
    prior, attempts = loop._capability_confidence("never_used")
    assert attempts == 0
    assert prior == pytest.approx(0.5)


def test_a_capability_used_twice_in_one_cycle_is_one_observation(
    loop: IntentionLoop,
) -> None:
    identity = loop.intend(intention="two calls", drive="curiosity")
    for success in (True, False):
        loop.record_action(
            identity, tool_name="dual", args={}, result="r",
            success=success, duration_ms=1.0,
        )
    loop.observe(identity, observation="tool_failed", actual_outcome="error")
    loop.revise(identity, success=False)
    record = next(r for r in loop._completed_intentions if r.id == identity)
    assert len(record.belief_updates) == 1
    assert record.belief_updates[0].new_confidence < record.belief_updates[0].old_confidence


def test_the_only_caller_no_longer_decides_there_is_nothing() -> None:
    source = Path("core/constitution.py").read_text()
    start = source.index("intention_loop.revise(")
    assert "belief_updates=[]" not in source[start : start + 400], (
        "the caller is back to hardcoding an empty list, which is the defect"
    )
