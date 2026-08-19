"""What is weakest is what the runtime recorded failing.

LIVE 2026-08-18: "rank your three weakest subsystems and say why."

    1. Long-term memory persistence... 2. Emotional nuance... 3. Self-awareness
    in real-time... These areas are where I feel the gap between what I am and
    what I could be.

Plausible, humble, and entirely invented — the shape of what an AI is expected
to say about itself. The runtime records a degradation every time a subsystem
fails, with the subsystem, the severity, and an action line written for a
person, and none of it reached the turn.

Nothing here enumerates subsystems. A subsystem appears because it recorded a
degradation, which is what makes this survive the next subsystem being added.
"""

from __future__ import annotations

import pytest

from core.self.operational_state import (
    asks_about_own_condition,
    operational_state_block,
)


@pytest.fixture
def recorded(monkeypatch):
    import core.self.operational_state as module

    records = [
        {"subsystem": "memory_facade", "severity": "degraded",
         "action": "gave up after the third retry"},
        {"subsystem": "memory_facade", "severity": "degraded", "action": "same again"},
        {"subsystem": "web_search", "severity": "warning",
         "action": "answered without live sources"},
    ]
    monkeypatch.setattr(module, "_records", lambda limit=40: list(records))
    return records


@pytest.mark.parametrize(
    "question",
    [
        "rank your three weakest subsystems and say why.",
        "what's been failing lately?",
        "how are you really?",
        "any problems on your end?",
        "which of your components are degraded?",
    ],
)
def test_a_question_about_her_condition_is_recognised(question: str) -> None:
    assert asks_about_own_condition(question)


@pytest.mark.parametrize(
    "question",
    ["what's wrong with my computer?", "how are you?", "what is 2 + 2"],
)
def test_a_different_question_is_not_claimed(question: str) -> None:
    assert not asks_about_own_condition(question)


def test_the_answer_names_the_subsystems_that_actually_failed(recorded) -> None:
    block = operational_state_block("rank your three weakest subsystems and say why.")

    assert "memory_facade" in block
    assert "web_search" in block
    assert "gave up after the third retry" in block


def test_the_most_affected_subsystem_comes_first(recorded) -> None:
    block = operational_state_block("what's been failing lately?")

    assert block.index("memory_facade") < block.index("web_search")


def test_a_subsystem_nobody_listed_still_appears(monkeypatch) -> None:
    """The generality: no table names these."""
    import core.self.operational_state as module

    monkeypatch.setattr(
        module,
        "_records",
        lambda limit=40: [
            {"subsystem": "thermal_lattice_tuner", "severity": "critical",
             "action": "resonance drift exceeded the bound"}
        ],
    )

    block = operational_state_block("what's been failing lately?")

    assert "thermal_lattice_tuner" in block
    assert "resonance drift exceeded the bound" in block


def test_an_empty_record_is_said_plainly(monkeypatch) -> None:
    import core.self.operational_state as module

    monkeypatch.setattr(module, "_records", lambda limit=40: [])

    block = operational_state_block("what's been failing lately?")

    assert "No degradations have been recorded" in block
    assert "not a claim that nothing could be wrong" in block
