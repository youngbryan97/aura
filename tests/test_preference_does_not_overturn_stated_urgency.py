"""Preference chooses between comparable options, not against a stated urgency.

Pre-existing failure, found while checking the authority work. Two initiatives
were proposed — "Investigate runtime drift" at urgency 0.7 and "Stabilize
thermal load" at 0.9 — and the arbiter promoted the drift one.

Nothing was broken in the scoring. Urgency carries the highest weight of eight
dimensions, so a 0.2 difference moves the blended score by about 0.03:
measured, 0.648 against 0.617. The subjective-choice layer then tipped that
gap, which is exactly what it is for when options are comparable.

They were not comparable. A caller that declares an urgency is not offering a
hint, and if preference can reverse it the parameter means nothing whenever
she happens to fancy something else.

Where no urgency was declared the dimension is inferred from age, and
preference keeps its say — an absent urgency is not an urgency of zero.
"""

from __future__ import annotations

import asyncio

import pytest

from core.agency.initiative_arbiter import _explicit_urgency, get_initiative_arbiter


class _Cognition:
    def __init__(self, initiatives):
        self.pending_initiatives = list(initiatives)
        self.current_objective = None
        self.modifiers: dict = {}


class _State:
    def __init__(self, initiatives):
        self.cognition = _Cognition(initiatives)


def _arbitrate(initiatives):
    return asyncio.run(get_initiative_arbiter().arbitrate(_State(initiatives)))


def test_the_more_urgent_initiative_is_promoted() -> None:
    selected = _arbitrate([
        {"goal": "Investigate runtime drift", "urgency": 0.7, "timestamp": 0},
        {"goal": "Stabilize thermal load", "urgency": 0.9, "timestamp": 0},
    ])

    assert selected is not None
    assert selected.initiative["goal"] == "Stabilize thermal load"


def test_order_of_proposal_does_not_decide_it() -> None:
    """The same two, proposed the other way round."""
    selected = _arbitrate([
        {"goal": "Stabilize thermal load", "urgency": 0.9, "timestamp": 0},
        {"goal": "Investigate runtime drift", "urgency": 0.7, "timestamp": 0},
    ])

    assert selected is not None
    assert selected.initiative["goal"] == "Stabilize thermal load"


def test_an_absent_urgency_is_not_an_urgency_of_zero() -> None:
    assert _explicit_urgency({"goal": "x"}) is None
    assert _explicit_urgency({"goal": "x", "urgency": 0.4}) == 0.4
    assert _explicit_urgency({"goal": "x", "metadata": {"urgency": 0.6}}) == 0.6
    assert _explicit_urgency({"goal": "x", "urgency": "not a number"}) is None


def test_preference_still_decides_when_nothing_was_declared() -> None:
    """Two undeclared initiatives remain a matter of preference."""
    selected = _arbitrate([
        {"goal": "Read about mycelium", "timestamp": 0},
        {"goal": "Tidy the log directory", "timestamp": 0},
    ])

    assert selected is not None
    assert selected.initiative["goal"] in {
        "Read about mycelium",
        "Tidy the log directory",
    }


@pytest.mark.parametrize("urgency", [0.9, 1.0])
def test_equal_urgency_leaves_the_choice_open(urgency: float) -> None:
    selected = _arbitrate([
        {"goal": "First", "urgency": urgency, "timestamp": 0},
        {"goal": "Second", "urgency": urgency, "timestamp": 0},
    ])

    assert selected is not None
    assert selected.initiative["goal"] in {"First", "Second"}
