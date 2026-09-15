"""A percept class in the content run is presented, not merely emitted.

The content run asks whether two of her mechanisms put the percept classes in
one geometry. Every class in every run so far recalled the same memories. A
class arrived at half strength into a stream that still held her own percepts
from the turn before, the turn at 0.83 and the host's pressure near 0.78, and
the most salient percept is the one that cues recall. Every class lost to the
same one. These pin that a class outranks what an ordinary turn leaves behind,
so the class is what she recalls about, and that the classes still differ in
their kind and in nothing else.
"""

from __future__ import annotations

import random

import pytest

from core.phases.memory_retrieval import _percept_cue
from core.state.aura_state import AuraState
from core.state.percepts import emit_percept
from core.subject.content_runtime import grid, present

pytestmark = pytest.mark.unit


def _stream_after_an_ordinary_turn() -> AuraState:
    """What the fork held in the probe: her own percepts, fresh for memory."""
    state = AuraState.default()
    emit_percept(
        state.world, "the_turn",
        content="this is better than it was, and I still have the low",
        intensity=0.8314, source="affect",
    )
    emit_percept(state.world, "resource_pressure", content="under load: 78% pressure, 70% cpu", intensity=0.7756)
    emit_percept(state.world, "goal_achieved", content="the log holds 2 lines", intensity=0.6892, source="filesystem")
    emit_percept(state.world, "expectation_met", content="that came out the way I thought it would", intensity=0.617)
    return state


def test_every_class_cues_recall_over_what_a_turn_leaves_behind() -> None:
    for cls in grid():
        state = _stream_after_an_ordinary_turn()
        present(cls)(state, random.Random(0))
        assert _percept_cue(state) == cls.content, cls.name


def test_the_classes_differ_in_their_kind_and_in_nothing_else() -> None:
    classes = grid()
    assert len({cls.intensity for cls in classes}) == 1
    assert len({cls.source for cls in classes}) == 1
    assert len({cls.kind for cls in classes}) == len(classes)
