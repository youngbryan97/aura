"""Perception reached the world model as three numbers and no content.

The observation carried how many percepts arrived, how strong the newest was,
and how novel — and nothing about what any of them were. On the seed-7 run of 25
September perception reached the world model at 0.17 against a bar of 0.30, and
that was one of the two channels that left the world domain with an in-degree of
two while every other domain had eight or nine.

The same argument `_recalled` already makes about recollection: a model shown how
loud an arrival was and not what it was cannot let what she perceived contribute
to what it infers.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.state.aura_state import AuraState
from core.world_model.observe_cycle import CONTENT_WIDTH, _perceived, observation_of


def _with(*contents: str) -> AuraState:
    state = AuraState.default()
    state.world.recent_percepts = [
        {"type": "interaction", "content": text, "intensity": 0.6} for text in contents
    ]
    return state


def test_nothing_arrived_is_a_coordinate_of_zeros():
    assert _perceived([]) == [0.0] * CONTENT_WIDTH


def test_what_arrived_is_a_coordinate():
    coordinate = _perceived([{"type": "interaction", "content": "a message arrived"}])
    assert len(coordinate) == CONTENT_WIDTH
    assert any(value != 0.0 for value in coordinate)


def test_two_arrivals_sharing_their_words_sit_close():
    near = _perceived([{"content": "Bryan asked about the deploy"}])
    same = _perceived([{"content": "Bryan asked about the deploy"}])
    far = _perceived([{"content": "the kettle is boiling downstairs"}])
    assert np.allclose(near, same)
    assert np.linalg.norm(np.array(near) - np.array(far)) > 0.0


def test_the_observation_moves_when_the_content_does():
    quiet = observation_of(AuraState.default())
    one = observation_of(_with("Bryan sent a message about the deploy"))
    other = observation_of(_with("a totally different thing entirely"))
    assert one.shape == quiet.shape
    assert int((np.abs(one - other) > 1e-9).sum()) >= CONTENT_WIDTH


def test_only_the_newest_few_arrivals_count():
    """A turn's perception is what came in during it."""
    many = _perceived([{"content": f"arrival {index}"} for index in range(40)])
    last_four = _perceived([{"content": f"arrival {index}"} for index in range(36, 40)])
    assert np.allclose(many, last_four)


def test_a_percept_that_is_not_a_mapping_does_not_stop_the_coordinate():
    coordinate = _perceived(["a bare string", {"content": "and a real one"}])
    assert len(coordinate) == CONTENT_WIDTH


def test_the_strength_and_the_content_are_different_inputs():
    """Two arrivals of the same words at different strengths differ, and so do
    two of different words at the same strength."""
    loud = AuraState.default()
    loud.world.recent_percepts = [{"content": "the same words", "intensity": 0.9}]
    soft = AuraState.default()
    soft.world.recent_percepts = [{"content": "the same words", "intensity": 0.1}]
    assert not np.allclose(observation_of(loud), observation_of(soft))
    assert not np.allclose(observation_of(loud), observation_of(_with("other words")))
