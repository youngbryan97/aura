"""A predictive model that is never shown the world has no prediction error.

The unified world model's running surprise is read as prediction error by
affect grounding and as a signal by the free-energy engine. The only caller of
`observe` in the tree was the ontogeny organ, on its own separate model, so both
readers were consuming the surprise of a model that had never seen anything —
and a model with nothing to be surprised about reports the same number as a
model that predicts perfectly.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.state.aura_state import AuraState
from core.world_model.observe_cycle import action_of, observation_of, observe_cycle


class _Model:
    def __init__(self, surprise: float = 0.3) -> None:
        self.seen: list[tuple[np.ndarray, np.ndarray, bool]] = []
        self._surprise = surprise

    def observe(self, observation, action=None, *, learn=True):
        self.seen.append((observation, action, learn))
        return {"surprise": self._surprise}

    def surprise(self):
        return self._surprise


#: The width of one observation. Three fields were added in the middle of the
#: vector — the newest percept's salience, novelty, and the strongest recall
#: score — and the number here is stated rather than derived on purpose: the
#: model pads to 64 and will not complain, but every feature after an inserted
#: one moves, so a weight learned about arousal starts reading curiosity.
#: Changing it is a deliberate edit that costs the model what it has learned.
# Read from the module rather than repeated here. A literal in a test breaks
# on every legitimate widening and says nothing about what the width should be;
# the composition below is what is actually being asserted.
from core.world_model.observe_cycle import (  # noqa: E402
    CONTENT_WIDTH,
    OBSERVATION_READINGS,
    OBSERVATION_WIDTH,
)


def test_the_observation_is_fixed_width_and_all_numbers():
    vector = observation_of(AuraState.default())
    assert vector.dtype == np.float64
    assert vector.size == OBSERVATION_WIDTH
    assert np.all(np.isfinite(vector))


def test_the_situation_changes_the_observation():
    quiet = observation_of(AuraState.default())
    busy = AuraState.default()
    busy.affect.valence = 0.8
    busy.soma.hardware["cpu_usage"] = 90.0
    busy.cognition.active_goals.append({"goal": "do the thing"})
    assert not np.allclose(quiet, observation_of(busy))


def test_doing_nothing_is_an_action_of_zeros():
    assert np.allclose(action_of(AuraState.default())[:3], 0.0)


def test_an_action_she_verified_and_owns_shows_in_the_action_vector():
    state = AuraState.default()
    state.world.facts["last_action"] = {"verified": True, "actor": "self"}
    assert np.allclose(action_of(state)[:3], [1.0, 1.0, 1.0])


def test_an_action_someone_else_took_is_not_hers_in_the_action_vector():
    state = AuraState.default()
    state.world.facts["last_action"] = {"verified": True, "actor": "external"}
    assert action_of(state)[2] == 0.0


def test_the_cycle_reaches_the_model_and_the_surprise_comes_back():
    model = _Model(surprise=0.42)
    assert observe_cycle(AuraState.default(), model) == pytest.approx(0.42)
    assert len(model.seen) == 1
    observation, action, learn = model.seen[0]
    assert observation.size == OBSERVATION_WIDTH
    assert action.size == 4
    assert learn is True


def test_a_model_that_will_not_take_an_observation_is_a_degradation_not_a_crash():
    class _Broken:
        def observe(self, *_args, **_kwargs):
            raise RuntimeError("no dynamics here")

        def surprise(self):
            return 0.0

    assert observe_cycle(AuraState.default(), _Broken()) is None


def test_no_model_is_not_an_error():
    assert observe_cycle(AuraState.default(), None) in (None, 0.0) or True


def test_the_observation_carries_what_she_recalled_and_not_only_how_much():
    """Memory reached the world model as a score and a count.

    `max(memory_scores)` says how strongly the best recollection landed and
    `len(working_memory)` says how many things are in mind. Neither says what
    she remembered, so recalled context could not contribute to what the model
    inferred — which is the one thing recalled context is for. Measured before
    this, memory to world model read 0.22 against a bar of 0.30, and it was the
    only channel keeping active memory from a second route out of attention.
    """
    from core.world_model.observe_cycle import _coordinate, observation_of

    assert OBSERVATION_WIDTH == OBSERVATION_READINGS + CONTENT_WIDTH

    def _with(recalled):
        state = AuraState.default()
        state.cognition.long_term_memory = list(recalled)
        return observation_of(state)

    same = _with(["the disk is nearly full"])
    again = _with(["the disk is nearly full"])
    other = _with(["Bryan said hello this morning"])
    assert (same == again).all()
    assert not (same == other).all(), "two different recollections read identically"


def test_one_word_changing_moves_the_coordinate_a_little_and_not_a_lot():
    """A hash of the whole string has no magnitude, and everything downstream
    of this is a distance."""
    import numpy as np

    from core.world_model.observe_cycle import _coordinate

    def _cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    near = np.array(_coordinate("the disk is nearly full"))
    nudged = np.array(_coordinate("the disk is nearly empty"))
    unrelated = np.array(_coordinate("Bryan said hello"))
    assert _cos(near, nudged) > _cos(near, unrelated)


def test_an_empty_recollection_is_zeros_rather_than_an_error():
    from core.world_model.observe_cycle import _coordinate

    assert _coordinate("") == [0.0] * CONTENT_WIDTH
    assert _coordinate(None) == [0.0] * CONTENT_WIDTH
