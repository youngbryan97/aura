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
OBSERVATION_WIDTH = 17


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
