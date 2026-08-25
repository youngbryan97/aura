"""z_Aura has to say what it does not know.

The failure this guards against is the one the codebase keeps finding: a
channel whose source is dead reporting a clean zero, which downstream reads as
a measurement. Every dimension here carries a presence bit, and a state that
nothing answered for has to be distinguishable from a state of all zeros that
something did answer for.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.brain.llm.endogenous_state import (
    CHANNELS,
    FEATURE_INDEX,
    FEATURES,
    STATE_DIM,
    EndogenousState,
    assemble_state,
    empty_state,
    layout_digest,
    pool_substrate,
)


def test_every_dimension_has_a_name_and_a_channel():
    assert len(FEATURES) == STATE_DIM
    assert len(FEATURE_INDEX) == STATE_DIM, "two dimensions share a name"
    for feature in FEATURES:
        assert feature.channel in CHANNELS
        assert feature.meaning.strip(), f"{feature.name} says nothing about itself"
        assert feature.low < feature.high


def test_the_layout_digest_changes_when_the_layout_does():
    before = layout_digest()
    assert before == layout_digest(), "the digest is not stable within a build"
    assert len(before) == 32


def test_an_unanswered_state_is_absent_not_zero():
    state = empty_state()
    assert state.coverage == 0.0
    assert all(value is None for value in state.named().values())
    assert not state.live_channels


def test_a_probe_that_raises_costs_one_channel_not_the_state():
    def explode() -> dict[str, float]:
        raise RuntimeError("organ down")

    state = assemble_state(
        probes={
            "affect": explode,
            "goal": lambda: {"goal.active": 1.0, "goal.priority": 0.8},
        }
    )
    assert state.sources["affect"] == "error"
    assert state.sources["goal"] == "live"
    assert state.is_present("goal.active")
    assert not state.is_present("affect.valence")


def test_do_returns_a_copy_and_records_the_intervention():
    state = empty_state()
    treated = state.do(**{"uncertainty.confidence": 0.9})
    assert state.get("uncertainty.confidence") == 0.0
    assert not state.is_present("uncertainty.confidence")
    assert treated.get("uncertainty.confidence") == pytest.approx(0.9, abs=1e-6)
    assert treated.is_present("uncertainty.confidence")
    assert treated.interventions[-1].feature == "uncertainty.confidence"
    assert treated.interventions[-1].was_present is False


def test_do_clamps_to_the_declared_range():
    state = empty_state().do(**{"uncertainty.confidence": 5.0})
    assert state.get("uncertainty.confidence") == pytest.approx(1.0)
    state = empty_state().do(**{"affect.valence": -9.0})
    assert state.get("affect.valence") == pytest.approx(-1.0)


def test_do_on_an_unknown_dimension_raises():
    with pytest.raises(KeyError):
        empty_state().do(**{"not.a.dimension": 1.0})


def test_ablation_clears_the_value_and_the_presence_bit():
    state = empty_state().do(**{"memory.recall_hits": 0.9, "goal.active": 1.0})
    ablated = state.ablate("memory")
    assert ablated.get("memory.recall_hits") == 0.0
    assert not ablated.is_present("memory.recall_hits")
    assert ablated.sources["memory"] == "ablated"
    assert ablated.is_present("goal.active"), "ablation reached another channel"


def test_the_payload_round_trips_and_refuses_another_layout():
    state = assemble_state(probes={"goal": lambda: {"goal.active": 1.0}})
    payload = state.to_payload()
    restored = EndogenousState.from_payload(payload)
    assert restored is not None
    assert np.allclose(restored.values, state.values)
    assert restored.present.tolist() == state.present.tolist()

    payload["layout"] = "0" * 32
    assert EndogenousState.from_payload(payload) is None


def test_interventions_survive_the_payload():
    """Otherwise an experimental state comes back looking observed."""
    state = empty_state().do(**{"uncertainty.confidence": 0.9})
    restored = EndogenousState.from_payload(state.to_payload())
    assert restored is not None
    assert [i.feature for i in restored.interventions] == ["uncertainty.confidence"]


def test_a_malformed_payload_is_refused_rather_than_repaired():
    state = empty_state()
    for broken in ({}, {"layout": layout_digest()}, {"layout": layout_digest(), "values": [1.0]}):
        assert EndogenousState.from_payload(broken) is None
    payload = state.to_payload()
    payload["values"] = [float("nan")] * STATE_DIM
    assert EndogenousState.from_payload(payload) is None


def test_substrate_pooling_is_declared_not_random():
    vector = np.arange(64, dtype=np.float32)
    first = pool_substrate(vector)
    second = pool_substrate(vector)
    assert np.allclose(first, second), "pooling is not reproducible"
    assert first.shape == (32,)
    assert np.all(np.isfinite(pool_substrate([float("nan"), float("inf"), 1.0])))
    assert pool_substrate([]).shape == (32,)
