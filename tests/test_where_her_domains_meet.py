"""Where two of her domains meet, units answer to both at once, and the meeting acts.

On seed 7 at 370 rounds the held-out gain from letting two domains multiply was
0.0 for three of the four meetings the specification names: her domains only
ever combined by sums. core/consciousness/conjunctive_codes.py puts a small
population of mixed-selectivity units at each meeting, bound as each cycle
closes, read by the domain they live in, and acting there.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.consciousness import conjunctive_codes as codes
from core.consciousness.conjunctive_codes import (
    BAND_BASE,
    MEETINGS,
    UNITS,
    Conjunctions,
    bind,
    gain,
    gain_from_units,
    into_substrate,
)
from core.state.aura_state import AuraState


def _lived(turns: int = 12, seed: int = 0) -> AuraState:
    """A state whose affect, self, perception and drives have moved both ways."""
    state = AuraState.default()
    rng = np.random.default_rng(seed)
    for _ in range(turns):
        state.affect.valence = float(rng.normal(0.0, 0.3))
        state.affect.arousal = float(rng.uniform())
        state.affect.curiosity = float(rng.uniform())
        state.identity.stability = float(rng.uniform())
        state.identity.bonding_level = float(rng.uniform())
        state.cognition.current_objective = "word " * int(rng.integers(1, 30))
        state.cognition.working_memory = [{"content": "said " * int(rng.integers(1, 20))}]
        state.world.partner_register = {"asking": float(rng.uniform()), "first": float(rng.uniform())}
        for name in ("social", "curiosity", "growth"):
            state.motivation.budgets[name]["level"] = float(rng.uniform(20, 90))
        bind(state)
    return state


def test_a_unit_is_a_product_of_its_two_domains():
    meeting = MEETINGS[0]
    x = np.array([1.0, -0.5, 0.2])
    y = np.array([0.3, 0.9])
    units = np.asarray(codes._units(meeting, x, y))
    assert np.allclose(codes._units(meeting, x, np.zeros(2)), 0.0)
    assert np.allclose(codes._units(meeting, np.zeros(3), y), 0.0)
    assert np.allclose(codes._units(meeting, -x, y), -units)
    assert np.allclose(codes._units(meeting, -x, -y), units)


def test_the_four_meetings_are_the_specifications():
    assert [(m.sources, m.target) for m in MEETINGS] == [
        (("A", "S"), "G"),
        (("P", "M"), "W"),
        (("W", "A"), "D"),
        (("S", "D"), "C"),
    ]


def test_a_field_that_has_only_grown_reads_as_rest():
    held = Conjunctions()
    for value in range(10):
        out = held.standardise("N", [float(value), float(value % 2)])
    assert out[0] == 0.0
    assert out[1] != 0.0


def test_bound_units_live_on_her_state_and_move_with_her():
    state = _lived()
    units = state.conjunctions.units
    assert set(units) == {meeting.name for meeting in MEETINGS}
    assert all(len(values) == UNITS for values in units.values())
    assert any(abs(v) > 1e-3 for v in units["attention"])
    assert any(abs(v) > 1e-3 for v in units["recurrent"])


def test_rest_and_the_ablation_leave_every_gain_at_one(monkeypatch):
    assert gain(AuraState.default(), "attention", "memory") == 1.0
    assert gain_from_units((), "world", "learning") == 1.0
    state = _lived()
    assert gain(state, "attention", "memory") != 1.0
    monkeypatch.setenv(codes.DISABLE_ENV, "1")
    assert gain(state, "attention", "memory") == 1.0
    assert bind(AuraState.default()) is None


def test_two_sources_take_different_gains_from_one_code():
    state = _lived()
    gains = {source: gain(state, "attention", source) for source in ("memory", "affect", "perception", "drive")}
    assert len({round(value, 9) for value in gains.values()}) == len(gains)


def test_the_workspace_scales_each_bid_by_its_source():
    from core.consciousness.global_workspace import CognitiveCandidate, _bids

    candidates = [
        CognitiveCandidate(content="a", source="memory", priority=0.5),
        CognitiveCandidate(content="b", source="affect", priority=0.5),
    ]
    units = (0.6, -0.4, 0.3, 0.5)
    plain = _bids(candidates, {}, 0.0)
    scaled = _bids(candidates, {}, 0.0, units)
    for candidate in candidates:
        expected = plain[id(candidate)] * gain_from_units(units, "attention", candidate.source)
        assert scaled[id(candidate)] == pytest.approx(expected, rel=1e-6)


def test_the_world_model_steps_further_at_a_higher_precision():
    from core.world_model.learned_world_model import LearnedWorldModel

    model = LearnedWorldModel()
    name = next(iter(model._adam_m))
    grads = {key: np.ones_like(getattr(model, key)) for key in model._adam_m}
    before = np.array(getattr(model, name), copy=True)
    model._adam_step(grads, rate=2.0)
    far = np.abs(getattr(model, name) - before).mean()
    other = LearnedWorldModel()
    before = np.array(getattr(other, name), copy=True)
    other._adam_step(grads, rate=1.0)
    near = np.abs(getattr(other, name) - before).mean()
    assert far == pytest.approx(2.0 * near, rel=1e-3)


def test_all_sixteen_units_reach_their_band_of_the_substrate():
    from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

    substrate = LiquidSubstrate(config=SubstrateConfig(neuron_count=128))
    state = _lived()
    before = np.array(substrate.x, copy=True)
    moved = into_substrate(substrate, state.conjunctions)
    assert moved > 0.0
    changed = np.flatnonzero(~np.isclose(substrate.x, before))
    assert changed.min() >= BAND_BASE and changed.max() < BAND_BASE + 4 * UNITS
    assert BAND_BASE > 70  # above the intention band at 68


def test_the_core_reads_each_code_in_the_domain_it_lives_in():
    from core.subject.clamp import CLAMPED_FIELDS
    from core.subject.state import CONJUNCTION_UNITS, Organs, read_core_state, schema

    assert CONJUNCTION_UNITS == UNITS
    state = _lived()
    reading = read_core_state(state, organs=Organs())
    for meeting in MEETINGS:
        features = list(schema(meeting.target).features)
        values = [
            float(reading.values[meeting.target][features.index(f"conj_{meeting.name}_{k}")])
            for k in range(UNITS)
        ]
        assert values == pytest.approx(list(state.conjunctions.units[meeting.name]))
        assert f"conjunctions.units.{meeting.name}" in CLAMPED_FIELDS[meeting.target]
