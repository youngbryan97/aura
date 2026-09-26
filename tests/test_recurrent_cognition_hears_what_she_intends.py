"""The substrate had no band for deliberation or the self.

Its input bands carry telemetry, the person's presence, the screen, the room's
sound, and three cross-modal products. Nothing of what she means to do and
nothing of how settled she is in herself reached it, so recurrent cognition
never heard either. The synergy line asks whether the self and deliberation
carry something about recurrent cognition jointly that neither carries alone,
and on the seed-7 run of 25 September its interaction gain was exactly zero on
three folds of five: there was no pathway for it to be about.

The band is built the way the one beside it is — two readings and their cross
term, because pressing hard while unsettled is a different state from pressing
hard while settled and a sum of the two cannot say which.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig


def _substrate() -> LiquidSubstrate:
    return LiquidSubstrate(SubstrateConfig(neuron_count=128))


def _band(substrate: LiquidSubstrate) -> np.ndarray:
    base = substrate._INTENTION_BASE
    return np.array(substrate.x[base : base + 3], dtype=np.float64)


def test_a_frame_with_neither_reading_leaves_the_band_alone():
    substrate = _substrate()
    before = _band(substrate)
    substrate.inject_perceptual_frame({"cpu_percent": 10.0})
    assert np.allclose(_band(substrate), before)


def test_what_she_means_to_do_reaches_the_band():
    substrate = _substrate()
    substrate.inject_perceptual_frame({"intent_urgency": 0.9, "self_stability": 0.0})
    assert _band(substrate)[0] != 0.0


def test_how_settled_she_is_reaches_the_band():
    substrate = _substrate()
    substrate.inject_perceptual_frame({"intent_urgency": 0.0, "self_stability": 0.9})
    assert _band(substrate)[1] != 0.0


def test_the_cross_term_needs_both_and_is_not_their_sum():
    """Pressing hard while unsettled is its own state."""
    unsettled = _substrate()
    unsettled.inject_perceptual_frame({"intent_urgency": 0.9, "self_stability": 0.0})
    settled = _substrate()
    settled.inject_perceptual_frame({"intent_urgency": 0.9, "self_stability": 1.0})
    assert _band(unsettled)[2] > _band(settled)[2]
    idle = _substrate()
    idle.inject_perceptual_frame({"intent_urgency": 0.0, "self_stability": 0.0})
    assert _band(idle)[2] == pytest.approx(0.0)


def test_the_band_does_not_reach_past_a_narrow_substrate():
    """`put` guards the index, the way it does for every band above sixteen."""
    narrow = LiquidSubstrate(SubstrateConfig(neuron_count=16))
    narrow.inject_perceptual_frame({"intent_urgency": 1.0, "self_stability": 1.0})
    assert narrow.x.shape[0] == 16


def test_the_band_does_not_land_on_another_one():
    substrate = _substrate()
    base = substrate._INTENTION_BASE
    assert base >= 67, "the cross-modal terms end at 66"


# ── the reading the loop hands it ────────────────────────────────────────


def test_the_hardest_thing_she_holds_is_what_presses():
    from core.phases.proprioceptive_loop import _urgency_now
    from core.state.aura_state import AuraState

    state = AuraState.default()
    assert _urgency_now(state) == pytest.approx(0.0)
    state.cognition.active_goals = [{"urgency": 0.2}, {"urgency": 0.7}]
    state.cognition.pending_initiatives = [{"urgency": 0.45}]
    assert _urgency_now(state) == pytest.approx(0.7)


def test_an_initiative_can_be_the_hardest_thing():
    from core.phases.proprioceptive_loop import _urgency_now
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.active_goals = [{"urgency": 0.1}]
    state.cognition.pending_initiatives = [{"urgency": 0.85}]
    assert _urgency_now(state) == pytest.approx(0.85)


def test_something_with_no_urgency_presses_by_nothing():
    from core.phases.proprioceptive_loop import _urgency_now
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.active_goals = [{"what": "no urgency here"}, {"urgency": "soon"}]
    assert _urgency_now(state) == pytest.approx(0.0)


def test_an_object_with_an_urgency_attribute_counts():
    from types import SimpleNamespace

    from core.phases.proprioceptive_loop import _urgency_now
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.active_goals = [SimpleNamespace(urgency=0.6)]
    assert _urgency_now(state) == pytest.approx(0.6)


def test_it_stays_inside_zero_and_one():
    from core.phases.proprioceptive_loop import _urgency_now
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.active_goals = [{"urgency": 40.0}, {"urgency": -3.0}]
    assert _urgency_now(state) == pytest.approx(1.0)
