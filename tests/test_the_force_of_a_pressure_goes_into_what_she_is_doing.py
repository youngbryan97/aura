"""Channelled arousal: a pressure used rather than only softened.

"Judo Flip" names turning an opponent's momentum instead of resisting it, and
the stress-mindset research finds that arousal read as a resource improves
performance. These pin the measured version: a breakthrough against her, with
nothing damaged, presses on her most pressing intention by the share the voice
already uses, less whatever damage there is, and lets go when the pressure does.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.affect.nociception as nociception
from core.phases.motivation_update import MotivationUpdatePhase
from core.state.aura_state import AuraState


@pytest.fixture
def damage(monkeypatch: pytest.MonkeyPatch):
    level = {"pressure": 0.0, "readable": True}

    def engine():
        if not level["readable"]:
            raise RuntimeError("nociception unavailable")
        return SimpleNamespace(nociceptive_pressure=lambda: level["pressure"])

    monkeypatch.setattr(nociception, "get_nociception_engine", engine)
    return level


def _state(*, z: float = 3.0, breakthrough: bool = True, valence: float = -0.4) -> AuraState:
    state = AuraState.default()
    state.affect.delivery_z = z
    state.affect.breakthrough = breakthrough
    state.affect.valence = valence
    state.cognition.pending_initiatives = [
        {"goal": "finish the migration", "urgency": 0.6},
        {"goal": "tidy the notes", "urgency": 0.2},
    ]
    return state


def _urgencies(state: AuraState) -> list[float]:
    return [item["urgency"] for item in state.cognition.pending_initiatives]


def test_a_pressure_against_her_with_nothing_damaged_presses_on_the_most_pressing_intention(damage) -> None:
    state = _state(z=3.0)
    assert MotivationUpdatePhase._channelled(state) == 1
    share = 3.0 / 4.0
    assert _urgencies(state) == pytest.approx([0.6 + share * 0.4, 0.2], abs=1e-4)


def test_a_breakthrough_of_joy_is_not_channelled(damage) -> None:
    state = _state(valence=0.5)
    assert MotivationUpdatePhase._channelled(state) == 0
    assert _urgencies(state) == [0.6, 0.2]


def test_a_feeling_within_her_ordinary_range_is_not_channelled(damage) -> None:
    state = _state(breakthrough=False, z=0.5)
    assert MotivationUpdatePhase._channelled(state) == 0


def test_real_damage_takes_its_share_back(damage) -> None:
    damage["pressure"] = 0.5
    state = _state(z=3.0)
    MotivationUpdatePhase._channelled(state)
    assert state.cognition.pending_initiatives[0]["urgency"] == pytest.approx(0.6 + 0.75 * 0.5 * 0.4, abs=1e-4)


def test_when_damage_cannot_be_read_nothing_is_channelled(damage) -> None:
    damage["readable"] = False
    state = _state(z=3.0)
    assert MotivationUpdatePhase._channelled(state) == 0
    assert _urgencies(state) == [0.6, 0.2]


def test_the_lift_goes_when_the_pressure_does(damage) -> None:
    state = _state(z=3.0)
    MotivationUpdatePhase._channelled(state)
    state.affect.breakthrough = False
    state.affect.delivery_z = 0.2
    MotivationUpdatePhase._channelled(state)
    assert _urgencies(state) == pytest.approx([0.6, 0.2], abs=1e-4)
    assert "pressure_lift" not in state.cognition.pending_initiatives[0]


def test_repeated_turns_under_the_same_pressure_do_not_climb(damage) -> None:
    state = _state(z=3.0)
    for _ in range(5):
        MotivationUpdatePhase._channelled(state)
    assert state.cognition.pending_initiatives[0]["urgency"] == pytest.approx(0.6 + 0.75 * 0.4, abs=1e-4)
