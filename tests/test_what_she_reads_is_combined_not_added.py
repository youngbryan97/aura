"""Three places where two things she reads are combined rather than added.

Every subject-core run failed synergy, and not because of its null. Scored on
how each target changed, the joint information two sources carried about it was
a few thousandths of a nat: nothing in the organism made a target depend on
the product of two of its readings. These pin the three mechanisms that do.

- Arousal is a gain on prediction error, in both directions (Mather et al.,
  2016). A surprise about herself claims attention in proportion to how aroused
  she is, and so does a surprising world pressing on her drives.
- An intention is obstructed in proportion to how hard it presses and how far
  it is beyond what she has shown she can do, and obstruction raises the
  continuous substrate's frustration.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.affect.arousal_gain import gained
from core.state.aura_state import AuraState

pytestmark = pytest.mark.unit


# ── the gain ──────────────────────────────────────────────────────────────


def test_arousal_moves_a_level_away_from_its_usual_point_in_both_directions() -> None:
    assert gained(0.8, 0.0, usual=0.5) == pytest.approx(0.8)
    assert gained(0.5, 0.9, usual=0.5) == pytest.approx(0.5)
    assert gained(0.7, 0.5, usual=0.5) == pytest.approx(0.8)
    assert gained(0.3, 0.5, usual=0.5) == pytest.approx(0.2)
    assert gained(0.95, 1.0, usual=0.5) == 1.0
    assert gained(0.05, 1.0, usual=0.5) == 0.0


def test_neither_reading_decides_the_effect_of_the_other() -> None:
    """The effect of arousal changes sign with the level, which a sum cannot do."""
    above = gained(0.7, 0.8, usual=0.5) - gained(0.7, 0.2, usual=0.5)
    below = gained(0.3, 0.8, usual=0.5) - gained(0.3, 0.2, usual=0.5)
    assert above > 0.0 > below


# ── a surprise about herself, gained by arousal, in the workspace ────────


def _self_bid(monkeypatch: pytest.MonkeyPatch, *, error: float, usual: float, arousal: float) -> float:
    import core.runtime.service_registry as registry
    from core.consciousness.workspace_feed import build_candidates

    snapshot = {
        "smoothed_error": error,
        "valence_error_ema": usual,
        "drive_error_ema": usual,
        "focus_error_ema": usual,
        "most_unpredictable": "affect_valence",
    }
    predictor = SimpleNamespace(get_snapshot=lambda: snapshot)
    monkeypatch.setattr(
        registry,
        "get_runtime_service",
        lambda name, default=None: predictor if name == "self_prediction" else default,
    )
    state = AuraState.default()
    state.affect.arousal = arousal
    bids = [bid for bid in build_candidates(state) if bid.source == "self" and "predicted" in bid.content]
    return max((float(bid.priority) for bid in bids), default=0.0)


def test_a_surprise_about_herself_claims_more_when_she_is_aroused(monkeypatch: pytest.MonkeyPatch) -> None:
    calm = _self_bid(monkeypatch, error=0.9, usual=0.2, arousal=0.1)
    aroused = _self_bid(monkeypatch, error=0.9, usual=0.2, arousal=0.9)
    assert aroused > calm


def test_being_easier_to_predict_than_usual_claims_less_when_she_is_aroused(monkeypatch: pytest.MonkeyPatch) -> None:
    calm = _self_bid(monkeypatch, error=0.1, usual=0.3, arousal=0.1)
    aroused = _self_bid(monkeypatch, error=0.1, usual=0.3, arousal=0.9)
    assert aroused < calm


# ── a surprising world, gained by arousal, pressing on the drives ────────


class _WorldModel:
    def __init__(self, surprise: float, usual: float) -> None:
        self._surprise = surprise
        self._usual = usual

    def surprise(self) -> float:
        return self._surprise

    def status(self) -> dict:
        return {"facets": {"learned": {"detail": {"mean_surprise": self._usual}}}}


def _pressure(monkeypatch: pytest.MonkeyPatch, model: object, arousal: float) -> float:
    import core.phases.motivation_update as motivation
    from core.phases.motivation_update import MotivationUpdatePhase

    monkeypatch.setattr(
        motivation,
        "get_runtime_service",
        lambda name, default=None: model if name == "unified_world_model" else default,
    )
    state = AuraState.default()
    state.affect.arousal = arousal
    return MotivationUpdatePhase(SimpleNamespace(organs={}))._surprise_pressure(state)


def test_a_world_worse_than_its_model_expects_presses_harder_on_an_aroused_organism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worse = _WorldModel(surprise=6.0, usual=3.0)
    assert _pressure(monkeypatch, worse, 0.9) > _pressure(monkeypatch, worse, 0.1)


def test_a_world_better_than_expected_presses_less_on_an_aroused_organism(monkeypatch: pytest.MonkeyPatch) -> None:
    better = _WorldModel(surprise=1.5, usual=3.0)
    assert _pressure(monkeypatch, better, 0.9) < _pressure(monkeypatch, better, 0.1)


def test_the_pressure_stays_within_its_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    for surprise in (0.0, 1.0, 3.0, 50.0):
        for arousal in (0.0, 0.5, 1.0):
            value = _pressure(monkeypatch, _WorldModel(surprise=surprise, usual=3.0), arousal)
            assert 0.0 <= value <= 1.0


# ── an obstructed intention raises the substrate's frustration ───────────


def _state_with(urgency: float | None, capacity: float | None) -> AuraState:
    state = AuraState.default()
    state.cognition.pending_initiatives = [] if urgency is None else [{"goal": "finish the index", "urgency": urgency}]
    state.cognition.active_goals = []
    if capacity is not None:
        state.identity.capacity = {"capacity": capacity}
    return state


def test_obstruction_is_how_hard_it_presses_times_how_far_it_is_beyond_her() -> None:
    from core.phases.affect_update import _obstruction

    assert _obstruction(_state_with(0.8, 0.25)) == pytest.approx(0.6)
    assert _obstruction(_state_with(0.8, 1.0)) == pytest.approx(0.0)
    assert _obstruction(_state_with(None, 0.25)) == pytest.approx(0.0)
    assert _obstruction(_state_with(0.8, None)) == pytest.approx(0.4)


class _Substrate:
    def __init__(self, frustration: float) -> None:
        self.frustration = frustration
        self.calls: list[dict] = []

    def get_substrate_affect(self) -> dict:
        return {"curiosity": 0.0}

    def current(self) -> SimpleNamespace:
        return SimpleNamespace(frustration=self.frustration)

    async def update(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _push(substrate: _Substrate, state: AuraState) -> dict:
    from core.phases.affect_update import AffectUpdatePhase

    phase = object.__new__(AffectUpdatePhase)
    asyncio.run(phase._push_to_substrate(substrate, state.affect, state))
    (call,) = substrate.calls
    return call


def test_an_obstructed_intention_raises_the_substrate_s_frustration_towards_it() -> None:
    call = _push(_Substrate(frustration=0.1), _state_with(0.8, 0.25))
    assert call["delta_frustration"] == pytest.approx(0.5)


def test_frustration_already_above_the_obstruction_is_left_to_decay_on_its_own() -> None:
    call = _push(_Substrate(frustration=0.9), _state_with(0.8, 0.25))
    assert call["delta_frustration"] == pytest.approx(0.0)
