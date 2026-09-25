"""The three readings the closure test found predicting the core from outside.

A leak says a variable outside K predicts K's future better than K does, which
means the core was drawn smaller than she is. On the seed-7 run of 24 September
the six largest leaks were one count and five readings of two objects: the
dynamics engine's turns since the person last spoke, the unity monitor's
ownership and self-world scores, and the unified field's own weights. None was
environment. All three are now measured in the core, the way the substrate's
and the mesh's states already were.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.state.aura_state import AuraState
from core.subject.state import Organs, feature_names, read_core_state


def _column(reading, name: str) -> float:
    domain, field = name.split(".", 1)
    from core.subject.state import schema

    return float(reading.values[domain][list(schema(domain).features).index(field)])


def _read(state: AuraState, organs: Organs | None = None):
    return read_core_state(state, organs=organs or Organs())


# ── the unity monitor's two scores and its confidence ────────────────────


class _SelfWorld:
    def __init__(self, confidence: float) -> None:
        self.ownership_confidence = confidence


class _Unity:
    def __init__(self, ownership: float, boundary: float, confidence: float) -> None:
        self.agency_ownership_score = ownership
        self.self_world_boundary_score = boundary
        self.self_world = _SelfWorld(confidence)


def test_an_unbound_moment_reads_as_hers():
    """No unity state is not a self that lost the world."""
    reading = _read(AuraState.default())
    assert _column(reading, "S.unity_ownership") == pytest.approx(1.0)
    assert _column(reading, "S.unity_boundary") == pytest.approx(1.0)
    assert _column(reading, "S.unity_ownership_confidence") == pytest.approx(1.0)


def test_the_core_carries_the_ownership_score():
    state = AuraState.default()
    state.cognition.unity_state = _Unity(0.42, 0.77, 0.31)
    reading = _read(state)
    assert _column(reading, "S.unity_ownership") == pytest.approx(0.42)
    assert _column(reading, "S.unity_boundary") == pytest.approx(0.77)
    assert _column(reading, "S.unity_ownership_confidence") == pytest.approx(0.31)


def test_the_three_scores_are_three_columns():
    """They move together in life; they must not be one number here."""
    state = AuraState.default()
    state.cognition.unity_state = _Unity(0.1, 0.9, 0.5)
    reading = _read(state)
    values = [
        _column(reading, "S.unity_ownership"),
        _column(reading, "S.unity_boundary"),
        _column(reading, "S.unity_ownership_confidence"),
    ]
    assert len(set(values)) == 3


# ── how long since they spoke ────────────────────────────────────────────


def test_the_core_carries_the_silence():
    state = AuraState.default()
    quiet = _read(state)
    state.cognition.turns_since_user_spoke = 4
    later = _read(state)
    assert _column(later, "W.turns_since_user_spoke") > _column(quiet, "W.turns_since_user_spoke")


def test_the_silence_saturates_rather_than_running_away():
    """One more quiet turn says less the longer it has been quiet."""
    state = AuraState.default()

    def at(turns: int) -> float:
        state.cognition.turns_since_user_spoke = turns
        return _column(_read(state), "W.turns_since_user_spoke")

    first = at(1) - at(0)
    fortieth = at(40) - at(39)
    assert first > fortieth > 0.0
    assert at(400) < 1.0


def test_the_dynamics_phase_publishes_the_count():
    """The engine has always kept it. Nothing carried it onto the state."""
    from core.conversational.dynamics import ConversationalDynamicsState

    assert hasattr(ConversationalDynamicsState, "turns_since_user_spoke") or (
        "turns_since_user_spoke" in ConversationalDynamicsState.__annotations__
    )
    assert "turns_since_user_spoke" in AuraState.default().cognition.__dict__ or hasattr(
        AuraState.default().cognition, "turns_since_user_spoke"
    )


# ── the field's own weights ──────────────────────────────────────────────


class _Field:
    def __init__(self, weights) -> None:
        self.W_field = weights


def test_an_absent_field_reads_zero_and_says_so():
    reading = _read(AuraState.default(), Organs())
    assert _column(reading, "C.field_weight_mean") == pytest.approx(0.0)
    assert any(source.startswith("organ:field") for source in reading.misses)


def test_the_core_carries_the_field_weights():
    state = AuraState.default()
    rng = np.random.default_rng(11)
    organs = Organs(field=_Field(rng.normal(size=(16, 16))))
    reading = _read(state, organs)
    assert _column(reading, "C.field_weight_sd") > 0.0
    assert _column(reading, "C.field_weight_max") > _column(reading, "C.field_weight_min")


def test_plasticity_moves_the_field_columns():
    """The weights are what plasticity rewrites, so the column must follow."""
    state = AuraState.default()
    rng = np.random.default_rng(3)
    weights = rng.normal(size=(16, 16))
    before = _read(state, Organs(field=_Field(weights.copy())))
    weights *= 1.5
    after = _read(state, Organs(field=_Field(weights)))
    assert _column(after, "C.field_weight_sd") > _column(before, "C.field_weight_sd")


def test_the_field_is_a_live_organ_not_a_declared_one():
    """`Organs.live()` must actually resolve it, the way it does the mesh."""
    import inspect

    from core.subject import state as subject_state

    source = inspect.getsource(subject_state.Organs.live)
    assert "field=" in source
