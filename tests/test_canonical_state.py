"""One authoritative value per canonical variable, many estimators.

Aura's largest architectural risk is ontological duplication inside her own
software. Affect was owned by the liquid substrate, the Damasio engine, the
interiority faculties, user-sentiment analysis and the phenomenal substrate;
selfhood by SelfObject, AuraNow.self_state, the identity engine, the
continuity engine, workspace ownership and the substrate. Each a coherent
answer to "how is Aura", none of them the same answer, nothing deciding.

These tests hold the shape of the fix: subsystems estimate, the canonical
state fuses, and where the estimators disagree that fact survives instead of
being averaged into a number describing none of them.
"""

from __future__ import annotations

import pytest

from core.canonical.channels import BY_ID, CHANNELS, Domain, channel, in_domain
from core.canonical.state import (
    DISAGREEMENT_THRESHOLD,
    FULL_WEIGHT_S,
    MIN_ESTIMATORS_FOR_DISAGREEMENT,
    NO_WEIGHT_S,
    CanonicalState,
)


@pytest.fixture
def state():
    clock = [1000.0]
    s = CanonicalState(now=lambda: clock[0])
    s._clock = clock
    return s


# ── the channels ─────────────────────────────────────────────────────────


def test_every_channel_has_a_meaning_a_second_implementer_could_use():
    """Same name and same range is not the same quantity."""
    for spec in CHANNELS:
        assert len(spec.meaning) > 40, f"{spec.id} has no usable meaning"
        assert spec.low < spec.high, f"{spec.id} has an empty range"
        assert spec.low <= spec.neutral <= spec.high, f"{spec.id} neutral is outside it"


def test_channel_ids_are_unique_and_domain_prefixed():
    assert len(BY_ID) == len(CHANNELS)
    for spec in CHANNELS:
        assert spec.id.startswith(f"{spec.domain}."), spec.id


def test_every_domain_has_at_least_one_channel():
    for domain in Domain:
        assert in_domain(domain), f"{domain} owns nothing"


def test_an_undeclared_channel_raises_rather_than_creating_one(state):
    """A misspelt id would otherwise be a second variable nothing reads."""
    with pytest.raises(KeyError):
        channel("affect.valance")
    with pytest.raises(KeyError):
        state.estimate("affect.valance", 0.5, confidence=1.0, producer="p")


# ── fusion ───────────────────────────────────────────────────────────────


def test_a_channel_nobody_estimates_reports_that_it_is_a_default(state):
    reading = state.get("affect.valence")
    assert reading.is_default is True
    assert reading.confidence == 0.0
    assert reading.value == channel("affect.valence").neutral


def test_a_confident_estimator_outweighs_an_unsure_one(state):
    state.estimate("affect.valence", 1.0, confidence=0.9, producer="sure")
    state.estimate("affect.valence", -1.0, confidence=0.1, producer="guessing")
    assert state.value("affect.valence") > 0.5


def test_one_producer_estimating_twice_has_changed_its_mind_not_gained_a_vote(state):
    state.estimate("affect.valence", 1.0, confidence=0.9, producer="p")
    state.estimate("affect.valence", 1.0, confidence=0.9, producer="p")
    state.estimate("affect.valence", -1.0, confidence=0.9, producer="q")
    assert state.value("affect.valence") == pytest.approx(0.0, abs=1e-6)
    assert state.get("affect.valence").contributors == ("p", "q")


def test_a_stale_estimate_stops_steering_the_answer(state):
    state.estimate("affect.valence", 1.0, confidence=1.0, producer="quiet")
    state._clock[0] += NO_WEIGHT_S + 1.0
    assert state.get("affect.valence").is_default is True


def test_an_estimate_inside_the_full_weight_window_is_not_decayed(state):
    state.estimate("affect.valence", 1.0, confidence=1.0, producer="p")
    state._clock[0] += FULL_WEIGHT_S - 1.0
    assert state.value("affect.valence") == pytest.approx(1.0)


def test_values_are_clamped_to_the_declared_range(state):
    state.estimate("affect.arousal", 5.0, confidence=1.0, producer="p")
    assert state.value("affect.arousal") == 1.0


def test_a_non_finite_estimate_is_refused(state):
    with pytest.raises(ValueError):
        state.estimate("affect.valence", float("nan"), confidence=1.0, producer="p")
    with pytest.raises(ValueError):
        state.estimate("affect.valence", float("inf"), confidence=1.0, producer="p")


def test_retracting_removes_a_producer(state):
    state.estimate("affect.valence", 1.0, confidence=1.0, producer="p")
    assert state.retract("affect.valence", "p") is True
    assert state.get("affect.valence").is_default is True


# ── disagreement ─────────────────────────────────────────────────────────


def test_disagreement_survives_instead_of_being_averaged(state):
    """0.2, 0.5 and 0.9 is not a system that believes 0.53."""
    for producer, value in (("tracker", 0.2), ("verifier", 0.5), ("calibration", 0.9)):
        state.estimate("epistemic.uncertainty", value, confidence=0.7, producer=producer)
    reading = state.get("epistemic.uncertainty")
    assert reading.disagreed is True
    assert reading.spread > DISAGREEMENT_THRESHOLD
    events = state.disagreements()
    assert len(events) == 1
    assert dict(events[0].positions)["tracker"] == pytest.approx(0.2)


def test_agreeing_estimators_raise_nothing(state):
    for producer, value in (("a", 0.30), ("b", 0.32), ("c", 0.29)):
        state.estimate("affect.valence", value, confidence=0.7, producer=producer)
    assert state.get("affect.valence").disagreed is False
    assert state.disagreements() == ()


def test_two_estimators_cannot_disagree(state):
    """With two there is no way to tell which is the outlier."""
    state.estimate("affect.valence", -1.0, confidence=0.9, producer="a")
    state.estimate("affect.valence", 1.0, confidence=0.9, producer="b")
    assert MIN_ESTIMATORS_FOR_DISAGREEMENT == 3
    assert state.get("affect.valence").disagreed is False


def test_reading_the_state_repeatedly_does_not_grow_the_event_list(state):
    for producer, value in (("a", 0.05), ("b", 0.5), ("c", 0.95)):
        state.estimate("epistemic.uncertainty", value, confidence=0.7, producer=producer)
    for _ in range(50):
        state.get("epistemic.uncertainty")
    assert len(state.disagreements()) == 1, "reading the state is a way to fill memory"


def test_taking_disagreements_clears_them(state):
    for producer, value in (("a", 0.05), ("b", 0.5), ("c", 0.95)):
        state.estimate("epistemic.uncertainty", value, confidence=0.7, producer=producer)
    state.get("epistemic.uncertainty")
    assert state.take_disagreements()
    assert state.disagreements() == ()


# ── disagreement as a cognitive event ────────────────────────────────────


def test_disagreement_becomes_evidence_about_how_sure_she_is(state):
    for producer, value in (("a", -0.9), ("b", 0.1), ("c", 0.85)):
        state.estimate("affect.valence", value, confidence=0.7, producer=producer)
    state.get("affect.valence")
    before = state.get("epistemic.uncertainty")
    assert before.is_default is True
    result = state.reconcile()
    assert result["reconciled"] == 1
    after = state.get("epistemic.uncertainty")
    assert after.is_default is False and after.value > before.value
    assert state.value("self.coherence") < channel("self.coherence").neutral


def test_agreement_produces_no_metacognitive_event(state):
    for producer, value in (("a", 0.30), ("b", 0.32), ("c", 0.29)):
        state.estimate("affect.valence", value, confidence=0.7, producer=producer)
    state.get("affect.valence")
    assert state.reconcile()["reconciled"] == 0
    assert state.get("epistemic.uncertainty").is_default is True


def test_reconciliation_cannot_run_away_on_its_own_evidence(state):
    """Disagreeing about coherence must not lower coherence, forever."""
    for producer, value in (("a", -0.9), ("b", 0.1), ("c", 0.85)):
        state.estimate("affect.valence", value, confidence=0.7, producer=producer)
    for producer, value in (("x", 0.05), ("y", 0.5), ("z", 0.95)):
        state.estimate("epistemic.uncertainty", value, confidence=0.7, producer=producer)
    seen = []
    for _ in range(8):
        state.reconcile()
        seen.append((state.value("epistemic.uncertainty"), state.value("self.coherence")))
    assert len(set(seen)) == 1, f"the reconciliation drifted on its own output: {seen}"


def test_reconciliation_does_not_depend_on_who_read_what(state):
    """Disagreement is detected on read, so an unread channel had none."""
    for producer, value in (("a", -0.9), ("b", 0.1), ("c", 0.85)):
        state.estimate("affect.valence", value, confidence=0.7, producer=producer)
    # Nobody has read affect.valence. reconcile() must still find the conflict.
    assert state.reconcile()["reconciled"] == 1


# ── the producers that used to own their own copy ────────────────────────


def test_the_subsystems_that_owned_affect_are_estimators_now():
    """Each of these kept a private answer to how she is."""
    import inspect

    from core.affect.damasio_v2 import DamasioMarkers
    from core.being.welfare_state import WelfareState
    from core.interiority.service import InteriorityService

    for owner, method in (
        (DamasioMarkers, "_estimate_canonical"),
        (WelfareState, "_estimate_canonical"),
        (InteriorityService, "_estimate_canonical"),
    ):
        assert hasattr(owner, method), f"{owner.__name__} no longer estimates"
        source = inspect.getsource(getattr(owner, method))
        assert "core.canonical.state" in source


def test_the_canonical_package_cannot_reach_its_estimators():
    """A state layer that can fetch a value grows a second answer."""
    deps = (
        pytest.importorskip("pathlib").Path("core/canonical/DEPS").read_text(encoding="utf-8")
    )
    assert '"-core",' in deps
    allowed = {line.strip().strip('",') for line in deps.splitlines() if line.strip().startswith('"+')}
    assert allowed == {"+core.canonical", "+core.runtime"}, (
        f"core/canonical may now import {allowed}; the estimated must not be "
        "able to reach the estimators"
    )
