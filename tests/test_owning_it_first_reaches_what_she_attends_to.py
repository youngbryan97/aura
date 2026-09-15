"""Owning a lapse first, wired: the events are noted where they happen and the reading moves attention.

The ledger is pinned in test_owning_a_lapse_before_it_is_raised.py. These pin
the wiring: a reply going out from a moment that held what she owed is her
owning it, a complaint about a reply that did not is theirs to raise, the
reading reaches her identity and the self domain, and a measured positive lift
gives what she owes a larger share of the moment.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.social.owning_it_first as owning
from core.social.other_agent_model import OtherAgentStateEstimator
from core.values.moral_responsibility import Amend


@pytest.fixture(autouse=True)
def fresh_ledger():
    owning.reset_for_test()
    yield
    owning.reset_for_test()


class _Signal:
    def __init__(self, value: float) -> None:
        self.value = value

    def decayed(self, _now: float) -> tuple[float, float]:
        return self.value, 0.5


def _model(frustration: float) -> SimpleNamespace:
    return SimpleNamespace(affect={"frustration": _Signal(frustration)})


def _observer(frustration: float) -> SimpleNamespace:
    model = _model(frustration)
    return SimpleNamespace(
        _models={"sam": model},
        _load_agent=lambda *_a, **_k: model,
        _frustration=OtherAgentStateEstimator._frustration,
    )


def _unity(*modalities: str) -> SimpleNamespace:
    return SimpleNamespace(contents=[SimpleNamespace(modality=m) for m in modalities])


def test_a_reply_made_from_what_she_owed_opens_an_owned_event(monkeypatch) -> None:
    from core.container import ServiceContainer

    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda cls, name, default=None: _unity("responsibility")))
    observer = _observer(0.4)
    OtherAgentStateEstimator._note_delivered(observer, "sam", 10.0)
    OtherAgentStateEstimator._note_heard(observer, "sam", _model(0.3), 20.0, complaint=False)
    reading = owning.get_owning_ledger().reading()
    assert reading.owned_events == 1 and reading.raised_events == 0


def test_a_complaint_about_a_reply_that_owed_nothing_is_raised_by_them(monkeypatch) -> None:
    from core.container import ServiceContainer

    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda cls, name, default=None: _unity("percept")))
    observer = _observer(0.2)
    OtherAgentStateEstimator._note_delivered(observer, "sam", 10.0)
    OtherAgentStateEstimator._note_heard(observer, "sam", _model(0.7), 20.0, complaint=True)
    OtherAgentStateEstimator._note_heard(observer, "sam", _model(0.6), 30.0, complaint=False)
    reading = owning.get_owning_ledger().reading()
    assert reading.raised_events == 1 and reading.owned_events == 0


def _measured_positive_lift() -> None:
    ledger = owning.get_owning_ledger()
    for _ in range(owning.MIN_SAMPLES):
        ledger.delivered("sam", owed_in_mind=True, frustration=0.4)
        ledger.heard("sam", frustration=0.2, complaint=False)
        ledger.delivered("sam", owed_in_mind=False, frustration=0.2)
        ledger.heard("sam", frustration=0.5, complaint=True)
        ledger.heard("sam", frustration=0.7, complaint=False)


def _owed_contents(monkeypatch) -> list:
    import core.values.moral_responsibility as moral
    from core.unity.runtime import UnityRuntime

    amend = Amend(kind="broken_commitment", subject="the summary", severity=0.4, owed_action="say so")
    monkeypatch.setattr(moral, "get_moral_responsibility", lambda: SimpleNamespace(owed_amends=lambda **_k: [amend]))
    state = SimpleNamespace(cognition=SimpleNamespace(current_partner="sam"))
    return UnityRuntime._accountability_contents(None, state)


def test_what_she_owes_takes_a_larger_share_when_owning_first_has_helped(monkeypatch) -> None:
    before = _owed_contents(monkeypatch)[0]
    _measured_positive_lift()
    after = _owed_contents(monkeypatch)[0]
    assert owning.get_owning_ledger().reading().lift > 0.0
    assert after.salience > before.salience
    assert after.action_relevance > before.action_relevance


def test_the_reading_reaches_her_identity_and_the_self_domain() -> None:
    from core.phases.affect_readings import AffectReadings
    from core.state.aura_state import AuraState
    from core.subject.state import feature_names

    _measured_positive_lift()
    state = AuraState.default()
    readings = AffectReadings.__new__(AffectReadings)
    readings.owning_first(state)
    assert state.identity.owning_first["measured"]
    assert "S.owning_first_lift" in feature_names("S")


def test_the_lesion_clamp_holds_it() -> None:
    from core.subject.clamp import CLAMPED_FIELDS

    assert "identity.owning_first" in CLAMPED_FIELDS["S"]
