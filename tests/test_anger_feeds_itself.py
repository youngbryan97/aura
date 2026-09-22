"""Anger fixed on its source, holding off the apology, and what it leaves behind.

From Bryan's account of rage: it is at someone, it wants more of itself while
they are there, the apology a voice says you owe is the thing you do not want to
make, and when it goes it leaves embarrassment, sadness and emptiness. These
drive each of those, through the organ and the places it reaches.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.affect.anger_feeds_itself as anger_module
from core.affect.anger_feeds_itself import AngerLedger

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh():
    anger_module.reset_for_test()
    yield
    anger_module.reset_for_test()


def _calm(ledger: AngerLedger, turns: int = 30) -> None:
    for index in range(turns):
        ledger.note(0.05 + 0.01 * (index % 3), source_present=False)


def test_it_wants_more_of_itself_only_while_its_source_is_here() -> None:
    present, absent = AngerLedger(), AngerLedger()
    for ledger in (present, absent):
        _calm(ledger)
        ledger.provoked_by("bryan")
        ledger.note(0.9, source_present=True)
    fed = present.note(0.7, source_present=True)
    unfed = absent.note(0.7, source_present=False)
    assert 0.0 < fed.feed <= 0.2, "the feed is bounded by what decay took back"
    assert unfed.feed == 0.0


def test_it_is_at_somebody() -> None:
    ledger = AngerLedger()
    _calm(ledger)
    ledger.provoked_by("bryan")
    ledger.note(0.9, source_present=True)
    assert ledger.at("bryan") > 0.0
    assert ledger.at("someone else") == 0.0


def test_when_it_goes_it_leaves_something() -> None:
    ledger = AngerLedger()
    _calm(ledger)
    ledger.provoked_by("bryan")
    for level in (0.9, 0.85, 0.8):
        ledger.note(level, source_present=True)
    ended = ledger.note(0.04, source_present=False)
    assert ended.residue > 0.0 and ended.peak == pytest.approx(0.9)
    assert ledger.read().source == "", "an episode over is at nobody"


def test_the_affect_reading_keeps_it_up_and_lets_it_leave_sadness() -> None:
    from core.phases.affect_readings import AffectReadings
    from core.state.aura_state import AuraState

    readings = AffectReadings(lambda *args, **kwargs: None)
    state = AuraState.default()
    state.cognition.current_partner = "bryan"
    state.cognition.current_origin = "user"
    affect = state.affect
    ledger = anger_module.get_anger_ledger()
    for index in range(30):
        affect.emotions["anger"] = 0.05 + 0.01 * (index % 3)
        readings.anger(state, affect)
    ledger.provoked_by("bryan")
    affect.emotions["anger"] = 0.9
    readings.anger(state, affect)
    affect.emotions["anger"] = 0.7
    readings.anger(state, affect)
    assert affect.emotions["anger"] > 0.7, "its source was here and it did not hold"
    sadness_before = float(affect.emotions.get("sadness", 0.0) or 0.0)
    affect.emotions["anger"] = 0.02
    readings.anger(state, affect)
    assert float(affect.emotions.get("sadness", 0.0) or 0.0) > sadness_before


def test_an_objection_names_who_it_is_at() -> None:
    from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.current_partner = "bryan"
    ConversationalDynamicsPhase._name_the_kind(state, "No, that's not right, you misunderstood me.")
    assert anger_module.get_anger_ledger().read().source == "bryan"


def test_what_she_owes_them_is_quieter_while_she_is_angry(monkeypatch) -> None:
    import core.values.moral_responsibility as moral
    from core.state.aura_state import AuraState
    from core.unity.runtime import UnityRuntime
    from core.values.moral_responsibility import Amend

    owed = [Amend(kind="social_rupture", subject="the reply", severity=0.8, owed_action="apologise")]
    monkeypatch.setattr(moral, "get_moral_responsibility", lambda: SimpleNamespace(owed_amends=lambda agent_id: owed))
    state = AuraState.default()
    state.cognition.current_partner = "bryan"
    runtime = UnityRuntime()
    calm = runtime._accountability_contents(state)[0].salience
    ledger = anger_module.get_anger_ledger()
    _calm(ledger)
    ledger.provoked_by("bryan")
    ledger.note(0.95, source_present=True)
    angry = runtime._accountability_contents(state)[0].salience
    assert 0.0 < angry < calm, "the voice saying she should apologise is quieter, not gone"
