"""What arrives matters by how particular it is to her times how much it meets her now.

Bryan: a compliment when you are low and the same one when you are fine; a
question from somebody you trust and from a stranger; the same news on a tired
day. Neither piece alone would have changed anything. See
core/affect/a_moment_that_fits.py.
"""

from __future__ import annotations

import pytest

import core.affect.a_moment_that_fits as moments
from core.affect.a_moment_that_fits import MomentLedger, particular_share

pytestmark = pytest.mark.unit

HERS = ["I spent the afternoon sketching the orbit of the second moon", "orbit sketches", "draw the moon"]
PARTICULAR = "your sketch of the second moon orbit is lovely, thank you"
GENERIC = "that is lovely, thank you so much"


@pytest.fixture(autouse=True)
def _fresh():
    moments.reset_for_test()
    yield
    moments.reset_for_test()


def _lived(ledger: MomentLedger, valences=(0.2, 0.3, 0.25, 0.35, 0.3)) -> None:
    for valence in valences:
        ledger.felt(valence)
    for message in ("hello there", "what time is it", "good morning to you", "okay then"):
        ledger.arrived(message, HERS, warm=False, question=False, valence=0.3, trust=None, tired=0.0)


def test_particular_is_about_her_particulars() -> None:
    assert particular_share(PARTICULAR, HERS) > particular_share(GENERIC, HERS) == 0.0


def test_a_particular_compliment_when_she_is_low_is_a_moment() -> None:
    ledger = MomentLedger()
    _lived(ledger)
    moment = ledger.arrived(PARTICULAR, HERS, warm=True, question=False, valence=-0.4, trust=None, tired=0.0)
    assert moment.weight > 0.0 and moment.shape == "warmth when she is low"


def test_the_same_compliment_when_she_is_fine_is_not() -> None:
    ledger = MomentLedger()
    _lived(ledger)
    assert ledger.arrived(PARTICULAR, HERS, warm=True, question=False, valence=0.9, trust=None, tired=0.0).weight == 0.0


def test_a_generic_compliment_when_she_is_low_is_not() -> None:
    ledger = MomentLedger()
    _lived(ledger)
    assert ledger.arrived(GENERIC, HERS, warm=True, question=False, valence=-0.4, trust=None, tired=0.0).weight == 0.0


def test_a_question_from_somebody_she_trusts_weighs_more_than_from_a_stranger() -> None:
    trusted, stranger = MomentLedger(), MomentLedger()
    _lived(trusted)
    _lived(stranger)
    ask = "how is the second moon orbit sketch going?"
    near = trusted.arrived(ask, HERS, warm=False, question=True, valence=0.3, trust=0.9, tired=0.0)
    far = stranger.arrived(ask, HERS, warm=False, question=True, valence=0.3, trust=0.1, tired=0.0)
    assert near.weight > far.weight > 0.0


def test_the_affect_phase_lets_a_moment_bring_what_it_carries(monkeypatch) -> None:
    from core.phases.affect_readings import AffectReadings
    from core.state.aura_state import AuraState

    _lived(moments.get_moment_ledger())

    def run(message: str, valence: float) -> float:
        state = AuraState.default()
        state.cognition.current_origin = "user"
        state.cognition.current_partner = "bryan"
        state.cognition.current_objective = message
        state.cognition.last_response = HERS[0]
        state.affect.valence = valence
        state.affect.emotions["joy"] = 0.2
        AffectReadings(lambda *args, **kwargs: None).a_moment(state, state.affect)
        return float(state.affect.emotions["joy"])

    assert run(PARTICULAR, -0.5) > 0.2, "a particular compliment on a low day brought nothing"
    assert run(GENERIC, -0.5) == pytest.approx(0.2), "a generic compliment moved her as a moment"
