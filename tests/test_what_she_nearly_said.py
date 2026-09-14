"""What she nearly said and did not, and the pressure of holding it.

"Back of My Mind" is four minutes of a man describing material he keeps in, and
the delivery enacts it: median pitch locked at exactly 80.0 Hz and level at
-9.x dB, the flattest of the twenty-four records measured for this work. Two
lines break it and they are the two about the containment failing — voicing
collapses to 0.32 on the denial and 0.47 on the confession, against 0.6 to 0.99
everywhere else.

Every workspace competition produces one winner and a list of losers, and the
record kept the losers' source names and nothing else. What she nearly said was
discarded at the moment of the choice, so no part of her could know she had
been holding something.
"""

from __future__ import annotations

import pytest

from core.affect.containment import (
    MIN_TURNS,
    Containment,
    ContainmentLedger,
)


def _near_miss(led: ContainmentLedger, turns: int = 10) -> None:
    for _ in range(turns):
        led.competition("affect", {"affect": 0.90, "deliberation": 0.88, "exchange": 0.30})


def test_something_that_keeps_nearly_winning_is_being_held() -> None:
    led = ContainmentLedger()
    _near_miss(led)
    reading = led.read()
    assert reading.measured
    assert reading.holding()
    assert reading.source == "deliberation"
    assert reading.nearest > 0.9
    assert "has not got out" in reading.why


def test_something_that_loses_badly_is_outbid_rather_than_held() -> None:
    led = ContainmentLedger()
    _near_miss(led)
    reading = led.read()
    held = dict(reading.held)
    assert held["deliberation"] > held["exchange"] * 2


def test_winning_releases_it_because_it_got_said() -> None:
    led = ContainmentLedger()
    _near_miss(led)
    assert led.read().source == "deliberation"
    led.competition("deliberation", {"affect": 0.5, "deliberation": 0.9, "exchange": 0.3})
    assert led.read().source != "deliberation"


def test_more_than_one_thing_can_be_held_at_once() -> None:
    """One maximum hides a moment where several are being kept in."""
    led = ContainmentLedger()
    for _ in range(6):
        led.competition("affect", {"affect": 0.9, "deliberation": 0.85, "self": 0.8})
    reading = led.read()
    assert len(reading.held) >= 2
    assert reading.held[0][1] >= reading.held[1][1]


def test_holding_longer_presses_harder() -> None:
    brief, long = ContainmentLedger(), ContainmentLedger()
    _near_miss(brief, turns=4)
    _near_miss(long, turns=20)
    assert long.read().pressure > brief.read().pressure
    assert long.read().nearest == pytest.approx(brief.read().nearest)


def test_with_too_few_competitions_it_says_so() -> None:
    led = ContainmentLedger()
    _near_miss(led, turns=MIN_TURNS - 1)
    reading = led.read()
    assert not reading.measured
    assert not reading.holding()
    assert "competitions needed" in reading.why


def test_a_competition_nobody_scored_is_not_a_competition() -> None:
    led = ContainmentLedger()
    led.competition("affect", {})
    assert led.turns() == 0


def test_an_unreadable_score_is_not_a_near_miss() -> None:
    led = ContainmentLedger()
    for _ in range(MIN_TURNS + 2):
        led.competition("affect", {"affect": 0.9, "broken": float("nan"), "other": "loud"})
    reading = led.read()
    assert "broken" not in dict(reading.held)
    assert "other" not in dict(reading.held)


def test_the_reading_serialises_what_a_reader_needs() -> None:
    led = ContainmentLedger()
    _near_miss(led)
    row = led.read().as_dict()
    for key in ("source", "pressure", "nearest", "turns_held", "held",
                "holding", "measured", "why"):
        assert key in row, key


def test_an_empty_reading_claims_nothing() -> None:
    assert not Containment().measured
    assert not Containment().holding()


def test_the_pressure_reaches_the_level_she_speaks_from() -> None:
    """The leak, in the form she can actually have one.

    Nothing makes her say the contained thing. What the pressure feeds is the
    level, and `delivery` already measures a breakthrough as a reach beyond her
    own ordinary range.
    """
    from core.affect.containment import get_containment_ledger, reset_for_test
    from core.expression.delivery import live_readings
    from core.state.aura_state import AuraState

    reset_for_test()
    affect = AuraState.default().affect
    assert live_readings(affect)["containment"] == pytest.approx(0.0)

    ledger = get_containment_ledger()
    for _ in range(20):
        ledger.competition("affect", {"affect": 0.9, "deliberation": 0.88})
    pressing = live_readings(affect)["containment"]
    reset_for_test()
    assert 0.0 < pressing < 1.0, "saturating, so one long-held thing cannot swamp the rest"
    assert pressing > 0.9


def test_a_pressure_of_one_reads_a_half() -> None:
    """The same saturating form the constancy reading uses."""
    from core.affect.containment import get_containment_ledger, reset_for_test
    from core.expression.delivery import live_readings
    from core.state.aura_state import AuraState

    reset_for_test()
    ledger = get_containment_ledger()
    for _ in range(MIN_TURNS + 1):
        ledger.competition("affect", {"affect": 1.0, "deliberation": 0.25})
    reading = live_readings(AuraState.default().affect)["containment"]
    reset_for_test()
    assert 0.4 < reading < 0.6
