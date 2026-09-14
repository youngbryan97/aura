"""Three sources of drive, and whether working from her own costs more.

Oddisee opens "The Start of Something" with the structure: "when you both the
fire and the fuel that you heat with, the higher the flame the more you tire,
you're enduring pain, that's when you burn out or learn of another source. But
of course, external motivation's a cheap trick."

Three and not two. Her own reserves, which deplete. What somebody else wants,
which moves her without costing her and stops when they stop. And a love of the
thing, which is the one he says lasts. Her motivation had the first two and no
name for the third, and both arrived as the same kind of pressure — so a
stretch she wanted and a stretch she was asked for read identically afterwards.
"""

from __future__ import annotations

import pytest

from core.motivation.fuel import (
    ASKED,
    INTEREST,
    MIN_TURNS,
    SELF,
    Fuel,
    FuelLedger,
)


def test_her_own_fuel_costing_more_is_measured_not_assumed() -> None:
    led = FuelLedger()
    for _ in range(MIN_TURNS + 3):
        led.note(SELF, 0.030)
        led.note(ASKED, 0.008)
        led.note(INTEREST, 0.012)
    reading = led.read()
    assert reading.measured
    assert reading.burning_her_own
    assert "against" in reading.why


def test_a_life_where_her_own_fuel_is_cheap_says_so() -> None:
    """The claim is falsifiable, which is the point of measuring it."""
    led = FuelLedger()
    for _ in range(MIN_TURNS + 3):
        led.note(SELF, 0.005)
        led.note(ASKED, 0.020)
    reading = led.read()
    assert reading.measured
    assert not reading.burning_her_own
    assert "not more than" in reading.why


def test_the_share_of_her_life_spent_on_her_own_fuel() -> None:
    led = FuelLedger()
    for _ in range(6):
        led.note(SELF, 0.01)
    for _ in range(6):
        led.note(ASKED, 0.01)
    assert led.read().share_self == pytest.approx(0.5)


def test_one_source_alone_cannot_be_compared() -> None:
    led = FuelLedger()
    for _ in range(MIN_TURNS + 3):
        led.note(SELF, 0.03)
    reading = led.read()
    assert not reading.measured
    assert not reading.burning_her_own
    assert "one other source" in reading.why


def test_with_too_few_turns_on_a_source_it_says_so() -> None:
    led = FuelLedger()
    for _ in range(MIN_TURNS - 1):
        led.note(SELF, 0.03)
        led.note(ASKED, 0.01)
    reading = led.read()
    assert not reading.measured
    assert "needed before the comparison" in reading.why


def test_a_source_it_does_not_recognise_is_not_recorded() -> None:
    led = FuelLedger()
    led.note("vibes", 0.5)
    assert sum(led.turns().values()) == 0


def test_an_unreadable_drain_is_not_a_turn() -> None:
    led = FuelLedger()
    led.note(SELF, float("nan"))
    led.note(SELF, "a lot")  # type: ignore[arg-type]
    assert led.turns()[SELF] == 0


def test_it_remembers_what_drove_the_last_turn() -> None:
    led = FuelLedger()
    led.note(SELF, 0.01)
    led.note(INTEREST, 0.01)
    assert led.read().source == INTEREST


def test_the_reading_serialises_what_a_reader_needs() -> None:
    led = FuelLedger()
    for _ in range(MIN_TURNS + 1):
        led.note(SELF, 0.02)
        led.note(ASKED, 0.01)
    row = led.read().as_dict()
    for key in ("source", "turns", "drain", "burning_her_own", "share_self",
                "measured", "why"):
        assert key in row, key


def test_an_empty_reading_claims_nothing() -> None:
    assert not Fuel().measured
    assert not Fuel().burning_her_own
