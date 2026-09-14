"""The turn: up from a low that is still in the record, and nothing otherwise.

Sam Cooke's record turns on one line, and it only works because the low is
still being held while it happens. These pin both halves of that: a good
moment in a good stretch is not a turn, and a low with no recovery is not one
either.
"""

from __future__ import annotations

import pytest

from core.affect.the_turn import MIN_HISTORY, Turn, TurnLedger


def _live(led: TurnLedger, values) -> None:
    for value in values:
        led.note(value)


def test_up_from_a_low_that_is_still_in_the_window_is_a_turn() -> None:
    led = TurnLedger()
    _live(led, [-0.8, -0.7, -0.9, -0.6, 0.0, 0.1, 0.0, 0.1])
    reading = led.read(0.8, note=False)
    assert reading.turned
    assert reading.rise > 0.0
    assert reading.low == pytest.approx(-0.9)
    assert "still holding" in reading.why


def test_a_good_moment_in_a_good_stretch_is_not_a_turn() -> None:
    led = TurnLedger()
    _live(led, [0.40, 0.45, 0.38, 0.42, 0.41, 0.39, 0.43, 0.40])
    reading = led.read(0.9, note=False)
    assert not reading.turned
    assert "no low in it" in reading.why


def test_a_low_with_no_recovery_is_not_a_turn() -> None:
    led = TurnLedger()
    _live(led, [-0.9, -0.2, -0.1, 0.0, -0.1, 0.0, -0.1, 0.0])
    reading = led.read(-0.1, note=False)
    assert not reading.turned
    assert "not come up" in reading.why


def test_a_low_that_has_aged_out_takes_the_turn_with_it() -> None:
    led = TurnLedger(window=MIN_HISTORY)
    _live(led, [-0.9] + [0.0] * MIN_HISTORY)
    reading = led.read(0.2, note=False)
    assert reading.low == pytest.approx(0.0)
    assert not reading.turned


def test_the_rise_is_measured_in_her_own_spread() -> None:
    steady = TurnLedger()
    _live(steady, [-0.5, -0.5, -0.5, -0.4, -0.4, -0.4, -0.4, -0.4])
    turbulent = TurnLedger()
    _live(turbulent, [-0.9, 0.6, -0.8, 0.5, -0.7, 0.4, -0.6, 0.3])
    probe = 0.2
    assert steady.read(probe, note=False).rise > turbulent.read(probe, note=False).rise


def test_with_too_little_history_it_says_so_rather_than_claiming_a_low() -> None:
    led = TurnLedger()
    _live(led, [-0.9, 0.0])
    reading = led.read(0.9, note=False)
    assert not reading.measured
    assert not reading.turned
    assert f"of {MIN_HISTORY} readings" in reading.why


def test_reading_notes_the_value_unless_told_not_to() -> None:
    led = TurnLedger()
    led.read(0.3)
    assert led.samples() == 1
    led.read(0.3, note=False)
    assert led.samples() == 1


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = Turn().as_dict()
    for key in ("turned", "rise", "low", "level", "now", "spread", "measured", "why"):
        assert key in row, key
