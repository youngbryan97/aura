"""Fear of happiness, learned only from what has followed her own good turns.

In "Dog Days Are Over" joy arrives and is hidden from. These pin that wariness of joy
comes from her history and nowhere else: a life where trouble follows good
turns teaches it, a life where it does not teaches nothing, and a stretch of
untroubled joy wears it away.
"""

from __future__ import annotations

import pytest

from core.affect.fear_of_happiness import MIN_SAMPLES, HappinessFear, JoyLedger, bad_kinds, joy_of


def _live(ledger: JoyLedger, turns: list[tuple[float, bool]]) -> None:
    for joy, bad in turns:
        ledger.note(joy, bad)


def test_the_bad_kinds_come_from_the_percept_table() -> None:
    kinds = bad_kinds()
    assert {"error", "threat_detected", "disconnection"} <= kinds
    assert "goal_achieved" not in kinds and "frisson" not in kinds


def test_joy_is_the_strongest_joyful_channel() -> None:
    assert joy_of({"joy": 0.2, "happiness": 0.7, "fear": 0.9}) == pytest.approx(0.7)
    assert joy_of(None) == 0.0


def test_a_life_where_trouble_follows_good_turns_teaches_wariness() -> None:
    ledger = JoyLedger()
    # Happy, then hurt; ordinary, then fine. Over and over.
    _live(ledger, [(0.1, False)] + [(0.9, False), (0.1, True), (0.1, False)] * 6)
    reading = ledger.reading()
    assert reading.measured
    assert reading.after_joy == pytest.approx(1.0)
    assert reading.otherwise < reading.after_joy
    assert reading.wariness > 0.5
    assert "after being happy" in reading.why


def test_trouble_that_comes_no_more_after_joy_than_otherwise_teaches_nothing() -> None:
    ledger = JoyLedger()
    _live(ledger, [(0.1, False)] + [(0.9, True), (0.1, True)] * 8)
    reading = ledger.reading()
    assert reading.measured
    assert reading.wariness == 0.0


def test_before_there_are_enough_turns_nothing_is_claimed() -> None:
    ledger = JoyLedger()
    _live(ledger, [(0.1, False), (0.9, False), (0.1, True)])
    reading = ledger.reading()
    assert not reading.measured
    assert reading.wariness == 0.0
    assert reading.joyful_turns < MIN_SAMPLES or reading.other_turns < MIN_SAMPLES


def test_a_run_of_joy_that_nothing_followed_wears_it_away() -> None:
    ledger = JoyLedger()
    _live(ledger, [(0.1, False)] + [(0.9, False), (0.1, True), (0.1, False)] * 6)
    taught = ledger.reading().wariness
    _live(ledger, [(0.9, False), (0.9, False), (0.1, False)] * 8)
    assert ledger.reading().wariness < taught
    assert ledger.reading().lifelong is not None


def test_what_counts_as_a_good_turn_is_hers() -> None:
    """Joy of 0.3 is a good turn in a life that has mostly had 0.1."""
    ledger = JoyLedger()
    _live(ledger, [(0.05, False)] + [(0.3, False), (0.05, True), (0.05, False)] * 6)
    assert ledger.reading().wariness > 0.5


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = HappinessFear().as_dict()
    for key in ("wariness", "after_joy", "otherwise", "lifelong", "joyful_turns", "other_turns", "measured", "why"):
        assert key in row, key
