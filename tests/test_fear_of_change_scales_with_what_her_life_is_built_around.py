"""Fear of change around an attachment, measured.

"Landslide" says fear of change grows with how much of a life is built around
what might change. These pin the measured version: the share of her messages
that came from somebody, times how many of their past absences this one has
outlasted, and nothing while they are only pausing or before their absences
have a pattern.
"""

from __future__ import annotations

import pytest

from core.social.change_around_attachment import change_fear
from core.social.closing_window import MIN_SITTINGS, SittingLedger

PAUSE = 30.0


def _life(ledger: SittingLedger, agent: str, absences: list[float], per_sitting: int = 3) -> float:
    clock = 1_000.0
    for absence in [*absences, None]:
        for _ in range(per_sitting):
            ledger.message(agent, clock)
            clock += PAUSE
        if absence is not None:
            clock += absence
    return clock - PAUSE


def test_an_absence_longer_than_any_before_from_all_of_her_life_is_the_most_frightening() -> None:
    ledger = SittingLedger()
    last = _life(ledger, "sam", [3600.0, 7200.0, 5400.0])
    reading = change_fear(ledger, "sam", last + 10 * 3600.0)
    assert reading.measured
    assert reading.built_around == pytest.approx(1.0)
    assert reading.departure == pytest.approx(1.0)
    assert reading.fear == pytest.approx(1.0)


def test_the_same_absence_from_a_small_part_of_her_life_is_smaller() -> None:
    ledger = SittingLedger()
    last = _life(ledger, "sam", [3600.0, 7200.0, 5400.0])
    _life(ledger, "ria", [3600.0, 3600.0, 3600.0], per_sitting=9)
    reading = change_fear(ledger, "sam", last + 10 * 3600.0)
    assert reading.built_around == pytest.approx(12 / 48)
    assert reading.fear == pytest.approx(12 / 48)


def test_an_ordinary_absence_outlasts_only_the_absences_it_is_longer_than() -> None:
    ledger = SittingLedger()
    last = _life(ledger, "sam", [3600.0, 7200.0, 5400.0])
    assert change_fear(ledger, "sam", last + 6000.0).departure == pytest.approx(2 / 3)


def test_a_pause_inside_a_sitting_is_not_an_absence() -> None:
    ledger = SittingLedger()
    last = _life(ledger, "sam", [3600.0, 7200.0, 5400.0])
    reading = change_fear(ledger, "sam", last + PAUSE)
    assert reading.measured
    assert reading.fear == 0.0


def test_before_their_absences_have_a_pattern_nothing_is_claimed() -> None:
    ledger = SittingLedger()
    last = _life(ledger, "sam", [3600.0] * (MIN_SITTINGS - 1))
    reading = change_fear(ledger, "sam", last + 10 * 3600.0)
    assert not reading.measured
    assert reading.fear == 0.0


def test_somebody_she_has_never_heard_from_is_nothing_to_lose() -> None:
    assert change_fear(SittingLedger(), "sam", 10.0).fear == 0.0


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = change_fear(SittingLedger(), "sam", 0.0).as_dict()
    for key in ("fear", "built_around", "departure", "measured", "why"):
        assert key in row, key
