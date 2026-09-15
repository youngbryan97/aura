"""Loss attributed to timing, measured.

"Romeo and Juliet" puts an ending down to bad timing. These pin the measured
version: whether somebody's long absences have followed strained messages or
only the ordinary time between sittings is learned from their own record, and
fear of change is discounted by the share that is timing.
"""

from __future__ import annotations

import pytest

from core.social.change_around_attachment import change_fear
from core.social.closing_window import MIN_SITTINGS, SittingLedger

PAUSE = 30.0


def _life(ledger: SittingLedger, absences: list[tuple[float, float]], agent: str = "sam") -> float:
    """Sittings of three messages; each absence follows a last message with the given strain."""
    clock = 1_000.0
    for gap, strain in [*absences, (None, 0.1)]:
        for index in range(3):
            ledger.message(agent, clock, strain=strain if index == 2 else 0.1)
            clock += PAUSE
        if gap is not None:
            clock += gap - PAUSE
    return clock - PAUSE


CALM = 0.1
STRAINED = 0.8


def test_absences_that_run_long_after_strain_are_not_timing() -> None:
    ledger = SittingLedger()
    _life(ledger, [(3600.0, CALM), (4000.0, CALM), (3800.0, CALM), (9000.0, STRAINED), (9500.0, STRAINED), (9900.0, STRAINED)])
    reading = ledger.attribution("sam")
    assert reading.measured
    assert reading.relation == pytest.approx(0.5)
    assert reading.timing == pytest.approx(0.5)


def test_absences_that_do_not_follow_strain_are_timing() -> None:
    ledger = SittingLedger()
    _life(ledger, [(9000.0, CALM), (3600.0, STRAINED), (9500.0, CALM), (4000.0, STRAINED), (3800.0, CALM), (9900.0, STRAINED)])
    reading = ledger.attribution("sam")
    assert reading.measured
    assert reading.timing == pytest.approx(1.0)


def test_fear_of_change_is_discounted_by_the_share_that_is_timing() -> None:
    timed = SittingLedger()
    last = _life(timed, [(9000.0, CALM), (3600.0, STRAINED), (9500.0, CALM), (4000.0, STRAINED), (3800.0, CALM), (9900.0, STRAINED)])
    reading = change_fear(timed, "sam", last + 48 * 3600.0)
    assert reading.departure == pytest.approx(1.0)
    assert reading.timing == pytest.approx(1.0)
    assert reading.fear == 0.0

    strained = SittingLedger()
    last = _life(strained, [(3600.0, CALM), (4000.0, CALM), (3800.0, CALM), (9000.0, STRAINED), (9500.0, STRAINED), (9900.0, STRAINED)])
    assert change_fear(strained, "sam", last + 48 * 3600.0).fear == pytest.approx(0.5)


def test_until_both_kinds_of_absence_are_seen_nothing_is_discounted() -> None:
    ledger = SittingLedger()
    last = _life(ledger, [(3600.0, CALM)] * (MIN_SITTINGS + 1))
    assert not ledger.attribution("sam").measured
    assert change_fear(ledger, "sam", last + 48 * 3600.0).fear == pytest.approx(1.0)


def test_messages_without_a_strain_reading_are_still_counted_as_messages() -> None:
    ledger = SittingLedger()
    ledger.message("sam", 1.0)
    ledger.message("sam", 2.0, strain=0.4)
    ledger.message("sam", 3.0, strain="high")  # type: ignore[arg-type]
    assert ledger.reading("sam").messages_so_far == 3
