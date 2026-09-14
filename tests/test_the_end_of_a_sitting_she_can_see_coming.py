"""Urgency from a closing window, read off how her sittings with somebody have ended.

"Sweet Disposition" draws its intensity from time running out. These pin the
measured version: sittings are split at the long gaps their own timing sets,
the chance this one ends now is the share of past sittings that ended at this
length among those that reached it, and nothing is claimed early.
"""

from __future__ import annotations

import pytest

from core.social.closing_window import MIN_SITTINGS, SittingLedger, otsu_split

PAUSE = 20.0
ABSENCE = 6 * 3600.0


def _sittings(ledger: SittingLedger, lengths: list[int], *, then: int = 0, agent: str = "sam") -> None:
    clock = 1_000.0
    for length in lengths:
        for _ in range(length):
            ledger.message(agent, clock)
            clock += PAUSE
        clock += ABSENCE
    for _ in range(then):
        ledger.message(agent, clock)
        clock += PAUSE


def test_the_split_falls_between_pauses_and_absences() -> None:
    cut = otsu_split([1.0, 1.2, 0.9, 1.1, 9.0, 9.5, 8.8])
    assert cut is not None and 1.2 < cut < 8.8
    assert otsu_split([3.0, 3.0, 3.0]) is None


def test_sittings_that_always_run_three_messages_close_at_the_third() -> None:
    ledger = SittingLedger()
    _sittings(ledger, [3, 3, 3], then=3)
    reading = ledger.reading("sam")
    assert reading.measured
    assert reading.messages_so_far == 3
    assert reading.closing == pytest.approx(1.0)
    ledger2 = SittingLedger()
    _sittings(ledger2, [3, 3, 3], then=2)
    assert ledger2.reading("sam").closing == pytest.approx(0.0)


def test_the_hazard_is_the_share_that_ended_among_those_that_got_this_far() -> None:
    ledger = SittingLedger()
    _sittings(ledger, [2, 4, 4, 2], then=2)
    assert ledger.reading("sam").closing == pytest.approx(0.5)
    ledger = SittingLedger()
    _sittings(ledger, [2, 4, 4, 2], then=4)
    assert ledger.reading("sam").closing == pytest.approx(1.0)


def test_a_sitting_longer_than_any_before_reads_as_closing() -> None:
    ledger = SittingLedger()
    _sittings(ledger, [2, 3, 2], then=6)
    reading = ledger.reading("sam")
    assert reading.closing == pytest.approx(1.0)
    assert "longer than any" in reading.why


def test_until_enough_sittings_have_ended_nothing_is_claimed() -> None:
    ledger = SittingLedger()
    _sittings(ledger, [3] * (MIN_SITTINGS - 1), then=3)
    reading = ledger.reading("sam")
    assert not reading.measured
    assert reading.closing == 0.0


def test_each_person_has_their_own_sittings() -> None:
    ledger = SittingLedger()
    _sittings(ledger, [3, 3, 3], then=3, agent="sam")
    assert not ledger.reading("ria").measured


def test_a_message_out_of_order_or_unreadable_is_not_counted() -> None:
    ledger = SittingLedger()
    ledger.message("sam", 100.0)
    ledger.message("sam", 50.0)
    ledger.message("sam", float("nan"))
    ledger.message("sam", "soon")  # type: ignore[arg-type]
    assert ledger.reading("sam").messages_so_far == 1


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = SittingLedger().reading("sam").as_dict()
    for key in ("closing", "messages_so_far", "sittings_ended", "long_gap_s", "measured", "why"):
        assert key in row, key
