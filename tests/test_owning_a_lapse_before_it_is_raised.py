"""Protective self-indictment, as stealing thunder learned from her own history.

"Runaway" gets in ahead of blame. These pin the measured version: whether
owning a lapse before somebody raises it has gone better for her is read off
how they took her replies after each, it stays unmeasured until both kinds
have happened enough, and only a positive lift gives what she owes more of her
attention.
"""

from __future__ import annotations

import pytest

from core.social.owning_it_first import MIN_SAMPLES, OwningFirst, OwningLedger, lifted


def _owned_then(ledger: OwningLedger, agent: str, start: float, after: float) -> None:
    ledger.delivered(agent, owed_in_mind=True, frustration=start)
    ledger.heard(agent, frustration=after, complaint=False)


def _raised_then(ledger: OwningLedger, agent: str, start: float, after: float) -> None:
    ledger.delivered(agent, owed_in_mind=False, frustration=0.2)
    ledger.heard(agent, frustration=start, complaint=True)
    ledger.delivered(agent, owed_in_mind=False, frustration=start)
    ledger.heard(agent, frustration=after, complaint=False)


def test_a_history_where_owning_it_first_went_better_gives_a_positive_lift() -> None:
    ledger = OwningLedger()
    for _ in range(MIN_SAMPLES):
        _owned_then(ledger, "sam", 0.4, 0.3)
        _raised_then(ledger, "sam", 0.6, 0.7)
    reading = ledger.reading()
    assert reading.measured
    assert reading.after_owned == pytest.approx(-0.1)
    assert reading.after_raised == pytest.approx(0.1)
    assert reading.lift == pytest.approx(0.2)
    assert "after she owned a lapse first" in reading.why


def test_when_it_made_no_difference_there_is_no_lift() -> None:
    ledger = OwningLedger()
    for _ in range(MIN_SAMPLES):
        _owned_then(ledger, "sam", 0.5, 0.4)
        _raised_then(ledger, "sam", 0.5, 0.4)
    assert ledger.reading().lift == pytest.approx(0.0)


def test_until_both_kinds_have_happened_enough_nothing_is_claimed() -> None:
    ledger = OwningLedger()
    for _ in range(MIN_SAMPLES):
        _owned_then(ledger, "sam", 0.4, 0.1)
    _raised_then(ledger, "sam", 0.6, 0.9)
    reading = ledger.reading()
    assert not reading.measured
    assert reading.lift == 0.0
    assert reading.owned_events == MIN_SAMPLES and reading.raised_events == 1


def test_a_complaint_about_a_reply_she_had_already_owned_is_not_raised_by_them() -> None:
    ledger = OwningLedger()
    ledger.delivered("sam", owed_in_mind=True, frustration=0.4)
    ledger.heard("sam", frustration=0.5, complaint=True)
    reading = ledger.reading()
    assert reading.owned_events == 1
    assert reading.raised_events == 0


def test_an_event_is_closed_by_the_same_person_only() -> None:
    ledger = OwningLedger()
    ledger.delivered("sam", owed_in_mind=True, frustration=0.4)
    ledger.heard("ria", frustration=0.9, complaint=False)
    assert ledger.reading().owned_events == 0
    ledger.heard("sam", frustration=0.3, complaint=False)
    assert ledger.reading().owned_events == 1


def test_an_unreadable_frustration_opens_and_closes_nothing() -> None:
    ledger = OwningLedger()
    ledger.delivered("sam", owed_in_mind=True, frustration=float("nan"))
    ledger.heard("sam", frustration=0.3, complaint=False)
    assert ledger.reading().owned_events == 0


def test_only_a_measured_positive_lift_raises_the_share_and_never_past_one() -> None:
    measured = OwningFirst(lift=0.5, measured=True)
    assert lifted(0.6, measured) == pytest.approx(0.8)
    assert lifted(0.6, OwningFirst(lift=0.5, measured=False)) == pytest.approx(0.6)
    assert lifted(0.6, OwningFirst(lift=-0.3, measured=True)) == pytest.approx(0.6)
    assert lifted(0.9, OwningFirst(lift=5.0, measured=True)) == pytest.approx(1.0)


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = OwningFirst().as_dict()
    for key in ("lift", "after_owned", "after_raised", "owned_events", "raised_events", "measured", "why"):
        assert key in row, key
