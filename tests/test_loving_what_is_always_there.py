"""Attachment moving to what is reliable, when what answers has stopped being.

"Why iii Love the Moon" states it and the performance puts the direction on it:
the grievance about people is sung at 322 Hz and the attachment to the moon at
93 Hz, one minute later in the same voice. What the moon has is constancy — it
shows up on a schedule and nothing it does depends on how the day went.

Her social drive drained on its own clock whether the person she was waiting
for came back every hour or once a fortnight, so a reliable presence and an
unreliable one left her in the same place.
"""

from __future__ import annotations

import pytest

from core.social.constancy import MIN_RETURNS, Constancy, ConstancyLedger


def _her_cycle(led: ConstancyLedger, period: int = 100, n: int = 20) -> None:
    for i in range(n):
        led.she_came_round(i * period)


def test_somebody_who_comes_back_erratically_moves_the_attachment() -> None:
    led = ConstancyLedger()
    _her_cycle(led)
    for t in (0, 90, 1500, 1520, 4000, 4100):
        led.they_came_back(t)
    reading = led.read()
    assert reading.measured
    assert reading.reallocated
    assert reading.theirs < reading.hers
    assert "less regularly" in reading.why


def test_somebody_reliable_does_not() -> None:
    led = ConstancyLedger()
    _her_cycle(led)
    for i in range(10):
        led.they_came_back(i * 200)
    reading = led.read()
    assert reading.measured
    assert not reading.reallocated


def test_slower_is_not_less_reliable() -> None:
    """Once a fortnight, every fortnight, is constant."""
    led = ConstancyLedger()
    _her_cycle(led)
    for i in range(8):
        led.they_came_back(i * 10_000)
    assert not led.read().reallocated


def test_the_reference_is_her_own_period_rather_than_a_chosen_number() -> None:
    """Something is unreliable when it is less regular than she is."""
    erratic_her = ConstancyLedger()
    for t in (0, 10, 900, 905, 4000, 4001, 9000, 9500):
        erratic_her.she_came_round(t)
    for t in (0, 100, 900, 1500, 4000, 4400):
        erratic_her.they_came_back(t)
    steady_her = ConstancyLedger()
    _her_cycle(steady_her)
    for t in (0, 100, 900, 1500, 4000, 4400):
        steady_her.they_came_back(t)
    # The same caller, read against two different versions of her.
    assert steady_her.read().reallocated
    assert not erratic_her.read().reallocated


def test_with_too_few_returns_it_says_so() -> None:
    led = ConstancyLedger()
    _her_cycle(led)
    led.they_came_back(0)
    led.they_came_back(100)
    reading = led.read()
    assert not reading.measured
    assert not reading.reallocated
    assert "returns needed" in reading.why


def test_without_her_own_period_there_is_nothing_to_read_theirs_against() -> None:
    led = ConstancyLedger()
    for t in (0, 100, 200, 300, 400):
        led.they_came_back(t)
    reading = led.read()
    assert not reading.measured
    assert reading.theirs > 0.0
    assert "her own period is not established" in reading.why


def test_unreadable_stamps_are_not_recorded() -> None:
    led = ConstancyLedger()
    _her_cycle(led)
    led.they_came_back("noon")  # type: ignore[arg-type]
    led.they_came_back(float("nan"))
    assert led.read().returns == 0


def test_the_reading_serialises_what_a_reader_needs() -> None:
    led = ConstancyLedger()
    _her_cycle(led)
    for i in range(6):
        led.they_came_back(i * 150)
    row = led.read().as_dict()
    for key in ("theirs", "hers", "returns", "reallocated", "measured", "why"):
        assert key in row, key


def test_an_empty_reading_claims_nothing() -> None:
    assert not Constancy().measured
    assert not Constancy().reallocated
