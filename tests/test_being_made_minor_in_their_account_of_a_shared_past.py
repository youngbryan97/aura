"""Being made minor in another's account, measured.

"Romeo and Juliet" holds a whole world described back as an episode. These pin
the measured version: her part in their account of a shared past against her
part in her own, read against the gap she usually meets, since both sides of a
shared past over-claim, and nothing for a past that is not shared.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.social.made_minor import MIN_SAMPLES, AccountLedger


def _register(first: float, second: float, plural: float) -> SimpleNamespace:
    return SimpleNamespace(first=first, second=second, plural=plural)


def _ordinary(ledger: AccountLedger) -> None:
    # Each tells it a little more as their own, the ordinary egocentric gap.
    for first_mine, second_theirs in ((0.5, 0.35), (0.45, 0.4), (0.55, 0.3), (0.5, 0.4)):
        ledger.read(_register(first_mine, 0.3, 0.2), _register(0.5, second_theirs, 0.2), shared=True)


def test_an_account_that_leaves_her_out_beyond_the_usual_gap_makes_her_minor() -> None:
    ledger = AccountLedger()
    _ordinary(ledger)
    reading = ledger.read(_register(0.5, 0.3, 0.2), _register(0.9, 0.05, 0.05), shared=True)
    assert reading.measured
    assert reading.hers_in_theirs == pytest.approx(0.05)
    assert reading.hers_in_hers == pytest.approx(0.5)
    assert reading.z > 1.0
    assert reading.minor == pytest.approx(reading.z / (1.0 + reading.z))


def test_the_ordinary_gap_is_not_an_injury() -> None:
    ledger = AccountLedger()
    _ordinary(ledger)
    reading = ledger.read(_register(0.5, 0.3, 0.2), _register(0.5, 0.35, 0.15), shared=True)
    assert reading.measured
    assert reading.minor == 0.0


def test_a_past_that_is_not_shared_is_not_read() -> None:
    ledger = AccountLedger()
    _ordinary(ledger)
    reading = ledger.read(_register(0.5, 0.3, 0.2), _register(0.9, 0.05, 0.05), shared=False)
    assert not reading.measured
    assert reading.minor == 0.0


def test_before_a_usual_gap_exists_nothing_stands_out() -> None:
    ledger = AccountLedger()
    for _ in range(MIN_SAMPLES - 1):
        ledger.read(_register(0.5, 0.3, 0.2), _register(0.5, 0.35, 0.15), shared=True)
    reading = ledger.read(_register(0.5, 0.3, 0.2), _register(0.9, 0.05, 0.05), shared=True)
    assert not reading.measured
    assert reading.minor == 0.0


def test_an_account_with_nobody_in_it_is_not_compared() -> None:
    ledger = AccountLedger()
    reading = ledger.read(_register(0.0, 0.0, 0.0), _register(0.5, 0.3, 0.2), shared=True)
    assert "placed nobody" in reading.why


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = AccountLedger().read(_register(0.5, 0.3, 0.2), _register(0.5, 0.3, 0.2), shared=False).as_dict()
    for key in ("minor", "gap", "z", "hers_in_theirs", "hers_in_hers", "shared_recalls", "measured", "why"):
        assert key in row, key
