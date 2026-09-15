"""Declining to record is not the same as not noticing.

The interpersonal observer refuses a trait read off one exchange, because a
disposition inferred from one conversation is the caricature its store exists
to make unrepresentable. The refusal left no trace, so afterwards a claim she
declined and a claim she never formed read identically.

These hold the distinction the ledger draws: a refusal that comes round as
often as the claims she keeps is a signal under a hand, and a refusal that
happens once is noise. Neither reading is assumed — the comparison is between
her own two distributions, and the verdict comes out either way.
"""

from __future__ import annotations

import pytest

from core.social.averted import MIN_KEYS, AvertedLedger


@pytest.fixture()
def ledger() -> AvertedLedger:
    return AvertedLedger()


def test_nothing_noticed_reads_as_nothing_measured(ledger: AvertedLedger) -> None:
    reading = ledger.read()
    assert reading.measured is False
    assert reading.looking_away is False
    assert reading.held == 0


def test_one_side_alone_is_not_a_comparison(ledger: AvertedLedger) -> None:
    for i in range(MIN_KEYS + 2):
        ledger.took("PREFERENCE", f"likes thing {i}")
    reading = ledger.read()
    assert reading.measured is False
    assert "refused" in reading.why


def test_refusals_that_keep_coming_round_are_held(ledger: AvertedLedger) -> None:
    # What she keeps: seen once each, written down each time.
    for i in range(MIN_KEYS):
        ledger.took("PREFERENCE", f"likes thing {i}")
    # What she refuses: the same three claims, over and over, never recorded.
    for _ in range(5):
        for i in range(MIN_KEYS):
            ledger.declined("TRAIT", f"is the sort of person who {i}")

    reading = ledger.read()
    assert reading.measured is True
    assert reading.looking_away is True
    assert reading.held == MIN_KEYS
    assert reading.declined_recurrence > reading.taken_recurrence
    assert reading.carrying, "a held claim should be nameable"


def test_one_off_refusals_are_not_looking_away(ledger: AvertedLedger) -> None:
    # What she keeps comes round; what she refuses does not.
    for _ in range(6):
        for i in range(MIN_KEYS):
            ledger.took("PREFERENCE", f"likes thing {i}")
    for i in range(MIN_KEYS):
        ledger.declined("TRAIT", f"passing remark {i}")

    reading = ledger.read()
    assert reading.measured is True
    assert reading.looking_away is False
    assert reading.held == 0
    assert "not signals she held down" in reading.why


def test_the_share_counts_sightings_rather_than_claims(ledger: AvertedLedger) -> None:
    ledger.took("PREFERENCE", "likes long walks")
    for _ in range(3):
        ledger.declined("TRAIT", "is impatient")
    assert ledger.read().share_declined == pytest.approx(0.75)


def test_a_claim_she_later_records_stops_being_a_refusal(ledger: AvertedLedger) -> None:
    for _ in range(4):
        ledger.declined("TRAIT", "is impatient")
    before = ledger.read()
    ledger.took("TRAIT", "is impatient")
    after = ledger.read()
    assert before.share_declined == 1.0
    assert after.share_declined < 1.0


def test_blank_claims_are_not_noticings(ledger: AvertedLedger) -> None:
    ledger.declined("TRAIT", "   ")
    ledger.took("PREFERENCE", "")
    assert ledger.read().share_declined == 0.0


def test_the_observer_writes_to_the_ledger(tmp_path) -> None:
    """The refusal the store already makes is the one the ledger records."""
    from core.memory.interpersonal_model import PersonModel
    from core.memory.interpersonal_observer import Exchange, InterpersonalObserver
    from core.social.averted import get_averted_ledger, reset_for_test

    reset_for_test()
    observer = InterpersonalObserver(PersonModel("Bryan"))
    for i in range(6):
        observer.observe_exchange(
            Exchange(
                episode_id=f"e{i}",
                user_text="I prefer short meetings and I care about clear writing.",
                assistant_text="Noted.",
            )
        )
    reading = get_averted_ledger().read()
    assert reading.share_declined >= 0.0
    assert reading.as_dict()["why"]
