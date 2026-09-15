"""A pull the scores do not explain.

"Georgia On My Mind" spends its middle acknowledging the alternatives and does
not argue with them — "other arms reach out to me, other eyes smile tenderly" —
and then says the road leads back anyway. The chooser takes the best score and
says why; nothing asked whether she comes back to what lost.

These hold what the ledger can and cannot conclude. Every comparison is against
her own history: her own rate of returning to anything, the spread of her own
per-option rates, and the usual margin her winners win by. A chooser that comes
back to everything equally reports no pull, which is the honest reading of it.
"""

from __future__ import annotations

import pytest

from core.motivation.returning import MIN_OPTIONS, MIN_PASSES, ReturningLedger


@pytest.fixture()
def ledger() -> ReturningLedger:
    return ReturningLedger()


def test_nothing_passed_over_reads_as_nothing(ledger: ReturningLedger) -> None:
    reading = ledger.read()
    assert reading.measured is False
    assert reading.survives_alternatives is False


def test_too_few_options_is_not_a_spread(ledger: ReturningLedger) -> None:
    for _ in range(MIN_PASSES + 1):
        ledger.note_choice(chosen_id="a", scores={"a": 0.9, "b": 0.2})
    reading = ledger.read()
    assert reading.measured is False
    assert str(MIN_OPTIONS) in reading.why


def test_something_she_keeps_coming_back_to_is_named(ledger: ReturningLedger) -> None:
    # Four options. "georgia" loses to a better score every time and is chosen
    # again afterwards; the other three lose and stay lost.
    for _ in range(4):
        ledger.note_choice(
            chosen_id="work",
            scores={"work": 0.90, "georgia": 0.30, "c": 0.31, "d": 0.32},
        )
        ledger.note_choice(
            chosen_id="georgia",
            scores={"georgia": 0.30, "c": 0.29, "d": 0.28},
        )

    reading = ledger.read()
    assert reading.measured is True
    assert reading.draws_her_back == "georgia"
    assert reading.survives_alternatives is True
    assert reading.pull > 1.0
    assert reading.conceded > 0.0


def test_coming_back_to_everything_equally_is_no_pull(ledger: ReturningLedger) -> None:
    options = ["a", "b", "c", "d"]
    for round_index in range(MIN_PASSES + 2):
        for winner in options:
            ledger.note_choice(
                chosen_id=winner, scores={name: 0.5 for name in options}
            )
    reading = ledger.read()
    assert reading.measured is True
    assert reading.survives_alternatives is False
    assert "same rate" in reading.why or reading.pull <= 1.0


def test_a_chooser_that_never_returns_reports_no_pull(ledger: ReturningLedger) -> None:
    for _ in range(MIN_PASSES + 2):
        ledger.note_choice(
            chosen_id="work", scores={"work": 0.9, "a": 0.1, "b": 0.2, "c": 0.3}
        )
    reading = ledger.read()
    assert reading.base_rate == 0.0
    assert reading.survives_alternatives is False


def test_losing_by_a_hair_is_not_surviving_an_alternative(ledger: ReturningLedger) -> None:
    """A near tie she comes back to is a coin landing, not a pull."""
    for _ in range(4):
        # Everything loses by a hair; one of them is returned to.
        ledger.note_choice(
            chosen_id="work",
            scores={"work": 0.50, "near": 0.499, "c": 0.498, "d": 0.497},
        )
        ledger.note_choice(chosen_id="near", scores={"near": 0.50, "c": 0.9, "d": 0.9})
    reading = ledger.read()
    assert reading.measured is True
    assert reading.conceded < 0.01


def test_a_bad_score_does_not_enter_the_book(ledger: ReturningLedger) -> None:
    ledger.note_choice(chosen_id="a", scores={"a": float("nan"), "b": 0.2})
    ledger.note_choice(chosen_id="", scores={"a": 0.5})
    assert ledger.read().measured is False


def test_the_chooser_writes_to_the_ledger(tmp_path, monkeypatch) -> None:
    """The receipt already named every option; now something reads it."""
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path))
    from core.agency.subjective_choice import ChoiceOption, SubjectiveChoiceEngine
    from core.motivation.returning import get_returning_ledger, reset_for_test

    reset_for_test()
    engine = SubjectiveChoiceEngine(
        tmp_path / "choices.json", mirror_identity=False
    )
    options = [
        ChoiceOption(id="a", label="read the paper"),
        ChoiceOption(id="b", label="write the note"),
    ]
    engine.choose(options, context="test")
    read = get_returning_ledger().read()
    assert read.as_dict()["why"]
