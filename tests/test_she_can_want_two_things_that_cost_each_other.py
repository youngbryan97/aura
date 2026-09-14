"""Ambivalence, and the phase-or-way question, measured from her own life.

Her budgets already compete and the lowest one wins the intention. That is
arbitration: it produces one answer and discards the other, so nothing
downstream could tell a moment when one drive pressed from a moment when two
pressed against each other. Only the second is a contradiction.

Nothing here names an opposed pair. Opposition is whatever her history shows
has cost each other, so these tests build histories and check that the reading
follows them.
"""

from __future__ import annotations

import pytest

from core.affect.ambivalence import (
    MIN_SAMPLES,
    PHASE,
    UNKNOWN,
    WAY,
    Ambivalence,
    OppositionLedger,
    drive_levels,
    tension,
)


def _trade(ledger: OppositionLedger, ticks: int, *, a: str = "growth", b: str = "social") -> None:
    """A history in which serving one has drained the other."""
    x, y = 0.5, 0.5
    for step in range(ticks):
        move = 0.05 if step % 2 == 0 else -0.05
        x = max(0.0, min(1.0, x + move))
        y = max(0.0, min(1.0, y - move))
        ledger.note({a: x, b: y, "energy": 0.9})


def _together(ledger: OppositionLedger, ticks: int) -> None:
    x = 0.5
    for step in range(ticks):
        x = max(0.0, min(1.0, x + (0.05 if step % 2 == 0 else -0.05)))
        ledger.note({"growth": x, "social": x, "energy": 0.9})


def test_a_pair_that_has_cost_each_other_is_opposed() -> None:
    ledger = OppositionLedger()
    _trade(ledger, 12)
    assert ledger.opposition("growth", "social") > 0.9
    assert ledger.lifelong("growth", "social") < 0.0


def test_a_pair_that_moves_together_is_not_opposed() -> None:
    ledger = OppositionLedger()
    _together(ledger, 12)
    assert ledger.opposition("growth", "social") == 0.0
    reading = tension({"growth": 0.1, "social": 0.1, "energy": 0.9}, ledger)
    assert not reading.held()
    assert "cost each other" in reading.why


def test_a_still_drive_has_no_correlation_rather_than_a_zero_one() -> None:
    """Two series cannot be opposed when one of them never moved."""
    ledger = OppositionLedger()
    for _ in range(12):
        ledger.note({"growth": 0.5, "social": 0.5})
    assert ledger.lifelong("growth", "social") is None
    assert ledger.opposition("growth", "social") == 0.0


def test_opposed_drives_that_are_both_full_are_not_a_contradiction() -> None:
    """A contradiction needs both wants live. Neither pressing is not one."""
    ledger = OppositionLedger()
    _trade(ledger, 12)
    reading = tension({"growth": 1.0, "social": 1.0, "energy": 1.0}, ledger)
    assert not reading.held(), reading
    assert reading.strength == 0.0


def test_both_pressing_and_opposed_is_the_contradiction() -> None:
    ledger = OppositionLedger()
    _trade(ledger, 12)
    reading = tension({"growth": 0.15, "social": 0.2, "energy": 0.95}, ledger)
    assert reading.held(), reading
    assert set(reading.pair) == {"growth", "social"}
    assert reading.strength > 0.5
    assert reading.pressure > 0.5


def test_one_side_pressing_alone_is_weaker_than_both(monkeypatch) -> None:
    """The measure moves with whichever side is weaker, smoothly."""
    ledger = OppositionLedger()
    _trade(ledger, 12)
    both = tension({"growth": 0.1, "social": 0.1, "energy": 0.9}, ledger).strength
    one = tension({"growth": 0.1, "social": 0.9, "energy": 0.9}, ledger).strength
    assert both > one > 0.0


def test_opposed_across_her_whole_life_is_the_way() -> None:
    ledger = OppositionLedger()
    _trade(ledger, 40)
    reading = tension({"growth": 0.1, "social": 0.15, "energy": 0.9}, ledger)
    assert reading.standing == WAY, reading
    assert "across her life" in reading.why


def test_opposed_only_lately_is_a_phase() -> None:
    """A long agreeable history, then a short stretch of trading off."""
    ledger = OppositionLedger()
    _together(ledger, 60)
    _trade(ledger, 6)
    reading = tension({"growth": 0.1, "social": 0.15, "energy": 0.9}, ledger)
    if reading.held():
        assert reading.standing == PHASE, reading
        assert "lately" in reading.why


def test_with_no_history_it_says_so_rather_than_guessing() -> None:
    ledger = OppositionLedger()
    reading = tension({"growth": 0.1, "social": 0.1}, ledger)
    assert reading.standing == UNKNOWN
    assert not reading.held()
    assert reading.why == "no history yet"


def test_recurrence_is_compared_against_chance_not_a_chosen_bar() -> None:
    ledger = OppositionLedger()
    _trade(ledger, 20)
    for _ in range(5):
        tension({"growth": 0.1, "social": 0.1, "energy": 0.9}, ledger)
    reading = tension({"growth": 0.1, "social": 0.1, "energy": 0.9}, ledger)
    assert reading.chance == pytest.approx(1.0 / ledger.pairs_seen())
    assert reading.recurrence > reading.chance, (
        "the pair she keeps being caught in should beat what chance would give it"
    )


def test_the_strongest_conflict_is_named_and_the_others_are_kept() -> None:
    """A single maximum hides a moment where more than two wants are fighting."""
    ledger = OppositionLedger()
    x, y, z = 0.5, 0.5, 0.5
    for step in range(30):
        move = 0.05 if step % 2 == 0 else -0.05
        x = max(0.0, min(1.0, x + move))
        y = max(0.0, min(1.0, y - move))
        z = max(0.0, min(1.0, z - move * 0.5))
        ledger.note({"growth": x, "social": y, "curiosity": z})
    reading = tension({"growth": 0.1, "social": 0.1, "curiosity": 0.3}, ledger)
    assert reading.held()
    assert len(reading.live_pairs) >= 2, reading.live_pairs
    assert reading.live_pairs[0][2] >= reading.live_pairs[1][2]
    assert reading.strength == pytest.approx(reading.live_pairs[0][2])


def test_the_short_window_is_derived_from_how_many_pairs_there_are() -> None:
    ledger = OppositionLedger()
    _trade(ledger, 8)
    # Three drives is three pairs, and a pair needs MIN_SAMPLES to be estimated.
    assert ledger.window() == 3 * MIN_SAMPLES


def test_a_drive_with_no_capacity_is_a_reading_that_cannot_be_taken() -> None:
    levels = drive_levels({
        "growth": {"level": 50.0, "capacity": 100.0},
        "broken": {"level": 5.0, "capacity": 0.0},
        "absent": {},
    })
    assert levels == {"growth": 0.5}


def test_the_reading_serialises_everything_a_reader_needs() -> None:
    ledger = OppositionLedger()
    _trade(ledger, 20)
    row = tension({"growth": 0.1, "social": 0.1, "energy": 0.9}, ledger).as_dict()
    for key in ("strength", "pair", "opposition", "pressure", "standing",
                "recurrence", "chance", "live_pairs", "why"):
        assert key in row, key
    assert isinstance(row["pair"], list) and len(row["pair"]) == 2


def test_an_empty_reading_is_not_held_and_claims_nothing() -> None:
    assert not Ambivalence().held()
    assert Ambivalence().standing == UNKNOWN


def test_early_in_a_life_the_two_windows_are_the_same_data() -> None:
    """So "the way" would be unfalsifiable, and the question stays open."""
    ledger = OppositionLedger()
    _trade(ledger, 6)
    assert ledger.samples("growth", "social") <= ledger.window()
    assert ledger.standing("growth", "social") == UNKNOWN
    reading = tension({"growth": 0.1, "social": 0.1, "energy": 0.9}, ledger)
    assert reading.held()
    assert reading.standing == UNKNOWN
    assert "not yet longer than the window" in reading.why


def test_once_the_history_outruns_the_window_it_can_answer() -> None:
    ledger = OppositionLedger()
    _trade(ledger, 60)
    assert ledger.samples("growth", "social") > ledger.window()
    assert ledger.standing("growth", "social") == WAY
