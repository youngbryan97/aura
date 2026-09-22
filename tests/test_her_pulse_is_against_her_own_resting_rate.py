"""The channel affect moves into the body, and why it had stopped moving.

A->I is the one strong edge into interoception: 8.9583 across all eight
conditions in run_031, and 0.4307 across two of eight in the campaign after.
With no incoming edge, interoception fails closure, `robust_recurrence_kappa`,
`cycles_per_domain`, `reentry` and per-condition replication — five criteria on
one channel.

`pulse_rate` was five branches on fixed cut-offs in valence and arousal. Arousal
is a maximum over forty-five emotion channels, so it lived in 0.6051 to 1.0000,
`arousal > 0.7` held almost always, and over 79,200 frames the pulse read 2.0 on
94.3% of them with two branches unreachable.
"""

from __future__ import annotations

import pytest

import core.soma.pulse as pulse_module
from core.soma.pulse import ALERT, RESTING, ArousalLedger, expression_for, rate


@pytest.fixture(autouse=True)
def _fresh():
    pulse_module.reset_for_test()
    yield
    pulse_module.reset_for_test()


def _run(values: list[float], window: int = 256) -> list[float]:
    ledger = ArousalLedger(window=window)
    return [rate(value, ledger=ledger) for value in values]


def test_the_rate_stays_on_the_span_the_branches_declared():
    out = _run([0.1 * (index % 11) for index in range(400)])
    assert min(out) >= RESTING
    assert max(out) <= ALERT


def test_an_arousal_that_never_leaves_the_top_still_varies():
    """The campaign's own range: nothing below 0.605, and the old channel sat
    at its ceiling for 94.3% of it."""
    rng = __import__("random").Random(7)
    high = [0.605 + 0.395 * rng.random() for _ in range(2000)]
    out = _run(high)
    at_ceiling = sum(1 for value in out if value >= ALERT - 1e-9) / len(out)
    assert at_ceiling < 0.02, at_ceiling
    assert len(set(round(value, 4) for value in out)) > 100


def test_a_rank_cannot_saturate_at_any_window():
    """Which is what makes this a channel rather than a flag."""
    rng = __import__("random").Random(11)
    high = [0.9 + 0.1 * rng.random() for _ in range(1500)]
    for window in (32, 128, 512, 1024):
        out = _run(high, window=window)
        share = sum(1 for value in out if value >= ALERT - 1e-9) / len(out)
        assert share < 0.05, (window, share)


def test_rising_arousal_reads_faster_than_falling_arousal():
    ledger = ArousalLedger()
    for value in [0.5] * 40:
        rate(value, ledger=ledger)
    assert rate(0.9, ledger=ledger) > rate(0.1, ledger=ledger)


def test_the_middle_is_returned_until_there_is_something_to_rank_against():
    ledger = ArousalLedger()
    first = rate(0.9, ledger=ledger)
    assert first == pytest.approx(RESTING + 0.5 * (ALERT - RESTING))


def test_a_reading_that_is_not_a_number_does_not_move_her():
    ledger = ArousalLedger()
    for value in [0.4] * 40:
        rate(value, ledger=ledger)
    assert rate("not a number", ledger=ledger) == pytest.approx(
        RESTING + 0.5 * (ALERT - RESTING)
    )


def test_the_word_is_the_branch_chain_unchanged():
    assert expression_for(0.6, 0.6) == "engaged"
    assert expression_for(-0.5, 0.5) == "contemplative"
    assert expression_for(0.0, 0.8) == "alert"
    assert expression_for(0.0, 0.2) == "resting"
    assert expression_for(0.0, 0.5) == "neutral"


def test_the_loop_reads_the_rate_from_here():
    """A rate computed and not used is the defect this replaced."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core" / "phases" / "proprioceptive_loop.py"
    ).read_text()
    tree = ast.parse(source)
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert "pulse_rate" in names
    assert "expression_for" in names
    assert "1.5" not in source.split("current_expression")[1][:400]
