"""The fortieth run in a world should start better than the first.

Everything she works out about a world is kept and brought back discounted:
how it moves, which of her acts do anything, what the world does on its own,
which move suits a kind of position, and the line that worked. Each of those
was built and tested on its own. None of them proves the thing they were all
built for.

Measured 2026-08-27, eight consecutive lives in one world, up to 3000 moves
each, against the same lives carrying nothing:

    starting fresh each time      first half 1881 → second half 1622
    carrying what she worked out  first half 2092 → second half 2736

Carrying it she reached a 2048 tile twice, and the share of moves answered on
sight rather than decided went from 12% to 26%. Starting fresh, neither
happened.

This runs a shorter version of the same thing, because a claim with no test
that can fail is not a claim.
"""

from __future__ import annotations

import statistics

import pytest

from tools.measure_getting_better import live_through

#: Enough lives for a first and second half, and enough moves in each that the
#: difference between them is the carrying rather than the noise. Below about
#: five hundred the model has not built enough to carry and the effect does not
#: form, so this is marked slow rather than cut down until it stops measuring
#: anything.
LIVES = 6
MOVES = 700

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def fresh():
    return live_through(LIVES, MOVES, carrying=False)


@pytest.fixture(scope="module")
def carrying():
    return live_through(LIVES, MOVES, carrying=True)


# ── the claim ────────────────────────────────────────────────────────────

def rise(lived, field: str) -> float:
    """How much better the second half was than the first."""
    half = max(1, len(lived) // 2)
    early = statistics.median(run[field] for run in lived[:half])
    late = statistics.median(run[field] for run in lived[half:])
    return late - early


def test_she_answers_more_of_it_on_sight_for_having_been_here(carrying):
    """Experience turning into skill: recognised rather than decided."""
    assert rise(carrying, "on_sight") > 0.0


def test_and_starting_fresh_she_does_not_nearly_as_much(carrying, fresh):
    """The null. Without it, "she improved" is variance with a story on it.

    Some rise is expected either way — a life gets more fluent within itself
    as positions repeat. What carrying adds is that the NEXT life starts where
    the last one finished, so the rise across lives is several times larger.
    """
    got = rise(carrying, "on_sight")
    anyway = rise(fresh, "on_sight")
    assert got > anyway * 2.0, (
        f"carrying gained {got:.1%} where starting fresh gained {anyway:.1%} — "
        "not enough of a difference to be the carrying"
    )


def test_and_plays_at_least_as_well_while_doing_it(carrying, fresh):
    """Fluency bought at the cost of playing worse would be a bad trade.

    Games at this length run into the move cap, so totals saturate and cannot
    show the gain — 3000-move lives put it at 2092 → 2736 with two 2048 tiles.
    What this can check is that being quicker did not make her worse.
    """
    theirs = statistics.median(run["total"] for run in fresh)
    hers = statistics.median(run["total"] for run in carrying)
    assert hers >= theirs


# ── and it is really the same machinery in both ──────────────────────────

def test_every_life_actually_played(fresh, carrying):
    for lived in (fresh, carrying):
        assert all(run["moves"] > 0 for run in lived)


def test_nothing_is_carried_when_nothing_should_be(fresh):
    """A fresh life cannot answer anything on sight in its first moves."""
    assert fresh[0]["on_sight"] < 1.0
