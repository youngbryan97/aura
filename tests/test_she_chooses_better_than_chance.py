"""A score is not a number until something else has been scored the same way.

The whole stack — a typed reading, a transition model worked out by watching,
a score over imagined states, a search as deep as there is time for — is worth
exactly what it beats. So it is measured against a null that changes only how
the move is picked, through the same perception and the same world.

Measured 2026-08-26 over 60 games of 400 moves: choosing at random reached a
median best of 128 and wasted 16.1% of its moves; choosing this way reached
512 and wasted 0.1%.
"""

from __future__ import annotations

import statistics

import pytest

from tools.measure_choosing import play


def _runs(how: str, games: int = 6, moves: int = 120) -> list[dict[str, float]]:
    return [play(how, moves=moves, seed=seed) for seed in range(games)]


@pytest.fixture(scope="module")
def chance() -> list[dict[str, float]]:
    return _runs("random")


@pytest.fixture(scope="module")
def hers() -> list[dict[str, float]]:
    return _runs("her scoring")


def test_choosing_this_way_gets_further_than_chance(chance, hers):
    assert statistics.median(r["best"] for r in hers) > statistics.median(
        r["best"] for r in chance
    )


def test_she_hardly_ever_makes_a_move_that_does_nothing(chance, hers):
    """Ruling one out before making it is the point of trying without doing."""
    wasted = sum(r["wasted"] for r in hers) / max(1, sum(r["moves"] for r in hers))
    by_chance = sum(r["wasted"] for r in chance) / max(1, sum(r["moves"] for r in chance))
    assert wasted < by_chance / 4


def test_she_works_out_how_the_world_moves_every_time(hers):
    assert all(run["knew"] for run in hers)


def test_the_null_is_the_same_world_and_the_same_perception():
    """Only the choosing differs, or the comparison means nothing."""
    import inspect

    from tools import measure_choosing

    source = inspect.getsource(measure_choosing.play)
    assert 'if how == "random"' in source
    assert source.count("knows.watched") == 1


def test_a_held_line_is_read_and_scored(hers):
    """It agrees with plain scoring on this world, which is worth saying."""
    with_line = _runs("her line")
    assert statistics.median(r["best"] for r in with_line) >= statistics.median(
        r["best"] for r in hers
    ) / 2
