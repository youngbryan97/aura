"""Whether one way is better than another, decided pair by pair with two error rates bounded
(core/cognition/whether_one_way_is_better.py).

LIVE 26 November 2026 her practice kept a change only where a fixed match beat a bar raised with every try; after
194 tries of one change the bar was out of reach, and nothing was kept.
"""
from __future__ import annotations

import random

import pytest

from core.cognition.whether_one_way_is_better import Comparison

pytestmark = pytest.mark.unit


def _a_pair(rng: random.Random, mean: float) -> float:
    """A pair's score of the given mean: each trial won, drawn or lost, a third of them drawn."""
    def one() -> float:
        draw = 1 / 3
        win = mean - draw / 2
        roll = rng.random()
        return 1.0 if roll < win else 0.5 if roll < win + draw else 0.0
    return (one() + one()) / 2


def _decide(rng: random.Random, mean: float, better: float, most: int = 20_000) -> tuple[str | None, int]:
    test = Comparison(better=better)
    for n in range(1, most + 1):
        said = test.add(_a_pair(rng, mean))
        if said:
            return said, n
    return None, most


def test_a_way_far_better_is_found_so_in_few_pairs() -> None:
    said, pairs = _decide(random.Random(1), mean=0.75, better=0.55)
    assert said == "better" and pairs < 40


def test_a_way_no_better_is_found_so() -> None:
    said, _pairs = _decide(random.Random(2), mean=0.5, better=0.55)
    assert said == "no better"


@pytest.mark.parametrize(("mean", "wrong"), [(0.5, "better"), (0.55, "no better")])
def test_each_error_is_about_as_rare_as_it_was_set(mean: float, wrong: str) -> None:
    rng = random.Random(3)
    told = [_decide(rng, mean=mean, better=0.55)[0] for _ in range(400)]
    # Set at one in twenty; the normal approximation and stopping at a pair let it run a little over.
    assert told.count(wrong) / len(told) < 0.09


def test_pairs_all_alike_say_little_until_there_are_many() -> None:
    test = Comparison(better=0.55)
    assert test.add(0.5) is None
    assert test.add(0.5) is None
    for _ in range(200):
        said = test.add(0.5)
    assert said == "no better"
