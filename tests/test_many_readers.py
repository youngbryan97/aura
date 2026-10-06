"""Where no test says pass or fail, many readers comparing two at a time decide, and say a tie where there is one."""
from __future__ import annotations

import random

import pytest

from core.stand_ins.many_readers import contrast_ratio, fitts_difficulty, judged_by_many, reading_ease

pytestmark = pytest.mark.unit

_QUALITY = {"plain": 0.2, "decent": 0.5, "strong": 0.9}


def _reader(seed: int, bias: float = 0.25, noise: float = 0.25):
    """A reader who sees quality through noise and leans to whichever is shown first."""
    rng = random.Random(seed)

    def read(criterion, first, second):
        lean = _QUALITY[first] - _QUALITY[second] + bias + rng.gauss(0, noise)
        return 1 if lean > 0 else 2

    return read


def test_the_better_version_is_found_through_noisy_readers_who_lean_to_what_they_see_first():
    standing = judged_by_many(list(_QUALITY), ["makes its case", "answers the strongest objection"], [_reader(s) for s in range(5)])
    assert standing.winner == 2, standing.says(list(_QUALITY))
    assert standing.strength[2] > standing.strength[1] > standing.strength[0]


def test_versions_no_reader_can_tell_apart_are_a_tie():
    rng = random.Random(3)
    standing = judged_by_many(["a", "b", "c"], ["reads well"], [lambda c, x, y: rng.choice([1, 2]) for _ in range(4)])
    assert standing.winner is None and "a tie" in standing.says()


def test_a_reader_who_always_picks_the_first_shown_decides_nothing():
    standing = judged_by_many(["a", "b"], ["reads well"], [lambda c, x, y: 1])
    assert standing.winner is None and abs(standing.strength[0] - standing.strength[1]) < 1e-9


def test_a_measure_counts_with_no_reader_unsure_of_it():
    easy, hard = "The cat sat. It was warm. We ate.", "Notwithstanding considerable epistemological uncertainty, macroeconomic institutionalization proceeds."
    assert reading_ease(easy) > 80 > reading_ease(hard)
    standing = judged_by_many([hard, easy], [], [lambda c, x, y: 0], measures={"reads easily": reading_ease})
    assert standing.winner == 1


def test_screen_measures_are_the_standard_ones():
    assert round(contrast_ratio("#000000", "#ffffff"), 1) == 21.0
    assert 4.4 < contrast_ratio("#777777", "#ffffff") < 4.6  # just under the 4.5 floor for body text
    assert fitts_difficulty(400, 40) > fitts_difficulty(100, 40)
