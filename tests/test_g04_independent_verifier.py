"""G08's verifier interprets and tests with its own code, and its own code is right."""

from __future__ import annotations

import pytest

from tools import verify_g04_independently as verifier


def test_its_interpreter_runs_the_six_primitives() -> None:
    steps = [("add", (0, 1)), ("mul", (3, 2)), ("sub", (4, 0)), ("idiv", (5, 1))]
    assert verifier._run(steps, (7, 3, 5)) == ((7 + 3) * 5 - 7) // 3
    assert verifier._run([("at", (0, 1))], ((4, 9, 16), 2)) == 16
    assert verifier._run([("count_of", (0, 1))], ((5, 1, 5, 5), 5)) == 3
    with pytest.raises(IndexError):
        verifier._run([("at", (0, 1))], ((4, 9), 2))
    with pytest.raises(ZeroDivisionError):
        verifier._run([("idiv", (0, 1))], (4, 0))


def test_its_exact_tail_matches_hand_counts() -> None:
    assert verifier.binomial_tail_at_least(9, 10) == pytest.approx(11 / 1024)
    assert verifier.binomial_tail_at_least(0, 0) == 1.0
    assert verifier.binomial_tail_at_least(5, 5) == pytest.approx(1 / 32)


def test_its_interval_brackets_the_rate() -> None:
    low, high = verifier.clopper_pearson(62, 62)
    assert high == 1.0 and 0.94 < low < 0.96
    low, high = verifier.clopper_pearson(31, 62)
    assert low < 0.5 < high
