"""Could a reader tell which state produced which reply?

The experiment this supports holds the prompt identical across two arms and
varies only z_Aura. If a weak reader can separate the replies, information
crossed by the endogenous route rather than through the context window. The
reader is tested here against arms built to be separable and arms built to be
identical, because a reader that always separates is not a reader.
"""

from __future__ import annotations

import numpy as np

from core.brain.llm.substrate_only_channel import (
    MIN_PER_ARM,
    measure_recoverability,
    state_entropy_bound,
)


def _replies(hedged: bool, count: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    shared = "the system reports a reading from the sensor and continues".split()
    marks = ["perhaps", "maybe", "unsure", "might"] if hedged else [
        "clearly",
        "certainly",
        "definitely",
        "is",
    ]
    out = []
    for _ in range(count):
        words = list(rng.choice(shared, size=12)) + marks
        rng.shuffle(words)
        out.append(" ".join(words))
    return out


def test_separable_arms_are_recovered():
    result = measure_recoverability(
        _replies(True, 14, 1), _replies(False, 14, 2), permutations=200
    )
    assert result is not None
    assert result.recovered is True
    assert result.accuracy > 0.5
    assert result.p_value <= 0.05
    assert result.as_dict()["what_this_means"]


def test_identical_arms_are_not_recovered():
    result = measure_recoverability(
        _replies(True, 14, 3), _replies(True, 14, 4), permutations=200
    )
    assert result is not None
    assert result.recovered is False


def test_too_few_replies_refuses_rather_than_reporting():
    assert (
        measure_recoverability(
            _replies(True, MIN_PER_ARM - 1, 5), _replies(False, MIN_PER_ARM, 6)
        )
        is None
    )


def test_a_vocabulary_of_one_word_refuses():
    """One word cannot separate anything, and the reader says so."""
    assert measure_recoverability(["a"] * 10, ["a"] * 10, permutations=10) is None


def test_arms_with_disjoint_vocabulary_are_trivially_recovered():
    result = measure_recoverability(["a"] * 10, ["b"] * 10, permutations=200)
    assert result is not None
    assert result.accuracy == 1.0
    assert result.recovered is True


def test_the_reader_names_the_words_it_used():
    result = measure_recoverability(
        _replies(True, 12, 7), _replies(False, 12, 8), permutations=50
    )
    assert result.top_words_low and result.top_words_high
    assert set(result.top_words_low) != set(result.top_words_high)


def test_length_does_not_decide_the_arm():
    """A longer reply must not be easier to classify because it is longer."""
    short = ["alpha beta gamma"] * 12
    long = ["alpha beta gamma " * 8] * 12
    result = measure_recoverability(short, long, permutations=100)
    assert result is None or result.recovered is False


def test_the_information_ceiling_is_stated():
    assert state_entropy_bound(74) > 6.0
    assert state_entropy_bound(0) == 0.0
