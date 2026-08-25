"""Does the state at one turn carry anything about the next?

Two corpora with a known answer. In one, the next turn's reply length is
decided by a named dimension of the current state. In the other, reply length
is drawn independently of it. The first must be reported as anticipation and
the second must not, or the test is a coin that always lands the same way.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.brain.llm.endogenous_anticipation import (
    MIN_PAIRS,
    channel_value,
    measure_anticipation,
    reply_length,
)
from core.brain.llm.endogenous_pair_recorder import RecordedPair
from core.brain.llm.endogenous_state import FEATURE_INDEX, STATE_DIM

DRIVER = FEATURE_INDEX["uncertainty.confidence"]


def _corpus(*, linked: bool, turns: int = 240, seed: int = 5):
    rng = np.random.default_rng(seed)
    out: list[RecordedPair] = []
    pending: float | None = None
    stamp = 1_000_000.0
    for _ in range(turns):
        values = rng.normal(0.0, 0.4, STATE_DIM).astype(np.float32)
        values[DRIVER] = float(rng.uniform(0.0, 1.0))
        # A reply whose length was decided by the PREVIOUS state, when linked.
        length = (
            int(6 + 40 * pending)
            if (linked and pending is not None)
            else int(rng.integers(6, 46))
        )
        pending = float(values[DRIVER])
        stamp += 1.0
        out.append(
            RecordedPair(
                values=values,
                present=np.ones(STATE_DIM, dtype=bool),
                text=" ".join(["word"] * max(1, length)),
                lane="chat",
                model="test",
                recorded_at=stamp,
            )
        )
    return out


def test_a_linked_corpus_is_reported_as_anticipation():
    result = measure_anticipation(_corpus(linked=True), permutations=200)
    assert result is not None
    assert result.anticipates is True
    assert result.correlation > 0.3
    assert result.p_value <= 0.05
    assert result.as_dict()["what_this_means"]


def test_an_unlinked_corpus_is_not():
    result = measure_anticipation(_corpus(linked=False), permutations=200)
    assert result is not None
    assert result.anticipates is False


def test_a_corpus_out_of_order_is_refused():
    """A shuffled corpus would make this measure neighbours, not sequence."""
    pairs = _corpus(linked=True)
    pairs[10], pairs[11] = pairs[11], pairs[10]
    assert measure_anticipation(pairs, permutations=10) is None


def test_too_few_turns_refuses():
    assert measure_anticipation(_corpus(linked=True, turns=MIN_PAIRS - 5)) is None


def test_a_state_dimension_can_be_the_target():
    result = measure_anticipation(
        _corpus(linked=True),
        target=channel_value("uncertainty.confidence"),
        target_name="next.uncertainty.confidence",
        permutations=100,
    )
    assert result is not None
    assert result.target == "next.uncertainty.confidence"


def test_a_constant_prediction_correlates_with_nothing():
    """Zero is the honest answer; numpy would hand back a nan."""
    flat = _corpus(linked=False)
    for pair in flat:
        pair.values[:] = 0.0
    result = measure_anticipation(flat, permutations=50)
    assert result is not None
    assert result.correlation == pytest.approx(0.0, abs=1e-9)
    assert result.anticipates is False


def test_reply_length_reads_the_recorded_text():
    pair = RecordedPair(
        values=np.zeros(STATE_DIM, dtype=np.float32),
        present=np.ones(STATE_DIM, dtype=bool),
        text="one two three",
        lane="",
        model="",
        recorded_at=1.0,
    )
    assert reply_length(pair) == 3.0
