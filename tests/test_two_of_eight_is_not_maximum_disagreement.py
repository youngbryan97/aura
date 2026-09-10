"""Disagreement is measured against the readinesses there are, not the active ones.

`interiority.tendency_conflict` is normalised entropy over action readinesses,
and `self.coherence` is one minus it. Normalised by the number ACTIVE, two
equally-weighted readinesses score 1.0 — the same as eight equally-weighted
ones — because two of two is as even as evenness gets. The most ordinary
interior state there is pinned the channel at its ceiling and coherence at
zero.

LIVE, 2026-09-07: `interiority.tendency_conflict went nominal -> red_high at
1.0ratio` within a minute of boot, beside a vocabulary of eight readinesses of
which two were active.

The denominator is measured, not declared: a faculty picks its tendency per
activation, so a list written into the module could not be trusted to be the
vocabulary. It grows as readinesses are seen and never shrinks, and every
reading carries the denominator it was taken against.
"""

from __future__ import annotations

import math

import pytest

from core.interiority.arbitration import _tendency_conflict


def test_one_readiness_is_no_disagreement() -> None:
    assert _tendency_conflict({"approach": 1.0}, 8) == 0.0


def test_two_of_eight_is_not_the_ceiling() -> None:
    conflict = _tendency_conflict({"approach": 1.0, "inhibit": 1.0}, 8)
    assert conflict == pytest.approx(math.log(2) / math.log(8))
    assert conflict < 0.4


def test_eight_of_eight_is_the_ceiling() -> None:
    even = {f"tendency{index}": 1.0 for index in range(8)}
    assert _tendency_conflict(even, 8) == pytest.approx(1.0)


def test_a_dominant_readiness_reads_low() -> None:
    weights = {"approach": 10.0, "inhibit": 0.2, "attend": 0.1}
    assert _tendency_conflict(weights, 8) < 0.25


def test_the_denominator_is_never_smaller_than_what_was_seen() -> None:
    """A reading cannot be against a vocabulary smaller than the active set."""

    even = {f"tendency{index}": 1.0 for index in range(5)}
    assert _tendency_conflict(even, 2) == pytest.approx(1.0)


def test_the_vocabulary_grows_from_what_faculties_produce() -> None:
    from core.interiority import arbitration

    before = set(arbitration._TENDENCIES_SEEN)
    arbitration._TENDENCIES_SEEN.update({"approach", "inhibit"})
    try:
        assert "approach" in arbitration.tendency_vocabulary()
        assert set(arbitration.tendency_vocabulary()) >= {"approach", "inhibit"}
    finally:
        arbitration._TENDENCIES_SEEN.clear()
        arbitration._TENDENCIES_SEEN.update(before)


def test_a_reading_carries_the_denominator_it_was_taken_against() -> None:
    """Without it, a 1.0 from two readinesses reads like a 1.0 from eight."""

    from core.interiority.arbitration import Arbitrated

    assert "tendency_vocabulary" in Arbitrated.__dataclass_fields__
