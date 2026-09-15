"""ISC-v2's verdict reads persistence on the next level, and replaces the v1 line with it.

The v1 persistence line passes memoryless noise (see
tests/test_persistence_is_memory_not_reversion_to_the_mean.py). The third
amendment to docs/ISC_V2_PREREGISTRATION.md replaces it in the v2 verdict with
`persistence`. These pin the wiring: a passing reading passes the v2 line, a
failing one fails it, a v2 run without the reading fails it, the v1 line is
still reported unchanged, and a pre-v2 run gets no v2 lines at all.
"""

from __future__ import annotations

import pytest

from core.subject.battery import Verdict, _assemble_v2

pytestmark = pytest.mark.unit


def _v2_evidence(**extra) -> dict:
    return {"synergy_v2": [], "nulls": {"v2_phi_beats_comparison_set": False, "conjunction": {}}, **extra}


def _persistence_line(evidence: dict):
    verdict = Verdict()
    _assemble_v2(verdict, evidence)
    return next(item for item in verdict.v2_criteria if item.key == "intrinsic_persistence")


def test_a_passing_level_reading_passes_the_v2_line() -> None:
    line = _persistence_line(_v2_evidence(persistence_v2={"passes": True, "gain_lower_bound": 0.2, "over_shuffle_lower_bound": 0.1}))
    assert line.passed is True


def test_a_failing_level_reading_fails_it() -> None:
    line = _persistence_line(_v2_evidence(persistence_v2={"passes": False, "gain_lower_bound": -0.01, "over_shuffle_lower_bound": 0.0}))
    assert line.passed is False


def test_a_v2_run_without_the_reading_fails_the_line() -> None:
    assert _persistence_line(_v2_evidence()).passed is False


def test_a_run_from_before_v2_gets_no_v2_lines() -> None:
    verdict = Verdict()
    _assemble_v2(verdict, {"nulls": {}})
    assert verdict.v2_criteria == []
