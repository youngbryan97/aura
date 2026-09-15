"""ISC-v2, assembled beside ISC-v1 and changing none of it.

docs/ISC_V2_PREREGISTRATION.md changes three lines: irreducibility is compared
against the v2 comparison set, synergy is scored on the target's change, and
the null line is decided across the campaign's declared seeds. These pin that
a run from before v2 reports no v2 lines, a v2 run reports all three beside an
untouched v1 verdict, and the scorecard decides the null line across seeds.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from core.subject.battery import assemble
from core.subject.null_verdicts import passes_the_v2_conjunction

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]


def _scorecard_module():
    spec = importlib.util.spec_from_file_location("subject_core_scorecard", REPO / "tools" / "subject_core_scorecard.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _evidence(*, v2: bool, reference_passes: bool = True, change_synergy: bool = True) -> dict:
    evidence = {
        "synergy": [{"sources": ["A", "S"], "target": "G", "passes": False, "synergy_fraction": 0.2}],
        "nulls": {
            "all_nulls_fail": False,
            "summary": {},
            "phi_beats_all": False,
            "conjunction": {"recurrent": reference_passes, "hub": False, "star": False},
        },
    }
    if v2:
        evidence["synergy_v2"] = [
            {"sources": ["A", "S"], "target": "G", "passes": change_synergy, "synergy_fraction": 0.4}
        ]
        evidence["nulls"]["v2_phi_beats_comparison_set"] = True
        evidence["nulls"]["v2_comparison_set"] = {"replay": 0.02}
    return evidence


def test_a_run_from_before_v2_reports_no_v2_lines() -> None:
    verdict = assemble(_evidence(v2=False))
    assert verdict.v2_criteria == []
    assert verdict.as_dict()["v2_passed"] is None


def test_a_v2_run_reports_all_three_lines_and_leaves_v1_alone() -> None:
    v1_only = assemble(_evidence(v2=False))
    verdict = assemble(_evidence(v2=True))
    assert [c.key for c in verdict.v2_criteria] == ["partition_beats_nulls", "synergy", "beats_every_null"]
    assert [(c.key, c.passed) for c in verdict.criteria] == [(c.key, c.passed) for c in v1_only.criteria]
    lines = {c.key: c.passed for c in verdict.v2_criteria}
    assert lines == {"partition_beats_nulls": True, "synergy": True, "beats_every_null": True}
    v1 = {c.key: c.passed for c in verdict.criteria}
    assert v1["synergy"] is False and v1["partition_beats_nulls"] is False


def test_the_v2_lines_stand_in_for_the_v1_lines_they_replace() -> None:
    verdict = assemble(_evidence(v2=True))
    keys = [c.key for c in verdict.v2_lines()]
    assert len(keys) == len(set(keys)) == len(verdict.criteria)


def test_a_reference_that_fails_on_this_seed_fails_the_seed_reading() -> None:
    verdict = assemble(_evidence(v2=True, reference_passes=False))
    assert {c.key: c.passed for c in verdict.v2_criteria}["beats_every_null"] is False


def test_an_unread_organ_invalidates_a_v2_line_as_it_does_the_v1_line() -> None:
    from core.subject.battery import CRITERION_ORGANS, MISSING_SHARE

    organs = {organ for key in ("synergy", "partition_beats_nulls", "beats_every_null") for organ in CRITERION_ORGANS.get(key, ())}
    if not organs:
        pytest.skip("no v2-replaced criterion rests on a named organ")
    evidence = _evidence(v2=True)
    evidence["recording"] = {"misses": {f"organ:{organ}.status": {"share": MISSING_SHARE} for organ in organs}}
    verdict = assemble(evidence)
    for item in verdict.v2_criteria:
        if CRITERION_ORGANS.get(item.key):
            assert item.passed is False
            assert "invalid" in item.detail


def test_the_v2_null_conjunction_judges_synergy_on_the_change() -> None:
    table = {
        "recurrent": {"kind": "architecture", "phi_do": 0.3, "one_component": True, "vertex_connectivity": 2,
                      "reentry": True, "closed": True, "synergy_passes": [False], "synergy_v2_passes": [True]},
        "replay": {"kind": "surrogate", "phi_do": 0.02},
    }
    assert passes_the_v2_conjunction("recurrent", table)
    table["recurrent"]["synergy_v2_passes"] = [False]
    assert not passes_the_v2_conjunction("recurrent", table)


def test_the_scorecard_decides_the_null_line_across_seeds() -> None:
    card = _scorecard_module()
    runs = [assemble(_evidence(v2=True, reference_passes=passes)).as_dict() for passes in (True, True, False)]
    reports = [
        {"verdict": run, "nulls": {"conjunction": {"recurrent": passes, "hub": False}}}
        for run, passes in zip(runs, (True, True, False), strict=True)
    ]
    null_verdict = {"all_nulls_fail": False, "reference_passes": False}
    summary = card._isc_v2(reports, null_verdict)
    assert summary["runs"] == 3
    assert "beats_every_null" in summary["failing"]
    assert summary["isc_v2"] is False
    assert card._isc_v2([{"verdict": assemble(_evidence(v2=False)).as_dict()}], None) is None
