"""ISC-v3, assembled beside ISC-v1 and ISC-v2 and changing neither.

docs/ISC_V3_PREREGISTRATION.md changes the synergy line and judges every null
architecture by the same line, in section 1's comparison set and in section 3's
conjunction. These pin that a run from before v3 reports no v3 lines, a v3 run
reports three beside untouched v1 and v2 verdicts, a null table without the v3
reading fails both null lines, and the scorecard decides v3's null line across
seeds.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from core.subject.battery import assemble
from core.subject.null_verdicts import (
    REFERENCE,
    beats_the_v3_comparison_set,
    comparison_set_v3,
    passes_the_v3_conjunction,
    v3_conjunctions,
    v3_readable,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]


def _scorecard_module():
    spec = importlib.util.spec_from_file_location("subject_core_scorecard", REPO / "tools" / "subject_core_scorecard.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _architecture(phi: float, *, v2: bool, v3: bool) -> dict:
    return {
        "kind": "architecture", "phi_do": phi, "one_component": True, "vertex_connectivity": 2,
        "reentry": True, "closed": True, "synergy_passes": [False], "synergy_v2_passes": [v2], "synergy_v3_passes": [v3],
    }


def _table(*, hub_v3: bool = False, hub_phi: float = 0.1) -> dict:
    return {
        REFERENCE: _architecture(0.3, v2=False, v3=True),
        "hub": _architecture(hub_phi, v2=False, v3=hub_v3),
        "replay": {"kind": "surrogate", "phi_do": 0.02},
    }


def _evidence(*, v3: bool = True, table: dict | None = None) -> dict:
    change = {"sources": ["A", "S"], "target": "G", "passes": False, "synergy_fraction": 0.4}
    if v3:
        change.update({"passes_v3": True, "margin_over_bootstrap": 0.05})
    return {
        "synergy": [{"sources": ["A", "S"], "target": "G", "passes": False, "synergy_fraction": 0.2}],
        "synergy_v2": [change],
        "persistence_v2": {"passes": True, "gain_lower_bound": 0.2, "over_shuffle_lower_bound": 0.1},
        "nulls": {
            "all_nulls_fail": False, "summary": {}, "phi_beats_all": False, "real_lower_bound": 0.25,
            "detail": _table() if table is None else table,
            "conjunction": {REFERENCE: False, "hub": False},
        },
    }


def test_a_run_from_before_v3_reports_no_v3_lines() -> None:
    verdict = assemble(_evidence(v3=False))
    assert len(verdict.v2_criteria) == 4
    assert verdict.v3_criteria == []
    assert verdict.as_dict()["v3_passed"] is None


def test_a_v3_run_reports_three_lines_and_leaves_v1_and_v2_alone() -> None:
    before = assemble(_evidence(v3=False))
    verdict = assemble(_evidence())
    assert [(c.key, c.passed) for c in verdict.criteria] == [(c.key, c.passed) for c in before.criteria]
    assert [(c.key, c.passed) for c in verdict.v2_criteria] == [(c.key, c.passed) for c in before.v2_criteria]
    assert {c.key: c.passed for c in verdict.v3_criteria} == {
        "partition_beats_nulls": True, "synergy": True, "beats_every_null": True,
    }
    v2 = {c.key: c.passed for c in verdict.v2_criteria}
    assert v2["synergy"] is False and v2["beats_every_null"] is False


def test_the_v3_lines_stand_in_for_the_lines_they_replace() -> None:
    verdict = assemble(_evidence())
    lines = verdict.v3_lines()
    keys = [c.key for c in lines]
    assert len(keys) == len(set(keys)) == len(verdict.criteria)
    by_key = {c.key: c for c in lines}
    assert "(ISC-v3)" in by_key["synergy"].statement
    assert "(ISC-v2)" in by_key["intrinsic_persistence"].statement


def test_a_null_that_passes_the_v3_line_joins_the_comparison_set_and_can_fail_aura() -> None:
    table = _table(hub_v3=True, hub_phi=0.5)
    assert comparison_set_v3(table) == {"hub": 0.5, "replay": 0.02}
    assert beats_the_v3_comparison_set(0.25, table)[0] is False
    lines = {c.key: c.passed for c in assemble(_evidence(table=table)).v3_criteria}
    assert lines["partition_beats_nulls"] is False
    assert lines["beats_every_null"] is False


def test_the_reference_is_held_to_the_same_synergy_line() -> None:
    table = _table()
    assert passes_the_v3_conjunction(REFERENCE, table)
    table[REFERENCE]["synergy_v3_passes"] = [False]
    assert not passes_the_v3_conjunction(REFERENCE, table)


def test_a_null_table_without_the_v3_reading_fails_both_null_lines() -> None:
    table = _table()
    del table["hub"]["synergy_v3_passes"]
    assert not v3_readable(table)
    assert v3_conjunctions(table) == {}
    assert beats_the_v3_comparison_set(0.25, table) == (False, {})
    lines = {c.key: c.passed for c in assemble(_evidence(table=table)).v3_criteria}
    assert lines == {"partition_beats_nulls": False, "synergy": True, "beats_every_null": False}


def test_an_unread_organ_invalidates_a_v3_line_as_it_does_the_others() -> None:
    from core.subject.battery import CRITERION_ORGANS, MISSING_SHARE

    organs = {organ for key in ("synergy", "partition_beats_nulls", "beats_every_null") for organ in CRITERION_ORGANS.get(key, ())}
    if not organs:
        pytest.skip("no v3-replaced criterion rests on a named organ")
    evidence = _evidence()
    evidence["recording"] = {"misses": {f"organ:{organ}.status": {"share": MISSING_SHARE} for organ in organs}}
    for item in assemble(evidence).v3_criteria:
        if CRITERION_ORGANS.get(item.key):
            assert item.passed is False
            assert "invalid" in item.detail


def test_the_scorecard_decides_v3s_null_line_across_seeds() -> None:
    card = _scorecard_module()
    tables = [_table(), _table(), _table()]
    tables[2][REFERENCE]["synergy_v3_passes"] = [False]
    reports = [
        {"verdict": assemble(_evidence(table=table)).as_dict(), "nulls": {"detail": table}} for table in tables
    ]
    from core.subject.null_verdicts import verdict_across_seeds

    null_verdict = verdict_across_seeds([card._v3_conjunction(report) for report in reports])
    assert null_verdict["instrument_failed"] is True
    summary = card._isc_v3(reports, null_verdict)
    assert summary["runs"] == 3
    assert "beats_every_null" in summary["failing"]
    assert summary["isc_v3"] is False
    assert card._isc_v3([{"verdict": assemble(_evidence(v3=False)).as_dict()}], None) is None
