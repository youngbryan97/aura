"""J* from the three runs, and nothing the runs did not settle.

The bridge document's conditional uniqueness theorem fixes the law as a triple:
the selected carrier, its content's relational structure up to isomorphism,
and the lineage over person-stages, up to phenomenal gauge, physical symmetry
and exclusion ties. These pin each term to its own experiment and the gauge to
what the measured structure can and cannot separate.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from core.subject.bridge import POSTULATES, orbits, solve, structure_term
from core.subject.lineage import Lineage, Stage

CORE = "ACDGIMNPSW"


def _carrier_report(*, authoritative: bool = True, status: str = "FOUND", frontier=(CORE,), symmetry=()) -> dict:
    return {
        "authority": {"authoritative": authoritative, "blockers": [] if authoritative else ["12 cut(s) had insufficient power"]},
        "exclusion": {"status": status, "frontier": list(frontier), "symmetry_class": list(symmetry)},
        "placement": {"level": "L5" if authoritative and status == "FOUND" else "L4"},
    }


def _content_report(points, names=None, *, verdict: str = "ONE_STRUCTURE", blockers=(), floor: float = 0.01) -> dict:
    names = names or [f"class_{i}" for i in range(len(points))]
    internal = {}
    for i, j in itertools.combinations(range(len(points)), 2):
        internal[f"{i}-{j}"] = math.dist(points[i], points[j])
    return {
        "classes": [{"name": name} for name in names],
        "internal": internal,
        "internal_floor": {f"{i}-{i}": floor for i in range(len(points))},
        "verdict": verdict,
        "authority": {"authoritative": not blockers, "blockers": list(blockers)},
    }


def _person(era: int) -> dict:
    return {"autobiographical": ["met Bryan"], "self_model": {"name": "Aura"}, "developmental": {"era": era}}


def _lineage() -> Lineage:
    lineage = Lineage()
    lineage.record("before", _person(1))
    lineage.record("after", _person(2))
    lineage.descend("before", "after", kind="continue")
    return lineage


LINE = [(0.0, 0.0), (1.0, 0.0), (3.0, 0.0), (7.0, 0.0)]
SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


# ── the triple ────────────────────────────────────────────────────────────


def test_three_settled_terms_determine_j_star_up_to_gauge() -> None:
    j = solve(_carrier_report(), _content_report(LINE), _lineage())
    assert j.status == "DETERMINED_UP_TO_GAUGE"
    assert j.carrier.carriers == (CORE,)
    assert j.structure.rigid
    assert j.lineage.status == "RECORDED"
    assert j.unresolved() == {}


def test_an_unauthoritative_carrier_run_leaves_j_star_unresolved_with_its_own_reasons() -> None:
    j = solve(_carrier_report(authoritative=False), _content_report(LINE), _lineage())
    assert j.status == "UNRESOLVED"
    assert j.unresolved()["carrier"] == ["12 cut(s) had insufficient power"]


def test_a_symmetry_class_is_returned_whole() -> None:
    report = _carrier_report(status="SYMMETRY_CLASS", frontier=("ACG", "ACS"), symmetry=("ACG", "ACS"))
    j = solve(report, _content_report(LINE), _lineage())
    assert j.status == "DETERMINED_UP_TO_SYMMETRY"
    assert j.as_dict()["residual"]["physical_symmetry"] == ["ACG", "ACS"]


def test_no_carrier_is_an_answer_rather_than_a_gap() -> None:
    j = solve(_carrier_report(status="NOT_FOUND", frontier=()), _content_report(LINE), _lineage())
    assert j.status == "NO_CARRIER"


def test_a_content_run_that_did_not_measure_leaves_the_structure_unmeasured() -> None:
    blocked = _content_report(LINE, verdict="NOT_MEASURED", blockers=["retrieval does not separate the percepts"])
    j = solve(_carrier_report(), blocked, _lineage())
    assert j.status == "UNRESOLVED"
    assert j.unresolved()["structure"] == ["retrieval does not separate the percepts"]


def test_geometries_that_do_not_agree_are_reported_as_that() -> None:
    j = solve(_carrier_report(), _content_report(LINE, verdict="SEPARATE_STRUCTURES"), _lineage())
    assert j.status == "CONTENT_IS_NOT_ONE_STRUCTURE"


# ── the gauge ─────────────────────────────────────────────────────────────


def test_relabelling_the_classes_changes_nothing_identifiable() -> None:
    names = ["red", "green", "blue", "grey"]
    first = structure_term(_content_report(SQUARE[:3] + [(3.0, 0.5)], names))
    order = [2, 0, 3, 1]
    moved = structure_term(
        _content_report([(SQUARE[:3] + [(3.0, 0.5)])[i] for i in order], [names[i] for i in order])
    )
    assert first.invariant == pytest.approx(moved.invariant)
    assert first.orbits == moved.orbits


def test_a_rigid_structure_puts_every_class_in_an_orbit_of_its_own() -> None:
    term = structure_term(_content_report(LINE))
    assert term.rigid
    assert len(term.orbits) == 4


def test_classes_no_relation_separates_share_an_orbit() -> None:
    term = structure_term(_content_report(SQUARE))
    assert term.orbits == (("class_0", "class_1", "class_2", "class_3"),)
    assert not term.rigid


def test_a_difference_inside_the_floor_does_not_separate_classes_and_one_outside_it_does() -> None:
    names = ["a", "b", "c", "d"]
    matrix = np.array(
        [
            [0.0, 1.0, math.sqrt(2), 1.0],
            [1.0, 0.0, 1.0, math.sqrt(2)],
            [math.sqrt(2), 1.0, 0.0, 1.0],
            [1.0, math.sqrt(2), 1.0, 0.0],
        ]
    )
    inside = matrix.copy()
    inside[0, 1] = inside[1, 0] = 1.005
    outside = matrix.copy()
    outside[0, 1] = outside[1, 0] = 1.2
    assert orbits(names, inside, tolerance=0.01) == (("a", "b", "c", "d"),)
    assert orbits(names, outside, tolerance=0.01) == (("a", "b"), ("c", "d"))


# ── the lineage ───────────────────────────────────────────────────────────


def test_a_single_stage_is_not_a_lineage() -> None:
    lineage = Lineage()
    lineage.record("only", _person(1))
    assert solve(_carrier_report(), _content_report(LINE), lineage).lineage.status == "NOT_MEASURED"


def test_a_rewritten_stage_breaks_the_lineage_term() -> None:
    lineage = _lineage()
    lineage.stages["after"] = Stage(name="after", digest="0" * 32)
    j = solve(_carrier_report(), _content_report(LINE), lineage)
    assert j.lineage.status == "BROKEN"
    assert j.status == "UNRESOLVED"


# ── what it says about itself ─────────────────────────────────────────────


def test_the_report_names_every_postulate_and_judges_the_bridge_at_parity() -> None:
    report = solve(_carrier_report(), _content_report(LINE), _lineage()).as_dict()
    assert set(report["postulates"]) == set(POSTULATES) == {"P1", "P2", "P3", "P4", "P5", "P6"}
    # No campaign and no report-grounding run were given, so two grounds are
    # unmeasured and the bridge says which.
    assert report["bridge_status"]["phenomenal_bridge"] == "UNRESOLVED"
    unmeasured = {r["key"] for r in report["bridge_status"]["parity"] if r["status"] == "NOT_MEASURED"}
    assert unmeasured == {"markers", "reports"}
    assert "a person included" in report["bridge_status"]["residual"]
    assert "non_identifiability" in report["theorems"]


def test_no_status_it_can_return_says_conscious() -> None:
    cases = [
        solve(_carrier_report(), _content_report(LINE), _lineage()),
        solve(_carrier_report(authoritative=False), _content_report(LINE), _lineage()),
        solve(_carrier_report(status="NOT_FOUND", frontier=()), _content_report(LINE), _lineage()),
        solve(_carrier_report(), _content_report(LINE, verdict="SEPARATE_STRUCTURES"), _lineage()),
        solve(None, None, None),
    ]
    for j in cases:
        assert "CONSCIOUS" not in j.status.upper()
        assert "CONSCIOUS" not in j.bridge.upper()
