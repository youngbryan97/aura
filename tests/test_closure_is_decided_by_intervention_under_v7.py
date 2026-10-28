"""ISC-v7: causal closure of the core is what intervention shows, alone and all at once.

An observational test cannot tell a unit outside that causes K from one that
records an unrecorded part of K: the two can leave identical recordings. On
worlds with known answers the gate read a record of K's past open seven times
in seven and a cause acting through two outside signs together closed twice in
three. v7 decides line 3 and the carrier's closure by swapping what is outside
K*, one unit at a time and all at once, and reports the observational reading
beside it. v6 and every earlier version are unchanged.
"""

from __future__ import annotations

import pytest

from core.subject.battery import assemble, closed_by_intervention
from core.subject.bridge import carrier_term

pytestmark = pytest.mark.unit

CLEAN = {"tested": 16, "moved_k": 0, "moved_held": 0, "could_not_run": 0}


def _intervention(verdict: str = "CLOSED", joint: dict | None = CLEAN) -> dict:
    out = {"verdict": verdict, "moves_k_star": [] if verdict != "OPEN" else ["service.x"], "results": {}}
    if joint is not None:
        out["results"]["joint-periphery"] = dict(joint)
    return out


def test_the_validation_runs_form_of_the_joint_swap_is_read_too() -> None:
    assert closed_by_intervention({"verdict": "CLOSED", "joint_swap": dict(CLEAN)}) == (True, "")


def test_closed_alone_and_all_at_once_is_closed() -> None:
    assert closed_by_intervention(_intervention()) == (True, "")


@pytest.mark.parametrize(
    "reading, why",
    [
        (_intervention("OPEN"), "intervention read OPEN"),
        (_intervention(joint=None), "never swapped at once"),
        (_intervention(joint={**CLEAN, "tested": 0}), "never swapped at once"),
        (_intervention(joint={**CLEAN, "could_not_run": 2}), "could not run"),
        (_intervention(joint={**CLEAN, "moved_k": 1}), "moved it"),
        (_intervention(joint={**CLEAN, "moved_held": 3}), "moved it"),
        ({}, "intervention read nothing"),
    ],
)
def test_anything_short_of_that_is_not(reading: dict, why: str) -> None:
    closed, reason = closed_by_intervention(reading)
    assert not closed and why in reason


def test_the_v7_line_stands_in_for_line_3_and_v6_is_untouched() -> None:
    evidence = {
        "closure": {"closed": False, "verdict": "OPEN", "leak": 0.012},
        "interventional_closure": _intervention(),
    }
    verdict = assemble(evidence)
    [line] = verdict.v7_criteria
    assert line.key == "causal_closure_of_the_core" and line.passed
    assert line.detail["observational"] == "OPEN"
    v6 = [item for item in verdict.v6_lines() if item.key == "causal_closure_of_the_core"]
    assert v6 and not v6[0].passed, "v6's line 3 still needs both tests"
    v7 = [item for item in verdict.v7_lines() if item.key == "causal_closure_of_the_core"]
    assert len(v7) == 1 and v7[0].passed


def test_without_an_interventional_reading_there_is_no_v7() -> None:
    assert assemble({"closure": {"closed": True}}).v7_criteria == []


def _carrier(*, authoritative: bool = True, components=("ACDGIMNPSW",), status: str = "UNRESOLVED", peak: float = 0.03) -> dict:
    return {
        "authority": {"authoritative": authoritative, "blockers": [] if authoritative else ["clock.valve"]},
        "exclusion": {
            "status": status,
            "components": list(components),
            "whole_core_is_one_component": len(components) == 1,
            "candidates": [{"support": components[0], "recurrent": True, "peak_rate": peak, "closed": False,
                            "closure": "OPEN"}],
            "frontier": [],
        },
    }


def test_a_whole_core_closed_under_intervention_is_found_whatever_the_gate_read() -> None:
    term = carrier_term(_carrier(), _intervention(), closure_by_intervention=True)
    assert term.status == "FOUND" and term.carriers == ("ACDGIMNPSW",)


def test_without_v7_the_same_run_is_not_found() -> None:
    assert carrier_term(_carrier(), _intervention()).status != "FOUND"


@pytest.mark.parametrize(
    "carrier, reading, status",
    [
        (_carrier(authoritative=False), _intervention(), "UNRESOLVED"),
        (_carrier(), _intervention(joint=None), "UNRESOLVED"),
        (_carrier(), _intervention("OPEN"), "NOT_FOUND"),
        (_carrier(components=("ACDG", "IMNPSW")), _intervention(), "UNRESOLVED"),
        (_carrier(peak=0.0), _intervention(), "NOT_FOUND"),
    ],
)
def test_v7_still_needs_authority_one_component_and_irreducibility(carrier: dict, reading: dict, status: str) -> None:
    assert carrier_term(carrier, reading, closure_by_intervention=True).status == status
