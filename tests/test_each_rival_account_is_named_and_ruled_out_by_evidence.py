"""Each rival account of a reading is named with the observation that tells it apart, and decided from the files.

core/subject/alternatives.py answers the construct-validity question by rival:
a rival is EXCLUDED only when her own evidence contains the observation it
cannot produce, PARTLY when only the null architectures rule it out, and OPEN
when the deciding run is still owed. These pin the rules that matter most:
nothing is excluded without its evidence, a control that tracks keeps its
rival open, and the per-ground calibration names the simple nulls that pass a
ground alone.
"""

from __future__ import annotations

import pytest

from core.subject.alternatives import RIVALS, newest_lines, null_calibration, rule_out

pytestmark = pytest.mark.unit


def _status(evidence: dict, name: str) -> str:
    return next(row["status"] for row in rule_out(evidence)["rivals"] if row["rival"] == name)


def test_with_no_evidence_nothing_is_excluded() -> None:
    read = rule_out({})
    assert read["excluded"] == []
    assert set(read["open"]) | set(read["partly"]) == {r.name for r in RIVALS}


def test_a_newer_version_replaces_a_line_of_the_same_name() -> None:
    verdict = {"verdict": {
        "criteria": [{"criterion": "causal_closure_of_the_core", "passed": False, "value": "LEAK"}],
        "v7_criteria": [{"criterion": "causal_closure_of_the_core", "passed": True, "value": "CLOSED"}],
    }}
    assert newest_lines(verdict)["causal_closure_of_the_core"] == {"passed": True, "value": "CLOSED", "version": "v7_criteria"}


GROUNDED = {"measured": True, "holds": True, "conditions": {"readings_of_herself": "withheld"}}


def test_the_text_route_is_excluded_only_when_steering_off_stops_the_tracking() -> None:
    assert _status({"reports": GROUNDED}, "report from text") == "PARTLY"
    stopped = {"reports": GROUNDED, "controls": {"steering-off": {"measured": True, "holds": False}}}
    assert _status(stopped, "report from text") == "EXCLUDED"
    still = {"reports": GROUNDED, "controls": {"steering-off": {"measured": True, "holds": True}}}
    assert _status(still, "report from text") == "OPEN"


def test_the_persona_is_excluded_only_when_the_tuned_model_stops_tracking_with_steering_off() -> None:
    assert _status({"controls": {"untuned": {"measured": True, "holds": False}}}, "persona") == "OPEN"
    stopped = {"controls": {"steering-off": {"measured": True, "holds": False}}}
    assert _status(stopped, "persona") == "EXCLUDED"


def test_a_tuned_model_that_tracks_without_steering_keeps_the_persona_open_whatever_an_untuned_one_does() -> None:
    for untuned in (None, {"measured": True, "holds": False}, {"measured": True, "holds": True}):
        controls = {"steering-off": {"measured": True, "holds": True}}
        if untuned is not None:
            controls["untuned"] = untuned
        assert _status({"controls": controls}, "persona") == "OPEN"


def test_her_domains_cut_apart_decide_the_fragmented_rival() -> None:
    nulls = {"conjunction_v3": {"independent": False, "high_dimensional_independent": False, "frozen_slow": False}}
    assert _status({"nulls": nulls}, "fragmented") == "PARTLY"
    apart = {"nulls": nulls, "disconnected": {"verdict": {"as_predicted": True, "why": "apart"}}}
    assert _status(apart, "fragmented") == "EXCLUDED"
    one = {"nulls": nulls, "disconnected": {"verdict": {"as_predicted": False, "why": "phi_do above its error"}}}
    assert _status(one, "fragmented") == "OPEN"


def test_only_a_confirmatory_seed_decides_chance_or_a_chosen_rule() -> None:
    assert _status({}, "chance or a chosen rule") == "OPEN"
    assert _status({"confirmation": {"read": True, "passed": True}}, "chance or a chosen rule") == "EXCLUDED"
    assert _status({"confirmation": {"read": True, "passed": False}}, "chance or a chosen rule") == "OPEN"


def test_the_language_model_rival_needs_a_carrier_found_without_it() -> None:
    assert _status({"carrier": {"scope": "substrate_only", "carrier": {"status": "FOUND"}}}, "language model") == "EXCLUDED"
    assert _status({"carrier": {"scope": "whole", "carrier": {"status": "FOUND"}}}, "language model") == "OPEN"


def test_calibration_names_the_simple_nulls_that_pass_a_ground_alone() -> None:
    row = {"kind": "architecture"}
    nulls = {"detail": {
        "recurrent": {**row, "closed": True, "one_component": True, "vertex_connectivity": 3, "reentry": True},
        "ring": {**row, "closed": True, "one_component": True, "vertex_connectivity": 1, "reentry": True},
        "star": {**row, "closed": False, "one_component": False, "vertex_connectivity": 0, "reentry": False},
    }, "conjunction_v3": {"recurrent": True, "ring": False, "star": False}}
    read = null_calibration(nulls)
    graph = read["grounds"][0]
    assert graph["passing"] == ["ring"] and graph["reference_passes"] is True
    robust = read["grounds"][1]
    assert robust["passing"] == [] and robust["pass_rate_upper_95"] == 1.5
    assert read["conjunction_passing"] == [] and read["conjunction_reference_passes"] is True
