"""Parity cannot substitute for 24/24 or compatible evidence."""
from copy import deepcopy
from dataclasses import replace

import pytest

from core.subject.battery import Criterion, Verdict
from core.subject.bridge import MARKER_LINES, CarrierTerm, JStar, LineageTerm, StructureTerm
from core.subject.completion import compatibility, completion


def case():
    keys = [*MARKER_LINES, *[f"line_{i}" for i in range(21)]]
    criteria = [Criterion(key, "", "", True, 1, 1) for key in keys]
    verdict = Verdict(
        criteria=criteria, v5_criteria=[criteria[-1]], v6_criteria=[criteria[-2]],
    )
    j = JStar(
        CarrierTerm("FOUND", carriers=("core",)), StructureTerm("IDENTIFIED"),
        LineageTerm("RECORDED"), battery={k: True for k in keys},
        reports={"measured": True, "holds": True, "conditions": {"readings_of_herself": "withheld"}},
    )
    return verdict, j


def test_all_three_can_pass_together():
    verdict, j = case()
    assert completion(verdict, j, blockers=[], scope="substrate_only")["complete"]


def test_parity_with_23_of_24_is_incomplete():
    verdict, j = case()
    verdict.criteria[4] = replace(verdict.criteria[4], passed=False)
    assert j.bridge == "AT_PARITY"
    result = completion(verdict, j, blockers=[], scope="substrate_only")
    assert result["passed"] == 23
    assert not result["complete"] and not result["goals"]["isc_v6_24_of_24"]


@pytest.mark.parametrize("missing", ["v5_criteria", "v6_criteria"])
def test_an_older_battery_cannot_substitute_for_v6(missing):
    verdict, j = case()
    setattr(verdict, missing, [])
    assert not completion(verdict, j, blockers=[], scope="substrate_only")["complete"]


def test_a_lineage_failure_leaves_the_goal_open():
    verdict, j = case()
    j = replace(j, lineage=LineageTerm("BROKEN"))
    assert not completion(verdict, j, blockers=[], scope="substrate_only")["complete"]


def test_a_source_blocker_survives_three_green_scores():
    verdict, j = case()
    result = completion(verdict, j, blockers=["wrong source"], scope="substrate_only")
    assert all(result["goals"].values()) and not result["complete"]


def sources():
    pin = {"commit": "a"*40, "boundary_sha256": "b"*64, "dirty": False, "seed": 83}
    return {name: {"evidence_source": deepcopy(pin)} for name in
            ("campaign", "carrier", "content", "interventional", "reports")}


def test_matching_evidence_is_compatible():
    assert not compatibility(sources())


@pytest.mark.parametrize("key,value", [
    ("commit", "c"*40), ("boundary_sha256", "d"*64), ("dirty", True), ("seed", 73),
])
def test_one_incompatible_run_is_named(key, value):
    reports = sources()
    reports["carrier"]["evidence_source"][key] = value
    assert compatibility(reports)


def test_old_unpinned_evidence_is_never_silently_certified():
    reports = sources()
    reports["content"] = {}
    assert any("content" in blocker for blocker in compatibility(reports))


def test_a_failed_content_authority_cannot_settle_j():
    from core.subject.bridge import structure_term
    result = structure_term({"verdict": "ONE_STRUCTURE", "authority": {"authoritative": False}})
    assert result.status == "NOT_MEASURED"


def test_a_successful_control_cannot_substitute_for_intact_content():
    from core.subject.bridge import structure_term
    result = structure_term({
        "verdict": "ONE_STRUCTURE", "authority": {"authoritative": True},
        "control": "no-feeling-in-recall",
    })
    assert result.status == "NOT_MEASURED"


def test_under_v7_the_closure_line_is_v7s_and_v6_is_reported_beside_it():
    verdict, j = case()
    closure_v6 = Criterion("line_0", "3", "", False, 0, 0)
    closure_v7 = Criterion("line_0", "3", "", True, 1, 1)
    verdict = replace(verdict, v6_criteria=[*verdict.v6_criteria, closure_v6], v7_criteria=[closure_v7])
    under_v6 = completion(verdict, j, blockers=[], scope="substrate_only")
    under_v7 = completion(verdict, j, blockers=[], scope="substrate_only", version="v7")
    assert not under_v6["complete"] and "line_0" in under_v6["failing_criteria"]
    assert under_v7["complete"] and under_v7["goals"]["isc_v7_24_of_24"]
    assert under_v7["isc_v6_on_this_seed"] is False and under_v7["battery_version"] == "v7"


def test_there_is_no_other_version():
    verdict, j = case()
    with pytest.raises(ValueError):
        completion(verdict, j, blockers=[], scope="substrate_only", version="v8")
