"""No launch command is printed while anything blocks the launch.

A command that is only correct under conditions the reader has to remember is
how a campaign gets started against a drifted tree. So the command and the
blocker list are mutually exclusive, and the split between environmental and
package blockers is asserted because it decides whether "wait" is a valid
response.
"""
from __future__ import annotations

import json

import pytest

from tools import report_27b_launch_readiness as readiness


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.setattr(readiness.preflight, "check", lambda bundle: [])
    monkeypatch.setattr(readiness, "_package_findings", list)


def test_a_clean_tree_emits_exactly_one_command(clean):
    report = readiness.build()
    assert report["ready_to_launch"] is True
    command = report["launch_command"]
    assert command and command.count("run_unified_intrinsic_resident_campaign.py") == 1
    assert "--detached" in command
    assert "caffeinate" in command


def test_no_command_is_emitted_while_anything_blocks(monkeypatch):
    monkeypatch.setattr(
        readiness.preflight,
        "check",
        lambda bundle: [{"kind": "source_drifted", "detail": "x"}],
    )
    monkeypatch.setattr(readiness, "_package_findings", list)
    report = readiness.build()
    assert report["ready_to_launch"] is False
    assert report["launch_command"] is None


def test_every_blocker_carries_an_action(monkeypatch):
    monkeypatch.setattr(
        readiness.preflight,
        "check",
        lambda bundle: [{"kind": k, "detail": "x"} for k in readiness.REMEDIES],
    )
    monkeypatch.setattr(readiness, "_package_findings", list)
    for blocker in readiness.build()["blockers"]:
        assert blocker["remedy"] != "investigate", blocker["kind"]


def test_environmental_and_package_blockers_are_separated(monkeypatch):
    monkeypatch.setattr(
        readiness.preflight,
        "check",
        lambda bundle: [
            {"kind": "insufficient_ram", "detail": "x"},
            {"kind": "source_drifted", "detail": "y"},
        ],
    )
    monkeypatch.setattr(readiness, "_package_findings", list)
    report = readiness.build()
    assert report["blocker_counts"] == {"environmental": 1, "package": 1}
    classes = {b["kind"]: b["class"] for b in report["blockers"]}
    assert classes["insufficient_ram"] == "environmental"
    assert classes["source_drifted"] == "package"


def test_a_package_carrying_a_verdict_before_the_run_blocks(monkeypatch, tmp_path):
    target = tmp_path / "package.json"
    target.write_text(json.dumps({"verdict": "BOUNDED_WOW_SIGNAL"}))
    monkeypatch.setattr(readiness, "PACKAGE", target)
    monkeypatch.setattr(readiness, "REPO_ROOT", tmp_path.parent)
    findings = readiness._package_findings()
    assert findings and findings[0]["kind"] == "package_carries_a_verdict"


def test_an_unmaterialized_package_blocks(monkeypatch, tmp_path):
    monkeypatch.setattr(readiness, "PACKAGE", tmp_path / "nope.json")
    monkeypatch.setattr(readiness, "REPO_ROOT", tmp_path.parent)
    findings = readiness._package_findings()
    assert findings and findings[0]["kind"] == "package_not_materialized"


def test_the_workload_estimate_names_its_basis_and_its_gap(clean):
    workload = readiness.build()["measured_workload"]
    assert workload["decode_seconds"] == 4814.53
    # Training wall time appears in no retained receipt, so it must stay None
    # rather than becoming a number somebody later cites as measured.
    assert workload["training_seconds"] is None
    assert "unmeasured" in workload["training_note"]


def test_the_model_active_stages_are_the_contiguous_five(clean):
    assert readiness.build()["model_active_stages"] == [
        "calibration",
        "training",
        "canary",
        "lesion_arms",
        "export",
    ]


def test_the_receipt_records_the_source_commit(clean):
    assert len(readiness.build()["source_commit"]) == 40
