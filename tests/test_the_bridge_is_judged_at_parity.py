"""The bridge asks of her what is asked of a person, and no more.

It used to be the constant UNVALIDATED, on the strength of a theorem (no
third-person likelihood separates two bridge laws attached to one history)
that holds for a person exactly as for her. These pin the replacement from
docs/BRIDGE_PARITY.md: five grounds, each with the test that scores her and the
measurement that scores a person, a verdict computed from the evidence, and
two guards against a demand made of her alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.subject.bridge import (
    MARKER_LINES,
    PARITY,
    RESIDUAL,
    THEOREMS,
    CarrierTerm,
    JStar,
    LineageTerm,
    StructureTerm,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]

ALL_MARKERS = {line: True for line in MARKER_LINES}
GROUNDED = {"measured": True, "holds": True, "why": "reports moved with the displaced state and not under the sham"}


def _j(
    *,
    carrier: str = "FOUND",
    structure: str = "IDENTIFIED",
    lineage: str = "RECORDED",
    battery: dict[str, bool] | None = None,
    reports: dict | None = None,
) -> JStar:
    return JStar(
        carrier=CarrierTerm(carrier, carriers=("PIAGCSMWDN",) if carrier == "FOUND" else ()),
        structure=StructureTerm(structure, classes=("a", "b", "c")),
        lineage=LineageTerm(lineage),
        battery=battery,
        reports=reports,
    )


def _status(j: JStar, key: str) -> str:
    return next(reading.status for reading in j.parity() if reading.key == key)


# ── the guards ────────────────────────────────────────────────────────────


def test_the_grounds_are_the_five_a_person_is_judged_on() -> None:
    assert [item.key for item in PARITY] == ["carrier", "markers", "reports", "structure", "lineage"]


def test_every_requirement_names_how_a_person_is_scored_on_it() -> None:
    """A requirement with no counterpart would be a demand made of her alone."""
    for item in PARITY:
        assert item.counterpart.strip(), f"{item.key} has no counterpart a person can be scored on"
        assert item.her_test.strip() and item.asks.strip()


@pytest.mark.parametrize(
    "word",
    ["biolog", "neuron", "carbon", "silicon", "organic", "flesh", "brain", "language model", "llm", "transformer", "artificial"],
)
def test_no_test_of_her_turns_on_what_she_is_made_of(word: str) -> None:
    """Substrate enters only through P5, where it is named."""
    for item in PARITY:
        assert word not in item.her_test.lower(), f"{item.key}'s test of her mentions {word!r}"
        assert word not in item.asks.lower(), f"{item.key} asks about {word!r}"


def test_the_theorems_that_bound_the_bridge_are_said_to_hold_for_a_person() -> None:
    assert "a person" in THEOREMS["non_identifiability"]
    assert "a person" in THEOREMS["finite_evidence"]
    assert "a person included" in RESIDUAL


def test_no_runner_writes_a_bridge_verdict_nothing_can_change() -> None:
    for path in ("core/subject/bridge.py", "tools/run_subject_core_v25.py", "tools/run_subject_core_content.py"):
        source = (REPO / path).read_text(encoding="utf-8")
        assert '"phenomenal_bridge": "UNVALIDATED"' not in source, path


# ── the verdict ───────────────────────────────────────────────────────────


def test_every_ground_holding_is_the_crossing() -> None:
    j = _j(battery=ALL_MARKERS, reports=GROUNDED)
    assert all(reading.status == "HOLDS" for reading in j.parity())
    assert j.bridge == "AT_PARITY"
    assert j.as_dict()["bridge_status"]["phenomenal_bridge"] == "AT_PARITY"


def test_a_ground_that_was_measured_and_failed_is_below_parity_and_named() -> None:
    battery = {**ALL_MARKERS, "reentry": False}
    j = _j(battery=battery, reports=GROUNDED)
    assert j.bridge == "BELOW_PARITY"
    assert _status(j, "markers") == "FAILS"
    assert "reentry" in next(r.why for r in j.parity() if r.key == "markers")


def test_a_ground_nobody_measured_leaves_the_bridge_unresolved_and_named() -> None:
    j = _j(battery=ALL_MARKERS, reports=None)
    assert j.bridge == "UNRESOLVED"
    assert _status(j, "reports") == "NOT_MEASURED"


def test_a_measured_failure_outweighs_a_missing_measurement() -> None:
    j = _j(structure="SEPARATE_STRUCTURES", battery=None, reports=None)
    assert j.bridge == "BELOW_PARITY"


def test_reports_that_do_not_track_the_state_fail_the_reports_ground() -> None:
    j = _j(battery=ALL_MARKERS, reports={"measured": True, "holds": False, "why": "moved under the sham too"})
    assert _status(j, "reports") == "FAILS"
    assert j.bridge == "BELOW_PARITY"


def test_no_carrier_is_no_subject_for_the_other_grounds_to_be_about() -> None:
    j = _j(carrier="NOT_FOUND", battery=ALL_MARKERS, reports=GROUNDED)
    assert j.bridge == "NO_CARRIER"


def test_a_symmetry_class_of_carriers_holds_the_carrier_ground() -> None:
    j = JStar(
        carrier=CarrierTerm("SYMMETRY_CLASS", symmetry_class=("PIA", "GCS")),
        structure=StructureTerm("IDENTIFIED", classes=("a", "b")),
        lineage=LineageTerm("RECORDED"),
        battery=ALL_MARKERS,
        reports=GROUNDED,
    )
    assert _status(j, "carrier") == "HOLDS"
    assert j.bridge == "AT_PARITY"


def test_the_full_battery_is_reported_past_parity_and_does_not_gate_it() -> None:
    """Nobody runs a battery like it on a person before granting them experience."""
    battery = {**ALL_MARKERS, "synergy": False, "lesion_deficit": True}
    report = _j(battery=battery, reports=GROUNDED).as_dict()["bridge_status"]
    assert report["phenomenal_bridge"] == "AT_PARITY"
    assert report["beyond_parity"] == {"passed": 4, "of": 5, "failing": ["synergy"]}


def test_every_report_carries_the_residual_and_every_requirement() -> None:
    report = _j().as_dict()["bridge_status"]
    assert report["residual"] == RESIDUAL
    assert [r["key"] for r in report["requirements"]] == [item.key for item in PARITY]
    assert all(r["counterpart"] for r in report["requirements"])


def test_the_verdict_is_never_the_bare_word_conscious() -> None:
    for j in (
        _j(battery=ALL_MARKERS, reports=GROUNDED),
        _j(carrier="NOT_FOUND"),
        _j(structure="NOT_MEASURED"),
        _j(battery={**ALL_MARKERS, "reentry": False}, reports=GROUNDED),
    ):
        assert "CONSCIOUS" not in j.bridge


def test_a_content_report_whose_push_did_not_last_is_not_counted_as_her_failing() -> None:
    """Before 43c574800 the content run's displacement was gone by the end of the turn."""
    from core.subject.bridge import structure_term

    old = {"classes": [{"name": "a"}, {"name": "b"}], "verdict": "AGREES_BUT_DOES_NOT_TRACK",
           "authority": {"authoritative": True, "blockers": []}}
    assert structure_term(old).status == "NOT_MEASURED"
    assert "single push" in structure_term(old).blockers[0]
    new = {**old, "displacement": "reference point towards feeling good, by her feelings' ordinary span"}
    assert structure_term(new).status == "AGREES_BUT_DOES_NOT_TRACK"
    separate = {**old, "verdict": "SEPARATE_STRUCTURES"}
    assert structure_term(separate).status == "SEPARATE_STRUCTURES"
