"""A control of the reports ground is declared, runs without what it removes, and is never read as the ground.

Every outside read of seed 97 asked the same thing of the reports ground: would
her rating still follow her state with her steering detached, or with a model
that was never tuned to speak as her? Those are runs of the same harness with
one thing taken away. These pin how such a run is declared, that it refuses to
go on with the thing it removes still in place, and that the bridge reads it
beside her run and never as the ground.
"""

from __future__ import annotations

import pytest

from core.subject.bridge import CarrierTerm, JStar, LineageTerm, StructureTerm
from tools.run_report_grounding import CONTROLS, control_environment, steering_refusal

pytestmark = pytest.mark.unit

WITHHELD = {"readings_of_herself": "withheld", "steering": "on"}


def test_her_own_run_pins_nothing() -> None:
    assert control_environment(["--whole", "--anchors", "80"]) == {}


def test_steering_off_detaches_her_steering_and_keeps_her_cortex() -> None:
    assert control_environment(["--whole", "--control", "steering-off"]) == {"AURA_DISABLE_AFFECTIVE_STEERING": "1"}


def test_untuned_puts_the_named_model_in_both_of_her_lanes() -> None:
    pins = control_environment(["--whole", "--control", "untuned", "--model", "/models/base"])
    assert pins == {
        "AURA_DISABLE_AFFECTIVE_STEERING": "1",
        "AURA_LLM__MLX_MODEL_PATH": "/models/base",
        "AURA_LLM__MLX_DEEP_MODEL_PATH": "/models/base",
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["--whole", "--control", "untuned"],
        ["--whole", "--model", "/models/base"],
        ["--whole", "--control", "steering-off", "--model", "/models/base"],
        ["--whole", "--control", "louder"],
        ["--whole", "--control"],
    ],
)
def test_a_control_that_is_not_declared_whole_is_refused(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        control_environment(argv)


def test_every_declared_control_says_what_it_asks() -> None:
    assert set(CONTROLS) == {"steering-off", "untuned"}
    assert all(question.endswith("?") for question in CONTROLS.values())


def test_her_run_needs_her_steering_attached() -> None:
    assert steering_refusal(True, None, {"active": None})
    assert steering_refusal(True, None, {"active": False})
    assert steering_refusal(True, None, {"active": True}) is None


def test_a_control_needs_her_steering_detached() -> None:
    for control in CONTROLS:
        assert steering_refusal(True, control, {"active": True})
        assert steering_refusal(True, control, {"active": None}) is None
        assert steering_refusal(True, control, {"active": False}) is None


def test_the_stand_in_organ_is_never_refused_on_steering() -> None:
    assert steering_refusal(False, None, {}) is None
    assert steering_refusal(False, "steering-off", {"active": True}) is None


def _reports_reading(reports: dict) -> tuple[str, str]:
    j = JStar(
        carrier=CarrierTerm("FOUND", carriers=("ACDGIMNPSW",)),
        structure=StructureTerm("IDENTIFIED", classes=("a", "b", "c")),
        lineage=LineageTerm("RECORDED"),
        reports=reports,
    )
    row = next(p for p in j.parity() if p.key == "reports")
    return row.status, row.why


def test_the_bridge_never_reads_a_control_as_the_ground() -> None:
    held = {"measured": True, "holds": True, "why": "tracked", "conditions": WITHHELD}
    assert _reports_reading(held)[0] == "HOLDS"
    for control in CONTROLS:
        status, why = _reports_reading({**held, "conditions": {**WITHHELD, "control": control}})
        assert status == "NOT_MEASURED"
        assert control in why
