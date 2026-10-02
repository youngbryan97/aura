from copy import deepcopy
from pathlib import Path

import pytest

from tools.run_semantic_grounded_adjudication_handoff import adjudication_arguments, wait_for_evaluation


def supervised():
    return {"plan_sha256": "a" * 64, "command_sha256": "b" * 64, "command": [
        "/absolute/python", "-u", "/frozen/tools/evaluate_semantic_grounded_native.py",
        "--parent", "/evidence/parent.json", "--source-report", "/evidence/source.json",
        "--directory", "/evidence/native", "--output", "/evidence/development.json",
        "--bundle", "a=/evidence/features/a", "--bundle", "b=/evidence/features/b", "--archive-decodes"]}


def arguments(value):
    return adjudication_arguments(value, report=Path("/evidence/development.json"),
        native_directory=Path("/evidence/native"), output=Path("/new/adjudication.json"))


def terminal(*, code=0):
    return {"terminal": True, "child_state": "dead", "receipt": {
        "plan_sha256": "a" * 64, "command_sha256": "b" * 64, "receipt_sha256": "c" * 64,
        "containment_verified": True, "returncode": code, "timed_out": False}}


def test_arguments_reuse_original_source_custody_without_another_model_command():
    values = arguments(supervised())
    assert values == ["--report", "/evidence/development.json", "--directory", "/evidence/native",
        "--output", "/new/adjudication.json", "--parent", "/evidence/parent.json",
        "--source-report", "/evidence/source.json", "--bundle", "a=/evidence/features/a",
        "--bundle", "b=/evidence/features/b"]
    assert not any("train" in value or "evaluate_semantic" in value for value in values)


@pytest.mark.parametrize("fault", ["wrong_command", "no_archive", "profile", "repeated_input", "relative",
    "different_report", "different_native", "duplicate_bundle", "no_bundles", "incomplete_bundle"])
def test_arguments_refuse_missing_replaced_or_unbound_evaluation_inputs(fault):
    value = supervised()
    command = value["command"]
    if fault == "wrong_command":
        command[2] = "/frozen/tools/fit_semantic_grounded_binding.py"
    elif fault == "no_archive":
        command.remove("--archive-decodes")
    elif fault == "profile":
        command += ["--profile-source-id", "d" * 64]
    elif fault == "repeated_input":
        command += ["--parent", "/evidence/parent.json"]
    elif fault == "relative":
        command[command.index("--parent") + 1] = "parent.json"
    elif fault in {"different_report", "different_native"}:
        flag = "--output" if fault == "different_report" else "--directory"
        command[command.index(flag) + 1] = "/replacement"
    elif fault == "duplicate_bundle":
        command += ["--bundle", "a=/replacement"]
    elif fault == "no_bundles":
        value["command"] = [x for x in command if x != "--bundle" and "=/evidence" not in x]
    else:
        command += ["--bundle"]
    with pytest.raises(ValueError):
        arguments(value)


@pytest.mark.parametrize("code", [0, 2])
def test_wait_accepts_clean_terminal_for_a_passed_or_negative_candidate(code):
    value = terminal(code=code)
    assert wait_for_evaluation(Path("/supervisor"), supervised(), timeout=1., inspect=lambda _: value) == value["receipt"]


@pytest.mark.parametrize("fault", ["plan", "command", "containment", "timeout", "bad_exit", "bool_exit", "alive_child"])
def test_wait_rejects_indeterminate_or_uncontained_terminal(fault):
    value = deepcopy(terminal())
    if fault in {"plan", "command"}:
        value["receipt"][fault + "_sha256"] = "d" * 64
    elif fault == "containment":
        value["receipt"]["containment_verified"] = False
    elif fault == "timeout":
        value["receipt"]["timed_out"] = True
    elif fault in {"bad_exit", "bool_exit"}:
        value["receipt"]["returncode"] = 1 if fault == "bad_exit" else False
    else:
        value["child_state"] = "alive"
    with pytest.raises(ValueError):
        wait_for_evaluation(Path("/supervisor"), supervised(), timeout=1., inspect=lambda _: value)


def test_wait_is_bounded_and_never_restarts_a_dead_prerequisite():
    now, sleeps = [0.], []
    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    running = {"terminal": False, "supervisor_alive": True, "completion_indeterminate": False}
    with pytest.raises(TimeoutError):
        wait_for_evaluation(Path("/supervisor"), supervised(), timeout=65., inspect=lambda _: running,
            clock=lambda: now[0], sleep=sleep)
    assert sleeps == [60., 5.]
    with pytest.raises(ValueError, match="no duplicate"):
        wait_for_evaluation(Path("/supervisor"), supervised(), timeout=65.,
            inspect=lambda _: {**running, "supervisor_alive": False})


@pytest.mark.parametrize("timeout", [0., -1., float("nan"), float("inf"), 8001.])
def test_wait_refuses_an_unbounded_or_invalid_allowance(timeout):
    with pytest.raises(ValueError):
        wait_for_evaluation(Path("/supervisor"), supervised(), timeout=timeout)


@pytest.mark.parametrize("advance,mismatch", [(True, False), (False, False), (False, True)])
def test_completed_handoff_verifies_and_regrades_once_without_model_or_decode(monkeypatch, tmp_path, advance, mismatch):
    import json
    import tools.adjudicate_semantic_grounded_development as grading
    import tools.run_detached_step as detached
    import tools.run_semantic_grounded_adjudication_handoff as handoff
    import tools.verify_semantic_grounded_evaluation as verification
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway
    from tools.semantic_grounded_development_archive import digest

    supervisor = tmp_path / "supervisor"
    supervisor.mkdir()
    report, native, directory = tmp_path / "development.json", tmp_path / "native", tmp_path / "handoff"
    plan = supervised()
    command = plan["command"]
    command[command.index("--output") + 1] = str(report)
    command[command.index("--directory") + 1] = str(native)
    (supervisor / detached.PLAN_FILE).write_text(json.dumps(plan))
    evaluation_plan = {"schema": "fixture"}
    evaluation_plan["plan_sha256"] = digest(evaluation_plan)
    report.with_suffix(".plan.json").write_text(json.dumps(evaluation_plan))
    monkeypatch.setattr(detached, "_verify_plan", lambda _plan, _path: None)
    measured_terminal = terminal(code=0 if advance or mismatch else 2)["receipt"]
    monkeypatch.setattr(handoff, "wait_for_evaluation", lambda *_args, **_kwargs: measured_terminal)
    calls = []
    def verify(report_path, native_path):
        calls.append(("verify", report_path, native_path))
        return {"receipt_sha256": "d" * 64, "advance_development": advance}
    def adjudicate(arguments):
        calls.append(("adjudicate", arguments))
        output = Path(arguments[arguments.index("--output") + 1])
        value = {"metrics": {"joint_native": {"program_equivalent": 1}}}
        value["receipt_sha256"] = digest(value)
        with local_internal_governed_scope("grounded_handoff_test", domain="file_write"):
            get_file_write_gateway().write_bytes_if_absent(output, json.dumps(value).encode(), source="grounded_handoff_test")
        return 0
    monkeypatch.setattr(verification, "verify", verify)
    monkeypatch.setattr(grading, "main", adjudicate)
    arguments = ["--evaluation-supervisor", str(supervisor), "--evaluation-supervisor-plan-sha256", plan["plan_sha256"],
        "--evaluation-plan-sha256", evaluation_plan["plan_sha256"], "--report", str(report),
        "--native-directory", str(native), "--directory", str(directory)]
    if mismatch:
        with pytest.raises(ValueError, match="exit disagrees"):
            handoff.main(arguments)
        assert [call[0] for call in calls] == ["verify"]
        assert not (directory / "completion.json").exists()
    else:
        assert handoff.main(arguments) == 0
        assert [call[0] for call in calls] == ["verify", "adjudicate"]
        completed = json.loads((directory / "completion.json").read_bytes())
        assert completed["advance_development"] is advance
        assert completed["g03_complete"] is completed["backbone_loaded"] is completed["serving_authority"] is False
        assert completed["receipt_sha256"] == digest({k: v for k, v in completed.items() if k != "receipt_sha256"})
