import json
from copy import deepcopy
from pathlib import Path

import pytest

from tools.run_semantic_grounded_handoff import check_terminal, evaluation_arguments


def supervised():
    return {"plan_sha256": "a" * 64, "command_sha256": "b" * 64,
        "command": ["/python", "-u", "/frozen/tools/fit_semantic_grounded_binding.py", "--native",
            "--parent", "/evidence/parent.json", "--source-report", "/evidence/source.json",
            "--folds", "/evidence/folds.json", "--bank", "/evidence/bank",
            "--directory", "/evidence/native", "--authority-key-file", "/private/key",
            "--bundle", "first=/evidence/first", "--bundle", "second=/evidence/second",
            "--steps", "512", "--resume-if-available"]}


def test_handoff_reuses_exact_fit_inputs_without_launching_another_fit():
    result = evaluation_arguments(supervised(), output=Path("/screen/development.json"), search_seconds=30.)
    assert result[result.index("--directory") + 1] == "/evidence/native"
    assert result.count("--bundle") == 2
    assert "--steps" not in result and "--resume-if-available" not in result
    assert "fit_semantic_grounded_binding.py" not in result


@pytest.mark.parametrize("fault", ["relative", "duplicate", "bundle", "prepare", "empty", "tool", "time"])
def test_handoff_rejects_changed_or_incomplete_command(fault):
    value = deepcopy(supervised())
    command = value["command"]
    seconds = 30.
    if fault == "relative":
        command[command.index("--parent") + 1] = "parent.json"
    elif fault == "duplicate":
        command += ["--parent", "/other/parent.json"]
    elif fault == "bundle":
        command += ["--bundle", "first=/other/first"]
    elif fault == "prepare":
        command += ["--prepare-only"]
    elif fault == "empty":
        command = [part for part in command if part != "--bundle" and "=/evidence/" not in part]
        value["command"] = command
    elif fault == "tool":
        command[2] = "/frozen/tools/another.py"
    else:
        seconds = float("nan")
    with pytest.raises(ValueError):
        evaluation_arguments(value, output=Path("/screen/development.json"), search_seconds=seconds)


def test_terminal_must_belong_to_bound_successful_contained_fit():
    value = supervised()
    receipt = {"plan_sha256": value["plan_sha256"], "command_sha256": value["command_sha256"],
        "passed": True, "containment_verified": True, "returncode": 0, "timed_out": False}
    check_terminal(receipt, value)
    for field, replacement in (("plan_sha256", "c" * 64), ("command_sha256", "d" * 64),
            ("passed", False), ("containment_verified", False), ("returncode", 2), ("timed_out", True)):
        with pytest.raises(ValueError, match="different or failed"):
            check_terminal({**receipt, field: replacement}, value)


@pytest.mark.parametrize("failure", ["trainer_failed", "initial_only", "other_plan"])
def test_handoff_prerequisites_fail_before_model_evaluation(tmp_path, monkeypatch, failure):
    from tools import (
        evaluate_semantic_grounded_native,
        run_detached_step,
        run_semantic_grounded_handoff,
        run_semantic_native_micro_stages,
        verify_semantic_grounded_fit,
    )

    value = supervised()
    supervisor = tmp_path / "supervisor"
    supervisor.mkdir()
    (supervisor / run_detached_step.PLAN_FILE).write_text(json.dumps(value))
    monkeypatch.setattr(run_detached_step, "_verify_plan", lambda *args: None)
    calls = []
    monkeypatch.setattr(evaluate_semantic_grounded_native, "main", lambda args: calls.append(args))
    def wait(*args, **kwargs):
        if failure == "trainer_failed":
            raise ValueError("trainer failed")
        return {"plan_sha256": value["plan_sha256"], "command_sha256": value["command_sha256"],
            "passed": True, "containment_verified": True, "returncode": 0, "timed_out": False}
    monkeypatch.setattr(run_semantic_native_micro_stages, "wait_for_fit", wait)
    monkeypatch.setattr(verify_semantic_grounded_fit, "verify", lambda directory: {"learned_checkpoint_selected": False})
    expected = "d" * 64 if failure == "other_plan" else value["plan_sha256"]
    monkeypatch.setattr("sys.argv", ["handoff", "--fit-supervisor", str(supervisor),
        "--fit-supervisor-plan-sha256", expected, "--directory", str(tmp_path / "handoff")])
    with pytest.raises(ValueError):
        run_semantic_grounded_handoff.main()
    assert calls == []
