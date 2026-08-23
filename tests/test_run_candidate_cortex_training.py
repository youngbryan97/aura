from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools import run_candidate_cortex_training as controller


def _plan(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    source = tmp_path / "frozen-source"
    tools = source / "tools"
    tools.mkdir(parents=True)
    (source / ".git").write_text("gitdir: frozen\n", encoding="utf-8")
    admission = tools / "adjudicate_candidate_cortex_checkpoint.py"
    admission.write_text("# frozen\n", encoding="utf-8")
    run_root = source / "artifacts" / "run"
    run_root.mkdir(parents=True)
    journal = run_root / "training_journal.jsonl"
    journal.write_text("", encoding="utf-8")
    key = tmp_path / "journal.key"
    key.write_bytes(b"k" * 64)
    plan = {
        "run_id": "bound-run",
        "python": "/usr/bin/python3",
        "admission": {"inputs": [{"path": str(admission)}]},
        "paths": {
            "run_root": str(run_root),
            "journal": str(journal),
        },
    }
    return plan, key, source


def test_launch_adaptive_binds_frozen_plan_and_current_runner_sources(
    tmp_path: Path, monkeypatch
) -> None:
    plan, key, source = _plan(tmp_path)
    observed: list[str] = []
    monkeypatch.setattr(
        controller,
        "read_authenticated_journal",
        lambda *_args, **_kwargs: [{"event_type": "canary_admitted"}],
    )
    monkeypatch.setattr(
        controller,
        "execution_admission",
        lambda *_args, **_kwargs: {"execution_authorized": True},
    )

    def fake_main(args: list[str]) -> int:
        observed.extend(args)
        return 0

    monkeypatch.setattr(controller.detached, "main", fake_main)
    monkeypatch.setattr(
        controller.detached,
        "_status",
        lambda path: {"state": "running", "run_dir": str(path)},
    )

    result = controller._launch_adaptive(plan, key, resume=False)

    assert result["state"] == "running"
    assert observed[observed.index("--cwd") + 1] == str(source)
    assert observed[observed.index("--execution-output-root") + 1] == str(
        Path(plan["paths"]["run_root"])
    )
    assert observed[observed.index("--resume-contract") + 1] == "target_checkpoint"
    verifier = json.loads(observed[observed.index("--resume-verifier-json") + 1])
    assert verifier[1].endswith("verify_candidate_cortex_adaptive_resume.py")
    target = observed[observed.index("--") + 1 :]
    assert target[1].endswith("run_candidate_cortex_adaptive_target.py")
    assert "--resume" not in observed


def test_launch_adaptive_requires_admitted_execution(
    tmp_path: Path, monkeypatch
) -> None:
    plan, key, _source = _plan(tmp_path)
    monkeypatch.setattr(
        controller,
        "read_authenticated_journal",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        controller,
        "execution_admission",
        lambda *_args, **_kwargs: {"execution_authorized": False},
    )

    try:
        controller._launch_adaptive(plan, key, resume=False)
    except controller.CandidateCortexTrainingError as exc:
        assert str(exc) == "adaptive_launch_not_authorized"
    else:
        raise AssertionError("unadmitted adaptive campaign was launched")


def test_launch_adaptive_resume_is_explicit(tmp_path: Path, monkeypatch) -> None:
    plan, key, _source = _plan(tmp_path)
    observed: list[str] = []
    monkeypatch.setattr(
        controller,
        "read_authenticated_journal",
        lambda *_args, **_kwargs: [{"event_type": "canary_admitted"}],
    )
    monkeypatch.setattr(
        controller,
        "execution_admission",
        lambda *_args, **_kwargs: {"execution_authorized": True},
    )
    monkeypatch.setattr(
        controller.detached,
        "main",
        lambda args: observed.extend(args) or 0,
    )
    monkeypatch.setattr(controller.detached, "_status", lambda _path: {})

    controller._launch_adaptive(plan, key, resume=True)

    assert "--resume" in observed


def test_adaptive_recovery_execution_gets_distinct_immutable_root(
    tmp_path: Path, monkeypatch
) -> None:
    plan, key, _source = _plan(tmp_path)
    observed: list[str] = []
    monkeypatch.setattr(
        controller,
        "read_authenticated_journal",
        lambda *_args, **_kwargs: [{"event_type": "canary_admitted"}],
    )
    monkeypatch.setattr(
        controller,
        "execution_admission",
        lambda *_args, **_kwargs: {"execution_authorized": True},
    )
    monkeypatch.setattr(
        controller.detached,
        "main",
        lambda args: observed.extend(args) or 0,
    )
    monkeypatch.setattr(
        controller.detached,
        "_status",
        lambda path: {"state": "running", "run_dir": str(path)},
    )

    result = controller._launch_adaptive(
        plan,
        key,
        resume=False,
        execution_id="cp921-recovery",
    )

    expected = (
        Path(plan["paths"]["run_root"])
        / "adaptive-execution"
        / "executions"
        / "cp921-recovery"
        / "detached"
    )
    assert result["run_dir"] == str(expected)
    assert observed[observed.index("--run-dir") + 1] == str(expected)
    assert observed[observed.index("--name") + 1].endswith("-cp921-recovery")
    assert "--resume" not in observed


def test_adaptive_execution_id_rejects_path_escape(tmp_path: Path) -> None:
    plan, _key_path, _source = _plan(tmp_path)

    try:
        controller.stage_detached_root(plan, execution_id="../escape")
    except controller.CandidateCortexTrainingError as exc:
        assert str(exc) == "adaptive_execution_id_invalid"
    else:
        raise AssertionError("unsafe adaptive execution id was accepted")
