from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_journal import CampaignJournal, CampaignPlan
from tools import run_resumable_latent_cortex_pilot as controller


def _git_repo(path: Path) -> str:
    path.mkdir()
    (path / "tracked.txt").write_text("immutable\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)
    return subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _config(tmp_path: Path) -> dict:
    source = tmp_path / "source"
    commit = _git_repo(source)
    campaign = source / "campaign"
    state = source / "state"
    terminal = campaign / "terminal.json"
    runner = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys; "
            f"p=Path({str(terminal)!r}); p.parent.mkdir(parents=True, exist_ok=True); "
            "p.write_text('{}'); sys.exit(2)"
        ),
    ]
    verifier = [
        sys.executable,
        "-c",
        f"from pathlib import Path; import sys; sys.exit(0 if Path({str(terminal)!r}).exists() else 1)",
    ]
    body = {
        "schema": controller.CONFIG_SCHEMA,
        "campaign_name": "pilot-test",
        "source_root": str(source),
        "source_commit": commit,
        "campaign_dir": str(campaign),
        "state_dir": str(state),
        "runner_command": runner,
        "runner_command_sha256": controller._sha256(runner),
        "runner_executable_sha256": controller._executable_sha256(runner[0]),
        "verifier_command": verifier,
        "verifier_command_sha256": controller._sha256(verifier),
        "verifier_executable_sha256": controller._executable_sha256(verifier[0]),
        "detached_broker_policy": [],
        "detached_broker_policy_sha256": controller._sha256([]),
        "detached_attempt_timeout_seconds": 60,
        "execution_output_root": str(campaign),
        "max_attempts": 2,
        "retry_backoff_seconds": 1,
        "heartbeat_seconds": 1,
    }
    return {**body, "config_sha256": controller._sha256(body)}


def test_controller_accepts_scientific_nonzero_after_independent_verification(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    assert controller.run(config) == 0
    status = json.loads((Path(config["state_dir"]) / "controller-status.json").read_text())
    state = json.loads((Path(config["state_dir"]) / "controller-state.json").read_text())
    assert status["phase"] == "complete"
    assert status["campaign_progress"]["availability"] == "waiting_for_plan"
    assert state["terminal"] is True
    assert state["attempts_started"] == 1
    assert state["active_child_pid"] == 0
    controller._reconcile_event_journal(
        Path(config["state_dir"]) / "controller-events.jsonl",
        state,
    )


def test_controller_retries_infrastructure_exit_from_durable_state(tmp_path: Path) -> None:
    config = _config(tmp_path)
    campaign = Path(config["campaign_dir"])
    counter = campaign / "attempt.txt"
    terminal = campaign / "terminal.json"
    runner = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys; "
            f"c=Path({str(counter)!r}); c.parent.mkdir(parents=True, exist_ok=True); "
            "n=int(c.read_text())+1 if c.exists() else 1; c.write_text(str(n)); "
            f"Path({str(terminal)!r}).write_text('{{}}') if n == 2 else None; "
            "sys.exit(2 if n == 2 else 1)"
        ),
    ]
    body = {
        **{key: value for key, value in config.items() if key != "config_sha256"},
        "runner_command": runner,
        "runner_command_sha256": controller._sha256(runner),
        "runner_executable_sha256": controller._executable_sha256(runner[0]),
    }
    config = {**body, "config_sha256": controller._sha256(body)}
    assert controller.run(config) == 0
    state = json.loads((Path(config["state_dir"]) / "controller-state.json").read_text())
    assert state["attempts_started"] == 2
    assert state["terminal"] is True


def test_controller_runs_broker_policy_through_detached_supervisor(tmp_path: Path) -> None:
    config = _config(tmp_path)
    campaign = Path(config["campaign_dir"])
    policy = [
        {
            "command": [sys.executable, "-c", "raise SystemExit(0)"],
            "cwd": config["source_root"],
            "stdout_path": str(campaign / "broker.log"),
            "timeout_s_max": 30.0,
            "max_invocations": 1,
        }
    ]
    body = {
        **{key: value for key, value in config.items() if key != "config_sha256"},
        "detached_broker_policy": policy,
        "detached_broker_policy_sha256": controller._sha256(policy),
    }
    config = {**body, "config_sha256": controller._sha256(body)}
    assert controller.run(config) == 0
    state = json.loads((Path(config["state_dir"]) / "controller-state.json").read_text())
    assert state["terminal"] is True
    assert state["attempts_started"] == 1
    detached_root = Path(config["state_dir"]) / "detached-attempts" / "attempt-0001"
    assert (detached_root / controller.detached.RECEIPT_FILE).exists()


def test_controller_adopts_detached_attempt_if_launch_return_is_lost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    campaign = Path(config["campaign_dir"])
    policy = [
        {
            "command": [sys.executable, "-c", "raise SystemExit(0)"],
            "cwd": config["source_root"],
            "stdout_path": str(campaign / "broker.log"),
            "timeout_s_max": 30.0,
            "max_invocations": 1,
        }
    ]
    body = {
        **{key: value for key, value in config.items() if key != "config_sha256"},
        "detached_broker_policy": policy,
        "detached_broker_policy_sha256": controller._sha256(policy),
    }
    config = {**body, "config_sha256": controller._sha256(body)}
    real_main = controller.detached.main
    launch_return_lost = False

    def _lose_first_launch_return(arguments: list[str]) -> int:
        nonlocal launch_return_lost
        result = real_main(arguments)
        if arguments[0] == "launch" and not launch_return_lost:
            launch_return_lost = True
            raise controller.PilotControllerError("simulated_controller_crash_after_launch")
        return result

    monkeypatch.setattr(controller.detached, "main", _lose_first_launch_return)
    with pytest.raises(controller.PilotControllerError, match="simulated_controller_crash"):
        controller.run(config)

    monkeypatch.setattr(controller.detached, "main", real_main)
    assert controller.run(config) == 0
    state = json.loads((Path(config["state_dir"]) / "controller-state.json").read_text())
    assert state["attempts_started"] == 1
    events = [
        json.loads(line)
        for line in (Path(config["state_dir"]) / "controller-events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events].count("ATTEMPT_RESERVED") == 1


def test_controller_rejects_tracked_source_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (Path(config["source_root"]) / "tracked.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(controller.PilotControllerError, match="source_identity_changed"):
        controller.run(config)


def test_config_hash_and_command_hash_are_fail_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = tmp_path / "config.json"
    path.write_bytes(controller.canonical_json_bytes(config) + b"\n")
    assert controller.load_config(path)["runner_command"] == config["runner_command"]
    config["runner_command_sha256"] = "0" * 64
    body = {key: value for key, value in config.items() if key != "config_sha256"}
    config["config_sha256"] = controller._sha256(body)
    path.write_bytes(controller.canonical_json_bytes(config) + b"\n")
    with pytest.raises(controller.PilotControllerError, match="runner_command_hash"):
        controller.load_config(path)


@pytest.mark.parametrize("owner", ["runner", "broker"])
def test_config_rejects_explicit_input_inside_mutable_output_root(
    tmp_path: Path,
    owner: str,
) -> None:
    config = _config(tmp_path)
    campaign = Path(config["campaign_dir"])
    campaign.mkdir()
    audit = campaign / "contamination-audit.json"
    audit.write_text("{}\n", encoding="utf-8")
    body = {key: value for key, value in config.items() if key != "config_sha256"}
    if owner == "runner":
        command = [*config["runner_command"], "--contamination-audit", str(audit)]
        body["runner_command"] = command
        body["runner_command_sha256"] = controller._sha256(command)
    else:
        policy = [
            {
                "command": [sys.executable, "-c", "raise SystemExit(0)", str(audit)],
                "cwd": config["source_root"],
                "stdout_path": str(campaign / "broker.log"),
                "timeout_s_max": 30.0,
                "max_invocations": 1,
            }
        ]
        body["detached_broker_policy"] = policy
        body["detached_broker_policy_sha256"] = controller._sha256(policy)
    config = {**body, "config_sha256": controller._sha256(body)}
    path = tmp_path / f"{owner}-config.json"
    path.write_bytes(controller.canonical_json_bytes(config) + b"\n")

    with pytest.raises(
        controller.PilotControllerError,
        match=rf"{owner}(_0)?_explicit_input_inside_execution_output_root",
    ):
        controller.load_config(path)


def test_config_accepts_explicit_input_outside_mutable_output_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = Path(config["source_root"])
    audit = source / "campaign-inputs" / "contamination-audit.json"
    audit.parent.mkdir()
    audit.write_text("{}\n", encoding="utf-8")
    command = [*config["runner_command"], "--contamination-audit", str(audit)]
    body = {
        **{key: value for key, value in config.items() if key != "config_sha256"},
        "runner_command": command,
        "runner_command_sha256": controller._sha256(command),
    }
    config = {**body, "config_sha256": controller._sha256(body)}
    path = tmp_path / "external-input-config.json"
    path.write_bytes(controller.canonical_json_bytes(config) + b"\n")

    assert controller.load_config(path)["runner_command"] == command


def test_campaign_progress_counts_only_hash_valid_sealed_cells(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    plan = CampaignPlan.build(
        "progress-test",
        [
            {"arm": "base_rlc", "domain": "logic", "task_id": "task-1"},
            {"arm": "adapter_vanilla", "domain": "math", "task_id": "task-2"},
        ],
    )
    (campaign / "plan.json").write_bytes(controller.canonical_json_bytes(plan.to_dict()) + b"\n")
    with CampaignJournal(campaign / "campaign.jsonl", plan) as journal:
        first = journal.start_cell(plan.cell_ids[0])
        journal.record_arm_result(
            plan.cell_ids[0],
            first,
            {"latency_s": 12.5, "text": "answer"},
        )
        journal.start_cell(plan.cell_ids[1])

    progress = controller._campaign_progress(campaign)

    assert progress["availability"] == "current"
    assert progress["total_cells"] == 2
    assert progress["started_attempts"] == 2
    assert progress["sealed_result_cells"] == 1
    assert progress["verified_cells"] == 0
    assert progress["committed_cells"] == 0
    assert progress["by_arm"]["base_rlc"]["sealed_results"] == 1
    assert progress["active_cells"][0]["arm"] == "adapter_vanilla"
    assert progress["projection"]["projected_remaining_inference_seconds"] == 12.5


def test_campaign_progress_rejects_rehashed_invalid_attempt_identity(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    plan = CampaignPlan.build(
        "tamper-test",
        [{"arm": "base_vanilla", "domain": "logic", "task_id": "task-1"}],
    )
    (campaign / "plan.json").write_bytes(controller.canonical_json_bytes(plan.to_dict()) + b"\n")
    with CampaignJournal(campaign / "campaign.jsonl", plan) as journal:
        journal.start_cell(plan.cell_ids[0])
    rows = (campaign / "campaign.jsonl").read_text().splitlines()
    event = json.loads(rows[-1])
    event["attempt_id"] = f"attempt-{'0' * 64}"
    body = {key: value for key, value in event.items() if key != "event_sha256"}
    event["event_sha256"] = controller._sha256(body)
    rows[-1] = controller.canonical_json_bytes(event).decode()
    (campaign / "campaign.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    progress = controller._campaign_progress(campaign)

    assert progress == {
        "schema": controller.PROGRESS_SCHEMA,
        "availability": "unavailable",
        "reason": "PilotControllerError",
    }


def test_event_journal_recovers_one_committed_event_after_state_write_gap(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    state = controller._default_state(config)
    journal = tmp_path / "events.jsonl"
    controller._append_event(journal, state, "VERIFIED_TERMINAL", {})
    stale = controller._default_state(config)
    assert controller._reconcile_event_journal(journal, stale) is True
    assert stale["terminal"] is True
    assert stale["journal_sequence"] == 1
