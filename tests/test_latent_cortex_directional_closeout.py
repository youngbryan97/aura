from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from tools import run_latent_cortex_directional_closeout as closeout


def _git_repo(path: Path) -> str:
    path.mkdir()
    (path / "tools").mkdir()
    (path / "tools" / Path(closeout.__file__).name).write_bytes(
        Path(closeout.__file__).read_bytes()
    )
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    subprocess.run(["git", "add", "tools"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)
    return subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _hashed(document: dict, key: str) -> dict:
    return {**document, key: closeout._sha(document)}


def _fixture(tmp_path: Path, *, phase: str = "complete") -> dict:
    source = tmp_path / "source"
    commit = _git_repo(source)
    campaign = tmp_path / "campaign"
    controller = tmp_path / "controller"
    output = tmp_path / "closeout"
    campaign.mkdir()
    controller.mkdir()
    (campaign / "plan.json").write_text("{}\n", encoding="utf-8")
    trust_root = tmp_path / "trust.pem"
    trust_root.write_text("public\n", encoding="utf-8")
    controller_config_body = {
        "schema": closeout.CONTROLLER_CONFIG_SCHEMA,
        "campaign_name": "directional-test",
        "source_commit": "b" * 40,
        "campaign_dir": str(campaign),
        "state_dir": str(controller),
        "execution_output_root": str(tmp_path),
    }
    controller_config = _hashed(controller_config_body, "config_sha256")
    controller_config_path = tmp_path / "controller-config.json"
    controller_config_path.write_bytes(canonical_json_bytes(controller_config) + b"\n")
    controller_status_body = {
        "schema": closeout.CONTROLLER_STATUS_SCHEMA,
        "campaign_name": "directional-test",
        "campaign_dir": str(campaign),
        "config_sha256": controller_config["config_sha256"],
        "source_commit": "b" * 40,
        "phase": phase,
        "reason": "",
        "heartbeat_at_unix": 1.0,
        "campaign_progress": {
            "sealed_result_cells": 4,
            "total_cells": 4,
            "failed_attempts": 0,
        },
    }
    (controller / "controller-status.json").write_bytes(
        canonical_json_bytes(_hashed(controller_status_body, "status_sha256")) + b"\n"
    )
    event_body = {
        "schema": closeout.CONTROLLER_EVENT_SCHEMA,
        "sequence": 1,
        "previous_event_sha256": "0" * 64,
        "event": "VERIFIED_TERMINAL",
        "recorded_at_unix": 1.0,
        "detail": {},
    }
    event = _hashed(event_body, "event_sha256")
    (controller / "controller-events.jsonl").write_bytes(canonical_json_bytes(event) + b"\n")
    state_body = {
        "schema": closeout.CONTROLLER_STATE_SCHEMA,
        "terminal": True,
        "journal_sequence": 1,
        "journal_head_sha256": event["event_sha256"],
    }
    (controller / "controller-state.json").write_bytes(
        canonical_json_bytes(_hashed(state_body, "state_sha256")) + b"\n"
    )
    independent = tmp_path / "independent-verdict.json"
    independent.write_text("{}\n", encoding="utf-8")
    config_body = {
        "schema": closeout.CONFIG_SCHEMA,
        "campaign_name": "directional-test",
        "source_root": str(source),
        "source_commit": commit,
        "closeout_tool_sha256": closeout._bytes_sha(
            (source / "tools" / Path(closeout.__file__).name).read_bytes()
        ),
        "controller_config": closeout._file_binding(
            controller_config_path, role="controller_config"
        ),
        "controller_config_identity_sha256": controller_config["config_sha256"],
        "controller_source_commit": "b" * 40,
        "controller_status_path": str(controller / "controller-status.json"),
        "controller_state_path": str(controller / "controller-state.json"),
        "controller_events_path": str(controller / "controller-events.jsonl"),
        "campaign_dir": str(campaign),
        "plan_sha256": "a" * 64,
        "independent_verdict_path": str(independent),
        "contamination_trust_root": closeout._file_binding(
            trust_root, role="contamination_trust_root"
        ),
        "output_root": str(output),
        "state_dir": str(output / "supervisor"),
        "directional_verdict_path": str(output / "directional-verdict.json"),
        "powered_handoff_path": str(output / "powered-campaign-handoff.json"),
        "receipt_path": str(output / "closeout-receipt.json"),
        "target_campaign_name": "powered-test",
        "launch_label": "com.aura.test.directional-closeout",
        "poll_seconds": 1,
        "stale_after_seconds": 30,
        "max_wait_seconds": 60,
    }
    return {**config_body, "config_sha256": closeout._sha(config_body)}


def test_positive_terminal_closeout_materializes_powered_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture(tmp_path)
    monkeypatch.setattr(closeout, "_verify_source_identity", lambda _config: None)
    verdict = {
        "schema": "aura.latent_cortex.directional_gate.v1",
        "decision": "advance_to_powered_external_campaign",
        "directional_gate_passed": True,
    }
    monkeypatch.setattr(closeout, "verify_directional", lambda **_kwargs: verdict)

    def fake_materialize(*, output: Path, **_kwargs) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"schema": "handoff", "decision": "prepare"}) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(closeout, "materialize", fake_materialize)
    result = closeout.closeout_once(config)

    assert result is not None
    assert result["exit_code"] == 0
    assert result["directional_gate_passed"] is True
    assert Path(config["powered_handoff_path"]).is_file()
    receipt = json.loads(Path(config["receipt_path"]).read_text())
    assert receipt["nonclaims"]["reasoning_gain_proven"] is False
    assert receipt["result"]["powered_handoff"]["schema"] == "handoff"


def test_negative_terminal_closeout_emits_diagnosis_without_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture(tmp_path)
    monkeypatch.setattr(closeout, "_verify_source_identity", lambda _config: None)
    monkeypatch.setattr(
        closeout,
        "verify_directional",
        lambda **_kwargs: {
            "schema": "aura.latent_cortex.directional_gate.v1",
            "decision": "repair_and_preregister_directional_revision",
            "directional_gate_passed": False,
            "diagnoses": ["positive_adapter_rlc_interaction_not_observed"],
        },
    )
    monkeypatch.setattr(
        closeout,
        "materialize",
        lambda **_kwargs: pytest.fail("negative result must not materialize a powered handoff"),
    )

    result = closeout.closeout_once(config)

    assert result is not None
    assert result["exit_code"] == 0
    assert result["directional_gate_passed"] is False
    assert result["directional_verdict"]["diagnoses"] == [
        "positive_adapter_rlc_interaction_not_observed"
    ]
    assert not Path(config["powered_handoff_path"]).exists()


def test_controller_status_tampering_fails_closed(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    status_path = Path(config["controller_status_path"])
    status = json.loads(status_path.read_text())
    status["phase"] = "running"
    status_path.write_text(json.dumps(status) + "\n", encoding="utf-8")

    with pytest.raises(closeout.DirectionalCloseoutError, match="controller_status_integrity_invalid"):
        closeout._controller_snapshot(config)


def test_controller_terminal_requires_verified_event_chain(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    events_path = Path(config["controller_events_path"])
    event = json.loads(events_path.read_text())
    event["event"] = "ATTEMPT_EXITED"
    body = dict(event)
    body.pop("event_sha256")
    events_path.write_bytes(
        canonical_json_bytes({**body, "event_sha256": closeout._sha(body)}) + b"\n"
    )

    with pytest.raises(closeout.DirectionalCloseoutError, match="controller_terminal_receipt_invalid"):
        closeout._controller_snapshot(config)


def test_closeout_rechecks_trust_root_binding_at_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture(tmp_path, phase="running")
    monkeypatch.setattr(closeout, "_verify_source_identity", lambda _config: None)
    Path(config["contamination_trust_root"]["path"]).write_text(
        "changed\n", encoding="utf-8"
    )

    with pytest.raises(closeout.DirectionalCloseoutError, match="binding_changed"):
        closeout.closeout_once(config)
