from __future__ import annotations

import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    VerifiedCampaignTrustPolicy,
    assemble_role_attestation,
    validate_campaign_trust_policy,
)
from core.brain.llm.latent_cortex.worker_origin import (
    ZERO_SHA256,
    verify_worker_result_origin,
)
from tools import run_detached_step as detached

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="strong containment requires macOS")


def _safe_resume_verifier() -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import hashlib,json,os; "
            "plan=os.environ['AURA_DETACHED_PLAN_SHA256']; "
            "command=os.environ['AURA_DETACHED_COMMAND_SHA256']; "
            "attempt=int(os.environ['AURA_DETACHED_PRIOR_ATTEMPT']); "
            "head=os.environ['AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256']; "
            "e={'schema':'aura.detached_step.resume_evidence.v2','plan_sha256':plan,"
            "'command_sha256':command,'prior_attempt':attempt,"
            "'prior_journal_head_sha256':head,'checkpoint_sequence':0,"
            "'checkpoint_state':'test-safe'}; "
            "raw=json.dumps(e,sort_keys=True,separators=(',',':')).encode(); "
            "esha=hashlib.sha256(raw).hexdigest(); "
            "identity=hashlib.sha256(json.dumps({'prior_attempt':attempt,"
            "'prior_journal_head_sha256':head,'checkpoint_sequence':0,"
            "'evidence_sha256':esha},sort_keys=True,separators=(',',':')).encode()).hexdigest(); "
            "print(json.dumps({'schema':'aura.detached_step.resume_verdict.v3',"
            "'plan_sha256':plan,'command_sha256':command,'prior_attempt':attempt,"
            "'prior_journal_head_sha256':head,'checkpoint_sequence':0,"
            "'checkpoint_identity':identity,'verdict':'safe_to_resume',"
            "'evidence_sha256':esha,'evidence':e}))"
        ),
    ]


def _indeterminate_resume_verifier() -> list[str]:
    command = _safe_resume_verifier()
    command[-1] = command[-1].replace("'safe_to_resume'", "'indeterminate'")
    return command


def _replaying_resume_verifier(state_path: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import hashlib,json,os,pathlib\n"
            f"state=pathlib.Path({str(state_path)!r})\n"
            "if state.exists():\n"
            "    print(state.read_text())\n"
            "else:\n"
            "    plan=os.environ['AURA_DETACHED_PLAN_SHA256']\n"
            "    command=os.environ['AURA_DETACHED_COMMAND_SHA256']\n"
            "    attempt=int(os.environ['AURA_DETACHED_PRIOR_ATTEMPT'])\n"
            "    head=os.environ['AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256']\n"
            "    evidence={'schema':'aura.detached_step.resume_evidence.v2',"
            "'plan_sha256':plan,'command_sha256':command,'prior_attempt':attempt,"
            "'prior_journal_head_sha256':head,'checkpoint_sequence':0}\n"
            "    raw=json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()\n"
            "    evidence_sha=hashlib.sha256(raw).hexdigest()\n"
            "    identity=hashlib.sha256(json.dumps({'prior_attempt':attempt,"
            "'prior_journal_head_sha256':head,'checkpoint_sequence':0,"
            "'evidence_sha256':evidence_sha},sort_keys=True,separators=(',',':')).encode()).hexdigest()\n"
            "    verdict={'schema':'aura.detached_step.resume_verdict.v3',"
            "'plan_sha256':plan,'command_sha256':command,'prior_attempt':attempt,"
            "'prior_journal_head_sha256':head,'checkpoint_sequence':0,"
            "'checkpoint_identity':identity,'verdict':'safe_to_resume',"
            "'evidence_sha256':evidence_sha,'evidence':evidence}\n"
            "    state.write_text(json.dumps(verdict,sort_keys=True,separators=(',',':')))\n"
            "    print(json.dumps(verdict))\n"
        ),
    ]


def _wait_for(path: Path, timeout_s: float = 8.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def _wait_for_text(path: Path, expected: str, timeout_s: float = 8.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.is_file() and path.read_text(encoding="utf-8") == expected:
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path} to contain {expected!r}")


def _wait_for_state(path: Path, expected: str, timeout_s: float = 8.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = {}
            if value.get("state") == expected:
                return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path} state {expected!r}")


def _wait_for_glob(directory: Path, pattern: str, timeout_s: float = 8.0) -> Path:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        matches = list(directory.glob(pattern))
        if len(matches) == 1 and matches[0].is_file():
            return matches[0]
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {directory / pattern}")


def _public_raw(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _campaign_role_pin(
    role: str,
    key: Ed25519PrivateKey,
) -> dict[str, str]:
    raw = _public_raw(key)
    return {
        "signer_id": f"{role}-signer",
        "organization_id": f"{role}-organization",
        "public_key_b64": base64.b64encode(raw).decode("ascii"),
        "key_id": hashlib.sha256(raw).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            f"{role}:implementation".encode()
        ).hexdigest(),
        "release_sha256": hashlib.sha256(
            f"{role}:release".encode()
        ).hexdigest(),
        "custody_class": "test_fixture",
        "custody_evidence_sha256": hashlib.sha256(
            f"{role}:custody".encode()
        ).hexdigest(),
    }


def _worker_origin_trust_fixture(
    directory: Path,
) -> tuple[
    Path,
    Path,
    VerifiedCampaignTrustPolicy,
    Ed25519PrivateKey,
]:
    directory.mkdir(mode=0o700)
    root = Ed25519PrivateKey.generate()
    role_keys = {
        role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES
    }
    now = int(time.time())
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "detached-worker-origin-test",
        "policy_revision": 1,
        "campaign_name": "detached-worker-origin-test",
        "protocol_sha256": "1" * 64,
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": now - 120,
        "not_before_unix": now - 60,
        "expires_at_unix": now + 3600,
        "roles": {
            role: _campaign_role_pin(role, role_keys[role])
            for role in CAMPAIGN_TRUST_ROLES
        },
    }
    signed = canonical_json_bytes(body)
    root_raw = _public_raw(root)
    document = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode(
                "ascii"
            ),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    policy_path = directory / "campaign-policy.json"
    root_path = directory / "campaign-root.pem"
    detached._atomic_write(policy_path, document, replace=False)
    root_path.write_bytes(
        root.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    root_path.chmod(0o600)
    policy = validate_campaign_trust_policy(
        document,
        trusted_root_public_key_pem=root_path.read_bytes(),
        now_unix=now,
    )
    return policy_path, root_path, policy, role_keys[CAMPAIGN_RUNNER]


def _worker_origin_policy(
    *,
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    trust_policy_path: Path,
    trust_root_path: Path,
    artifact_dir: Path,
    timeout_s: float,
) -> list[dict]:
    return [
        {
            "command": command,
            "cwd": str(cwd),
            "stdout_path": str(stdout_path),
            "timeout_s_max": timeout_s,
            "max_invocations": 1,
            "worker_origin": {
                "schema": detached.WORKER_ORIGIN_POLICY_SCHEMA,
                "campaign_name": "detached-worker-origin-test",
                "protocol_sha256": "1" * 64,
                "trust_policy_path": str(trust_policy_path),
                "trust_root_path": str(trust_root_path),
                "artifact_dir": str(artifact_dir),
                "arm": "adapter_rlc",
                "worker_attempt_slot": 1,
                "allowed_cells": [
                    {
                        "cell_id": "cell-0001",
                        "cell_type": "reasoning",
                    }
                ],
                "model_identity_sha256": "8" * 64,
                "adapter_identity_sha256": "9" * 64,
                "authorization_ttl_seconds": 300,
            },
        }
    ]


def _launch(
    run_dir: Path,
    command: list[str],
    *,
    timeout_s: float = 5.0,
    resume: bool = False,
    resume_contract: str = "none",
    resume_verifier: list[str] | None = None,
    broker_policy: list[dict] | None = None,
    cwd: Path | None = None,
    execution_output_roots: list[Path] | None = None,
) -> dict:
    resume_args = ["--resume"] if resume else []
    if resume_contract == "target_checkpoint" and resume_verifier is None:
        resume_verifier = _safe_resume_verifier()
    verifier_args = (
        ["--resume-verifier-json", json.dumps(resume_verifier)]
        if resume_verifier is not None
        else []
    )
    broker_args = (
        ["--broker-policy-json", json.dumps(broker_policy)]
        if broker_policy is not None
        else []
    )
    output_args = [
        argument
        for root in execution_output_roots or []
        for argument in ("--execution-output-root", str(root))
    ]
    result = subprocess.run(
        [
            sys.executable,
            str(Path(detached.__file__).resolve()),
            "launch",
            "--run-dir",
            str(run_dir),
            "--name",
            "test-step",
            "--cwd",
            str(cwd or run_dir.parent),
            "--timeout",
            str(timeout_s),
            "--resume-contract",
            resume_contract,
            *verifier_args,
            *broker_args,
            *output_args,
            *resume_args,
            "--",
            *command,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    return json.loads(result.stdout)


def test_launcher_source_never_forks_the_importing_python_process() -> None:
    source = Path(detached.__file__).read_text(encoding="utf-8")
    assert "os.fork(" not in source
    assert "os.posix_spawn(" in source


def test_nonzero_target_runs_once_and_survives_launcher(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    counter = tmp_path / "counter.txt"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys; "
            f"p=Path({str(counter)!r}); "
            "p.write_text((p.read_text() if p.exists() else '')+'once\\n'); "
            "sys.exit(7)"
        ),
    ]
    launch = _launch(run_dir, command)
    assert launch["restart_policy"] == "never"
    receipt_path = run_dir / detached.RECEIPT_FILE
    receipt = _wait_for(receipt_path)
    first_receipt = receipt_path.read_bytes()
    assert receipt["status"] == "failed"
    assert receipt["returncode"] == 7
    assert receipt["restart_count"] == 0
    assert receipt["supervisor_attempt"] == 1
    assert counter.read_text(encoding="utf-8") == "once\n"

    time.sleep(0.4)
    assert receipt_path.read_bytes() == first_receipt
    assert counter.read_text(encoding="utf-8") == "once\n"
    inspection = detached._status(run_dir)
    assert inspection["terminal"] is True
    assert inspection["supervisor_alive"] is False


def test_timeout_kills_target_group_and_writes_terminal_receipt(tmp_path: Path) -> None:
    run_dir = tmp_path / "timeout"
    _launch(
        run_dir,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_s=0.25,
    )
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE)
    assert receipt["status"] == "timed_out"
    assert receipt["timed_out"] is True
    assert receipt["returncode"] == 124
    assert receipt["restart_count"] == 0


def test_duplicate_run_directory_is_immutable(tmp_path: Path) -> None:
    run_dir = tmp_path / "immutable"
    command = [sys.executable, "-c", "pass"]
    _launch(run_dir, command)
    _wait_for(run_dir / detached.RECEIPT_FILE)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(detached.__file__).resolve()),
            "launch",
            "--run-dir",
            str(run_dir),
            "--name",
            "test-step",
            "--cwd",
            str(tmp_path),
            "--timeout",
            "5",
            "--",
            *command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert result.returncode == 2
    assert "terminal receipt already exists" in result.stderr


def test_plan_and_receipt_hashes_replay(tmp_path: Path) -> None:
    run_dir = tmp_path / "hashes"
    _launch(run_dir, [sys.executable, "-c", "pass"])
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE)
    plan = json.loads((run_dir / detached.PLAN_FILE).read_text(encoding="utf-8"))
    plan_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    assert plan["plan_sha256"] == detached._sha256(plan_body)
    assert receipt["plan_sha256"] == plan["plan_sha256"]
    assert receipt["receipt_sha256"] == detached._sha256(receipt_body)
    attempts = detached._read_attempts(run_dir)
    assert [event["event"] for event in attempts] == [
        "LAUNCHED",
        "CONTROL_READY",
        "TARGET_STARTED",
        "TERMINAL",
    ]
    assert attempts[0]["previous_event_sha256"] == ""
    assert attempts[1]["previous_event_sha256"] == attempts[0]["event_sha256"]
    assert attempts[2]["previous_event_sha256"] == attempts[1]["event_sha256"]
    assert attempts[3]["previous_event_sha256"] == attempts[2]["event_sha256"]


def test_explicit_resume_reaps_stale_child_and_increments_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "resumed"
    counter = tmp_path / "resume-counter.txt"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import time; "
            f"p=Path({str(counter)!r}); "
            "n=int(p.read_text())+1 if p.exists() else 1; "
            "p.write_text(str(n)); "
            "time.sleep(30) if n == 1 else None"
        ),
    ]
    first = _launch(run_dir, command, timeout_s=60.0, resume_contract="target_checkpoint")
    status = _wait_for_state(run_dir / detached.STATUS_FILE, "running")
    assert status["supervisor_attempt"] == 1
    _wait_for_text(counter, "1")

    os.kill(first["supervisor_pid"], signal.SIGKILL)
    deadline = time.time() + 5.0
    while time.time() < deadline and detached._pid_matches(
        first["supervisor_pid"], first["supervisor_start_token"]
    ):
        time.sleep(0.05)
    assert not detached._pid_matches(first["supervisor_pid"], first["supervisor_start_token"])

    resumed = _launch(
        run_dir,
        command,
        timeout_s=60.0,
        resume=True,
        resume_contract="target_checkpoint",
    )
    assert resumed["resumed"] is True
    assert resumed["recovered_stale_child"] is True
    assert resumed["supervisor_attempt"] == 2
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE)
    assert receipt["status"] == "passed"
    assert receipt["supervisor_attempt"] == 2
    assert counter.read_text(encoding="utf-8") == "2"
    events = detached._read_attempts(run_dir)
    assert [(event["event"], event["attempt"]) for event in events] == [
        ("LAUNCHED", 1),
        ("CONTROL_READY", 1),
        ("TARGET_STARTED", 1),
        ("LAUNCHED", 2),
        ("CONTROL_READY", 2),
        ("TARGET_STARTED", 2),
        ("TERMINAL", 2),
    ]


def test_resume_is_rejected_while_supervisor_is_alive(tmp_path: Path) -> None:
    run_dir = tmp_path / "live"
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    _launch(run_dir, command, timeout_s=60.0, resume_contract="target_checkpoint")
    _wait_for_state(run_dir / detached.STATUS_FILE, "running")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(detached.__file__).resolve()),
            "launch",
            "--run-dir",
            str(run_dir),
            "--name",
            "test-step",
            "--cwd",
            str(tmp_path),
            "--timeout",
            "60",
            "--resume-contract",
            "target_checkpoint",
            "--resume-verifier-json",
            json.dumps(_safe_resume_verifier()),
            "--resume",
            "--",
            *command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert result.returncode == 2
    assert "supervisor is already alive" in result.stderr
    stopped = detached._stop(run_dir)
    assert stopped["stopped"] is True
    assert stopped["control"] == "authenticated_socket"
    _wait_for(run_dir / detached.RECEIPT_FILE)


def test_resume_requires_existing_plan(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing"
    command = [sys.executable, "-c", "pass"]
    with pytest.raises(subprocess.CalledProcessError) as raised:
        _launch(run_dir, command, resume=True)
    assert "--resume requires an existing detached plan" in raised.value.stderr


def test_generic_incomplete_execution_cannot_be_replayed(tmp_path: Path) -> None:
    run_dir = tmp_path / "generic"
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    first = _launch(run_dir, command, timeout_s=60.0)
    _wait_for_state(run_dir / detached.STATUS_FILE, "running")
    os.kill(first["supervisor_pid"], signal.SIGKILL)
    deadline = time.time() + 5.0
    while time.time() < deadline and detached._pid_matches(
        first["supervisor_pid"], first["supervisor_start_token"]
    ):
        time.sleep(0.05)

    result = subprocess.run(
        [
            sys.executable,
            str(Path(detached.__file__).resolve()),
            "launch",
            "--run-dir",
            str(run_dir),
            "--name",
            "test-step",
            "--cwd",
            str(tmp_path),
            "--timeout",
            "60",
            "--resume-contract",
            "none",
            "--resume",
            "--",
            *command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert result.returncode == 2
    assert "completion-indeterminate" in result.stderr
    target = next(
        event
        for event in detached._read_attempts(run_dir)
        if event["event"] == "TARGET_STARTED"
    )
    assert detached._terminate_stale_target(target) is True


def test_checkpoint_resume_requires_verifier_safe_verdict(tmp_path: Path) -> None:
    run_dir = tmp_path / "verifier-refusal"
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    verifier = _indeterminate_resume_verifier()
    first = _launch(
        run_dir,
        command,
        timeout_s=60.0,
        resume_contract="target_checkpoint",
        resume_verifier=verifier,
    )
    _wait_for_state(run_dir / detached.STATUS_FILE, "running")
    os.kill(first["supervisor_pid"], signal.SIGKILL)
    deadline = time.time() + 5.0
    while time.time() < deadline and detached._pid_matches(
        first["supervisor_pid"], first["supervisor_start_token"]
    ):
        time.sleep(0.05)
    with pytest.raises(subprocess.CalledProcessError) as raised:
        _launch(
            run_dir,
            command,
            timeout_s=60.0,
            resume=True,
            resume_contract="target_checkpoint",
            resume_verifier=verifier,
        )
    assert "verifier returned indeterminate" in raised.value.stderr


def test_authoritative_terminal_journal_recreates_receipt(tmp_path: Path) -> None:
    run_dir = tmp_path / "reconcile"
    _launch(run_dir, [sys.executable, "-c", "pass"])
    receipt_path = run_dir / detached.RECEIPT_FILE
    _wait_for(receipt_path)
    expected = receipt_path.read_bytes()
    receipt_path.unlink()

    inspection = detached._status(run_dir)
    assert inspection["terminal"] is True
    assert receipt_path.read_bytes() == expected


def test_terminal_journal_crash_boundary_reconciles_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "terminal-crash"
    monkeypatch.setenv("AURA_DETACHED_TEST_CRASH_POINT", "after_terminal_journal")
    launch = _launch(
        run_dir,
        [sys.executable, "-c", "import time; time.sleep(0.2)"],
        timeout_s=15.0,
    )
    deadline = time.time() + 20.0
    while time.time() < deadline and detached._pid_matches(
        launch["supervisor_pid"], launch["supervisor_start_token"]
    ):
        time.sleep(0.05)
    assert not detached._pid_matches(
        launch["supervisor_pid"],
        launch["supervisor_start_token"],
    )
    assert not (run_dir / detached.RECEIPT_FILE).exists()
    terminal = [
        event for event in detached._read_attempts(run_dir) if event["event"] == "TERMINAL"
    ]
    assert len(terminal) == 1

    inspection = detached._status(run_dir)
    assert inspection["terminal"] is True
    assert (run_dir / detached.RECEIPT_FILE).is_file()


@pytest.mark.parametrize(
    "crash_point",
    ("after_supervisor_fork_before_reservation", "after_reservation_before_release"),
)
def test_handoff_crash_boundaries_never_duplicate_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_point: str,
) -> None:
    run_dir = tmp_path / crash_point
    counter = tmp_path / f"{crash_point}.txt"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(counter)!r}).write_text('once')",
    ]
    monkeypatch.setenv("AURA_DETACHED_TEST_CRASH_POINT", crash_point)
    with pytest.raises(subprocess.CalledProcessError):
        _launch(
            run_dir,
            command,
            timeout_s=30.0,
            resume_contract="target_checkpoint",
        )
    assert not counter.exists()
    monkeypatch.delenv("AURA_DETACHED_TEST_CRASH_POINT")
    status_path = run_dir / detached.STATUS_FILE
    if status_path.is_file():
        stale = json.loads(status_path.read_text(encoding="utf-8"))
        deadline = time.time() + 5.0
        while time.time() < deadline and detached._pid_matches(
            stale["supervisor_pid"], stale["supervisor_start_token"]
        ):
            time.sleep(0.05)

    resumed = _launch(
        run_dir,
        command,
        timeout_s=30.0,
        resume=True,
        resume_contract="target_checkpoint",
    )
    assert resumed["resumed"] is True
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE)
    assert receipt["status"] == "passed"
    assert counter.read_text(encoding="utf-8") == "once"


def test_forged_status_target_identity_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "forged-status"
    _launch(run_dir, [sys.executable, "-c", "pass"])
    _wait_for(run_dir / detached.RECEIPT_FILE)
    status_path = run_dir / detached.STATUS_FILE
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["child_process_group_id"] += 1
    detached._atomic_write(status_path, status)

    with pytest.raises(detached.DetachedStepError, match="status target identity mismatch"):
        detached._status(run_dir)


def test_log_symlink_is_rejected_without_touching_target(tmp_path: Path) -> None:
    run_dir = tmp_path / "symlink-log"
    run_dir.mkdir(mode=0o700)
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged", encoding="utf-8")
    (run_dir / detached.LOG_FILE).symlink_to(victim)

    _launch(run_dir, [sys.executable, "-c", "print('unsafe')"])
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE)
    assert receipt["status"] == "supervisor_failed"
    assert receipt["child_pid"] == 0
    assert victim.read_text(encoding="utf-8") == "unchanged"


def test_custodied_detached_write_rejects_root_exchange_without_redirect(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    with detached._run_directory_custody(run_dir, create=True) as custody:
        displaced = tmp_path / "displaced"
        run_dir.rename(displaced)
        replacement.rename(run_dir)
        with pytest.raises(detached.DetachedStepError, match="custodied artifact write failed"):
            detached._atomic_write(run_dir / detached.STATUS_FILE, {"state": "forbidden"})
        assert custody.identity["st_ino"] == displaced.stat().st_ino
    assert list(run_dir.iterdir()) == []
    assert not (displaced / detached.STATUS_FILE).exists()


def test_custodied_detached_write_rejects_nested_symlink_without_redirect(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    outside.mkdir()
    with detached._run_directory_custody(run_dir, create=True):
        (run_dir / "nested").symlink_to(outside, target_is_directory=True)
        with pytest.raises(detached.DetachedStepError, match="custodied artifact write failed"):
            detached._atomic_write(run_dir / "nested" / "status.json", {"state": "forbidden"})
    assert list(outside.iterdir()) == []


def test_supervisor_fault_after_target_release_cleans_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "fault-cleanup"
    monkeypatch.setenv("AURA_DETACHED_TEST_FAULT_POINT", "after_target_release")
    _launch(run_dir, [sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=60.0)
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE)
    assert receipt["status"] == "supervisor_failed"
    assert receipt["process_group_empty"] is True
    assert receipt["descendant_cleanup_performed"] is True
    assert detached._identity_state(receipt["child_pid"], receipt["child_start_token"]) == "dead"


def test_kernel_policy_rejects_term_ignoring_grandchild(tmp_path: Path) -> None:
    run_dir = tmp_path / "grandchild-rejected"
    grandchild_pid = tmp_path / "grandchild.pid"
    command = [
        sys.executable,
        "-c",
        (
            "import pathlib,signal,subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c',"
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']); "
            f"pathlib.Path({str(grandchild_pid)!r}).write_text(str(p.pid))"
        ),
    ]
    _launch(run_dir, command, timeout_s=60.0)
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE, timeout_s=15.0)
    assert receipt["status"] == "failed"
    assert receipt["returncode"] != 0
    assert receipt["fork_policy"] == "kernel_denied"
    assert receipt["containment_verified"] is True
    assert receipt["process_group_empty"] is True
    assert not grandchild_pid.exists()


def test_environment_stripping_new_session_descendant_is_kernel_denied(tmp_path: Path) -> None:
    run_dir = tmp_path / "escaped-lineage-denied"
    outcome = tmp_path / "escaped.outcome"
    command = [
        sys.executable,
        "-c",
        (
            "exec(\"import os, pathlib, subprocess, sys\\n"
            f"outcome = pathlib.Path({str(outcome)!r})\\n"
            "try:\\n"
            "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
            "start_new_session=True, env={'PATH': os.environ['PATH']})\\n"
            "except PermissionError:\\n"
            "    outcome.write_text('kernel-denied')\\n"
            "else:\\n"
            "    outcome.write_text(f'escaped:{child.pid}')\")"
        ),
    ]
    _launch(run_dir, command, timeout_s=60.0)
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE, timeout_s=15.0)
    assert receipt["status"] == "passed"
    assert receipt["containment_verified"] is True
    assert receipt["lineage_empty"] is True
    assert receipt["fork_policy"] == "kernel_denied"
    assert outcome.read_text(encoding="utf-8") == "kernel-denied"


def test_terminal_duration_uses_monotonic_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = detached._build_plan("clock", [sys.executable, "-c", "pass"], tmp_path, 5.0, "none")
    monkeypatch.setattr(detached.time, "time", lambda: 10.0)
    monkeypatch.setattr(detached.time, "monotonic_ns", lambda: 9_000_000_000)
    receipt = detached._terminal_receipt(
        plan=plan,
        attempt=1,
        supervisor_pid=100,
        supervisor_start_token="token",
        child_pid=0,
        child_process_group_id=0,
        child_start_token="",
        started_at=999.0,
        started_monotonic_ns=1_000_000_000,
        returncode=0,
        timed_out=False,
        stop_signal=None,
        descendant_cleanup_performed=False,
        lineage_cleanup_count=0,
        containment_verified=True,
        supervisor_error=None,
    )
    assert receipt["finished_at"] == 10.0
    assert receipt["duration_s"] == 8.0


def test_checkpoint_contract_reconciles_indeterminate_completed_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "completed-before-receipt"
    invocations = tmp_path / "invocations.txt"
    durable_effect = tmp_path / "durable-effect.txt"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            f"calls=Path({str(invocations)!r}); effect=Path({str(durable_effect)!r}); "
            "calls.write_text((calls.read_text() if calls.exists() else '')+'call\\n'); "
            "effect.write_text('once') if not effect.exists() else None"
        ),
    ]
    monkeypatch.setenv("AURA_DETACHED_TEST_CRASH_POINT", "after_target_exit")
    first = _launch(
        run_dir,
        command,
        timeout_s=60.0,
        resume_contract="target_checkpoint",
    )
    _wait_for_text(durable_effect, "once")
    deadline = time.time() + 5.0
    while time.time() < deadline and detached._pid_matches(
        first["supervisor_pid"], first["supervisor_start_token"]
    ):
        time.sleep(0.05)
    assert not (run_dir / detached.RECEIPT_FILE).exists()
    assert detached._status(run_dir)["completion_indeterminate"] is True

    monkeypatch.delenv("AURA_DETACHED_TEST_CRASH_POINT")
    resumed = _launch(
        run_dir,
        command,
        timeout_s=60.0,
        resume=True,
        resume_contract="target_checkpoint",
    )
    assert resumed["prior_completion_indeterminate"] is True
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE)
    assert receipt["status"] == "passed"
    assert invocations.read_text(encoding="utf-8") == "call\ncall\n"
    assert durable_effect.read_text(encoding="utf-8") == "once"


def test_unobservable_process_identity_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        detached,
        "_inspect_process",
        lambda _pid: detached.ProcessObservation("unknown"),
    )
    assert detached._identity_state(123, "token") == "unknown"
    with pytest.raises(detached.DetachedStepError, match="unobservable"):
        detached._wait_for_pid_exit(123, "token", 0.01)

    monkeypatch.setattr(
        detached,
        "_inspect_process",
        lambda _pid: detached.ProcessObservation("alive", token=""),
    )
    assert detached._identity_state(123, "token") == "unknown"


def test_reaped_direct_child_does_not_depend_on_libproc_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    containment_token = "a" * 64
    child = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        start_new_session=True,
    )
    child_token = detached._process_start_token(child.pid)
    process_group_id = child.pid
    assert child.wait(timeout=5.0) == 0

    monkeypatch.setattr(
        detached,
        "_inspect_process",
        lambda _pid: detached.ProcessObservation("unknown"),
    )
    cleanup_performed, lineage_cleanup_count = detached._cleanup_child_process(
        child,
        child_token,
        process_group_id,
        containment_token,
    )

    assert cleanup_performed is False
    assert lineage_cleanup_count == 0


def test_direct_child_observation_tolerates_transient_libproc_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        start_new_session=True,
    )
    original_inspect = detached._inspect_process
    child_token = detached._process_start_token(child.pid)
    calls = 0

    def transient_observation(pid: int) -> detached.ProcessObservation:
        nonlocal calls
        if pid == child.pid and calls < 5:
            calls += 1
            return detached.ProcessObservation("unknown")
        return original_inspect(pid)

    monkeypatch.setattr(detached, "_inspect_process", transient_observation)
    try:
        observation, returncode = detached._observe_direct_child(
            child,
            child_token,
            1.0,
        )
        assert observation.state == "alive"
        assert observation.token == child_token
        assert returncode is None
        assert calls == 5
    finally:
        child.terminate()
        child.wait(timeout=5.0)


def test_direct_child_exit_during_libproc_gap_is_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.05)"],
        start_new_session=True,
    )
    child_token = detached._process_start_token(child.pid)
    monkeypatch.setattr(
        detached,
        "_inspect_process",
        lambda _pid: detached.ProcessObservation("unknown"),
    )

    observation, returncode = detached._observe_direct_child(
        child,
        child_token,
        1.0,
    )

    assert observation.state == "dead"
    assert returncode == 0


def test_attempt_journal_tampering_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "tampered-journal"
    _launch(run_dir, [sys.executable, "-c", "pass"])
    _wait_for(run_dir / detached.RECEIPT_FILE)
    journal_path = run_dir / detached.ATTEMPTS_FILE
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["supervisor_pid"] += 1
    lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(detached.DetachedStepError, match="journal hash mismatch"):
        detached._status(run_dir)


def test_plan_freezes_absolute_executable_and_secret_free_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-cross-boundary")
    plan = detached._build_plan(
        "frozen-execution",
        [sys.executable, "-c", "pass"],
        tmp_path,
        5.0,
        "none",
    )
    assert Path(plan["command"][0]).is_absolute()
    assert plan["executable_sha256"] == detached._sha256_file(Path(plan["command"][0]))
    assert plan["executable_binding"]["resolved_sha256"] == plan["executable_sha256"]
    assert "ANTHROPIC_API_KEY" not in plan["execution_environment"]
    assert "must-not-cross-boundary" not in json.dumps(plan)


def test_precontained_sandbox_plan_is_bound_and_not_nested(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "contained.sb"
    profile.write_text(
        "\n".join(
            (
                "(version 1)",
                "(deny default)",
                "(deny network*)",
                "(deny process-fork)",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    command = [
        str(detached._DARWIN_SANDBOX),
        "-f",
        str(profile),
        sys.executable,
        "-c",
        "pass",
    ]
    plan = detached._build_plan(
        "precontained",
        command,
        tmp_path,
        5.0,
        "none",
        containment_mode=detached._PRECONTAINED_SANDBOX_MODE,
    )
    assert plan["execution_sandbox"]["mode"] == "precontained-sandbox"
    assert plan["execution_sandbox"]["profile_path"] == str(profile)
    assert detached._executed_command(plan) == plan["command"]
    assert [
        root
        for root in plan["target_execution_manifest"]["roots"]
        if root.get("path") == str(profile)
    ]
    detached._verify_plan(plan, tmp_path / "detached_plan.json")

    profile.write_text(
        "(version 1)\n(deny default)\n(deny network*)\n"
        "(deny process-fork)\n# changed\n",
        encoding="utf-8",
    )
    with pytest.raises(
        detached.DetachedStepError,
        match="precontained sandbox profile drift",
    ):
        detached._verify_plan(plan, tmp_path / "detached_plan.json")


def test_precontained_sandbox_rejects_allow_default_profile(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "unsafe.sb"
    profile.write_text(
        "(version 1)\n(allow default)\n(deny default)\n"
        "(deny network*)\n(deny process-fork)\n",
        encoding="utf-8",
    )
    with pytest.raises(
        detached.DetachedStepError,
        match="lacks the required deny boundary",
    ):
        detached._build_plan(
            "unsafe-precontained",
            [
                str(detached._DARWIN_SANDBOX),
                "-f",
                str(profile),
                sys.executable,
                "-c",
                "pass",
            ],
            tmp_path,
            5.0,
            "none",
            containment_mode=detached._PRECONTAINED_SANDBOX_MODE,
        )


def test_target_receives_exact_detached_evidence_paths_and_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "evidence-environment"
    observed_path = tmp_path / "observed-environment.json"
    keys = [
        "AURA_DETACHED_RUN_DIR",
        "AURA_DETACHED_PLAN_PATH",
        "AURA_DETACHED_ATTEMPTS_PATH",
        "AURA_DETACHED_PLAN_SHA256",
        "AURA_DETACHED_SUPERVISOR_ATTEMPT",
    ]
    script = (
        "import json,os,pathlib; "
        f"keys={keys!r}; "
        f"pathlib.Path({str(observed_path)!r}).write_text("
        "json.dumps({key:os.environ[key] for key in keys},sort_keys=True))"
    )

    _launch(run_dir, [sys.executable, "-c", script])
    _wait_for(run_dir / detached.RECEIPT_FILE)
    observed = json.loads(observed_path.read_text(encoding="utf-8"))
    plan = json.loads((run_dir / detached.PLAN_FILE).read_text(encoding="utf-8"))

    assert observed == {
        "AURA_DETACHED_RUN_DIR": str(run_dir),
        "AURA_DETACHED_PLAN_PATH": str(run_dir / detached.PLAN_FILE),
        "AURA_DETACHED_ATTEMPTS_PATH": str(run_dir / detached.ATTEMPTS_FILE),
        "AURA_DETACHED_PLAN_SHA256": plan["plan_sha256"],
        "AURA_DETACHED_SUPERVISOR_ATTEMPT": "1",
    }


def test_command_resolution_preserves_virtualenv_launcher(tmp_path: Path) -> None:
    environment = detached._frozen_environment()
    environment["PATH"] = str(tmp_path / "venv" / "bin")
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    launcher = bin_dir / "python"
    launcher.symlink_to(Path(sys.executable))
    pyvenv = tmp_path / "venv" / "pyvenv.cfg"
    pyvenv.write_text("home = /frozen/interpreter\n", encoding="utf-8")

    resolved = detached._resolve_command([str(launcher), "-c", "pass"], tmp_path, environment)
    assert resolved[0] == str(launcher)
    binding = detached._launcher_binding(Path(resolved[0]))
    assert binding["invocation_kind"] == "symlink"
    assert binding["resolved_path"] == str(Path(sys.executable).resolve())
    assert binding["pyvenv"]["sha256"] == detached._sha256_file(pyvenv)

    pyvenv.write_text("home = /mutated/interpreter\n", encoding="utf-8")
    with pytest.raises(detached.DetachedStepError, match="launcher binding changed"):
        detached._verify_launcher_binding(binding, launcher)


def test_execution_manifest_detects_interpreted_script_mutation(tmp_path: Path) -> None:
    script = tmp_path / "target.py"
    script.write_text("print('first')\n", encoding="utf-8")
    plan = detached._build_plan(
        "script-freeze",
        [sys.executable, str(script)],
        tmp_path,
        5.0,
        "none",
    )
    detached._verify_execution_manifest_current(plan["target_execution_manifest"])

    script.write_text("print('mutated')\n", encoding="utf-8")
    with pytest.raises(detached.DetachedStepError, match="execution source changed"):
        detached._verify_execution_manifest_current(plan["target_execution_manifest"])


def test_execution_manifest_allows_bound_run_artifacts_but_not_other_source(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    script = repository / "target.py"
    script.write_text("print('stable')\n", encoding="utf-8")
    subprocess.run(["git", "add", "target.py"], cwd=repository, check=True)
    run_dir = repository / "artifacts" / "run"
    plan = detached._build_plan(
        "run-artifact-freeze",
        [sys.executable, str(script)],
        repository,
        5.0,
        "none",
        execution_exclusion_roots=(run_dir,),
    )

    run_dir.mkdir(parents=True)
    (run_dir / "detached_plan.json").write_text("{}\n", encoding="utf-8")
    detached._verify_execution_manifest_current(plan["target_execution_manifest"])

    (repository / "late_source.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
    with pytest.raises(detached.DetachedStepError, match="execution source changed"):
        detached._verify_execution_manifest_current(plan["target_execution_manifest"])


def test_execution_manifest_ignores_unrelated_runtime_json_but_binds_explicit_input(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    script = repository / "target.py"
    script.write_text("print('stable')\n", encoding="utf-8")
    subprocess.run(["git", "add", "target.py"], cwd=repository, check=True)
    live_status = repository / "detached_status.json"
    live_status.write_text('{"heartbeat":1}\n', encoding="utf-8")

    unrelated_plan = detached._build_plan(
        "runtime-json-exclusion",
        [sys.executable, str(script)],
        repository,
        5.0,
        "none",
    )
    live_status.write_text('{"heartbeat":2}\n', encoding="utf-8")
    detached._verify_execution_manifest_current(
        unrelated_plan["target_execution_manifest"]
    )

    explicit_config = repository / "explicit-config.json"
    explicit_config.write_text('{"mode":"first"}\n', encoding="utf-8")
    explicit_plan = detached._build_plan(
        "explicit-json-binding",
        [sys.executable, str(script), str(explicit_config)],
        repository,
        5.0,
        "none",
    )
    explicit_config.write_text('{"mode":"other"}\n', encoding="utf-8")
    with pytest.raises(detached.DetachedStepError, match="execution source changed"):
        detached._verify_execution_manifest_current(
            explicit_plan["target_execution_manifest"]
        )


def test_execution_manifest_rejects_excluding_tracked_source(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    source_dir = repository / "tools"
    source_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    script = source_dir / "target.py"
    script.write_text("print('stable')\n", encoding="utf-8")
    subprocess.run(["git", "add", "tools/target.py"], cwd=repository, check=True)

    with pytest.raises(detached.DetachedStepError, match="target source"):
        detached._build_plan(
            "unsafe-run-exclusion",
            [sys.executable, str(script)],
            repository,
            5.0,
            "none",
            execution_exclusion_roots=(source_dir,),
        )


def test_execution_manifest_binds_explicit_config_inside_output_root(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    output_root = repository / "artifacts" / "run"
    output_root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    script = repository / "target.py"
    config = output_root / "plan.json"
    script.write_text("raise SystemExit(0)\n", encoding="ascii")
    config.write_text('{"mode":"first"}\n', encoding="ascii")
    subprocess.run(["git", "add", "target.py"], cwd=repository, check=True)

    plan = detached._build_plan(
        "explicit-output-config",
        [sys.executable, str(script), str(config)],
        repository,
        5.0,
        "none",
        execution_exclusion_roots=(output_root,),
    )
    manifest = plan["target_execution_manifest"]

    assert any(
        root.get("kind") == "file" and root.get("path") == str(config)
        for root in manifest["roots"]
    )
    detached._verify_execution_manifest_current(manifest)
    config.write_text('{"mode":"changed"}\n', encoding="ascii")
    with pytest.raises(detached.DetachedStepError, match="execution source changed"):
        detached._verify_execution_manifest_current(manifest)


def test_detached_launch_allows_declared_generated_source_output(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    script = repository / "target.py"
    output_root = repository / "artifacts" / "training"
    script.write_text(
        (
            "from pathlib import Path\n"
            f"root = Path({str(output_root)!r})\n"
            "root.mkdir(parents=True)\n"
            "(root / 'source_snapshot.py').write_text(\"stable = True\\n\")\n"
            "import time\n"
            "time.sleep(1.5)\n"
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "target.py"], cwd=repository, check=True)

    launch = _launch(
        repository / "artifacts" / "detached",
        [sys.executable, str(script)],
        timeout_s=10.0,
        cwd=repository,
        execution_output_roots=[output_root],
    )
    receipt = _wait_for(Path(launch["receipt_path"]), timeout_s=10.0)

    assert receipt["status"] == "passed"
    plan = detached._read_json(Path(launch["run_dir"]) / detached.PLAN_FILE)
    assert str(output_root) in plan["target_execution_manifest"]["excluded_roots"]


def test_mutated_resume_verifier_is_rejected_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "mutated-verifier"
    verifier_script = tmp_path / "resume_verifier.py"
    verifier_script.write_text("raise SystemExit(2)\n", encoding="utf-8")
    resume_verifier = [sys.executable, str(verifier_script)]
    monkeypatch.setenv("AURA_DETACHED_TEST_CRASH_POINT", "after_target_exit")
    launched = _launch(
        run_dir,
        [sys.executable, "-c", "pass"],
        timeout_s=30.0,
        resume_contract="target_checkpoint",
        resume_verifier=resume_verifier,
    )
    deadline = time.time() + 8.0
    while time.time() < deadline and detached._pid_matches(
        launched["supervisor_pid"], launched["supervisor_start_token"]
    ):
        time.sleep(0.05)
    monkeypatch.delenv("AURA_DETACHED_TEST_CRASH_POINT")
    verifier_script.write_text("print('replacement approved')\n", encoding="utf-8")

    with pytest.raises(subprocess.CalledProcessError) as raised:
        _launch(
            run_dir,
            [sys.executable, "-c", "pass"],
            timeout_s=30.0,
            resume=True,
            resume_contract="target_checkpoint",
            resume_verifier=resume_verifier,
        )
    assert "existing detached plan differs" in raised.value.stderr


def test_stale_resume_verdict_cannot_authorize_later_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "stale-verdict"
    resume_verifier = _replaying_resume_verifier(tmp_path / "stale-verdict.state")
    command = [sys.executable, "-c", "pass"]
    monkeypatch.setenv("AURA_DETACHED_TEST_CRASH_POINT", "after_target_exit")
    first = _launch(
        run_dir,
        command,
        timeout_s=30.0,
        resume_contract="target_checkpoint",
        resume_verifier=resume_verifier,
    )
    deadline = time.time() + 8.0
    while time.time() < deadline and detached._pid_matches(
        first["supervisor_pid"], first["supervisor_start_token"]
    ):
        time.sleep(0.05)

    second = _launch(
        run_dir,
        command,
        timeout_s=30.0,
        resume=True,
        resume_contract="target_checkpoint",
        resume_verifier=resume_verifier,
    )
    deadline = time.time() + 8.0
    while time.time() < deadline and detached._pid_matches(
        second["supervisor_pid"], second["supervisor_start_token"]
    ):
        time.sleep(0.05)
    monkeypatch.delenv("AURA_DETACHED_TEST_CRASH_POINT")

    with pytest.raises(subprocess.CalledProcessError) as raised:
        _launch(
            run_dir,
            command,
            timeout_s=30.0,
            resume=True,
            resume_contract="target_checkpoint",
            resume_verifier=resume_verifier,
        )
    assert "verdict binding is invalid" in raised.value.stderr


def test_compact_broker_request_resolves_exact_large_frozen_policy(tmp_path: Path) -> None:
    command = [sys.executable, "x" * 10_000]
    stdout_path = tmp_path / "worker.log"
    policy = detached._build_broker_policy(
        [
            {
                "command": command,
                "cwd": str(tmp_path),
                "stdout_path": str(stdout_path),
                "timeout_s_max": 30.0,
                "max_invocations": 1,
            }
        ],
        detached._frozen_environment(),
    )[0]
    request = {
        "schema": detached.broker_protocol.REQUEST_SCHEMA,
        "action": "run",
        "command_sha256": policy["command_sha256"],
        "request_binding_sha256": detached.broker_protocol.compute_broker_request_binding(
            policy["command"],
            cwd=policy["cwd"],
            stdout_path=policy["stdout_path"],
        ),
        "timeout_s": 30.0,
    }

    assert detached._matching_broker_policy({"broker_policy": [policy]}, request) == policy
    assert len(detached._canonical_bytes(request)) < 2_048
    request["request_binding_sha256"] = "0" * 64
    with pytest.raises(detached.BrokerRequestError, match="exceeds its frozen policy"):
        detached._matching_broker_policy({"broker_policy": [policy]}, request)


def test_exact_broker_policy_runs_worker_inside_strict_target_boundary(tmp_path: Path) -> None:
    run_dir = tmp_path / "brokered-run"
    worker_log = tmp_path / "broker-worker.log"
    coordinator_result = tmp_path / "coordinator-result.txt"
    repo_root = Path(detached.__file__).resolve().parent.parent
    python = str(Path(sys.executable).resolve())
    worker_command = [python, "-c", "print('broker-worker-ok', flush=True)"]
    coordinator_code = (
        "from pathlib import Path; "
        "from core.runtime.detached_subprocess_broker import run_brokered_process; "
        f"result=run_brokered_process({worker_command!r}, cwd=Path({str(repo_root)!r}), "
        f"stdout_path=Path({str(worker_log)!r}), timeout_s=10.0); "
        f"Path({str(coordinator_result)!r}).write_text(str(result.returncode))"
    )
    broker_policy = [
        {
            "command": worker_command,
            "cwd": str(repo_root),
            "stdout_path": str(worker_log),
            "timeout_s_max": 10.0,
            "max_invocations": 1,
        }
    ]

    _launch(
        run_dir,
        [python, "-c", coordinator_code],
        timeout_s=30.0,
        broker_policy=broker_policy,
        cwd=repo_root,
    )
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE, timeout_s=20.0)
    assert receipt["status"] == "passed"
    assert receipt["containment_verified"] is True
    assert coordinator_result.read_text(encoding="utf-8") == "0"
    assert "broker-worker-ok" in worker_log.read_text(encoding="utf-8")
    events = detached._read_attempts(run_dir)
    event_types = [event["event"] for event in events]
    assert event_types == [
        "LAUNCHED",
        "CONTROL_READY",
        "TARGET_STARTED",
        "BROKER_STARTED",
        "BROKER_TERMINAL",
        "TERMINAL",
    ]
    broker_terminal = next(event for event in events if event["event"] == "BROKER_TERMINAL")
    assert broker_terminal["response"]["status"] == "passed"
    assert broker_terminal["response"]["containment_verified"] is True
    assert len(broker_terminal["response"]["response_hmac_sha256"]) == 64


def test_broker_rejects_command_outside_frozen_policy(tmp_path: Path) -> None:
    run_dir = tmp_path / "broker-rejection"
    worker_log = tmp_path / "broker-rejection.log"
    outcome = tmp_path / "broker-rejection.txt"
    repo_root = Path(detached.__file__).resolve().parent.parent
    python = str(Path(sys.executable).resolve())
    allowed = [python, "-c", "print('allowed')"]
    denied = [python, "-c", "print('denied')"]
    coordinator_code = (
        "from pathlib import Path\n"
        "from core.runtime.detached_subprocess_broker import DetachedBrokerError, run_brokered_process\n"
        "try:\n"
        f"    run_brokered_process({denied!r}, cwd=Path({str(repo_root)!r}), "
        f"stdout_path=Path({str(worker_log)!r}), timeout_s=5.0)\n"
        "except DetachedBrokerError:\n"
        f"    Path({str(outcome)!r}).write_text('rejected')\n"
    )
    policy = [{
        "command": allowed,
        "cwd": str(repo_root),
        "stdout_path": str(worker_log),
        "timeout_s_max": 5.0,
        "max_invocations": 1,
    }]

    _launch(
        run_dir,
        [python, "-c", coordinator_code],
        timeout_s=20.0,
        broker_policy=policy,
        cwd=repo_root,
    )
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE, timeout_s=15.0)
    assert receipt["status"] == "passed"
    assert outcome.read_text(encoding="utf-8") == "rejected"
    assert not any(
        event["event"] == "BROKER_STARTED" for event in detached._read_attempts(run_dir)
    )


def test_broker_worker_cannot_fork_or_escape_its_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "broker-worker-no-fork"
    worker_log = tmp_path / "broker-worker-no-fork.log"
    worker_outcome = tmp_path / "broker-worker-no-fork.txt"
    coordinator_outcome = tmp_path / "broker-worker-coordinator.txt"
    repo_root = Path(detached.__file__).resolve().parent.parent
    python = str(Path(sys.executable).resolve())
    worker_source = (
        "from pathlib import Path\n"
        "import subprocess, sys\n"
        "try:\n"
        "    subprocess.run([sys.executable, '-c', 'pass'], check=True)\n"
        "except PermissionError:\n"
        f"    Path({str(worker_outcome)!r}).write_text('kernel-denied')\n"
    )
    worker = [python, "-c", worker_source]
    coordinator_code = (
        "from pathlib import Path; "
        "from core.runtime.detached_subprocess_broker import run_brokered_process; "
        f"result=run_brokered_process({worker!r}, cwd=Path({str(repo_root)!r}), "
        f"stdout_path=Path({str(worker_log)!r}), timeout_s=5.0); "
        f"Path({str(coordinator_outcome)!r}).write_text(str(result.returncode))"
    )
    policy = [{
        "command": worker,
        "cwd": str(repo_root),
        "stdout_path": str(worker_log),
        "timeout_s_max": 5.0,
        "max_invocations": 1,
    }]

    _launch(
        run_dir,
        [python, "-c", coordinator_code],
        timeout_s=20.0,
        broker_policy=policy,
        cwd=repo_root,
    )
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE, timeout_s=15.0)
    assert receipt["status"] == "passed"
    assert coordinator_outcome.read_text(encoding="utf-8") == "0"
    assert worker_outcome.read_text(encoding="utf-8") == "kernel-denied"


def test_broker_log_substitution_is_rejected_without_touching_victim(tmp_path: Path) -> None:
    run_dir = tmp_path / "broker-log-substitution"
    worker_log = tmp_path / "broker-log-substitution.log"
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged", encoding="utf-8")
    repo_root = Path(detached.__file__).resolve().parent.parent
    python = str(Path(sys.executable).resolve())
    worker = [python, "-c", "print('must-not-run')"]
    coordinator_code = (
        "from pathlib import Path; import os; "
        "from core.runtime.detached_subprocess_broker import run_brokered_process; "
        f"os.symlink({str(victim)!r}, {str(worker_log)!r}); "
        f"run_brokered_process({worker!r}, cwd=Path({str(repo_root)!r}), "
        f"stdout_path=Path({str(worker_log)!r}), timeout_s=5.0)"
    )
    policy = [{
        "command": worker,
        "cwd": str(repo_root),
        "stdout_path": str(worker_log),
        "timeout_s_max": 5.0,
        "max_invocations": 1,
    }]

    _launch(
        run_dir,
        [python, "-c", coordinator_code],
        timeout_s=20.0,
        broker_policy=policy,
        cwd=repo_root,
    )
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE, timeout_s=15.0)
    assert receipt["status"] == "failed"
    assert receipt["containment_verified"] is True
    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert not any(
        event["event"] == "BROKER_STARTED" for event in detached._read_attempts(run_dir)
    )


def test_broker_enforces_invocation_bound(tmp_path: Path) -> None:
    run_dir = tmp_path / "broker-bound"
    worker_log = tmp_path / "broker-bound.log"
    outcome = tmp_path / "broker-bound.txt"
    repo_root = Path(detached.__file__).resolve().parent.parent
    python = str(Path(sys.executable).resolve())
    worker = [python, "-c", "print('bounded')"]
    coordinator_code = (
        "from pathlib import Path\n"
        "from core.runtime.detached_subprocess_broker import DetachedBrokerError, run_brokered_process\n"
        f"command={worker!r}\n"
        f"kwargs={{'cwd':Path({str(repo_root)!r}),'stdout_path':Path({str(worker_log)!r}),'timeout_s':5.0}}\n"
        "first=run_brokered_process(command, **kwargs)\n"
        "try:\n"
        "    run_brokered_process(command, **kwargs)\n"
        "except DetachedBrokerError:\n"
        f"    Path({str(outcome)!r}).write_text(f'{{first.returncode}}:bounded')\n"
    )
    policy = [{
        "command": worker,
        "cwd": str(repo_root),
        "stdout_path": str(worker_log),
        "timeout_s_max": 5.0,
        "max_invocations": 1,
    }]

    _launch(
        run_dir,
        [python, "-c", coordinator_code],
        timeout_s=30.0,
        broker_policy=policy,
        cwd=repo_root,
    )
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE, timeout_s=20.0)
    assert receipt["status"] == "passed"
    assert outcome.read_text(encoding="utf-8") == "0:bounded"
    assert sum(
        event["event"] == "BROKER_STARTED" for event in detached._read_attempts(run_dir)
    ) == 1


def test_broker_timeout_kills_worker_and_returns_bounded_result(tmp_path: Path) -> None:
    run_dir = tmp_path / "broker-timeout"
    worker_log = tmp_path / "broker-timeout.log"
    outcome = tmp_path / "broker-timeout.txt"
    worker_pid_path = tmp_path / "broker-timeout.pid"
    repo_root = Path(detached.__file__).resolve().parent.parent
    python = str(Path(sys.executable).resolve())
    worker = [
        python,
        "-c",
        f"import os,time; from pathlib import Path; Path({str(worker_pid_path)!r}).write_text(str(os.getpid())); time.sleep(30)",
    ]
    coordinator_code = (
        "from pathlib import Path; "
        "from core.runtime.detached_subprocess_broker import run_brokered_process; "
        f"result=run_brokered_process({worker!r}, cwd=Path({str(repo_root)!r}), "
        f"stdout_path=Path({str(worker_log)!r}), timeout_s=0.5); "
        f"Path({str(outcome)!r}).write_text(f'{{result.returncode}}:{{result.timed_out}}')"
    )
    policy = [{
        "command": worker,
        "cwd": str(repo_root),
        "stdout_path": str(worker_log),
        "timeout_s_max": 0.5,
        "max_invocations": 1,
    }]

    _launch(
        run_dir,
        [python, "-c", coordinator_code],
        timeout_s=30.0,
        broker_policy=policy,
        cwd=repo_root,
    )
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE, timeout_s=20.0)
    assert receipt["status"] == "passed"
    assert outcome.read_text(encoding="utf-8") == "124:True"
    worker_pid = int(worker_pid_path.read_text(encoding="utf-8"))
    assert detached._inspect_process(worker_pid).state == "dead"
    terminal = next(
        event for event in detached._read_attempts(run_dir) if event["event"] == "BROKER_TERMINAL"
    )
    assert terminal["response"]["status"] == "timed_out"
    assert terminal["response"]["containment_verified"] is True


def test_resume_reaps_broker_worker_after_supervisor_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "broker-crash-resume"
    worker_log = tmp_path / "broker-crash-resume.log"
    worker_pid_path = tmp_path / "broker-crash-resume.pid"
    outcome = tmp_path / "broker-crash-resume.txt"
    repo_root = Path(detached.__file__).resolve().parent.parent
    python = str(Path(sys.executable).resolve())
    worker = [
        python,
        "-c",
        f"import os,time; from pathlib import Path; Path({str(worker_pid_path)!r}).write_text(str(os.getpid())); time.sleep(30)",
    ]
    coordinator_code = (
        "from pathlib import Path; "
        "from core.runtime.detached_subprocess_broker import run_brokered_process; "
        f"marker=Path({str(worker_pid_path)!r}); outcome=Path({str(outcome)!r}); "
        f"run_brokered_process({worker!r}, cwd=Path({str(repo_root)!r}), "
        f"stdout_path=Path({str(worker_log)!r}), timeout_s=20.0) if not marker.exists() else None; "
        "outcome.write_text('resumed')"
    )
    policy = [{
        "command": worker,
        "cwd": str(repo_root),
        "stdout_path": str(worker_log),
        "timeout_s_max": 20.0,
        "max_invocations": 1,
    }]
    monkeypatch.setenv("AURA_DETACHED_TEST_CRASH_POINT", "after_broker_release")
    first = _launch(
        run_dir,
        [python, "-c", coordinator_code],
        timeout_s=60.0,
        resume_contract="target_checkpoint",
        broker_policy=policy,
        cwd=repo_root,
    )
    marker_deadline = time.time() + 8.0
    while time.time() < marker_deadline and not worker_pid_path.is_file():
        time.sleep(0.05)
    assert worker_pid_path.is_file()
    deadline = time.time() + 8.0
    while time.time() < deadline and detached._pid_matches(
        first["supervisor_pid"], first["supervisor_start_token"]
    ):
        time.sleep(0.05)
    stale_worker_pid = int(worker_pid_path.read_text(encoding="utf-8"))
    assert detached._inspect_process(stale_worker_pid).state == "alive"

    monkeypatch.delenv("AURA_DETACHED_TEST_CRASH_POINT")
    _launch(
        run_dir,
        [python, "-c", coordinator_code],
        timeout_s=60.0,
        resume=True,
        resume_contract="target_checkpoint",
        broker_policy=policy,
        cwd=repo_root,
    )
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE, timeout_s=20.0)
    assert receipt["status"] == "passed"
    assert outcome.read_text(encoding="utf-8") == "resumed"
    assert detached._inspect_process(stale_worker_pid).state == "dead"


def test_resume_quarantines_worker_origin_after_supervisor_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    repo_root = Path(detached.__file__).resolve().parent.parent
    python = str(Path(sys.executable).resolve())
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)

    worker_pid_path = tmp_path / "worker.pid"
    worker_log = tmp_path / "worker.log"
    coordinator_outcome = tmp_path / "coordinator.outcome"
    authorization_wait = tmp_path / "authorization.wait"
    worker_script = workspace / "worker.py"
    coordinator_script = workspace / "coordinator.py"
    worker_script.write_text(
        "\n".join(
            [
                "import os",
                "import pathlib",
                "import time",
                f"pathlib.Path({str(worker_pid_path)!r}).write_text(str(os.getpid()))",
                "time.sleep(30)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    worker_command = [python, str(worker_script)]
    coordinator_script.write_text(
        "\n".join(
            [
                "import pathlib",
                "import sys",
                "import time",
                f"sys.path.insert(0, {str(repo_root)!r})",
                (
                    "from core.runtime.detached_subprocess_broker "
                    "import DetachedBrokerError, run_brokered_process"
                ),
                f"worker_marker = pathlib.Path({str(worker_pid_path)!r})",
                f"outcome = pathlib.Path({str(coordinator_outcome)!r})",
                f"command = {worker_command!r}",
                "if not worker_marker.exists():",
                "    while True:",
                "        try:",
                (
                    "            run_brokered_process("
                    f"command, cwd=pathlib.Path({str(workspace)!r}), "
                    f"stdout_path=pathlib.Path({str(worker_log)!r}), "
                    "timeout_s=20.0)"
                ),
                "        except DetachedBrokerError as exc:",
                (
                    "            if 'external authorization required at ' "
                    "not in str(exc):"
                ),
                "                raise",
                (
                    f"            pathlib.Path({str(authorization_wait)!r})"
                    ".write_text(str(exc), encoding='utf-8')"
                ),
                "            time.sleep(0.05)",
                "            continue",
                "        break",
                "outcome.write_text('resumed', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "worker.py", "coordinator.py"],
        cwd=workspace,
        check=True,
    )

    trust_policy_path, trust_root_path, trust_policy, runner_key = (
        _worker_origin_trust_fixture(tmp_path / "trust")
    )
    origin_dir = tmp_path / "origins"
    run_dir = tmp_path / "run"
    broker_policy = _worker_origin_policy(
        command=worker_command,
        cwd=workspace,
        stdout_path=worker_log,
        trust_policy_path=trust_policy_path,
        trust_root_path=trust_root_path,
        artifact_dir=origin_dir,
        timeout_s=20.0,
    )

    monkeypatch.setenv(
        "AURA_DETACHED_TEST_CRASH_POINT",
        "after_broker_release",
    )
    first = _launch(
        run_dir,
        [python, str(coordinator_script)],
        timeout_s=60.0,
        resume_contract="target_checkpoint",
        broker_policy=broker_policy,
        cwd=workspace,
    )
    request_path = _wait_for_glob(origin_dir, "*.request.json", timeout_s=15.0)
    _wait_for_glob(tmp_path, authorization_wait.name, timeout_s=5.0)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    signature = runner_key.sign(
        base64.b64decode(request["signed_payload_b64"], validate=True)
    )
    attestation = assemble_role_attestation(
        trust_policy,
        request,
        signature_b64=base64.b64encode(signature).decode("ascii"),
        role=CAMPAIGN_RUNNER,
    )
    detached._atomic_write(
        request_path.with_name(
            request_path.name.replace(".request.json", ".attestation.json")
        ),
        attestation,
        replace=False,
    )

    _wait_for_glob(tmp_path, worker_pid_path.name, timeout_s=10.0)
    deadline = time.time() + 8.0
    while time.time() < deadline and detached._pid_matches(
        first["supervisor_pid"],
        first["supervisor_start_token"],
    ):
        time.sleep(0.05)
    stale_worker_pid = int(worker_pid_path.read_text(encoding="utf-8"))
    assert detached._inspect_process(stale_worker_pid).state == "alive"

    monkeypatch.setenv(
        "AURA_DETACHED_TEST_CRASH_POINT",
        "after_worker_origin_quarantine",
    )
    with pytest.raises(subprocess.CalledProcessError):
        _launch(
            run_dir,
            [python, str(coordinator_script)],
            timeout_s=60.0,
            resume=True,
            resume_contract="target_checkpoint",
            broker_policy=broker_policy,
            cwd=workspace,
        )
    assert detached._inspect_process(stale_worker_pid).state == "dead"
    crash_boundary_events = detached._read_attempts(run_dir)
    assert sum(
        event["event"] == "BROKER_ORIGIN_QUARANTINED"
        for event in crash_boundary_events
    ) == 1
    assert not any(
        event["event"] == "LAUNCHED" and event["attempt"] == 2
        for event in crash_boundary_events
    )

    monkeypatch.delenv("AURA_DETACHED_TEST_CRASH_POINT")
    _launch(
        run_dir,
        [python, str(coordinator_script)],
        timeout_s=60.0,
        resume=True,
        resume_contract="target_checkpoint",
        broker_policy=broker_policy,
        cwd=workspace,
    )
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE, timeout_s=20.0)
    assert receipt["status"] == "passed"
    assert coordinator_outcome.read_text(encoding="utf-8") == "resumed"
    assert detached._inspect_process(stale_worker_pid).state == "dead"

    events = detached._read_attempts(run_dir)
    quarantines = [
        event
        for event in events
        if event["event"] == "BROKER_ORIGIN_QUARANTINED"
    ]
    assert len(quarantines) == 1
    quarantine = quarantines[0]
    quarantine_receipt = quarantine["quarantine_receipt"]
    assert quarantine["attempt"] == 1
    assert quarantine_receipt["claim_eligible"] is False
    assert quarantine_receipt["lifecycle_recoverable"] is False
    assert quarantine_receipt["authority_key_recoverable"] is False
    assert quarantine_receipt["worker_identity_observed"] == "dead"
    assert quarantine_receipt["worker_process_group_empty"] is True
    assert quarantine_receipt["prior_journal_head_sha256"] == (
        quarantine["previous_event_sha256"]
    )
    assert not any(
        event["event"] == "BROKER_TERMINAL" and event["attempt"] == 1
        for event in events
    )
    assert any(
        event["event"] == "LAUNCHED" and event["attempt"] == 2
        for event in events
    )
    plan, _events, _status, _receipt = detached._verify_run_locked(run_dir)

    policy = plan["broker_policy"][0]
    contract = policy["worker_origin"]
    broker_start = next(
        event
        for event in events
        if event["event"] == "BROKER_STARTED" and event["attempt"] == 1
    )
    paths = detached._worker_origin_artifact_paths(
        contract,
        supervisor_attempt=1,
        broker_policy_sha256=policy["policy_sha256"],
    )
    authorization = json.loads(paths["payload"].read_text(encoding="utf-8"))
    forged_quarantine = json.loads(json.dumps(quarantine))
    forged_receipt = forged_quarantine["quarantine_receipt"]
    forged_receipt["claim_eligible"] = True
    forged_body = {
        key: value
        for key, value in forged_receipt.items()
        if key != "receipt_sha256"
    }
    forged_receipt["receipt_sha256"] = detached._sha256(forged_body)
    with pytest.raises(
        detached.DetachedStepError,
        match="quarantine receipt binding is invalid",
    ):
        detached._verify_persisted_worker_origin_quarantine(
            plan=plan,
            policy=policy,
            contract=contract,
            attempt=1,
            broker_start=broker_start,
            quarantine_event=forged_quarantine,
            authorization=authorization,
            paths=paths,
            lifecycle_artifact_sha256=None,
        )


def test_broker_worker_origin_is_externally_authorized_and_supervisor_signed(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    repo_root = Path(detached.__file__).resolve().parent.parent
    python = str(Path(sys.executable))
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)

    result_path = tmp_path / "signed-result.json"
    worker_log = tmp_path / "worker.log"
    coordinator_outcome = tmp_path / "coordinator.outcome"
    authorization_wait = tmp_path / "authorization.wait"
    worker_script = workspace / "worker.py"
    coordinator_script = workspace / "coordinator.py"
    worker_script.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "import pathlib",
                "import sys",
                f"sys.path.insert(0, {str(repo_root)!r})",
                (
                    "from core.runtime.detached_worker_origin_channel "
                    "import DetachedWorkerOriginChannelClient"
                ),
                (
                    "visible = {key: value for key, value in os.environ.items() "
                    "if key.startswith('AURA_DETACHED_WORKER_ORIGIN_')}"
                ),
                (
                    "assert set(visible) == "
                    "{'AURA_DETACHED_WORKER_ORIGIN_FD', "
                    "'AURA_DETACHED_WORKER_ORIGIN_SESSION'}"
                ),
                "assert not any('PRIVATE' in key or 'SIGNING' in key for key in visible)",
                "with DetachedWorkerOriginChannelClient.from_environment() as client:",
                (
                    "    result = client.record_result("
                    "{'answer': '42'}, cell_id='cell-0001', "
                    "cell_type='reasoning', attempt_id='attempt-0001')"
                ),
                (
                    f"pathlib.Path({str(result_path)!r}).write_text("
                    "json.dumps(result, sort_keys=True, separators=(',', ':')) + "
                    "'\\n', encoding='utf-8')"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    worker_command = [python, str(worker_script)]
    coordinator_script.write_text(
        "\n".join(
            [
                "import pathlib",
                "import sys",
                "import time",
                f"sys.path.insert(0, {str(repo_root)!r})",
                (
                    "from core.runtime.detached_subprocess_broker "
                    "import DetachedBrokerError, run_brokered_process"
                ),
                f"command = {worker_command!r}",
                "rejections = 0",
                "while True:",
                "    try:",
                (
                    "        result = run_brokered_process("
                    f"command, cwd=pathlib.Path({str(workspace)!r}), "
                    f"stdout_path=pathlib.Path({str(worker_log)!r}), "
                    "timeout_s=10.0)"
                ),
                "    except DetachedBrokerError as exc:",
                "        if 'external authorization required at ' not in str(exc):",
                "            raise",
                "        rejections += 1",
                (
                    f"        pathlib.Path({str(authorization_wait)!r}).write_text("
                    "str(exc), encoding='utf-8')"
                ),
                "        time.sleep(0.05)",
                "        continue",
                "    break",
                (
                    f"pathlib.Path({str(coordinator_outcome)!r}).write_text("
                    "f'{rejections}:{result.returncode}', encoding='utf-8')"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "worker.py", "coordinator.py"],
        cwd=workspace,
        check=True,
    )

    trust_policy_path, trust_root_path, trust_policy, runner_key = (
        _worker_origin_trust_fixture(tmp_path / "trust")
    )
    origin_dir = tmp_path / "origins"
    run_dir = tmp_path / "run"
    broker_policy = [
        {
            "command": worker_command,
            "cwd": str(workspace),
            "stdout_path": str(worker_log),
            "timeout_s_max": 10.0,
            "max_invocations": 1,
            "worker_origin": {
                "schema": detached.WORKER_ORIGIN_POLICY_SCHEMA,
                "campaign_name": "detached-worker-origin-test",
                "protocol_sha256": "1" * 64,
                "trust_policy_path": str(trust_policy_path),
                "trust_root_path": str(trust_root_path),
                "artifact_dir": str(origin_dir),
                "arm": "adapter_rlc",
                "worker_attempt_slot": 1,
                "allowed_cells": [
                    {
                        "cell_id": "cell-0001",
                        "cell_type": "reasoning",
                    }
                ],
                "model_identity_sha256": "8" * 64,
                "adapter_identity_sha256": "9" * 64,
                "authorization_ttl_seconds": 300,
            },
        }
    ]

    _launch(
        run_dir,
        [python, str(coordinator_script)],
        timeout_s=30.0,
        broker_policy=broker_policy,
        cwd=workspace,
    )
    request_path = _wait_for_glob(
        origin_dir,
        "*.request.json",
        timeout_s=15.0,
    )
    _wait_for_glob(tmp_path, authorization_wait.name, timeout_s=5.0)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    signature = runner_key.sign(
        base64.b64decode(request["signed_payload_b64"], validate=True)
    )
    attestation = assemble_role_attestation(
        trust_policy,
        request,
        signature_b64=base64.b64encode(signature).decode("ascii"),
        role=CAMPAIGN_RUNNER,
    )
    attestation_path = request_path.with_name(
        request_path.name.replace(".request.json", ".attestation.json")
    )
    detached._atomic_write(attestation_path, attestation, replace=False)

    receipt = _wait_for(run_dir / detached.RECEIPT_FILE, timeout_s=30.0)
    assert receipt["status"] == "passed"
    assert coordinator_outcome.read_text(encoding="utf-8") == "1:0"
    events = detached._read_attempts(run_dir)
    assert sum(event["event"] == "BROKER_STARTED" for event in events) == 1
    broker_start = next(
        event for event in events if event["event"] == "BROKER_STARTED"
    )
    broker_terminal = next(
        event for event in events if event["event"] == "BROKER_TERMINAL"
    )
    response = broker_terminal["response"]
    assert response["status"] == "passed"
    assert response["worker_origin_lifecycle"]["event_type"] == "terminal"
    assert response["worker_origin_lifecycle"]["result_count"] == 1

    signed_result = json.loads(result_path.read_text(encoding="utf-8"))
    lifecycle_path = Path(
        response["worker_origin_lifecycle"]["artifact_path"]
    )
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    authorization = lifecycle["authorization_payload"]
    verified_result = verify_worker_result_origin(
        trust_policy,
        authorization_attestation=attestation,
        expected_authorization_payload=authorization,
        result=signed_result,
        expected_cell_id="cell-0001",
        expected_cell_type="reasoning",
        expected_attempt_id="attempt-0001",
        expected_sequence=1,
        expected_previous_origin_sha256=ZERO_SHA256,
        authorization_not_before_unix=request["signed_payload"][
            "signed_at_unix"
        ],
        authorization_not_after_unix=request["signed_payload"][
            "signed_at_unix"
        ],
    )
    assert verified_result["session_id"] == broker_start["worker_origin"][
        "session_id"
    ]
    assert detached._status(run_dir)["terminal"] is True

    signature_b64 = lifecycle["event_origin"]["signature_b64"]
    lifecycle["event_origin"]["signature_b64"] = (
        ("A" if signature_b64[0] != "A" else "B") + signature_b64[1:]
    )
    lifecycle_body = {
        key: value
        for key, value in lifecycle.items()
        if key != "artifact_sha256"
    }
    lifecycle["artifact_sha256"] = detached._sha256(lifecycle_body)
    detached._atomic_write(lifecycle_path, lifecycle)
    with pytest.raises(
        detached.DetachedStepError,
        match="lifecycle signature is invalid",
    ):
        detached._status(run_dir)
