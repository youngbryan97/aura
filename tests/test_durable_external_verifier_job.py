"""Bounded lifecycle and fault tests for durable external verifier jobs."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from core.learning.durable_external_verifier_job import (
    DurableExternalVerifierJob,
    DurableExternalVerifierJobError,
)
from core.learning.verified_transition_episode import canonical_json_bytes

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="run_detached_step strong containment requires macOS",
)

_ROOT = Path(__file__).resolve().parents[1]
_DETACHED_RUNNER = _ROOT / "tools" / "run_detached_step.py"
_RESUME_HELPER = _ROOT / "tools" / "resume_durable_external_verifier_job.py"


def _request(*, purpose: str = "test:durable-replay") -> dict[str, Any]:
    body = {
        "schema": "test.durable_external_verifier.request.v1",
        "purpose": purpose,
        "payload": {"sequence": 7},
    }
    return {
        **body,
        "request_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _target_script(tmp_path: Path) -> Path:
    release = tmp_path / "target-release"
    release.mkdir(mode=0o700)
    script = release / "tiny-file-verifier.py"
    script.write_text(
        f"""#!{sys.executable}
import argparse
import json
import os
import pathlib
import time

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="success")
parser.add_argument("--delay", type=float, default=0.0)
parser.add_argument("--request", required=True)
parser.add_argument("--result", required=True)
args = parser.parse_args()
request_path = pathlib.Path(args.request)
result_path = pathlib.Path(args.result)
request_raw = request_path.read_bytes()
request = json.loads(request_raw)
canonical = lambda value: json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode("ascii")
if request_raw != canonical(request):
    raise SystemExit(20)
if result_path.exists():
    try:
        existing_raw = result_path.read_bytes()
        existing = json.loads(existing_raw)
    except Exception:
        raise SystemExit(21)
    if (
        existing_raw != canonical(existing)
        or existing.get("request_sha256") != request["request_sha256"]
    ):
        raise SystemExit(22)
    raise SystemExit(0)
if args.delay:
    time.sleep(args.delay)
if args.mode == "hang":
    time.sleep(30)
if args.mode == "partial":
    fd = os.open(result_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, b"{{")
        os.fsync(fd)
    finally:
        os.close(fd)
    raise SystemExit(0)
response = {{
    "schema": "test.durable_external_verifier.response.v1",
    "request_sha256": request["request_sha256"],
    "verified": True,
}}
raw = canonical(response)
fd = os.open(result_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    os.fchmod(fd, 0o600)
    offset = 0
    while offset < len(raw):
        offset += os.write(fd, raw[offset:])
    os.fsync(fd)
finally:
    os.close(fd)
if args.mode == "fail_after_result":
    raise SystemExit(7)
""",
        encoding="ascii",
    )
    script.chmod(0o700)
    return script


def _job(
    tmp_path: Path,
    target: Path,
    *,
    mode: str = "success",
    delay: float = 0.0,
    timeout_seconds: float = 5.0,
) -> DurableExternalVerifierJob:
    return DurableExternalVerifierJob(
        job_root=tmp_path / "jobs",
        executable=target,
        executable_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        arguments=(
            "--mode",
            mode,
            "--delay",
            str(delay),
        ),
        cwd=target.parent,
        detached_runner=_DETACHED_RUNNER,
        resume_helper=_RESUME_HELPER,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=0.02,
        runner_call_timeout_seconds=15.0,
    )


def _job_dir(tmp_path: Path, request: dict[str, Any]) -> Path:
    return tmp_path / "jobs" / hashlib.sha256(canonical_json_bytes(request)).hexdigest()


def _wait_for(
    predicate: Any,
    *,
    timeout_seconds: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition did not become true")


def _detached_call(*arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(_DETACHED_RUNNER), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    value = json.loads(completed.stdout)
    assert isinstance(value, dict)
    return value


def test_success_is_private_create_once_and_idempotent(
    tmp_path: Path,
) -> None:
    target = _target_script(tmp_path)
    job = _job(tmp_path, target)
    request = _request()

    first = job.run_file_protocol(
        request,
        job.target_command,
        job.timeout_seconds,
        request["purpose"],
    )
    second = _job(tmp_path, target).execute(request)

    assert first == second
    submission = job.begin(request)
    assert stat.S_IMODE(submission.request_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(submission.result_path.stat().st_mode) == 0o600
    assert submission.request_path.stat().st_nlink == 1
    assert submission.result_path.stat().st_nlink == 1
    inspection = _detached_call(
        "status",
        "--run-dir",
        str(submission.run_dir),
    )
    assert inspection["terminal"] is True
    assert inspection["supervisor_attempt"] == 1
    assert inspection["receipt"]["status"] == "passed"
    plan = json.loads((submission.run_dir / "detached_plan.json").read_text(encoding="utf-8"))
    assert plan["resume_contract"] == "target_checkpoint"
    assert plan["power_assertion"]["path"] == "/usr/bin/caffeinate"


def test_caller_timeout_does_not_cancel_and_new_caller_recovers(
    tmp_path: Path,
) -> None:
    target = _target_script(tmp_path)
    request = _request()
    first_caller = _job(tmp_path, target, delay=0.4)
    submission = first_caller.begin(request)

    with pytest.raises(
        DurableExternalVerifierJobError,
        match="caller_wait_timeout",
    ):
        first_caller.wait(submission, caller_timeout_seconds=0.05)

    recovered = _job(tmp_path, target, delay=0.4).execute(request)
    assert recovered["verified"] is True
    inspection = _detached_call(
        "status",
        "--run-dir",
        str(submission.run_dir),
    )
    assert inspection["supervisor_attempt"] == 1
    assert inspection["receipt"]["status"] == "passed"


def test_indeterminate_attempt_resumes_valid_candidate_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target_script(tmp_path)
    request = _request()
    first_caller = _job(tmp_path, target)
    monkeypatch.setenv(
        "AURA_DETACHED_TEST_CRASH_POINT",
        "after_target_exit",
    )
    submission = first_caller.begin(request)
    _wait_for(submission.candidate_result_path.is_file)
    _wait_for(
        lambda: _detached_call(
            "status",
            "--run-dir",
            str(submission.run_dir),
        )["completion_indeterminate"]
    )

    monkeypatch.delenv("AURA_DETACHED_TEST_CRASH_POINT")
    result = _job(tmp_path, target).execute(request)

    assert result["verified"] is True
    inspection = _detached_call(
        "status",
        "--run-dir",
        str(submission.run_dir),
    )
    assert inspection["supervisor_attempt"] == 2
    assert inspection["receipt"]["status"] == "passed"
    assert inspection["receipt"]["command"] == list(submission.target_command)


def test_resume_rejects_malformed_partial_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target_script(tmp_path)
    request = _request()
    first_caller = _job(tmp_path, target, mode="partial")
    monkeypatch.setenv(
        "AURA_DETACHED_TEST_CRASH_POINT",
        "after_target_exit",
    )
    submission = first_caller.begin(request)
    _wait_for(submission.candidate_result_path.is_file)
    _wait_for(
        lambda: _detached_call(
            "status",
            "--run-dir",
            str(submission.run_dir),
        )["completion_indeterminate"]
    )

    monkeypatch.delenv("AURA_DETACHED_TEST_CRASH_POINT")
    with pytest.raises(
        DurableExternalVerifierJobError,
        match="detached_runner_failed",
    ):
        _job(tmp_path, target, mode="partial").execute(request)
    assert not submission.result_path.exists()


@pytest.mark.parametrize(
    ("mode", "timeout_seconds"),
    [
        ("partial", 5.0),
        ("fail_after_result", 5.0),
        ("hang", 0.2),
    ],
)
def test_no_success_on_partial_nonzero_or_timeout(
    tmp_path: Path,
    mode: str,
    timeout_seconds: float,
) -> None:
    target = _target_script(tmp_path)
    job = _job(
        tmp_path,
        target,
        mode=mode,
        timeout_seconds=timeout_seconds,
    )
    request = _request()

    with pytest.raises(DurableExternalVerifierJobError):
        job.execute(request, caller_timeout_seconds=5.0)

    job_dir = _job_dir(tmp_path, request)
    assert not (job_dir / "result").exists()
    inspection = _detached_call(
        "status",
        "--run-dir",
        str(job_dir / "detached"),
    )
    assert inspection["receipt"]["status"] in {
        "passed",
        "failed",
        "timed_out",
    }
    if mode == "partial":
        assert inspection["receipt"]["status"] == "passed"
    elif mode == "fail_after_result":
        assert inspection["receipt"]["status"] == "failed"
    else:
        assert inspection["receipt"]["status"] == "timed_out"


def test_no_success_after_authenticated_cancellation(
    tmp_path: Path,
) -> None:
    target = _target_script(tmp_path)
    job = _job(tmp_path, target, mode="hang", timeout_seconds=10.0)
    request = _request()
    submission = job.begin(request)

    stop = _detached_call("stop", "--run-dir", str(submission.run_dir))
    assert stop["stopped"] is True
    with pytest.raises(
        DurableExternalVerifierJobError,
        match="execution_not_successful",
    ):
        job.wait(submission, caller_timeout_seconds=5.0)
    assert not submission.result_path.exists()
    inspection = _detached_call(
        "status",
        "--run-dir",
        str(submission.run_dir),
    )
    assert inspection["receipt"]["status"] == "stopped"


def test_call_contract_and_long_timeout_bounds(tmp_path: Path) -> None:
    target = _target_script(tmp_path)
    job = _job(tmp_path, target, timeout_seconds=93_600.0)
    request = _request()
    assert job.timeout_seconds == 93_600.0

    with pytest.raises(
        DurableExternalVerifierJobError,
        match="target_command_mismatch",
    ):
        job.run_file_protocol(
            request,
            [*job.target_command, "--different"],
            job.timeout_seconds,
            request["purpose"],
        )
    with pytest.raises(
        DurableExternalVerifierJobError,
        match="timeout_mismatch",
    ):
        job.run_file_protocol(
            request,
            job.target_command,
            93_599.0,
            request["purpose"],
        )
    with pytest.raises(
        DurableExternalVerifierJobError,
        match="purpose_mismatch",
    ):
        job.run_file_protocol(
            request,
            job.target_command,
            job.timeout_seconds,
            "test:different-purpose",
        )
    with pytest.raises(
        DurableExternalVerifierJobError,
        match="timeout_invalid",
    ):
        _job(tmp_path, target, timeout_seconds=93_600.1)


def test_runner_output_volume_is_distinguishable_from_runner_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A noisy runner and a failed runner must not report the same code.

    Both once raised ``durable_verifier_detached_runner_failed`` with no
    detail, so a resume that died on a contract mismatch was indistinguishable
    from one that merely printed too much.
    """
    target = _target_script(tmp_path)
    job = _job(tmp_path, target)
    arguments = ["status", "--run-dir", str(tmp_path / "run")]

    # The runner is spawned through the subprocess gateway (2c2db21b9f), so
    # the stand-in sits on the gateway the module resolves at call time, not
    # on `subprocess.run`, which the real spawn no longer reaches.
    import core.learning.durable_external_verifier_job as module

    class _Gateway:
        def __init__(self, run: Any) -> None:
            self.run = run

    def _stub(returncode: int, stdout: bytes, stderr: bytes) -> Any:
        def _run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(
                args=["runner"],
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )

        return _run

    def _use(run: Any) -> None:
        monkeypatch.setattr(module, "get_subprocess_gateway", lambda: _Gateway(run))

    _use(_stub(2, b"", b"contract mismatch"))
    with pytest.raises(DurableExternalVerifierJobError) as failed:
        job._runner_call(arguments)
    assert failed.value.code == "durable_verifier_detached_runner_failed"
    # The detail is what makes such a failure localisable at all.
    assert "returncode=2" in failed.value.detail
    assert "contract mismatch" in failed.value.detail

    flood = b"x" * ((1 << 20) + 1)
    _use(_stub(0, flood, b""))
    with pytest.raises(DurableExternalVerifierJobError) as flooded:
        job._runner_call(arguments)
    assert flooded.value.code == "durable_verifier_detached_runner_output_too_large"

    _use(_stub(0, b"", flood))
    with pytest.raises(DurableExternalVerifierJobError) as noisy:
        job._runner_call(arguments)
    assert noisy.value.code == "durable_verifier_detached_runner_output_too_large"
