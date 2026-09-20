#!/usr/bin/env python3
"""Bounded external proof for Aura's monotonic shutdown contract.

Each case starts a fresh real runtime, injects SIGINT/SIGTERM at a named
lifecycle boundary, and verifies the process from outside. A pass requires the
terminal root-exit receipt, all canonical shutdown phases exactly once, no
surviving post-latch admission, no descendant process, a free port, and a
reacquirable singleton lock. The driver hard-kills only its own process group
after a bounded failure; it never uses broad process-name cleanup.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.runtime.resource_observation import (  # noqa: E402
    ProcessObservation,
    ResourceObserver,
    get_resource_observer,
)
from tools.live_boot_proof import (  # noqa: E402
    build_safe_boot_env,
    live_proof_rss_abort_mb,
    resolve_launch_python,
)

SHUTDOWN_PHASES = (
    "output_flush",
    "memory_commit",
    "state_vault",
    "actors",
    "model_runtime",
    "event_bus",
    "task_supervisor",
)
PHASE_MARKER = "ShutdownCoordinator: phase started (phase={phase} "
READY_MARKER = "Registry Locked. Aura Ready (Desktop)."
FOREGROUND_MARKER = "Foreground chat reservation acquired"
PROBE_START_MARKER = "Lifecycle probe hold started (target={target} "
#: The three cortex lines, matched on the part that does not name the model.
#:
#: They were pinned with "32B" in them, and the runtime writes
#: `"Primary %s cortex is cold. Starting warmup"` with the resident lane's
#: signed label — which stopped being a 32B when the cortex became
#: Aura-Qwen3.8-27B. `model_warmup_signal` therefore waited its full ten
#: minutes for a line that can no longer be printed and reported every check
#: failed, on 2026-09-20 and on every run since the model changed. The lane
#: label is exactly the part `_primary_lane_label` exists to keep out of a
#: reader's way, so the marker does not read it.
MODEL_WARMUP_START_MARKER = "cortex is cold. Starting warmup"
MODEL_WARMUP_COMPLETE_MARKER = "cortex warmup complete."
MODEL_RECOVERY_START_MARKER = "cortex is dead. Triggering background respawn"
MODEL_OWNER_SCRIPT_MARKERS = (
    "evaluate_unified_intrinsic_decoding.py",
    "mlx_worker.py",
    "resident_recurrent_sft.py",
    "train_recurrent",
    "train_and_fuse.py",
)
REQUIRED_OWNER_CLASSES = (
    "process",
    "thread",
    "task",
    "listener",
    "sentinel",
    "actor",
    "model_worker",
    "lock",
)

FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "[DEGRADATION]",
    "RuntimeWarning:",
    "Task exception was never retrieved",
    "Exception in callback",
    "Root process exit receipt persistence failed",
    "shutdown complete (clean=False",
    "FATAL BOOT ERROR",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    trigger_marker: str
    first_signal: signal.Signals
    repeat_signal: signal.Signals | None = None
    repeat_marker: str | None = None
    repeat_delay_s: float = 0.2
    probe_target: str | None = None
    start_foreground_chat: bool = False
    boot_mode: str = "desktop"
    kill_model_worker_after_trigger: bool = False
    post_kill_marker: str | None = None
    trigger_timeout_s: float = 180.0
    shutdown_timeout_s: float = 140.0


_BOUNDARY_CASE_SPECS: dict[str, CaseSpec] = {
    "launcher_bootstrap": CaseSpec(
        name="launcher_bootstrap",
        trigger_marker="Instance lock acquired: orchestrator",
        first_signal=signal.SIGTERM,
        trigger_timeout_s=45.0,
    ),
    "orchestrator_boot_repeated": CaseSpec(
        name="orchestrator_boot_repeated",
        trigger_marker="Orchestrator boot beginning",
        first_signal=signal.SIGTERM,
        repeat_signal=signal.SIGINT,
    ),
    "ready_repeated": CaseSpec(
        name="ready_repeated",
        trigger_marker=READY_MARKER,
        first_signal=signal.SIGINT,
        repeat_signal=signal.SIGTERM,
    ),
    "model_warmup_signal": CaseSpec(
        name="model_warmup_signal",
        trigger_marker=MODEL_WARMUP_START_MARKER,
        first_signal=signal.SIGTERM,
        boot_mode="headless",
        trigger_timeout_s=600.0,
        shutdown_timeout_s=180.0,
    ),
    "model_recovery_signal": CaseSpec(
        name="model_recovery_signal",
        trigger_marker=MODEL_WARMUP_COMPLETE_MARKER,
        first_signal=signal.SIGINT,
        boot_mode="headless",
        kill_model_worker_after_trigger=True,
        post_kill_marker=MODEL_RECOVERY_START_MARKER,
        trigger_timeout_s=600.0,
        shutdown_timeout_s=180.0,
    ),
    "container_repeated": CaseSpec(
        name="container_repeated",
        trigger_marker=READY_MARKER,
        first_signal=signal.SIGTERM,
        repeat_signal=signal.SIGINT,
        repeat_marker=PROBE_START_MARKER.format(target="container"),
        probe_target="container",
    ),
    "root_finalization_repeated": CaseSpec(
        name="root_finalization_repeated",
        trigger_marker=READY_MARKER,
        first_signal=signal.SIGINT,
        repeat_signal=signal.SIGTERM,
        repeat_marker=PROBE_START_MARKER.format(target="root_finalization"),
        probe_target="root_finalization",
    ),
    "active_foreground_repeated": CaseSpec(
        name="active_foreground_repeated",
        trigger_marker=READY_MARKER,
        first_signal=signal.SIGTERM,
        repeat_signal=signal.SIGINT,
        start_foreground_chat=True,
        repeat_delay_s=0.25,
        trigger_timeout_s=300.0,
    ),
}


def _coordinator_phase_case(phase: str, index: int) -> CaseSpec:
    """Inject a second signal while one exact shutdown owner group is wedged."""

    name = f"{phase}_repeated"
    target = f"coordinator:{phase}"
    first_signal = signal.SIGTERM if index % 2 == 0 else signal.SIGINT
    repeat_signal = signal.SIGINT if first_signal is signal.SIGTERM else signal.SIGTERM
    return CaseSpec(
        name=name,
        trigger_marker=READY_MARKER,
        first_signal=first_signal,
        repeat_signal=repeat_signal,
        repeat_marker=PROBE_START_MARKER.format(target=target),
        probe_target=target,
    )


COORDINATOR_PHASE_CASES = tuple(f"{phase}_repeated" for phase in SHUTDOWN_PHASES)
CASE_SPECS: dict[str, CaseSpec] = {
    "launcher_bootstrap": _BOUNDARY_CASE_SPECS["launcher_bootstrap"],
    "orchestrator_boot_repeated": _BOUNDARY_CASE_SPECS["orchestrator_boot_repeated"],
    "ready_repeated": _BOUNDARY_CASE_SPECS["ready_repeated"],
    "model_warmup_signal": _BOUNDARY_CASE_SPECS["model_warmup_signal"],
    "model_recovery_signal": _BOUNDARY_CASE_SPECS["model_recovery_signal"],
    **{
        f"{phase}_repeated": _coordinator_phase_case(phase, index)
        for index, phase in enumerate(SHUTDOWN_PHASES)
    },
    "container_repeated": _BOUNDARY_CASE_SPECS["container_repeated"],
    "root_finalization_repeated": _BOUNDARY_CASE_SPECS["root_finalization_repeated"],
    "active_foreground_repeated": _BOUNDARY_CASE_SPECS["active_foreground_repeated"],
}

DEFAULT_CASES = tuple(CASE_SPECS)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    create_time: float
    cmdline: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "create_time": self.create_time,
            "cmdline": list(self.cmdline),
        }


class LogCursor:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.text = ""

    def refresh(self) -> str:
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self.offset)
                chunk = handle.read()
                self.offset = handle.tell()
        except OSError:
            return ""
        if chunk:
            self.text += chunk
        return chunk


def _is_aura_main_process(cmdline: list[str] | tuple[str, ...]) -> bool:
    if not cmdline:
        return False
    executable = Path(str(cmdline[0])).name.lower()
    if "python" not in executable:
        return False
    return any(Path(str(arg)).name == "aura_main.py" for arg in cmdline[1:])


def _running_aura_main_pids(
    *,
    observer: ResourceObserver | None = None,
) -> list[int]:
    table = (observer or get_resource_observer()).process_table()
    if not table.available:
        raise RuntimeError(f"process table observation unavailable: {table.error}")
    return sorted(
        process.pid
        for process in table.processes
        if _is_aura_main_process(process.cmdline)
    )


def _is_competing_model_owner(cmdline: list[str] | tuple[str, ...]) -> bool:
    command = " ".join(str(item) for item in cmdline).lower()
    return any(marker.lower() in command for marker in MODEL_OWNER_SCRIPT_MARKERS)


def _running_competing_model_pids(
    *,
    observer: ResourceObserver | None = None,
) -> list[int]:
    table = (observer or get_resource_observer()).process_table()
    if not table.available:
        raise RuntimeError(f"process table observation unavailable: {table.error}")
    return sorted(
        process.pid
        for process in table.processes
        if _is_competing_model_owner(process.cmdline)
    )


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", int(port))) != 0


def _orchestrator_lock_available() -> tuple[bool, str]:
    lock_path = Path.home() / ".aura" / "locks" / "orchestrator.lock"
    if not lock_path.exists():
        return True, "lock file absent"
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except OSError as exc:
        return False, f"open failed: {type(exc).__name__}: {exc}"
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False, "lock is still held"
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True, "lock reacquired"
    finally:
        os.close(fd)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _nested_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def evaluate_terminal_report(
    report: dict[str, Any],
    *,
    root_pid: int,
    expected_first_reason: str,
    minimum_signal_requests: int,
) -> dict[str, bool]:
    components = _nested_dict(report, "components")
    coordinator = _nested_dict(components, "coordinator")
    container = _nested_dict(components, "container")
    hygiene = _nested_dict(components, "runtime_hygiene")
    final_tasks = _nested_dict(components, "final_tasks")
    request = _nested_dict(report, "request")
    admission = _nested_dict(report, "admission")
    counts = _nested_dict(admission, "counts")
    root_exit = _nested_dict(report, "root_exit")
    root_resources = _nested_dict(root_exit, "resources")
    verdict = _nested_dict(report, "verdict")

    return {
        "schema": report.get("schema") == "aura.shutdown_verdict.v1",
        "root_pid": report.get("pid") == root_pid,
        "terminal_stage": report.get("stage") == "root_process_exit",
        "final": report.get("final") is True,
        "terminal_receipt_once": report.get("terminal_receipt_sequence") == 1,
        "verdict_clean": verdict.get("clean") is True and not verdict.get("blockers"),
        "first_reason": request.get("first_reason") == expected_first_reason,
        "signal_request_count": int(request.get("request_count", 0) or 0)
        >= minimum_signal_requests,
        "all_phases_once": coordinator.get("completed_phases") == list(SHUTDOWN_PHASES),
        "coordinator_clean": coordinator.get("clean") is True,
        "container_clean": container.get("clean") is True,
        "runtime_hygiene_clean": hygiene.get("clean") is True,
        "final_tasks_empty": final_tasks.get("count") == 0,
        "no_shutdown_resurrection": int(counts.get("survived", 0) or 0) == 0,
        "root_lock_released": root_exit.get("lock_released") is True,
        "root_finalizers_complete": root_exit.get("multiprocessing_finalizers_completed")
        is True,
        "root_logging_flushed": root_exit.get("logging_shutdown_completed") is True,
        "root_resources_clean": root_resources.get("clean") is True,
        "root_exit_zero": root_exit.get("exit_code") == 0,
    }


def evaluate_owner_class_witnesses(
    report: dict[str, Any],
    case_verdict: dict[str, Any],
) -> dict[str, bool]:
    """Project one case onto the owner classes required by CTX2-SHUTDOWN-002."""

    components = _nested_dict(report, "components")
    hygiene = _nested_dict(components, "runtime_hygiene")
    before = _nested_dict(hygiene, "before")
    after = _nested_dict(hygiene, "after")
    before_processes = _nested_dict(before, "processes")
    before_threads = _nested_dict(before, "threads")
    before_tasks = _nested_dict(before, "tasks")
    after_processes = _nested_dict(after, "processes")
    after_threads = _nested_dict(after, "threads")
    after_tasks = _nested_dict(after, "tasks")
    after_native = _nested_dict(after, "native_resources")
    coordinator = _nested_dict(components, "coordinator")
    handler_statuses = _nested_dict(coordinator, "handler_statuses")
    container = _nested_dict(components, "container")
    pre_signal = _nested_dict(case_verdict, "pre_signal_evidence")
    checks = _nested_dict(case_verdict, "checks")

    process_observed = any(
        int(before_processes.get(key, 0) or 0) > 0
        for key in (
            "active_registered",
            "active_subprocesses",
            "active_multiprocessing",
            "owned_descendant_processes",
        )
    ) or bool(case_verdict.get("recovery_injection"))
    process_clean = all(
        int(after_processes.get(key, 0) or 0) == 0
        for key in (
            "active_registered",
            "active_subprocesses",
            "active_multiprocessing",
            "owned_descendant_processes",
            "rogue_child_processes",
        )
    )
    thread_observed = int(before_threads.get("active", 0) or 0) > 0
    thread_clean = (
        int(after_threads.get("active", 0) or 0) == 0
        and int(after_threads.get("active_non_daemon", 0) or 0) == 0
        and int(after_threads.get("stale_non_daemon", 0) or 0) == 0
    )
    task_observed = any(
        int(before_tasks.get(key, 0) or 0) > 0
        for key in ("total_observed", "total_tracked", "active")
    )
    task_clean = (
        int(after_tasks.get("active", 0) or 0) == 0
        and int(after_tasks.get("shutdown_critical_active", 0) or 0) == 0
    )
    sentinel_observed = any("sentinel" in str(name) for name in handler_statuses)
    sentinel_clean = all(
        str(status) == "completed"
        for name, status in handler_statuses.items()
        if "sentinel" in str(name)
    )
    completed_phases = list(coordinator.get("completed_phases") or ())
    completed_services = list(container.get("completed_services") or ())

    return {
        "process": process_observed and process_clean,
        "thread": thread_observed and thread_clean,
        "task": task_observed and task_clean,
        "listener": (
            pre_signal.get("port_listening") is True
            and int(after_native.get("listening_socket_count", 0) or 0) == 0
            and checks.get("port_free") is True
        ),
        "sentinel": sentinel_observed and sentinel_clean,
        "actor": (
            bool(completed_services)
            and "actors" in completed_phases
            and container.get("clean") is True
        ),
        "model_worker": (
            pre_signal.get("model_worker_observed") is True
            and process_clean
        ),
        "lock": (
            pre_signal.get("singleton_lock_held") is True
            and checks.get("singleton_lock_available") is True
        ),
    }


def aggregate_owner_class_witnesses(
    case_verdicts: list[dict[str, Any]],
) -> dict[str, bool]:
    aggregate = {name: False for name in REQUIRED_OWNER_CLASSES}
    for verdict in case_verdicts:
        report = verdict.get("shutdown_report")
        if not isinstance(report, dict):
            continue
        observed = evaluate_owner_class_witnesses(report, verdict)
        for name in aggregate:
            aggregate[name] = aggregate[name] or observed.get(name, False)
    return aggregate


class SignalMatrixCase:
    def __init__(
        self,
        *,
        spec: CaseSpec,
        port: int,
        artifact_dir: Path,
        rss_abort_mb: float | None,
        observer: ResourceObserver | None = None,
    ) -> None:
        self.spec = spec
        self.port = int(port)
        self.artifact_dir = artifact_dir.resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_path = self.artifact_dir / "runtime_stdout.log"
        self.transcript_path = self.artifact_dir / "events.jsonl"
        self.verdict_path = self.artifact_dir / "case_verdict.json"
        self.report_path = self.artifact_dir / "shutdown_report.json"
        self.cursor = LogCursor(self.stdout_path)
        self.proc: subprocess.Popen[bytes] | None = None
        self.stdout_handle: Any = None
        self.seen_identities: dict[tuple[int, float], ProcessIdentity] = {}
        self.peak_rss_mb = 0.0
        self.rss_abort_mb_override = rss_abort_mb
        self.rss_abort_mb = 0.0
        self.resource_observer = observer or get_resource_observer()
        self.started_monotonic = time.monotonic()
        self.chat_thread: threading.Thread | None = None
        self.chat_result: dict[str, Any] = {}
        self.signal_events: list[dict[str, Any]] = []
        self.recovery_injection: dict[str, Any] = {}
        self.pre_signal_evidence: dict[str, Any] = {}

    def record(self, event: str, **detail: Any) -> None:
        payload = {
            "at_unix": time.time(),
            "elapsed_seconds": round(time.monotonic() - self.started_monotonic, 3),
            "event": event,
            "resource_observation": self.resource_observer.provenance.to_dict(),
            **detail,
        }
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        print(f"[{self.spec.name}] {event}: {detail.get('summary', '')}", flush=True)

    def _capture_identity(self, process: ProcessObservation) -> None:
        identity = ProcessIdentity(
            pid=process.pid,
            create_time=process.create_time,
            cmdline=process.cmdline,
        )
        self.seen_identities[(identity.pid, identity.create_time)] = identity

    def sample_process_tree(self) -> float:
        if self.proc is None:
            return 0.0
        table = self.resource_observer.process_table()
        if not table.available:
            raise RuntimeError(f"process table observation unavailable: {table.error}")
        processes = [
            process
            for process in table.processes
            if process.pid == self.proc.pid or self.proc.pid in process.ancestor_pids
        ]
        total = 0
        for process in processes:
            self._capture_identity(process)
            total += process.rss_bytes
        rss_mb = total / (1024 * 1024)
        self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
        if self.rss_abort_mb > 0 and rss_mb > self.rss_abort_mb:
            raise RuntimeError(
                f"process tree RSS {rss_mb:.0f}MB exceeded {self.rss_abort_mb:.0f}MB ceiling"
            )
        return rss_mb

    def wait_for_marker(self, marker: str, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.1, timeout_s)
        while time.monotonic() < deadline:
            self.cursor.refresh()
            if marker in self.cursor.text:
                self.record("marker_observed", summary=marker, marker=marker)
                return True
            if self.proc is not None and self.proc.poll() is not None:
                self.cursor.refresh()
                return marker in self.cursor.text
            self.sample_process_tree()
            time.sleep(0.05)
        self.cursor.refresh()
        return marker in self.cursor.text

    def send_signal(self, sig: signal.Signals, *, role: str) -> None:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(f"cannot send {sig.name}: root process already exited")
        os.kill(self.proc.pid, sig)
        event = {
            "role": role,
            "signal": sig.name,
            "at_unix": time.time(),
            "log_offset": self.cursor.offset,
        }
        self.signal_events.append(event)
        self.record("signal_sent", summary=f"{role} {sig.name}", **event)

    def start_foreground_chat(self) -> None:
        def _request() -> None:
            started = time.monotonic()
            try:
                with httpx.Client(
                    timeout=180.0,
                    headers={
                        "X-Aura-Surface": "desktop-ui",
                        "X-Aura-Require-CognitiveEngine": "true",
                    },
                ) as client:
                    response = client.post(
                        f"http://127.0.0.1:{self.port}/api/chat",
                        json={
                            "message": (
                                "Reason carefully about a three-stage reliability migration and "
                                "state the strongest failure invariant for each stage."
                            ),
                            "session_id": f"shutdown-matrix-{self.spec.name}",
                        },
                    )
                self.chat_result.update(
                    status_code=response.status_code,
                    body=response.text[:1000],
                )
            except (httpx.HTTPError, OSError) as exc:
                self.chat_result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                self.chat_result["duration_seconds"] = round(time.monotonic() - started, 3)

        self.chat_thread = threading.Thread(
            target=_request,
            name=f"shutdown-matrix-chat-{self.spec.name}",
            daemon=True,
        )
        self.chat_thread.start()
        self.record("foreground_chat_started", summary="POST /api/chat dispatched")

    def kill_owned_model_worker(self, timeout_s: float = 30.0) -> None:
        if self.proc is None:
            raise RuntimeError("cannot inject recovery without a root process")
        deadline = time.monotonic() + max(0.1, timeout_s)
        while time.monotonic() < deadline:
            table = self.resource_observer.process_table()
            if not table.available:
                raise RuntimeError(
                    f"process table observation unavailable: {table.error}"
                )
            candidates = [
                process
                for process in table.processes
                if self.proc.pid in process.ancestor_pids
                and any(
                    Path(str(arg)).name == "mlx_worker.py"
                    for arg in process.cmdline[1:]
                )
            ]
            if len(candidates) > 1:
                raise RuntimeError(
                    "recovery injection found multiple owned MLX workers: "
                    f"{sorted(item.pid for item in candidates)}"
                )
            if candidates:
                candidate = candidates[0]
                current = self.resource_observer.process(candidate.pid)
                if (
                    current is None
                    or abs(current.create_time - candidate.create_time) > 0.001
                    or self.proc.pid not in current.ancestor_pids
                ):
                    time.sleep(0.05)
                    continue
                self._capture_identity(current)
                os.kill(current.pid, signal.SIGKILL)
                self.recovery_injection = {
                    "worker_pid": current.pid,
                    "worker_create_time": current.create_time,
                    "signal": "SIGKILL",
                    "at_unix": time.time(),
                }
                self.record(
                    "owned_model_worker_killed",
                    summary=f"pid={current.pid} identity-bound recovery injection",
                    **self.recovery_injection,
                )
                return
            self.sample_process_tree()
            time.sleep(0.1)
        raise TimeoutError("owned MLX worker was not observed for recovery injection")

    def spawn(self) -> None:
        env = build_safe_boot_env(os.environ, mode=self.spec.boot_mode)
        env["PYTHONUNBUFFERED"] = "1"
        env["AURA_SHUTDOWN_REPORT_PATH"] = str(self.report_path)
        env["AURA_ARTIFACTS_DIR"] = str(self.artifact_dir / "runtime_artifacts")
        env["AURA_LIVENESS_HEARTBEAT_FILE"] = str(
            self.artifact_dir / "liveness_heartbeat.json"
        )
        env["AURA_REAPER_MANIFEST"] = str(self.artifact_dir / "reaper_manifest.json")
        if self.spec.probe_target:
            env["AURA_SHUTDOWN_PROBE_ENABLED"] = "1"
            env["AURA_SHUTDOWN_PROBE_TARGET"] = self.spec.probe_target
            env["AURA_SHUTDOWN_PROBE_HOLD_SECONDS"] = "0.75"
        derived_rss = live_proof_rss_abort_mb(env)
        self.rss_abort_mb = (
            min(float(self.rss_abort_mb_override), derived_rss)
            if self.rss_abort_mb_override is not None
            else derived_rss
        )
        self.stdout_handle = self.stdout_path.open("wb")
        command = [
            resolve_launch_python(),
            "aura_main.py",
            "--desktop" if self.spec.boot_mode == "desktop" else "--headless",
            "--port",
            str(self.port),
        ]
        self.proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=self.stdout_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self.record(
            "spawned",
            summary=f"pid={self.proc.pid} port={self.port}",
            pid=self.proc.pid,
            port=self.port,
            command=command,
            rss_abort_mb=self.rss_abort_mb,
        )
        self.sample_process_tree()

    def wait_for_exit(self) -> bool:
        if self.proc is None:
            return False
        deadline = time.monotonic() + self.spec.shutdown_timeout_s
        while time.monotonic() < deadline:
            self.cursor.refresh()
            self.sample_process_tree()
            if self.proc.poll() is not None:
                self.cursor.refresh()
                return True
            time.sleep(0.1)
        return self.proc.poll() is not None

    def hard_kill_owned_group(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                self.proc.kill()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass

    def residual_processes(self) -> list[ProcessIdentity]:
        residuals: list[ProcessIdentity] = []
        for identity in self.seen_identities.values():
            process = self.resource_observer.process(identity.pid)
            if process is None:
                continue
            if abs(process.create_time - identity.create_time) > 0.001:
                continue
            if process.status.lower() not in {"dead", "zombie"}:
                residuals.append(identity)
        return sorted(residuals, key=lambda item: item.pid)

    def _failure_lines(self) -> list[str]:
        lines: list[str] = []
        for line in self.cursor.text.splitlines():
            if any(marker.lower() in line.lower() for marker in FAILURE_MARKERS):
                lines.append(line[:1000])
        return lines[:50]

    def capture_pre_signal_evidence(self) -> None:
        self.sample_process_tree()
        lock_available, lock_detail = _orchestrator_lock_available()
        model_workers = sorted(
            identity.pid
            for identity in self.seen_identities.values()
            if any(
                Path(str(arg)).name == "mlx_worker.py"
                for arg in identity.cmdline[1:]
            )
        )
        self.pre_signal_evidence = {
            "port_listening": not _port_is_free(self.port),
            "singleton_lock_held": not lock_available,
            "singleton_lock_detail": lock_detail,
            "owned_process_identity_count": len(self.seen_identities),
            "model_worker_observed": bool(model_workers),
            "model_worker_pids": model_workers,
        }
        self.record(
            "pre_signal_evidence",
            summary=(
                f"port_listening={self.pre_signal_evidence['port_listening']} "
                f"lock_held={self.pre_signal_evidence['singleton_lock_held']} "
                f"model_worker={self.pre_signal_evidence['model_worker_observed']}"
            ),
            **self.pre_signal_evidence,
        )

    def _probe_window(self) -> dict[str, Any]:
        target = self.spec.probe_target
        if not target:
            return {}
        escaped_target = re.escape(target)
        starts = re.findall(
            rf"Lifecycle probe hold started \(target={escaped_target} "
            rf"started_at_unix=([0-9.]+) hold_seconds=([0-9.]+)\)",
            self.cursor.text,
        )
        completions = re.findall(
            rf"Lifecycle probe hold completed \(target={escaped_target} "
            rf"completed_at_unix=([0-9.]+)\)",
            self.cursor.text,
        )
        repeat_events = [
            event for event in self.signal_events if event.get("role") == "repeat"
        ]
        started_at = float(starts[0][0]) if len(starts) == 1 else 0.0
        completed_at = float(completions[0]) if len(completions) == 1 else 0.0
        repeat_at = (
            float(repeat_events[0].get("at_unix", 0.0) or 0.0)
            if len(repeat_events) == 1
            else 0.0
        )
        return {
            "target": target,
            "started_count": len(starts),
            "completed_count": len(completions),
            "started_at_unix": started_at,
            "completed_at_unix": completed_at,
            "repeat_at_unix": repeat_at,
            "repeat_inside_hold": bool(
                started_at > 0.0
                and completed_at >= started_at
                and started_at <= repeat_at <= completed_at
            ),
        }

    def finalize_verdict(self, *, trigger_seen: bool, repeat_seen: bool, exited: bool) -> bool:
        if self.stdout_handle is not None:
            self.stdout_handle.flush()
            self.stdout_handle.close()
            self.stdout_handle = None
        self.cursor.refresh()
        if self.chat_thread is not None:
            self.chat_thread.join(timeout=5.0)
        time.sleep(1.5)

        report = _read_json(self.report_path)
        root_pid = self.proc.pid if self.proc is not None else -1
        expected_first_reason = f"desktop_signal:{self.spec.first_signal.name}"
        report_checks = evaluate_terminal_report(
            report,
            root_pid=root_pid,
            expected_first_reason=expected_first_reason,
            minimum_signal_requests=len(self.signal_events),
        )
        residuals = self.residual_processes()
        lock_available, lock_detail = _orchestrator_lock_available()
        phase_counts = {
            phase: self.cursor.text.count(PHASE_MARKER.format(phase=phase))
            for phase in SHUTDOWN_PHASES
        }
        history = list((self.report_path.parent / "shutdown_history").glob("*.json"))
        failure_lines = self._failure_lines()
        probe_window = self._probe_window()
        external_checks = {
            "trigger_seen": trigger_seen,
            "repeat_trigger_seen": repeat_seen,
            "root_exited": exited,
            "root_exit_code_zero": self.proc is not None and self.proc.returncode == 0,
            "phase_markers_once": all(count == 1 for count in phase_counts.values()),
            "history_receipt_once": len(history) == 1,
            "no_residual_processes": not residuals,
            "port_free": _port_is_free(self.port),
            "singleton_lock_available": lock_available,
            "runtime_stream_clean": not failure_lines,
        }
        if self.spec.start_foreground_chat:
            try:
                chat_body = json.loads(str(self.chat_result.get("body") or ""))
            except (json.JSONDecodeError, TypeError, ValueError):
                chat_body = {}
            external_checks["foreground_shutdown_outcome_explicit"] = bool(
                self.chat_result.get("status_code") in {200, 503}
                and isinstance(chat_body, dict)
                and chat_body.get("status") == "runtime_shutdown"
            )
        if self.spec.kill_model_worker_after_trigger:
            external_checks.update(
                model_worker_recovery_injected=bool(self.recovery_injection),
                recovery_boundary_observed=bool(
                    self.spec.post_kill_marker
                    and self.spec.post_kill_marker in self.cursor.text
                ),
            )
        if self.spec.probe_target:
            external_checks.update(
                probe_hold_started_once=probe_window.get("started_count") == 1,
                probe_hold_completed_once=probe_window.get("completed_count") == 1,
                repeat_inside_probe_hold=probe_window.get("repeat_inside_hold") is True,
            )
        checks = {**report_checks, **external_checks}
        passed = all(checks.values())
        verdict = {
            "schema": "aura.shutdown_signal_case.v1",
            "case": self.spec.name,
            "passed": passed,
            "root_pid": root_pid,
            "root_returncode": self.proc.returncode if self.proc is not None else None,
            "port": self.port,
            "checks": checks,
            "signal_events": self.signal_events,
            "probe_window": probe_window,
            "phase_marker_counts": phase_counts,
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "rss_abort_mb": round(self.rss_abort_mb, 1),
            "residual_processes": [item.as_dict() for item in residuals],
            "lock_detail": lock_detail,
            "failure_lines": failure_lines,
            "chat_result": self.chat_result,
            "recovery_injection": self.recovery_injection,
            "pre_signal_evidence": self.pre_signal_evidence,
            "shutdown_report_path": str(self.report_path),
            "stdout_path": str(self.stdout_path),
            "history_paths": [str(path) for path in history],
            "resource_observation": self.resource_observer.provenance.to_dict(),
        }
        self.verdict_path.write_text(
            json.dumps(verdict, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.record(
            "verdict",
            summary=f"passed={passed} failed={[key for key, ok in checks.items() if not ok]}",
            passed=passed,
            failed_checks=[key for key, ok in checks.items() if not ok],
        )
        return passed

    def run(self) -> bool:
        trigger_seen = False
        repeat_seen = self.spec.repeat_signal is None
        exited = False
        try:
            self.spawn()
            trigger_seen = self.wait_for_marker(
                self.spec.trigger_marker,
                self.spec.trigger_timeout_s,
            )
            if not trigger_seen:
                raise TimeoutError(f"trigger marker not observed: {self.spec.trigger_marker}")

            if self.spec.start_foreground_chat:
                self.start_foreground_chat()
                trigger_seen = self.wait_for_marker(FOREGROUND_MARKER, 180.0)
                if not trigger_seen:
                    raise TimeoutError(f"foreground marker not observed: {FOREGROUND_MARKER}")

            if self.spec.kill_model_worker_after_trigger:
                self.kill_owned_model_worker()
                if not self.spec.post_kill_marker:
                    raise RuntimeError("recovery injection has no post-kill marker")
                trigger_seen = self.wait_for_marker(
                    self.spec.post_kill_marker,
                    self.spec.trigger_timeout_s,
                )
                if not trigger_seen:
                    raise TimeoutError(
                        f"recovery marker not observed: {self.spec.post_kill_marker}"
                    )

            self.capture_pre_signal_evidence()
            self.send_signal(self.spec.first_signal, role="first")
            if self.spec.repeat_signal is not None:
                if self.spec.repeat_marker:
                    repeat_seen = self.wait_for_marker(
                        self.spec.repeat_marker,
                        self.spec.shutdown_timeout_s,
                    )
                    if not repeat_seen:
                        raise TimeoutError(
                            f"repeat marker not observed: {self.spec.repeat_marker}"
                        )
                else:
                    time.sleep(max(0.0, self.spec.repeat_delay_s))
                    repeat_seen = True
                self.send_signal(self.spec.repeat_signal, role="repeat")

            exited = self.wait_for_exit()
            if not exited:
                raise TimeoutError(
                    f"root process did not exit within {self.spec.shutdown_timeout_s:.1f}s"
                )
        except (OSError, RuntimeError, TimeoutError) as exc:
            self.record("case_error", summary=f"{type(exc).__name__}: {exc}")
            self.hard_kill_owned_group()
        finally:
            if self.proc is not None and self.proc.poll() is None:
                self.hard_kill_owned_group()
            if self.proc is not None:
                self.proc.poll()
        return self.finalize_verdict(
            trigger_seen=trigger_seen,
            repeat_seen=repeat_seen,
            exited=exited,
        )


def _parse_cases(values: list[str] | None) -> list[str]:
    if not values or values == ["all"]:
        return list(DEFAULT_CASES)
    selected: list[str] = []
    for value in values:
        for name in str(value).split(","):
            normalized = name.strip()
            if not normalized:
                continue
            if normalized == "all":
                return list(DEFAULT_CASES)
            if normalized not in CASE_SPECS:
                raise ValueError(f"unknown case '{normalized}'")
            if normalized not in selected:
                selected.append(normalized)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Case name, comma-separated names, or 'all' (default: all)",
    )
    parser.add_argument("--base-port", type=int, default=8870)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--rss-abort-mb", type=float)
    parser.add_argument("--list", action="store_true", help="List case names and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for name in DEFAULT_CASES:
            print(name)
        return 0
    try:
        selected = _parse_cases(args.cases)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    observer = get_resource_observer()
    try:
        existing = _running_aura_main_pids(observer=observer)
    except RuntimeError as exc:
        print(f"Refusing shutdown matrix: {exc}", file=sys.stderr)
        return 2
    if existing:
        print(
            f"Refusing shutdown matrix while another aura_main.py is running: {existing}",
            file=sys.stderr,
        )
        return 2
    try:
        model_owners = _running_competing_model_pids(observer=observer)
    except RuntimeError as exc:
        print(f"Cannot verify competing model owners: {exc}", file=sys.stderr)
        return 2
    if model_owners:
        print(
            "Refusing to run beside an existing resident model owner "
            f"(pids={model_owners}).",
            file=sys.stderr,
        )
        return 2
    occupied = [args.base_port + index for index in range(len(selected)) if not _port_is_free(args.base_port + index)]
    if occupied:
        print(f"Refusing shutdown matrix because ports are occupied: {occupied}", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    artifact_root = (
        args.artifact_dir
        or ROOT / "artifacts" / "current" / f"shutdown_signal_matrix_{stamp}"
    ).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results: list[dict[str, object]] = []
    case_verdicts: list[dict[str, Any]] = []
    for index, name in enumerate(selected):
        case = SignalMatrixCase(
            spec=CASE_SPECS[name],
            port=args.base_port + index,
            artifact_dir=artifact_root / name,
            rss_abort_mb=args.rss_abort_mb,
            observer=observer,
        )
        passed = case.run()
        results.append(
            {
                "case": name,
                "passed": passed,
                "verdict_path": str(case.verdict_path),
            }
        )
        verdict_payload = _read_json(case.verdict_path)
        if verdict_payload:
            # Keep case receipts bounded.  The canonical shutdown report remains
            # a separate artifact and is joined only for in-memory aggregation.
            report_payload = _read_json(case.report_path)
            if report_payload:
                verdict_payload = {
                    **verdict_payload,
                    "shutdown_report": report_payload,
                }
            case_verdicts.append(verdict_payload)
        try:
            contamination = _running_aura_main_pids(observer=observer)
        except RuntimeError as exc:
            print(f"Process-table verification failed after {name}: {exc}")
            break
        if contamination:
            print("A case left an aura_main.py process; refusing to contaminate later cases.")
            break

    phase_coverage_complete = all(
        case_name in selected for case_name in COORDINATOR_PHASE_CASES
    )
    owner_class_witnesses = aggregate_owner_class_witnesses(case_verdicts)
    full_matrix_requested = selected == list(DEFAULT_CASES)
    owner_class_coverage_complete = all(owner_class_witnesses.values())
    matrix_passed = (
        len(results) == len(selected)
        and all(bool(item["passed"]) for item in results)
        and (
            not full_matrix_requested
            or (phase_coverage_complete and owner_class_coverage_complete)
        )
    )
    summary = {
        "schema": "aura.shutdown_signal_matrix.v1",
        "passed": matrix_passed,
        "selected_cases": selected,
        "coordinator_phase_cases": list(COORDINATOR_PHASE_CASES),
        "coordinator_phase_coverage_complete": phase_coverage_complete,
        "owner_class_witnesses": owner_class_witnesses,
        "owner_class_coverage_complete": owner_class_coverage_complete,
        "full_matrix_requested": full_matrix_requested,
        "completed_cases": results,
        "duration_seconds": round(time.monotonic() - started, 3),
        "artifact_root": str(artifact_root),
        "resource_observation": observer.provenance.to_dict(),
    }
    summary_path = artifact_root / "MATRIX_VERDICT.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"Shutdown signal matrix passed={matrix_passed} "
        f"duration={summary['duration_seconds']}s artifact={summary_path}",
        flush=True,
    )
    return 0 if matrix_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
