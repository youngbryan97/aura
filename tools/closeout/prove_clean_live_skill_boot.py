#!/usr/bin/env python3
"""Externally prove a clean Aura boot publishes one ready capability catalog.

Unlike the in-process catalog audit, this proof starts ``aura_main.py`` in a
fresh process group, observes the real HTTP surfaces, and verifies a clean
bounded shutdown from outside the runtime. It refuses to contend with a live
Aura or another resident model owner and hard-kills only the process group it
created after a bounded failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from core.skills.discovery import PARITY_STATES_THAT_AGREE

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
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
from tools.shutdown_signal_matrix import (  # noqa: E402
    FAILURE_MARKERS,
    _is_aura_main_process,
    _is_competing_model_owner,
    evaluate_terminal_report,
)

READY_MARKER = "Registry Locked. Aura Ready"


def _sha256_json(value: Any) -> str:
    canonical = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _health_identity(health: dict[str, Any]) -> dict[str, Any]:
    preflight = dict(health.get("execution_preflight") or {})
    return {
        "backend": health.get("backend"),
        "digest": health.get("digest"),
        "expected_live_count": health.get("expected_live_count"),
        "live_count": health.get("live_count"),
        "missing_live": health.get("missing_live"),
        "parity_status": health.get("parity_status"),
        "preflight_complete": preflight.get("complete"),
        "preflight_failed": preflight.get("failed"),
        "preflight_ok": preflight.get("ok"),
        "quarantined": health.get("quarantined"),
        "quarantined_count": health.get("quarantined_count"),
        "ready": health.get("ready"),
        "reason": health.get("reason"),
    }


def evaluate_skill_surfaces(
    tools_payload: dict[str, Any],
    skills_payload: dict[str, Any],
    bootstrap_payload: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate only externally observable live catalog facts."""

    tools = list(tools_payload.get("tools") or ())
    skills = list(skills_payload.get("catalog") or ())
    bootstrap_tools = list(bootstrap_payload.get("tools") or ())
    tool_names = sorted(str(item.get("name") or "") for item in tools)
    skill_names = sorted(str(item.get("name") or "") for item in skills)
    bootstrap_names = sorted(
        str(item.get("name") or "") for item in bootstrap_tools
    )
    health = {
        "tools": _health_identity(dict(tools_payload.get("health") or {})),
        "skills": _health_identity(dict(skills_payload.get("health") or {})),
        "bootstrap": _health_identity(
            dict(bootstrap_payload.get("skill_catalog") or {})
        ),
    }
    canonical_health = health["tools"]
    ui_flags = list((bootstrap_payload.get("ui") or {}).get("status_flags") or ())
    blocked_flags = sorted(
        set(ui_flags)
        & {"skill_catalog_blocked", "skill_missing_live", "skill_quarantined"}
    )
    checks = {
        "catalog_nonempty": bool(tool_names),
        "catalog_names_nonempty": bool(tool_names) and all(tool_names),
        "catalog_counts_match_payloads": (
            tools_payload.get("count") == len(tools)
            and skills_payload.get("count") == len(skills)
        ),
        "catalogs_identical": tool_names == skill_names == bootstrap_names,
        "health_identical": all(item == canonical_health for item in health.values()),
        "catalog_ready": canonical_health.get("ready") is True,
        # Against the vocabulary the catalog actually reports. This asked
        # for "ready", which `core.skills.discovery` has never produced, so
        # the check was False on every boot whatever the two
        # implementations said — and its unit test passed because the
        # fixture wrote "ready" into a health dict by hand.
        "catalog_parity_ready": (
            canonical_health.get("parity_status") in PARITY_STATES_THAT_AGREE
        ),
        "no_missing_live": not canonical_health.get("missing_live"),
        "no_quarantined": (
            not canonical_health.get("quarantined")
            and int(canonical_health.get("quarantined_count") or 0) == 0
        ),
        "preflight_complete": canonical_health.get("preflight_complete") is True,
        "preflight_ok": canonical_health.get("preflight_ok") is True,
        "preflight_has_no_failures": not canonical_health.get("preflight_failed"),
        "ui_has_no_skill_blocker": not blocked_flags,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "catalog_count": len(tool_names),
        "catalog_names_sha256": hashlib.sha256(
            "\n".join(tool_names).encode("utf-8")
        ).hexdigest(),
        "health": health,
        "payload_sha256": {
            "tools": _sha256_json(tools_payload),
            "skills": _sha256_json(skills_payload),
            "bootstrap": _sha256_json(bootstrap_payload),
        },
        "ui_status_flags": ui_flags,
    }


def is_competing_model_owner(process: ProcessObservation) -> bool:
    return _is_competing_model_owner(process.cmdline)


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", int(port))) != 0


class CleanLiveSkillBootProof:
    def __init__(
        self,
        *,
        port: int,
        boot_timeout_s: float,
        shutdown_timeout_s: float,
        artifact_dir: Path,
        observer: ResourceObserver | None = None,
    ) -> None:
        self.port = int(port)
        self.boot_timeout_s = float(boot_timeout_s)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self.artifact_dir = artifact_dir.resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_path = self.artifact_dir / "runtime_stdout.log"
        self.report_path = self.artifact_dir / "shutdown_report.json"
        self.verdict_path = self.artifact_dir / "VERDICT.json"
        self.observer = observer or get_resource_observer()
        self.proc: subprocess.Popen[bytes] | None = None
        self.stdout_handle: Any = None
        self.seen_identities: dict[tuple[int, float], ProcessObservation] = {}
        self.started_at = time.time()
        self.peak_rss_mb = 0.0
        self.rss_abort_mb = 0.0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _process_table(self) -> tuple[ProcessObservation, ...]:
        table = self.observer.process_table()
        if not table.available:
            raise RuntimeError(f"process table unavailable: {table.error}")
        return table.processes

    def preflight(self) -> dict[str, Any]:
        processes = self._process_table()
        aura_pids = sorted(
            item.pid for item in processes if _is_aura_main_process(item.cmdline)
        )
        model_owners = sorted(
            item.pid for item in processes if is_competing_model_owner(item)
        )
        checks = {
            "port_free": _port_is_free(self.port),
            "no_live_aura": not aura_pids,
            "no_competing_model_owner": not model_owners,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "aura_pids": aura_pids,
            "model_owner_pids": model_owners,
        }

    def _sample_owned_tree(self) -> float:
        if self.proc is None:
            return 0.0
        total = 0
        for item in self._process_table():
            if item.pid == self.proc.pid or self.proc.pid in item.ancestor_pids:
                self.seen_identities[(item.pid, item.create_time)] = item
                total += item.rss_bytes
        rss_mb = total / (1024 * 1024)
        self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
        if self.rss_abort_mb and rss_mb > self.rss_abort_mb:
            raise RuntimeError(
                f"owned process tree RSS {rss_mb:.0f}MB exceeded "
                f"{self.rss_abort_mb:.0f}MB ceiling"
            )
        return rss_mb

    def _spawn(self) -> None:
        env = build_safe_boot_env(os.environ, mode="headless", observer=self.observer)
        env["AURA_LOG_DIR"] = str(self.artifact_dir / "logs")
        env["AURA_ARTIFACTS_DIR"] = str(self.artifact_dir / "runtime_artifacts")
        env["AURA_SHUTDOWN_REPORT_PATH"] = str(self.report_path)
        env["PYTHONUNBUFFERED"] = "1"
        self.rss_abort_mb = live_proof_rss_abort_mb(env)
        self.stdout_handle = self.stdout_path.open("wb")
        command = [
            resolve_launch_python(),
            "aura_main.py",
            "--headless",
            "--port",
            str(self.port),
        ]
        self.proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=self.stdout_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._sample_owned_tree()

    def _runtime_text(self) -> str:
        if self.stdout_handle is not None:
            self.stdout_handle.flush()
        try:
            return self.stdout_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _wait_for_surface(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.boot_timeout_s
        last_error = "runtime_not_observed"
        last_result: dict[str, Any] = {}
        headers = {"X-Aura-Surface": "desktop-ui"}
        while time.monotonic() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError(
                    f"runtime exited before catalog readiness: "
                    f"rc={None if self.proc is None else self.proc.returncode}"
                )
            self._sample_owned_tree()
            try:
                with httpx.Client(timeout=10.0, headers=headers) as client:
                    tools_response = client.get(f"{self.base_url}/api/tools/catalog")
                    skills_response = client.get(f"{self.base_url}/api/skills")
                    bootstrap_response = client.get(f"{self.base_url}/api/ui/bootstrap")
                status_codes = {
                    "tools": tools_response.status_code,
                    "skills": skills_response.status_code,
                    "bootstrap": bootstrap_response.status_code,
                }
                if any(code != 200 for code in status_codes.values()):
                    last_error = f"route_status:{status_codes}"
                else:
                    last_result = evaluate_skill_surfaces(
                        tools_response.json(),
                        skills_response.json(),
                        bootstrap_response.json(),
                    )
                    last_result["status_codes"] = status_codes
                    ready_marker_seen = READY_MARKER in self._runtime_text()
                    last_result["ready_marker_seen"] = ready_marker_seen
                    if last_result.get("passed") and ready_marker_seen:
                        return last_result
                    failed = [
                        name
                        for name, passed in last_result.get("checks", {}).items()
                        if not passed
                    ]
                    last_error = f"catalog_not_ready:{failed}"
            except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5)
        raise RuntimeError(f"clean live catalog did not become ready: {last_error}")

    def _stop(self) -> dict[str, Any]:
        if self.proc is None:
            return {"passed": False, "reason": "runtime_not_spawned"}
        if self.proc.poll() is None:
            os.kill(self.proc.pid, signal.SIGTERM)
        deadline = time.monotonic() + self.shutdown_timeout_s
        while self.proc.poll() is None and time.monotonic() < deadline:
            self._sample_owned_tree()
            time.sleep(0.1)
        graceful = self.proc.poll() is not None
        if not graceful:
            self._kill_owned_group()
        if self.stdout_handle is not None:
            self.stdout_handle.flush()
            self.stdout_handle.close()
            self.stdout_handle = None
        time.sleep(1.0)
        residuals: list[int] = []
        current = {(item.pid, item.create_time): item for item in self._process_table()}
        for identity in self.seen_identities:
            item = current.get(identity)
            if item is not None and item.status.lower() not in {"dead", "zombie"}:
                residuals.append(item.pid)
        report: dict[str, Any] = {}
        try:
            value = json.loads(self.report_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                report = value
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
        report_checks = evaluate_terminal_report(
            report,
            root_pid=self.proc.pid,
            expected_first_reason="desktop_signal:SIGTERM",
            minimum_signal_requests=1,
        )
        checks = {
            "graceful_exit": graceful,
            "exit_code_zero": self.proc.returncode == 0,
            "no_residual_owned_process": not residuals,
            "port_free": _port_is_free(self.port),
            "terminal_shutdown_receipt_clean": bool(report_checks)
            and all(report_checks.values()),
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "report_checks": report_checks,
            "residual_pids": sorted(residuals),
        }

    def _kill_owned_group(self) -> None:
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

    def run(self) -> int:
        preflight = self.preflight()
        surface: dict[str, Any] = {}
        shutdown: dict[str, Any] = {"passed": False, "reason": "not_started"}
        error = ""
        if preflight["passed"]:
            try:
                self._spawn()
                surface = self._wait_for_surface()
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                shutdown = self._stop()
        runtime_text = self._runtime_text()
        failure_lines = [
            line[:1000]
            for line in runtime_text.splitlines()
            if any(marker.lower() in line.lower() for marker in FAILURE_MARKERS)
        ][:50]
        checks = {
            "preflight": preflight.get("passed") is True,
            "live_surface": surface.get("passed") is True,
            "shutdown": shutdown.get("passed") is True,
            "runtime_stream_clean": not failure_lines,
            "no_internal_error": not error,
        }
        verdict = {
            "schema": "aura.clean_live_skill_boot_proof.v1",
            "passed": all(checks.values()),
            "checks": checks,
            "preflight": preflight,
            "surface": surface,
            "shutdown": shutdown,
            "error": error,
            "failure_lines": failure_lines,
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "rss_abort_mb": round(self.rss_abort_mb, 1),
            "port": self.port,
            "started_at": self.started_at,
            "finished_at": time.time(),
            "resource_observation": self.observer.provenance.to_dict(),
        }
        self.verdict_path.write_text(
            json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            "AURA_CLEAN_LIVE_SKILL_BOOT_PROOF="
            + json.dumps(verdict, separators=(",", ":"), sort_keys=True),
            flush=True,
        )
        return 0 if verdict["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--boot-timeout", type=float, default=600.0)
    parser.add_argument("--shutdown-timeout", type=float, default=180.0)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    artifact_dir = args.artifact_dir or (
        ROOT / "artifacts" / "current" / f"clean_live_skill_boot_{stamp}"
    )
    proof = CleanLiveSkillBootProof(
        port=args.port,
        boot_timeout_s=args.boot_timeout,
        shutdown_timeout_s=args.shutdown_timeout,
        artifact_dir=artifact_dir,
    )
    return proof.run()


if __name__ == "__main__":
    raise SystemExit(main())
