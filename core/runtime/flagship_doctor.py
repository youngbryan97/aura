"""Aura Flagship Doctor.

A one-command operational doctor for flagship readiness. This module avoids
importing heavyweight Aura subsystems at module import time. It checks the repo
layout, runtime version, expected guard modules, known ports, optional local log
evidence, and available quality gates.

It is intentionally conservative: a PASS means "basic operational readiness
checks are green", not "Aura is proven conscious" or "the whole product is
perfect".
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


from core.runtime.atomic_writer import atomic_write_text

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class DoctorFinding:
    code: str
    status: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DoctorReport:
    root: str
    created_at: float
    overall: str
    findings: list[DoctorFinding]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.flagship.doctor.v1",
            "root": self.root,
            "created_at": self.created_at,
            "overall": self.overall,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=repr)


REQUIRED_FILES = [
    "aura_main.py",
    "core/runtime/task_ownership.py",
    "core/runtime/persistence_ownership.py",
    "core/runtime/flagship_readiness.py",
    "core/morphogenesis/runtime.py",
    "core/morphogenesis/hooks.py",
    "core/morphogenesis/registry.py",
]

EXPECTED_SCRIPTS = [
    "scripts/aura_task_ownership_codemod.py",
    "scripts/aura_persistence_audit.py",
    "scripts/aura_collect_flagship_evidence.py",
]

EXPECTED_PORTS = [8000, 9090, 10003]


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.15) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return False


def _run(cmd: list[str], root: Path, timeout: float = 45.0) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(root), text=True, capture_output=True, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-6000:],
            "stderr_tail": proc.stderr[-6000:],
            "duration_s": round(time.time() - started, 3),
        }
    except (subprocess.SubprocessError, OSError) as exc:
        record_degradation('flagship_doctor', exc)
        return {
            "cmd": cmd,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.time() - started, 3),
        }


def _log_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for base in (root / "logs", Path.home() / ".aura" / "logs"):
        if base.exists():
            try:
                candidates.extend(sorted(base.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:10])
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass  # no-op: intentional
    return candidates


def _tail(path: Path, max_chars: int = 8000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[-max_chars:]
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return ""


def check_python() -> DoctorFinding:
    ok = sys.version_info >= (3, 12)
    return DoctorFinding(
        code="python_version",
        status="pass" if ok else "fail",
        message=f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        suggestion="Use Python 3.12+." if not ok else "",
    )


def check_layout(root: Path) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    for rel in REQUIRED_FILES:
        exists = (root / rel).exists()
        findings.append(
            DoctorFinding(
                code="required_file",
                status="pass" if exists else "fail",
                message=f"{rel} {'exists' if exists else 'missing'}",
                detail={"path": rel},
                suggestion=f"Restore or apply patch that provides {rel}." if not exists else "",
            )
        )
    for rel in EXPECTED_SCRIPTS:
        exists = (root / rel).exists()
        findings.append(
            DoctorFinding(
                code="expected_script",
                status="pass" if exists else "warn",
                message=f"{rel} {'exists' if exists else 'missing'}",
                detail={"path": rel},
                suggestion=f"Apply closure patches to install {rel}." if not exists else "",
            )
        )
    return findings


def check_aura_main(root: Path) -> list[DoctorFinding]:
    path = root / "aura_main.py"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    checks = [
        ("asyncio_patch_import", "core.utils.asyncio_patch" in text, "Import core.utils.asyncio_patch near the top of aura_main.py."),
        ("morphogenesis_boot", "start_morphogenesis_runtime" in text, "Start morphogenesis after core services are registered."),
        ("python_312_guard", "Python 3.12+" in text or "sys.version_info < (3, 12)" in text, "Normalize runtime guard to Python 3.12+."),
        ("security_guard", "validate_security_config" in text and "AURA_API_TOKEN" in text, "Keep fail-closed API security validation."),
    ]
    out: list[DoctorFinding] = []
    for code, ok, suggestion in checks:
        out.append(
            DoctorFinding(
                code=code,
                status="pass" if ok else "warn",
                message=f"{code}: {'present' if ok else 'not detected'}",
                suggestion="" if ok else suggestion,
            )
        )
    return out


def check_ports() -> list[DoctorFinding]:
    out: list[DoctorFinding] = []
    for port in EXPECTED_PORTS:
        open_ = _port_open(port)
        # Open ports are informational, not failure: Aura may not be running.
        out.append(
            DoctorFinding(
                code="port_probe",
                status="info",
                message=f"127.0.0.1:{port} {'open' if open_ else 'closed'}",
                detail={"port": port, "open": open_},
            )
        )
    return out


def check_logs(root: Path) -> list[DoctorFinding]:
    logs = _log_candidates(root)
    if not logs:
        return [
            DoctorFinding(
                code="logs",
                status="warn",
                message="No local Aura log files found.",
                suggestion="Run Aura once, then collect flagship evidence.",
            )
        ]
    joined = "\n".join(_tail(p) for p in logs[:5])
    probes = [
        ("log_morphogenesis_started", "MorphogeneticRuntime started" in joined),
        ("log_hooks_wired", "Morphogenesis hooks" in joined or "Morphogenesis hooks wired" in joined),
        ("log_task_supervisor", "Task Supervisor active" in joined or "TaskTracker" in joined),
        ("log_consciousness_online", "Consciousness System ONLINE" in joined),
    ]
    out: list[DoctorFinding] = [
        DoctorFinding("logs_found", "pass", f"Found {len(logs)} log file(s).", {"logs": [str(p) for p in logs[:5]]})
    ]
    for code, ok in probes:
        out.append(
            DoctorFinding(
                code=code,
                status="pass" if ok else "warn",
                message=f"{code}: {'detected' if ok else 'not detected in recent logs'}",
                suggestion="Run a fresh boot and collect evidence." if not ok else "",
            )
        )
    return out


def check_optional_gates(root: Path) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    commands = [
        ("flagship_readiness", [sys.executable, "-m", "core.runtime.flagship_readiness", "--json", "."]),
        ("task_ownership_report", [sys.executable, "scripts/aura_task_ownership_codemod.py", ".", "--json"]),
        ("persistence_audit", [sys.executable, "scripts/aura_persistence_audit.py", ".", "--json"]),
    ]
    for code, cmd in commands:
        module_or_script = cmd[1]
        runnable = module_or_script == "-m" or (root / module_or_script).exists()
        if not runnable:
            findings.append(DoctorFinding(code, "warn", f"{module_or_script} not available."))
            continue
        result = _run(cmd, root)
        rc = result.get("returncode")
        status = "pass" if rc == 0 else "warn"
        findings.append(DoctorFinding(code, status, f"{code} return code: {rc}", result))
    return findings


def run_doctor(root: str | Path, *, include_gates: bool = True) -> DoctorReport:
    root = Path(root).resolve()
    findings: list[DoctorFinding] = []
    findings.append(check_python())
    findings.extend(check_layout(root))
    findings.extend(check_aura_main(root))
    findings.extend(check_ports())
    findings.extend(check_logs(root))
    if include_gates:
        findings.extend(check_optional_gates(root))

    if any(f.status == "fail" for f in findings):
        overall = "fail"
    elif any(f.status == "warn" for f in findings):
        overall = "warn"
    else:
        overall = "pass"

    return DoctorReport(root=str(root), created_at=time.time(), overall=overall, findings=findings)


import gc
import sqlite3
import threading
import logging

logger = logging.getLogger("Aura.FlagshipDoctor")


class FlagshipDoctorDaemon:
    """Active background doctor daemon for event-loop latency tracking and database/memory self-healing."""

    def __init__(
        self,
        root_dir: str | Path | None = None,
        check_interval: float = 1.0,
        lag_threshold: float = 5.0,
        ram_threshold: float = 90.0,
    ) -> None:
        self.root = Path(root_dir or ".").resolve()
        self.check_interval = check_interval
        self.lag_threshold = lag_threshold
        self.ram_threshold = ram_threshold
        self._last_heartbeat = time.time()
        self._running = False
        self._monitor_thread: threading.Thread | None = None
        self._loop: Any = None

    def start(self, loop: Any = None) -> None:
        """Start the background monitoring thread and event-loop heartbeat updater."""
        if self._running:
            return
        
        import asyncio
        try:
            self._loop = loop or asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("FlagshipDoctorDaemon: No running event loop found during start. Heartbeat updater deferred.")
        
        self._running = True
        self._last_heartbeat = time.time()
        
        # Schedule the heartbeat task on the event loop
        if self._loop and self._loop.is_running():
            self._loop.create_task(self._heartbeat_updater())
            
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="AuraFlagshipDoctorDaemon"
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

    async def _heartbeat_updater(self) -> None:
        """Async task that constantly updates the heartbeat timestamp on the event loop."""
        import asyncio
        while self._running:
            self._last_heartbeat = time.time()
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break

    def _monitor_loop(self) -> None:
        """Standard thread loop running in the background to detect event-loop stalls or high memory."""
        logger.info("FlagshipDoctorDaemon background thread started.")
        
        while self._running:
            time.sleep(self.check_interval)
            if not self._running:
                break
                
            # 1. Event Loop Lag check
            lag = time.time() - self._last_heartbeat
            
            # 2. RAM Pressure check
            ram_percent = 0.0
            try:
                import psutil
                ram_percent = psutil.virtual_memory().percent
            except ImportError:
                pass
                
            # Trigger self-healing if limits are violated
            if lag > self.lag_threshold or (ram_percent > 0.0 and ram_percent >= self.ram_threshold):
                logger.warning(
                    "⚠️ [HEALTH DEGRADED] FlagshipDoctorDaemon triggered self-healing. Lag: %.2fs, RAM: %.1f%%",
                    lag,
                    ram_percent
                )
                try:
                    self._execute_self_healing(lag, ram_percent)
                except (RuntimeError, OSError, AttributeError, ValueError, TypeError, ImportError, sqlite3.Error) as e:
                    logger.error("FlagshipDoctorDaemon self-healing failed: %s", e)

    def _execute_self_healing(self, lag: float, ram_percent: float) -> None:
        """Executes garbage collection and compacts SQLite databases under pressure."""
        # 1. Run CPU Garbage Collection
        logger.info("♻️ Reclaiming memory via gc.collect()...")
        gc.collect()
        
        # 2. Compact SQLite Databases (specifically test_projects.db)
        db_paths = [
            self.root / "tests" / "test_projects.db",
            Path.home() / ".aura" / "live-source" / "tests" / "test_projects.db",
        ]
        
        compacted_count = 0
        for db_path in db_paths:
            if db_path.exists():
                try:
                    logger.info("🗄️ Compacting SQLite database: %s", db_path)
                    conn = sqlite3.connect(str(db_path), timeout=5.0)
                    conn.execute("VACUUM;")
                    conn.close()
                    compacted_count += 1
                except (sqlite3.Error, OSError, RuntimeError, ValueError) as e:
                    logger.error("Failed to compact DB %s: %s", db_path, e)
                    
        # 3. Trigger global database maintenance if available
        try:
            from core.persistence.db_maintenance import get_db_maintenance
            maint = get_db_maintenance()
            logger.info("🗄️ Triggering global DatabaseMaintenance pass...")
            maint.run_maintenance(force=True)
            compacted_count += 1
        except ImportError:
            pass
        except (RuntimeError, AttributeError, ValueError, TypeError, OSError) as e:
            logger.error("Global database maintenance run failed: %s", e)
            
        # 4. Record systemic degradation telemetry
        try:
            from core.runtime.errors import record_degradation
            record_degradation(
                "flagship_doctor",
                RuntimeError(f"Self-healing active: lag={lag:.2f}s, RAM={ram_percent:.1f}%"),
                severity="warning",
                action=f"reclaimed RAM with gc.collect() and compacted {compacted_count} databases"
            )
        except (ImportError, RuntimeError, AttributeError, ValueError, TypeError, OSError) as e:
            logger.error("Failed to record degradation telemetry: %s", e)


_daemon_instance: FlagshipDoctorDaemon | None = None


def get_flagship_doctor_daemon(root_dir: str | Path | None = None) -> FlagshipDoctorDaemon:
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = FlagshipDoctorDaemon(root_dir=root_dir)
    return _daemon_instance


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Aura flagship operational doctor")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-gates", action="store_true", help="Skip running slower external gates.")
    parser.add_argument("--out", default="", help="Optional report JSON path.")
    args = parser.parse_args(argv)

    report = run_doctor(args.root, include_gates=not args.no_gates)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out, report.to_json(), encoding="utf-8")
    if args.json:
        print(report.to_json())
    else:
        print(f"Aura flagship doctor: {report.overall.upper()}")
        for finding in report.findings:
            print(f"[{finding.status.upper()}] {finding.code}: {finding.message}")
            if finding.suggestion:
                print(f"  -> {finding.suggestion}")

    return 1 if report.overall == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
