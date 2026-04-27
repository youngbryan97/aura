"""core/resilience/integrity_monitor.py — System Integrity Monitor

Periodic health sweep that validates database integrity, service registration,
memory usage, and resource health. Runs every 5 minutes.
"""
try:
    from core.utils.exceptions import capture_and_log
except ImportError:
    def capture_and_log(e, ctx=None):
        logging.getLogger("Aura.IntegrityMonitor").error(f"Integrity Error: {e} | Context: {ctx}")
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import os
import sqlite3
import platform
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("Aura.IntegrityMonitor")


class IntegrityReport:
    """Results from an integrity check cycle."""
    def __init__(self):
        self.timestamp = time.time()
        self.db_checks = {}  # db_name -> "ok" | error
        self.service_checks = {}
        self.memory_mb: float = 0.0
        self.memory_percent: float = 0.0
        self.cpu_percent: float = 0.0
        self.thermal_level: int = 0  # 0=Nominal, 1=Fair, 2=Serious, 3=Critical
        self.warnings = []
        self.errors = []
        self.passed = True


class SystemIntegrityMonitor:
    """Runs periodic integrity checks across all subsystems."""

    _DEFAULT_MEMORY_WARNING_MB = 2048
    _DEFAULT_MEMORY_CRITICAL_MB = 4096
    _WARNING_RAM_FRACTION = 0.20
    _CRITICAL_RAM_FRACTION = 0.35

    def __init__(self, data_dir: str = "data", interval: float = 300.0):
        self._data_dir = Path(data_dir)
        self._interval = interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_report: Optional[IntegrityReport] = None
        self._check_count = 0
        self._proc = None
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
        except Exception as e:
            capture_and_log(e, {'module': __name__})

        # Critical services that must exist
        self._critical_services = [
            "cognitive_engine", "knowledge_graph", "mycelial_network",
            "capability_engine", "context_manager"
        ]
        # Non-critical but expected
        self._expected_services = [
            "agency_core", "subsystem_audit", "voice_engine",
            "personality_engine", "metabolic_monitor"
        ]
        # Memory thresholds scale to the host and still allow manual overrides.
        self._memory_warning_mb, self._memory_critical_mb = self._resolve_memory_thresholds()

    async def start(self):
        """Start periodic integrity checks."""
        self._running = True
        self._task = get_task_tracker().create_task(self._monitor_loop())
        logger.info("🔍 System Integrity Monitor started (interval=%ds)", self._interval)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def _monitor_loop(self):
        # Initial delay — let system boot before first check
        await asyncio.sleep(5)

        while self._running:
            try:
                report = await self.run_check()
                self._last_report = report
                self._check_count += 1

                if report.errors:
                    logger.error("🔍 INTEGRITY ERRORS: %s", report.errors)
                elif report.warnings:
                    logger.warning("🔍 Integrity warnings: %s", report.warnings)
                else:
                    logger.info("🔍 Integrity check #%d passed", self._check_count)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Integrity monitor error: %s", e)

            await asyncio.sleep(self._interval)

    async def run_check(self) -> IntegrityReport:
        """Run a full integrity check."""
        report = IntegrityReport()

        # 1. Database integrity
        await asyncio.to_thread(self._check_databases, report)

        # 2. Service registration
        self._check_services(report)

        # 3. System resources
        await asyncio.to_thread(self._check_resources, report)

        # 4. Determine overall status
        report.passed = len(report.errors) == 0
        return report

    def _resolve_memory_thresholds(self) -> tuple[int, int]:
        warning_override = os.getenv("AURA_INTEGRITY_MEMORY_WARNING_MB")
        critical_override = os.getenv("AURA_INTEGRITY_MEMORY_CRITICAL_MB")

        try:
            import psutil
            total_mb = int(psutil.virtual_memory().total / (1024 * 1024))
        except Exception:
            total_mb = 0

        warning_mb = self._DEFAULT_MEMORY_WARNING_MB
        critical_mb = self._DEFAULT_MEMORY_CRITICAL_MB
        if total_mb > 0:
            warning_mb = max(warning_mb, int(total_mb * self._WARNING_RAM_FRACTION))
            critical_mb = max(critical_mb, int(total_mb * self._CRITICAL_RAM_FRACTION))

        try:
            if warning_override:
                warning_mb = max(1, int(warning_override))
            if critical_override:
                critical_mb = max(1, int(critical_override))
        except ValueError:
            logger.warning(
                "Invalid integrity memory threshold override(s): warning=%r critical=%r",
                warning_override,
                critical_override,
            )

        if critical_mb <= warning_mb:
            critical_mb = warning_mb + 1024

        return warning_mb, critical_mb

    def _check_databases(self, report: IntegrityReport):
        """Run PRAGMA integrity_check on all .db files."""
        if not self._data_dir.exists():
            report.warnings.append(f"Data directory {self._data_dir} does not exist")
            return

        db_files = list(self._data_dir.glob("*.db"))
        for db_path in db_files:
            db_name = db_path.name
            try:
                conn = sqlite3.connect(str(db_path), timeout=10)
                conn.execute("PRAGMA busy_timeout=5000;")
                result = conn.execute("PRAGMA integrity_check;").fetchone()
                conn.close()

                if result and result[0] == "ok":
                    report.db_checks[db_name] = "ok"
                else:
                    msg = f"DB integrity failed: {db_name} — {result}"
                    report.db_checks[db_name] = str(result)
                    report.errors.append(msg)
                    logger.error("🔍 %s", msg)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    report.db_checks[db_name] = "locked (skipped)"
                    report.warnings.append(f"{db_name} locked during integrity check")
                else:
                    report.db_checks[db_name] = f"error: {e}"
                    report.errors.append(f"DB check failed: {db_name} — {e}")
            except Exception as e:
                report.db_checks[db_name] = f"error: {e}"
                report.warnings.append(f"DB check skipped: {db_name} — {e}")

    def _check_services(self, report: IntegrityReport):
        """Verify critical services exist in ServiceContainer. (FIXED: BUG-043)"""
        try:
            from core.container import ServiceContainer
            
            # Phase 43: Get all registered services once to avoid N lookups
            # if the container supports listing. If not, we just use the names.
            # Assuming ServiceContainer.get is the overhead, we'll try to get
            # the full registry if possible.
            registry = getattr(ServiceContainer, "_services", {})
            if not registry and hasattr(ServiceContainer, "get_all_services"):
                registry = ServiceContainer.get_all_services()
            
            registered_names = set(registry.keys()) if registry else None

            for svc in self._critical_services:
                if registered_names is not None:
                    exists = svc in registered_names
                else:
                    exists = ServiceContainer.get(svc, default=None) is not None
                
                report.service_checks[svc] = exists
                if not exists:
                    report.errors.append(f"Critical service missing: {svc}")

            for svc in self._expected_services:
                if registered_names is not None:
                    exists = svc in registered_names
                else:
                    exists = ServiceContainer.get(svc, default=None) is not None
                
                report.service_checks[svc] = exists
                if not exists:
                    report.warnings.append(f"Expected service missing: {svc}")
        except Exception as e:
            report.errors.append(f"Service check failed: {e}")

    def _check_resources(self, report: IntegrityReport):
        """Check system resource usage."""
        try:
            import psutil
            if not self._proc:
                self._proc = psutil.Process(os.getpid())
            
            # Use cached process to get meaningful diffs
            report.memory_mb = self._proc.memory_info().rss / (1024 * 1024)
            report.memory_percent = self._proc.memory_percent()
            
            # Normalize CPU usage by core count for 0-100% scale
            raw_cpu = self._proc.cpu_percent(interval=0.1)
            report.cpu_percent = raw_cpu / psutil.cpu_count()
            
            # --- Phase 7: Thermal Resonance ---
            report.thermal_level = self._get_thermal_level()
            if report.thermal_level >= 2: # Serious or Critical
                report.errors.append(f"CRITICAL thermal pressure: level {report.thermal_level}")
            elif report.thermal_level == 1:
                report.warnings.append("Thermal pressure is fair")
            if report.memory_mb > self._memory_critical_mb:
                report.errors.append(
                    f"CRITICAL memory usage: {report.memory_mb:.0f}MB "
                    f"(threshold: {self._memory_critical_mb}MB)"
                )
            elif report.memory_mb > self._memory_warning_mb:
                report.warnings.append(
                    f"High memory usage: {report.memory_mb:.0f}MB "
                    f"(threshold: {self._memory_warning_mb}MB)"
                )
            
            # --- Phase 5: Sentient Resource Resonance ---
            # Pulse stress to HomeostaticCoupling if we hit high load or thermal pressure
            if report.memory_mb > self._memory_warning_mb or report.cpu_percent > 80.0 or report.thermal_level > 0:
                try:
                    from core.container import ServiceContainer
                    homeostasis = ServiceContainer.get("homeostatic_coupling", default=None)
                    if homeostasis and hasattr(homeostasis, "process_resource_stress"):
                        homeostasis.process_resource_stress(
                            cpu_load=report.cpu_percent,
                            mem_mb=report.memory_mb,
                            thermal_level=report.thermal_level
                        )
                except Exception as e:
                    capture_and_log(e, {'module': __name__})
        except ImportError:
            report.warnings.append("psutil not available for resource checks")
        except Exception as e:
            report.warnings.append(f"Resource check failed: {e}")

    def _get_thermal_level(self) -> int:
        """
        Retrieves the thermal level on macOS using sysctl.
        Returns:
            int: 0=Nominal, 1=Fair, 2=Serious, 3=Critical.
                 Returns 0 if not macOS or sysctl fails.
        """
        if platform.system() == "Darwin":
            try:
                # sysctl -n hw.thermallevel (Standard Intel/Some M1)
                result = subprocess.run(
                    ["sysctl", "-n", "hw.thermallevel"],
                    capture_output=True,
                    text=True,
                    check=False # Don't raise if missing
                )
                if result.returncode == 0 and result.stdout.strip():
                    return int(result.stdout.strip())
            except (subprocess.CalledProcessError, ValueError):
                logger.debug('Ignored Exception in integrity_monitor.py: %s', "unknown_error")
            
            try:
                # Alternate check: Thermal Pressure
                # (This is often 0 for nominal, 1 for fair, etc.)
                result_alt = subprocess.run(
                    ["sysctl", "-n", "kern.thermal_pressure"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result_alt.returncode == 0 and result_alt.stdout.strip():
                    return int(result_alt.stdout.strip())
            except (subprocess.CalledProcessError, ValueError):
                logger.debug('Ignored Exception in integrity_monitor.py: %s', "unknown_error")
        return 0 # Default to nominal if not macOS or error

    def get_stats(self) -> dict:
        report = self._last_report
        return {
            "check_count": self._check_count,
            "last_check": report.timestamp if report else time.time(),
            "last_passed": report.passed if report else True,
            "memory_mb": report.memory_mb if report else 0.0,
            "memory_percent": report.memory_percent if report else 0.0,
            "cpu_percent": report.cpu_percent if report else 0.0,
            "db_status": report.db_checks if report else {},
            "warnings": list(report.warnings[:5]) if report and report.warnings else [],
            "errors": list(report.errors[:5]) if report and report.errors else [],
        }
