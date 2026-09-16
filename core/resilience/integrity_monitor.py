"""core/resilience/integrity_monitor.py — System Integrity Monitor

Periodic health sweep that validates database integrity, service registration,
memory usage, and resource health. Runs every 5 minutes.
"""
try:
    from core.utils.exceptions import capture_and_log
except ImportError:
    def capture_and_log(e, ctx=None):
        logging.getLogger("Aura.IntegrityMonitor").error(f"Integrity Error: {e} | Context: {ctx}")
import asyncio
import logging
import os
import sqlite3
import time
from pathlib import Path

from core.resilience.substrate_monitor import SubstrateMonitor
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.IntegrityMonitor")

_SQLITE_MAGIC = b"SQLite format 3\x00"
_SQLITE_SUFFIXES = (".db", ".sqlite3", ".sqlite")
# Heavy artifact trees that hold no SQLite state worth sweeping every cycle.
_SWEEP_EXCLUDED_DIRS = {"training", "error_logs", "bench", "__pycache__"}

_DB_SWEEP_EVERY_N = declare(
    "AURA_INTEGRITY_DB_SWEEP_EVERY_N",
    kind=FlagKind.INT,
    default=6,
    description=(
        "Run the SQLite quick_check sweep every Nth integrity cycle (cycle 1 "
        "always sweeps, so boot is covered). At the default 300s cycle, 6 = "
        "one sweep per half hour instead of full integrity_check page scans "
        "of every store every 5 minutes."
    ),
    owner="core/resilience/integrity_monitor.py",
)


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
        self._task: asyncio.Task | None = None
        self._last_report: IntegrityReport | None = None
        self._check_count = 0
        self._last_findings: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
        self._last_db_checks: dict[str, str] = {}
        self._last_db_errors: list[str] = []
        self._proc = None
        self._substrate_monitor = SubstrateMonitor()
        try:
            from core.runtime import resource_psutil as psutil
            self._proc = psutil.Process(os.getpid())
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('integrity_monitor', e)
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

                self._report_findings(report)

            except asyncio.CancelledError:
                break
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('integrity_monitor', e)
                logger.error("Integrity monitor error: %s", e)

            await asyncio.sleep(self._interval)

    def _report_findings(self, report: IntegrityReport) -> None:
        """Say a finding once when it appears and once when it clears.

        "Thermal pressure is fair" was logged at warning every five minutes
        for a whole session — 230 lines for one standing condition. A
        condition that has not changed since the last check is not an event.
        """
        errors = tuple(str(e) for e in report.errors)
        warnings = tuple(str(w) for w in report.warnings)
        previous_errors, previous_warnings = self._last_findings
        self._last_findings = (errors, warnings)
        if errors:
            if errors != previous_errors:
                logger.error("🔍 INTEGRITY ERRORS: %s", list(errors))
            else:
                logger.info("🔍 Integrity errors unchanged (%d): %s", len(errors), list(errors))
            return
        if previous_errors:
            logger.info("🔍 Integrity errors cleared: %s", list(previous_errors))
        if warnings:
            if warnings != previous_warnings:
                logger.warning("🔍 Integrity warnings: %s", list(warnings))
            else:
                logger.info("🔍 Integrity warnings unchanged (%d): %s", len(warnings), list(warnings))
            return
        if previous_warnings:
            logger.info("🔍 Integrity warnings cleared: %s", list(previous_warnings))
        logger.info("🔍 Integrity check #%d passed", self._check_count)

    async def run_check(self, include_databases: bool | None = None) -> IntegrityReport:
        """Run a full integrity check.

        The DB sweep runs on the first cycle (boot coverage) and then every
        Nth cycle (AURA_INTEGRITY_DB_SWEEP_EVERY_N); services and resources
        are cheap and run every cycle. Callers can force either way.
        """
        report = IntegrityReport()

        if include_databases is None:
            every_n = max(1, int(_DB_SWEEP_EVERY_N.value() or 1))
            include_databases = self._check_count % every_n == 0

        # 1. Database integrity. A corrupt store is STATE, not an event:
        # skip-cycles re-report the last sweep's verdict so health surfaces
        # never show green over known corruption (degradations still fire
        # only on real sweeps).
        if include_databases:
            await asyncio.to_thread(self._check_databases, report)
            self._last_db_checks = dict(report.db_checks)
            self._last_db_errors = [e for e in report.errors if "DB " in e or "db" in e.lower()]
        else:
            report.db_checks.update(self._last_db_checks)
            report.errors.extend(self._last_db_errors)

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
            from core.runtime import resource_psutil as psutil
            total_mb = int(psutil.virtual_memory().total / (1024 * 1024))
        except (ImportError, AttributeError, RuntimeError):
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

    def _discover_sqlite_stores(self, max_stores: int = 200) -> list[Path]:
        """Every real SQLite store under the state roots, header-verified.

        The old sweep saw only top-level data/*.db — nested stores and
        *.sqlite3 files (most of the 28 live stores) were never checked.
        Suffix pre-filter keeps the walk cheap; the 16-byte header read
        confirms the file is genuinely SQLite before it is ever opened.
        """
        roots = [self._data_dir]
        for sibling in ("storage", ".aura_runtime"):
            candidate = self._data_dir.parent / sibling
            if candidate.is_dir():
                roots.append(candidate)
        stores: list[Path] = []
        for root in roots:
            if not root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(
                    d for d in dirnames if d not in _SWEEP_EXCLUDED_DIRS
                )
                for fname in sorted(filenames):
                    if len(stores) >= max_stores:
                        return stores
                    if fname.endswith(("-wal", "-shm")):
                        continue
                    path = Path(dirpath) / fname
                    if path.suffix.lower() not in _SQLITE_SUFFIXES:
                        continue
                    try:
                        with path.open("rb") as fh:
                            if fh.read(16) != _SQLITE_MAGIC:
                                continue
                    except OSError:
                        continue
                    stores.append(path)
        return stores

    def _check_databases(self, report: IntegrityReport):
        """PRAGMA quick_check every discovered SQLite store.

        quick_check, not integrity_check: the full scan reads every page of
        every store (~700MB per pass on a mature instance) and belongs on
        db_maintenance's weekly schedule, not a live monitor. quick_check
        still catches malformed pages and broken btrees — the corruption
        classes that actually strike this host.
        """
        if not self._data_dir.exists():
            report.warnings.append(f"Data directory {self._data_dir} does not exist")
            return

        for db_path in self._discover_sqlite_stores():
            try:
                db_name = str(db_path.relative_to(self._data_dir.parent))
            except ValueError:
                db_name = str(db_path)
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
                try:
                    conn.execute("PRAGMA busy_timeout=5000;")
                    result = conn.execute("PRAGMA quick_check(1);").fetchone()
                finally:
                    conn.close()

                if result and result[0] == "ok":
                    report.db_checks[db_name] = "ok"
                else:
                    msg = (
                        f"DB integrity failed: {db_name} — {result} "
                        "(see docs/runbooks/memory-corruption.md)"
                    )
                    report.db_checks[db_name] = str(result)
                    report.errors.append(msg)
                    logger.error("🔍 %s", msg)
                    record_degradation(
                        'integrity_monitor',
                        RuntimeError(msg),
                        severity="critical",
                        action="reported corrupt store; operator runbook memory-corruption",
                    )
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    report.db_checks[db_name] = "locked (skipped)"
                    report.warnings.append(f"{db_name} locked during integrity check")
                else:
                    report.db_checks[db_name] = f"error: {e}"
                    report.errors.append(f"DB check failed: {db_name} — {e}")
            except (sqlite3.Error, OSError) as e:
                record_degradation('integrity_monitor', e)
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
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('integrity_monitor', e)
            report.errors.append(f"Service check failed: {e}")

    def _check_resources(self, report: IntegrityReport):
        """Check system resource usage."""
        try:
            telemetry = self._substrate_monitor.sample(process=self._proc)
            report.memory_mb = telemetry.memory_mb
            report.memory_percent = telemetry.memory_percent
            report.cpu_percent = telemetry.cpu_percent
            if not telemetry.psutil_available:
                report.warnings.append("psutil unavailable; using generic resource fallback")
            # Preserve the overridable method for existing tests/operators while
            # the monitor supplies cross-platform thermal adapters underneath.
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
                except (ImportError, AttributeError, RuntimeError) as e:
                    record_degradation('integrity_monitor', e)
                    capture_and_log(e, {'module': __name__})
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('integrity_monitor', e)
            report.warnings.append(f"Resource check failed: {e}")

    def _get_thermal_level(self) -> int:
        """
        Retrieves host thermal level through the substrate monitor.
        Returns:
            int: 0=Nominal, 1=Fair, 2=Serious, 3=Critical.
                 Returns 0 if unavailable.
        """
        try:
            level, _pressure, _source = self._substrate_monitor.thermal()
            return int(level)
        except (RuntimeError, TypeError, ValueError, OSError) as exc:
            record_degradation('integrity_monitor', exc)
            logger.debug("Thermal probe failed: %s", exc)
            return 0

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
