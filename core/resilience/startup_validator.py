from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import async_atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.resource_observation import get_resource_observer

logger = logging.getLogger("Aura.StartupValidator")


@dataclass
class ValidationCheck:
    id: str
    name: str
    description: str
    critical: bool  # If True, failure prevents startup
    passed: bool = False
    message: str = ""


class StartupValidator:
    """
    Performs final checks before Aura is considered 'ONLINE'.
    Checks for:
    - Dangerous files (must NOT exist)
    - Essential services (must exist)
    - State integrity
    - Ethics safety configuration
    - Resource availability
    """

    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.checks: list[ValidationCheck] = [
            # Safety Checks
            ValidationCheck("safe_01", "Dangerous Files Purged", "Verify ethics-bypass files are deleted", True),
            ValidationCheck("safe_02", "Safe Backup Active", "Verify SafeBackupSystem is registered", False),
            
            # Resilience Checks
            ValidationCheck("res_01", "Stability Guardian Online", "Verify health monitoring is active", False),
            ValidationCheck("res_02", "Error Boundary Registry", "Verify circuit registry is initialized", True),
            ValidationCheck("res_03", "Research Cycle Ready", "Verify autonomous research engine is loaded", False),
            
            # Core Wiring
            ValidationCheck("core_01", "Kernel Interface Ready", "Verify communication with Aura Kernel", True),
            ValidationCheck("core_02", "LLM Protocol Valid", "Verify LLM organ is loaded and responding", True),
            ValidationCheck("core_03", "State Repository Bound", "Verify database connectivity", True),
            
            # Resources
            ValidationCheck("sys_01", "Memory Check", "Verify sufficient RAM for operation", False),
            ValidationCheck("sys_02", "Storage Check", "Verify write access to data directory", True),
            ValidationCheck("sys_03", "Zombie Reaper", "Clean up orphaned MLX workers", False),
        ]

    async def validate_all(self) -> bool:
        """Run all checks and return True if system is safe to start."""
        logger.info("StartupValidator: commencing system verification...")
        
        for check in self.checks:
            try:
                # Dispatch to specific handler
                handler_name = f"_check_{check.id}"
                handler = getattr(self, handler_name, None)
                if handler:
                    await handler(check)
                else:
                    # A declared check with no handler used to set passed=True
                    # and say "not implemented (ignored)". Every one of the
                    # eleven checks has a handler today, so nothing was being
                    # skipped — but "Dangerous Files Purged" is critical, and
                    # renaming its handler would have turned it green and let
                    # the boot through. An unrun check is not a passed check.
                    check.passed = False
                    check.message = f"NOT VERIFIED: no {handler_name} implemented"
                    record_degradation(
                        "startup_validator",
                        RuntimeError(f"{check.id} declared without {handler_name}"),
                        action="startup check reported as failed rather than passed",
                    )
            except (RuntimeError, AttributeError, TypeError) as e:
                record_degradation('startup_validator', e)
                check.passed = False
                check.message = f"Check crashed: {e}"

        # Calculate results
        failed_critical = [c for c in self.checks if not c.passed and c.critical]

        self.print_report()

        if failed_critical:
            logger.critical("STARTUP BLOCKED: %d critical validation failures.", len(failed_critical))
            return False

        logger.info("Startup validation SUCCESS. System state: SAFE.")
        return True

    # ── Handlers ──────────────────────────────────────────────────────────────

    async def _check_safe_01(self, c: ValidationCheck):
        from core.container import ServiceContainer

        dangerous = [
            "core/self_preservation_integration.py",
            "core/self_preservation_skills.py",
            "core/existential_awareness.py",
        ]
        found = []
        for path in dangerous:
            if Path(path).exists():
                found.append(path)

        legacy_runtime_active = bool(
            getattr(self.orchestrator, "self_preservation", None)
            or ServiceContainer.get("self_preservation", default=None)
        )
        safe_backup_active = bool(
            getattr(self.orchestrator, "backup_system", None)
            or ServiceContainer.get("backup_system", default=None)
        )

        if found:
            c.passed = False
            c.message = (
                "DANGER: unsafe legacy self-preservation files are present: "
                + ", ".join(found)
            )
        elif legacy_runtime_active:
            c.passed = False
            c.message = (
                "DANGER: legacy self-preservation runtime is active; "
                "safe backup path is not authoritative."
            )
        elif safe_backup_active:
            c.passed = True
            c.message = "Unsafe self-preservation path removed; safe backup active."
        else:
            c.passed = False
            c.message = (
                "Backup hardening missing; cannot confirm safe self-preservation "
                "runtime path."
            )

    async def _check_safe_02(self, c: ValidationCheck):
        from core.container import ServiceContainer
        if ServiceContainer.has("backup_system"):
            c.passed = True
            c.message = "SafeBackupSystem registered."
        else:
            c.passed = False
            c.message = "Backup system missing."

    async def _check_res_01(self, c: ValidationCheck):
        from core.container import ServiceContainer
        if ServiceContainer.has("stability_guardian"):
            c.passed = True
            c.message = "StabilityGuardian registered."
        else:
            c.passed = False
            c.message = "Guardian missing."

    async def _check_res_02(self, c: ValidationCheck):
        try:
            from core.resilience.error_boundary import get_circuit_registry
            reg = get_circuit_registry()
            c.passed = True
            c.message = f"Registry active with {len(reg.circuits)} circuits."
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('startup_validator', e)
            c.passed = False
            c.message = f"Registry failure: {e}"

    async def _check_res_03(self, c: ValidationCheck):
        from core.container import ServiceContainer
        if ServiceContainer.has("research_cycle"):
            c.passed = True
            c.message = "ResearchCycle active."
        else:
            c.passed = False
            c.message = "Cycle daemon offline (optional)."

    async def _check_core_01(self, c: ValidationCheck):
        ki = getattr(self.orchestrator, "kernel_interface", None)
        if ki and ki.is_ready():
            c.passed = True
            c.message = f"Kernel interface online (v{getattr(ki.kernel.state, 'version', 'unknown')})."
        else:
            c.passed = False
            c.message = "Kernel interface not ready."

    async def _check_core_02(self, c: ValidationCheck):
        ki = getattr(self.orchestrator, "kernel_interface", None)
        if not ki:
            c.passed = False; c.message = "No kernel interface"; return

        brain = ki.kernel.organs.get("brain") or ki.kernel.organs.get("llm")
        if brain and getattr(brain, "instance", None):
            c.passed = True
            c.message = f"Brain (LLM) active: {brain.instance.__class__.__name__}"
        else:
            c.passed = False
            c.message = "LLM organ or instance missing."

    async def _check_core_03(self, c: ValidationCheck):
        from core.container import ServiceContainer
        repo = ServiceContainer.get("state_repository", default=None)
        if not repo:
            c.passed = False; c.message = "State repository missing from container."; return

        # Read the repository once, then accept the already-authoritative kernel
        # or repository state immediately. The former loop slept for five seconds
        # before consulting that fallback on every healthy desktop boot.
        state = await repo.get_current()
        ki = getattr(self.orchestrator, "kernel_interface", None)
        kernel_state = getattr(getattr(ki, "kernel", None), "state", None) if ki else None
        fallback_state = kernel_state or getattr(repo, "_current", None)
        state_source = "repository"
        if state is None and fallback_state is not None:
            state = fallback_state
            state_source = "authoritative fallback"

        # A genuinely empty boot may still be waiting for the vault actor's SHM
        # handoff. Keep that bounded retry only when no authoritative state exists.
        for _attempt in range(9):
            if state is not None:
                break
            await asyncio.sleep(0.5)
            state = await repo.get_current()

        if state:
            c.passed = True
            c.message = f"State bound via {state_source} (v{state.version})."
        else:
            c.passed = False
            c.message = "State repository unreachable or empty (SHM sync failed)."

    async def _check_sys_01(self, c: ValidationCheck):
        try:
            memory = get_resource_observer().memory()
            if not memory.available:
                c.passed = False
                c.message = f"Memory observation unavailable: {memory.error or 'unknown error'}"
            elif memory.available_bytes < 500 * 1024 * 1024:  # 500MB
                c.passed = False
                c.message = f"Low memory available: {memory.available_bytes // 1024 // 1024}MB"
            else:
                c.passed = True
                c.message = (
                    f"Memory OK: {memory.available_bytes // 1024 // 1024}MB available "
                    f"({memory.provenance.source.value})."
                )
        except (ImportError, AttributeError, RuntimeError):
            c.passed = False
            c.message = "Memory observation failed."

    async def _check_sys_02(self, c: ValidationCheck):
        try:
            from core.config import config
            test_file = config.paths.data_dir / ".write_test"
            await async_atomic_write_text(test_file, "ok")
            test_file.unlink()
            c.passed = True
            c.message = f"Data dir writable: {config.paths.data_dir}"
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('startup_validator', e)
            c.passed = False
            c.message = f"Storage inaccessible: {e}"

    async def _check_sys_03(self, c: ValidationCheck):
        """Scans for and reaps orphaned mlx_worker processes."""
        try:
            import os
            import signal

            table = await asyncio.to_thread(get_resource_observer().process_table)
            if not table.available:
                raise RuntimeError(f"process_table_unavailable:{table.error}")

            reaped_count = 0
            for process in table.processes:
                try:
                    cmdline = " ".join(process.cmdline)
                    # Orphaned if PPID is 1 or parent PID is not running
                    is_orphaned = process.ppid == 1
                    
                    if "mlx_worker" in cmdline and is_orphaned:
                        logger.warning("💉 [REAPER] Reaping orphaned MLX worker (PID: %d)", process.pid)
                        os.kill(process.pid, signal.SIGKILL)
                        reaped_count += 1
                except (ProcessLookupError, PermissionError):
                    continue
            
            c.passed = True
            if reaped_count > 0:
                c.message = f"Reaped {reaped_count} orphaned workers."
            else:
                c.message = "No zombies found."
                
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('startup_validator', e)
            c.passed = False
            c.message = f"Reaper skipped: {e}"

    # ── UI ────────────────────────────────────────────────────────────────────

    def print_report(self) -> None:
        logger.info("\n" + "="*60)
        logger.info(" AURA STARTUP VALIDATION REPORT")
        logger.info("="*60)

        for check in self.checks:
            icon = "✓" if check.passed else ("!" if not check.critical else "✗")
            label = f"[{icon}] {check.name}"
            logger.info("%s | %s", f"{label:<30}", check.message)

        logger.info("="*60)
        final_status = "PASSED" if all(c.passed or not c.critical for c in self.checks) else "FAILED"
        logger.info(" FINAL STATUS: %s", final_status)
        logger.info("="*60 + "\n")
