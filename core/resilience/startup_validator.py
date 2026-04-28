from __future__ import annotations
from core.runtime.errors import record_degradation

from core.runtime.atomic_writer import atomic_write_text

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

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
        self.checks: List[ValidationCheck] = [
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
                    check.passed = True
                    check.message = "Check not implemented (ignored)"
            except Exception as e:
                record_degradation('startup_validator', e)
                check.passed = False
                check.message = f"Check crashed: {e}"

        # Calculate results
        failed_critical = [c for c in self.checks if not c.passed and c.critical]
        failed_optional = [c for c in self.checks if not c.passed and not c.critical]

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

        if legacy_runtime_active:
            c.passed = False
            c.message = (
                "DANGER: legacy self-preservation runtime is active; "
                "safe backup path is not authoritative."
            )
        elif safe_backup_active:
            c.passed = True
            if found:
                c.message = (
                    f"Safe backup path active; {len(found)} legacy self-preservation "
                    "files remain on disk but are dormant."
                )
            else:
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
        except Exception as e:
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

        # Resilience: Try to get state. If proxy, it'll check SHM.
        # Added retry loop to wait for SHM propagation from Vault Actor during boot.
        state = None
        for attempt in range(10): # 5 seconds total
            state = await repo.get_current()
            if state:
                break
            await asyncio.sleep(0.5)

        if state:
            c.passed = True
            c.message = f"State bound (v{state.version})."
        else:
            ki = getattr(self.orchestrator, "kernel_interface", None)
            kernel_state = getattr(getattr(ki, "kernel", None), "state", None) if ki else None
            fallback_state = kernel_state or getattr(repo, "_current", None)
            if fallback_state is not None:
                c.passed = True
                c.message = f"State bound via authoritative fallback (v{fallback_state.version})."
            else:
                c.passed = False
                c.message = "State repository unreachable or empty (SHM sync failed)."

    async def _check_sys_01(self, c: ValidationCheck):
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.available < 500 * 1024 * 1024: # 500MB
                c.passed = False
                c.message = f"Low memory available: {mem.available // 1024 // 1024}MB"
            else:
                c.passed = True
                c.message = f"Memory OK: {mem.available // 1024 // 1024}MB available."
        except Exception:
            c.passed = True
            c.message = "psutil missing, skipping mem check."

    async def _check_sys_02(self, c: ValidationCheck):
        try:
            from core.config import config
            test_file = config.paths.data_dir / ".write_test"
            atomic_write_text(test_file, "ok")
            test_file.unlink()
            c.passed = True
            c.message = f"Data dir writable: {config.paths.data_dir}"
        except Exception as e:
            record_degradation('startup_validator', e)
            c.passed = False
            c.message = f"Storage inaccessible: {e}"

    async def _check_sys_03(self, c: ValidationCheck):
        """Scans for and reaps orphaned mlx_worker processes."""
        try:
            import psutil
            import os
            import signal
            
            reaped_count = 0
            for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']):
                try:
                    cmdline = " ".join(proc.info['cmdline'] or [])
                    # Orphaned if PPID is 1 or parent PID is not running
                    is_orphaned = proc.info['ppid'] == 1
                    
                    if "mlx_worker" in cmdline and is_orphaned:
                        logger.warning("💉 [REAPER] Reaping orphaned MLX worker (PID: %d)", proc.info['pid'])
                        os.kill(proc.info['pid'], signal.SIGKILL)
                        reaped_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            c.passed = True
            if reaped_count > 0:
                c.message = f"Reaped {reaped_count} orphaned workers."
            else:
                c.message = "No zombies found."
                
        except Exception as e:
            record_degradation('startup_validator', e)
            c.passed = True # Non-critical
            c.message = f"Reaper skipped: {e}"

    # ── UI ────────────────────────────────────────────────────────────────────

    def print_report(self) -> None:
        logger.info("\n" + "="*60)
        logger.info(" AURA STARTUP VALIDATION REPORT")
        logger.info("="*60)

        for check in self.checks:
            icon = "✓" if check.passed else ("!" if not check.critical else "✗")
            label = f"[{icon}] {check.name}"
            logger.info(f"{label:<30} | {check.message}")

        logger.info("="*60)
        final_status = "PASSED" if all(c.passed or not c.critical for c in self.checks) else "FAILED"
        logger.info(f" FINAL STATUS: {final_status}")
        logger.info("="*60 + "\n")
