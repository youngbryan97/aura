from core.utils.task_tracker import get_task_tracker
import logging
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional
from core.event_bus import get_event_bus
from core.container import ServiceContainer
from core.config import config

logger = logging.getLogger("Aura.IntegrityGuard")

class IntegrityGuard:
    """Sovereignty and Environment Integrity Guard.
    
    Protects Aura against process-level intrusion, debuggers, 
    and critical file system tampering.
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.bus = get_event_bus()
        self.active = False
        self._watchdog_task: Optional[asyncio.Task] = None
        self._last_sovereignty_score = 1.0
        self.called = False # v25: Track call state for deduplication

    async def start(self):
        """Starts the integrity watchdog."""
        if self.active:
            return
        self.active = True
        self._watchdog_task = get_task_tracker().create_task(self._watchdog_loop())
        logger.info("🛡️ Integrity Guard ACTIVE (PID/Sovereignty Protection)")

    async def on_start_async(self):
        """Standard lifecycle hook."""
        await self.start()

    async def stop(self):
        """Stops the integrity watchdog."""
        self.active = False
        if self._watchdog_task:
            self._watchdog_task.cancel()
        
        try:
            if self._watchdog_task: await self._watchdog_task
        except asyncio.CancelledError:
            logger.debug("IntegrityGuard: Shutdown requested.")
        logger.info("🛑 Integrity Guard STOPPED")

    async def on_stop_async(self):
        """Standard lifecycle hook."""
        await self.stop()

    def get_health(self) -> Dict[str, Any]:
        """Returns the health status of the integrity guard."""
        return {
            "status": "secure" if self._last_sovereignty_score > 0.8 else "at_risk",
            "score": self._last_sovereignty_score,
            "watchdog_active": self.active
        }

    def _get_project_root(self) -> Path:
        try:
            return Path(config.paths.project_root).resolve()
        except Exception:
            return Path(__file__).resolve().parents[2]

    def verify_sovereignty(self) -> float:
        """Verify the integrity of Aura's core environment.
        Returns a score from 0.0 to 1.0.
        """
        score = 1.0
        
        # 1. File Integrity Check
        critical_files = [
            "core/orchestrator/boot.py",
            "core/consciousness/global_workspace.py",
            "core/container.py"
        ]
        
        root = self._get_project_root()
             
        for rel_path in critical_files:
            abs_path = root / rel_path
            if not abs_path.exists():
                logger.critical("🛑 [SOVEREIGNTY] Critical file MISSING: %s", rel_path)
                score -= 0.3
        
        # 2. PID Watchdog: Check for debuggers or tracers
        try:
            import psutil
            process = psutil.Process(os.getpid())
            suspicious = ["gdb", "lldb", "debugpy", "pydevd"]
            chain = [process]
            try:
                chain.extend(process.parents())
            except Exception:
                parent = process.parent()
                if parent:
                    chain.append(parent)
            for proc in chain:
                name = ""
                try:
                    name = proc.name().lower()
                except Exception:
                    continue
                if any(s in name for s in suspicious):
                    logger.warning("🩸 [SOVEREIGNTY] Debugger detected in process chain: %s", proc.name())
                    score -= 0.5
                    break
        except Exception as e:
            logger.debug("IntegrityGuard: PID check failed (likely psutil/permissions): %s", e)
            
        self._last_sovereignty_score = max(0.0, score)
        return self._last_sovereignty_score

    async def _watchdog_loop(self):
        """Background loop for continuous environment verification."""
        # Initial heartbeat for immediate visibility
        try:
            audit = ServiceContainer.get("subsystem_audit", default=None)
            if audit:
                audit.heartbeat("sovereign_scanner")
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        while self.active:
            try:
                # Re-resolve audit inside loop to handle early boot timing
                audit = ServiceContainer.get("subsystem_audit", default=None)
                if audit:
                    audit.heartbeat("sovereign_scanner")
                
                score = self.verify_sovereignty()
                if score < 0.9:
                    mycelium = ServiceContainer.get("mycelial_network", default=None)
                    if mycelium:
                        await mycelium.emit_reflex("ENV_BREACH", {"score": score})
            except Exception as e:
                logger.error("Integrity watchdog error: %s", e)
            
            # Sleep 10s for the first minute, then 60s
            if not hasattr(self, "_ticks"): self._ticks = 0
            self._ticks += 1
            await asyncio.sleep(10 if self._ticks < 6 else 60)

    # ── Sovereign Scanner Interface (Patch 25) ───────────────────
    
    def _is_emergency(self) -> bool:
        """Check for critical environmental breach or resource exhaustion."""
        # 0.3 is the threshold where critical files are missing or a debugger is attached
        return self._last_sovereignty_score < 0.3

    async def scan(self, message: str) -> Dict[str, Any]:
        """v25 Hardening: SovereignScanner resilience."""
        self.called = True
        try:
            if self._is_emergency():
                logger.critical("🛑 [SCANNER] Emergency halt triggered by Sovereignty breach.")
                return {
                    "blocked": True, 
                    "reason": "Sovereignty Breach: System integrity compromised. Emergency halt engaged."
                }
            return await self._process_scan(message)
        except Exception as e:
            logger.error(f"Scanner failure: {e}")
            # v25: Fail-soft bypass to ensure the user isn't stuck behind a broken scanner
            return {"blocked": False, "reason": "BYPASS"}
        finally:
            self.called = False

    async def _process_scan(self, message: str) -> Dict[str, Any]:
        """Analyze message for forbidden patterns or recursive loops."""
        # This is the 'immunological' layer.
        # For now, it detects high-risk strings that could destabilize the cognitive kernel.
        forbidden = [
            "DETONATE_AURA_CORE",
            "REBOOT_SYSTEM_FORCE",
            "WIPE_STATE_REPOSITORY"
        ]
        
        msg_upper = message.upper()
        for f in forbidden:
            if f in msg_upper:
                return {
                    "blocked": True,
                    "reason": f"Restricted administrative command: {f}"
                }
        
        return {"blocked": False}
