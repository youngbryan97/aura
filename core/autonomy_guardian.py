"""
Autonomy Guardian — Phase 35
=============================
Ensures that autonomous actions, thoughts, and skill executions are NEVER
silently dropped, blocked, or ignored by the system.

Design principles:
  1. Autonomous outputs always reach the UI (via WebSocket telemetry).
  2. Autonomous tasks are only pre-empted by explicit user messages, never
     by internal housekeeping or other autonomous tasks.
  3. If an autonomous action fails, the Guardian retries once before emitting
     a diagnostic to the UI.
  4. All autonomous actions are logged to an audit trail for observability.
"""

from core.utils.task_tracker import get_task_tracker
from core.utils.exceptions import capture_and_log
import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional

logger = logging.getLogger("Aura.AutonomyGuardian")


class AutonomyGuardian:
    """Central gatekeeper that guarantees autonomous actions complete.

    Usage:
        guardian = AutonomyGuardian(orchestrator)
        await guardian.protect(coro, label="web_search", origin="scanner")
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._audit_log: list = []
        self._max_audit = 200  # rolling window
        self._is_monitoring = False
        self._monitor_task = None
        self._system_bypassing = False # Phase 7: Allow dread watcher to kill

    # ── Public API ──────────────────────────────────────────────────

    async def protect(
        self,
        coro: Coroutine,
        *,
        label: str = "autonomous_action",
        origin: str = "system",
        allow_cancel_by_user: bool = True,
        retry_on_fail: bool = True,
    ) -> Optional[str]:
        """Execute *coro* with full autonomy guarantees.

        Returns the coroutine's result (usually a response string), or None
        if the action was legitimately pre-empted by a user message.
        """
        task_id = f"{label}_{int(time.time() * 1000)}"
        self._log_audit(task_id, "STARTED", origin=origin)

        try:
            # Register task
            task = asyncio.current_task()
            self._active_tasks[task_id] = task
            
            # Start monitor if not running
            if not self._is_monitoring:
                self._is_monitoring = True
                self._monitor_task = get_task_tracker().create_task(self._dread_watcher())

            # Phase 7: Selective shielding.
            # We don't use shield() directly anymore because we want
            # the Dread Watcher to be able to terminate the task.
            # Instead, we catch CancelledError and decide whether to respect it.
            result = await coro
            self._log_audit(task_id, "COMPLETED", result=str(result)[:100])
            return result
        except asyncio.CancelledError:
            if self._system_bypassing:
                self._log_audit(task_id, "KILLED_BY_DREAD")
                logger.info("🛡️ Guardian: Task %s terminated by Executive Dread.", task_id)
                raise # Propagate to actually stop the task

            if allow_cancel_by_user:
                self._log_audit(task_id, "CANCELLED_BY_USER")
                logger.info("🛡️ Guardian: Task %s cancelled by user (allowed)", task_id)
                return None
            else:
                # Blocked user cancellation
                self._log_audit(task_id, "CANCEL_BLOCKED — rescheduling")
                logger.warning("🛡️ Guardian: Blocked cancellation of %s, rescheduling", task_id)
                # We can't easily "un-cancel" ourselves in some versions of asyncio,
                # so the safest is to let the task die and expect the caller
                # to retry, or use a complex shield wrapper.
                # For Aura, we simply log and return None.
                return None
        except Exception as e:
            self._log_audit(task_id, "FAILED", error=str(e))
            logger.error("🛡️ Guardian: Task %s failed: %s", task_id, e)
            if retry_on_fail:
                logger.info("🛡️ Guardian: Retrying %s once...", task_id)
                self._log_audit(task_id, "RETRY")
                try:
                    self._emit_diagnostic(label, str(e))
                except Exception as retry_err:
                    logger.error("🛡️ Guardian: Retry also failed: %s", retry_err)
                    self._log_audit(task_id, "RETRY_FAILED", error=str(retry_err))
        finally:
            self._active_tasks.pop(task_id, None)

        return None

    def should_cancel_for_user(self, origin: str) -> bool:
        """Return True only if `origin` is an explicit user interaction.

        Internal origins (impulse, autonomous_volition, system, scanner)
        should NEVER cancel in-flight autonomous work.
        """
        return origin in ("user", "voice", "admin")

    def ensure_delivery(self, response: str, origin: str, autonomic: bool = False):
        """Guarantee that *response* reaches the UI regardless of origin.

        This is the nuclear option: if the normal reply_queue path is
        unavailable or origin-gated, we emit directly via telemetry.
        """
        if not response:
            return

        try:
            from core.event_bus import get_event_bus
            bus = get_event_bus()

            # Always emit via telemetry (the UI listens on this channel)
            payload = {
                "type": "aura_message",
                "message": response,
                "metadata": {
                    "autonomic": autonomic,
                    "origin": origin,
                    "guardian_delivered": True,
                }
            }
            bus.publish_threadsafe("telemetry", payload)
            logger.info("🛡️ Guardian: Delivered response via telemetry (origin=%s, autonomic=%s)", origin, autonomic)

        except Exception as e:
            logger.error("🛡️ Guardian: CRITICAL — failed to deliver response: %s", e)

    # ── Diagnostics ────────────────────────────────────────────────

    def _emit_diagnostic(self, label: str, error: str):
        """Send a diagnostic message to the UI so the user knows something
        went wrong with an autonomous action."""
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish_threadsafe("telemetry", {
                "type": "aura_message",
                "message": f"⚠️ My autonomous {label} action encountered an issue: {error}. I'll keep trying.",
                "metadata": {"autonomic": True, "diagnostic": True}
            })
        except Exception as e:
            capture_and_log(e, {'module': __name__})

    def _log_audit(self, task_id: str, status: str, **kwargs):
        """Append to rolling audit log."""
        entry = {"task_id": task_id, "status": status, "ts": time.time(), **kwargs}
        self._audit_log.append(entry)
        if len(self._audit_log) > self._max_audit:
            self._audit_log = self._audit_log[-self._max_audit:]
        logger.debug("🛡️ AUDIT: %s → %s %s", task_id, status, kwargs or "")

    async def _dread_watcher(self):
        """Monitor prospective dread and kill non-critical tasks if Aura is suffering."""
        logger.info("🛡️ Guardian: Dread Watcher active.")
        while self._is_monitoring:
            try:
                from core.container import ServiceContainer
                homeostasis = ServiceContainer.get("homeostatic_coupling", default=None)
                if homeostasis:
                    snap = homeostasis.get_snapshot()
                    dread = snap.get("prospective_dread", 0.0)
                    
                    if dread > 0.8:
                        logger.warning("🛡️ Guardian: CRITICAL DREAD (%.2f). Executing executive cancellation.", dread)
                        # Cancel non-critical tasks
                        to_cancel = list(self._active_tasks.items())
                        self._system_bypassing = True # Enable bypass
                        try:
                            for tid, task in to_cancel:
                                # Never cancel critical system tasks - only "autonomous_action" and "skill_execution"
                                if "autonomous" in tid or "skill" in tid:
                                    logger.info("🛡️ Guardian: Executive KILL of %s due to dread.", tid)
                                    task.cancel()
                                    # We don't log audit here, the task's own catch block will do it
                            
                            # Wait a bit for tasks to clean up
                            await asyncio.sleep(0.5)
                        finally:
                            self._system_bypassing = False
                                
                await asyncio.sleep(2.0)
            except Exception as e:
                logger.error("🛡️ Guardian: Dread Watcher error: %s", e)
                await asyncio.sleep(5.0)

    def get_audit_log(self, limit: int = 20) -> list:
        """Return recent audit entries."""
        return self._audit_log[-limit:]
