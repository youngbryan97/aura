"""core/planning/recovery_engine.py — Failure Recovery & Replanning
===================================================================
Real desktops are messy. Aura needs to recover.

This engine handles failures without collapsing the entire mission.
Strategies (in order of preference):
  1. Retry with same adapter (transient failure)
  2. Switch to fallback adapter (e.g., Notes → TextEdit)
  3. Use programmatic fallback (e.g., PDF via renderer instead of app export)
  4. Skip non-critical step and continue
  5. Ask user for help (last resort)
  6. Honest failure report — NEVER pretend success

The recovery engine NEVER smoothly pretends things worked.
If she fails, she says so. No canned fallbacks.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.container import ServiceContainer
from core.planning.task_graph import TaskNode
from core.runtime.errors import record_degradation

if TYPE_CHECKING:
    from core.planning.mission_state import Mission

logger = logging.getLogger("Aura.RecoveryEngine")


# ---------------------------------------------------------------------------
# Recovery patterns
# ---------------------------------------------------------------------------

@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    node_id: str
    strategy: str
    original_error: str
    result: str
    success: bool
    timestamp: float = field(default_factory=time.time)


# Known error patterns → recovery strategies
ERROR_RECOVERY_MAP: dict[str, str] = {
    # App not found / not installed
    "application can't be found": "use_fallback_app",
    "is not running": "launch_and_retry",
    "does not exist": "use_fallback_app",
    "no such application": "use_fallback_app",

    # Permission / access
    "not allowed": "escalate_permission",
    "permission denied": "escalate_permission",
    "not authorized": "escalate_permission",
    "accessibility": "escalate_permission",

    # Timeout / hang
    "timed out": "retry_with_delay",
    "timeout": "retry_with_delay",

    # UI / interaction failures
    "can't get window": "retry_with_delay",
    "no front window": "launch_and_retry",
    "menu item": "use_hotkey_alternative",
    "click": "retry_with_screenshot",

    # Network
    "network": "retry_with_backoff",
    "connection": "retry_with_backoff",
    "dns": "retry_with_backoff",
    "ssl": "retry_with_backoff",

    # File conflicts
    "already exists": "version_filename",
    "file exists": "version_filename",

    # Login required
    "login": "ask_user",
    "sign in": "ask_user",
    "authenticate": "ask_user",

    # Verification failures
    "verification failed": "retry_then_fallback",

    # Generic
    "blocked": "skip_or_ask",
    "refused": "skip_or_ask",
}


class RecoveryEngine:
    """Recovers from failures without collapsing the entire mission.

    Never pretends success. If all recovery fails, produces an honest
    error message for narration.
    """

    def __init__(self, sleep: Callable[[float], Awaitable[None]] | None = None) -> None:
        self._attempts: list[RecoveryAttempt] = []
        self._max_attempts = 200
        self._started = False
        if sleep is None:
            import asyncio
            self._sleep = asyncio.sleep
        else:
            self._sleep = sleep

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("recovery_engine", self, required=False)
        self._started = True
        logger.info("RecoveryEngine ONLINE — failure recovery ready")

    async def recover(
        self,
        mission: Mission,
        node: TaskNode,
        error: str,
    ) -> bool:
        """Attempt to recover from a failed node.

        Returns True if recovery succeeded and the node can be considered
        handled (either succeeded via retry/fallback, or non-critical and skipped).
        Returns False if recovery failed and the node should be marked FAILED.
        """
        # Determine recovery strategy from error
        strategy = self._classify_error(error)
        logger.info(
            "Recovery for '%s' (error: %s) → strategy: %s",
            node.task_id, error[:60], strategy,
        )

        success = False
        result_msg = ""

        if strategy == "retry_with_delay":
            success, result_msg = await self._retry_with_delay(mission, node, error)

        elif strategy == "launch_and_retry":
            success, result_msg = await self._launch_and_retry(mission, node, error)

        elif strategy == "use_fallback_app":
            success, result_msg = await self._use_fallback_app(mission, node, error)

        elif strategy == "retry_then_fallback":
            success, result_msg = await self._retry_then_fallback(mission, node, error)

        elif strategy == "use_hotkey_alternative":
            success, result_msg = await self._use_hotkey_alternative(mission, node, error)

        elif strategy == "retry_with_screenshot":
            success, result_msg = await self._retry_with_screenshot(mission, node, error)

        elif strategy == "retry_with_backoff":
            success, result_msg = await self._retry_with_backoff(mission, node, error)

        elif strategy == "version_filename":
            success, result_msg = await self._version_filename(mission, node, error)

        elif strategy == "escalate_permission":
            success, result_msg = await self._escalate_permission(mission, node, error)

        elif strategy == "ask_user":
            success, result_msg = await self._ask_user(mission, node, error)

        elif strategy == "skip_or_ask":
            success, result_msg = await self._skip_or_ask(mission, node, error)

        else:
            # Unknown strategy — try generic retry then skip if non-critical
            success, result_msg = await self._generic_recovery(mission, node, error)

        # Record the attempt
        self._attempts.append(RecoveryAttempt(
            node_id=node.task_id,
            strategy=strategy,
            original_error=error[:200],
            result=result_msg[:200],
            success=success,
        ))
        if len(self._attempts) > self._max_attempts:
            self._attempts = self._attempts[-self._max_attempts:]

        if success:
            logger.info("Recovery SUCCEEDED for '%s': %s", node.task_id, result_msg[:80])
        else:
            logger.info("Recovery FAILED for '%s': %s", node.task_id, result_msg[:80])

        # Cross-episode learning ("learn from death"): record whether this recovery
        # strategy worked for this kind of step, so future planning can stop choosing
        # strategies that keep failing. LLM-free, automatic, best-effort.
        try:
            from core.planning.plan_failure_memory import get_plan_failure_memory

            goal = str(getattr(node, "description", "") or getattr(node, "task_id", "") or "")
            get_plan_failure_memory().record_outcome(
                goal, strategy, success=success, failure_mode="" if success else error,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass

        return success

    def _classify_error(self, error: str) -> str:
        """Map error text to a recovery strategy."""
        error_lower = error.lower()
        for pattern, strategy in ERROR_RECOVERY_MAP.items():
            if pattern in error_lower:
                return strategy
        return "generic_retry"

    # ------------------------------------------------------------------
    # Recovery strategies
    # ------------------------------------------------------------------

    async def _retry_with_delay(
        self, mission: Mission, node: TaskNode, error: str
    ) -> tuple[bool, str]:
        """Wait briefly, then retry the same action."""
        if node.retries_used >= node.retry_count:
            return False, f"Exhausted {node.retry_count} retries"

        await self._sleep(1.5)
        node.retries_used += 1
        if mission.graph:
            mission.graph.mark_retrying(node.task_id)

        # Re-execute via mission state
        try:
            ms = ServiceContainer.get("mission_state", default=None)
            if ms:
                # Pass the graph: a retried step still has placeholders to fill
                # from the steps that already succeeded. Retrying without it
                # would re-run the step with "{{generated_content}}" verbatim.
                result = await ms._execute_node(node, mission.graph)
                if result.get("success", False):
                    verified = await ms._verify_node(node)
                    if verified:
                        if mission.graph:
                            mission.graph.mark_succeeded(
                                node.task_id,
                                result=result,
                                receipt_id=result.get("receipt_id", ""),
                            )
                        return True, f"Retry #{node.retries_used} succeeded"
            return False, f"Retry #{node.retries_used} failed"
        except (ImportError, AttributeError, RuntimeError) as e:
            return False, f"Retry error: {e}"

    async def _launch_and_retry(
        self, mission: Mission, node: TaskNode, error: str
    ) -> tuple[bool, str]:
        """Launch the required app, then retry the action."""
        app_name = node.params.get("name", "") or node.params.get("app", "")
        if not app_name:
            return False, "No app name to launch"

        try:
            host = ServiceContainer.get("host_automation", default=None)
            if host is None:
                from core.capabilities.host_automation import get_host_automation
                host = get_host_automation()

            # Launch the app
            receipt = await host.launch_app(app_name)
            if not receipt.success:
                return False, f"Could not launch {app_name}: {receipt.error}"

            await self._sleep(1.0)

            # Retry original action
            return await self._retry_with_delay(mission, node, error)
        except (ImportError, AttributeError, RuntimeError) as e:
            return False, f"Launch+retry error: {e}"

    async def _use_fallback_app(
        self, mission: Mission, node: TaskNode, error: str
    ) -> tuple[bool, str]:
        """Switch to a fallback application for this step."""
        if not node.fallback_action:
            # Try to find an alternative from AppRegistry
            registry_error: str = ""
            try:
                registry = ServiceContainer.get("app_registry", default=None)
                if registry:
                    from core.capabilities.app_registry import AppAffordance
                    # Map action to affordance
                    action_affordance = {
                        "launch_app": None,
                        "create_text_file": AppAffordance.CREATE_TEXT,
                        "create_pdf": AppAffordance.EXPORT_PDF,
                        "open_url": AppAffordance.OPEN_URL,
                    }
                    affordance = action_affordance.get(node.action)
                    if affordance:
                        alt_app = registry.get_best_app_for(affordance)
                        if alt_app:
                            node.params["name"] = alt_app.name
                            return await self._retry_with_delay(mission, node, error)
            except (ImportError, AttributeError, RuntimeError) as exc:
                registry_error = f"; registry unavailable: {exc}"
                record_degradation("recovery_engine.fallback_registry", exc)

            # No fallback available — skip if non-critical
            if not node.critical:
                if mission.graph:
                    mission.graph.mark_skipped(node.task_id, f"No fallback for {error[:50]}")
                return True, f"Non-critical step skipped: {node.description}{registry_error}"
            return False, f"No fallback app available{registry_error}"

        # Use the explicit fallback
        node.action = node.fallback_action
        node.params = node.fallback_params or node.params
        node.retries_used = 0
        mission.narration_log.append(f"Trying fallback: {node.fallback_action}")
        return await self._retry_with_delay(mission, node, error)

    async def _retry_then_fallback(
        self, mission: Mission, node: TaskNode, error: str
    ) -> tuple[bool, str]:
        """Retry once, then try fallback if retry fails."""
        success, msg = await self._retry_with_delay(mission, node, error)
        if success:
            return True, msg
        return await self._use_fallback_app(mission, node, error)

    async def _use_hotkey_alternative(
        self, mission: Mission, node: TaskNode, error: str
    ) -> tuple[bool, str]:
        """If menu_select fails, try a keyboard shortcut instead."""
        # Common menu → hotkey mappings
        menu_to_hotkey = {
            "save": ["command", "s"],
            "copy": ["command", "c"],
            "paste": ["command", "v"],
            "undo": ["command", "z"],
            "print": ["command", "p"],
            "new": ["command", "n"],
            "open": ["command", "o"],
            "close": ["command", "w"],
            "quit": ["command", "q"],
            "select all": ["command", "a"],
            "find": ["command", "f"],
        }

        menu_path = node.params.get("path", [])
        if menu_path:
            last_item = menu_path[-1].lower()
            for key, hotkey in menu_to_hotkey.items():
                if key in last_item:
                    node.action = "hotkey"
                    node.params = {"keys": hotkey}
                    mission.narration_log.append(f"Menu failed, using hotkey: {'+'.join(hotkey)}")
                    return await self._retry_with_delay(mission, node, error)

        return False, "No hotkey alternative found"

    async def _retry_with_screenshot(
        self, mission: Mission, node: TaskNode, error: str
    ) -> tuple[bool, str]:
        """Take a screenshot to re-perceive the screen, then retry."""
        try:
            host = ServiceContainer.get("host_automation", default=None)
            if host:
                await host.take_screenshot()
                await self._sleep(0.5)
                return await self._retry_with_delay(mission, node, error)
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            record_degradation("recovery_engine.screenshot_retry", exc)
            return False, f"Screenshot+retry failed: {exc}"
        return False, "Screenshot+retry failed: host automation unavailable"

    async def _retry_with_backoff(
        self, mission: Mission, node: TaskNode, error: str
    ) -> tuple[bool, str]:
        """Exponential backoff retry for network errors."""
        max_retries = 3
        last_error = ""
        for attempt in range(max_retries):
            delay = 2.0 ** attempt  # 1s, 2s, 4s
            await self._sleep(delay)
            node.retries_used += 1
            try:
                ms = ServiceContainer.get("mission_state", default=None)
                if ms:
                    result = await ms._execute_node(node, mission.graph)
                    if result.get("success", False):
                        verified = await ms._verify_node(node)
                        if verified and mission.graph:
                            mission.graph.mark_succeeded(node.task_id, result=result)
                            return True, f"Backoff retry #{attempt + 1} succeeded"
                        last_error = "post-action verification failed"
            except (ImportError, AttributeError, RuntimeError) as exc:
                last_error = str(exc)
                record_degradation("recovery_engine.backoff_retry", exc)
                continue
        suffix = f": {last_error}" if last_error else ""
        return False, f"All {max_retries} backoff retries failed{suffix}"

    async def _version_filename(
        self, mission: Mission, node: TaskNode, error: str
    ) -> tuple[bool, str]:
        """If a file already exists, version the filename and retry."""
        from pathlib import Path
        path_str = node.params.get("path", "") or node.params.get("destination", "")
        if not path_str:
            return False, "No path to version"

        path = Path(path_str)
        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        for v in range(2, 10):
            new_path = parent / f"{stem}_v{v}{suffix}"
            if not new_path.exists():
                node.params["path"] = str(new_path)
                if "destination" in node.params:
                    node.params["destination"] = str(new_path)
                mission.narration_log.append(f"File exists, using: {new_path.name}")
                return await self._retry_with_delay(mission, node, error)

        return False, "Could not find available version number"

    async def _escalate_permission(
        self, mission: Mission, node: TaskNode, error: str
    ) -> tuple[bool, str]:
        """Log permission escalation need and pause for user."""
        mission.narration_log.append(
            f"I need permission for: {node.description or node.action}. "
            f"Error: {error[:80]}"
        )
        if not node.critical:
            if mission.graph:
                mission.graph.mark_skipped(node.task_id, f"Permission needed: {error[:60]}")
            return True, "Non-critical step skipped due to permission"
        return False, f"Permission required: {error[:100]}"

    async def _ask_user(
        self, mission: Mission, node: TaskNode, error: str
    ) -> tuple[bool, str]:
        """Pause the mission and ask the user for help."""
        mission.narration_log.append(
            f"I need your help with: {node.description or node.action}. "
            f"Reason: {error[:80]}"
        )
        from core.planning.mission_state import MissionStatus
        mission.status = MissionStatus.PAUSED
        return False, f"Mission paused — user help needed: {error[:80]}"

    async def _skip_or_ask(
        self, mission: Mission, node: TaskNode, error: str
    ) -> tuple[bool, str]:
        """Skip if non-critical, otherwise ask user."""
        if not node.critical:
            if mission.graph:
                mission.graph.mark_skipped(node.task_id, f"Blocked: {error[:60]}")
            return True, f"Non-critical step skipped: {node.description}"
        return await self._ask_user(mission, node, error)

    async def _generic_recovery(
        self, mission: Mission, node: TaskNode, error: str
    ) -> tuple[bool, str]:
        """Generic recovery: retry once, then skip if non-critical."""
        if node.retries_used < 1:
            success, msg = await self._retry_with_delay(mission, node, error)
            if success:
                return True, msg

        if not node.critical:
            if mission.graph:
                mission.graph.mark_skipped(node.task_id, f"Generic recovery: {error[:60]}")
            return True, "Non-critical step skipped after failed retry"

        return False, f"No recovery strategy for: {error[:100]}"

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_recent_attempts(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            {
                "node_id": a.node_id,
                "strategy": a.strategy,
                "error": a.original_error[:80],
                "result": a.result[:80],
                "success": a.success,
            }
            for a in self._attempts[-limit:]
        ]

    def get_status(self) -> dict[str, Any]:
        total = len(self._attempts)
        successes = sum(1 for a in self._attempts if a.success)
        return {
            "total_attempts": total,
            "successes": successes,
            "recovery_rate": round(successes / max(1, total), 3),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: RecoveryEngine | None = None


def get_recovery_engine() -> RecoveryEngine:
    global _instance
    if _instance is None:
        _instance = RecoveryEngine()
    return _instance


__all__ = ["RecoveryEngine", "RecoveryAttempt", "get_recovery_engine"]
