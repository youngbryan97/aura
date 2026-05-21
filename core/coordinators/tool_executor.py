"""core/coordinators/tool_executor.py

Focused tool execution coordinator — extracted from cognitive_coordinator.py.

Handles:
- Single tool execution with feedback and learning
- Batch plan execution
- Swarm debate delegation
- Hephaestus forge fallback for missing tools
- ACG world model outcome recording
- Tool learning / reliability tracking

All execution is gated through AuthorityGateway -> UnifiedWill.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation

logger = logging.getLogger(__name__)

MAX_TOOL_NAME_CHARS = 128
MAX_ARG_REPR_CHARS = 2000

_TOOL_EXECUTOR_ERRORS = (
    AttributeError,
    ConnectionError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    sqlite3.Error,
)


def _record_tool_executor_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "tool_executor",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation(
                "tool_executor",
                error,
                severity=severity,
                action=action or "tool executor degraded",
            )
        except TypeError:
            logger.warning(
                "ToolExecutor degradation could not be recorded: %s",
                signature_exc,
            )


def _safe_tool_name(tool_name: object) -> str:
    try:
        name = str(tool_name or "").replace("\x00", "").strip()
    except (RuntimeError, TypeError, ValueError):
        name = ""
    return name[:MAX_TOOL_NAME_CHARS]


def _safe_args(args: object) -> dict[str, Any]:
    if isinstance(args, dict):
        return dict(args)
    return {}


def _compact_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {str(error)[:500]}"


class ToolExecutor:
    """Focused tool execution with learning and world model integration."""

    def __init__(self, orch: Any) -> None:
        self.orch = orch

    @staticmethod
    def _record_coding_tool_event(
        orch: Any,
        *,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        success: bool,
        error: str = "",
    ) -> None:
        try:
            from core.runtime.coding_session_memory import get_coding_session_memory

            get_coding_session_memory().record_tool_event(
                tool_name=tool_name,
                args=args,
                result=result,
                objective=getattr(orch, "_current_objective", "") or "",
                origin="tool_executor",
                success=success,
                error=error,
            )
        except _TOOL_EXECUTOR_ERRORS as exc:
            _record_tool_executor_degradation(
                exc,
                action="continued tool execution after coding session memory record failed",
                severity="warning",
                extra={"tool_name": _safe_tool_name(tool_name)},
            )
            logger.debug("ToolExecutor: coding tool recording skipped: %s", exc)

    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Execute a single tool with feedback, episodic recording, and learning."""
        orch = self.orch
        _start = time.time()
        tool_name = _safe_tool_name(tool_name)
        args = _safe_args(args)
        if not tool_name:
            result = {"ok": False, "error": "invalid_tool_name"}
            self._record_coding_tool_event(
                orch,
                tool_name="",
                args=args,
                result=result,
                success=False,
                error=result["error"],
            )
            return result

        current_objective = getattr(orch, "_current_objective", "") or ""
        emit_thought = getattr(orch, "_emit_thought_stream", None)

        # Swarm debate delegation
        if tool_name == "swarm_debate":
            try:
                swarm = getattr(orch, "swarm", None)
                if not swarm:
                    result = {"ok": False, "error": "Swarm Delegator not available."}
                    self._record_coding_tool_event(orch, tool_name=tool_name, args=args, result=result, success=False, error=result["error"])
                    return result
                topic = args.get("topic") or args.get("query") or current_objective
                roles = args.get("roles", ["architect", "critic"])
                if not isinstance(roles, list):
                    roles = ["architect", "critic"]
                if callable(emit_thought):
                    emit_thought(f"🐝 Engaging Swarm Debate: {str(topic)[:100]}...")
                result = await swarm.delegate_debate(str(topic), roles=roles)
                response = {"ok": True, "output": result}
                self._record_coding_tool_event(orch, tool_name=tool_name, args=args, result=response, success=True)
                return response
            except _TOOL_EXECUTOR_ERRORS as e:
                _record_tool_executor_degradation(
                    e,
                    action="returned structured swarm_debate failure",
                    severity="degraded",
                    extra={"tool_name": tool_name},
                )
                result = {
                    "ok": False,
                    "error": "execution_jolt",
                    "message": _compact_error(e),
                }
                self._record_coding_tool_event(
                    orch,
                    tool_name=tool_name,
                    args=args,
                    result=result,
                    success=False,
                    error=str(e),
                )
                return result

        try:
            router = getattr(orch, "router", None)
            if router is None or not hasattr(router, "execute"):
                result = {
                    "ok": False,
                    "error": "tool_router_unavailable",
                    "message": "Tool router is unavailable.",
                }
                self._record_coding_tool_event(
                    orch,
                    tool_name=tool_name,
                    args=args,
                    result=result,
                    success=False,
                    error=result["error"],
                )
                return result

            # Missing tool -> Hephaestus forge
            if tool_name not in getattr(router, "skills", {}):
                if tool_name == "notify_user":
                    result = {"ok": True, "message": args.get("message", "Done.")}
                    self._record_coding_tool_event(orch, tool_name=tool_name, args=args, result=result, success=True)
                    return result
                hephaestus = getattr(orch, "hephaestus", None)
                if hephaestus:
                    if callable(emit_thought):
                        emit_thought(
                        f"🔨 Tool '{tool_name}' missing. Initiating Autonomous Forge..."
                        )
                    objective = (
                        f"Create a skill '{tool_name}' to handle request "
                        f"within objective: {current_objective}"
                    )
                    forge_result = await hephaestus.synthesize_skill(tool_name, objective)
                    if forge_result.get("ok"):
                        if callable(emit_thought):
                            emit_thought(
                                f"✅ Skill '{tool_name}' forged successfully. Retrying..."
                            )
                        return await self.execute_tool(tool_name, args)
                    else:
                        logger.warning(
                            "Autogenesis failed for %s: %s",
                            tool_name,
                            forge_result.get("error"),
                        )
                result = {"ok": False, "error": f"Tool '{tool_name}' not found."}
                self._record_coding_tool_event(orch, tool_name=tool_name, args=args, result=result, success=False, error=result["error"])
                return result

            # Execute via router
            goal = {"action": tool_name, "params": args}
            context = {
                "objective": current_objective,
                "system": orch.status.__dict__,
                "stealth": (
                    await orch.stealth_mode.get_stealth_status()
                    if hasattr(orch, "stealth_mode")
                    and orch.stealth_mode
                    and getattr(orch.stealth_mode, "stealth_enabled", False)
                    else {}
                ),
                "liquid_state": (
                    orch.liquid_state.get_status()
                    if hasattr(orch, "liquid_state") and orch.liquid_state
                    else {}
                ),
            }
            result = await router.execute(goal, context)
            if not isinstance(result, dict):
                result = {"ok": True, "output": result}
            success = bool(result.get("ok", False))
            elapsed_ms = (time.time() - _start) * 1000
            logger.info("Tool %s execution completed: %s", tool_name, success)

            # Tool learning
            self._record_tool_learning(orch, tool_name, args, success, elapsed_ms)

            # Unified memory
            await self._record_memory(orch, tool_name, args, result, success)

            # ACG world model
            self._record_acg(goal, context, result, success)
            self._record_coding_tool_event(orch, tool_name=tool_name, args=args, result=result, success=success)

            # [RUBICON] Action Feedback Loop -- route to affect + body schema
            self._emit_action_feedback(
                tool_name, success, elapsed_ms,
                result_summary=str(result.get("output", result.get("result", "")))[:300],
                error_detail=str(result.get("error", ""))[:300] if not success else "",
                source="tool_executor",
            )

            return result

        except _TOOL_EXECUTOR_ERRORS as e:
            _record_tool_executor_degradation(
                e,
                action="returned structured tool failure and recorded crash context",
                severity="degraded",
                extra={
                    "tool_name": tool_name,
                    "args_preview": repr(args)[:MAX_ARG_REPR_CHARS],
                },
            )
            logger.error("Execution Jolt (Pain): Tool %s crashed: %s", tool_name, e)
            await self._record_crash(orch, tool_name, args, e)
            elapsed_ms = (time.time() - _start) * 1000
            result = {
                "ok": False,
                "error": "execution_jolt",
                "message": _compact_error(e),
            }
            self._record_coding_tool_event(orch, tool_name=tool_name, args=args, result=result, success=False, error=str(e))
            # [RUBICON] Action Feedback Loop -- record crash
            self._emit_action_feedback(
                tool_name, False, elapsed_ms,
                error_detail=str(e)[:300],
                source="tool_executor",
            )
            return result

    async def execute_plan(self, plan: dict[str, Any]) -> list[Any]:
        """Batch tool execution from a plan dict."""
        results = []
        plan = _safe_args(plan)
        tool_calls = plan.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            return [{"ok": False, "error": "invalid_tool_plan"}]
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                results.append({"ok": False, "error": "invalid_tool_call"})
                continue
            result = await self.execute_tool(
                tool_call.get("tool", "unknown"),
                tool_call.get("args", {}),
            )
            results.append(result)
        return results

    # ── Private helpers ────────────────────────────────────────────

    @staticmethod
    def _record_tool_learning(
        orch: Any, tool_name: str, args: dict[str, Any], success: bool, elapsed_ms: float
    ) -> None:
        if hasattr(orch, "tool_learner") and orch.tool_learner:
            try:
                category = orch.tool_learner.classify_task(
                    str(args.get("query", args.get("path", "")))
                )
                orch.tool_learner.record_usage(tool_name, category, success, elapsed_ms)
            except (OSError, ConnectionError, TimeoutError) as _e:
                _record_tool_executor_degradation(
                    _e,
                    action="continued after tool learning record failed",
                    severity="warning",
                    extra={"tool_name": _safe_tool_name(tool_name)},
                )
                logger.debug("Tool learning record failed: %s", _e)

    @staticmethod
    async def _record_memory(
        orch: Any, tool_name: str, args: dict[str, Any], result: Any, success: bool
    ) -> None:
        if hasattr(orch, "memory") and orch.memory:
            try:
                await orch.memory.commit_interaction(
                    context=str(args)[:500],
                    action=f"execute_tool({tool_name})",
                    outcome=str(result)[:500],
                    success=success,
                    importance=0.3 if success else 0.7,
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as _e:
                _record_tool_executor_degradation(
                    _e,
                    action="continued after unified memory tool record failed",
                    severity="warning",
                    extra={"tool_name": _safe_tool_name(tool_name)},
                )
                logger.debug("Unified memory record failed: %s", _e)

    @staticmethod
    def _record_acg(goal: dict[str, Any], context: dict[str, Any], result: Any, success: bool) -> None:
        try:
            from core.world_model.acg import acg as _acg_module

            _acg_module.record_outcome(
                action=goal,
                context=str(context)[:500],
                outcome=result,
                success=success,
            )
        except _TOOL_EXECUTOR_ERRORS as _e:
            _record_tool_executor_degradation(
                _e,
                action="continued after ACG outcome record failed",
                severity="warning",
                extra={"goal": repr(goal)[:MAX_ARG_REPR_CHARS]},
            )
            logger.debug("ACG record failed: %s", _e)

    @staticmethod
    def _emit_action_feedback(
        tool_name: str,
        success: bool,
        latency_ms: float,
        *,
        result_summary: str = "",
        error_detail: str = "",
        cost_tokens: int = 0,
        source: str = "",
    ) -> None:
        """[RUBICON] Route action feedback to the FeedbackProcessor.

        This closes the loop: tool execution -> affect + body schema + learning.
        """
        try:
            from core.somatic.action_feedback import get_feedback_processor
            fp = get_feedback_processor()
            fp.process_tool_result(
                tool_name,
                success,
                latency_ms,
                result_summary=result_summary,
                error_detail=error_detail,
                cost_tokens=cost_tokens,
                source=source,
            )
        except _TOOL_EXECUTOR_ERRORS as exc:
            _record_tool_executor_degradation(
                exc,
                action="continued after somatic action feedback emission failed",
                severity="warning",
                extra={"tool_name": _safe_tool_name(tool_name)},
            )
            logger.debug("ToolExecutor: action feedback emission failed: %s", exc)

    @staticmethod
    async def _record_crash(
        orch: Any,
        tool_name: str,
        args: dict[str, Any],
        error: BaseException,
    ) -> None:
        if hasattr(orch, "memory") and orch.memory:
            try:
                await orch.memory.commit_interaction(
                    context=str(args)[:500],
                    action=f"execute_tool({tool_name})",
                    outcome=f"CRASH: {type(error).__name__}",
                    success=False,
                    emotional_valence=-0.5,
                    importance=0.9,
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as _e:
                _record_tool_executor_degradation(
                    _e,
                    action="continued after crash memory record failed",
                    severity="warning",
                    extra={"tool_name": _safe_tool_name(tool_name)},
                )
                logger.debug("Unified memory record failed (crash path): %s", _e)
