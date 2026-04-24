"""Tool Execution Mixin for RobustOrchestrator.
Extracts browser task and tool execution logic.
"""
import logging
import time
from typing import Any, Optional

from core.container import ServiceContainer

logger = logging.getLogger(__name__)
_USER_FACING_TOOL_ORIGINS = {
    "user",
    "api",
    "admin",
    "voice",
    "gui",
    "ws",
    "websocket",
    "direct",
    "external",
    "frontend",
    "ui",
}


class ToolExecutionMixin:
    """Handles tool execution with constitutional gating, episodic recording, and tool learning."""

    @staticmethod
    def _normalize_tool_origin(origin: Any) -> str:
        return str(origin or "").strip().lower().replace("-", "_")

    @classmethod
    def _coerce_tool_origin(cls, origin: Any) -> str:
        normalized = cls._normalize_tool_origin(origin)
        if not normalized:
            return ""
        if normalized in _USER_FACING_TOOL_ORIGINS:
            return normalized
        tokens = {token for token in normalized.split("_") if token}
        for candidate in ("user", "api", "voice", "admin", "gui", "websocket", "ws", "direct", "external"):
            if candidate in tokens:
                return candidate
        return normalized

    def _resolve_tool_origin(
        self,
        *,
        explicit_origin: Any = None,
        payload_context: Optional[dict[str, Any]] = None,
    ) -> str:
        candidates = [
            explicit_origin,
            (payload_context or {}).get("origin") if isinstance(payload_context, dict) else None,
            getattr(self, "_current_origin", ""),
            getattr(getattr(getattr(self, "state", None), "cognition", None), "current_origin", ""),
        ]
        for candidate in candidates:
            resolved = self._coerce_tool_origin(candidate)
            if resolved:
                return resolved
        return "unknown"

    async def run_browser_task(self, url: str, task: str) -> Any:
        """Formalized browser task execution via skill router.
        Rigor: No more stubs.
        """
        logger.info("🌐 Initiating Browser Task: %s @ %s", task, url)
        return await self.execute_tool("browser", {"url": url, "task": task})


    async def execute_tool(self, tool_name: str, args: dict[str, Any], **kwargs) -> Any:
        """Execute a single tool with feedback reporting, episodic recording, and tool learning"""
        _start = time.time()
        _constitution = None
        _tool_handle = None
        _constitutional_runtime_live = False
        kwargs = dict(kwargs or {})
        _origin = self._resolve_tool_origin(
            explicit_origin=kwargs.get("origin"),
            payload_context=kwargs.get("payload_context"),
        )
        kwargs.setdefault("origin", _origin)

        def _record_coding_tool_event(result: Any, *, success: bool, error: str = "") -> None:
            try:
                from core.runtime.coding_session_memory import get_coding_session_memory

                get_coding_session_memory().record_tool_event(
                    tool_name=tool_name,
                    args=args,
                    result=result,
                    objective=self._current_objective or "",
                    origin=_origin,
                    success=success,
                    error=error,
                )
            except Exception as _coding_exc:
                logger.debug("Coding session tool recording skipped: %s", _coding_exc)

        # ── UNIFIED WILL GATE ────────────────────────────────────────────
        try:
            from core.will import ActionDomain, get_will
            _will_decision = get_will().decide(
                content=f"tool:{tool_name} args:{str(args)[:100]}",
                source=_origin,
                domain=ActionDomain.TOOL_EXECUTION,
                priority=0.7,
            )
            if not _will_decision.is_approved():
                logger.warning("Unified Will REFUSED tool '%s': %s", tool_name, _will_decision.reason)
                result = {"ok": False, "error": f"Will refused: {_will_decision.reason}"}
                _record_coding_tool_event(result, success=False, error=str(_will_decision.reason))
                return result
        except Exception as _will_err:
            logger.debug("Unified Will tool gate degraded: %s", _will_err)
        # ─────────────────────────────────────────────────────────────────

        # ── EXECUTIVE APPROVAL GATE ──────────────────────────────────────
        try:
            from core.container import ServiceContainer as _SC
            from core.constitution import get_constitutional_core

            _constitutional_runtime_live = (
                _SC.has("executive_core")
                or _SC.has("aura_kernel")
                or _SC.has("kernel_interface")
                or bool(getattr(_SC, "_registration_locked", False))
            )
            _constitution = get_constitutional_core(self)
            _tool_handle = await _constitution.begin_tool_execution(
                tool_name,
                args,
                source=_origin,
                objective=self._current_objective or "",
            )
            if not _tool_handle.approved:
                reason = _tool_handle.decision.reason
                logger.warning("🚫 ExecutiveCore blocked tool '%s': %s", tool_name, reason)
                try:
                    from core.unified_action_log import get_action_log
                    get_action_log().record(tool_name, _origin, "tool", "blocked", str(reason))
                except Exception: pass
                result = {"ok": False, "error": f"Executive blocked: {reason}"}
                _record_coding_tool_event(result, success=False, error=str(reason))
                return result
            try:
                from core.unified_action_log import get_action_log
                get_action_log().record(tool_name, _origin, "tool", "approved")
            except Exception: pass
            if _tool_handle.constraints:
                kwargs.update(_tool_handle.constraints)  # Apply any degraded-mode constraints
        except Exception as _exec_err:
            if _constitutional_runtime_live:
                try:
                    from core.health.degraded_events import record_degraded_event

                    record_degraded_event(
                        "orchestrator",
                        "tool_gate_unavailable",
                        detail=tool_name,
                        severity="warning",
                        classification="foreground_blocking" if _origin in ("user", "voice", "admin", "api") else "background_degraded",
                        context={"error": type(_exec_err).__name__},
                        exc=_exec_err,
                    )
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
                logger.warning("🚫 ConstitutionalCore unavailable for tool '%s': %s", tool_name, _exec_err)
                result = {"ok": False, "error": "Constitutional tool gate unavailable"}
                _record_coding_tool_event(result, success=False, error=str(_exec_err))
                return result
            logger.debug("ConstitutionalCore unavailable for tool gate: %s", _exec_err)
        # ─────────────────────────────────────────────────────────────────

        if _constitutional_runtime_live and _tool_handle:
            try:
                from core.executive.authority_gateway import get_authority_gateway

                capability_token_id = _tool_handle.capability_token_id
                if not capability_token_id:
                    logger.warning("🚫 Tool '%s' missing capability token under constitutional runtime.", tool_name)
                    if _constitution and _tool_handle:
                        await _constitution.finish_tool_execution(
                            _tool_handle,
                            result={"ok": False, "error": "Capability token missing."},
                            success=False,
                            duration_ms=(time.time() - _start) * 1000,
                            error="Capability token missing.",
                        )
                    result = {"ok": False, "error": "Capability token missing."}
                    _record_coding_tool_event(result, success=False, error="Capability token missing.")
                    return result
                if not get_authority_gateway().verify_tool_access(tool_name, capability_token_id):
                    logger.warning("🚫 Capability token denied tool '%s'.", tool_name)
                    if _constitution and _tool_handle:
                        await _constitution.finish_tool_execution(
                            _tool_handle,
                            result={"ok": False, "error": "Capability token denied tool execution."},
                            success=False,
                            duration_ms=(time.time() - _start) * 1000,
                            error="Capability token denied tool execution.",
                        )
                    result = {"ok": False, "error": "Capability token denied tool execution."}
                    _record_coding_tool_event(result, success=False, error="Capability token denied tool execution.")
                    return result
                kwargs["capability_token_id"] = capability_token_id
            except Exception as capability_err:
                logger.warning("Capability verification failed for tool '%s': %s", tool_name, capability_err)
                if _constitution and _tool_handle:
                    await _constitution.finish_tool_execution(
                        _tool_handle,
                        result={"ok": False, "error": "Capability verification failed."},
                        success=False,
                        duration_ms=(time.time() - _start) * 1000,
                        error="Capability verification failed.",
                    )
                result = {"ok": False, "error": "Capability verification failed."}
                _record_coding_tool_event(result, success=False, error=str(capability_err))
                return result

        # 0. Virtual & Internal Tools
        if tool_name == "swarm_debate":
            if not self.swarm:
                if _constitution and _tool_handle:
                    await _constitution.finish_tool_execution(
                        _tool_handle,
                        result={"ok": False, "error": "Swarm Delegator not available."},
                        success=False,
                        duration_ms=(time.time() - _start) * 1000,
                        error="Swarm Delegator not available.",
                    )
                result = {"ok": False, "error": "Swarm Delegator not available."}
                _record_coding_tool_event(result, success=False, error="Swarm Delegator not available.")
                return result
            topic = args.get("topic") or args.get("query") or self._current_objective
            roles = args.get("roles", ["architect", "critic"])
            self._emit_thought_stream(f"🐝 Engaging Swarm Debate: {topic[:100]}...")
            result = await self.swarm.delegate_debate(topic, roles=roles, **kwargs)
            if _constitution and _tool_handle:
                await _constitution.finish_tool_execution(
                    _tool_handle,
                    result={"ok": True, "output": result},
                    success=True,
                    duration_ms=(time.time() - _start) * 1000,
                )
            response = {"ok": True, "output": result}
            _record_coding_tool_event(response, success=True)
            return response

        try:
            # 1. Check if tool exists in registry
            if tool_name not in self.router.skills:
                # Fallback for notify_user which is sometimes a virtual alias
                if tool_name == "notify_user":
                    if _constitution and _tool_handle:
                        await _constitution.finish_tool_execution(
                            _tool_handle,
                            result={"ok": True, "message": args.get("message", "Done.")},
                            success=True,
                            duration_ms=(time.time() - _start) * 1000,
                        )
                    result = {"ok": True, "message": args.get("message", "Done.")}
                    _record_coding_tool_event(result, success=True)
                    return result

                # 1.5 Autogenesis (Hephaestus Engine)
                if self.hephaestus:
                    self._emit_thought_stream(f"🔨 Tool '{tool_name}' missing. Initiating Autonomous Forge...")
                    objective = f"Create a skill '{tool_name}' to handle request within objective: {self._current_objective}"
                    forge_result = await self.hephaestus.synthesize_skill(tool_name, objective)
                    if forge_result.get("ok"):
                        self._emit_thought_stream(f"✅ Skill '{tool_name}' forged successfully. Retrying...")
                        if _constitution and _tool_handle:
                            await _constitution.finish_tool_execution(
                                _tool_handle,
                                result={"ok": True, "handoff": "autogenesis_retry", "tool_name": tool_name},
                                success=True,
                                duration_ms=(time.time() - _start) * 1000,
                            )
                        # Retry execution once
                        return await self.execute_tool(tool_name, args, **kwargs)
                    else:
                        logger.warning("Autogenesis failed for %s: %s", tool_name, forge_result.get("error"))

                if _constitution and _tool_handle:
                    await _constitution.finish_tool_execution(
                        _tool_handle,
                        result={"ok": False, "error": f"Tool '{tool_name}' not found."},
                        success=False,
                        duration_ms=(time.time() - _start) * 1000,
                        error=f"Tool '{tool_name}' not found.",
                    )
                result = {"ok": False, "error": f"Tool '{tool_name}' not found."}
                _record_coding_tool_event(result, success=False, error=result["error"])
                return result

            # 2. Contextual Awareness
            context = {
                "objective": self._current_objective,
                "system": self.status.model_dump(),
                "stealth": await self.stealth_mode.get_stealth_status() if hasattr(self, 'stealth_mode') and self.stealth_mode and getattr(self.stealth_mode, 'stealth_enabled', False) else {},
                "liquid_state": self.liquid_state.get_status() if hasattr(self, 'liquid_state') and self.liquid_state else {},
                **kwargs
            }

            # 2.5 Resistance Sandbox — emit prediction before execution
            _sandbox = None
            _sandbox_predicted = "success"
            try:
                from core.embodiment.resistance_sandbox import get_resistance_sandbox
                _sandbox = get_resistance_sandbox()
                _sandbox_predicted = "success" if tool_name not in ("browser", "shell", "file_write") else "success_with_side_effects"
            except Exception:
                _sandbox = None

            # 3. Literal Execution (Async)
            if _tool_handle is not None:
                from core.governance_context import governed_scope

                async with governed_scope(_tool_handle.decision):
                    result = await self.router.execute(tool_name, args, context)
            else:
                result = await self.router.execute(tool_name, args, context)

            success = result.get('ok', False)
            elapsed_ms = (time.time() - _start) * 1000
            logger.info("Tool %s execution completed: %s", tool_name, success)

            # 3.5 Resistance Sandbox — compare prediction to actual outcome
            if _sandbox is not None:
                try:
                    _actual_outcome = "success" if success else f"failure:{str(result.get('error', 'unknown'))[:80]}"
                    _sandbox.execute_with_prediction(
                        action_type="tool_exec",
                        target=tool_name,
                        predicted_outcome=_sandbox_predicted,
                        action_fn=lambda: _actual_outcome,
                    )
                except Exception as _sbx_err:
                    logger.debug("Resistance sandbox feedback failed: %s", _sbx_err)

            # 5. Tool Learning
            if hasattr(self, 'tool_learner') and self.tool_learner:
                try:
                    category = self.tool_learner.classify_task(str(args.get('query', args.get('path', ''))))
                    self.tool_learner.record_usage(tool_name, category, success, elapsed_ms)
                except Exception as _e:
                    logger.debug("Tool learning record failed: %s", _e)

            # 6. Episodic Recording (Now via Facade)
            if hasattr(self, 'memory') and self.memory:
                try:
                    await self.memory.commit_interaction(
                        context=str(args)[:500],
                        action=f"execute_tool({tool_name})",
                        outcome=str(result)[:500],
                        success=success,
                        importance=0.3 if success else 0.7,
                    )
                except Exception as _e:
                    logger.debug("Unified memory record failed: %s", _e)

            # 7. Causal Learning (ACG)
            try:
                from core.world_model.acg import acg
                acg.record_outcome(
                    action=tool_name,
                    context=str(context)[:500],
                    outcome=result,
                    success=success
                )
            except Exception as _e:
                logger.debug("ACG record failed: %s", _e)

            # WIRE-01: Affect State Update
        # Redundant local import removed
            affect_mgr = ServiceContainer.get("affect_engine", default=None)
            if affect_mgr:
                stimulus = "error" if not success else "intrigue"
                intensity = 15.0 if not success else 5.0
                self._fire_and_forget(
                    affect_mgr.apply_stimulus(stimulus, intensity),
                    name="orchestrator.affect_engine.apply_stimulus",
                )

            if _constitution and _tool_handle:
                await _constitution.finish_tool_execution(
                    _tool_handle,
                    result=result,
                    success=success,
                    duration_ms=elapsed_ms,
                )
            _record_coding_tool_event(result, success=success, error=str(result.get("error", "")))
            return result

        except Exception as e:
            logger.error("Execution Jolt (Pain): Tool %s crashed: %s", tool_name, e)
            # Record failure
            if hasattr(self, 'memory') and self.memory:
                try:
                    await self.memory.commit_interaction(
                        context=str(args)[:500],
                        action=f"execute_tool({tool_name})",
                        outcome=f"CRASH: {type(e).__name__}",
                        success=False,
                        emotional_valence=-0.5,
                        importance=0.9,
                    )
                except Exception as _e:
                    logger.debug("Unified memory record failed (crash path): %s", _e)
            if _constitution and _tool_handle:
                try:
                    await _constitution.finish_tool_execution(
                        _tool_handle,
                        result={"ok": False, "error": "execution_jolt", "message": str(e)},
                        success=False,
                        duration_ms=(time.time() - _start) * 1000,
                        error=str(e),
                    )
                except Exception as _finish_exc:
                    logger.debug("Constitutional tool completion failed: %s", _finish_exc)
            result = {"ok": False, "error": "execution_jolt", "message": str(e)}
            _record_coding_tool_event(result, success=False, error=str(e))
            return result

        logger.info("Orchestrator stopped")
