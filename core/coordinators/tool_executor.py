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
import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Focused tool execution with learning and world model integration."""

    def __init__(self, orch: Any) -> None:
        self.orch = orch

    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a single tool with feedback, episodic recording, and learning."""
        from core.utils.task_tracker import task_tracker
        from core.world_model.acg import acg as _acg_module

        orch = self.orch
        _start = time.time()

        # Swarm debate delegation
        if tool_name == "swarm_debate":
            if not orch.swarm:
                return {"ok": False, "error": "Swarm Delegator not available."}
            topic = args.get("topic") or args.get("query") or orch._current_objective
            roles = args.get("roles", ["architect", "critic"])
            orch._emit_thought_stream(f"🐝 Engaging Swarm Debate: {topic[:100]}...")
            result = await orch.swarm.delegate_debate(topic, roles=roles)
            return {"ok": True, "output": result}

        try:
            # Missing tool -> Hephaestus forge
            if tool_name not in orch.router.skills:
                if tool_name == "notify_user":
                    return {"ok": True, "message": args.get("message", "Done.")}
                if orch.hephaestus:
                    orch._emit_thought_stream(
                        f"🔨 Tool '{tool_name}' missing. Initiating Autonomous Forge..."
                    )
                    objective = (
                        f"Create a skill '{tool_name}' to handle request "
                        f"within objective: {orch._current_objective}"
                    )
                    forge_result = await orch.hephaestus.synthesize_skill(tool_name, objective)
                    if forge_result.get("ok"):
                        orch._emit_thought_stream(
                            f"✅ Skill '{tool_name}' forged successfully. Retrying..."
                        )
                        return await self.execute_tool(tool_name, args)
                    else:
                        logger.warning(
                            "Autogenesis failed for %s: %s",
                            tool_name,
                            forge_result.get("error"),
                        )
                return {"ok": False, "error": f"Tool '{tool_name}' not found."}

            # Execute via router
            goal = {"action": tool_name, "params": args}
            context = {
                "objective": orch._current_objective,
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
            result = await orch.router.execute(goal, context)
            success = result.get("ok", False)
            elapsed_ms = (time.time() - _start) * 1000
            logger.info("Tool %s execution completed: %s", tool_name, success)

            # Tool learning
            self._record_tool_learning(orch, tool_name, args, success, elapsed_ms)

            # Unified memory
            await self._record_memory(orch, tool_name, args, result, success)

            # ACG world model
            self._record_acg(goal, context, result, success)

            return result

        except Exception as e:
            logger.error("Execution Jolt (Pain): Tool %s crashed: %s", tool_name, e)
            await self._record_crash(orch, tool_name, args, e)
            return {"ok": False, "error": "execution_jolt", "message": str(e)}

    async def execute_plan(self, plan: Dict[str, Any]) -> List[Any]:
        """Batch tool execution from a plan dict."""
        results = []
        for tool_call in plan.get("tool_calls", []):
            result = await self.execute_tool(
                tool_call.get("tool", "unknown"),
                tool_call.get("args", {}),
            )
            results.append(result)
        return results

    # ── Private helpers ────────────────────────────────────────────

    @staticmethod
    def _record_tool_learning(
        orch: Any, tool_name: str, args: Dict, success: bool, elapsed_ms: float
    ) -> None:
        if hasattr(orch, "tool_learner") and orch.tool_learner:
            try:
                category = orch.tool_learner.classify_task(
                    str(args.get("query", args.get("path", "")))
                )
                orch.tool_learner.record_usage(tool_name, category, success, elapsed_ms)
            except Exception as _e:
                logger.debug("Tool learning record failed: %s", _e)

    @staticmethod
    async def _record_memory(
        orch: Any, tool_name: str, args: Dict, result: Any, success: bool
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
            except Exception as _e:
                logger.debug("Unified memory record failed: %s", _e)

    @staticmethod
    def _record_acg(goal: Dict, context: Dict, result: Any, success: bool) -> None:
        try:
            from core.world_model.acg import acg as _acg_module

            _acg_module.record_outcome(
                action=goal,
                context=str(context)[:500],
                outcome=result,
                success=success,
            )
        except Exception as _e:
            logger.debug("ACG record failed: %s", _e)

    @staticmethod
    async def _record_crash(orch: Any, tool_name: str, args: Dict, error: Exception) -> None:
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
            except Exception as _e:
                logger.debug("Unified memory record failed (crash path): %s", _e)
