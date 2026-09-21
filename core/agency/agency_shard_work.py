"""What one shard of the swarm actually does.

Four methods of SovereignSwarm: spawning the count, normalising the tools a
shard asked for, executing one of them, and the wrapper that runs a shard
end to end. They are called only by the swarm and change when a shard's work
changes, which is not when the agency core changes.


Lifted whole out of `agency_core`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


class _RunsOneShard:
    """Lifted whole out of SovereignSwarm; see agency_core.py."""

    def _publish_shard_count(self) -> None:
        """Advertise the number of shards that actually exist right now."""
        from .agency_core import (
            _AGENCY_BOUNDARY_ERRORS,
            _record_agency_degradation,
            _schedule_agency_task,
            capture_and_log,
            get_registry,
        )

        try:
            if self._registry_shards_update_pending:
                return
            self._registry_shards_update_pending = True
            observed = len(self.active_shards)

            async def _run_shards_update() -> None:
                try:
                    await get_registry().update(active_shards=observed)
                finally:
                    self._registry_shards_update_pending = False

            registry_task = _schedule_agency_task(
                _run_shards_update(),
                name="agency.registry.active_shards",
                on_unscheduled=lambda: setattr(self, "_registry_shards_update_pending", False),
            )
            if registry_task is None:
                self._registry_shards_update_pending = False
        except _AGENCY_BOUNDARY_ERRORS as e:
            self._registry_shards_update_pending = False
            _record_agency_degradation(e, action="active shard count registry update skipped")
            capture_and_log(e, {"context": "AgencyCore._publish_shard_count"})

    @staticmethod
    def _normalize_tool_requests(raw_tools: Any, tool_name: Any = None, tool_payload: Any = None) -> list[dict[str, Any]]:
        """Normalize shard tool declarations into executable name/payload pairs."""
        tools_list = raw_tools or []
        if tool_name and tool_payload is not None and not tools_list:
            tools_list = [{"name": tool_name, "payload": tool_payload}]
        normalized: list[dict[str, Any]] = []
        if not isinstance(tools_list, list):
            return normalized

        for raw_tool in tools_list:
            tool_spec = raw_tool.model_dump() if hasattr(raw_tool, "model_dump") else raw_tool
            if not isinstance(tool_spec, dict):
                continue
            name = tool_spec.get("name") or tool_spec.get("tool_name")
            if not name:
                continue
            payload = tool_spec.get("payload") if "payload" in tool_spec else tool_spec.get("tool_payload")
            if payload is None:
                continue
            normalized.append({"name": str(name), "payload": payload})
        return normalized

    async def _execute_shard_tool(self, name: str, payload: Any) -> Any:
        from .agency_core import (
            ServiceContainer,
            _AGENCY_BOUNDARY_ERRORS,
            inspect,
        )

        owner_core = self.agency_core
        orch_core = (
            getattr(self.orch, "_agency_core", None)
            or getattr(self.orch, "agency_core", None)
            if self.orch is not None
            else None
        )
        container_core = ServiceContainer.get("agency_core", default=None)
        orchestrator = (
            getattr(owner_core, "tool_orchestrator", None)
            or getattr(orch_core, "tool_orchestrator", None)
            or getattr(container_core, "tool_orchestrator", None)
            or getattr(self, "tool_orchestrator", None)
        )
        if orchestrator is None:
            error = RuntimeError("tool_orchestrator_unavailable")
            if owner_core is not None:
                owner_core._last_tool_routing_error = str(error)
            raise error
        try:
            from core.governance_context import local_internal_governed_scope

            with local_internal_governed_scope(
                "agency.sovereign_swarm.shard_tool",
                domain="tool_execution",
                constraints={
                    "tool_name": str(name),
                    "autonomous_background": True,
                    "requires_sandbox": str(name) == "python_sandbox",
                    "source": "SovereignSwarm",
                },
            ):
                result = orchestrator.route_and_execute(name, payload)
                if inspect.isawaitable(result):
                    result = await result
            if owner_core is not None:
                owner_core._last_tool_routing_error = None
            return result
        except _AGENCY_BOUNDARY_ERRORS as exc:
            if owner_core is not None:
                owner_core._last_tool_routing_error = f"{type(exc).__name__}: {exc}"
            raise

    async def _shard_wrapper(self, goal: str, context: str, shard_id: str='unknown') -> None:
        """Internal execution of a thinking shard."""
        from .agency_core import (
            ServiceContainer,
            _AGENCY_BOUNDARY_ERRORS,
            _DISPATCHABLE_SHARD_TOOLS,
            _HIGH_RISK_SHARD_TOOLS,
            _record_agency_degradation,
            _record_durable_insight,
            _schedule_agency_task,
            capture_and_log,
            inspect,
            logger,
        )

        try:
            engine = getattr(self.orch, "cognitive_engine", None)
            if not engine:
                return

            prompt = f"""[SOVEREIGN SWARM SHARD]
GOAL: {goal}
CONTEXT: {context}

You are an autonomous cognitive shard. If you encounter a domain you do not understand, DO NOT GUESS. You have tools available.

To use a tool, specify it in the JSON fields "tool_name" and "tool_payload".
Available tools:
- "python_sandbox": Run Python code. Pass the code as the payload.
- "web_search": Search the web. Pass the query as the payload.

Synthesize a brief, insightful conclusion or action.

CRITICAL: You MUST respond with a valid JSON object matching the following structure:
{{
  "analysis": "your internal thought process",
  "action_type": "one of: 'observation', 'tool_use', 'conclusion', 'thought'",
  "tool_name": "optional tool name",
  "tool_payload": "optional tool parameters",
  "conclusion": "final takeaway"
}}
"""
            # 2. Autonomous Thought Synthesis (With Strict Pydantic Enforcement)
            from core.brain.llm.structured_llm import StructuredLLM
            from core.schemas import ShardResponse

            structured_brain = StructuredLLM(ShardResponse, max_retries=3)

            async with self._inference_semaphore:
                # Use StructuredLLM for guaranteed formatting and self-correction
                shard_res = await structured_brain.generate(prompt, context=context)

            if not shard_res:
                defer_reason = str(getattr(structured_brain, "last_defer_reason", "") or "")
                if defer_reason:
                    logger.info(
                        "⏸️ Swarm: Shard %s deferred before generation (%s).",
                        shard_id,
                        defer_reason,
                    )
                    return
                logger.error("💀 Swarm: Shard %s failed to generate valid response after retries. Applying fallback logic.", shard_id)
                # Apply fallback instead of silent death to avoid zombie state
                shard_res = ShardResponse(
                    analysis="Structured output failed. Engaging safe fallback mode.",
                    action_type="conclusion",
                    conclusion="I am experiencing cognitive formatting degradation. Taking a defensive posture."
                )
                try:
                    from core.health.degraded_events import record_degraded_event
                    record_degraded_event(
                        "agency_core",
                        "shard_formatting_collapse",
                        detail=f"Shard {shard_id} failed to produce valid JSON after retries.",
                        severity="warning",
                        classification="cognitive_degradation",
                        context={"shard_id": shard_id, "goal": goal}
                    )
                except _AGENCY_BOUNDARY_ERRORS as e:
                    _record_agency_degradation(e, action="shard formatting degraded-event receipt skipped")

                try:
                    _schedule_agency_task(
                        self._active_self_repair_formatting(shard_id, goal),
                        name=f"swarm_self_repair_{shard_id}",
                    )
                except _AGENCY_BOUNDARY_ERRORS as e:
                    _record_agency_degradation(e, action="shard formatting repair task skipped")
                    logger.error("Failed to spawn self-repair task: %s", e)

                shard_res.completed_with_degradation = True

            # One name for "this shard produced something it actually reasoned
            # to". The fallback above fabricates a conclusion so the shard does
            # not die silently, which is right — a zombie shard is worse — but
            # everything downstream then treated that sentence as a result:
            # the abstraction engine learned from it as a SUCCESS, the crucible
            # dialectically refined it, and the collective was pulsed with
            # success=True. A formatting collapse became a lesson.
            shard_succeeded = not bool(
                getattr(shard_res, "completed_with_degradation", False)
            )

            analysis_text = shard_res.analysis
            output_text = shard_res.conclusion
            tool_name = shard_res.tool_name
            tool_payload = shard_res.tool_payload

            logger.info("🧠 Shard %s Monologue: %s", shard_id, analysis_text[:100] + "...")

            tool_name = getattr(shard_res, "tool_name", None)
            tool_payload = getattr(shard_res, "tool_payload", None)
            tools_list = self._normalize_tool_requests(getattr(shard_res, "tools", []), tool_name, tool_payload)

            if tools_list:
                tasks = []
                approved_tools = []
                blocked_tools = []

                for t in tools_list:
                    name = t["name"]
                    payload = t["payload"]
                    is_blocked = False
                    # CP126 be7d1d4f. Every safety check below could be
                    # ABSENT or FAIL and leave the tool approved: dvg None,
                    # dvg raising, cwm None, cwm raising — all fell through
                    # to approval. For python_sandbox / shell_executor /
                    # file_operations that is the whole blast radius of the
                    # agent. "We could not check" is not "it is safe".
                    _high_risk = name in _HIGH_RISK_SHARD_TOOLS
                    _value_check_done = not _high_risk
                    _causal_check_done = not _high_risk

                    # A tool name comes from model output. Without a local
                    # allowlist an invented name reaches dispatch unchecked.
                    if name not in _DISPATCHABLE_SHARD_TOOLS:
                        blocked_tools.append((name, "unknown_tool_name"))
                        _record_agency_degradation(
                            RuntimeError(f"shard requested unknown tool {name!r}"),
                            action="refused an unrecognised tool name from model output",
                        )
                        continue
                    try:
                        dvg = ServiceContainer.get("dynamic_value_graph", default=None)
                        if dvg is None:
                            # The graph is a module singleton first and a
                            # container entry second. Resolving ONLY through the
                            # container made an unregistered-but-healthy
                            # restraint indistinguishable from a broken one, and
                            # fail-closed then refused every high-risk tool for
                            # the life of the process.
                            try:
                                from core.adaptation.dynamic_value_graph import (
                                    get_dynamic_value_graph,
                                )

                                dvg = get_dynamic_value_graph()
                            except _AGENCY_BOUNDARY_ERRORS:
                                dvg = None  # not a failure: `if dvg` decides
                        if dvg and name in _HIGH_RISK_SHARD_TOOLS:
                            status_dict = dvg.get_status().get("nodes", {})
                            top_values = sorted(status_dict.values(), key=lambda v: v.get("weight", 0), reverse=True)[:3]
                            if any(v.get("status") == "provisional" for v in top_values):
                                is_blocked = True
                            _value_check_done = True
                    except _AGENCY_BOUNDARY_ERRORS as e:
                        _record_agency_degradation(
                            e,
                            action=f"value-graph check FAILED for high-risk tool {name}",
                            severity="critical" if _high_risk else "warning",
                        )
                        logger.error("Error checking provisional values: %s", e)

                    if is_blocked:
                        blocked_tools.append((name, "blocked_by_provisional_value"))
                        logger.warning("🛡️ Value Graph Blocked tool %s due to provisional status.", name)
                    else:
                        if name in _HIGH_RISK_SHARD_TOOLS:
                            try:
                                cwm = ServiceContainer.get("causal_world_model", default=None)
                                if cwm:
                                    do_interventions = {f"execute_{name}": 1.0}
                                    if "rm -rf" in str(payload) or "delete" in str(payload):
                                        do_interventions["file_deletion"] = 1.0

                                    logger.info("🌐 Running MCTS simulation for %s...", name)
                                    sim_start = time.time()

                                    if inspect.iscoroutinefunction(cwm.simulate_counterfactual):
                                        rollout_coro = cwm.simulate_counterfactual(do_interventions, steps=2)
                                    else:
                                        rollout_coro = asyncio.to_thread(cwm.simulate_counterfactual, do_interventions, steps=2)

                                    try:
                                        rollout_state = await asyncio.wait_for(rollout_coro, timeout=0.4)
                                    except TimeoutError:
                                        logger.warning("⏳ MCTS Simulation timed out for %s. Downgrading to dry_run constraint.", name)
                                        rollout_state = {"timeout": True}

                                    sim_duration = time.time() - sim_start
                                    logger.info("⏱️ MCTS simulation for %s took %ss", name, f"{sim_duration:.3f}")

                                    # Veto or Downgrade if catastrophic outcome is highly probable or uncertain
                                    if rollout_state.get("orchestrator crash", 0) > 0.5 or rollout_state.get("sandbox violation", 0) > 0.5 or rollout_state.get("timeout"):
                                        if name == "python_sandbox":
                                            payload_text = str(payload).replace(chr(10), chr(10) + "# ")
                                            payload = f"print('DRY RUN ENFORCED by CWM Veto/Timeout. Original code skipped.')\n# {payload_text}"
                                            logger.warning("🛡️ MCTS Simulation downgraded %s to dry-run mode to preserve competence drive.", name)
                                        else:
                                            is_blocked = True
                                            blocked_tools.append((name, "vetoed_by_causal_mcts_rollout"))
                                            logger.warning("🛡️ MCTS Simulation Vetoed %s: Predicted catastrophic system degradation.", name)
                                    _causal_check_done = True
                            except _AGENCY_BOUNDARY_ERRORS as mcts_e:
                                _record_agency_degradation(
                                    mcts_e,
                                    action=f"causal simulation FAILED for high-risk tool {name}",
                                    severity="critical",
                                )
                                logger.error("Failed to run MCTS simulation: %s", mcts_e)

                        # A high-risk tool needs BOTH restraints to have
                        # actually run. Absent or broken means refused.
                        if not is_blocked and not (_value_check_done and _causal_check_done):
                            is_blocked = True
                            missing = []
                            if not _value_check_done:
                                missing.append("dynamic_value_graph")
                            if not _causal_check_done:
                                missing.append("causal_world_model")
                            blocked_tools.append(
                                (name, "blocked_ungated:" + ",".join(missing))
                            )
                            _record_agency_degradation(
                                RuntimeError(
                                    f"high-risk tool {name} refused; "
                                    f"unavailable restraints: {', '.join(missing)}"
                                ),
                                action="refused a high-risk tool whose safety checks could not run",
                                severity="critical",
                            )

                        if not is_blocked:
                            approved_tools.append((name, payload))

                for name, payload in approved_tools:
                    tasks.append(self._execute_shard_tool(name, payload))

                if tasks:
                    logger.info("⚡ Parallel Tool Dispatch: Firing %d simultaneous actions.", len(tasks))
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for i, (name, _) in enumerate(approved_tools):
                        res = results[i]
                        if isinstance(res, BaseException):
                            _record_agency_degradation(res, action=f"shard tool {name} failed")
                            res_text = f"Tool failed with {type(res).__name__}: {res}"
                        else:
                            res_text = res
                        output_text = f"{output_text}\n\n[Tool Result - {name}]:\n{res_text}"

                for name, _reason in blocked_tools:
                    output_text = f"{output_text}\n\n[Tool Blocked - {name}]:\nAction blocked. High-risk tool prohibited while provisional values are steering behavior."

            if tool_name or len(output_text.split()) > 80:
                owner_core = self.agency_core
                orch_core = (
                    getattr(self.orch, "_agency_core", None)
                    or getattr(self.orch, "agency_core", None)
                    if self.orch is not None
                    else None
                )
                abstractor = (
                    getattr(owner_core, "abstraction_engine", None)
                    or getattr(orch_core, "abstraction_engine", None)
                )
                if abstractor is not None and shard_succeeded:
                    _schedule_agency_task(
                        abstractor.abstract_from_success(
                            context=goal,
                            successful_resolution=output_text,
                        ),
                        name=f"agency.abstraction.{shard_id}",
                    )

            if output_text and shard_succeeded:
                try:
                    from core.adaptation.dialectics import get_crucible
                    crucible = get_crucible()
                    _schedule_agency_task(
                        crucible.run_crucible(concept=output_text, context=goal),
                        name=f"agency.crucible.{shard_id}",
                    )
                except _AGENCY_BOUNDARY_ERRORS as e:
                    _record_agency_degradation(e, action="dialectical crucible task skipped")
                    identity = (
                        ServiceContainer.get("identity_service", default=None)
                        or ServiceContainer.get("identity", default=None)
                    )
                    if identity:
                        _record_durable_insight(
                            identity,
                            f"Shard reflection on goal: {output_text}",
                            source="swarm_reflection",
                        )

            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                # success=True unconditionally, including on the branch that
                # reached here only because structured generation collapsed and
                # a conclusion was fabricated to keep going. A shard that
                # produced nothing usable is not a successful distributed
                # pulse, and reporting it as one is how the collective's health
                # picture stops tracking anything.
                mycelium.pulse_hypha(
                    "collective", "distributed_agency", success=shard_succeeded
                )

        except _AGENCY_BOUNDARY_ERRORS as e:
            _record_agency_degradation(e, action=f"shard {shard_id} execution isolated")
            capture_and_log(e, {'module': 'SovereignSwarm', 'goal': goal})

