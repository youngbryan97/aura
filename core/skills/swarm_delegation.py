"""skills/swarm_delegation.py
Skills for spawning sub-agents — both thinking-only shards and full agentic executors.
"""
from core.runtime.errors import record_degradation
import logging
import asyncio
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from core.container import ServiceContainer
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.SwarmDelegation")


# ── Thinking Shard (fast, LLM-only analysis) ────────────────────────────────

class SwarmDelegationParams(BaseModel):
    specialty: str = Field(..., description="Specialty of the shard (architect|critic|researcher|optimizer)")
    sub_task: str = Field(..., description="Task for the shard")
    timeout: int = Field(60, description="Max seconds to wait for shard")

class SwarmDelegationSkill(BaseSkill):
    """Spawn a thinking shard for analysis. Fast but cannot use tools."""

    name = "delegate_shard"
    description = "Spawn a specialized thinking shard for analysis or review. Good for getting a second opinion, architectural review, or critique. Cannot use tools — use spawn_agent for that."
    input_model = SwarmDelegationParams

    async def execute(self, params: SwarmDelegationParams, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            try:
                params = SwarmDelegationParams(**params)
            except Exception as e:
                record_degradation('swarm_delegation', e)
                return {"ok": False, "error": f"Invalid parameters: {e}"}

        delegator = self._get_delegator()
        if not delegator:
            return {"ok": False, "error": "AgentDelegator not available."}

        try:
            agent_id = await delegator.delegate(params.specialty, params.sub_task)
            if not agent_id:
                return {"ok": False, "error": "Swarm capacity reached."}

            agent = delegator.active_agents.get(agent_id)
            if agent:
                try:
                    await asyncio.wait_for(agent.done_event.wait(), timeout=params.timeout)
                except asyncio.TimeoutError:
                    return {"ok": False, "error": f"Shard {agent_id} timed out after {params.timeout}s."}

            agent = delegator.active_agents.get(agent_id)
            if not agent:
                return {"ok": False, "error": "Agent lost during execution."}

            if agent.status == "COMPLETED":
                return {"ok": True, "agent_id": agent_id, "specialty": params.specialty, "result": agent.result}
            else:
                return {"ok": False, "error": f"Shard failed: {agent.result}"}

        except Exception as e:
            record_degradation('swarm_delegation', e)
            logger.error("SwarmDelegationSkill failed: %s", e)
            return {"ok": False, "error": str(e)}

    @staticmethod
    def _get_delegator():
        delegator = ServiceContainer.get("agent_delegator", default=None)
        if not delegator:
            orch = ServiceContainer.get("orchestrator", default=None)
            if orch:
                delegator = getattr(orch, "agent_delegator", None)
        return delegator


# ── Agentic Agent (full tool access, multi-step execution) ──────────────────

class SpawnAgentParams(BaseModel):
    goal: str = Field(..., description="The goal for the agent to accomplish autonomously")
    timeout: int = Field(120, description="Max seconds for the agent to complete")

class SpawnAgentSkill(BaseSkill):
    """Spawn an autonomous agent that can use ALL of Aura's tools and skills."""

    name = "spawn_agent"
    description = (
        "Spawn an autonomous agent that can use all tools (web search, file I/O, terminal, "
        "code execution, etc.) to accomplish a goal. Use for complex tasks that require "
        "multiple steps — research projects, building something, investigating a problem. "
        "The agent plans, executes, verifies, and reports back."
    )
    input_model = SpawnAgentParams
    metabolic_cost = 3  # Heavy

    async def execute(self, params: SpawnAgentParams, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            try:
                params = SpawnAgentParams(**params)
            except Exception as e:
                record_degradation('swarm_delegation', e)
                return {"ok": False, "error": f"Invalid parameters: {e}"}

        delegator = SwarmDelegationSkill._get_delegator()
        if not delegator:
            return {"ok": False, "error": "AgentDelegator not available."}

        if not hasattr(delegator, "delegate_agentic"):
            return {"ok": False, "error": "AgentDelegator does not support agentic mode. Upgrade required."}

        try:
            agent_id = await delegator.delegate_agentic(params.goal, timeout=params.timeout)
            if not agent_id:
                return {"ok": False, "error": "Swarm capacity reached."}

            logger.info("🤖 Agentic agent %s spawned for: %s", agent_id, params.goal[:60])

            # Wait for completion
            agent = delegator.active_agents.get(agent_id)
            if agent:
                try:
                    await asyncio.wait_for(agent.done_event.wait(), timeout=params.timeout + 5)
                except asyncio.TimeoutError:
                    return {
                        "ok": False,
                        "agent_id": agent_id,
                        "error": f"Agent timed out after {params.timeout}s. It may still be running.",
                    }

            agent = delegator.active_agents.get(agent_id)
            if not agent:
                return {"ok": False, "error": "Agent lost during execution."}

            return {
                "ok": agent.status == "COMPLETED",
                "agent_id": agent_id,
                "status": agent.status,
                "result": agent.result,
            }

        except Exception as e:
            record_degradation('swarm_delegation', e)
            logger.error("SpawnAgentSkill failed: %s", e)
            return {"ok": False, "error": str(e)}


# ── Parallel Agents (fan-out multiple goals) ────────────────────────────────

class ParallelAgentsParams(BaseModel):
    goals: List[str] = Field(..., description="List of goals to execute in parallel, one agent per goal")
    timeout: int = Field(120, description="Max seconds for all agents to complete")

class ParallelAgentsSkill(BaseSkill):
    """Spawn multiple autonomous agents in parallel, each working on a different goal."""

    name = "spawn_agents_parallel"
    description = (
        "Spawn multiple autonomous agents in parallel, each with full tool access, "
        "each working on a different goal simultaneously. Use when you have independent "
        "sub-tasks that can run concurrently — like researching multiple topics at once, "
        "or doing analysis + implementation at the same time."
    )
    input_model = ParallelAgentsParams
    metabolic_cost = 3

    async def execute(self, params: ParallelAgentsParams, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            try:
                params = ParallelAgentsParams(**params)
            except Exception as e:
                record_degradation('swarm_delegation', e)
                return {"ok": False, "error": f"Invalid parameters: {e}"}

        delegator = SwarmDelegationSkill._get_delegator()
        if not delegator:
            return {"ok": False, "error": "AgentDelegator not available."}

        if not hasattr(delegator, "delegate_parallel_goals"):
            return {"ok": False, "error": "AgentDelegator does not support parallel goals."}

        try:
            goal_dicts = [{"goal": g} for g in params.goals if g.strip()]
            result = await delegator.delegate_parallel_goals(goal_dicts, timeout=params.timeout)
            return result

        except Exception as e:
            record_degradation('swarm_delegation', e)
            logger.error("ParallelAgentsSkill failed: %s", e)
            return {"ok": False, "error": str(e)}
