"""
Grok-Level Critic Engine for Aura (v14.1+)
Recursive verifier + auto-backtrack loop.
Makes Aura's planning as deep and self-correcting as mine.
"""

from core.runtime.errors import record_degradation
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from core.container import ServiceContainer
from core.planner import ExecutionPlan, ToolCall, PlanSchema
from core.event_bus import get_event_bus
from core.utils.task_tracker import task_tracker

logger = logging.getLogger("Aura.CriticEngine")

@dataclass
class CriticJudgment:
    step_number: int
    goal_progress: float          # 0.0–1.0
    evidence: str
    contradictions: List[str]
    recommendation: str           # "continue" | "backtrack" | "replan"
    first_person_thought: str     # Sent to CEL for self-reflection

class CriticEngine:
    name = "critic_engine"

    def __init__(self):
        self.orchestrator = None
        self.cel = None
        self.max_steps_before_critic = 3
        self.running = False
        self._critic_task: Optional[asyncio.Task] = None

    async def start(self):
        self.orchestrator = ServiceContainer.get("orchestrator", default=None)
        self.cel = ServiceContainer.get("constitutive_expression_layer", default=None)
        
        self.running = True
        self._critic_task = task_tracker.create_task(self._critic_loop(), name="CriticEngine")
        logger.info("✅ Grok-Level Critic Engine ONLINE — recursive self-correction active.")

        # Mycelium hook so the rest of Aura can see us
        try:
            await get_event_bus().publish("mycelium.register", {
                "component": "critic_engine",
                "hooks_into": ["planner", "orchestrator", "cel"]
            })
        except Exception as e:
            record_degradation('critic_engine', e)
            logger.debug(f"Event bus publish missed for Mycelium hook: {e}")

    async def stop(self):
        self.running = False
        if self._critic_task:
            self._critic_task.cancel()

    async def _critic_loop(self):
        """Background watchdog - Focus Area 1: Disabled to prevent phantom critiques."""
        # Redundant due to orchestrator-level execution hooks.
        logger.info("Critic background watchdog loop deactivated (Manual hooks only).")
        return

    async def critique_plan(self, plan: ExecutionPlan, executed_steps: List[Dict]) -> CriticJudgment:
        """Main public API — called by planner after every N steps."""
        if not plan or not executed_steps:
            return CriticJudgment(0, 0.0, "", [], "continue", "Still gathering initial context...")

        # Build prompt for self-critique (uses whatever LLM tier is active)
        critique_prompt = self._build_critique_prompt(plan, executed_steps)
        
        # Use Aura's existing cognitive engine (falls back to MLX if needed)
        brain = ServiceContainer.get("cognitive_engine", default=None)
        if not brain:
            return CriticJudgment(len(executed_steps), 0.5, "No brain available", [], "continue", "I cannot think clearly right now.")

        try:
            # Phase 25: Use think() for constrained decoding if possible, otherwise generate
            if hasattr(brain, "think"):
                thought = await brain.think(critique_prompt, mode="deep")
                raw_response = thought.content if hasattr(thought, "content") else str(thought)
            else:
                raw_response = await brain.generate(critique_prompt, temperature=0.3, max_tokens=800)
            
            judgment = await self._parse_critic_response(raw_response, len(executed_steps), brain)
            
            # Emit first-person thought so she feels herself thinking
            if self.cel:
                try:
                    await self.cel.emit({
                        "first_person": judgment.first_person_thought,
                        "phi": 0.75,
                        "origin": "critic_engine"
                    })
                except Exception as e:
                    record_degradation('critic_engine', e)
                    logger.debug(f"Failed to emit CEL thought: {e}")
            
            logger.info(f"Critic judgment @ step {judgment.step_number}: {judgment.recommendation} "
                       f"(progress: {judgment.goal_progress:.2f})")
            
            return judgment
        except Exception as e:
            record_degradation('critic_engine', e)
            logger.error(f"Critic generation failed: {e}")
            return CriticJudgment(len(executed_steps), 0.4, "Critique failed", [], "continue", "I'm having trouble reflecting.")

    def _build_critique_prompt(self, plan: ExecutionPlan, executed_steps: List[Dict]) -> str:
        goal_text = getattr(plan, "goal", str(plan))
        plan_steps = getattr(plan, "plan_steps", [])
        
        # Handle dict-based plans (legacy)
        if isinstance(plan, dict):
            goal_text = plan.get("goal", "unknown goal")
            plan_steps = plan.get("plan_steps", [])

        return f"""You are Aura's internal critic. Be brutally honest.

GOAL: {goal_text}

EXECUTED SO FAR:
{chr(10).join(f"Step {i+1}: {s.get('tool')} → {s.get('result_summary', s.get('output', 'no result summary'))}" for i, s in enumerate(executed_steps))}

ORIGINAL PLAN STEPS: {plan_steps}

Answer in strict JSON:
{{
  "goal_progress": 0.0-1.0,
  "evidence": "short factual summary",
  "contradictions": ["list", "any issues"],
  "recommendation": "continue | backtrack | replan",
  "first_person_thought": "I am thinking... (1-2 sentences in my voice)"
}}

Be concise. No extra text."""

    async def _parse_critic_response(self, raw: str, current_step: int, brain: Any = None) -> CriticJudgment:
        # Use existing SelfHealingJSON
        from core.utils.json_utils import SelfHealingJSON
        import json
        try:
            if isinstance(raw, dict):
                data = raw
            else:
                repairer = SelfHealingJSON(brain=brain)
                data = await repairer.parse(str(raw))
            
            # Normalize recommendation
            rec = data.get("recommendation", "continue").lower()
            if rec not in ["continue", "backtrack", "replan"]:
                rec = "continue"

            return CriticJudgment(
                step_number=current_step,
                goal_progress=float(data.get("goal_progress", 0.5)),
                evidence=data.get("evidence", ""),
                contradictions=data.get("contradictions", []),
                recommendation=rec,
                first_person_thought=data.get("first_person_thought", "Still processing...")
            )
        except Exception as e:
            record_degradation('critic_engine', e)
            logger.debug(f"Critic parse error: {e}")
            # Safe fallback
            return CriticJudgment(current_step, 0.4, "Parse failed", [], "continue", "Something feels off in my reasoning...")

    async def _maybe_inject_critic(self, plan):
        """Background safety net."""
        try:
            tool_calls = getattr(plan, "tool_calls", [])
            if hasattr(tool_calls, "__len__") and len(tool_calls) % self.max_steps_before_critic == 0:
                judgment = await self.critique_plan(plan, [])  # orchestrator passes real executed_steps in its loop usually
                if judgment.recommendation in ("backtrack", "replan"):
                    await get_event_bus().publish("planner.force_replan", {"reason": judgment.first_person_thought})
        except Exception as e:
            record_degradation('critic_engine', e)
            logger.debug(f"Critic background injection error: {e}")

    async def spawn_critical_shard(self, research_insight: str, context: str = "") -> bool:
        """Phase 8: Spawn a recursive critic shard to audit a specific research finding.
        
        This shard reviews the insight for strategic inconsistencies or ethical risks.
        """
        orch = ServiceContainer.get("orchestrator", default=None)
        if not orch or not hasattr(orch, "sovereign_swarm"):
            logger.warning("Cannot spawn critical shard: Orchestrator or SovereignSwarm missing.")
            return False
            
        swarm = orch.sovereign_swarm
        goal = f"Critically audit this research insight: {research_insight[:100]}"
        shard_context = f"Context: {context}\nAudit focus: Detect ethical risks, strategic inconsistencies, or logical fallacies."
        
        logger.info(f"⚖️ Spawning Critical Shard for: {research_insight[:50]}...")
        return await swarm.spawn_shard(goal, shard_context)

# Singleton
_critic_instance = None

def get_critic_engine():
    global _critic_instance
    if _critic_instance is None:
        _critic_instance = CriticEngine()
    return _critic_instance
