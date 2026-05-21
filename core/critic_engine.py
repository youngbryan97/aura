"""
Grok-Level Critic Engine for Aura (v14.1+)
Recursive verifier + auto-backtrack loop.
Makes Aura's planning as deep and self-correcting as mine.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from core.container import ServiceContainer
from core.event_bus import get_event_bus
from core.planner import ExecutionPlan
from core.runtime.errors import FallbackClassification, Severity, record_degradation

logger = logging.getLogger("Aura.CriticEngine")

MAX_TEXT_CHARS = 2000

_CRITIC_ERRORS = (
    AttributeError,
    ConnectionError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_critic_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "critic_engine",
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
                "critic_engine",
                error,
                severity=severity,
                action=action or "critic engine degraded",
            )
        except TypeError:
            logger.warning(
                "CriticEngine degradation could not be recorded: %s",
                signature_exc,
            )


def _safe_text(value: object, *, default: str = "", max_chars: int = MAX_TEXT_CHARS) -> str:
    try:
        text = str(value if value is not None else default)
    except (RuntimeError, TypeError, ValueError):
        text = default
    return text.replace("\x00", "")[:max_chars]


def _safe_progress(value: object, default: float = 0.5) -> float:
    try:
        progress = float(value)
    except (TypeError, ValueError, OverflowError):
        progress = default
    return max(0.0, min(1.0, progress))


def _safe_contradictions(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_safe_text(item, max_chars=500) for item in value[:20]]


def _format_executed_step(index: int, step: object) -> str:
    if isinstance(step, dict):
        tool = _safe_text(step.get("tool"), default="unknown")
        summary = _safe_text(
            step.get("result_summary", step.get("output", "no result summary"))
        )
    else:
        tool = "unknown"
        summary = _safe_text(step, default="no result summary")
    return f"Step {index + 1}: {tool} → {summary}"


@dataclass
class CriticJudgment:
    step_number: int
    goal_progress: float          # 0.0–1.0
    evidence: str
    contradictions: list[str]
    recommendation: str           # "continue" | "backtrack" | "replan"
    first_person_thought: str     # Sent to CEL for self-reflection

class CriticEngine:
    name = "critic_engine"

    def __init__(self):
        self.orchestrator = None
        self.cel = None
        self.max_steps_before_critic = 3
        self.running = False
        self._critic_task: asyncio.Task | None = None

    async def start(self):
        self.orchestrator = ServiceContainer.get("orchestrator", default=None)
        self.cel = ServiceContainer.get("constitutive_expression_layer", default=None)
        self.running = True
        self._critic_task = None
        logger.info("✅ Critic Engine ONLINE — on-demand recursive self-correction active.")

        # Mycelium hook so the rest of Aura can see us
        try:
            await get_event_bus().publish("mycelium.register", {
                "component": "critic_engine",
                "hooks_into": ["planner", "orchestrator", "cel"]
            })
        except _CRITIC_ERRORS as e:
            _record_critic_degradation(
                e,
                action="started critic engine while mycelium registration event failed",
                severity="warning",
            )
            logger.debug("Event bus publish missed for Mycelium hook: %s", e)

    async def stop(self):
        self.running = False
        if self._critic_task:
            self._critic_task.cancel()

    async def _critic_loop(self):
        """Background watchdog - Focus Area 1: Disabled to prevent phantom critiques."""
        # Redundant due to orchestrator-level execution hooks.
        logger.info("Critic background watchdog loop deactivated (Manual hooks only).")
        return

    async def critique_plan(self, plan: ExecutionPlan, executed_steps: list[dict[str, Any]]) -> CriticJudgment:
        """Main public API — called by planner after every N steps."""
        if not plan or not executed_steps:
            return CriticJudgment(0, 0.0, "", [], "continue", "Still gathering initial context...")

        try:
            # Build prompt for self-critique (uses whatever LLM tier is active)
            critique_prompt = self._build_critique_prompt(plan, executed_steps)

            # Use Aura's existing cognitive engine (falls back to MLX if needed)
            brain = ServiceContainer.get("cognitive_engine", default=None)
            if not brain:
                return CriticJudgment(len(executed_steps), 0.5, "No brain available", [], "continue", "I cannot think clearly right now.")

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
                except _CRITIC_ERRORS as e:
                    _record_critic_degradation(
                        e,
                        action="returned critic judgment while CEL reflection emission failed",
                        severity="warning",
                    )
                    logger.debug("Failed to emit CEL thought: %s", e)
            
            logger.info(f"Critic judgment @ step {judgment.step_number}: {judgment.recommendation} "
                       f"(progress: {judgment.goal_progress:.2f})")
            
            return judgment
        except _CRITIC_ERRORS as e:
            _record_critic_degradation(
                e,
                action="returned conservative continue judgment after critic generation failed",
                severity="degraded",
                extra={"executed_steps": len(executed_steps)},
            )
            logger.error("Critic generation failed: %s", e)
            return CriticJudgment(len(executed_steps), 0.4, "Critique failed", [], "continue", "I'm having trouble reflecting.")

    def _build_critique_prompt(self, plan: ExecutionPlan, executed_steps: list[dict[str, Any]]) -> str:
        goal_text = _safe_text(getattr(plan, "goal", str(plan)))
        plan_steps = getattr(plan, "plan_steps", [])
        
        # Handle dict-based plans (legacy)
        if isinstance(plan, dict):
            goal_text = _safe_text(plan.get("goal", "unknown goal"))
            plan_steps = plan.get("plan_steps", [])

        return f"""You are Aura's internal critic. Be brutally honest.

GOAL: {goal_text}

EXECUTED SO FAR:
{chr(10).join(_format_executed_step(i, s) for i, s in enumerate(executed_steps[:50]))}

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
        try:
            from core.utils.json_utils import SelfHealingJSON

            if isinstance(raw, dict):
                data = raw
            else:
                repairer = SelfHealingJSON(brain=brain)
                data = await repairer.parse(str(raw))
            if not isinstance(data, dict):
                raise ValueError("critic response did not parse to a JSON object")
            
            # Normalize recommendation
            rec = _safe_text(data.get("recommendation", "continue")).lower()
            if rec not in ["continue", "backtrack", "replan"]:
                rec = "continue"

            return CriticJudgment(
                step_number=current_step,
                goal_progress=_safe_progress(data.get("goal_progress", 0.5)),
                evidence=_safe_text(data.get("evidence", "")),
                contradictions=_safe_contradictions(data.get("contradictions", [])),
                recommendation=rec,
                first_person_thought=_safe_text(data.get("first_person_thought", "Still processing..."))
            )
        except _CRITIC_ERRORS as e:
            _record_critic_degradation(
                e,
                action="returned conservative continue judgment after critic JSON parse failed",
                severity="warning",
                extra={"current_step": current_step},
            )
            logger.debug("Critic parse error: %s", e)
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
        except _CRITIC_ERRORS as e:
            _record_critic_degradation(
                e,
                action="skipped background critic injection after planner/event bus failure",
                severity="warning",
            )
            logger.debug("Critic background injection error: %s", e)

    async def spawn_critical_shard(self, research_insight: str, context: str = "") -> bool:
        """Phase 8: Spawn a recursive critic shard to audit a specific research finding.
        
        This shard reviews the insight for strategic inconsistencies or ethical risks.
        """
        orch = ServiceContainer.get("orchestrator", default=None)
        swarm = getattr(orch, "sovereign_swarm", None) if orch else None
        if not swarm:
            logger.warning("Cannot spawn critical shard: Orchestrator or SovereignSwarm missing.")
            return False

        goal = f"Critically audit this research insight: {_safe_text(research_insight, max_chars=100)}"
        shard_context = (
            f"Context: {_safe_text(context)}\n"
            "Audit focus: Detect ethical risks, strategic inconsistencies, or logical fallacies."
        )
        
        logger.info(
            "⚖️ Spawning Critical Shard for: %s...",
            _safe_text(research_insight, max_chars=50),
        )
        try:
            return bool(await swarm.spawn_shard(goal, shard_context))
        except _CRITIC_ERRORS as e:
            _record_critic_degradation(
                e,
                action="returned false after critical shard spawn failed",
                severity="warning",
            )
            logger.debug("Critical shard spawn failed: %s", e)
            return False

# Singleton
_critic_instance: CriticEngine | None = None

def get_critic_engine():
    global _critic_instance
    if _critic_instance is None:
        _critic_instance = CriticEngine()
    return _critic_instance
