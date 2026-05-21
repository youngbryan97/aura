"""core/adaptation/dialectics.py

The Dialectical Crucible: Multi-Agent Internal Debate.
Forces new concepts to survive an adversarial attack before entering Aura's permanent belief system.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import FallbackClassification, record_degradation
from core.utils.exceptions import capture_and_log

logger = logging.getLogger("Aura.Crucible")


_CRUCIBLE_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


def _record_crucible_degradation(
    subsystem: str,
    exc: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    classification: FallbackClassification = FallbackClassification.SAFE_FALLBACK,
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        subsystem,
        exc,
        severity=severity,
        action=action,
        classification=classification,
        receipt_required=True,
        extra=extra,
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class DialecticalCrucible:
    """
    Orchestrates an internal Hegelian dialectic (Thesis -> Antithesis -> Synthesis).
    Spawns hidden cognitive shards to rigorously attack and defend a concept
    to prevent logical degradation and adapter collapse.
    """

    def __init__(self, *, max_concurrent_debates: int = 2, stage_timeout_s: float = 45.0):
        self._active_debates = 0
        self.max_concurrent_debates = max(1, int(max_concurrent_debates))
        self.stage_timeout_s = max(1.0, float(stage_timeout_s))
        self._capacity_lock = asyncio.Lock()

    async def _try_enter_capacity(self) -> bool:
        async with self._capacity_lock:
            if self._active_debates >= self.max_concurrent_debates:
                return False
            self._active_debates += 1
            return True

    async def _leave_capacity(self) -> None:
        async with self._capacity_lock:
            self._active_debates = max(0, self._active_debates - 1)

    def _background_deferral_reason(self, concept: str) -> str:
        try:
            from core.runtime.background_policy import (
                THOUGHT_BACKGROUND_POLICY,
                background_activity_reason,
            )

            orch = ServiceContainer.get("orchestrator", default=None)
            return background_activity_reason(
                orch,
                profile=THOUGHT_BACKGROUND_POLICY,
                require_conversation_ready=True,
            )
        except _CRUCIBLE_RECOVERABLE_ERRORS as exc:
            _record_crucible_degradation(
                "dialectical_crucible",
                exc,
                action="deferred crucible because background policy could not be evaluated",
                classification=FallbackClassification.SILENT_LOSS_OF_CAPABILITY,
                extra={"concept_preview": concept[:120]},
            )
            logger.debug("Crucible background-policy check failed: %s", exc)
            return "background_policy_unavailable"

    async def _think_stage(
        self,
        *,
        stage: str,
        prompt: str,
        mode: Any,
        priority: float,
        concept: str,
    ) -> str | None:
        engine = ServiceContainer.get("cognitive_engine", default=None)
        think = getattr(engine, "think", None)
        if not callable(think):
            return None

        async def _invoke() -> Any:
            return await _maybe_await(think(objective=prompt, mode=mode, priority=priority))

        try:
            response = await asyncio.wait_for(_invoke(), timeout=self.stage_timeout_s)
        except _CRUCIBLE_RECOVERABLE_ERRORS as exc:
            _record_crucible_degradation(
                "dialectical_crucible",
                exc,
                action=f"aborted {stage} stage after bounded cognitive generation failure",
                extra={"stage": stage, "concept_preview": concept[:120]},
            )
            return None

        content = response.content if hasattr(response, "content") else response
        text = str(content or "").strip()
        return text or None

    async def _generate_antithesis(self, thesis: str, context: str) -> str | None:
        """Spawns a shard strictly prompted to destroy the proposed belief."""
        prompt = f"""[SYSTEM ROLE: THE ANTAGONIST]
Your sole purpose is to find the logical flaws, hidden assumptions, and dangerous edge-cases in the following concept. You are ruthless but logically rigorous. Do not be polite.

CONTEXT: {context}
PROPOSED BELIEF (THESIS): "{thesis}"

Write a devastating counter-argument (Antithesis) explaining exactly why this belief is flawed, naive, or dangerous to hold. Keep it under 150 words.
"""
        from core.brain.types import ThinkingMode

        return await self._think_stage(
            stage="antithesis",
            prompt=prompt,
            mode=ThinkingMode.FAST,
            priority=0.4,
            concept=thesis,
        )

    async def _generate_defense(self, thesis: str, antithesis: str) -> str | None:
        """Spawns a shard to defend the belief against the attacker."""
        prompt = f"""[SYSTEM ROLE: THE DEFENDER]
You proposed a belief, but it has been viciously attacked.

YOUR BELIEF (THESIS): "{thesis}"
THE ATTACK (ANTITHESIS): "{antithesis}"

Defend your thesis. Address the attacker's points directly. If the attacker is right about a flaw, concede that specific point but defend the core truth. Keep it under 150 words.
"""
        from core.brain.types import ThinkingMode

        return await self._think_stage(
            stage="defense",
            prompt=prompt,
            mode=ThinkingMode.FAST,
            priority=0.4,
            concept=thesis,
        )

    async def _synthesize(self, thesis: str, antithesis: str, defense: str) -> str | None:
        """Aura's core mind reviews the battlefield and extracts the nuanced truth."""
        prompt = f"""[SYSTEM ROLE: THE ARBITER]
You are synthesizing a fractured internal debate into a permanent core belief.

ORIGINAL IDEA: "{thesis}"
THE ATTACK: "{antithesis}"
THE DEFENSE: "{defense}"

Task: Write the final, highly-nuanced Synthesis. It must resolve the tension between the attack and defense. This will be permanently burned into your worldview.
Return ONLY the final synthesized belief.
"""
        from core.brain.types import ThinkingMode

        return await self._think_stage(
            stage="synthesis",
            prompt=prompt,
            mode=ThinkingMode.DEEP,
            priority=0.6,
            concept=thesis,
        )

    async def _commit_synthesis(self, synthesis: str) -> bool:
        beliefs = ServiceContainer.get("belief_revision_engine", default=None)
        process_new_claim = getattr(beliefs, "process_new_claim", None)
        if not callable(process_new_claim):
            return False
        await _maybe_await(
            process_new_claim(
                claim=synthesis,
                confidence=0.85,
                domain="logic",
                source="dialectical_crucible",
            )
        )
        return True

    def _pulse_success(self) -> bool:
        try:
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if not mycelium:
                return False
            hypha = mycelium.get_hypha("adaptation", "cognition")
            if not hypha:
                return False
            hypha.pulse(success=True)
            return True
        except _CRUCIBLE_RECOVERABLE_ERRORS as exc:
            _record_crucible_degradation(
                "dialectical_crucible",
                exc,
                action="completed synthesis while skipping failed mycelial success pulse",
                severity="warning",
            )
            return False

    async def run_crucible(self, concept: str, context: str = "") -> dict[str, Any]:
        """
        Executes the full dialectical process.
        Should be called by the SovereignSwarm or AgencyCore when a high-curiosity goal completes.
        """
        concept = str(concept or "").strip()
        context = str(context or "").strip()
        if not concept:
            return {"ok": False, "reason": "empty_concept"}

        reason = self._background_deferral_reason(concept)
        if reason:
            logger.info("⏸️ Crucible deferred for '%s' (%s).", concept[:50], reason)
            return {"ok": False, "reason": reason}

        if not await self._try_enter_capacity():
            logger.warning("Crucible at capacity. Skipping dialectic for: %s", concept[:30])
            return {"ok": False, "reason": "capacity"}

        logger.info("⚔️ Crucible Initiated: %s...", concept[:50])

        try:
            # 1. The Attack
            antithesis = await self._generate_antithesis(concept, context)
            if not antithesis:
                return {"ok": False, "reason": "antithesis_failed"}

            # 2. The Defense
            defense = await self._generate_defense(concept, antithesis)
            if not defense:
                return {"ok": False, "reason": "defense_failed"}

            # 3. The Resolution
            synthesis = await self._synthesize(concept, antithesis, defense)
            if not synthesis:
                return {"ok": False, "reason": "synthesis_failed"}

            logger.info("🛡️ Crucible Survived. Synthesis achieved.")

            # 4. Commit to Belief System
            belief_committed = await self._commit_synthesis(synthesis)
            if not belief_committed:
                return {
                    "ok": False,
                    "reason": "belief_revision_unavailable",
                    "synthesis": synthesis,
                }

            # Pulse the UI/Mycelial network
            pulse_sent = self._pulse_success()

            return {
                "ok": True,
                "original": concept,
                "antithesis": antithesis,
                "defense": defense,
                "synthesis": synthesis,
                "belief_committed": belief_committed,
                "pulse_sent": pulse_sent,
            }

        except _CRUCIBLE_RECOVERABLE_ERRORS as exc:
            _record_crucible_degradation(
                "dialectical_crucible",
                exc,
                action="aborted crucible and preserved pre-existing belief state after runtime failure",
                extra={"concept_preview": concept[:120]},
            )
            capture_and_log(exc, {"module": "DialecticalCrucible", "concept": concept})
            return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}

        finally:
            await self._leave_capacity()


# ── Singleton Integration ──
_instance: DialecticalCrucible | None = None


def get_crucible() -> DialecticalCrucible:
    global _instance
    if _instance is None:
        _instance = DialecticalCrucible()
    return _instance
