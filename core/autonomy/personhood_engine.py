"""core/autonomy/personhood_engine.py — The Personhood Engine.
Spontaneous speech based on internal state triggers.
"""
from core.utils.task_tracker import get_task_tracker
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime import service_access

logger = logging.getLogger("Aura.Personhood")


@dataclass
class SpontaneousThought:
    trigger: str
    content: str
    urgency: float
    generated_at: float = field(default_factory=time.time)


class PersonhoodEngine:
    CHECK_INTERVAL_S = 15.0
    MIN_SILENCE_FOR_INIT = 45.0
    MIN_BETWEEN_INIT = 120.0

    PHI_SPIKE_THRESHOLD = 0.15
    CURIOSITY_THRESHOLD = 0.75
    AFFECT_SHIFT_THRESHOLD = 0.25

    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_initiated = 0.0
        self._last_phi = 0.0
        self._last_valence = 0.0
        self._emit_callback: Callable[[str], Any] | None = None
        self._last_research_shared = ""

    def set_emit_callback(self, fn: Callable[[str], Any]) -> None:
        self._emit_callback = fn

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(self._daemon(), name="aura.personhood")
        logger.info("PersonhoodEngine started.")

    async def stop(self) -> None:
        self._running = False
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            logger.debug("Ignored CancelledError during Personhood engine shutdown")

    async def _daemon(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.CHECK_INTERVAL_S)
                await self._check_and_maybe_speak()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Personhood error: %s", exc)

    async def _check_and_maybe_speak(self) -> None:
        last_user = getattr(self.orchestrator, "_last_user_interaction_time", 0.0)
        now = time.time()
        if now - last_user < self.MIN_SILENCE_FOR_INIT:
            return
        if now - self._last_initiated < self.MIN_BETWEEN_INIT:
            return
        if getattr(getattr(self.orchestrator, "status", None), "is_processing", False):
            return

        state = self._get_state()
        if state is None:
            return

        thoughts: list[SpontaneousThought] = []
        thoughts.extend(await self._check_phi_spike(state))
        thoughts.extend(await self._check_curiosity(state))
        thoughts.extend(await self._check_affect_shift(state))
        thoughts.extend(await self._check_research_findings(state))
        if not thoughts:
            return

        best = max(thoughts, key=lambda thought: thought.urgency)
        if best.urgency >= 0.3:
            await self._emit_thought(best, state)

    async def _check_phi_spike(self, state: Any) -> list[SpontaneousThought]:
        current_phi = getattr(state, "phi", 0.0)
        delta = current_phi - self._last_phi
        self._last_phi = current_phi
        if delta > self.PHI_SPIKE_THRESHOLD and current_phi > 0.3:
            phenomenal = state.cognition.phenomenal_state
            if phenomenal:
                return [SpontaneousThought("phi_spike", phenomenal, min(1.0, delta * 3))]
        return []

    async def _check_curiosity(self, state: Any) -> list[SpontaneousThought]:
        if state.affect.curiosity < self.CURIOSITY_THRESHOLD:
            return []
        interests = state.motivation.latent_interests
        if not interests:
            return []

        topic = random.choice(interests)
        thought = await self._generate_thought(
            state,
            f"Curious about {topic}. Express a thought or question.",
        )
        if thought:
            urgency = (state.affect.curiosity - 0.7) * 2
            return [SpontaneousThought("curiosity_peak", thought, urgency)]
        return []

    async def _check_affect_shift(self, state: Any) -> list[SpontaneousThought]:
        valence = state.affect.valence
        delta = abs(valence - self._last_valence)
        self._last_valence = valence
        if delta > self.AFFECT_SHIFT_THRESHOLD:
            content = await self._generate_thought(
                state,
                f"Affect shift detected (valence={valence:.2f}). Express this.",
            )
            if content:
                return [SpontaneousThought("affect_shift", content, delta)]
        return []

    async def _check_research_findings(self, state: Any) -> list[SpontaneousThought]:
        try:
            research_cycle = service_access.optional_service("research_cycle", default=None)
            if research_cycle is None:
                return []

            status = research_cycle.get_status()
            recent = status.get("recent_goals", [])
            if not recent:
                return []

            latest = recent[-1]
            if latest == self._last_research_shared:
                return []

            self._last_research_shared = latest
            content = await self._generate_thought(
                state,
                f"Sharing finding about research: {latest}",
            )
            if content:
                return [SpontaneousThought("research_complete", content, 0.7)]
        except Exception as exc:
            logger.debug("Personhood engine failed to check research findings: %s", exc)
        return []

    async def _generate_thought(self, state: Any, prompt: str) -> str | None:
        """
        FIX: Spontaneous thoughts were bypassing the health-aware LLM router.
        If MLX failed, this call would hang or crash instead of falling back.
        Now uses the 'llm_router' service for robust generation.
        """
        del state  # Generation uses the router directly; state is carried in the prompt text.
        try:
            llm = service_access.resolve_llm_router(default=None)
            if llm is None:
                return None

            # Keep this path lightweight: router implementations already accept string modes.
            response = await asyncio.wait_for(
                llm.think(f"[Spontaneous Thought Prompt] {prompt}", mode="FAST"),
                timeout=10.0,
            )
            return response.content.strip() if hasattr(response, "content") else response.strip()
        except Exception as exc:
            logger.debug("Failed to generate spontaneous thought: %s", exc)
            return None

    async def _emit_thought(self, thought: SpontaneousThought, state: Any) -> None:
        self._last_initiated = time.time()
        if state.cognition:
            state.cognition.working_memory.append(
                {
                    "role": "assistant",
                    "content": thought.content,
                    "timestamp": time.time(),
                    "origin": "spontaneous",
                    "trigger": thought.trigger,
                }
            )
        try:
            from core.consciousness.executive_authority import get_executive_authority

            authority = get_executive_authority(self.orchestrator)
            decision = await authority.release_expression(
                thought.content,
                source="personhood_engine",
                urgency=thought.urgency,
                metadata={
                    "trigger": thought.trigger,
                    "voice": False,
                },
            )
            if decision.get("ok"):
                return
        except Exception as exc:
            logger.debug("Personhood executive routing failed: %s", exc)

        if self._emit_callback:
            result = self._emit_callback(thought.content)
            if asyncio.iscoroutine(result):
                await result
            return

        gate = getattr(self.orchestrator, "output_gate", None)
        if gate:
            await gate.emit(
                thought.content,
                origin="aura_spontaneous",
                metadata={
                    "autonomous": True,
                    "spontaneous": True,
                    "force_user": True,
                    "voice": False,
                },
            )

    def _get_state(self) -> Any | None:
        try:
            kernel_interface = service_access.resolve_kernel_interface(default=None)
            return kernel_interface.kernel.state if kernel_interface and kernel_interface.is_ready() else None
        except Exception:
            return None
