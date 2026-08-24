"""core/brain/deep_deliberation.py

Deep Deliberation  (lineage: Deep Thought — The Hitchhiker's Guide to the Galaxy)
================================================================================
"42" is the joke that lands a real lesson: the answer was useless because no one
had worked out the actual QUESTION. So for problems flagged as hard, this engine
refines the question first, then spends an extended reasoning budget on the
refined version. The refinement step is the value — most systems answer the
literal question; this one fixes the question before answering. It lives in
brain/ beside deliberation.py and reasoning_amplifier.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.service_registry import get_runtime_service, register_runtime_service
from core.utils.engine_support import coerce_text, record_engine_degradation, resolve_brain

logger = logging.getLogger("Aura.DeepDeliberation")


def _degrade(exc: BaseException, *, action: str, severity: str = "warning") -> None:
    record_engine_degradation("deep_deliberation", exc, action=action, severity=severity)


@dataclass
class DeliberationResult:
    original_question: str
    refined_question: str
    answer: str
    passes: int
    used_model: bool
    # True when the answer came from a Recursive Latent Cortex episode on the
    # resident model (workspace recurrence), not ordinary token generation.
    used_latent_cortex: bool = False
    timestamp: float = field(default_factory=time.time)


# A deliberation that fell back to the sharpened question is not a failure
# of correctness, but a run of them means no model is reachable.
_UNHEALTHY_FAILURE_STREAK = 3
# Below this share of model-backed deliberations the engine is degraded: it
# is still answering, but not by deliberating.
_MIN_MODEL_BACKED_RATE = 0.25



def _latent_episode_seconds(objective: str, *, floor_s: float) -> float:
    """How long a complete answer to this actually needs.

    Measured where there are measurements: `measured_admission` keeps p90
    prefill, decode and overhead per task shape from completed generations,
    and falls back to its own static prior for a shape it has not seen. Never
    returns less than the caller's existing allowance, so this can only give a
    hard question more room, never take room from an easy one.
    """
    from core.runtime.response_policy import USER_FACING_COMPLETION_DEADLINE_MAX_S

    try:
        from core.brain.llm.measured_admission import recommended_foreground_deadline
        from core.brain.llm.model_registry import runtime_model_measurement_key
        from core.runtime.structured_input import answer_surface_planning_tokens

        needed, _confidence, _samples = recommended_foreground_deadline(
            model=runtime_model_measurement_key(),
            prompt_tokens=max(2048, 1800 + len(str(objective or "")) // 4),
            decode_tokens=max(1, answer_surface_planning_tokens(str(objective or ""))),
            minimum_seconds=float(floor_s),
            maximum_seconds=float(USER_FACING_COMPLETION_DEADLINE_MAX_S),
        )
        return float(needed)
    except (ArithmeticError, ImportError, TypeError, ValueError) as exc:
        _degrade(exc, action="sized the latent episode from the caller's allowance alone")
        return float(floor_s)


class DeepDeliberationEngine:
    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._deliberations = 0
        # CP126 6b3e534c: health has to be derived from something. These are
        # the observations get_status() reports on.
        self._model_backed = 0
        self._unbacked = 0
        self._failures = 0
        self._consecutive_failures = 0
        self._last_latency_s = 0.0
        self._last_completed_at = 0.0
        logger.info("🪐 DeepDeliberationEngine initialized (Deep Thought lineage)")

    @staticmethod
    def _heuristic_refine(question: str) -> str:
        q = question.strip()
        vague = (
            "how do i",
            "what should i",
            "can you help",
            "what is the best",
            "fix this",
            "make it better",
        )
        low = q.lower()
        if any(v in low for v in vague) or len(q.split()) < 6:
            return (
                f"{q.rstrip('?')} — specifically: what concrete outcome defines success, "
                "what constraints apply, and what is the single most important sub-question?"
            )
        return q

    def refine_question(self, question: str) -> str:
        """Synchronous question refinement (no model call) for idle/background callers."""
        return self._heuristic_refine(question)

    async def deliberate(
        self,
        question: str,
        context: dict | None = None,
        budget: int = 2,
        *,
        timeout_s: float = 45.0,
        foreground_request: bool = True,
    ) -> DeliberationResult:
        if type(foreground_request) is not bool:
            raise ValueError("foreground_request must be boolean")
        self._deliberations += 1
        _started_at = time.time()
        refined = self._heuristic_refine(question)
        answer = ""
        used_model = False
        used_latent_cortex = False
        passes = 0

        brain = resolve_brain(self.orchestrator)
        if brain is not None and hasattr(brain, "think"):
            try:
                import asyncio

                from core.brain.types import ThinkingMode

                refine_prompt = (
                    "Restate the user's question as the *real* question they need answered. "
                    "One sentence.\nQUESTION: " + question[:400]
                )
                refine_out = coerce_text(
                    await asyncio.wait_for(
                        brain.think(
                            refine_prompt,
                            mode=ThinkingMode.FAST,
                            origin="deep_thought",
                            is_background=not foreground_request,
                        ),
                        timeout=min(20.0, timeout_s),
                    )
                )
                if refine_out:
                    refined = refine_out.strip()[:400]
                    passes += 1
                # DEEP pass, first choice: a Recursive Latent Cortex episode
                # on the resident model — workspace recurrence buys real
                # computational depth before any token is committed. Honest
                # refusals (busy lane, disabled, no worker) fall through to
                # ordinary generation below.
                try:
                    from core.brain.latent_cortex_service import get_latent_cortex_service

                    # Compiled understanding: digest-first conceptual context
                    # for the episode — dense, provenance-carrying concept
                    # digests instead of raw retrieval, sized for the
                    # episode's bounded compaction budget. Absent or failed
                    # ⇒ the episode proceeds on the question alone.
                    episode_messages = None
                    try:
                        from core.knowledge.compiled_understanding import (
                            get_compiled_understanding,
                        )

                        understanding = await asyncio.wait_for(
                            get_compiled_understanding().understand(refined),
                            timeout=min(20.0, timeout_s),
                        )
                        compiled_context = str(
                            understanding.get("context") or ""
                        ).strip()
                        if compiled_context:
                            episode_messages = [
                                {
                                    "role": "system",
                                    "content": (
                                        "Compiled understanding (provenance-"
                                        "tracked concept digests):\n"
                                        + compiled_context
                                    ),
                                },
                                {"role": "user", "content": refined},
                            ]
                    except (ImportError, AttributeError, RuntimeError,
                            TypeError, ValueError, TimeoutError) as cu_exc:
                        _degrade(
                            cu_exc,
                            action=(
                                "ran latent episode without compiled "
                                "understanding context"
                            ),
                        )

                    # These were flat ceilings of 120s and 150s. The latent
                    # cortex refuses before executing when the answer surface
                    # cannot fit the window it is given, and the smallest
                    # compound surface needs more than 120s allows — so every
                    # multi-part question was refused before starting, and the
                    # harder the question the more certain the refusal. Size
                    # the window by what the answer actually needs, keeping the
                    # old allowance as the floor and the runtime's own
                    # published deadline as the ceiling.
                    latent_timeout_s = _latent_episode_seconds(
                        refined, floor_s=min(120.0, timeout_s * 2)
                    )
                    latent = await asyncio.wait_for(
                        get_latent_cortex_service(self.orchestrator).deep_reason(
                            None if episode_messages else refined,
                            messages=episode_messages,
                            stakes=0.6,
                            uncertainty=0.7,
                            domain="deliberation",
                            timeout_s=latent_timeout_s,
                            foreground_request=foreground_request,
                        ),
                        # The outer net kept 30s over the inner window; that
                        # margin is preserved rather than replaced.
                        timeout=latent_timeout_s + 30.0,
                    )
                    if latent.get("ok") and str(latent.get("text") or "").strip():
                        answer = str(latent["text"]).strip()
                        passes += 1
                        used_model = True
                        used_latent_cortex = True
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                    _degrade(exc, action="fell back to ordinary generation after latent cortex episode failed")
                for _ in range(max(1, budget)):
                    if answer:
                        break
                    ans_out = coerce_text(
                        await asyncio.wait_for(
                            brain.think(
                                "Answer thoroughly and precisely:\n" + refined,
                                mode=(
                                    ThinkingMode.DEEP
                                    if hasattr(ThinkingMode, "DEEP")
                                    else ThinkingMode.FAST
                                ),
                                origin="deep_thought",
                                is_background=not foreground_request,
                            ),
                            timeout=timeout_s,
                        )
                    )
                    if ans_out:
                        answer = ans_out.strip()
                        passes += 1
                        used_model = True
                        break
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                _degrade(exc, action="returned refined-question with heuristic note after model deliberation failed")

        if not answer:
            answer = (
                "No model was available to answer, but the question has been sharpened. "
                f"Answer the refined question: {refined}"
            )
        # Record what this deliberation actually achieved. A run that
        # produced only the sharpened-question fallback did NOT deliberate
        # with a model, and health must be able to say so.
        self._last_latency_s = max(0.0, time.time() - _started_at)
        self._last_completed_at = time.time()
        if used_model:
            self._model_backed += 1
            self._consecutive_failures = 0
        else:
            self._unbacked += 1
            self._consecutive_failures += 1
        return DeliberationResult(
            original_question=question[:300],
            refined_question=refined,
            answer=answer,
            passes=passes,
            used_model=used_model,
            used_latent_cortex=used_latent_cortex,
        )

    def get_status(self) -> dict[str, Any]:
        """Health derived from what this engine actually managed to do.

        CP126 6b3e534c. ``healthy`` was the literal ``True`` — it ignored
        model availability, latency, failure streaks and whether any
        deliberation had ever completed with a model behind it. A health
        surface that cannot report ill health is not a health surface; it is
        a constant that happens to be shaped like one, and every consumer
        polling it was being told nothing.

        An engine is healthy when it is either untested (nothing has been
        asked of it yet — that is honestly unknown, not broken) or has been
        completing deliberations with a real model recently enough to
        believe it still can.
        """
        completed = self._model_backed + self._unbacked
        success_rate = (self._model_backed / completed) if completed else None
        untested = completed == 0
        healthy = bool(
            untested
            or (
                self._consecutive_failures < _UNHEALTHY_FAILURE_STREAK
                and (success_rate or 0.0) >= _MIN_MODEL_BACKED_RATE
            )
        )
        reasons: list[str] = []
        if not untested:
            if self._consecutive_failures >= _UNHEALTHY_FAILURE_STREAK:
                reasons.append(
                    f"consecutive_unbacked_deliberations={self._consecutive_failures}"
                )
            if (success_rate or 0.0) < _MIN_MODEL_BACKED_RATE:
                reasons.append(f"model_backed_rate={success_rate:.2f}")
        return {
            "deliberations": self._deliberations,
            "healthy": healthy,
            # "unknown" is distinct from "well": nothing has been asked yet.
            "state": "untested" if untested else ("healthy" if healthy else "degraded"),
            "model_backed": self._model_backed,
            "unbacked": self._unbacked,
            "consecutive_unbacked": self._consecutive_failures,
            "model_backed_rate": success_rate,
            "last_latency_s": round(self._last_latency_s, 3),
            "last_completed_at": self._last_completed_at,
            "unhealthy_reasons": reasons,
        }


_INSTANCE: DeepDeliberationEngine | None = None


def get_deep_deliberation(orchestrator: Any = None) -> DeepDeliberationEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = DeepDeliberationEngine(orchestrator=orchestrator)
    return _INSTANCE


def register_deep_deliberation(orchestrator: Any = None) -> DeepDeliberationEngine:
    from core.service_names import ServiceNames

    inst = get_runtime_service(
        ServiceNames.DEEP_THOUGHT,
        default=None,
    ) or get_deep_deliberation(orchestrator)
    register_runtime_service(
        ServiceNames.DEEP_THOUGHT,
        inst,
        required=False,
        owner="core/brain/deep_deliberation.py",
        registered_by="register_deep_deliberation",
    )
    register_runtime_service(
        "deep_thought",
        inst,
        required=False,
        owner="core/brain/deep_deliberation.py",
        registered_by="register_deep_deliberation",
    )
    return inst


__all__ = [
    "DeepDeliberationEngine",
    "DeliberationResult",
    "get_deep_deliberation",
    "register_deep_deliberation",
]
