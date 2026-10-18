"""core/agency/self_play.py

Asynchronous Adversarial Self-Play.
Spawns competing cognitive shards during system idle time to generate novel 
problems and solve them, pushing failures to the DistillationPipe for nightly learning.
"""
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import logging
import random
import re
import time
from core.runtime.service_registry import get_runtime_service
from core.runtime.background_policy import (
    RESEARCH_BACKGROUND_POLICY,
    background_activity_reason,
)

logger = logging.getLogger("Aura.SelfPlay")

class ContinuousSelfPlay:
    _UNCERTAINTY_PATTERNS = (
        "i'm not sure",
        "i am not sure",
        "cannot determine",
        "can't determine",
        "insufficient information",
        "not enough information",
        "unclear",
        "i guess",
        "maybe",
    )
    _REASONING_MARKERS = (
        "because",
        "therefore",
        "if ",
        "then ",
        "step",
        "trade-off",
        "constraint",
        "assume",
        "first,",
        "second,",
    )
    _RESOLUTION_MARKERS = (
        "final answer",
        "conclusion",
        "therefore",
        "the best resolution",
        "the solution is",
    )

    def __init__(self, idle_threshold_seconds: int = 1800):
        # Default: Wait for 30 minutes of chat silence before initiating self-play
        self.idle_threshold = idle_threshold_seconds
        self.is_playing = False

    async def _ask_her_model(self, request: str, *, purpose: str) -> str:
        """A request to her model, not an objective.

        Sent through the cognitive engine, these requests became the kernel's
        objectives. The kernel routed them as tasks, the task engine started
        218 goals named "[SYSTEM ROLE: THE ADVERSARY]..." and the commitment
        engine kept 224 commitments to them (her log, 10 to 18 October 2026).
        The swarm's role table left the same path on 15 September
        (core/collective/delegator.py).
        """
        router = get_runtime_service("llm_router", default=None)
        if router is None or not hasattr(router, "think"):
            return ""
        response = await router.think(prompt=request, prefer_tier="primary", is_background=True,
                                      origin="self_play", purpose=purpose)
        return str(response if isinstance(response, str) else getattr(response, "content", "") or "").strip()

    async def _generate_adversarial_problem(self) -> str:
        """An original problem in a domain chosen at random, without its solution."""
        domains = ["resource_allocation_crisis", "philosophical_paradox", "multi_variable_logic_puzzle", "game_theory_dilemma"]
        selected_domain = random.choice(domains).replace("_", " ")
        return await self._ask_her_model(
            f"Write one original {selected_domain} with competing variables and hidden trade-offs that takes "
            "step-by-step deduction to resolve. Give the problem only, without its solution.",
            purpose="self_play_problem",
        )

    async def _attempt_solution(self, problem: str) -> str:
        """Her resolution of the problem, worked step by step."""
        return await self._ask_her_model(
            f"Resolve this step by step and end with the final answer.\n\n{problem}",
            purpose="self_play_solution",
        )

    def _evaluate_solution_quality(self, solution: str) -> tuple[bool, float, str]:
        text = str(solution or "").strip()
        if not text:
            return False, 0.0, "empty_response"

        lowered = text.lower()
        word_count = len(text.split())
        if word_count < 50:
            return False, 0.1, "too_short"

        score = 0.3
        reasons: list[str] = []

        uncertainty_hits = sum(1 for pattern in self._UNCERTAINTY_PATTERNS if pattern in lowered)
        if uncertainty_hits:
            score -= min(0.5, uncertainty_hits * 0.2)
            reasons.append("uncertain_language")

        if any(marker in lowered for marker in self._REASONING_MARKERS):
            score += 0.25
        else:
            reasons.append("missing_reasoning_markers")

        if any(marker in lowered for marker in self._RESOLUTION_MARKERS):
            score += 0.2
        else:
            reasons.append("missing_resolution_marker")

        if re.search(r"(^|\n)\s*(\d+\.|[-*])\s+", text):
            score += 0.15

        paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
        if len(paragraphs) >= 2:
            score += 0.1

        score = max(0.0, min(score, 0.95))
        return score >= 0.55, score, ",".join(reasons) or "structured"

    async def trigger_cycle(self, last_user_interaction: float):
        """Main loop trigger, called by the AgencyCore heartbeat."""
        if self.is_playing:
            return

        if float(last_user_interaction or 0.0) <= 0.0:
            logger.debug("♟️ Self-play deferred: no valid user-idle anchor yet.")
            return

        time_since_interaction = time.time() - last_user_interaction
        if time_since_interaction < self.idle_threshold:
            return

        orchestrator = get_runtime_service("orchestrator", default=None)
        defer_reason = background_activity_reason(
            orchestrator,
            profile=RESEARCH_BACKGROUND_POLICY,
            min_idle_seconds=float(self.idle_threshold),
            max_failure_pressure=0.35,
            require_conversation_ready=False,
        )
        if defer_reason:
            logger.debug("♟️ Self-play deferred: %s.", defer_reason)
            return

        self.is_playing = True
        logger.info("♟️ System idle. Initiating Continuous Self-Play cycle...")

        try:
            # 1. Generate the problem
            problem = await self._generate_adversarial_problem()
            if not problem:
                return

            # 2. Attempt to solve it
            solution = await self._attempt_solution(problem)
            if not solution:
                return

            # 3. Evaluate the effort and route failed local attempts to the
            # governed distillation dataset for later resident-model training.
            succeeded, confidence, reason = self._evaluate_solution_quality(solution)
            if not succeeded:
                logger.info(
                    "❌ Self-Play Solver failed quality gate (%s, confidence=%.2f). Routing to Distillation Pipe.",
                    reason,
                    confidence,
                )
                
                # The pipe the learning phase and the dreamer use. It is a
                # module singleton that nothing registers as a service, so the
                # container lookup this was sent failed solutions to read None.
                from core.adaptation.distillation_pipe import get_distillation_pipe

                distillation = get_distillation_pipe()
                if distillation and hasattr(distillation, 'flag_for_distillation'):
                    await distillation.flag_for_distillation(
                        prompt=problem,
                        local_response=solution,
                        confidence=max(0.05, confidence),
                    )
            else:
                logger.info("✅ Self-Play Solver succeeded (confidence=%.2f).", confidence)
                
                # Optional: Send highly successful complex logic to the Abstraction Engine
                abstraction = get_runtime_service("abstraction_engine", default=None)
                if abstraction:
                    # Note: abstract_from_success is async
                    task = get_task_tracker().create_task(
                        abstraction.abstract_from_success(context=problem, successful_resolution=solution)
                    )
                    task.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('self_play', e)
            logger.error("Self-Play Cycle Error: %s", e)
        finally:
            self.is_playing = False
