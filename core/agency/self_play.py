"""core/agency/self_play.py

Asynchronous Adversarial Self-Play.
Spawns competing cognitive shards during system idle time to generate novel 
problems and solve them, pushing failures to the DistillationPipe for nightly learning.
"""
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import random
import re
import time
from core.container import ServiceContainer
from core.brain.cognitive_engine import ThinkingMode

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

    async def _generate_adversarial_problem(self) -> str:
        """The Adversary Shard creates a novel, highly complex scenario."""
        engine = ServiceContainer.get("cognitive_engine", default=None)
        if not engine:
            return ""

        domains = ["resource_allocation_crisis", "philosophical_paradox", "multi_variable_logic_puzzle", "game_theory_dilemma"]
        selected_domain = random.choice(domains)

        prompt = f"""[SYSTEM ROLE: THE ADVERSARY]
Your task is to invent a highly complex, original problem designed to test the limits of a reasoning engine.
DOMAIN: {selected_domain}

The problem must involve competing variables, hidden trade-offs, and require step-by-step logical deduction. 
Do not provide the solution. Output ONLY the problem statement.
"""
        res = await engine.think(objective=prompt, mode=ThinkingMode.FAST, priority=0.2, block_user=False)
        return res.content if hasattr(res, 'content') else str(res)

    async def _attempt_solution(self, problem: str) -> str:
        """The Solver Shard attempts to resolve the Adversary's problem."""
        engine = ServiceContainer.get("cognitive_engine", default=None)
        if not engine:
            return ""

        prompt = f"""[SYSTEM ROLE: THE SOLVER]
You are presented with a highly complex scenario.
PROBLEM:
"{problem}"

Task: Think step-by-step to arrive at the most logically sound resolution. 
Detail your logical chain of thought before providing the final answer.
"""
        # Deep mode to force chain-of-thought
        res = await engine.think(objective=prompt, mode=ThinkingMode.DEEP, priority=0.2, block_user=False)
        return res.content if hasattr(res, 'content') else str(res)

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
            
        time_since_interaction = time.time() - last_user_interaction
        if time_since_interaction < self.idle_threshold:
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

            # 3. Evaluate the effort and route to Distillation
            # If the solver's response is short, confused, or lacks structured logic,
            # we consider it a failure and send it to the Teacher model.
            succeeded, confidence, reason = self._evaluate_solution_quality(solution)
            if not succeeded:
                logger.info(
                    "❌ Self-Play Solver failed quality gate (%s, confidence=%.2f). Routing to Distillation Pipe.",
                    reason,
                    confidence,
                )
                
                distillation = ServiceContainer.get("distillation_pipe", default=None)
                if distillation and hasattr(distillation, 'flag_for_distillation'):
                    # This uses existing pipeline to ask Gemini for the ideal answer
                    # and saves it to lora_dataset.jsonl
                    await distillation.flag_for_distillation(
                        prompt=problem,
                        local_response=solution,
                        confidence=max(0.05, confidence),
                    )
            else:
                logger.info("✅ Self-Play Solver succeeded (confidence=%.2f).", confidence)
                
                # Optional: Send highly successful complex logic to the Abstraction Engine
                abstraction = ServiceContainer.get("abstraction_engine", default=None)
                if abstraction:
                    # Note: abstract_from_success is async
                    task = get_task_tracker().create_task(
                        abstraction.abstract_from_success(context=problem, successful_resolution=solution)
                    )
                    task.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)

        except Exception as e:
            record_degradation('self_play', e)
            logger.error("Self-Play Cycle Error: %s", e)
        finally:
            self.is_playing = False
