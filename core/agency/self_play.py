"""core/agency/self_play.py

Asynchronous Adversarial Self-Play.
Spawns competing cognitive shards during system idle time to generate novel 
problems and solve them, pushing failures to the DistillationPipe for nightly learning.
"""
import asyncio
import logging
import random
import time
from core.container import ServiceContainer
from core.brain.cognitive_engine import ThinkingMode

logger = logging.getLogger("Aura.SelfPlay")

class ContinuousSelfPlay:
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
            if "I'm not sure" in solution or len(solution.split()) < 50:
                logger.info("❌ Self-Play Solver failed. Routing to Distillation Pipe.")
                
                distillation = ServiceContainer.get("distillation_pipe", default=None)
                if distillation and hasattr(distillation, 'flag_for_distillation'):
                    # This uses existing pipeline to ask Gemini for the ideal answer
                    # and saves it to lora_dataset.jsonl
                    await distillation.flag_for_distillation(
                        prompt=problem,
                        local_response=solution,
                        confidence=0.1 # Low confidence triggered this
                    )
            else:
                logger.info("✅ Self-Play Solver succeeded. Logic is sound.")
                
                # Optional: Send highly successful complex logic to the Abstraction Engine
                abstraction = ServiceContainer.get("abstraction_engine", default=None)
                if abstraction:
                    # Note: abstract_from_success is async
                    task = asyncio.create_task(
                        abstraction.abstract_from_success(context=problem, successful_resolution=solution)
                    )
                    task.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)

        except Exception as e:
            logger.error("Self-Play Cycle Error: %s", e)
        finally:
            self.is_playing = False
