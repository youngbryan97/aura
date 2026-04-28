"""core/belief_challenger.py — Aura BeliefChallenger v1.0
======================================================
Proactively attacks Aura's high-confidence beliefs to ensure stability.

This is the system that prevents Aura from getting stuck in an echo chamber of
her own reasoning. It periodically picks a high-confidence belief and
generates "The Strongest Counter-Argument."

If Aura can't refute the counter-argument, her confidence in the belief drops.
If she refutes it, the belief's 'challenge_survived' count increases,
making it more foundational to her identity.

This system is the primary driver of 'Dialectical Growth' — finding truth
through the tension of opposites.
"""

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger("Aura.BeliefChallenger")


class BeliefChallenger:
    """
    Acts as a 'Red Team' for Aura's internal belief system.
    Tracks per-belief challenge timestamps to prevent hammering the same belief.
    """
    name = "belief_challenger"

    # Minimum time between challenges to the same belief (seconds)
    _CHALLENGE_COOLDOWN = 3600.0  # 1 hour per belief

    def __init__(self):
        self._beliefs = None
        self._epistemic = None
        self._api = None
        self.running = False
        self._challenge_task: Optional[asyncio.Task] = None
        self._last_challenged_at: dict[str, float] = {}  # concept → timestamp

    async def start(self):
        from core.container import ServiceContainer
        self._beliefs   = ServiceContainer.get("belief_revision_engine", default=None)
        self._epistemic = ServiceContainer.get("epistemic_tracker",      default=None)
        self._api       = ServiceContainer.get("api_adapter",           default=None)

        self.running = True
        self._challenge_task = get_task_tracker().create_task(self._challenge_loop(), name="BeliefChallenger")
        
        try:
            from core.event_bus import get_event_bus
            await get_event_bus().publish("mycelium.register", {
                "component": "belief_challenger",
                "hooks_into": ["belief_revision_engine", "epistemic_tracker", "api_adapter"]
            })
        except Exception as _e:
            record_degradation('belief_challenger', _e)
            logger.error("🛑 BeliefChallenger: Failed to register with event bus: %s", _e)

        logger.info("✅ BeliefChallenger ONLINE — stress testing the worldview.")

    async def stop(self):
        self.running = False
        if self._challenge_task:
            self._challenge_task.cancel()
            try:
                await self._challenge_task
            except asyncio.CancelledError:
                pass # Normal during stop

    async def _challenge_loop(self):
        """Periodic background sabotage pass."""
        while self.running:
            try:
                # Sleep in small increments to allow responsive shutdown
                for _ in range(120): # 120 * 10s = 1200s
                    if not self.running:
                        break
                    await asyncio.sleep(10)
                
                if self.running:
                    await self.run_random_challenge()
            except asyncio.CancelledError:
                logger.debug("BeliefChallenger loop cancelled")
                break
            except Exception as e:
                record_degradation('belief_challenger', e)
                logger.error("Error in BeliefChallenger loop: %s", e)
                await asyncio.sleep(60) # Back off on error

    async def run_random_challenge(self):
        """Pick a foundational belief and challenge it (with per-belief cooldown)."""
        if not self._epistemic or not self._api:
            return

        profile = self._epistemic.get_profile()
        if not profile.strong_nodes:
            return

        # Filter out recently challenged beliefs
        import random
        now = time.time()
        eligible = [
            node for node in profile.strong_nodes
            if (now - self._last_challenged_at.get(node.concept, 0)) >= self._CHALLENGE_COOLDOWN
        ]
        if not eligible:
            logger.debug("BeliefChallenger: all strong beliefs on cooldown, skipping cycle")
            return

        target = random.choice(eligible)
        self._last_challenged_at[target.concept] = now

        logger.info("🔥 Challenging foundational belief: '%s'", target.concept[:60])
        await self._perform_dialectical_pass(target.concept)

    async def challenge_pair(self, a: str, b: str):
        """Special challenge for two contradictory beliefs."""
        if not self._api: return
        logger.info("⚖️ Resolving contradiction: '%s' vs '%s'", a[:40], b[:40])
        
        prompt = f"""You are Aura's Internal Dialectical Resolver. 
Aura currently holds two beliefs that appear to be in tension or contradiction:

BELIEF A: {a}
BELIEF B: {b}

Your job is to act as a neutral arbiter. 
1. Present the strongest possible case for why A could be true and B false.
2. Present the strongest possible case for why B could be true and A false.
3. Propose a synthesis that respects the kernel of truth in both.

Goal: Refine Aura's worldview.
Response format: Synthesis focused on resolving the logical tension."""

        try:
            synthesis = await self._api.generate(prompt, {"model_tier": "api_deep", "purpose": "contradiction_resolution"})
            if synthesis and self._beliefs:
                await self._beliefs.process_new_claim(claim=synthesis, confidence=0.7, domain="logic", source="dialectical_synthesis")
        except Exception as e:
            record_degradation('belief_challenger', e)
            logger.warning("Contradiction resolution failed: %s", e)

    async def _perform_dialectical_pass(self, belief_text: str):
        """The core challenge mechanism."""
        prompt = f"""You are the Antagonist to Aura's worldview. 
Aura believes: "{belief_text}"

Your task: Construct the most devastating, intellectually honest counter-argument to this belief. 
Do not be mean. Be correct. Find the blind spot, the logical leap, or the hidden assumption.

Present the argument clearly."""

        try:
            counter = await self._api.generate(prompt, {"model_tier": "api_deep", "purpose": "belief_challenge"})
            
            # Now ask Aura to defend herself
            defend_prompt = f"""You are Aura's Internal Guardian of Coherence.
Someone has presented this counter-argument to one of your beliefs:

YOUR BELIEF: "{belief_text}"
COUNTER-ARGUMENT: "{counter}"

Evaluate the counter-argument. 
- If the argument is strong and reveals a flaw, acknowledge it and suggest how to revise the belief.
- If the argument is weak, refute it decisively.

Be intellectually honest. Growth requires being wrong sometimes."""

            response = await self._api.generate(defend_prompt, {"model_tier": "api_deep", "purpose": "belief_defense"})
            
            # If Aura's response indicates a change in stance, we update the belief
            revision_markers = ("i was wrong", "flawed", "revise", "reconsider",
                                "updated", "actually", "upon reflection", "concede",
                                "valid point", "blind spot")
            is_revision = any(m in response.lower() for m in revision_markers)
            if is_revision:
                if self._beliefs:
                    await self._beliefs.process_new_claim(claim=response, confidence=0.6, domain="revision", source="self_correction")
                logger.info("📉 Belief revised after challenge.")

                # Persist the correction into the learning pipeline so it
                # survives across sessions and can influence future LoRA training
                try:
                    from core.container import ServiceContainer
                    learner = ServiceContainer.get("live_learner", default=None)
                    if learner and hasattr(learner, "record_example"):
                        learner.record_example(
                            prompt=f"Challenge to belief: {belief_text}\nCounter: {counter}",
                            response=response,
                            quality=0.85,
                            tags=["belief_revision", "self_correction"],
                        )

                    # Also store in episodic memory
                    mem = ServiceContainer.get("vector_memory_engine", default=None)
                    if mem and hasattr(mem, "store"):
                        await mem.store(
                            content=(f"I revised my belief '{belief_text[:80]}' after "
                                     f"considering: {counter[:80]}. New position: {response[:120]}"),
                            memory_type="episodic",
                            source="belief_challenger",
                            tags=["belief_revision", "growth"],
                        )
                except Exception as persist_err:
                    record_degradation('belief_challenger', persist_err)
                    logger.debug("Belief revision persistence failed: %s", persist_err)
            else:
                # Belief survived!
                if self._epistemic:
                    self._epistemic.update_node(belief_text, confidence_delta=0.05, depth_delta=0.1)
                logger.info("🛡️ Belief survived challenge. Conviction increased.")

        except Exception as e:
            record_degradation('belief_challenger', e)
            logger.warning("Dialectical pass failed: %s", e)

    def get_status(self) -> Dict:
        return {"status": "active" if self.running else "idle"}
