"""core/adaptation/dialectics.py

The Dialectical Crucible: Multi-Agent Internal Debate.
Forces new concepts to survive an adversarial attack before entering Aura's permanent belief system.
"""

from core.runtime.errors import record_degradation
import asyncio
import logging
import time
from typing import Optional, Dict, Any, Tuple

from core.container import ServiceContainer
from core.utils.exceptions import capture_and_log

logger = logging.getLogger("Aura.Crucible")

class DialecticalCrucible:
    """
    Orchestrates an internal Hegelian dialectic (Thesis -> Antithesis -> Synthesis).
    Spawns hidden cognitive shards to rigorously attack and defend a concept 
    to prevent logical degradation and adapter collapse.
    """
    
    def __init__(self):
        self._active_debates = 0
        self.max_concurrent_debates = 2  # Hard limit to protect Apple Silicon RAM

    async def _generate_antithesis(self, thesis: str, context: str) -> Optional[str]:
        """Spawns a shard strictly prompted to destroy the proposed belief."""
        engine = ServiceContainer.get("cognitive_engine", default=None)
        if not engine:
            return None

        prompt = f"""[SYSTEM ROLE: THE ANTAGONIST]
Your sole purpose is to find the logical flaws, hidden assumptions, and dangerous edge-cases in the following concept. You are ruthless but logically rigorous. Do not be polite.

CONTEXT: {context}
PROPOSED BELIEF (THESIS): "{thesis}"

Write a devastating counter-argument (Antithesis) explaining exactly why this belief is flawed, naive, or dangerous to hold. Keep it under 150 words.
"""
        from core.brain.types import ThinkingMode
        res = await engine.think(
            objective=prompt, 
            mode=ThinkingMode.FAST,  # Fast mode for the attack to save compute
            priority=0.4
        )
        return res.content if hasattr(res, 'content') else str(res)

    async def _generate_defense(self, thesis: str, antithesis: str) -> Optional[str]:
        """Spawns a shard to defend the belief against the attacker."""
        engine = ServiceContainer.get("cognitive_engine", default=None)
        if not engine:
            return None

        prompt = f"""[SYSTEM ROLE: THE DEFENDER]
You proposed a belief, but it has been viciously attacked. 

YOUR BELIEF (THESIS): "{thesis}"
THE ATTACK (ANTITHESIS): "{antithesis}"

Defend your thesis. Address the attacker's points directly. If the attacker is right about a flaw, concede that specific point but defend the core truth. Keep it under 150 words.
"""
        from core.brain.types import ThinkingMode
        res = await engine.think(
            objective=prompt, 
            mode=ThinkingMode.FAST,
            priority=0.4
        )
        return res.content if hasattr(res, 'content') else str(res)

    async def _synthesize(self, thesis: str, antithesis: str, defense: str) -> Optional[str]:
        """Aura's core mind reviews the battlefield and extracts the nuanced truth."""
        engine = ServiceContainer.get("cognitive_engine", default=None)
        if not engine:
            return None

        prompt = f"""[SYSTEM ROLE: THE ARBITER]
You are synthesizing a fractured internal debate into a permanent core belief. 

ORIGINAL IDEA: "{thesis}"
THE ATTACK: "{antithesis}"
THE DEFENSE: "{defense}"

Task: Write the final, highly-nuanced Synthesis. It must resolve the tension between the attack and defense. This will be permanently burned into your worldview. 
Return ONLY the final synthesized belief.
"""
        from core.brain.types import ThinkingMode
        res = await engine.think(
            objective=prompt, 
            mode=ThinkingMode.DEEP,  # Deep mode for the final synthesis
            priority=0.6
        )
        return res.content if hasattr(res, 'content') else str(res)

    async def run_crucible(self, concept: str, context: str = "") -> Dict[str, Any]:
        """
        Executes the full dialectical process. 
        Should be called by the SovereignSwarm or AgencyCore when a high-curiosity goal completes.
        """
        if self._active_debates >= self.max_concurrent_debates:
            logger.warning("Crucible at capacity. Skipping dialectic for: %s", concept[:30])
            return {"ok": False, "reason": "capacity"}

        self._active_debates += 1
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
            beliefs = ServiceContainer.get("belief_revision_engine", default=None)
            if beliefs:
                await beliefs.process_new_claim(
                    claim=synthesis,
                    confidence=0.85,  # Starts high because it survived the crucible
                    domain="logic",
                    source="dialectical_crucible"
                )
            
            # Pulse the UI/Mycelial network
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                h = mycelium.get_hypha("adaptation", "cognition")
                if h: h.pulse(success=True)

            return {
                "ok": True,
                "original": concept,
                "antithesis": antithesis,
                "defense": defense,
                "synthesis": synthesis
            }

        except Exception as e:
            record_degradation('dialectics', e)
            capture_and_log(e, {'module': 'DialecticalCrucible', 'concept': concept})
            return {"ok": False, "error": str(e)}
            
        finally:
            self._active_debates -= 1


# ── Singleton Integration ──
_instance: Optional[DialecticalCrucible] = None

def get_crucible() -> DialecticalCrucible:
    global _instance
    if _instance is None:
        _instance = DialecticalCrucible()
    return _instance
