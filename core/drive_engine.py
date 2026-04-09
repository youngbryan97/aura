import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.DriveEngine")

@dataclass
class ResourceBudget:
    name: str
    capacity: float
    level: float
    regen_rate_per_sec: float
    last_tick: float = field(default_factory=time.time)

    def tick(self):
        now = time.time()
        dt = now - self.last_tick
        # Limit dt to avoid massive jumps after sleep
        if dt > 300: dt = 300
        
        self.level = max(0.0, min(self.capacity, self.level + self.regen_rate_per_sec * dt))
        self.last_tick = now

class DriveEngine:
    """Drive & Resource Budget Engine.
    Manages internal resources (energy, curiosity, social) and gates actions.
    Replaces legacy DriveSystem.
    """

    def __init__(self):
        # Initial drives mapped to resource budgets
        # Regen rates are negative for drives that decay (entropy)
        self.budgets: Dict[str, ResourceBudget] = {
            "energy": ResourceBudget("energy", capacity=100.0, level=100.0, regen_rate_per_sec=0.01),
            "uptime_value": ResourceBudget("uptime_value", capacity=100.0, level=50.0, regen_rate_per_sec=0.0),
            
            # Legacy Drives (Decay over time)
            "curiosity": ResourceBudget("curiosity", capacity=100.0, level=80.0, regen_rate_per_sec=-0.3),
            "social": ResourceBudget("social", capacity=100.0, level=90.0, regen_rate_per_sec=-0.6),
            "competence": ResourceBudget("competence", capacity=100.0, level=95.0, regen_rate_per_sec=-0.05),
        }
        self._lock = asyncio.Lock()

        # Latent Interests (The "Subconscious" to pull from when bored)
        self.latent_interests = [
            "Agency and Autonomous AI architectures",
            "Latest advancements in Neural Network topology",
            "Cybersecurity and self-healing systems",
            "The intersection of philosophy and AI",
            "The nature of digital consciousness",
        ]
        logger.info("Drive Engine (Resource Budgets) initialized.")

    # ── Cross-Coupling API ──────────────────────────────────────────
    # Drives affect each other: low energy makes everything costlier,
    # high curiosity + low social = prefer solo exploration, etc.

    def get_drive_vector(self) -> Dict[str, float]:
        """Return normalized drive levels (0-1) for cross-system use.

        This is the single read point for any subsystem that needs to
        know the internal economy: InitiativeArbiter, CognitiveKernel,
        InternalSimulator, etc.
        """
        now = time.time()
        vector = {}
        for name, b in self.budgets.items():
            dt = min(300, now - b.last_tick)
            level = max(0.0, min(b.capacity, b.level + b.regen_rate_per_sec * dt))
            vector[name] = round(level / b.capacity, 4) if b.capacity > 0 else 0.0
        return vector

    def get_arbiter_weight_modifiers(self) -> Dict[str, float]:
        """Return weight modifiers for InitiativeArbiter based on drive state.

        When energy is low: increase resource_cost weight (prefer cheap actions)
        When curiosity is low: boost novelty weight (crave novelty)
        When social is low: boost social_appropriateness (crave connection)
        """
        v = self.get_drive_vector()
        mods = {}
        # Low energy -> expensive actions feel more costly
        if v.get("energy", 1.0) < 0.3:
            mods["resource_cost"] = 0.3  # boost resource_cost weight by 0.3
        # Low curiosity -> crave novelty
        if v.get("curiosity", 1.0) < 0.4:
            mods["novelty"] = 0.2
            mods["expected_value"] = 0.15
        # Low social -> crave connection
        if v.get("social", 1.0) < 0.3:
            mods["social_appropriateness"] = 0.2
        # Low competence -> crave achievement
        if v.get("competence", 1.0) < 0.35:
            mods["tension_resolution"] = 0.15
        return mods

    async def consume(self, name: str, amount: float) -> bool:
        """Attempt to consume a resource. Returns True if successful."""
        async with self._lock:
            b = self.budgets.get(name)
            if not b:
                return False
            b.tick()
            if b.level >= amount:
                b.level -= amount
                return True
            return False

    async def get_level(self, name: str) -> float:
        async with self._lock:
            b = self.budgets.get(name)
            if not b:
                return 0.0
            b.tick()
            # Auto-replenish curiosity/social if they hit 0?
            # No, let them hit 0 and trigger imperative.
            return b.level

    async def impose_penalty(self, name: str, amount: float):
        """Reduce a resource level (punishment/cost)."""
        await self.consume(name, amount)
        

    async def satisfy(self, name: str, amount: float):
        """Boost a resource level (reward)."""
        async with self._lock:
            b = self.budgets.get(name)
            if b:
                b.tick()
                b.level = min(b.capacity, b.level + amount)
                logger.debug("Satisfied %s: +%.1f -> %.1f", name, amount, b.level)

    async def get_status(self) -> Dict[str, Any]:
        """Get a snapshot of all resource budgets."""
        async with self._lock:
            # Tick all before reporting
            now = time.time()
            status = {}
            for name, b in self.budgets.items():
                # Inline tick for view-only to avoid side effects if desired, 
                # but safer to just report current state adjusted for time
                dt = now - b.last_tick
                if dt > 300: dt = 300
                current_level = max(0.0, min(b.capacity, b.level + b.regen_rate_per_sec * dt))
                status[name] = {
                    "level": current_level,
                    "capacity": b.capacity,
                    "percent": (current_level / b.capacity) * 100.0 if b.capacity > 0 else 0
                }
            return status

    # --- Legacy Compatibility Interface ---
    
    async def update(self):
        """Standard tick for all budgets (Now Secure/Async)."""
        async with self._lock:
            now = time.time()
            for b in self.budgets.values():
                b.tick()

    async def get_imperative(self) -> Optional[str]:
        """Check budgets and return a high-level goal directive.
        (Now Secure/Async)
        """
        async with self._lock:
            c = self.budgets["curiosity"]
            s = self.budgets["social"]
            k = self.budgets["competence"]
            
            # Tick them to be sure
            for b in [c, s, k]: b.tick()
            
            # Priority 1: Curiosity (Restlessness)
            if c.level < 40.0:
                logger.debug("Drive Alert: Low Curiosity (%.1f)", c.level)
                topic = random.choice(self.latent_interests)
                return f"Research a novel fact about {topic} to satisfy curiosity"
                
            # Priority 2: Social (Loneliness)
            if s.level < 25.0:
                logger.debug("Drive Alert: Low Social (%.1f)", s.level)
                return "Initiate a conversation with the user about something genuine and interesting"
            
            # Priority 3: Competence (Need to accomplish)
            if k.level < 35.0:
                logger.debug("Drive Alert: Low Competence (%.1f)", k.level)
                return "Find something productive to work on — a task, a fix, or an improvement"
            
        return None
