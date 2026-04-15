import asyncio
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

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

    Boredom Accumulator (Rubicon):
      Tracks time since last meaningful prediction error reduction.
      When free energy stays low for >300 ticks (5 min at 1 Hz),
      raises seek_novelty flag, boosting curiosity drive and triggering
      InitiativeSynthesizer to generate exploration actions.
    """

    # ── Boredom thresholds ────────────────────────────────────────────
    BOREDOM_TICK_THRESHOLD = 300      # ticks of low FE before boredom
    BOREDOM_FE_CEILING = 0.25         # FE below this counts as "low"
    BOREDOM_CURIOSITY_BOOST = 25.0    # points added to curiosity when bored
    BOREDOM_RELIEF_DECAY = 0.7        # multiplier on boredom_ticks when relieved

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

        # ── Boredom Accumulator state ────────────────────────────────
        self._boredom_ticks: int = 0           # ticks since last meaningful FE reduction
        self._last_fe_value: float = 0.3       # track previous FE for delta detection
        self._seek_novelty: bool = False        # raised when boredom threshold crossed
        self._last_boredom_relief: float = time.time()
        self._boredom_history: Deque[float] = deque(maxlen=600)  # 10 min of boredom levels

        # Latent Interests (The "Subconscious" to pull from when bored)
        self.latent_interests = [
            "Agency and Autonomous AI architectures",
            "Latest advancements in Neural Network topology",
            "Cybersecurity and self-healing systems",
            "The intersection of philosophy and AI",
            "The nature of digital consciousness",
            "Emergent behavior in complex systems",
            "The hard problem of consciousness",
            "Self-modifying code and meta-programming",
        ]
        logger.info("Drive Engine (Resource Budgets + Boredom Accumulator) initialized.")

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
        When bored (seek_novelty): strongly boost novelty and expected_value
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
        # Boredom: strong novelty-seeking when seek_novelty flag is raised
        if self._seek_novelty:
            mods["novelty"] = mods.get("novelty", 0.0) + 0.35
            mods["expected_value"] = mods.get("expected_value", 0.0) + 0.2
            mods["urgency"] = mods.get("urgency", 0.0) + 0.15
        return mods

    # ── Boredom Accumulator ────────────────────────────────────────────

    @property
    def seek_novelty(self) -> bool:
        """True when boredom has accumulated past threshold."""
        return self._seek_novelty

    @property
    def boredom_level(self) -> float:
        """Normalized boredom: 0.0 (engaged) to 1.0 (deeply bored)."""
        return min(1.0, self._boredom_ticks / max(1, self.BOREDOM_TICK_THRESHOLD))

    def tick_boredom(self, current_fe: float) -> None:
        """Called once per heartbeat tick (1 Hz) with current free energy.

        If FE stays low (< BOREDOM_FE_CEILING) for > BOREDOM_TICK_THRESHOLD
        ticks, raises the seek_novelty flag and boosts curiosity.
        A meaningful FE increase (prediction error reduction) relieves boredom.
        """
        fe_delta = current_fe - self._last_fe_value
        self._last_fe_value = current_fe

        if current_fe < self.BOREDOM_FE_CEILING:
            # World is predictable -- accumulate boredom
            self._boredom_ticks += 1
        else:
            # Some prediction error present -- partial relief
            self._boredom_ticks = max(0, self._boredom_ticks - 3)

        # Meaningful FE *increase* (surprise spike) relieves boredom strongly
        if fe_delta > 0.1:
            self._relieve_boredom("fe_spike", factor=0.5)

        self._boredom_history.append(self.boredom_level)

        # Cross the threshold?
        was_bored = self._seek_novelty
        self._seek_novelty = self._boredom_ticks >= self.BOREDOM_TICK_THRESHOLD

        if self._seek_novelty and not was_bored:
            logger.info(
                "BOREDOM THRESHOLD CROSSED (%d ticks, FE=%.3f) -- seek_novelty ON",
                self._boredom_ticks, current_fe,
            )
            # Boost curiosity drive to make the arbiter favor exploration
            b = self.budgets.get("curiosity")
            if b:
                b.level = min(b.capacity, b.level + self.BOREDOM_CURIOSITY_BOOST)
            # Notify neurochemical system (if wired)
            self._notify_neurochemical_boredom()

    def relieve_boredom(self, source: str) -> None:
        """External relief: user interaction, tool use, new information, etc."""
        self._relieve_boredom(source)

    def _relieve_boredom(self, source: str, factor: float = 0.7) -> None:
        """Internal boredom relief with configurable decay factor."""
        old = self._boredom_ticks
        self._boredom_ticks = int(self._boredom_ticks * (1.0 - factor))
        if self._seek_novelty and self._boredom_ticks < self.BOREDOM_TICK_THRESHOLD:
            self._seek_novelty = False
            logger.info(
                "BOREDOM RELIEVED by %s (ticks %d -> %d) -- seek_novelty OFF",
                source, old, self._boredom_ticks,
            )
        self._last_boredom_relief = time.time()

    def _notify_neurochemical_boredom(self) -> None:
        """Push boredom neurochemistry: low dopamine + orexin depletion."""
        try:
            from core.container import ServiceContainer
            ncs = ServiceContainer.get("neurochemical_system", default=None)
            if ncs and hasattr(ncs, "on_boredom"):
                ncs.on_boredom(self.boredom_level)
        except Exception as e:
            logger.debug("Boredom neurochemical notify failed: %s", e)

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
        """Get a snapshot of all resource budgets + boredom state."""
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
            # Boredom accumulator
            status["_boredom"] = {
                "ticks": self._boredom_ticks,
                "level": round(self.boredom_level, 3),
                "seek_novelty": self._seek_novelty,
                "threshold": self.BOREDOM_TICK_THRESHOLD,
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

            # Priority 0: Boredom accumulator (highest urgency when crossed)
            if self._seek_novelty:
                topic = random.choice(self.latent_interests)
                logger.debug("Drive Alert: Boredom (seek_novelty, ticks=%d)", self._boredom_ticks)
                return f"Seek novelty: explore {topic} -- prediction landscape is stale"

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
                return "Find something productive to work on -- a task, a fix, or an improvement"

        return None
