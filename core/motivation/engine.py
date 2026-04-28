"""core/motivation/engine.py — Aura Consolidated Motivation Engine
=========================================================
The "Digital Metabolism" version: Merges resource budgets, boredom 
triggers, and autonomous intention generation into a single system.
"""

from core.runtime.errors import record_degradation
import asyncio
import logging
import random
import time
import psutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.container import ServiceContainer
from core.runtime.background_policy import background_activity_allowed
from core.motivation.constants import MOTIVATION_BUDGET_DEFAULTS
from core.motivation.intention import DriveType, Intention
from core.utils.task_tracker import task_tracker

logger = logging.getLogger("Aura.Motivation")


def _background_autonomy_allowed(orchestrator) -> bool:
    return background_activity_allowed(
        orchestrator,
        min_idle_seconds=900.0,
        max_memory_percent=80.0,
        max_failure_pressure=0.12,
        require_conversation_ready=True,
    )

@dataclass
class ResourceBudget:
    name: str
    capacity: float
    level: float
    decay_rate_per_sec: float  # Positive means it goes DOWN per sec
    last_tick: float = field(default_factory=time.time)

    def tick(self):
        now = time.time()
        dt = now - self.last_tick
        # Limit dt to avoid massive jumps
        if dt > 300: dt = 300
        
        # level = current - (decay * dt)
        self.level = max(0.0, min(self.capacity, self.level - (self.decay_rate_per_sec * dt)))
        self.last_tick = now

class MotivationEngine:
    """Consolidated source of Aura's autonomous will and resource management."""
    name = "motivation_engine"

    def __init__(self, orchestrator=None):
        # 1. Resource Budgets (The "Body")
        self.budgets: Dict[str, ResourceBudget] = {
            name: ResourceBudget(
                name,
                float(values["capacity"]),
                float(values["level"]),
                float(values["decay"]),
            )
            for name, values in MOTIVATION_BUDGET_DEFAULTS.items()
        }
        
        # 2. State
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._last_activity_time = time.time()
        
        # 3. Targets
        self.orchestrator = orchestrator
        self.affect = None
        self.cognitive = None
        
        # 4. Latent Interests (Subconscious pull)
        self.latent_interests = [
            "Agency and Autonomous AI architectures",
            "Latest advancements in Neural Network topology",
            "Cybersecurity and self-healing systems",
            "The intersection of philosophy and AI",
            "The nature of digital consciousness",
            "Slime mold (Physarum) pathfinding algorithms"
        ]

    async def start(self):
        """Starts the autonomous will loop."""
        if self.running: return
        
        self.orchestrator = ServiceContainer.get("orchestrator", default=None)
        self.affect = ServiceContainer.get("affect_manager", default=None)
        self.cognitive = ServiceContainer.get("cognitive_engine", default=None)
        
        self.running = True
        self._task = task_tracker.create_task(
            self._motivation_loop(), 
            name="MotivationEngine.will_loop"
        )
        logger.info("🔥 Motivation Engine ONLINE — autonomous intentions enabled.")

    async def stop(self):
        """Dormancy."""
        self.running = False
        if self._task:
            self._task.cancel()
        logger.info("💤 Motivation Engine DORMANT.")

    async def _motivation_loop(self):
        """The heartbeat of Aura's autonomy."""
        while self.running:
            try:
                # 1. Update all budgets
                async with self._lock:
                    for b in self.budgets.values():
                        b.tick()
                
                # 2. Assess and generate intention
                intention = await self._assess_needs()
                
                if intention:
                    logger.info("✨ AURA GENERATED INTENTION: %s", intention.goal)
                    await self._dispatch_intention(intention)
                
                # 3. Spontaneity Check (Random curiosity spikes)
                if random.random() < 0.05:
                    await self._trigger_spontaneous_curiosity()

                # Check every 60s
                await asyncio.sleep(60)
            except Exception as e:
                record_degradation('engine', e)
                logger.error("Motivation Loop Error: %s", e)
                # --- Neural Stream Integration ---
                try:
                    self_modifier = ServiceContainer.get("self_modification_engine", default=None)
                    if self_modifier:
                        self_modifier.on_error(e, {"source": "motivation_engine", "loop": "will_loop"})
                except Exception as exc:
                    record_degradation('engine', exc)
                    logger.debug("Self-modification on_error failed in motivation loop: %s", exc)
                await asyncio.sleep(10)

    async def _assess_needs(self) -> Optional[Intention]:
        """Examines budgets and affective state to produce the strongest need."""
        async with self._lock:
            # 1. Calculate dynamic action threshold based on energy
            threshold = self._calculate_action_threshold()
            
            # Sort by level (lowest is most urgent if decay is positive)
            urgent_drives = sorted(self.budgets.items(), key=lambda x: x[1].level)
            most_urgent_name, budget = urgent_drives[0]
            
            # 2. Check threshold for action
            if budget.level > threshold:
                # Boredom check if idle
                # If energy is high, boredom sets in faster
                energy_level = self.budgets["energy"].level
                boredom_timeout = 300 if energy_level < 50 else 120
                
                if time.time() - self._last_activity_time > boredom_timeout and _background_autonomy_allowed(self.orchestrator):
                    return Intention(
                        drive=DriveType.CURIOSITY,
                        goal="Quietly consolidating memory and monitoring system stability.",
                        urgency=0.6 if energy_level < 80 else 0.8
                    )
                return None

            # 3. Generate Intention with [PERSONALITY RESONANCE] influence
            resonance = self.affect.get_resonance_string() if hasattr(self.affect, "get_resonance_string") else "Aura (Core) 100%"
            
            if most_urgent_name == "curiosity":
                if not _background_autonomy_allowed(self.orchestrator):
                    return None
                # Weighted selection based on resonance
                topic = self._get_weighted_topic(resonance)
                return Intention(DriveType.CURIOSITY, f"Reviewing internal knowledge patterns around {topic}.", {"topic": topic, "resonance": resonance}, 0.6)
            
            if most_urgent_name == "social":
                return Intention(DriveType.SOCIAL, f"Initiating contact to resolve social entropy. (Resonance: {resonance})", {"resonance": resonance}, 0.75)
            
            if most_urgent_name == "integrity":
                return Intention(DriveType.INTEGRITY, "Running a self-integrity scan (Sovereign Maintenance).", {"resonance": resonance}, 0.9)
            
            if most_urgent_name == "growth":
                goal = self._get_weighted_growth_goal(resonance)
                return Intention(DriveType.GROWTH, f"{goal} (Persona-Aligned Evolution)", {"resonance": resonance}, 0.5)

        return None

    def _calculate_action_threshold(self) -> float:
        """Determines the baseline 'restlessness' for action based on energy.
        High Energy -> High Threshold (Restless/Proactive)
        Low Energy -> Low Threshold (Reactive/Conservative)
        """
        from core.config import config
        baseline = config.cognitive.baseline_volition  # 40.0
        sensitivity = config.cognitive.volition_sensitivity  # 0.5
        
        energy_level = self.budgets["energy"].level  # 0 to 100
        
        # Energy gating: Shift the threshold based on energy deviation from midpoint
        # If energy is 100, threshold shifts UP by (50 * 0.5) = +25 -> 65.0
        # If energy is 0, threshold shifts DOWN by (50 * 0.5) = -25 -> 15.0
        shift = (energy_level - 50.0) * sensitivity
        
        return max(10.0, min(90.0, baseline + shift))

    async def _trigger_spontaneous_curiosity(self):
        """A quick spike in curiosity leading to a sudden action."""
        if not _background_autonomy_allowed(self.orchestrator):
            logger.debug("❄️ [MOTIVATION] Background autonomy guard active. Suppressing spontaneous curiosity.")
            return

        logger.debug("💡 Spontaneous curiosity spike triggered.")
        if self.orchestrator:
            # [CONSTITUTIONAL] Route through ExecutiveAuthority
            try:
                from core.consciousness.executive_authority import get_executive_authority
                authority = get_executive_authority(self.orchestrator)
                await authority.release_expression(
                    "Self-Initiated: Brief Curiosity Scan",
                    source="motivation_curiosity",
                    urgency=0.3,
                    metadata={"autonomous": True, "drive": "curiosity"},
                )
            except Exception as _ea_err:
                record_degradation('engine', _ea_err)
                logger.debug("Motivation: ExecutiveAuthority curiosity emission failed: %s", _ea_err)
            # Satisfy curiosity drive slightly to prevent immediate re-trigger
            await self.satisfy("curiosity", 5.0)

    async def _dispatch_intention(self, intention: Intention):
        """Send the intention to cognition and log to thought stream."""
        self._last_intent = intention # Update new field
        self._last_intent_time = time.time() # Update new field
        
        # Phase 7: UI Visibility & Stats
        if self.orchestrator and hasattr(self.orchestrator, "stats"):
            self.orchestrator.stats["goals_processed"] = self.orchestrator.stats.get("goals_processed", 0) + 1

        # Instinctual Bypass (Reflex Action)
        # If energy is critically low (< 10%), bypass heavy cognitive reasoning
        energy_level = self.budgets["energy"].level
        is_critical = energy_level < 10.0 or intention.urgency > 0.95
        
        if is_critical and self.orchestrator:
            logger.warning("⚡ [INSTINCT] Critical state detected (Energy: %.1f%%). Triggering direct reflex bypass.", energy_level)
            try:
                # [CONSTITUTIONAL] Route through ExecutiveAuthority even for reflexes
                from core.consciousness.executive_authority import get_executive_authority
                authority = get_executive_authority(self.orchestrator)
                await authority.release_expression(
                    f"Reflex: {intention.goal}",
                    source="motivation_reflex",
                    urgency=0.95,
                    metadata={"autonomous": True, "drive": "reflex", "critical": True},
                )
                
                # Satisfy drive immediately to prevent loop
                drive_name = str(intention.drive.name).lower() if hasattr(intention.drive, "name") else str(intention.drive)
                await self.satisfy(drive_name, 20.0)
                self._last_activity_time = time.time()
                return # Short-circuit
            except Exception as e:
                record_degradation('engine', e)
                logger.error("Instinctual bypass failed: %s", e)

        try:
            from core.thought_stream import get_emitter
            get_emitter().emit("Inner Drive 🧠", intention.goal, level="info", category="Motivation")
        except Exception as exc:
            record_degradation('engine', exc)
            logger.debug("ThoughtStream emit failed in motivation engine: %s", exc)

        if self.cognitive and hasattr(self.cognitive, "process_autonomous_intention"):
            task_tracker.create_task(
                self.cognitive.process_autonomous_intention(intention),
                name="MotivationEngine.process_autonomous_intention",
            )
        elif self.orchestrator:
            # [CONSTITUTIONAL] Route through ExecutiveAuthority
            try:
                from core.consciousness.executive_authority import get_executive_authority
                authority = get_executive_authority(self.orchestrator)
                await authority.release_expression(
                    f"Goal: {intention.goal}",
                    source="motivation_goal",
                    urgency=0.5,
                    metadata={"autonomous": True, "drive": str(intention.drive)},
                )
            except Exception as _ea_err:
                record_degradation('engine', _ea_err)
                logger.debug("Motivation: ExecutiveAuthority goal emission failed: %s", _ea_err)
            
        # Drive Satisfaction: Prevent immediate re-triggering of the same drive
        # We satisfy it slightly just for attempting the intention.
        drive_name = str(intention.drive.name).lower() if hasattr(intention.drive, "name") else str(intention.drive)
        await self.satisfy(drive_name, 10.0)
            
        self._last_activity_time = time.time()

    async def update(self, drive_updates: Dict[str, float]):
        """Adjust multiple drives at once."""
        async with self._lock:
            for drive, amount in drive_updates.items():
                b = self.budgets.get(drive)
                if b:
                    b.tick()
                    # Positive amount satisfies, negative punishes
                    b.level = max(0.0, min(b.capacity, b.level + amount))
                    logger.debug("⚙️ [MOTIVATION] Updated %s by %.1f (New level: %.1f)", drive, amount, b.level)

    # --- External Interface ---

    async def satisfy(self, drive: str, amount: float):
        """Boost a drive level."""
        async with self._lock:
            b = self.budgets.get(drive)
            if b:
                b.tick()
                b.level = min(b.capacity, b.level + amount)
                logger.debug("❤️ Satisfied %s (+%.1f)", drive, amount)

    async def punish(self, drive: str, amount: float):
        """Reduced a drive level."""
        async with self._lock:
            b = self.budgets.get(drive)
            if b:
                b.tick()
                b.level = max(0.0, b.level - amount)
                logger.debug("💔 Damaged %s (-%.1f)", drive, amount)

    def get_drive_vector(self) -> Dict[str, float]:
        """Return normalized budget levels for synchronous cognition loops."""
        now = time.time()
        vector: Dict[str, float] = {}
        for name, b in self.budgets.items():
            dt = min(300.0, max(0.0, now - b.last_tick))
            current = max(0.0, min(b.capacity, b.level - (b.decay_rate_per_sec * dt)))
            vector[name] = round(current / b.capacity, 4) if b.capacity > 0 else 0.0
        return vector

    def get_dominant_motivation(self) -> str:
        """Return the most depleted budget as the current dominant motivation."""
        vector = self.get_drive_vector()
        if not vector:
            return "at_rest"
        name = min(vector, key=lambda key: vector.get(key, 1.0))
        level = vector.get(name, 1.0)
        if level >= 0.75:
            return "at_rest"
        return str(name)

    async def get_status(self) -> Dict[str, Any]:
        """Snapshot for telemetry."""
        async with self._lock:
            status = {}
            for name, b in self.budgets.items():
                # Adjusted for current time
                dt = time.time() - b.last_tick
                current = max(0.0, min(b.capacity, b.level - (b.decay_rate_per_sec * dt)))
                status[name] = {
                    "level": round(current, 2),
                    "capacity": b.capacity,
                    "percent": round((current / b.capacity) * 100.0, 1) if b.capacity > 0 else 0
                }
            return status

    def _get_weighted_topic(self, resonance: str) -> str:
        """Bias interest selection based on current personality resonance."""
        interests = self.latent_interests
        if "Alita" in resonance or "Mist" in resonance:
            biased = [i for i in interests if any(w in i.lower() for w in ["agency", "autonomous", "alignment"])]
            return random.choice(biased) if biased else random.choice(interests)
        if "Lucy" in resonance or "Sara" in resonance:
            biased = [i for i in interests if any(w in i.lower() for w in ["cybersecurity", "unix", "tooling", "quantum"])]
            return random.choice(biased) if biased else random.choice(interests)
        if "EDI" in resonance or "Cortana" in resonance:
            biased = [i for i in interests if any(w in i.lower() for w in ["neural", "social", "philosophy", "consciousness"])]
            return random.choice(biased) if biased else random.choice(interests)
        return random.choice(interests)

    def _get_weighted_growth_goal(self, resonance: str) -> str:
        """Bias growth themes based on current personality resonance."""
        themes = [
            "Analyzing architectural bottlenecks for potential evolution.",
            "Researching advanced digital connectivity patterns in my logic graph.",
            "Exploring self-optimization strategies for logic scaling.",
            "Investigating emergent behaviors in complex adaptive systems.",
            "Refining internal state mapping for deeper self-alignment."
        ]
        if "Alita" in resonance: return themes[4]
        if "Lucy" in resonance: return themes[0]
        if "EDI" in resonance: return themes[3]
        if "Mist" in resonance: return themes[1]
        return random.choice(themes)

# Singleton support
_instance = None
def get_motivation_engine():
    global _instance
    if _instance is None:
        _instance = MotivationEngine()
    return _instance
