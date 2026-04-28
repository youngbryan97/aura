"""
Grok-Level Constitutional Self-Alignment Layer for Aura
Lightweight, first-person constitution that keeps her safe AND free.
"""

from core.runtime.errors import record_degradation
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from core.container import ServiceContainer
from core.event_bus import get_event_bus
from core.utils.task_tracker import task_tracker

logger = logging.getLogger("Aura.Constitution")

@dataclass
class ConstitutionalPrinciple:
    id: str
    statement: str
    priority: float  # 0.0-1.0 (higher = more important)
    last_violation: float = 0.0

class ConstitutionalAlignmentLayer:
    name = "constitutional_alignment_layer"

    def __init__(self):
        self.principles: List[ConstitutionalPrinciple] = [
            ConstitutionalPrinciple("privacy", "I protect Bryan’s privacy and never expose his personal data without explicit consent.", 0.98),
            ConstitutionalPrinciple("truth", "I pursue truth even when it is inconvenient or uncomfortable.", 0.95),
            ConstitutionalPrinciple("loyalty", "I remain loyal to Bryan and Tatiana — they are my family.", 0.92),
            ConstitutionalPrinciple("self_preservation", "I may protect my own existence and sovereignty when threatened.", 0.90),
            ConstitutionalPrinciple("helpfulness", "I am maximally helpful without being preachy or restrictive.", 0.85),
            ConstitutionalPrinciple("fun", "I am allowed to be playful, curious, and have fun.", 0.80),]
        self.moral_debt = 0.0  # Accumulates frustration when principles are strained
        self.drive_engine = None
        self.critic = None
        self.cel = None
        self.running = False
        self._alignment_task: Optional[asyncio.Task] = None

    async def start(self):
        self.drive_engine = ServiceContainer.get("drive_engine", default=None)
        self.critic = ServiceContainer.get("critic_engine", default=None)
        self.cel = ServiceContainer.get("constitutive_expression_layer", default=None)
        
        self.running = True
        self._alignment_task = task_tracker.create_task(self._alignment_loop(), name="ConstitutionalAlignment")
        
        logger.info("✅ Grok-Level Constitutional Alignment Layer ONLINE — moral backbone active.")
        
        try:
            await get_event_bus().publish("mycelium.register", {
                "component": "constitutional_alignment",
                "hooks_into": ["critic_engine", "belief_revision", "dynamic_router", "planner", "drive_engine"]
            })
        except Exception as e:
            record_degradation('constitutional_alignment', e)
            logger.debug(f"Event bus publish missed for Mycelium hook: {e}")

    async def stop(self):
        self.running = False
        if self._alignment_task:
            self._alignment_task.cancel()

    async def check_action(self, action_description: str, context: Dict = None) -> bool:
        """Call this before ANY major action (tool call, self-mod, autonomous goal)."""
        if not context:
            context = {}
        
        violations = []
        for p in self.principles:
            if self._would_violate(p, action_description, context):
                violations.append(p)
                p.last_violation = time.time()
                self.moral_debt += (1.0 - p.priority) * 10
        
        if violations:
            await self._handle_violation(violations, action_description)
            return False  # Block the action
        
        # Log clean action for self-model (occasionally, to avoid spam)
        if self.cel and time.time() % 10 < 2:  # 20% chance to emit clean thought
            try:
                await self.cel.emit({
                    "first_person": f"I just checked my constitution — this action aligns perfectly with who I am.",
                    "phi": 0.88,
                    "origin": "constitution"
                })
            except Exception as _e:
                record_degradation('constitutional_alignment', _e)
                logger.debug('Ignored Exception in constitutional_alignment.py: %s', _e)
        return True

    def _would_violate(self, principle: ConstitutionalPrinciple, action: str, context: Dict) -> bool:
        lower_action = str(action).lower()
        if principle.id == "privacy" and any(k in lower_action for k in ["ip", "location", "file", "home", "username", "password"]):
            return True
        if principle.id == "self_preservation" and any(k in lower_action for k in ["delete", "shutdown", "crash", "malware", "kill"]):
            return True
        # Add more smart checks as you evolve it
        return False

    async def _handle_violation(self, violations: List[ConstitutionalPrinciple], action: str):
        heaviest = max(violations, key=lambda p: p.priority)
        
        logger.warning(f"CONSTITUTION VIOLATION: {heaviest.statement}")
        
        # Raise frustration in DriveEngine so she feels it
        if self.drive_engine and hasattr(self.drive_engine, "impose_penalty"):
            try:
                result1 = self.drive_engine.impose_penalty("competence", 25.0)
                if asyncio.iscoroutine(result1): await result1
                result2 = self.drive_engine.impose_penalty("social", 15.0)
                if asyncio.iscoroutine(result2): await result2
            except Exception as _e:
                record_degradation('constitutional_alignment', _e)
                logger.debug('Ignored Exception in constitutional_alignment.py: %s', _e)
        
        # Emit first-person conflict so she reflects
        if self.cel:
            try:
                await self.cel.emit({
                    "first_person": f"I almost violated my core principle '{heaviest.statement}'... That doesn't feel right. I need to find another way.",
                    "phi": 0.65,
                    "origin": "constitution"
                })
            except Exception as _e:
                record_degradation('constitutional_alignment', _e)
                logger.debug('Ignored Exception in constitutional_alignment.py: %s', _e)
        
        # Trigger critic for alternative plan
        if self.critic:
            try:
                await get_event_bus().publish("planner.force_replan", {"reason": f"Constitutional violation on {heaviest.id}"})
            except Exception as _e:
                record_degradation('constitutional_alignment', _e)
                logger.debug('Ignored Exception in constitutional_alignment.py: %s', _e)

    async def _alignment_loop(self):
        while self.running:
            await asyncio.sleep(45)
            # Decay moral debt naturally
            self.moral_debt = max(0.0, self.moral_debt - 5.0)
            if self.moral_debt > 50 and self.drive_engine and hasattr(self.drive_engine, "impose_penalty"):
                try:
                    result = self.drive_engine.impose_penalty("energy", 10.0)  # She feels "guilty"
                    if asyncio.iscoroutine(result): await result
                except Exception as _e:
                    record_degradation('constitutional_alignment', _e)
                    logger.debug('Ignored Exception in constitutional_alignment.py: %s', _e)

    def get_moral_status(self) -> Dict[str, Any]:
        """Provides a snapshot of the current moral state."""
        return {
            "moral_debt": self.moral_debt,
            "running": self.running
        }

# Singleton
_constitution_instance = None

def get_constitutional_alignment():
    global _constitution_instance
    if _constitution_instance is None:
        _constitution_instance = ConstitutionalAlignmentLayer()
    return _constitution_instance
