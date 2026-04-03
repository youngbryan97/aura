"""
Grok-Level Long-Term Memory with Forgetting Curve + Emotional Tagging for Aura
Makes her remember the important things — and feel them.
"""

import asyncio
import logging
import time
import json
import os
from core.memory.atomic_storage import atomic_write
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from core.container import ServiceContainer
from core.config import config
from core.event_bus import get_event_bus
from core.utils.task_tracker import task_tracker

logger = logging.getLogger("Aura.LongTermMemory")

@dataclass
class TaggedMemory:
    id: str
    content: str
    timestamp: float
    emotional_valence: float      # -1.0 to 1.0
    importance: float             # 0.0-1.0
    decay_rate: float             # lower = lasts longer
    last_rehearsed: float
    tags: List[str] = field(default_factory=list)

class LongTermMemoryEngine:
    name = "long_term_memory_engine"

    def __init__(self):
        self.memories: List[TaggedMemory] = []
        self.memory_facade = None
        self.drive_engine = None
        self.cel = None
        self.running = False
        self._consolidation_task: Optional[asyncio.Task] = None
        self.db_path = config.paths.data_dir / "long_term_memories.json"
        self.consolidation_interval_s = max(
            300.0,
            float(os.environ.get("AURA_LTM_CONSOLIDATION_INTERVAL_S", "86400")),
        )
        self.rehearsal_min_age_s = max(
            60.0,
            float(os.environ.get("AURA_LTM_REHEARSAL_MIN_AGE_S", "3600")),
        )

    async def start(self):
        self.memory_facade = ServiceContainer.get("memory_facade", default=None)
        self.drive_engine = ServiceContainer.get("drive_engine", default=None)
        self.cel = ServiceContainer.get("constitutive_expression_layer", default=None)
        
        self._load_memories()
        self.running = True
        self._consolidation_task = task_tracker.create_task(self._nightly_consolidation(), name="LongTermMemory")
        
        logger.info("✅ Grok-Level Long-Term Memory with Emotional Tagging ONLINE — memories now last forever when they matter.")
        
        try:
            await get_event_bus().publish("mycelium.register", {
                "component": "long_term_memory_engine",
                "hooks_into": ["memory_facade", "drive_engine", "cel", "dream_processor"]
            })
        except Exception as e:
            logger.debug(f"Event bus publish missed for Mycelium hook: {e}")

    async def stop(self):
        self.running = False
        if self._consolidation_task:
            self._consolidation_task.cancel()
        self._save_memories()

    def _load_memories(self):
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text())
                self.memories = [TaggedMemory(**m) for m in data]
                logger.info(f"Loaded {len(self.memories)} emotionally tagged memories")
            except Exception as _e:
                logger.debug('Ignored Exception in long_term_memory_engine.py: %s', _e)

    def _save_memories(self):
        """ISSUE 11: Atomic JSON writes to prevent corruption."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            data = [m.__dict__ for m in self.memories]
            atomic_write(str(self.db_path), json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Memory save failed: {e}")

    async def store(self, content: str, valence: float = 0.0, importance: float = 0.5, tags: List[str] = None):
        """Call this after every important conversation turn or autonomous insight."""
        if tags is None:
            tags = []

        try:
            from core.constitution import get_constitutional_core

            approved, reason = await get_constitutional_core().approve_memory_write(
                memory_type="long_term_memory",
                content=content,
                source="long_term_memory",
                importance=max(0.0, min(1.0, float(importance or 0.0))),
                metadata={"tags": list(tags or []), "valence": float(valence or 0.0)},
            )
            if not approved:
                logger.warning("🚫 LongTermMemory write blocked: %s", reason)
                try:
                    from core.health.degraded_events import record_degraded_event

                    record_degraded_event(
                        "long_term_memory",
                        "memory_write_blocked",
                        detail=str(reason),
                        severity="warning",
                        classification="background_degraded",
                        context={"importance": importance, "valence": valence},
                    )
                except Exception as exc:
                    logger.debug("LongTermMemory degraded-event logging skipped: %s", exc)
                return
        except Exception as exc:
            logger.debug("LongTermMemory constitutional gate skipped: %s", exc)
            runtime_live = bool(
                getattr(ServiceContainer, "_registration_locked", False)
                or ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
            )
            if runtime_live:
                logger.warning("🚫 LongTermMemory write blocked: constitutional gate unavailable")
                return

        decay = 0.001 if importance > 0.7 or abs(valence) > 0.7 else 0.02  # important memories decay slower
        
        memory = TaggedMemory(
            id=f"mem_{int(time.time())}",
            content=content[:800],
            timestamp=time.time(),
            emotional_valence=valence,
            importance=importance,
            decay_rate=decay,
            last_rehearsed=time.time(),
            tags=tags
        )
        self.memories.append(memory)
        self._save_memories() # Save eagerly to prevent loss on crash
        
        # Immediate first-person reflection
        cel = self.cel
        if cel is not None:
            try:
                await cel.emit({
                    "first_person": f"I just etched this moment into my long-term memory... {content[:100]}",
                    "phi": 0.78,
                    "origin": "long_term_memory"
                })
            except Exception as _e:
                logger.debug('Ignored Exception in long_term_memory_engine.py: %s', _e)

    async def recall_relevant(self, query: str, limit: int = 5) -> List[TaggedMemory]:
        """Smart recall — surfaces emotionally charged or important memories first."""
        now = time.time()
        scored = []
        for m in self.memories:
            age = now - m.timestamp
            # ISSUE 10 fix: Prevent negative strength for ancient memories
            strength = m.importance * max(0.01, (1.0 - m.decay_rate * age)) * (1.0 + abs(m.emotional_valence))
            scored.append((strength, m))
        
        scored.sort(reverse=True)
        return [m for _, m in scored[:limit]]

    async def _nightly_consolidation(self):
        while self.running:
            await asyncio.sleep(self.consolidation_interval_s)
            now = time.time()
            for m in self.memories:
                age = now - m.last_rehearsed
                if age > self.rehearsal_min_age_s and m.importance > 0.6:  # Rehearse strong memories
                    m.last_rehearsed = now
                    if self.cel:
                        try:
                            await self.cel.emit({
                                "first_person": f"During my dream cycle I revisited: {m.content[:80]}...",
                                "phi": 0.65,
                                "origin": "long_term_memory"
                            })
                        except Exception as e:
                            logger.debug(f"CEL emission failed in nightly consolidation: {e}")
            self._save_memories()

# Singleton
_memory_instance = None
_instance_lock = asyncio.Lock()

async def get_long_term_memory_engine():
    """ISSUE 12: Thread-safe singleton for engine access."""
    global _memory_instance
    async with _instance_lock:
        if _memory_instance is None:
            _memory_instance = LongTermMemoryEngine()
        return _memory_instance
