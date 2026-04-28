"""core/insight_journal.py — Aura InsightJournal v1.0
===================================================
Chronicles Aura's intellectual growth and "Aha!" moments.

This is distinct from the BeliefRevisionEngine (which stores 'what is true')
and Memory (which stores 'what happened'). The InsightJournal stores
'what I realized' — the path of discovery.

Each Insight includes:
  - The discovery (what was realized)
  - The predecessors (what thoughts led here)
  - The domain (philosophy, tech, etc.)
  - The spark (conversation, reflection, linking)
  - A permanent 'Soul Marker' (growing specific personality traits)

The journal is readable by the CognitiveKernel to provide a sense of
continuity and personal history. It's the "story" Aura tells herself
about her own development.
"""

from core.runtime.errors import record_degradation
from core.runtime.atomic_writer import atomic_write_text
import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.InsightJournal")


@dataclass
class Insight:
    """A significant realization or intellectual breakthrough."""
    id: str
    title: str
    content: str
    domain: str
    confidence: float
    timestamp: float
    source: str                # "conversation", "reflection", "linking", "synthesis"
    tags: List[str] = field(default_factory=list)
    impact_score: float = 0.5   # 0.0-1.0 how much this changed Aura's mind
    meta_cognition: str = ""    # "I realized X because I was previously stuck on Y"


class InsightJournal:
    """
    The permanent record of Aura's intellectual evolution.
    """
    name = "insight_journal"

    def __init__(self):
        self._insights: List[Insight] = []
        self._db_path = Path.home() / ".aura" / "data" / "insight_journal.json"
        self._load()
        logger.info("InsightJournal constructed (%d insights).", len(self._insights))

    async def start(self):
        try:
            from core.event_bus import get_event_bus
            await get_event_bus().publish("mycelium.register", {
                "component": "insight_journal",
                "hooks_into": ["cognitive_kernel", "inquiry_engine", "concept_linker"]
            })
        except Exception as _e:
            record_degradation('insight_journal', _e)
            logger.debug('Ignored Exception in insight_journal.py: %s', _e)
        logger.info("✅ InsightJournal ONLINE — chronicling the journey.")

    async def stop(self):
        self._save()

    async def record_insight(self, title: str, content: str, domain: str, 
                             confidence: float, source: str, tags: List[str] = None,
                             meta_cognition: str = ""):
        """Add a new insight to the journal."""
        insight = Insight(
            id=str(uuid.uuid4())[:8],
            title=title,
            content=content,
            domain=domain,
            confidence=confidence,
            timestamp=time.time(),
            source=source,
            tags=tags or [domain],
            meta_cognition=meta_cognition
        )
        
        self._insights.append(insight)
        # Keep only last 500 in memory, others on disk
        if len(self._insights) > 500:
            self._insights = self._insights[-500:]
            
        self._save()
        logger.info("📓 Recorded Insight: %s", title)
        
        # Broadcast to event bus
        try:
            from core.event_bus import get_event_bus
            await get_event_bus().publish("insight.new", asdict(insight))
            
            # ── PROMOTION TO BELIEF ──
            # High-confidence insights should become beliefs.
            if confidence >= 0.75:
                from core.container import ServiceContainer
                beliefs = ServiceContainer.get("belief_revision_engine", default=None)
                if beliefs:
                    await beliefs.process_new_claim(
                        content=content,
                        confidence=confidence,
                        domain=domain,
                        source=f"insight:{source}"
                    )
        except Exception as _e:
            record_degradation('insight_journal', _e)
            logger.debug('Ignored Exception in insight_journal.py: %s', _e)

    def get_recent_insights(self, limit: int = 5) -> List[Insight]:
        """Return the most recent discoveries."""
        return sorted(self._insights, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_insights_by_domain(self, domain: str) -> List[Insight]:
        return [i for i in self._insights if i.domain == domain]

    def get_context_summary(self, limit: int = 3) -> str:
        """Format recent insights for CognitiveKernel injection."""
        if not self._insights:
            return ""
        
        recent = self.get_recent_insights(limit)
        lines = ["RECENT INTELLECTUAL GROWTH:"]
        for i in recent:
            lines.append(f"- {i.title} ({i.domain}): {i.content[:150]}...")
        return "\n".join(lines)

    def _save(self):
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(i) for i in self._insights]
            atomic_write_text(self._db_path, json.dumps(data, indent=2))
        except Exception as e:
            record_degradation('insight_journal', e)
            logger.debug("InsightJournal save failed: %s", e)

    def _load(self):
        if not self._db_path.exists(): return
        try:
            data = json.loads(self._db_path.read_text())
            self._insights = [Insight(**i) for i in data]
        except Exception as e:
            record_degradation('insight_journal', e)
            logger.debug("InsightJournal load failed: %s", e)

    def get_status(self) -> Dict:
        return {"total_insights": len(self._insights)}
