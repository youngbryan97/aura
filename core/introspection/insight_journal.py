"""core/introspection/insight_journal.py — Aura InsightJournal v1.0
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

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.config import config
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.executors import off_the_loop

logger = logging.getLogger("Aura.InsightJournal")


def _record_insight_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    classification: FallbackClassification = FallbackClassification.SAFE_FALLBACK,
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "insight_journal",
        error,
        severity=severity,
        action=action,
        classification=classification,
        receipt_required=severity in {"degraded", "critical"},
        extra=extra,
    )


def _clamp(value: Any, *, low: float, high: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, numeric))


def _default_db_path() -> Path:
    return config.paths.data_dir / "insight_journal.json"


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
    tags: list[str] = field(default_factory=list)
    impact_score: float = 0.5   # 0.0-1.0 how much this changed Aura's mind
    meta_cognition: str = ""    # "I realized X because I was previously stuck on Y"


class InsightJournal:
    """
    The permanent record of Aura's intellectual evolution.
    """
    name = "insight_journal"

    def __init__(self):
        self._insights: list[Insight] = []
        self._db_path = _default_db_path()
        self._load()
        logger.info("InsightJournal constructed (%d insights).", len(self._insights))

    async def start(self):
        try:
            from core.event_bus import get_event_bus
            await get_event_bus().publish("mycelium.register", {
                "component": "insight_journal",
                "hooks_into": ["cognitive_kernel", "inquiry_engine", "concept_linker"]
            })
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_insight_degradation(
                exc,
                action="continued online after mycelium registration failed",
                classification=FallbackClassification.AUDIT_GAP,
                extra={"component": "insight_journal"},
            )
            logger.debug("InsightJournal mycelium registration failed: %s", exc)
        logger.info("✅ InsightJournal ONLINE — chronicling the journey.")

    async def stop(self):
        await off_the_loop(self._save)

    async def record_insight(self, title: str, content: str, domain: str,
                             confidence: float, source: str, tags: list[str] | None = None,
                             meta_cognition: str = ""):
        """Add a new insight to the journal."""
        safe_domain = str(domain or "general")[:120]
        safe_confidence = _clamp(confidence, low=0.0, high=1.0, default=0.5)
        insight = Insight(
            id=str(uuid.uuid4())[:8],
            title=str(title or "Untitled insight")[:200],
            content=str(content or "")[:5000],
            domain=safe_domain,
            confidence=safe_confidence,
            timestamp=time.time(),
            source=str(source or "unknown")[:80],
            tags=[str(tag)[:80] for tag in (tags or [safe_domain]) if str(tag or "").strip()][:20],
            meta_cognition=str(meta_cognition or "")[:1200],
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
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_insight_degradation(
                exc,
                action="kept insight durably after event-bus broadcast failed",
                classification=FallbackClassification.AUDIT_GAP,
                extra={"insight_id": insight.id, "domain": insight.domain},
            )
            logger.debug("InsightJournal broadcast failed: %s", exc)

        # High-confidence insights should become beliefs.
        if safe_confidence >= 0.75:
            try:
                from core.container import ServiceContainer
                beliefs = ServiceContainer.get("belief_revision_engine", default=None)
                if beliefs:
                    await beliefs.process_new_claim(
                        content=insight.content,
                        confidence=safe_confidence,
                        domain=safe_domain,
                        source=f"insight:{insight.source}"
                    )
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _record_insight_degradation(
                    exc,
                    action="kept insight durably after belief promotion failed",
                    extra={"insight_id": insight.id, "domain": insight.domain},
                )
                logger.debug("InsightJournal belief promotion failed: %s", exc)

    def get_recent_insights(self, limit: int = 5) -> list[Insight]:
        """Return the most recent discoveries."""
        return sorted(self._insights, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_highest_confidence_insights(self, limit: int = 5) -> list[Insight]:
        """Return the insights with the highest confidence scores."""
        return sorted(self._insights, key=lambda x: x.confidence, reverse=True)[:limit]

    def get_insights_by_domain(self, domain: str) -> list[Insight]:
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

    def _restore_insight(self, payload: Any) -> Insight:
        if not isinstance(payload, dict):
            raise TypeError("insight payload must be an object")
        title = str(payload.get("title", "") or "").strip()
        content = str(payload.get("content", "") or "").strip()
        if not title or not content:
            raise ValueError("insight title and content are required")
        tags = payload.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        return Insight(
            id=str(payload.get("id") or str(uuid.uuid4())[:8])[:80],
            title=title[:200],
            content=content[:5000],
            domain=str(payload.get("domain", "general") or "general")[:120],
            confidence=_clamp(payload.get("confidence", 0.5), low=0.0, high=1.0, default=0.5),
            timestamp=_clamp(payload.get("timestamp", time.time()), low=0.0, high=time.time(), default=time.time()),
            source=str(payload.get("source", "unknown") or "unknown")[:80],
            tags=[str(tag)[:80] for tag in tags[:20] if str(tag or "").strip()],
            impact_score=_clamp(payload.get("impact_score", 0.5), low=0.0, high=1.0, default=0.5),
            meta_cognition=str(payload.get("meta_cognition", "") or "")[:1200],
        )

    def _quarantine_corrupt_store(self) -> None:
        quarantine_path = self._db_path.with_name(
            f"{self._db_path.stem}.corrupt-{int(time.time())}{self._db_path.suffix}"
        )
        try:
            self._db_path.replace(quarantine_path)
        except OSError as exc:
            _record_insight_degradation(
                exc,
                action="left corrupt insight journal in place after quarantine move failed",
                severity="degraded",
                classification=FallbackClassification.AUDIT_GAP,
                extra={"path": str(self._db_path)},
            )

    def _save(self):
        try:
            data = [asdict(i) for i in self._insights]
            atomic_write_text(self._db_path, json.dumps(data, indent=2))
        except (OSError, TypeError, ValueError) as e:
            _record_insight_degradation(
                e,
                action="kept insights in memory after durable save failed",
                severity="degraded",
                extra={"path": str(self._db_path), "insight_count": len(self._insights)},
            )
            logger.debug("InsightJournal save failed: %s", e)

    def _load(self):
        if not self._db_path.exists():
            return
        try:
            raw = self._db_path.read_text(encoding="utf-8")
        except OSError as e:
            _record_insight_degradation(
                e,
                action="started with empty insight journal after durable load failed",
                severity="degraded",
                extra={"path": str(self._db_path)},
            )
            logger.debug("InsightJournal load failed: %s", e)
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            _record_insight_degradation(
                e,
                action="started with empty insight journal and quarantined corrupt store",
                severity="degraded",
                classification=FallbackClassification.AUDIT_GAP,
                extra={"path": str(self._db_path)},
            )
            self._quarantine_corrupt_store()
            return
        if not isinstance(data, list):
            _record_insight_degradation(
                TypeError("insight journal root must be a list"),
                action="started with empty insight journal after invalid store root",
                severity="degraded",
                classification=FallbackClassification.AUDIT_GAP,
                extra={"path": str(self._db_path)},
            )
            return
        restored: list[Insight] = []
        for item in data:
            try:
                restored.append(self._restore_insight(item))
            except (TypeError, ValueError) as exc:
                _record_insight_degradation(
                    exc,
                    action="skipped invalid insight while loading durable store",
                    extra={"path": str(self._db_path)},
                )
        self._insights = restored[-500:]

    def get_status(self) -> dict[str, Any]:
        return {
            "total_insights": len(self._insights),
            "domains": sorted({insight.domain for insight in self._insights}),
        }
