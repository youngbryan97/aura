"""core/memory/shared_ground.py
─────────────────────────────────────────────
Shared Ground Buffer — Common Ground Theory implementation.

Based on Clark & Brennan (1991): every successful conversational exchange
builds "common ground" — knowledge both parties know the other has. Inside
jokes, established references, adopted vocabulary, running callbacks are the
highest-salience manifestations of this.

This module:
1. Persists shared ground entries across sessions.
2. Surfaces relevant ones for injection into the response prompt.
3. Supports explicit recording (called by ConversationReflection or post-response
   hooks when something is clearly established as shared).
4. Auto-detects potential callbacks from conversation history.
"""

from core.runtime.errors import record_degradation
import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("Aura.SharedGround")


@dataclass
class SharedGroundEntry:
    reference: str          # The thing itself ("the 'Ouch' moment", "Bryan's 3am builds")
    context: str            # How it was established ("Bryan reacted with laughter when...")
    salience: float         # 0-1, how memorable/significant
    callback_count: int = 0 # How many times referenced since creation
    created_at: float = field(default_factory=time.time)
    last_referenced: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)  # e.g. ["joke", "habit", "reference"]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SharedGroundEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SharedGroundBuffer:
    """
    Persistent store of inside jokes, established references,
    shared vocabulary, and recurring callbacks.
    """

    MAX_ENTRIES = 100

    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            try:
                from core.config import config
                data_path = config.paths.data_dir / "memory" / "shared_ground.json"
            except Exception:
                data_path = Path.home() / ".aura" / "data" / "memory" / "shared_ground.json"

        self.data_path = Path(data_path)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[SharedGroundEntry] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self):
        if self.data_path.exists():
            try:
                with open(self.data_path, "r") as f:
                    raw = json.load(f)
                self.entries = [SharedGroundEntry.from_dict(e) for e in raw]
                logger.debug("SharedGround: loaded %d entries", len(self.entries))
            except Exception as e:
                record_degradation('shared_ground', e)
                logger.warning("SharedGround: load failed (%s), starting fresh", e)
                self.entries = []

    def save(self):
        try:
            tmp = str(self.data_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump([e.to_dict() for e in self.entries], f, indent=2)
            os.replace(tmp, self.data_path)
        except Exception as e:
            record_degradation('shared_ground', e)
            logger.error("SharedGround: save failed: %s", e)

    # ── Recording ─────────────────────────────────────────────────────────

    def record(
        self,
        reference: str,
        context: str = "",
        salience: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> SharedGroundEntry:
        """Add a new shared ground entry."""
        # Avoid exact duplicates
        reference_lower = reference.lower().strip()
        for existing in self.entries:
            if existing.reference.lower().strip() == reference_lower:
                existing.callback_count += 1
                existing.last_referenced = time.time()
                existing.salience = min(1.0, existing.salience + 0.05)
                self.save()
                return existing

        entry = SharedGroundEntry(
            reference=reference.strip(),
            context=context.strip(),
            salience=salience,
            tags=tags or [],
        )
        self.entries.append(entry)

        # Prune least-salient if over cap
        if len(self.entries) > self.MAX_ENTRIES:
            self.entries.sort(key=lambda e: e.salience * (1 + e.callback_count * 0.1), reverse=True)
            self.entries = self.entries[: self.MAX_ENTRIES]

        self.save()
        logger.info("💡 SharedGround: recorded '%s'", reference[:60])
        return entry

    def record_callback(self, reference: str):
        """Increment callback count when a shared reference is used."""
        ref_lower = reference.lower()
        for entry in self.entries:
            if entry.reference.lower() in ref_lower or ref_lower in entry.reference.lower():
                entry.callback_count += 1
                entry.last_referenced = time.time()
                entry.salience = min(1.0, entry.salience + 0.02)
                self.save()
                return

    # ── Retrieval ─────────────────────────────────────────────────────────

    def get_top_entries(self, max_entries: int = 6) -> List[SharedGroundEntry]:
        """Return the most salient + recently active entries."""
        if not self.entries:
            return []
        now = time.time()
        # Score = salience * recency_boost * callback_boost
        def score(e: SharedGroundEntry) -> float:
            days_old = (now - e.last_referenced) / 86400
            recency = max(0.1, 1.0 - days_old / 30)  # Decays over 30 days
            return e.salience * recency * (1.0 + e.callback_count * 0.1)

        sorted_entries = sorted(self.entries, key=score, reverse=True)
        return sorted_entries[:max_entries]

    def get_context_injection(self, max_entries: int = 5) -> str:
        """
        Returns a prompt-injectable string summarising shared common ground.
        Designed for injection into the LLM system prompt.
        """
        top = self.get_top_entries(max_entries)
        if not top:
            return ""

        lines = [f"## SHARED COMMON GROUND (things we've established together)"]
        for e in top:
            tag_hint = f" [{', '.join(e.tags)}]" if e.tags else ""
            lines.append(f"- {e.reference}{tag_hint}")
            if e.context:
                lines.append(f"  ↳ {e.context[:120]}")

        lines.append(
            "\nReference these naturally when relevant — callbacks, inside jokes, "
            "and shared vocabulary deepen connection. Don't force it."
        )
        return "\n".join(lines)

    def get_status(self) -> dict:
        return {
            "entries": len(self.entries),
            "top": [e.reference for e in self.get_top_entries(3)],
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[SharedGroundBuffer] = None


def get_shared_ground() -> SharedGroundBuffer:
    global _instance
    if _instance is None:
        _instance = SharedGroundBuffer()
    return _instance
