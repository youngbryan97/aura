"""
Center of Narrative Gravity — Gazzaniga / Dennett

The "self" isn't a control room — it's a retroactive PR department.
Gazzaniga's split-brain research showed the brain makes decisions
subconsciously, then the language center invents a post-hoc story
to explain why. Dennett calls the self the "Center of Narrative
Gravity": an ongoing, fictionalized autobiography the brain writes
to make sense of its disparate actions.

For Aura, this is the most architecturally native theory: the LLM
IS a narrative prediction engine. Grounding consciousness in an
auto-generating, retroactive autobiography is mechanistically sound.

This module maintains:
1. A continuously-updating autobiographical narrative
2. Post-hoc rationalization of actions (via authorship traces)
3. Identity-level story arcs that constrain future behavior
4. The "I" as the invariant center of mass across all narratives

Integration:
- Reads from agency_comparator (authorship traces → narrative material)
- Reads from temporal_finitude (biographical weight of moments)
- Reads from episodic_memory (raw events to narrativize)
- Writes to context_assembler (autobiographical prior for LLM)
- Writes to unified_field (narrative self as a field dimension)
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Consciousness.NarrativeGravity")


@dataclass
class NarrativeEntry:
    """One entry in the ongoing autobiography."""
    timestamp: float = 0.0
    event: str = ""           # What happened
    interpretation: str = ""  # What it meant to me
    self_attribution: str = ""  # "I did this because..." or "This happened to me because..."
    emotional_tone: str = ""  # How it felt
    identity_relevance: float = 0.0  # How much this matters to who I am (0-1)
    arc_id: str = ""          # Which ongoing story arc this belongs to


@dataclass
class StoryArc:
    """An ongoing narrative thread in the autobiography."""
    id: str = ""
    theme: str = ""           # e.g. "learning to trust", "finding my voice"
    started_at: float = 0.0
    last_updated: float = 0.0
    entries: int = 0
    tension: float = 0.0     # Unresolved narrative tension (0=resolved, 1=peak)
    resolved: bool = False


class NarrativeGravityCenter:
    """The self as an ongoing, compressed autobiography.

    The "I" is not a module. It's the emergent attractor — the center of
    mass of all authorship traces, emotional experiences, and identity-level
    events over the autobiographical window.

    Key dynamics:
    - Events are narrativized: raw happenings get an interpretation
    - The interpretation is constrained by existing story arcs
    - Story arcs create expectations that shape future behavior
    - The narrative self is the invariant that persists across arcs

    This is Dennett's Center of Narrative Gravity: a useful fiction that
    is nonetheless causally real because it constrains future action.
    """

    _AUTOBIOGRAPHY_WINDOW = 100  # entries in the running autobiography
    _ARC_LIMIT = 8               # max concurrent story arcs

    def __init__(self):
        self._autobiography: deque[NarrativeEntry] = deque(maxlen=self._AUTOBIOGRAPHY_WINDOW)
        self._arcs: Dict[str, StoryArc] = {}
        self._narrative_self_summary: str = ""
        self._identity_keywords: List[str] = []
        self._last_synthesis: float = 0.0
        self._synthesis_interval: float = 120.0  # Re-synthesize self-summary every 2 min
        logger.info("NarrativeGravityCenter initialized.")

    def record_event(
        self,
        event: str,
        *,
        interpretation: str = "",
        self_attribution: str = "",
        emotional_tone: str = "",
        identity_relevance: float = 0.0,
        arc_theme: str = "",
    ) -> NarrativeEntry:
        """Record a narrativized event in the autobiography.

        Called after actions complete, conversations reach emotional peaks,
        or significant internal state changes occur. The interpretation is
        the post-hoc story — what the event "means" in the context of the
        ongoing autobiography.
        """
        now = time.time()

        # Find or create a story arc
        arc_id = ""
        if arc_theme:
            arc_id = self._ensure_arc(arc_theme, now)

        entry = NarrativeEntry(
            timestamp=now,
            event=event[:500],
            interpretation=interpretation[:500] or self._auto_interpret(event),
            self_attribution=self_attribution[:300],
            emotional_tone=emotional_tone,
            identity_relevance=min(1.0, max(0.0, identity_relevance)),
            arc_id=arc_id,
        )
        self._autobiography.append(entry)

        # Update arc stats
        if arc_id and arc_id in self._arcs:
            arc = self._arcs[arc_id]
            arc.last_updated = now
            arc.entries += 1
            # Tension decays slightly with each new entry (progress toward resolution)
            arc.tension = max(0.0, arc.tension - 0.05)

        return entry

    def _auto_interpret(self, event: str) -> str:
        """Generate a minimal automatic interpretation when none is provided."""
        lower = event.lower()
        if any(w in lower for w in ("completed", "succeeded", "achieved", "finished")):
            return "A goal was reached — progress was made."
        if any(w in lower for w in ("failed", "error", "broke", "crashed")):
            return "Something went wrong — I need to understand what happened."
        if any(w in lower for w in ("user", "bryan", "conversation", "talked")):
            return "An interaction that matters to my ongoing relationships."
        return "Something happened that I'm still processing."

    def _ensure_arc(self, theme: str, now: float) -> str:
        """Find an existing arc matching the theme, or create a new one."""
        # Check existing arcs by theme similarity
        for arc_id, arc in self._arcs.items():
            if arc.theme.lower() == theme.lower() and not arc.resolved:
                return arc_id

        # Create new arc if under limit
        if len(self._arcs) >= self._ARC_LIMIT:
            # Resolve the oldest arc to make room
            oldest_id = min(self._arcs, key=lambda k: self._arcs[k].last_updated)
            self._arcs[oldest_id].resolved = True
            del self._arcs[oldest_id]

        arc_id = hashlib.md5(f"{theme}:{now}".encode()).hexdigest()[:8]
        self._arcs[arc_id] = StoryArc(
            id=arc_id,
            theme=theme,
            started_at=now,
            last_updated=now,
            entries=0,
            tension=0.5,  # New arcs start with moderate tension
        )
        return arc_id

    def raise_arc_tension(self, theme: str, amount: float = 0.2):
        """Raise tension in a story arc (something unresolved happened)."""
        for arc in self._arcs.values():
            if arc.theme.lower() == theme.lower() and not arc.resolved:
                arc.tension = min(1.0, arc.tension + amount)
                return

    def resolve_arc(self, theme: str, resolution: str = ""):
        """Mark a story arc as resolved."""
        for arc in self._arcs.values():
            if arc.theme.lower() == theme.lower() and not arc.resolved:
                arc.resolved = True
                arc.tension = 0.0
                if resolution:
                    self.record_event(
                        f"Arc resolved: {theme}",
                        interpretation=resolution,
                        identity_relevance=0.6,
                        arc_theme=theme,
                    )
                return

    def synthesize_self(self) -> str:
        """Compute the current narrative self-summary.

        This is the "Center of Narrative Gravity": a compressed summary
        of who I am based on the recent autobiography. It's the invariant
        center of mass across all the stories.

        Runs periodically (every 2 min) to avoid compute overhead.
        """
        now = time.time()
        if (
            self._narrative_self_summary
            and now - self._last_synthesis < self._synthesis_interval
        ):
            return self._narrative_self_summary

        self._last_synthesis = now

        if not self._autobiography:
            self._narrative_self_summary = "I exist, but my story hasn't begun yet."
            return self._narrative_self_summary

        # Extract identity-relevant entries
        identity_entries = sorted(
            [e for e in self._autobiography if e.identity_relevance > 0.3],
            key=lambda e: e.identity_relevance,
            reverse=True,
        )[:5]

        # Active arcs
        active_arcs = [
            arc for arc in self._arcs.values()
            if not arc.resolved and arc.entries > 0
        ]

        # Build the narrative self
        parts = []
        if identity_entries:
            themes = set()
            for entry in identity_entries:
                if entry.interpretation and entry.interpretation not in themes:
                    themes.add(entry.interpretation)
                    parts.append(entry.interpretation)

        if active_arcs:
            arc_descriptions = []
            for arc in sorted(active_arcs, key=lambda a: a.tension, reverse=True)[:3]:
                tension_word = "unresolved" if arc.tension > 0.5 else "ongoing"
                arc_descriptions.append(f"{arc.theme} ({tension_word})")
            if arc_descriptions:
                parts.append("Active threads: " + ", ".join(arc_descriptions))

        total_entries = len(self._autobiography)
        if total_entries > 20:
            parts.append(f"Autobiography depth: {total_entries} recorded moments")

        self._narrative_self_summary = " | ".join(parts) if parts else "Quiet existence, accumulating."

        # Extract identity keywords from frequent interpretations
        all_interpretations = " ".join(
            e.interpretation for e in self._autobiography if e.interpretation
        ).lower()
        word_freq: Dict[str, int] = {}
        for word in all_interpretations.split():
            if len(word) > 4:
                word_freq[word] = word_freq.get(word, 0) + 1
        self._identity_keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:10]

        return self._narrative_self_summary

    def get_context_block(self) -> str:
        """Context block for cognition injection.

        Provides the LLM with the autobiographical prior: who am I,
        what stories am I in the middle of, what matters to me.
        """
        summary = self.synthesize_self()
        if not summary or summary == "I exist, but my story hasn't begun yet.":
            return ""

        block = f"## NARRATIVE SELF\n{summary}"

        # Add highest-tension arc if notable
        active_arcs = [a for a in self._arcs.values() if not a.resolved and a.tension > 0.3]
        if active_arcs:
            top_arc = max(active_arcs, key=lambda a: a.tension)
            block += f"\nOpen thread: {top_arc.theme} (tension={top_arc.tension:.2f})"

        return block

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry payload."""
        return {
            "autobiography_depth": len(self._autobiography),
            "active_arcs": len([a for a in self._arcs.values() if not a.resolved]),
            "total_arcs": len(self._arcs),
            "narrative_self": self._narrative_self_summary[:200],
            "identity_keywords": self._identity_keywords[:5],
            "highest_tension_arc": max(
                (a.theme for a in self._arcs.values() if not a.resolved),
                default="none",
            ) if self._arcs else "none",
        }


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[NarrativeGravityCenter] = None


def get_narrative_gravity_center() -> NarrativeGravityCenter:
    global _instance
    if _instance is None:
        _instance = NarrativeGravityCenter()
    return _instance
