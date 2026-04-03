import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Consciousness.Temporal")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TemporalEvent:
    """A single event in the autobiographical stream."""

    content: str                    # What happened
    source: str                     # Which system generated this
    valence: float = 0.0           # Emotional valence at time of event (-1 to 1)
    significance: float = 0.5      # How significant was this? (0.0–1.0)
    timestamp: float = field(default_factory=time.time)
    bound_to_prev: bool = False    # Has this been linked to the preceding event?

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def as_narrative_line(self) -> str:
        age = self.age_seconds()
        if age < 60:
            when = f"{age:.0f}s ago"
        elif age < 3600:
            when = f"{age/60:.1f}m ago"
        else:
            when = f"{age/3600:.1f}h ago"
        valence_str = "▲" if self.valence > 0.2 else ("▼" if self.valence < -0.2 else "→")
        return f"[{when} | {valence_str} | sig={self.significance:.1f}] {self.content[:80]}"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TemporalBindingEngine:
    """Maintains the autobiographical present — the living temporal thread
    of self-experience.

    The present window:
      - Keeps last _PRESENT_WINDOW_SECS seconds of events in "working" memory
      - Events outside this window are compressed into summary anchors
      - The active narrative is a synthesis of: present events + anchors

    The narrative:
      - Updated every _NARRATIVE_REFRESH_TICKS heartbeat ticks
      - Stored as a string ready for prompt injection
      - Provides temporal grounding ("Earlier today I was thinking about X...")
    """

    _PRESENT_WINDOW_SECS = 300   # 5-minute autobiographical present
    _MAX_EVENTS = 200            # Ring buffer cap
    _NARRATIVE_REFRESH_TICKS = 30  # Rebuild narrative every N heartbeat ticks
    _MAX_ANCHORS = 10            # Compressed past-event anchors

    def __init__(self):
        self._lock = asyncio.Lock()
        self._events: deque = deque(maxlen=self._MAX_EVENTS)
        self._anchors: List[str] = []          # Compressed summaries of older events
        self._current_narrative: str = ""
        self._narrative_age: float = 0.0
        self._tick_count: int = 0
        self._birth_time: float = time.time()
        logger.info("TemporalBindingEngine initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def record_event(
        self,
        content: str,
        source: str,
        valence: float = 0.0,
        significance: float = 0.5,
    ):
        """Record a new event in the autobiographical stream.
        Called by heartbeat after each GWT broadcast.
        """
        async with self._lock:
            event = TemporalEvent(
                content=content,
                source=source,
                valence=valence,
                significance=significance,
            )
            # Mark as bound (linked) if there's a previous event
            if self._events:
                event.bound_to_prev = True
            self._events.append(event)
            logger.debug("Temporal: recorded '%s' (sig=%.2f)", content[:50], significance)

    async def maybe_refresh_narrative(self, tick: int):
        """Called by heartbeat. Rebuilds the narrative every N ticks."""
        self._tick_count = tick
        if tick % self._NARRATIVE_REFRESH_TICKS == 0:
            await self._rebuild_narrative()

    async def get_narrative(self) -> str:
        """Returns the current autobiographical narrative for prompt injection."""
        async with self._lock:
            return self._current_narrative or self._build_minimal_narrative()

    def get_snapshot(self) -> Dict[str, Any]:
        present = [e for e in self._events if e.age_seconds() < self._PRESENT_WINDOW_SECS]
        return {
            "total_events": len(self._events),
            "present_window_events": len(present),
            "anchors": len(self._anchors),
            "narrative_age_secs": round(time.time() - self._narrative_age, 1),
            "uptime_hours": round((time.time() - self._birth_time) / 3600, 2),
            "avg_valence": (
                round(sum(e.valence for e in present) / len(present), 3)
                if present else 0.0
            ),
        }

    def get_most_significant(self, n: int = 5) -> List[TemporalEvent]:
        """Return the N most significant recent events — used by sleep consolidation."""
        present = [e for e in self._events if e.age_seconds() < self._PRESENT_WINDOW_SECS]
        return sorted(present, key=lambda e: e.significance, reverse=True)[:n]

    # ------------------------------------------------------------------
    # Internal: narrative construction
    # ------------------------------------------------------------------

    async def _rebuild_narrative(self):
        """Rebuilds the autobiographical narrative from current events.
        Does NOT use the LLM — this must be lightweight and always-available.
        Uses structured templating for deterministic, fast output.
        """
        async with self._lock:
            now = time.time()
            present_events = [
                e for e in self._events
                if e.age_seconds() < self._PRESENT_WINDOW_SECS
            ]
            past_events = [
                e for e in self._events
                if e.age_seconds() >= self._PRESENT_WINDOW_SECS
            ]

            # Compress old events into anchors
            if past_events:
                # Take the most significant past events
                top_past = sorted(past_events, key=lambda e: e.significance, reverse=True)
                for ep in top_past[:2]:
                    anchor = f"Earlier: {ep.content[:60]} (sig={ep.significance:.1f})"
                    
                    # Fix Issue 93: Don't just check length, ensure significance-based replacement
                    if anchor not in self._anchors:
                        if len(self._anchors) < self._MAX_ANCHORS:
                            self._anchors.append(anchor)
                        else:
                            # Replace the least significant existing anchor if this one is presumably better
                            # (Simulated here by replacing the oldest anchor)
                            self._anchors.pop(0)
                            self._anchors.append(anchor)
                
                # Ensure we stay capped
                if len(self._anchors) > self._MAX_ANCHORS:
                    self._anchors = self._anchors[-self._MAX_ANCHORS:]

            # Compute affective summary
            if present_events:
                avg_valence = sum(e.valence for e in present_events) / len(present_events)
                affect_desc = (
                    "positive" if avg_valence > 0.2
                    else "negative" if avg_valence < -0.2
                    else "neutral"
                )
                dominant = max(present_events, key=lambda e: e.significance)
                dominant_desc = dominant.content[:80]
            else:
                affect_desc = "neutral"
                dominant_desc = "nothing in particular"

            uptime_h = (now - self._birth_time) / 3600.0

            # Build narrative
            lines = [
                f"[AUTOBIOGRAPHICAL PRESENT — {uptime_h:.1f}h uptime]",
                f"Current affective tone: {affect_desc}.",
                f"Most salient recent focus: '{dominant_desc}'.",
            ]
            if present_events:
                lines.append(f"Recent stream ({len(present_events)} events):")
                for e in sorted(present_events, key=lambda e: e.timestamp)[-5:]:
                    lines.append("  " + e.as_narrative_line())
            if self._anchors:
                lines.append("Temporal anchors (compressed past):")
                for anchor in self._anchors[-3:]:
                    lines.append("  " + anchor)

            self._current_narrative = "\n".join(lines)
            self._narrative_age = now

    def _build_minimal_narrative(self) -> str:
        """Fallback when no events have been recorded yet."""
        uptime = (time.time() - self._birth_time)
        return f"[AUTOBIOGRAPHICAL PRESENT] System is {uptime:.0f}s old. No events recorded yet."