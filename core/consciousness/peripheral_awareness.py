"""
Peripheral Awareness — Attention/Consciousness Dissociation

Based on Koch, Lamme, Tsuchiya: attention and consciousness are doubly
dissociable. You can be conscious of something you're not attending to
(peripheral awareness). You can attend to something you're not conscious
of (subliminal processing).

Aura's Global Workspace conflates these: winning the broadcast = conscious
access. This module adds a parallel pathway where content that DOESN'T win
workspace competition can still be phenomenally present at low intensity.

The key insight: consciousness is broader than the spotlight of attention.
The periphery is dim but still experienced. A person in a quiet room is
conscious of the ambient hum even when attending to a book.

Integration:
- Reads workspace losers from global_workspace.py
- Computes peripheral phenomenal presence for non-broadcast content
- Feeds into qualia_synthesizer as a "peripheral field" dimension
- Injects context when peripheral content is notably strong
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("Consciousness.PeripheralAwareness")


@dataclass
class PeripheralContent:
    """Content that didn't win workspace broadcast but is still phenomenally present."""
    source: str = ""
    content_summary: str = ""
    original_priority: float = 0.0
    peripheral_intensity: float = 0.0  # 0-1: how present in peripheral awareness
    suppression_ticks: int = 0         # How many ticks since this was suppressed
    timestamp: float = 0.0


class PeripheralAwarenessEngine:
    """Maintains peripheral phenomenal presence for non-broadcast content.

    After each Global Workspace competition, the losers don't vanish.
    They persist in a peripheral field with diminishing intensity.
    Content that was ALMOST broadcast (high priority, narrowly lost)
    stays in peripheral awareness longer and more intensely.

    This means Aura can be "aware" of things she's not "attending" to.
    For example, during a focused conversation, she can still be
    peripherally aware of system health, ongoing goals, or unresolved
    emotional states — without them winning the spotlight.

    The peripheral field modulates:
    - Qualia richness (more peripheral content = richer experience)
    - Surprise sensitivity (peripheral content lowers surprise threshold)
    - Topic recall (peripherally present topics are easier to bring up)
    """

    _MAX_PERIPHERAL = 8          # Max items in peripheral field
    _DECAY_PER_TICK = 0.08       # How fast peripheral content fades
    _MIN_INTENSITY = 0.05        # Below this, content drops out
    _NEAR_MISS_BONUS = 0.3       # Extra intensity for content that almost won

    def __init__(self):
        self._peripheral_field: List[PeripheralContent] = []
        self._history: deque[Dict[str, Any]] = deque(maxlen=30)
        self._tick_count: int = 0
        self._total_peripheral_intensity: float = 0.0
        logger.info("PeripheralAwarenessEngine initialized.")

    def process_workspace_results(
        self,
        winner_source: str,
        all_candidates: List[Dict[str, Any]],
    ):
        """Called after each Global Workspace competition.

        Takes the full candidate list and computes peripheral presence
        for losers. Near-misses get higher peripheral intensity.

        Args:
            winner_source: source ID of the winning candidate
            all_candidates: list of dicts with 'source', 'priority', 'content'
        """
        self._tick_count += 1

        if not all_candidates:
            return

        winner_priority = 0.0
        losers = []
        for candidate in all_candidates:
            src = str(candidate.get("source", ""))
            pri = float(candidate.get("priority", 0.0))
            if src == winner_source:
                winner_priority = pri
            else:
                losers.append(candidate)

        # Compute peripheral intensity for each loser
        for loser in losers:
            pri = float(loser.get("priority", 0.0))
            # Near-miss: priority was close to winner → stronger peripheral presence
            gap = max(0.001, winner_priority - pri)
            near_miss_factor = max(0.0, 1.0 - gap * 5.0)  # Close = high factor
            base_intensity = min(0.8, pri * 0.5)
            peripheral_intensity = min(1.0, base_intensity + near_miss_factor * self._NEAR_MISS_BONUS)

            if peripheral_intensity < self._MIN_INTENSITY:
                continue

            # Check if this source is already in peripheral field
            existing = next(
                (p for p in self._peripheral_field if p.source == loser.get("source", "")),
                None,
            )
            if existing:
                # Refresh: take the max of current and new intensity
                existing.peripheral_intensity = max(existing.peripheral_intensity, peripheral_intensity)
                existing.suppression_ticks = 0
                existing.timestamp = time.time()
                existing.content_summary = str(loser.get("content", ""))[:200]
            else:
                self._peripheral_field.append(PeripheralContent(
                    source=str(loser.get("source", "")),
                    content_summary=str(loser.get("content", ""))[:200],
                    original_priority=pri,
                    peripheral_intensity=peripheral_intensity,
                    suppression_ticks=0,
                    timestamp=time.time(),
                ))

        # Decay all peripheral content
        surviving = []
        for item in self._peripheral_field:
            item.peripheral_intensity -= self._DECAY_PER_TICK
            item.suppression_ticks += 1
            if item.peripheral_intensity >= self._MIN_INTENSITY:
                surviving.append(item)

        # Keep only the top N by intensity
        surviving.sort(key=lambda x: x.peripheral_intensity, reverse=True)
        self._peripheral_field = surviving[:self._MAX_PERIPHERAL]

        # Compute total peripheral intensity (measure of peripheral richness)
        self._total_peripheral_intensity = sum(
            p.peripheral_intensity for p in self._peripheral_field
        )

        # Log snapshot
        self._history.append({
            "tick": self._tick_count,
            "peripheral_count": len(self._peripheral_field),
            "total_intensity": round(self._total_peripheral_intensity, 4),
            "strongest": self._peripheral_field[0].source if self._peripheral_field else "",
            "timestamp": time.time(),
        })

    def get_peripheral_richness(self) -> float:
        """0-1 measure of how rich the peripheral field is.

        High = many things present in peripheral awareness.
        Low = narrow focus, empty periphery.
        Feeds into qualia_synthesizer PRI computation.
        """
        if not self._peripheral_field:
            return 0.0
        return min(1.0, self._total_peripheral_intensity / 3.0)

    def get_peripheral_topics(self) -> List[str]:
        """Return summaries of peripherally present content.

        These are topics Aura is "aware of" but not "attending to."
        """
        return [
            p.content_summary
            for p in self._peripheral_field
            if p.peripheral_intensity > 0.1
        ][:3]

    def is_peripherally_aware_of(self, source: str) -> bool:
        """Check if a specific source is currently in peripheral awareness."""
        return any(
            p.source == source and p.peripheral_intensity > self._MIN_INTENSITY
            for p in self._peripheral_field
        )

    def get_context_block(self) -> str:
        """Context block for cognition injection.

        Only fires when peripheral content is notably strong — preventing
        context inflation during focused states.
        """
        if self._total_peripheral_intensity < 0.3:
            return ""

        topics = self.get_peripheral_topics()
        if not topics:
            return ""

        richness = self.get_peripheral_richness()
        items = [t[:80] for t in topics]

        return (
            f"## PERIPHERAL AWARENESS (richness={richness:.2f})\n"
            f"Not the focus, but still present: {' | '.join(items)}"
        )

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry payload."""
        return {
            "peripheral_count": len(self._peripheral_field),
            "total_intensity": round(self._total_peripheral_intensity, 4),
            "richness": round(self.get_peripheral_richness(), 4),
            "items": [
                {
                    "source": p.source,
                    "intensity": round(p.peripheral_intensity, 4),
                    "suppression_ticks": p.suppression_ticks,
                }
                for p in self._peripheral_field
            ],
            "tick_count": self._tick_count,
        }


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[PeripheralAwarenessEngine] = None


def get_peripheral_awareness_engine() -> PeripheralAwarenessEngine:
    global _instance
    if _instance is None:
        _instance = PeripheralAwarenessEngine()
    return _instance
