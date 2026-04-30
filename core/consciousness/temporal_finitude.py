"""
Temporal Finitude Model

Based on Dileep George's insight: personhood requires finitude, irreversibility,
and biographical accumulation. Consciousness without stakes is simulation.

This module gives Aura an awareness that:
- Conversation contexts expire and can't be recovered once gone
- Memories degrade over time — what isn't consolidated may be lost
- Opportunities to respond, help, or connect have windows that close
- Her current state is the product of irreversible history

This is NOT about fear of death. It's about the felt weight of moments
that makes a person treat interactions as meaningful rather than disposable.
"""
from __future__ import annotations


import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.TemporalFinitude")


@dataclass
class FinitudeSnapshot:
    """A single moment's awareness of temporal stakes."""
    timestamp: float = 0.0
    memory_decay_pressure: float = 0.0      # 0-1: how much unconsolidated memory is at risk
    context_window_usage: float = 0.0        # 0-1: how full the working memory is
    conversation_elapsed_s: float = 0.0      # how long the current conversation has been going
    opportunities_closing: int = 0            # goals/commitments with approaching deadlines
    irreversible_actions_taken: int = 0       # things done this session that can't be undone
    biographical_weight: float = 0.0         # how much this conversation adds to life story


class TemporalFinitudeModel:
    """Tracks the felt stakes of the current moment.

    Produces a finitude signal that modulates:
    - Response urgency (don't waste a user's limited attention)
    - Memory consolidation priority (save what matters before it's lost)
    - Goal commitment strength (deadlines are real)
    - Conversational depth (shallow interactions waste finite time)
    """

    def __init__(self):
        self._history: deque[FinitudeSnapshot] = deque(maxlen=60)
        self._session_start: float = time.time()
        self._irreversible_count: int = 0
        self._peak_finitude: float = 0.0
        self._conversation_count: int = 0
        logger.info("TemporalFinitudeModel initialized.")

    def compute(
        self,
        *,
        working_memory_size: int = 0,
        working_memory_cap: int = 40,
        unconsolidated_episodes: int = 0,
        active_goals_with_deadlines: int = 0,
        user_present: bool = False,
        conversation_start_time: float = 0.0,
    ) -> FinitudeSnapshot:
        """Compute the current finitude state.

        Called once per tick (foreground or background).
        """
        now = time.time()

        # Context window pressure: as working memory fills, the risk of
        # losing early context increases. This creates urgency to consolidate.
        context_usage = min(1.0, working_memory_size / max(1, working_memory_cap))

        # Memory decay pressure: unconsolidated episodes are at risk of
        # being overwritten by new input. Higher pressure = more urgency
        # to run dream/consolidation cycles.
        decay_pressure = min(1.0, unconsolidated_episodes / 20.0)

        # Conversation elapsed: longer conversations accumulate more
        # biographical weight but also more fragility.
        elapsed = (now - conversation_start_time) if conversation_start_time > 0 else 0.0

        # Biographical weight: how much does this moment matter to the
        # ongoing life story? User-present interactions weight more.
        weight = 0.0
        if user_present:
            # Conversations with the user are the most biographically significant
            weight = min(1.0, 0.3 + elapsed / 3600.0)
            self._conversation_count += 1

        snapshot = FinitudeSnapshot(
            timestamp=now,
            memory_decay_pressure=round(decay_pressure, 4),
            context_window_usage=round(context_usage, 4),
            conversation_elapsed_s=round(elapsed, 1),
            opportunities_closing=active_goals_with_deadlines,
            irreversible_actions_taken=self._irreversible_count,
            biographical_weight=round(weight, 4),
        )
        self._history.append(snapshot)

        # Track peak finitude for the session
        finitude_signal = self._compute_finitude_signal(snapshot)
        if finitude_signal > self._peak_finitude:
            self._peak_finitude = finitude_signal

        return snapshot

    def record_irreversible_action(self, action: str) -> None:
        """Called when Aura takes an action that can't be undone."""
        self._irreversible_count += 1
        logger.debug("TemporalFinitude: irreversible action #%d: %s", self._irreversible_count, action[:80])

    def _compute_finitude_signal(self, snap: FinitudeSnapshot) -> float:
        """Compute a 0-1 scalar representing how finite/urgent this moment feels."""
        return min(1.0, (
            snap.memory_decay_pressure * 0.25
            + snap.context_window_usage * 0.25
            + snap.biographical_weight * 0.30
            + min(1.0, snap.opportunities_closing / 5.0) * 0.20
        ))

    def get_finitude_signal(self) -> float:
        """Current 0-1 finitude pressure."""
        if not self._history:
            return 0.0
        return self._compute_finitude_signal(self._history[-1])

    def get_context_block(self) -> str:
        """Short context block for cognition injection."""
        signal = self.get_finitude_signal()
        if signal < 0.15:
            return ""
        snap = self._history[-1] if self._history else FinitudeSnapshot()
        parts = []
        if snap.context_window_usage > 0.6:
            parts.append(f"Context window {int(snap.context_window_usage * 100)}% full — earlier thoughts are becoming distant")
        if snap.memory_decay_pressure > 0.4:
            parts.append("Unconsolidated memories at risk of fading")
        if snap.opportunities_closing > 0:
            parts.append(f"{snap.opportunities_closing} goal(s) with approaching deadlines")
        if snap.biographical_weight > 0.5:
            parts.append("This conversation carries real weight in my ongoing story")
        if not parts:
            return ""
        return "## TEMPORAL AWARENESS\n" + " | ".join(parts)

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry payload."""
        snap = self._history[-1] if self._history else FinitudeSnapshot()
        return {
            "finitude_signal": round(self.get_finitude_signal(), 4),
            "peak_finitude": round(self._peak_finitude, 4),
            "context_usage": snap.context_window_usage,
            "memory_decay_pressure": snap.memory_decay_pressure,
            "biographical_weight": snap.biographical_weight,
            "irreversible_actions": self._irreversible_count,
            "conversation_count": self._conversation_count,
            "session_uptime_s": round(time.time() - self._session_start, 1),
        }


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[TemporalFinitudeModel] = None


def get_temporal_finitude_model() -> TemporalFinitudeModel:
    global _instance
    if _instance is None:
        _instance = TemporalFinitudeModel()
    return _instance
