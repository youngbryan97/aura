from __future__ import annotations
import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from core.container import ServiceContainer

if TYPE_CHECKING:
    from core.orchestrator import RobustOrchestrator
    from core.agency_core import AgencyCore, AgencyState, EngagementMode
    from core.cognitive_integration_layer import CognitiveIntegrationLayer
    from core.resilience.inhibition_manager import InhibitionManager

logger = logging.getLogger("Consciousness.GlobalWorkspace")



# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContentType(Enum):
    """Types of cognitive content for workspace processing."""
    UNKNOWN = auto()
    PERCEPTUAL = auto()
    AFFECTIVE = auto()
    MEMORIAL = auto()
    INTENTIONAL = auto()
    LINGUISTIC = auto()
    SOMATIC = auto()
    SOCIAL = auto()
    META = auto()


@dataclass(order=True)
class WorkItem:
    """Backward compatibility for legacy AttentionSummarizer."""
    priority: float
    ts: float = field(compare=False)
    id: str = field(compare=False)
    source: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False)
    reason: Optional[str] = field(compare=False)


@dataclass
class CognitiveCandidate:
    """A bid for the global workspace broadcast slot.
    Any subsystem can submit one each tick.
    """

    content: str                       # What wants to be broadcast
    source: str                        # e.g. "drive_curiosity", "affect_distress", "memory"
    priority: float                    # 0.0–1.0 base weight
    content_type: ContentType = ContentType.UNKNOWN
    affect_weight: float = 0.0        # Emotional urgency boost (from AffectEngine)
    focus_bias: float = 0.0           # Priority boost for focused attention (from AttentionSchema)
    submitted_at: float = field(default_factory=time.time)

    @property
    def salience(self) -> float:
        """Alias for effective_priority for downstream compatibility."""
        return self.effective_priority

    @property
    def effective_priority(self) -> float:
        """Priority decays slightly with age (recent events are more salient)."""
        age = time.time() - self.submitted_at
        recency = max(0.0, 1.0 - (age / 10.0))  # Full weight within 10s, then decays
        return min(1.0, (self.priority + self.affect_weight * 0.3 + self.focus_bias) * (0.7 + 0.3 * recency))


@dataclass
class BroadcastEvent:
    """The formal event emitted on a workspace competition win.
    Compatible with PhenomenologicalExperiencer.
    """
    winners: List[CognitiveCandidate]
    timestamp: float = field(default_factory=time.time)


@dataclass
class BroadcastRecord:
    winner: CognitiveCandidate
    losers: List[str]          # source names of losers
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Processor registration type
# ---------------------------------------------------------------------------

ProcessorFn = Callable[[CognitiveCandidate], Coroutine]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GlobalWorkspace:
    """The competitive bottleneck. One winner per cognitive tick.

    Inhibition model:
      - Losing subsystems are placed in a cooldown dict.
      - They cannot re-submit for _INHIBIT_TICKS ticks.
      - This prevents the same subsystem from dominating every cycle
        and forces genuine competition.
    """

    _INHIBIT_TICKS: int = 3       # How many ticks a loser is inhibited
    _MAX_CANDIDATES: int = 20     # Hard cap — prevents memory leak if submissions pile up
    _IGNITION_THRESHOLD: float = 0.6  # Priority above which workspace "ignites"
    _PHI_PRIORITY_BOOST: float = 0.15  # Max priority bonus for high-Φ sources

    def __init__(self, attention_schema: Any = None):
        self._lock: Optional[asyncio.Lock] = None
        self._candidates: List[CognitiveCandidate] = []
        self._inhibited: Dict[str, int] = {}   # source -> ticks_remaining
        self._processors: List[ProcessorFn] = []
        self._history: List[BroadcastRecord] = []
        self._tick: int = 0
        self.attention_schema: Any = attention_schema
        self.last_winner: Optional[CognitiveCandidate] = None
        
        # [UNITY] Global Inhibition Link
        self._global_inhibition: Optional[InhibitionManager] = None
        
        # --- Ignition Detection (GWT) ---
        self.ignition_level: float = 0.0    # 0.0-1.0 current ignition intensity
        self.ignited: bool = False          # True when ignition_level >= threshold
        self._ignition_count: int = 0       # Total ignition events
        self._current_phi: float = 0.0      # Φ from substrate (updated externally)
        
        logger.info("GlobalWorkspace initialized (ignition_threshold=%.2f).", self._IGNITION_THRESHOLD)

    @property
    def history(self) -> List[BroadcastRecord]:
        """Backward compatibility for AttentionSummarizer."""
        return self._history

    @history.setter
    def history(self, value: List[BroadcastRecord]):
        self._history = value

    # ------------------------------------------------------------------
    # Submission API — called by subsystems every heartbeat tick
    # ------------------------------------------------------------------

    async def submit(self, candidate: CognitiveCandidate) -> bool:
        """Submit a candidate for the next broadcast competition.
        Returns False if the source is currently inhibited.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        async with self._lock:
            # Check internal inhibition
            if candidate.source in self._inhibited and self._inhibited[candidate.source] > 0:
                logger.debug("GW: %s is internal-inhibited (%d ticks)", candidate.source, self._inhibited[candidate.source])
                return False
                
            # Check global inhibition
            if self._global_inhibition is None:
                self._global_inhibition = ServiceContainer.get("inhibition_manager", default=None)
            
            if self._global_inhibition:
                if await self._global_inhibition.is_inhibited(candidate.source):
                    logger.debug("GW: %s is GLOBAL-inhibited", candidate.source)
                    return False
            
            # Φ-aware priority boost: high integration → higher salience
            if self._current_phi > 0.1:
                phi_boost = min(self._PHI_PRIORITY_BOOST, self._current_phi * 0.1)
                # Fix Issue 68: Don't mutate candidate.priority; use focus_bias instead
                candidate.focus_bias = min(1.0, candidate.focus_bias + phi_boost)
            
            # --- Seizure Guard (Phase 23.5) ---
            if len(self._candidates) >= self._MAX_CANDIDATES:
                # If we're at the limit, additional submissions are dropped
                # AND we trigger a system-wide tension event.
                logger.warning("🧠 [SEIZURE GUARD] GlobalWorkspace is FLOODED (%d candidates). Dropping bid from %s.", 
                               len(self._candidates), candidate.source)
                
                # Use Mycelium to broadcast a tension reflex if possible
                try:
                    mycelium = ServiceContainer.get("mycelial_network", default=None)
                    if mycelium:
                        # Establish a 'tension' state via mycelium
                        h = mycelium.get_hypha("consciousness", "workspace")
                        if h: h.strength = 10.0 # Thicken the visual noise
                        asyncio.create_task(mycelium.emit_reflex("NEURAL_FLOOD", {"source": candidate.source}))
                except Exception as _e:
                    logger.debug('Ignored Exception in global_workspace.py: %s', _e)
                return False

            # Replace any existing candidate from same source (only one bid per source)
            self._candidates = [c for c in self._candidates if c.source != candidate.source]
            self._candidates.append(candidate)
            return True

    # ------------------------------------------------------------------
    # Processor registration — subsystems register to receive broadcasts
    # ------------------------------------------------------------------

    def register_processor(self, fn: ProcessorFn) -> None:
        """Register a coroutine function to be called when a winner is broadcast."""
        self._processors.append(fn)

    def subscribe(self, fn: ProcessorFn) -> None:
        """Alias for register_processor to support AgencyCore subscriptions."""
        self.register_processor(fn)

    # ------------------------------------------------------------------
    # Competition — called once per heartbeat tick
    # ------------------------------------------------------------------

    async def run_competition(self) -> Optional[CognitiveCandidate]:
        """Run the competitive selection. Returns the winner (or None if no candidates).
        Inhibits losers and broadcasts winner to all registered processors.
        """
        self._tick += 1

        if self._lock is None:
            self._lock = asyncio.Lock()

        # Mycelial Pulse (Proof of Life for Workspace)
        try:
            from core.container import ServiceContainer
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                hypha = mycelium.get_hypha("consciousness", "workspace")
                if hypha: hypha.pulse(success=True)
        except Exception as _e:
            logger.debug('Ignored Exception in global_workspace.py: %s', _e)

        async with self._lock:
            # Decay inhibition counters
            self._inhibited = {
                src: count - 1
                for src, count in self._inhibited.items()
                if count > 1
            }

            if not self._candidates:
                return None

            # Sort by effective priority (highest wins)
            self._candidates.sort(key=lambda c: c.effective_priority, reverse=True)
            winner = self._candidates[0]
            losers = self._candidates[1:]

            # Inhibit all losers
            for loser in losers:
                self._inhibited[loser.source] = self._INHIBIT_TICKS

            # Clear candidate pool
            self._candidates = []

            # Record
            record = BroadcastRecord(
                winner=winner,
                losers=[l.source for l in losers]
            )
            self._history.append(record)
            if len(self._history) > 100:
                self._history = self._history[-100:]

            self.last_winner = winner

        # --- Peripheral Awareness (Attention/Consciousness Dissociation) ---
        # Feed losers into the peripheral field so content that didn't win
        # broadcast can still be phenomenally present at low intensity.
        try:
            from core.consciousness.peripheral_awareness import get_peripheral_awareness_engine
            all_candidates_data = [
                {"source": winner.source, "priority": winner.effective_priority, "content": str(winner.content)[:200]}
            ] + [
                {"source": l.source, "priority": l.effective_priority, "content": str(l.content)[:200]}
                for l in losers
            ]
            get_peripheral_awareness_engine().process_workspace_results(
                winner_source=winner.source,
                all_candidates=all_candidates_data,
            )
        except Exception as _pa_exc:
            logger.debug("GW peripheral awareness feed skipped: %s", _pa_exc)

        # --- Ignition Detection ---
        winner_priority = winner.effective_priority
        self.ignition_level = min(1.0, winner_priority / self._IGNITION_THRESHOLD)
        was_ignited = self.ignited
        self.ignited = winner_priority >= self._IGNITION_THRESHOLD
        
        if self.ignited and not was_ignited:
            self._ignition_count += 1
            logger.info(
                "⚡ GW IGNITION #%d: source=%s, priority=%.3f, phi=%.4f",
                self._ignition_count, winner.source, winner_priority, self._current_phi,
            )

            # ── Theory Arbitration: GWT predicts broadcast improves accessibility ──
            try:
                from core.consciousness.theory_arbitration import get_theory_arbitration
                arb = get_theory_arbitration()
                event_id = f"gw_ignition_{self._ignition_count}"
                arb.log_prediction(
                    theory="gwt",
                    event_id=event_id,
                    prediction="broadcast_improves_coherence",
                    confidence=min(1.0, winner_priority),
                )
                # IIT counter-prediction: integration matters more than broadcast
                arb.log_prediction(
                    theory="iit_4_0",
                    event_id=event_id,
                    prediction="phi_determines_coherence_not_broadcast",
                    confidence=0.6,
                )
            except Exception:
                pass  # Theory arbitration is optional

        # 4. Neural Feed Transparency (Phase 13)
        try:
            from core.thought_stream import get_emitter
            emitter = get_emitter()
            if emitter:
                emitter.emit(
                    title="Neural Competition",
                    content=f"Winner: {winner.source} | Content: {winner.content[:100]}",
                    level="info",
                    metadata={
                        "tick": self._tick,
                        "winner_priority": round(winner.effective_priority, 3),
                        "losers": [l.source for l in losers[:3]]
                    }
                )
        except Exception as e:
            logger.debug("Failed to emit Neural Feed match: %s", e)

        # Update attention schema with winner (outside lock)
        if self.attention_schema:
            await self.attention_schema.set_focus(
                content=winner.content,
                source=winner.source,
                priority=winner.effective_priority,
            )

        # Broadcast to all registered processors (outside lock, concurrent)
        if self._processors:
            event = BroadcastEvent(winners=[winner], timestamp=time.time())
            await asyncio.gather(
                *[self._safe_call(proc, event) for proc in self._processors],
                return_exceptions=True
            )

        logger.debug(
            "GW tick %d: winner='%s' (pri=%.2f), inhibited=%s",
            self._tick, winner.source, winner.effective_priority, list(self._inhibited.keys())
        )
        return winner

    async def _safe_call(self, fn: ProcessorFn, event: Union[BroadcastEvent, CognitiveCandidate]):
        try:
            # Handle both legacy single-candidate and new broadcast-event formats
            res = fn(event)
            if res is not None and inspect.isawaitable(res):
                await res
        except Exception as e:
            logger.error("GW processor error: %s", e)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def get_snapshot(self) -> Dict[str, Any]:
        last = self.last_winner
        return {
            "tick": self._tick,
            "last_winner": last.source if last else None,
            "last_content": last.content[:80] if last else None,
            "last_priority": round(last.effective_priority, 3) if last else 0.0,
            "pending_candidates": len(self._candidates),
            "inhibited_sources": list(self._inhibited.keys()),
            "broadcast_history_len": len(self._history),
            "ignition_level": round(self.ignition_level, 3),
            "ignited": self.ignited,
            "ignition_count": self._ignition_count,
            "phi": round(self._current_phi, 4),
        }

    def update_phi(self, phi: float) -> None:
        """Update the current Φ value from the LiquidSubstrate.
        Called by the heartbeat or consciousness system each tick.
        """
        self._current_phi = max(0.0, float(phi))

    def is_ignited(self) -> bool:
        """Whether the workspace is currently in an ignited state."""
        return self.ignited

    def get_ignition_level(self) -> float:
        """Current ignition intensity (0.0-1.0)."""
        return self.ignition_level

    def get_last_n_winners(self, n: int = 5) -> List[Dict]:
        return [
            {
                "winner": r.winner.source,
                "content": r.winner.content[:60],
                "losers": r.losers,
                "timestamp": r.timestamp,
            }
            for r in self._history[-n:]
        ]

    def get_context_stream(self, n: int = 5) -> str:
        """Return a formatted string of the last N winners for prompt injection."""
        winners = self.get_last_n_winners(n)
        if not winners:
            return ""
        
        lines = []
        for w in winners:
            lines.append(f"- [{w['winner']}] {w['content']}")
        return "\n".join(lines)