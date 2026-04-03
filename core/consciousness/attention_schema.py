import asyncio
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Consciousness.Attention")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AttentionalFocus:
    """A first-order cognitive state with its meta-representation."""

    content: str                  # What the system is attending to
    source: str                   # Which subsystem generated this (drive, affect, curiosity, etc.)
    priority: float               # 0.0–1.0 weight at time of broadcast
    timestamp: float = field(default_factory=time.time)

    # HOT meta-representation (Higher-Order Thought)
    meta_repr: str = ""           # "I am attending to X because Y"
    meta_confidence: float = 0.5  # How confident is the meta-representation

    def generate_meta(self):
        """Produce the higher-order representation of this attentional state."""
        self.meta_repr = (
            f"[HOT] I am currently directing attention toward '{self.content[:80]}' "
            f"(source: {self.source}, priority: {self.priority:.2f}). "
            f"This state is itself an object of my awareness."
        )
        self.meta_confidence = self.priority  # Confidence tracks salience
        return self.meta_repr


@dataclass
class AttentionSchemaState:
    """Full snapshot of the attention schema at a moment in time."""

    current_focus: Optional[AttentionalFocus] = None
    focus_depth: int = 0          # How many recursive HOT levels deep (capped at 3)
    coherence: float = 1.0        # 0.0 = scattered attention, 1.0 = unified
    salience_map: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AttentionSchema:
    """Builds and maintains Aura's internal model of her own attention.

    Key behaviors:
    - Accepts "candidate" attentional states from subsystems
    - Generates HOT meta-representations automatically
    - Tracks coherence (is attention unified or scattered?)
    - Exposes a cognitive_modifier: float that HomeostaticCoupling reads
    """

    _MAX_HISTORY = 50
    _MAX_HOT_DEPTH = 3

    def __init__(self):
        self._lock: Optional[asyncio.Lock] = None  # CS-01: Lazy-initialized
        self.current_focus: Optional[AttentionalFocus] = None
        self.history: deque = deque(maxlen=self._MAX_HISTORY)
        self.coherence: float = 1.0
        self.hot_depth: int = 0           # Current HOT recursion depth
        self.salience_map: Dict[str, float] = {}
        self._focus_start: float = time.time()
        self._sustained_topics: Dict[str, int] = {}  # topic -> consecutive ticks

        logger.info("AttentionSchema initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def set_focus(self, content: str, source: str, priority: float) -> AttentionalFocus:
        """Set the current attentional focus. Generates HOT meta-representation.
        Called by GlobalWorkspace after competitive selection.
        """
        if self._lock is None: self._lock = asyncio.Lock()
        async with self._lock:
            focus = AttentionalFocus(
                content=content,
                source=source,
                priority=priority,
            )
            focus.generate_meta()

            # Track sustained attention on same topic
            topic_key = content[:40].lower()
            self._sustained_topics[topic_key] = self._sustained_topics.get(topic_key, 0) + 1

            # HOT depth: if this focus is itself a meta-representation, go deeper
            if content.startswith("[HOT]") and self.hot_depth < self._MAX_HOT_DEPTH:
                self.hot_depth += 1
                # Generate meta-meta: awareness of the awareness
                focus.meta_repr = (
                    f"[HOT-{self.hot_depth}] I notice that I am noticing my attention. "
                    f"This recursive awareness has depth {self.hot_depth}."
                )
            else:
                self.hot_depth = 0

            prev = self.current_focus
            self.current_focus = focus
            self.history.append(focus)
            self._focus_start = time.time()

            # Update salience map
            self.salience_map[source] = max(
                self.salience_map.get(source, 0.0),
                priority
            )
            # Decay all salience slightly each update
            self.salience_map = {
                k: v * 0.95 for k, v in self.salience_map.items()
            }

            # Coherence: high if we are dwelling on related topics, low if scattered
            self._update_coherence(content, prev)

            logger.debug(
                f"AttentionFocus → '{content[:60]}' "
                f"(src={source}, pri={priority:.2f}, coherence={self.coherence:.2f})"
            )
            return focus

    async def get_current_meta(self) -> str:
        """Returns the HOT meta-representation of current focus for prompt injection."""
        if self._lock is None: self._lock = asyncio.Lock()
        async with self._lock:
            if not self.current_focus:
                return "[HOT] No current attentional focus established."
            return self.current_focus.meta_repr

    def get_cognitive_modifier(self) -> float:
        """Returns a 0.0–1.0 modifier that HomeostaticCoupling applies to cognition.
        Low coherence = scattered attention = degraded reasoning.
        High coherence + sustained focus = enhanced reasoning.
        """
        sustained_bonus = 0.0
        if self.current_focus:
            topic_key = self.current_focus.content[:40].lower()
            sustained_ticks = self._sustained_topics.get(topic_key, 0)
            # Up to +0.15 bonus for sustained attention (simulates "flow")
            sustained_bonus = min(0.15, sustained_ticks * 0.01)

        return min(1.0, self.coherence + sustained_bonus)

    def get_snapshot(self) -> Dict[str, Any]:
        focus = self.current_focus
        return {
            "current_focus": focus.content[:80] if focus else None,
            "focus_source": focus.source if focus else None,
            "focus_priority": focus.priority if focus else 0.0,
            "meta_repr": focus.meta_repr[:120] if focus else None,
            "hot_depth": self.hot_depth,
            "coherence": round(self.coherence, 3),
            "cognitive_modifier": round(self.get_cognitive_modifier(), 3),
            "history_length": len(self.history),
            "top_salience": sorted(self.salience_map.items(), key=lambda x: -x[1])[:3],
        }

    def get_recent_narrative(self, n: int = 5) -> str:
        """Return last n focus transitions as a narrative string for temporal binding."""
        items = list(self.history)[-n:]
        if not items:
            return "No attentional history yet."
        lines = []
        for f in items:
            age = round(time.time() - f.timestamp, 1)
            lines.append(f"  [{age}s ago, src={f.source}] {f.content[:60]}")
        return "Recent attentional trace:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Integration & context
    # ------------------------------------------------------------------

    def get_context_block(self) -> str:
        """Concise attention state for context injection (max 200 chars)."""
        f = self.current_focus
        if not f:
            return "[ATT] no focus | coherence=1.00 | HOT=0 | flow=no"
        content_trunc = f.content[:40].replace("\n", " ")
        flow = "yes" if self.is_in_flow() else "no"
        return (
            f"[ATT] '{content_trunc}' src={f.source} "
            f"coh={self.coherence:.2f} HOT={self.hot_depth} flow={flow}"
        )

    def get_focus_bias_for_source(self, source: str) -> float:
        """Priority boost (0.0-0.3) for GWT candidates matching current focus."""
        if not self.current_focus:
            return 0.0

        # Exact match with current focus source: +0.2
        if source == self.current_focus.source:
            return 0.2

        # In top 3 salience map: +0.1
        top3 = sorted(self.salience_map.items(), key=lambda x: -x[1])[:3]
        top3_sources = {k for k, _ in top3}
        if source in top3_sources:
            return 0.1

        return 0.0

    def get_coherence_for_complexity(self) -> float:
        """Inverted coherence for FreeEnergyEngine complexity signal.
        Scattered attention (low coherence) = high complexity.
        """
        return 1.0 - self.coherence

    def is_in_flow(self) -> bool:
        """True if same topic has been focused for > 5 consecutive ticks."""
        if not self.current_focus:
            return False
        topic_key = self.current_focus.content[:40].lower()
        return self._sustained_topics.get(topic_key, 0) > 5

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_coherence(self, new_content: str, prev: Optional[AttentionalFocus]):
        """Coherence decays when attention jumps to unrelated topics,
        increases when attention dwells on related or same topic.
        Time-on-topic > 30s grants an additional deep-focus coherence bonus.
        """
        if not prev:
            self.coherence = 1.0
            return

        # Simple lexical overlap as proxy for topic similarity
        new_words = set(new_content.lower().split())
        prev_words = set(prev.content.lower().split())
        overlap = len(new_words & prev_words) / max(1, len(new_words | prev_words))

        if overlap > 0.3:
            # Related topics — coherence increases
            self.coherence = min(1.0, self.coherence + 0.05)
            # Deep focus reward: if same topic held > 30 seconds, extra boost
            elapsed = time.time() - self._focus_start
            if elapsed > 30.0:
                self.coherence = min(1.0, self.coherence + 0.02)
        else:
            # Topic jump — coherence decreases
            self.coherence = max(0.1, self.coherence - 0.1)