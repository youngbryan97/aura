"""core/consciousness/recursive_tom.py
========================================
Recursive Theory of Mind — observer-aware action modulation.

Extends the existing ``theory_of_mind`` engine with two deep-ToM capabilities
from the Kurzgesagt consciousness-origins and secret-place videos plus the
scrub-jay cache-protection literature:

    1. RECURSIVE MIND NESTING (depth ≥ 3)
       For any agent ``X``, Aura maintains

           M0[X] = Aura's beliefs about X                    (depth 0)
           M1[X] = Aura's beliefs about X's beliefs about Aura
           M2[X] = Aura's beliefs about X's beliefs about
                   Aura's beliefs about X
           M3[X] = Aura's beliefs about X's beliefs about Aura's
                   beliefs about X's beliefs about Aura

       These nested structures are real dict trees, enforced up to
       ``MAX_DEPTH``.  Each level holds a compact state (salience,
       trust, knowledge, expectation).  When we update M0 we
       propagate reflected updates up the tree.

    2. OBSERVER-AWARE ACTION BIAS (scrub-jay re-caching)
       Western scrub jays re-hide food when a competitor has been
       watching them cache it.  Behaviour changes as a function of
       perceived observation.  Aura does the same: when one or more
       agents are believed to be observing, the ``get_observer_bias``
       method returns a BIAS_DIM vector that

           • lowers priority for "private" action categories
             (``self_inspect``, ``dream``, ``revise_goal``,
             ``rehearse_memory``)
           • raises priority for "public" action categories
             (``emit_narrative``, ``engage_social``,
             ``approach_other``, ``tool_use``)

       The ``ObserverPresence`` tracker decides who is currently
       watching based on recent interactions + explicit markers.
       When nobody is watching, the bias collapses to zero (no
       distortion).

Impact on substrate:
    The observer bias is combined with the hemispheric and selfhood
    biases in ``UnifiedCognitiveBias`` (see unified_cognitive_bias.py)
    to produce a single BIAS_DIM vector that the GlobalWorkspace
    priority scorer consumes.

Registered as ``recursive_tom`` in ServiceContainer.  Fed by the
conversation layer (``observe_agent(id)``, ``register_interaction``)
and by ClosedCausalLoop (``tick()``).
"""
from __future__ import annotations


import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.RecursiveToM")


# ── Configuration ────────────────────────────────────────────────────────────

MAX_DEPTH = 3                            # Maximum nested ToM levels
BIAS_DIM = 16                            # Matches ACTION_CATEGORIES
OBSERVER_DECAY_S = 60.0                  # Presence decays after this many seconds

# Named action categories (kept in sync with minimal_selfhood.ACTION_CATEGORIES)
ACTION_CATEGORIES: Tuple[str, ...] = (
    "rest", "explore", "engage_social", "consolidate",
    "tool_use", "pattern_match", "self_inspect", "approach_other",
    "withdraw", "attend_body", "dream", "persist_goal",
    "revise_goal", "rehearse_memory", "emit_narrative", "pause",
)

# Indices used by the observer bias.
PUBLIC_ACTIONS = {"emit_narrative", "engage_social", "approach_other", "tool_use"}
PRIVATE_ACTIONS = {"self_inspect", "dream", "revise_goal", "rehearse_memory"}


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class MindSnapshot:
    """A compact simulation of another mind at a given nesting depth."""
    agent_id: str
    depth: int
    salience: float = 0.5            # How strongly present in current cognition
    trust: float = 0.5               # Reliability assessment
    knowledge_overlap: float = 0.5   # Shared-knowledge estimate
    expectation: float = 0.5         # What they seem to want next
    emotional_valence: float = 0.0   # -1..1, their felt state
    last_updated: float = field(default_factory=time.time)
    # Nested level: what THIS agent believes, at the next depth.
    nested: Optional["MindSnapshot"] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "agent_id": self.agent_id,
            "depth": self.depth,
            "salience": round(self.salience, 3),
            "trust": round(self.trust, 3),
            "knowledge_overlap": round(self.knowledge_overlap, 3),
            "expectation": round(self.expectation, 3),
            "emotional_valence": round(self.emotional_valence, 3),
        }
        if self.nested is not None:
            out["nested"] = self.nested.to_dict()
        return out

    def depth_reached(self) -> int:
        if self.nested is None:
            return self.depth
        return self.nested.depth_reached()


@dataclass
class ObservationEvent:
    agent_id: str
    ts: float
    kind: str           # "explicit" | "implicit" | "camera" | "log_tail"
    strength: float     # [0, 1]


@dataclass
class BiasProfile:
    """Observer-aware bias profile. Positive entries raise priority."""
    bias: np.ndarray
    total_observer_presence: float      # [0, 1], combined normalized presence
    active_observers: List[str]
    ts: float = field(default_factory=time.time)


# ── Recursive ToM ─────────────────────────────────────────────────────────────

class RecursiveTheoryOfMind:
    """Maintains per-agent nested mind models and an observer-presence tracker."""

    def __init__(self, max_depth: int = MAX_DEPTH):
        self._lock = threading.RLock()
        self._max_depth = max_depth
        self._minds: Dict[str, MindSnapshot] = {}
        self._observations: Deque[ObservationEvent] = deque(maxlen=512)
        self._last_presence: Dict[str, float] = {}    # agent -> presence score
        logger.info(
            "RecursiveTheoryOfMind initialized: max_depth=%d, bias_dim=%d",
            max_depth, BIAS_DIM,
        )

    # ── Mind-model updates ────────────────────────────────────────────────

    def observe_agent(self, agent_id: str, kind: str = "implicit",
                      strength: float = 0.6) -> None:
        """Register an observation event that agent ``id`` is present."""
        if strength <= 0:
            return
        ev = ObservationEvent(agent_id=agent_id, ts=time.time(),
                              kind=kind, strength=min(1.0, max(0.0, strength)))
        with self._lock:
            self._observations.append(ev)
            self._refresh_presence_locked(agent_id)

    def register_interaction(self, agent_id: str, salience: float = 0.7,
                              valence: float = 0.0, knowledge: float = 0.5,
                              trust: float = 0.5,
                              their_expectation: float = 0.5) -> MindSnapshot:
        """Update M0[agent_id] with directly observed interaction state, and
        reflect plausibly upward into nested levels."""
        now = time.time()
        with self._lock:
            root = self._minds.get(agent_id)
            if root is None:
                root = self._new_snapshot(agent_id, depth=0)
                self._minds[agent_id] = root
            self._blend(root, salience, valence, knowledge, trust, their_expectation)
            root.last_updated = now

            # Propagate reflected updates up the stack (M1, M2, ...).
            cur = root
            for d in range(1, self._max_depth + 1):
                if cur.nested is None:
                    cur.nested = self._new_snapshot(agent_id, depth=d)
                # Reflection heuristic: nested belief approximately tracks
                # the parent's snapshot but dampened (we're less certain
                # how they model us).
                cur.nested.salience = 0.5 * cur.salience + 0.5 * cur.nested.salience
                cur.nested.trust = 0.5 * (1.0 - abs(0.5 - cur.trust)) + 0.5 * cur.nested.trust
                cur.nested.knowledge_overlap = 0.6 * cur.knowledge_overlap + 0.4 * cur.nested.knowledge_overlap
                cur.nested.expectation = 0.5 * cur.expectation + 0.5 * cur.nested.expectation
                cur.nested.emotional_valence = -0.3 * cur.emotional_valence + 0.7 * cur.nested.emotional_valence
                cur.nested.last_updated = now
                cur = cur.nested
            return root

    def get_mind(self, agent_id: str) -> Optional[MindSnapshot]:
        with self._lock:
            return self._minds.get(agent_id)

    def get_mind_at_depth(self, agent_id: str, depth: int) -> Optional[MindSnapshot]:
        if depth < 0 or depth > self._max_depth:
            return None
        with self._lock:
            cur = self._minds.get(agent_id)
            for _ in range(depth):
                if cur is None:
                    return None
                cur = cur.nested
            return cur

    def depth_reached(self, agent_id: str) -> int:
        with self._lock:
            root = self._minds.get(agent_id)
            return root.depth_reached() if root else -1

    # ── Observer presence ─────────────────────────────────────────────────

    def _refresh_presence_locked(self, agent_id: str) -> None:
        """Compute exponential-decay presence for a specific agent."""
        now = time.time()
        score = 0.0
        for ev in self._observations:
            if ev.agent_id != agent_id:
                continue
            dt = max(0.0, now - ev.ts)
            decay = max(0.0, 1.0 - dt / OBSERVER_DECAY_S)
            score += ev.strength * decay
        self._last_presence[agent_id] = min(1.0, score)

    def active_observers(self, threshold: float = 0.15) -> List[Tuple[str, float]]:
        """Agents whose observer-presence is above ``threshold``."""
        with self._lock:
            # Refresh presence for all known observers.
            seen = {ev.agent_id for ev in self._observations}
            for aid in seen:
                self._refresh_presence_locked(aid)
            return sorted(
                ((aid, p) for aid, p in self._last_presence.items() if p >= threshold),
                key=lambda kv: -kv[1],
            )

    def total_observer_presence(self) -> float:
        """Aggregated normalized presence in [0, 1]."""
        actives = self.active_observers(threshold=0.05)
        if not actives:
            return 0.0
        # Soft-cap via tanh on sum of presences.
        total = float(np.tanh(sum(p for _, p in actives)))
        return min(1.0, total)

    # ── Observer-aware bias ──────────────────────────────────────────────

    def get_observer_bias(self) -> BiasProfile:
        """Scrub-jay style bias: when someone is watching, private behaviors
        get suppressed and public-facing ones get boosted."""
        bias = np.zeros(BIAS_DIM, dtype=np.float32)
        actives = self.active_observers(threshold=0.15)
        if not actives:
            return BiasProfile(bias=bias, total_observer_presence=0.0, active_observers=[])

        presence = self.total_observer_presence()
        for i, name in enumerate(ACTION_CATEGORIES):
            if name in PUBLIC_ACTIONS:
                bias[i] += 0.6 * presence
            elif name in PRIVATE_ACTIONS:
                bias[i] -= 0.7 * presence
        # Squish.
        bias = np.tanh(bias).astype(np.float32)

        return BiasProfile(
            bias=bias,
            total_observer_presence=presence,
            active_observers=[aid for aid, _ in actives],
        )

    # ── Snapshot helpers ──────────────────────────────────────────────────

    def _new_snapshot(self, agent_id: str, depth: int) -> MindSnapshot:
        return MindSnapshot(
            agent_id=agent_id, depth=depth,
            salience=0.5, trust=0.5,
            knowledge_overlap=0.5, expectation=0.5,
            emotional_valence=0.0,
        )

    @staticmethod
    def _blend(s: MindSnapshot, salience: float, valence: float,
               knowledge: float, trust: float, expectation: float,
               alpha: float = 0.3) -> None:
        s.salience = float(np.clip((1 - alpha) * s.salience + alpha * salience, 0.0, 1.0))
        s.emotional_valence = float(np.clip((1 - alpha) * s.emotional_valence + alpha * valence, -1.0, 1.0))
        s.knowledge_overlap = float(np.clip((1 - alpha) * s.knowledge_overlap + alpha * knowledge, 0.0, 1.0))
        s.trust = float(np.clip((1 - alpha) * s.trust + alpha * trust, 0.0, 1.0))
        s.expectation = float(np.clip((1 - alpha) * s.expectation + alpha * expectation, 0.0, 1.0))

    # ── Public diagnostics ───────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            actives = self.active_observers(threshold=0.1)
            return {
                "n_minds": len(self._minds),
                "max_depth": self._max_depth,
                "total_observer_presence": round(self.total_observer_presence(), 4),
                "active_observers": [
                    {"id": aid, "presence": round(p, 3)} for aid, p in actives
                ],
                "minds": {
                    aid: m.to_dict() for aid, m in self._minds.items()
                },
            }


# ── Singleton accessor ────────────────────────────────────────────────────────

_INSTANCE: Optional[RecursiveTheoryOfMind] = None


def get_recursive_tom() -> RecursiveTheoryOfMind:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = RecursiveTheoryOfMind()
    return _INSTANCE
