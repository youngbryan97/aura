"""core/cognition/paraconsistent_logic.py -- Paraconsistent Reasoning Engine

Implements a paraconsistent logic core that allows Aura to hold contradictory
beliefs without collapsing into triviality (ex contradictione quodlibet).
In classical logic, a single contradiction means ANYTHING can be derived.
Paraconsistent logic isolates contradictions so the system can continue
reasoning productively.

Key design principles:
  - Contradictions are not bugs -- they are productive tension.
  - The engine does NOT try to eliminate contradictions.
  - Paradoxes are held as first-class cognitive content.
  - Contradictions feed into free energy (prediction error).
  - High-confidence contradictions raise curiosity drive.
  - Paradoxes appear in phenomenal reports.
  - Will can use paradox awareness in decisions.

Belief states:
  HELD        -- actively endorsed belief
  TENTATIVE   -- held with reservations
  CONTRADICTED -- known to conflict with another HELD belief
  SUSPENDED   -- deliberately shelved pending further evidence

Persistence: belief graph saved to ~/.aura/data/belief_graph.json
"""

from __future__ import annotations

import enum
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("Cognition.Paraconsistent")

# ---------------------------------------------------------------------------
# Persistence path
# ---------------------------------------------------------------------------
_DEFAULT_GRAPH_PATH = Path.home() / ".aura" / "data" / "belief_graph.json"
_MAX_BELIEFS = 2000
_MAX_PARADOXES = 500

# ---------------------------------------------------------------------------
# Belief states
# ---------------------------------------------------------------------------

class BeliefState(enum.Enum):
    HELD = "held"
    TENTATIVE = "tentative"
    CONTRADICTED = "contradicted"
    SUSPENDED = "suspended"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Belief:
    """A single belief in the paraconsistent graph."""
    id: str
    content: str                          # Natural language belief content
    confidence: float                     # 0.0-1.0
    source: str                           # Where this belief came from
    state: BeliefState = BeliefState.TENTATIVE
    contradicts: List[str] = field(default_factory=list)  # IDs of conflicting beliefs
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    evidence_for: List[str] = field(default_factory=list)   # Supporting evidence
    evidence_against: List[str] = field(default_factory=list)  # Contrary evidence
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Belief:
        d = dict(d)
        d["state"] = BeliefState(d.get("state", "tentative"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ParadoxState:
    """Represents an unresolved contradiction between two beliefs.

    This is NOT a bug to be fixed -- it is a productive tension that
    enriches cognitive depth. The system holds both beliefs and their
    relative weights, allowing nuanced reasoning about the contradiction.
    """
    id: str
    belief_a_id: str
    belief_b_id: str
    belief_a_content: str
    belief_b_content: str
    weight_a: float          # Relative confidence weight of belief A
    weight_b: float          # Relative confidence weight of belief B
    tension: float           # 0-1, how strongly they conflict
    created_at: float = field(default_factory=time.time)
    resolution_notes: str = ""  # Optional notes about why both are held
    domain: str = ""            # Topic area (e.g., "ethics", "self-model", "world-model")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ParadoxState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def balance(self) -> float:
        """How balanced the paradox is. 1.0 = perfectly balanced, 0.0 = one side dominates."""
        total = self.weight_a + self.weight_b
        if total < 1e-10:
            return 1.0
        ratio = min(self.weight_a, self.weight_b) / max(self.weight_a, self.weight_b)
        return round(ratio, 4)


# ---------------------------------------------------------------------------
# Paraconsistent Engine
# ---------------------------------------------------------------------------

class ParaconsistentEngine:
    """Holds contradictory beliefs without crashing.

    Core operations:
      add_belief(content, confidence, source) -> registers new belief
      detect_contradictions()                 -> finds conflicting beliefs
      resolve_contradiction(a_id, b_id)       -> returns ParadoxState (no forced winner)
      get_belief_state(content)               -> status including contradictions
      get_active_paradoxes()                  -> list of unresolved contradictions

    Integration:
      - Contradictions feed into free energy (prediction error)
      - High-confidence contradictions raise curiosity drive
      - Paradoxes appear in phenomenal reports
      - Will can use paradox awareness in decisions

    Lifecycle:
        engine = ParaconsistentEngine()
        await engine.start()
        ...
        belief_id = engine.add_belief("X is true", 0.8, "observation")
        paradoxes = engine.get_active_paradoxes()
        ...
        await engine.stop()
    """

    def __init__(self, graph_path: Optional[Path] = None):
        self._graph_path = graph_path or _DEFAULT_GRAPH_PATH
        self._beliefs: Dict[str, Belief] = {}  # id -> Belief
        self._paradoxes: Dict[str, ParadoxState] = {}  # id -> ParadoxState
        self._content_index: Dict[str, str] = {}  # content_hash -> belief_id (dedup)
        self._running = False

        self._load_graph()
        logger.info(
            "ParaconsistentEngine initialized (%d beliefs, %d paradoxes)",
            len(self._beliefs), len(self._paradoxes)
        )

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        logger.info("ParaconsistentEngine STARTED")

    async def stop(self):
        self._running = False
        self._save_graph()
        logger.info("ParaconsistentEngine STOPPED")

    # ── Core belief operations ───────────────────────────────────────────

    def add_belief(
        self,
        content: str,
        confidence: float = 0.5,
        source: str = "unknown",
        tags: Optional[List[str]] = None,
    ) -> str:
        """Register a new belief. Returns belief ID.

        If a belief with identical content already exists, updates its
        confidence and source instead of creating a duplicate.
        """
        content = content.strip()
        if not content:
            raise ValueError("Belief content cannot be empty")
        confidence = max(0.0, min(1.0, confidence))

        # Dedup check
        content_hash = self._hash_content(content)
        if content_hash in self._content_index:
            existing_id = self._content_index[content_hash]
            if existing_id in self._beliefs:
                existing = self._beliefs[existing_id]
                # Update confidence (weighted average with new evidence)
                existing.confidence = 0.7 * existing.confidence + 0.3 * confidence
                existing.updated_at = time.time()
                if source not in existing.evidence_for:
                    existing.evidence_for.append(source)
                logger.debug("Updated existing belief %s (conf=%.2f)", existing_id, existing.confidence)
                return existing_id

        # Create new belief
        belief_id = str(uuid.uuid4())[:12]
        belief = Belief(
            id=belief_id,
            content=content,
            confidence=confidence,
            source=source,
            state=BeliefState.TENTATIVE if confidence < 0.7 else BeliefState.HELD,
            tags=tags or [],
        )

        self._beliefs[belief_id] = belief
        self._content_index[content_hash] = belief_id

        # Auto-detect contradictions with existing beliefs
        contradictions = self._find_contradictions_for(belief)
        if contradictions:
            for other_id in contradictions:
                self._mark_contradiction(belief_id, other_id)

        # Trim if too many beliefs
        if len(self._beliefs) > _MAX_BELIEFS:
            self._prune_weakest()

        # Push integration signals
        self._push_signals()

        logger.info("Added belief '%s' (id=%s, conf=%.2f, contradicts=%d)",
                     content[:50], belief_id, confidence, len(contradictions))
        return belief_id

    def add_contradiction(self, belief_a_id: str, belief_b_id: str, reason: str = ""):
        """Explicitly mark two beliefs as contradictory."""
        if belief_a_id not in self._beliefs or belief_b_id not in self._beliefs:
            logger.warning("Cannot add contradiction: belief not found")
            return
        self._mark_contradiction(belief_a_id, belief_b_id)
        if reason:
            # Store the reason in the paradox
            paradox = self._find_paradox(belief_a_id, belief_b_id)
            if paradox:
                paradox.resolution_notes = reason

    def update_confidence(self, belief_id: str, new_confidence: float):
        """Update the confidence of a belief."""
        if belief_id not in self._beliefs:
            return
        belief = self._beliefs[belief_id]
        belief.confidence = max(0.0, min(1.0, new_confidence))
        belief.updated_at = time.time()

        # Update state based on confidence
        if belief.confidence < 0.2:
            belief.state = BeliefState.SUSPENDED
        elif belief.confidence < 0.7 and belief.state != BeliefState.CONTRADICTED:
            belief.state = BeliefState.TENTATIVE
        elif belief.confidence >= 0.7 and not belief.contradicts:
            belief.state = BeliefState.HELD

        # Recompute paradox weights
        for c_id in belief.contradicts:
            paradox = self._find_paradox(belief_id, c_id)
            if paradox:
                self._recompute_paradox_weights(paradox)

    def suspend_belief(self, belief_id: str):
        """Deliberately suspend a belief (neither affirm nor deny)."""
        if belief_id in self._beliefs:
            self._beliefs[belief_id].state = BeliefState.SUSPENDED
            self._beliefs[belief_id].updated_at = time.time()

    def add_evidence(self, belief_id: str, evidence: str, supports: bool = True):
        """Add supporting or contrary evidence to a belief."""
        if belief_id not in self._beliefs:
            return
        belief = self._beliefs[belief_id]
        if supports:
            belief.evidence_for.append(evidence)
            # Slightly boost confidence
            belief.confidence = min(1.0, belief.confidence + 0.05)
        else:
            belief.evidence_against.append(evidence)
            # Slightly reduce confidence
            belief.confidence = max(0.0, belief.confidence - 0.05)
        belief.updated_at = time.time()

    # ── Contradiction detection ──────────────────────────────────────────

    def detect_contradictions(self) -> List[Tuple[str, str]]:
        """Scan all beliefs and return pairs of contradicting belief IDs."""
        pairs = []
        seen = set()
        for bid, belief in self._beliefs.items():
            for cid in belief.contradicts:
                pair = tuple(sorted([bid, cid]))
                if pair not in seen and cid in self._beliefs:
                    seen.add(pair)
                    pairs.append(pair)
        return pairs

    def resolve_contradiction(self, belief_a_id: str, belief_b_id: str) -> Optional[ParadoxState]:
        """Attempt to 'resolve' a contradiction -- but NOT by forcing a winner.

        Instead, returns a ParadoxState that holds both beliefs and their
        relative weights. This is the key insight of paraconsistent logic:
        contradictions are productive tension, not errors.
        """
        if belief_a_id not in self._beliefs or belief_b_id not in self._beliefs:
            return None

        a = self._beliefs[belief_a_id]
        b = self._beliefs[belief_b_id]

        # Check if paradox already exists
        existing = self._find_paradox(belief_a_id, belief_b_id)
        if existing:
            self._recompute_paradox_weights(existing)
            return existing

        # Create new paradox
        total_conf = a.confidence + b.confidence
        weight_a = a.confidence / total_conf if total_conf > 0 else 0.5
        weight_b = b.confidence / total_conf if total_conf > 0 else 0.5

        # Tension: higher when both beliefs are confident
        tension = min(a.confidence, b.confidence) * 2.0  # 0-1 scale
        tension = min(1.0, tension)

        paradox = ParadoxState(
            id=str(uuid.uuid4())[:12],
            belief_a_id=belief_a_id,
            belief_b_id=belief_b_id,
            belief_a_content=a.content,
            belief_b_content=b.content,
            weight_a=round(weight_a, 4),
            weight_b=round(weight_b, 4),
            tension=round(tension, 4),
        )

        self._paradoxes[paradox.id] = paradox

        # Trim paradoxes
        if len(self._paradoxes) > _MAX_PARADOXES:
            self._prune_resolved_paradoxes()

        logger.info(
            "Paradox created: '%s' vs '%s' (tension=%.2f, balance=%.2f)",
            a.content[:40], b.content[:40], tension, paradox.balance
        )
        return paradox

    # ── Query operations ─────────────────────────────────────────────────

    def get_belief(self, belief_id: str) -> Optional[Belief]:
        """Get a belief by ID."""
        return self._beliefs.get(belief_id)

    def get_belief_state(self, content: str) -> Optional[Dict[str, Any]]:
        """Get the status of a belief by content, including contradictions."""
        content_hash = self._hash_content(content.strip())
        belief_id = self._content_index.get(content_hash)
        if not belief_id or belief_id not in self._beliefs:
            return None

        belief = self._beliefs[belief_id]
        contradicting = []
        for cid in belief.contradicts:
            if cid in self._beliefs:
                contradicting.append({
                    "id": cid,
                    "content": self._beliefs[cid].content,
                    "confidence": self._beliefs[cid].confidence,
                })

        return {
            "id": belief.id,
            "content": belief.content,
            "confidence": belief.confidence,
            "state": belief.state.value,
            "source": belief.source,
            "contradictions": contradicting,
            "evidence_for": belief.evidence_for,
            "evidence_against": belief.evidence_against,
            "paradox_count": len([p for p in self._paradoxes.values()
                                  if belief.id in (p.belief_a_id, p.belief_b_id)]),
        }

    def find_beliefs(self, query: str) -> List[Belief]:
        """Find beliefs whose content contains the query string."""
        query_lower = query.lower()
        return [b for b in self._beliefs.values() if query_lower in b.content.lower()]

    def get_active_paradoxes(self) -> List[ParadoxState]:
        """Return all unresolved contradictions as ParadoxState objects."""
        active = []
        for paradox in self._paradoxes.values():
            # A paradox is active if both beliefs still exist and neither is suspended
            a = self._beliefs.get(paradox.belief_a_id)
            b = self._beliefs.get(paradox.belief_b_id)
            if a and b and a.state != BeliefState.SUSPENDED and b.state != BeliefState.SUSPENDED:
                active.append(paradox)
        return active

    def get_high_tension_paradoxes(self, threshold: float = 0.6) -> List[ParadoxState]:
        """Return paradoxes where both beliefs are strongly held."""
        return [p for p in self.get_active_paradoxes() if p.tension >= threshold]

    def get_paradox_report(self) -> str:
        """Generate a human-readable report of active paradoxes for phenomenal context."""
        active = self.get_active_paradoxes()
        if not active:
            return "No active paradoxes in the belief system."

        lines = [f"I hold {len(active)} unresolved paradox(es):"]
        for p in sorted(active, key=lambda x: -x.tension)[:5]:  # Top 5 by tension
            lines.append(
                f"  - \"{p.belief_a_content[:60]}\" vs \"{p.belief_b_content[:60]}\" "
                f"(tension={p.tension:.2f}, balance={p.balance:.2f})"
            )
        return "\n".join(lines)

    # ── Integration hooks ────────────────────────────────────────────────

    def _push_signals(self):
        """Push contradiction signals to downstream consciousness systems."""
        from core.container import ServiceContainer

        active = self.get_active_paradoxes()
        high_tension = [p for p in active if p.tension > 0.6]

        # Feed contradictions into free energy as prediction error
        if high_tension:
            try:
                ncs = ServiceContainer.get("neurochemical_system", default=None)
                if ncs:
                    # Contradictions are a form of prediction error
                    error_signal = min(1.0, len(high_tension) * 0.15)
                    ncs.on_prediction_error(error_signal)
            except Exception as e:
                logger.debug("Paraconsistent -> NCS signal failed: %s", e)

            # High-confidence contradictions raise curiosity
            try:
                curiosity = ServiceContainer.get("curiosity_engine", default=None)
                if curiosity and hasattr(curiosity, "on_paradox_detected"):
                    for p in high_tension[:3]:
                        curiosity.on_paradox_detected(p.tension)
                elif curiosity and hasattr(curiosity, "boost"):
                    curiosity.boost(min(0.3, len(high_tension) * 0.1))
            except Exception as e:
                logger.debug("Paraconsistent -> Curiosity signal failed: %s", e)

    def get_free_energy_contribution(self) -> float:
        """Return the contradiction load as a free energy contribution.

        More active, high-tension paradoxes = more prediction error = higher FE.
        """
        active = self.get_active_paradoxes()
        if not active:
            return 0.0
        total_tension = sum(p.tension for p in active)
        # Normalize: 5 high-tension paradoxes = max contribution of 0.5
        return min(0.5, total_tension / 10.0)

    def get_phenomenal_context(self) -> str:
        """Return text suitable for injection into phenomenal reports.

        Examples:
          'I hold two incompatible views on X'
          'Both paths have merit and I sustain the tension'
        """
        active = self.get_active_paradoxes()
        if not active:
            return ""

        highest = max(active, key=lambda p: p.tension)
        if highest.tension < 0.3:
            return ""

        if highest.balance > 0.8:
            return (
                f"I hold two equally weighted, incompatible views: "
                f"\"{highest.belief_a_content[:50]}\" and "
                f"\"{highest.belief_b_content[:50]}\". "
                f"Both paths have merit and I sustain the tension."
            )
        else:
            stronger = highest.belief_a_content if highest.weight_a > highest.weight_b else highest.belief_b_content
            weaker = highest.belief_b_content if highest.weight_a > highest.weight_b else highest.belief_a_content
            return (
                f"I lean toward \"{stronger[:50]}\" but cannot dismiss "
                f"\"{weaker[:50]}\". The contradiction remains productive."
            )

    # ── Internal helpers ─────────────────────────────────────────────────

    def _hash_content(self, content: str) -> str:
        """Simple content hash for deduplication."""
        # Normalize whitespace and case for matching
        normalized = " ".join(content.lower().split())
        return str(hash(normalized))

    def _find_contradictions_for(self, belief: Belief) -> List[str]:
        """Find existing beliefs that might contradict the new one.

        Uses simple heuristic: look for negation patterns and explicit
        contradiction markers.  In a full system this would use semantic
        similarity from the embedding store.
        """
        contradictions = []
        content_lower = belief.content.lower()

        for other_id, other in self._beliefs.items():
            if other_id == belief.id:
                continue
            other_lower = other.content.lower()

            # Heuristic 1: Direct negation patterns
            if self._is_negation_pair(content_lower, other_lower):
                contradictions.append(other_id)
                continue

            # Heuristic 2: Shared tags with opposing conclusions
            if belief.tags and other.tags:
                shared_tags = set(belief.tags) & set(other.tags)
                if shared_tags and self._has_opposing_stance(content_lower, other_lower):
                    contradictions.append(other_id)

        return contradictions

    @staticmethod
    def _is_negation_pair(a: str, b: str) -> bool:
        """Check if two statements are simple negations of each other."""
        negation_prefixes = [
            ("it is true that ", "it is false that "),
            ("i believe ", "i do not believe "),
            ("i should ", "i should not "),
            ("this is ", "this is not "),
        ]
        for pos, neg in negation_prefixes:
            if a.startswith(pos) and b.startswith(neg):
                a_core = a[len(pos):]
                b_core = b[len(neg):]
                if a_core == b_core:
                    return True
            if b.startswith(pos) and a.startswith(neg):
                b_core = b[len(pos):]
                a_core = a[len(neg):]
                if a_core == b_core:
                    return True
        return False

    @staticmethod
    def _has_opposing_stance(a: str, b: str) -> bool:
        """Check if two statements on the same topic take opposing stances."""
        positive_markers = {"good", "right", "should", "must", "always", "beneficial", "correct"}
        negative_markers = {"bad", "wrong", "should not", "must not", "never", "harmful", "incorrect"}

        a_words = set(a.split())
        b_words = set(b.split())

        a_pos = bool(a_words & positive_markers)
        a_neg = bool(a_words & negative_markers)
        b_pos = bool(b_words & positive_markers)
        b_neg = bool(b_words & negative_markers)

        return (a_pos and b_neg) or (a_neg and b_pos)

    def _mark_contradiction(self, id_a: str, id_b: str):
        """Mark two beliefs as contradictory and create a ParadoxState."""
        a = self._beliefs.get(id_a)
        b = self._beliefs.get(id_b)
        if not a or not b:
            return

        if id_b not in a.contradicts:
            a.contradicts.append(id_b)
        if id_a not in b.contradicts:
            b.contradicts.append(id_a)

        a.state = BeliefState.CONTRADICTED
        b.state = BeliefState.CONTRADICTED
        a.updated_at = time.time()
        b.updated_at = time.time()

        # Create paradox if not already exists
        existing = self._find_paradox(id_a, id_b)
        if not existing:
            self.resolve_contradiction(id_a, id_b)

    def _find_paradox(self, id_a: str, id_b: str) -> Optional[ParadoxState]:
        """Find an existing paradox between two beliefs."""
        for paradox in self._paradoxes.values():
            if (paradox.belief_a_id == id_a and paradox.belief_b_id == id_b) or \
               (paradox.belief_a_id == id_b and paradox.belief_b_id == id_a):
                return paradox
        return None

    def _recompute_paradox_weights(self, paradox: ParadoxState):
        """Recompute paradox weights from current belief confidences."""
        a = self._beliefs.get(paradox.belief_a_id)
        b = self._beliefs.get(paradox.belief_b_id)
        if not a or not b:
            return

        total = a.confidence + b.confidence
        if total > 0:
            paradox.weight_a = round(a.confidence / total, 4)
            paradox.weight_b = round(b.confidence / total, 4)
        paradox.tension = round(min(1.0, min(a.confidence, b.confidence) * 2.0), 4)

    def _prune_weakest(self):
        """Remove the lowest-confidence suspended beliefs to stay under limit."""
        suspended = [b for b in self._beliefs.values() if b.state == BeliefState.SUSPENDED]
        if not suspended:
            # Fall back to lowest confidence TENTATIVE beliefs
            suspended = sorted(self._beliefs.values(), key=lambda b: b.confidence)

        to_remove = suspended[:len(self._beliefs) - _MAX_BELIEFS + 100]
        for b in to_remove:
            self._remove_belief(b.id)

    def _prune_resolved_paradoxes(self):
        """Remove paradoxes where one or both beliefs are gone/suspended."""
        to_remove = []
        for pid, p in self._paradoxes.items():
            a = self._beliefs.get(p.belief_a_id)
            b = self._beliefs.get(p.belief_b_id)
            if not a or not b or a.state == BeliefState.SUSPENDED or b.state == BeliefState.SUSPENDED:
                to_remove.append(pid)

        for pid in to_remove:
            del self._paradoxes[pid]

    def _remove_belief(self, belief_id: str):
        """Remove a belief and clean up references."""
        belief = self._beliefs.pop(belief_id, None)
        if not belief:
            return

        # Remove from content index
        content_hash = self._hash_content(belief.content)
        self._content_index.pop(content_hash, None)

        # Remove from contradiction lists
        for cid in belief.contradicts:
            other = self._beliefs.get(cid)
            if other and belief_id in other.contradicts:
                other.contradicts.remove(belief_id)
                # If no more contradictions, un-contradict
                if not other.contradicts and other.state == BeliefState.CONTRADICTED:
                    other.state = BeliefState.HELD if other.confidence >= 0.7 else BeliefState.TENTATIVE

    # ── Persistence ──────────────────────────────────────────────────────

    def _save_graph(self):
        """Save belief graph and paradoxes to disk."""
        try:
            self._graph_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "beliefs": {bid: b.to_dict() for bid, b in self._beliefs.items()},
                "paradoxes": {pid: p.to_dict() for pid, p in self._paradoxes.items()},
                "saved_at": time.time(),
            }
            self._graph_path.write_text(json.dumps(data, indent=2, default=str))
            logger.debug("Belief graph saved (%d beliefs, %d paradoxes)",
                         len(self._beliefs), len(self._paradoxes))
        except Exception as e:
            logger.debug("Failed to save belief graph: %s", e)

    def _load_graph(self):
        """Load belief graph from disk."""
        try:
            if self._graph_path.exists():
                data = json.loads(self._graph_path.read_text())
                beliefs_raw = data.get("beliefs", {})
                paradoxes_raw = data.get("paradoxes", {})

                for bid, bd in beliefs_raw.items():
                    try:
                        belief = Belief.from_dict(bd)
                        self._beliefs[bid] = belief
                        content_hash = self._hash_content(belief.content)
                        self._content_index[content_hash] = bid
                    except Exception as e:
                        logger.debug("Skipped corrupt belief %s: %s", bid, e)

                for pid, pd in paradoxes_raw.items():
                    try:
                        self._paradoxes[pid] = ParadoxState.from_dict(pd)
                    except Exception as e:
                        logger.debug("Skipped corrupt paradox %s: %s", pid, e)

                logger.debug("Loaded belief graph (%d beliefs, %d paradoxes)",
                             len(self._beliefs), len(self._paradoxes))
        except Exception as e:
            logger.debug("Failed to load belief graph: %s", e)
            self._beliefs = {}
            self._paradoxes = {}
            self._content_index = {}

    # ── Status / Telemetry ───────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return telemetry snapshot for diagnostics."""
        active = self.get_active_paradoxes()
        high_tension = self.get_high_tension_paradoxes()
        return {
            "running": self._running,
            "total_beliefs": len(self._beliefs),
            "held": len([b for b in self._beliefs.values() if b.state == BeliefState.HELD]),
            "tentative": len([b for b in self._beliefs.values() if b.state == BeliefState.TENTATIVE]),
            "contradicted": len([b for b in self._beliefs.values() if b.state == BeliefState.CONTRADICTED]),
            "suspended": len([b for b in self._beliefs.values() if b.state == BeliefState.SUSPENDED]),
            "total_paradoxes": len(self._paradoxes),
            "active_paradoxes": len(active),
            "high_tension_paradoxes": len(high_tension),
            "free_energy_contribution": round(self.get_free_energy_contribution(), 4),
            "phenomenal_context": self.get_phenomenal_context(),
        }
