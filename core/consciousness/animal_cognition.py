"""core/consciousness/animal_cognition.py — Bio-Inspired Cognitive Strategies

Implements cognitive strategies observed in animal intelligence research,
adapted for Aura's digital architecture. These are not metaphors — they
are functional analogs that give the system capabilities inspired by
real biological cognition.

Implemented strategies:
  1. PathIntegration    — Desert ant navigation: tracks cognitive drift from goals
  2. ReplayPreplay      — Rat hippocampal replay: consolidates and pre-caches
  3. QuorumDecision     — Bee swarm consensus: multi-evaluator voting for decisions
  4. PhysarumRouter     — Slime mold optimization: self-optimizing routing topology
  5. EmotionalStateTracker — Dog social cognition: reads user emotional signals
  6. CognitiveWeb       — Spider extended cognition: tension-weighted knowledge graph
  7. CamouflageAdapter  — Cephalopod adaptation: reflexive style matching

Scientific basis for each is documented inline.
"""
from __future__ import annotations


import asyncio
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Consciousness.AnimalCognition")


# ---------------------------------------------------------------------------
# 1. Path Integration (Cataglyphis desert ant)
# ---------------------------------------------------------------------------
# Desert ants maintain a running vector sum of all movements, giving them
# a "home vector" that points directly back to the nest from any position.
# For Aura: tracks how far reasoning has drifted from the original goal.

@dataclass
class PathIntegrationState:
    """Tracks cognitive drift from the original goal during multi-step reasoning."""
    goal_keywords: set = field(default_factory=set)
    step_count: int = 0
    drift_score: float = 0.0  # 0 = on target, >1 = significantly drifted
    max_drift_before_correction: float = 1.5

    def record_step(self, step_content: str):
        """Record a reasoning step and measure drift from goal."""
        step_words = set(step_content.lower().split())
        if not self.goal_keywords:
            return
        overlap = len(step_words & self.goal_keywords)
        relevance = overlap / max(1, len(self.goal_keywords))
        # Drift increases when steps are irrelevant to goal
        self.drift_score += max(0.0, 0.5 - relevance)
        self.step_count += 1

    def needs_correction(self) -> bool:
        return self.drift_score > self.max_drift_before_correction

    def get_home_vector(self) -> str:
        """Return a correction prompt to re-orient toward the goal."""
        if not self.goal_keywords:
            return ""
        return (
            f"You have drifted from the original question (drift={self.drift_score:.1f}). "
            f"Re-focus on: {', '.join(sorted(self.goal_keywords)[:8])}"
        )


class PathIntegrationEngine:
    """Maintains path integration state for active reasoning chains."""

    def __init__(self):
        self._active: Optional[PathIntegrationState] = None

    def begin_navigation(self, goal: str):
        """Start tracking a new goal."""
        keywords = set(goal.lower().split()) - {
            "the", "a", "an", "is", "are", "was", "were", "what", "how",
            "why", "when", "where", "do", "does", "did", "can", "could",
            "would", "should", "will", "to", "of", "in", "for", "and",
            "or", "but", "not", "with", "this", "that", "it", "you", "i",
            "my", "your", "me", "we", "they", "he", "she",
        }
        self._active = PathIntegrationState(goal_keywords=keywords)

    def record_step(self, content: str):
        if self._active:
            self._active.record_step(content)

    def check_drift(self) -> Optional[str]:
        """Returns correction prompt if drifted, None if on track."""
        if self._active and self._active.needs_correction():
            correction = self._active.get_home_vector()
            self._active.drift_score *= 0.5  # Partial reset after correction
            return correction
        return None

    def end_navigation(self):
        self._active = None


# ---------------------------------------------------------------------------
# 2. Replay/Preplay (Rat hippocampus)
# ---------------------------------------------------------------------------
# During rest, rats replay recent experiences in compressed time.
# Before entering new environments, they "preplay" candidate trajectories.

@dataclass
class ReplayTrace:
    """A compressed trace of a reasoning chain for replay."""
    objective: str
    steps: List[str]  # key decision points
    outcome_quality: float  # -1 bad, 0 neutral, +1 good
    timestamp: float = field(default_factory=time.time)


class ReplayPreplayEngine:
    """Consolidates experience through replay and pre-caches likely queries."""

    def __init__(self, max_traces: int = 50, max_preplay: int = 5):
        self._traces: deque = deque(maxlen=max_traces)
        self._preplay_cache: Dict[str, Dict[str, Any]] = {}
        self._preplay_ttl: float = 300.0  # 5 minutes

    def record_trace(self, objective: str, steps: List[str], quality: float):
        """Record a compressed interaction trace for later replay."""
        self._traces.append(ReplayTrace(
            objective=objective,
            steps=steps[:10],  # Cap at 10 key steps
            outcome_quality=max(-1.0, min(1.0, quality)),
        ))

    def get_replay_batch(self, limit: int = 5) -> List[ReplayTrace]:
        """Get recent traces for replay during idle."""
        recent = sorted(self._traces, key=lambda t: t.timestamp, reverse=True)
        return recent[:limit]

    def get_positive_patterns(self, limit: int = 3) -> List[ReplayTrace]:
        """Get traces of successful interactions for pattern reinforcement."""
        positive = [t for t in self._traces if t.outcome_quality > 0.3]
        return sorted(positive, key=lambda t: t.outcome_quality, reverse=True)[:limit]

    def store_preplay(self, query_hypothesis: str, precomputed: Dict[str, Any]):
        """Pre-cache a likely upcoming query result."""
        self._preplay_cache[query_hypothesis.lower().strip()] = {
            "data": precomputed,
            "stored_at": time.time(),
        }

    def check_preplay(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if we pre-computed results for this query."""
        key = query.lower().strip()
        for cached_key, entry in self._preplay_cache.items():
            if cached_key in key or key in cached_key:
                age = time.time() - entry["stored_at"]
                if age < self._preplay_ttl:
                    logger.info("Preplay cache HIT for query: %s", key[:50])
                    return entry["data"]
        return None

    def clear_stale_preplay(self):
        """Remove expired preplay entries."""
        now = time.time()
        stale = [k for k, v in self._preplay_cache.items()
                 if now - v["stored_at"] > self._preplay_ttl]
        for k in stale:
            del self._preplay_cache[k]


# ---------------------------------------------------------------------------
# 3. Quorum Decision (Honeybee swarm)
# ---------------------------------------------------------------------------
# Swarms commit to a decision only when enough scouts agree.
# High-stakes decisions require consensus, not just a single evaluation.

@dataclass
class QuorumVote:
    voter_id: str
    choice: str
    confidence: float
    reasoning: str = ""


class QuorumDecisionGate:
    """Requires multiple evaluators to agree before committing to high-stakes decisions."""

    def __init__(self, quorum_threshold: float = 0.6):
        self._threshold = quorum_threshold  # fraction that must agree
        self._pending_votes: Dict[str, List[QuorumVote]] = {}

    def open_vote(self, decision_id: str):
        """Open voting for a decision."""
        self._pending_votes[decision_id] = []

    def cast_vote(self, decision_id: str, voter_id: str, choice: str,
                  confidence: float, reasoning: str = ""):
        if decision_id not in self._pending_votes:
            self.open_vote(decision_id)
        self._pending_votes[decision_id].append(
            QuorumVote(voter_id=voter_id, choice=choice,
                       confidence=confidence, reasoning=reasoning)
        )

    def check_quorum(self, decision_id: str) -> Tuple[bool, str, float]:
        """Check if quorum has been reached.
        Returns (quorum_reached, winning_choice, agreement_ratio).
        """
        votes = self._pending_votes.get(decision_id, [])
        if not votes:
            return False, "", 0.0

        # Count weighted votes per choice
        choice_scores: Dict[str, float] = defaultdict(float)
        for v in votes:
            choice_scores[v.choice] += v.confidence

        total = sum(choice_scores.values())
        if total == 0:
            return False, "", 0.0

        best_choice = max(choice_scores, key=choice_scores.get)
        agreement = choice_scores[best_choice] / total

        return agreement >= self._threshold, best_choice, agreement

    def close_vote(self, decision_id: str):
        self._pending_votes.pop(decision_id, None)


# ---------------------------------------------------------------------------
# 4. Physarum Router (Slime mold network optimization)
# ---------------------------------------------------------------------------
# Physarum polycephalum finds near-optimal networks without neurons.
# Reinforce productive pathways, atrophy unused ones.

class PhysarumRouter:
    """Self-optimizing routing topology between Aura subsystems."""

    def __init__(self, subsystems: List[str], decay: float = 0.98,
                 reinforce: float = 1.05, min_diameter: float = 0.1):
        self._subsystems = subsystems
        self._decay = decay
        self._reinforce = reinforce
        self._min_diameter = min_diameter
        # Initialize all-to-all with equal bandwidth
        self._diameters: Dict[Tuple[str, str], float] = {}
        for i, a in enumerate(subsystems):
            for b in subsystems[i + 1:]:
                self._diameters[(a, b)] = 1.0
                self._diameters[(b, a)] = 1.0

    def reinforce_path(self, source: str, target: str):
        """A message on this path contributed to good output — reinforce."""
        key = (source, target)
        if key in self._diameters:
            self._diameters[key] = min(3.0, self._diameters[key] * self._reinforce)

    def decay_all(self):
        """Apply decay to all paths (called periodically)."""
        for key in self._diameters:
            self._diameters[key] = max(
                self._min_diameter,
                self._diameters[key] * self._decay,
            )

    def get_bandwidth(self, source: str, target: str) -> float:
        return self._diameters.get((source, target), self._min_diameter)

    def get_topology(self) -> Dict[str, float]:
        """Return current topology for visualization."""
        return {
            f"{a}->{b}": round(d, 3)
            for (a, b), d in sorted(self._diameters.items(), key=lambda x: -x[1])
            if d > self._min_diameter * 1.5  # Only show active paths
        }

    def exploration_pulse(self):
        """Temporarily boost dormant paths to check for new utility."""
        for key in self._diameters:
            if self._diameters[key] < self._min_diameter * 2:
                self._diameters[key] = 0.5  # Restore to moderate level


# ---------------------------------------------------------------------------
# 5. Emotional State Tracker (Dog social cognition)
# ---------------------------------------------------------------------------
# Dogs read human emotional states through facial expression, vocal tone,
# and behavior patterns. This module reads user emotional signals from text.

@dataclass
class UserEmotionalState:
    """Running model of the user's emotional state."""
    valence: float = 0.0       # -1 negative, +1 positive
    arousal: float = 0.5       # 0 calm, 1 activated
    frustration: float = 0.0   # 0 none, 1 high
    engagement: float = 0.5    # 0 disengaged, 1 highly engaged
    confidence: float = 0.5    # user's apparent confidence in their questions
    last_updated: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "valence": round(self.valence, 2),
            "arousal": round(self.arousal, 2),
            "frustration": round(self.frustration, 2),
            "engagement": round(self.engagement, 2),
            "confidence": round(self.confidence, 2),
        }


class EmotionalStateTracker:
    """Reads user emotional signals from text input patterns."""

    # Sentiment markers (lightweight, no LLM needed)
    _POSITIVE_MARKERS = frozenset({
        "thanks", "thank", "great", "awesome", "perfect", "love", "amazing",
        "excellent", "nice", "cool", "wonderful", "brilliant", "exactly",
        "yes", "yep", "yeah", "right", "good", "happy", "glad",
    })
    _NEGATIVE_MARKERS = frozenset({
        "no", "wrong", "bad", "terrible", "hate", "awful", "broken",
        "frustrated", "annoyed", "confused", "lost", "stuck", "help",
        "failing", "error", "bug", "crash", "doesn't work", "won't work",
    })
    _FRUSTRATION_MARKERS = frozenset({
        "again", "still", "already", "told you", "said", "why won't",
        "come on", "seriously", "ugh", "sigh", "ffs", "wtf",
    })

    def __init__(self, decay_rate: float = 0.1):
        self.state = UserEmotionalState()
        self._decay_rate = decay_rate
        self._message_lengths: deque = deque(maxlen=10)
        self._message_times: deque = deque(maxlen=10)

    def update(self, user_message: str):
        """Update emotional state estimate from a new user message."""
        now = time.time()
        text = user_message.lower()
        words = set(text.split())

        # Decay toward neutral
        self.state.valence *= (1.0 - self._decay_rate)
        self.state.frustration *= (1.0 - self._decay_rate * 2)

        # Sentiment signals
        pos_count = len(words & self._POSITIVE_MARKERS)
        neg_count = len(words & self._NEGATIVE_MARKERS)
        frust_count = len(words & self._FRUSTRATION_MARKERS)

        self.state.valence = max(-1.0, min(1.0, self.state.valence + pos_count * 0.15 - neg_count * 0.2))

        self.state.frustration = max(0.0, min(1.0, self.state.frustration + frust_count * 0.2))

        # Arousal from punctuation density
        excl_count = text.count("!") + text.count("?")
        caps_ratio = sum(1 for c in user_message if c.isupper()) / max(1, len(user_message))
        self.state.arousal = min(1.0, 0.3 + excl_count * 0.1 + caps_ratio * 2.0)

        # Engagement from message length trend
        self._message_lengths.append(len(text))
        if len(self._message_lengths) >= 3:
            recent_avg = sum(list(self._message_lengths)[-3:]) / 3
            older_avg = sum(list(self._message_lengths)[:3]) / max(1, min(3, len(self._message_lengths)))
            if recent_avg > older_avg * 1.3:
                self.state.engagement = min(1.0, self.state.engagement + 0.1)
            elif recent_avg < older_avg * 0.5:
                self.state.engagement = max(0.0, self.state.engagement - 0.15)

        # Response time (user taking longer may indicate frustration or disengagement)
        self._message_times.append(now)

        self.state.last_updated = now

    def get_response_guidance(self) -> Dict[str, Any]:
        """Get guidance for response generation based on user emotional state."""
        guidance = {}
        if self.state.frustration > 0.5:
            guidance["tone"] = "direct_and_concrete"
            guidance["verbosity"] = "shorter"
            guidance["hedging"] = "minimal"
        if self.state.engagement > 0.7:
            guidance["depth"] = "more_detail"
        if self.state.valence < -0.3:
            guidance["warmth"] = "increased"
        return guidance

    def get_neurochemical_triggers(self) -> Dict[str, float]:
        """Map user emotional state to Aura's neurochemical triggers."""
        triggers = {}
        if self.state.frustration > 0.5:
            triggers["norepinephrine_surge"] = self.state.frustration * 0.3
        if self.state.engagement > 0.7:
            triggers["dopamine_surge"] = (self.state.engagement - 0.5) * 0.2
        if self.state.valence > 0.3:
            triggers["oxytocin_surge"] = self.state.valence * 0.15
        return triggers


# ---------------------------------------------------------------------------
# 6. Cognitive Web (Spider extended cognition)
# ---------------------------------------------------------------------------
# Orb-weaving spiders use their webs as external memory and cognitive tools.
# The web's tension encodes information about prey location and size.

class CognitiveWeb:
    """Tension-weighted knowledge graph that acts as external cognitive scaffolding."""

    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def add_node(self, node_id: str, content: str, **metadata):
        self._nodes[node_id] = {
            "content": content,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "access_count": 0,
            **metadata,
        }

    def add_edge(self, source: str, target: str, relation: str = "related",
                 weight: float = 1.0):
        if source not in self._nodes or target not in self._nodes:
            return
        self._edges[(source, target)] = {
            "relation": relation,
            "weight": weight,
            "last_confirmed": time.time(),
            "propagation_strength": 1.0,
        }

    def tug(self, node_id: str, depth: int = 2) -> List[Tuple[str, float]]:
        """Pull on a node and get activated neighbors ranked by relevance.

        Like a spider touching a silk strand — vibrations propagate outward
        with decreasing strength.
        """
        if node_id not in self._nodes:
            return []

        self._nodes[node_id]["last_accessed"] = time.time()
        self._nodes[node_id]["access_count"] += 1

        activated: Dict[str, float] = {}
        frontier = [(node_id, 1.0)]
        visited = {node_id}

        for _ in range(depth):
            next_frontier = []
            for current, strength in frontier:
                for (src, tgt), edge in self._edges.items():
                    neighbor = tgt if src == current else (src if tgt == current else None)
                    if neighbor is None or neighbor in visited:
                        continue
                    propagated = strength * edge["weight"] * edge["propagation_strength"]
                    # Recency bonus
                    age = time.time() - edge["last_confirmed"]
                    recency_factor = math.exp(-age / 86400.0)  # 1-day half-life
                    propagated *= (0.5 + 0.5 * recency_factor)

                    if propagated > 0.05:  # Threshold
                        activated[neighbor] = max(activated.get(neighbor, 0), propagated)
                        visited.add(neighbor)
                        next_frontier.append((neighbor, propagated))
            frontier = next_frontier

        return sorted(activated.items(), key=lambda x: -x[1])

    def decay_stale_edges(self, max_age_days: float = 30.0):
        """Periodic web maintenance — flag and weaken stale connections."""
        now = time.time()
        max_age_seconds = max_age_days * 86400
        to_remove = []
        for key, edge in self._edges.items():
            age = now - edge["last_confirmed"]
            if age > max_age_seconds:
                to_remove.append(key)
            elif age > max_age_seconds * 0.5:
                edge["propagation_strength"] *= 0.9

        for key in to_remove:
            del self._edges[key]


# ---------------------------------------------------------------------------
# 7. Camouflage Adapter (Cephalopod chromatophore system)
# ---------------------------------------------------------------------------
# Cuttlefish match their environment in <300ms without conscious design.
# This module reflexively adapts response style to match the user.

class CamouflageAdapter:
    """Reflexive style adaptation based on observed user patterns."""

    def __init__(self):
        self._vocab_level: float = 5.0   # 1=simple, 10=technical
        self._formality: float = 5.0     # 1=casual, 10=formal
        self._length_pref: str = "medium"  # short, medium, long
        self._samples: deque = deque(maxlen=20)

    def observe_user(self, message: str):
        """Update style model based on user message patterns."""
        words = message.split()
        self._samples.append(message)

        # Vocabulary complexity (average word length as proxy)
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            self._vocab_level = 0.7 * self._vocab_level + 0.3 * min(10, avg_word_len * 1.5)

        # Formality signals
        has_greeting = any(w.lower() in ("hi", "hey", "yo", "sup") for w in words[:3])
        has_formal = any(w.lower() in ("regarding", "furthermore", "accordingly", "therefore") for w in words)
        if has_greeting and not has_formal:
            self._formality = 0.8 * self._formality + 0.2 * 3.0
        elif has_formal:
            self._formality = 0.8 * self._formality + 0.2 * 8.0

        # Length preference
        if len(words) < 10:
            self._length_pref = "short"
        elif len(words) > 50:
            self._length_pref = "long"
        else:
            self._length_pref = "medium"

    def get_style_cues(self) -> Dict[str, Any]:
        """Get style guidance for response generation."""
        return {
            "vocabulary_level": round(self._vocab_level, 1),
            "formality": round(self._formality, 1),
            "length_preference": self._length_pref,
            "style_hint": self._compose_hint(),
        }

    def _compose_hint(self) -> str:
        parts = []
        if self._vocab_level < 4:
            parts.append("Use simple, everyday words.")
        elif self._vocab_level > 7:
            parts.append("Technical vocabulary is appropriate.")
        if self._formality < 4:
            parts.append("Keep it casual and conversational.")
        elif self._formality > 7:
            parts.append("Maintain a professional tone.")
        if self._length_pref == "short":
            parts.append("Be brief.")
        elif self._length_pref == "long":
            parts.append("You can elaborate.")
        return " ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

_path_integration: Optional[PathIntegrationEngine] = None
_replay_engine: Optional[ReplayPreplayEngine] = None
_quorum_gate: Optional[QuorumDecisionGate] = None
_emotional_tracker: Optional[EmotionalStateTracker] = None
_camouflage: Optional[CamouflageAdapter] = None
_cognitive_web: Optional[CognitiveWeb] = None


def get_path_integration() -> PathIntegrationEngine:
    global _path_integration
    if _path_integration is None:
        _path_integration = PathIntegrationEngine()
    return _path_integration


def get_replay_engine() -> ReplayPreplayEngine:
    global _replay_engine
    if _replay_engine is None:
        _replay_engine = ReplayPreplayEngine()
    return _replay_engine


def get_quorum_gate() -> QuorumDecisionGate:
    global _quorum_gate
    if _quorum_gate is None:
        _quorum_gate = QuorumDecisionGate()
    return _quorum_gate


def get_emotional_tracker() -> EmotionalStateTracker:
    global _emotional_tracker
    if _emotional_tracker is None:
        _emotional_tracker = EmotionalStateTracker()
    return _emotional_tracker


def get_camouflage_adapter() -> CamouflageAdapter:
    global _camouflage
    if _camouflage is None:
        _camouflage = CamouflageAdapter()
    return _camouflage


def get_cognitive_web() -> CognitiveWeb:
    global _cognitive_web
    if _cognitive_web is None:
        _cognitive_web = CognitiveWeb()
    return _cognitive_web
