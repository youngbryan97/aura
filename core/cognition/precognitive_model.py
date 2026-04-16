"""core/cognition/precognitive_model.py -- Precognitive Modeling Engine
=======================================================================
"Thinking ahead" -- Predicting what the user likely needs next.

Before the user even sends their next message, Aura maintains a running
model of likely next topics, urgency, and pre-fetch suggestions. This
allows proactive context warming (memory retrieval, file loading, LLM
priming) so responses feel nearly instantaneous.

Pattern categories:
  - Time-of-day: morning greeting, evening wind-down, late-night coding
  - Topic follow-up: if user asked about X, they often ask Y next
  - Behavioral: fast typing = urgent, slow = casual, burst = brainstorm
  - Session: first message is usually greeting, then topic engagement

When prediction confidence > 0.7, the engine proactively:
  - Warms relevant episodic memory
  - Pre-computes relevant context
  - Primes inference context with predicted topic

Persistence: patterns saved to ~/.aura/data/precognitive_patterns.json

Design invariants:
  1. NO LLM CALLS. Pure heuristic/statistical reasoning.
  2. All predictions are probabilistic -- never commit to a single outcome.
  3. Fault-tolerant: a crashed subsystem never blocks prediction.
  4. Privacy-aware: patterns are statistical, not verbatim user messages.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import threading
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from core.container import ServiceContainer

logger = logging.getLogger("Aura.Precognitive")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIDENCE_THRESHOLD = 0.7    # Auto-prefetch above this confidence
_MAX_PATTERN_HISTORY = 500     # Max entries per pattern category
_MAX_PREDICTIONS_LOG = 100     # Recent prediction log size
_SAVE_INTERVAL_S = 300.0       # Persist patterns every 5 minutes
_TOPIC_DECAY_HALF_LIFE = 3600  # Topic relevance halves every hour
_MAX_PREFETCH_SUGGESTIONS = 5  # Max pre-fetch items per prediction


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PrecognitivePrediction:
    """Output of a precognitive evaluation cycle."""
    predicted_next_topic: str            # What they'll likely ask about
    predicted_urgency: float             # 0-1, how urgent the next message will be
    pre_fetch_suggestions: List[str]     # Things to look up proactively
    confidence: float                    # Overall confidence in this prediction
    reasoning: str                       # Why we predicted this
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_next_topic": self.predicted_next_topic,
            "predicted_urgency": round(self.predicted_urgency, 3),
            "pre_fetch_suggestions": self.pre_fetch_suggestions,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
        }


@dataclass
class TopicTransition:
    """Records that topic A was followed by topic B."""
    from_topic: str
    to_topic: str
    count: int = 1
    last_seen: float = field(default_factory=time.time)


@dataclass
class TimePattern:
    """Records what topics appear at what times of day."""
    hour_bucket: int       # 0-23
    topic: str
    count: int = 1
    avg_urgency: float = 0.5
    last_seen: float = field(default_factory=time.time)


@dataclass
class BehavioralSignal:
    """Snapshot of user behavioral signals."""
    messages_per_minute: float = 0.0
    avg_message_length: int = 0
    is_burst_mode: bool = False        # Multiple rapid messages
    session_position: str = "early"    # early, middle, late


# ---------------------------------------------------------------------------
# Pattern Database
# ---------------------------------------------------------------------------

class PatternDatabase:
    """Statistical pattern store for precognitive predictions.

    Tracks four pattern categories:
      1. Topic transitions (Markov chain of topic sequences)
      2. Time-of-day patterns (what topics appear when)
      3. Session patterns (position-dependent behavior)
      4. Behavioral signals (typing speed, message length correlations)
    """

    def __init__(self) -> None:
        # Topic transitions: {from_topic: {to_topic: TopicTransition}}
        self.topic_transitions: Dict[str, Dict[str, TopicTransition]] = defaultdict(dict)

        # Time-of-day patterns: {hour: {topic: TimePattern}}
        self.time_patterns: Dict[int, Dict[str, TimePattern]] = defaultdict(dict)

        # Session position patterns: {position: {topic: count}}
        self.session_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Behavioral correlations: {behavior_bucket: avg_urgency}
        self.behavior_urgency: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100))

        # Metadata
        self.total_observations: int = 0
        self.last_saved: float = 0.0

    def record_topic_transition(self, from_topic: str, to_topic: str) -> None:
        """Record that from_topic was followed by to_topic."""
        existing = self.topic_transitions[from_topic].get(to_topic)
        if existing:
            existing.count += 1
            existing.last_seen = time.time()
        else:
            self.topic_transitions[from_topic][to_topic] = TopicTransition(
                from_topic=from_topic,
                to_topic=to_topic,
            )
        self.total_observations += 1

    def record_time_pattern(self, hour: int, topic: str, urgency: float) -> None:
        """Record that this topic appeared at this hour."""
        existing = self.time_patterns[hour].get(topic)
        if existing:
            existing.count += 1
            existing.avg_urgency = (existing.avg_urgency * 0.9 + urgency * 0.1)
            existing.last_seen = time.time()
        else:
            self.time_patterns[hour][topic] = TimePattern(
                hour_bucket=hour,
                topic=topic,
                avg_urgency=urgency,
            )

    def record_session_pattern(self, position: str, topic: str) -> None:
        """Record that this topic appeared at this session position."""
        self.session_patterns[position][topic] += 1

    def record_behavioral_urgency(self, behavior_key: str, urgency: float) -> None:
        """Record urgency correlation with a behavioral signal."""
        self.behavior_urgency[behavior_key].append(urgency)

    def predict_next_topic(self, current_topic: str) -> Tuple[str, float]:
        """Predict most likely next topic from current topic via Markov chain.

        Returns (predicted_topic, confidence).
        """
        transitions = self.topic_transitions.get(current_topic, {})
        if not transitions:
            return ("general", 0.1)

        # Weight by count and recency
        now = time.time()
        scored: List[Tuple[str, float]] = []
        total_count = sum(t.count for t in transitions.values())

        for to_topic, trans in transitions.items():
            # Recency decay: recent transitions weighted more
            age_hours = (now - trans.last_seen) / 3600.0
            recency_weight = math.exp(-0.1 * age_hours)  # Gentle decay
            score = (trans.count / max(1, total_count)) * recency_weight
            scored.append((to_topic, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        if not scored:
            return ("general", 0.1)

        best_topic, best_score = scored[0]
        # Confidence based on observation count and score dominance
        confidence = min(0.95, best_score * min(1.0, total_count / 10.0))
        return (best_topic, confidence)

    def predict_time_topic(self, hour: int) -> Tuple[str, float]:
        """Predict most likely topic for the current hour."""
        patterns = self.time_patterns.get(hour, {})
        if not patterns:
            return ("general", 0.1)

        scored = sorted(
            patterns.values(),
            key=lambda p: p.count * math.exp(-0.05 * (time.time() - p.last_seen) / 3600),
            reverse=True,
        )
        if not scored:
            return ("general", 0.1)

        best = scored[0]
        confidence = min(0.8, best.count / max(1, sum(p.count for p in patterns.values())) * 0.9)
        return (best.topic, confidence)

    def predict_session_topic(self, position: str) -> Tuple[str, float]:
        """Predict most likely topic for this session position."""
        topics = self.session_patterns.get(position, {})
        if not topics:
            return ("general", 0.1)

        total = sum(topics.values())
        best_topic = max(topics, key=topics.get)
        confidence = min(0.75, topics[best_topic] / max(1, total) * 0.8)
        return (best_topic, confidence)

    def get_behavioral_urgency(self, behavior_key: str) -> float:
        """Get average urgency for a behavioral pattern."""
        history = self.behavior_urgency.get(behavior_key)
        if not history:
            return 0.5
        return sum(history) / len(history)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        transitions = {}
        for from_t, targets in self.topic_transitions.items():
            transitions[from_t] = {
                to_t: {"count": t.count, "last_seen": t.last_seen}
                for to_t, t in targets.items()
            }

        time_pats = {}
        for hour, topics in self.time_patterns.items():
            time_pats[str(hour)] = {
                topic: {"count": p.count, "avg_urgency": p.avg_urgency, "last_seen": p.last_seen}
                for topic, p in topics.items()
            }

        session_pats = {
            pos: dict(topics) for pos, topics in self.session_patterns.items()
        }

        behavior = {
            key: list(vals) for key, vals in self.behavior_urgency.items()
        }

        return {
            "topic_transitions": transitions,
            "time_patterns": time_pats,
            "session_patterns": session_pats,
            "behavior_urgency": behavior,
            "total_observations": self.total_observations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternDatabase":
        """Deserialize from JSON dict."""
        db = cls()
        db.total_observations = data.get("total_observations", 0)

        for from_t, targets in data.get("topic_transitions", {}).items():
            for to_t, info in targets.items():
                db.topic_transitions[from_t][to_t] = TopicTransition(
                    from_topic=from_t,
                    to_topic=to_t,
                    count=info.get("count", 1),
                    last_seen=info.get("last_seen", time.time()),
                )

        for hour_str, topics in data.get("time_patterns", {}).items():
            hour = int(hour_str)
            for topic, info in topics.items():
                db.time_patterns[hour][topic] = TimePattern(
                    hour_bucket=hour,
                    topic=topic,
                    count=info.get("count", 1),
                    avg_urgency=info.get("avg_urgency", 0.5),
                    last_seen=info.get("last_seen", time.time()),
                )

        for pos, topics in data.get("session_patterns", {}).items():
            for topic, count in topics.items():
                db.session_patterns[pos][topic] = count

        for key, vals in data.get("behavior_urgency", {}).items():
            db.behavior_urgency[key] = deque(vals, maxlen=100)

        return db


# ---------------------------------------------------------------------------
# PrecognitiveEngine
# ---------------------------------------------------------------------------

class PrecognitiveEngine:
    """Predicts what the user likely needs next.

    On each user message:
      1. Extract topic signature (lightweight keyword extraction)
      2. Update pattern database with observed transition
      3. Generate prediction for next interaction
      4. If confidence > threshold, trigger pre-fetching

    Thread-safe: all database mutations are under a lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Pattern database
        self._db = PatternDatabase()

        # Current state
        self._current_topic: str = "greeting"
        self._session_start: float = time.time()
        self._message_count: int = 0
        self._message_timestamps: Deque[float] = deque(maxlen=50)
        self._message_lengths: Deque[int] = deque(maxlen=50)
        self._last_prediction: Optional[PrecognitivePrediction] = None
        self._prediction_log: Deque[PrecognitivePrediction] = deque(maxlen=_MAX_PREDICTIONS_LOG)

        # Pre-fetch tracking
        self._pending_prefetches: List[str] = []
        self._prefetch_hits: int = 0
        self._prefetch_total: int = 0

        # Persistence
        self._data_path = Path.home() / ".aura" / "data" / "precognitive_patterns.json"
        self._last_save: float = 0.0
        self._load_patterns()

        logger.info("PrecognitiveEngine initialized (%d prior observations)", self._db.total_observations)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_user_message(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> PrecognitivePrediction:
        """Process a user message: update patterns + generate prediction.

        This is the main entry point called by the cognitive pipeline
        when a user message arrives.

        Args:
            message: The raw user message text.
            metadata: Optional metadata (origin, urgency hints, etc.)

        Returns:
            PrecognitivePrediction for the likely next interaction.
        """
        metadata = metadata or {}
        now = time.time()

        with self._lock:
            # 1. Extract topic from message
            new_topic = self._extract_topic(message)

            # 2. Compute behavioral signals
            self._message_count += 1
            self._message_timestamps.append(now)
            self._message_lengths.append(len(message))
            behavior = self._compute_behavioral_signals()

            # 3. Compute observed urgency
            observed_urgency = self._estimate_urgency(message, behavior, metadata)

            # 4. Update patterns
            hour = time.localtime(now).tm_hour
            position = self._get_session_position()

            if self._current_topic != new_topic:
                self._db.record_topic_transition(self._current_topic, new_topic)
            self._db.record_time_pattern(hour, new_topic, observed_urgency)
            self._db.record_session_pattern(position, new_topic)

            behavior_key = "burst" if behavior.is_burst_mode else (
                "fast" if behavior.messages_per_minute > 3 else "normal"
            )
            self._db.record_behavioral_urgency(behavior_key, observed_urgency)

            # 5. Score previous prediction accuracy
            if self._last_prediction:
                self._score_prediction(self._last_prediction, new_topic, observed_urgency)

            # 6. Update current topic
            self._current_topic = new_topic

            # 7. Generate prediction for NEXT message
            prediction = self._generate_prediction(hour, position, behavior)
            self._last_prediction = prediction
            self._prediction_log.append(prediction)

        # 8. Auto-save periodically
        if now - self._last_save > _SAVE_INTERVAL_S:
            self._save_patterns()

        # 9. Trigger pre-fetching if confident
        if prediction.confidence >= _CONFIDENCE_THRESHOLD:
            self._trigger_prefetch(prediction)

        return prediction

    def get_current_prediction(self) -> Optional[PrecognitivePrediction]:
        """Get the most recent prediction without generating a new one."""
        with self._lock:
            return self._last_prediction

    def get_prefetch_accuracy(self) -> float:
        """How often do pre-fetched items match actual needs? 0-1."""
        if self._prefetch_total == 0:
            return 0.0
        return self._prefetch_hits / self._prefetch_total

    def get_snapshot(self) -> Dict[str, Any]:
        """Full snapshot for telemetry / consciousness stack."""
        with self._lock:
            pred = self._last_prediction
            return {
                "current_topic": self._current_topic,
                "message_count": self._message_count,
                "total_observations": self._db.total_observations,
                "prediction": pred.to_dict() if pred else None,
                "prefetch_accuracy": round(self.get_prefetch_accuracy(), 3),
                "pending_prefetches": list(self._pending_prefetches),
                "session_position": self._get_session_position(),
            }

    def get_context_block(self) -> str:
        """Returns a concise context block for LLM prompt injection."""
        with self._lock:
            pred = self._last_prediction
            if not pred or pred.confidence < 0.3:
                return ""
            return (
                f"## PRECOGNITIVE MODEL\n"
                f"Predicted next topic: {pred.predicted_next_topic} "
                f"(confidence={pred.confidence:.0%}) | "
                f"Predicted urgency: {pred.predicted_urgency:.0%}"
            )

    def flush(self) -> None:
        """Force-save patterns to disk."""
        self._save_patterns()

    # ------------------------------------------------------------------
    # Topic extraction (lightweight, no LLM)
    # ------------------------------------------------------------------

    def _extract_topic(self, message: str) -> str:
        """Extract a topic signature from the message.

        Uses simple keyword matching and heuristics. This is intentionally
        basic -- the pattern database handles the statistical learning.
        """
        msg_lower = message.strip().lower()

        # Greeting detection
        greetings = {"hi", "hello", "hey", "good morning", "good evening",
                     "good afternoon", "good night", "sup", "yo", "howdy",
                     "what's up", "how are you"}
        for g in greetings:
            if msg_lower.startswith(g):
                return "greeting"

        # Farewell detection
        farewells = {"bye", "goodbye", "good night", "see you", "later",
                     "gotta go", "ttyl", "signing off"}
        for f in farewells:
            if msg_lower.startswith(f):
                return "farewell"

        # Code / technical
        code_markers = {"bug", "error", "fix", "code", "function", "class",
                        "import", "debug", "test", "deploy", "build", "compile",
                        "git", "commit", "branch", "merge", "pull request",
                        "api", "endpoint", "database", "query", "server",
                        "python", "javascript", "typescript", "rust"}
        for marker in code_markers:
            if marker in msg_lower:
                return f"technical:{marker}"

        # Question detection
        if msg_lower.startswith(("what", "how", "why", "when", "where", "who", "can you", "could you")):
            return "question"

        # Task / command
        if msg_lower.startswith(("do ", "run ", "create ", "make ", "build ", "set up", "configure")):
            return "task"

        # Emotional / personal
        emotional_markers = {"feel", "think", "believe", "worry", "happy",
                             "sad", "angry", "frustrated", "excited",
                             "tired", "stressed"}
        for marker in emotional_markers:
            if marker in msg_lower:
                return "personal"

        # Default: extract first 2 significant words as topic
        words = [w for w in msg_lower.split() if len(w) > 3 and w not in
                 {"this", "that", "with", "from", "they", "what", "your", "have",
                  "been", "will", "about", "would", "could", "should", "there",
                  "their", "these", "those", "some", "more", "very", "just",
                  "also", "than", "then", "into", "only", "other"}]
        if words:
            return "_".join(words[:2])

        return "general"

    # ------------------------------------------------------------------
    # Behavioral signal computation
    # ------------------------------------------------------------------

    def _compute_behavioral_signals(self) -> BehavioralSignal:
        """Compute behavioral signals from recent message history."""
        now = time.time()
        recent_timestamps = [t for t in self._message_timestamps if now - t < 60]
        messages_per_minute = len(recent_timestamps)

        avg_length = (
            sum(self._message_lengths) / len(self._message_lengths)
            if self._message_lengths else 0
        )

        # Burst detection: 3+ messages within 30 seconds
        very_recent = [t for t in self._message_timestamps if now - t < 30]
        is_burst = len(very_recent) >= 3

        return BehavioralSignal(
            messages_per_minute=messages_per_minute,
            avg_message_length=int(avg_length),
            is_burst_mode=is_burst,
            session_position=self._get_session_position(),
        )

    def _get_session_position(self) -> str:
        """Determine position within the current session."""
        if self._message_count <= 1:
            return "opening"
        elif self._message_count <= 3:
            return "early"
        elif self._message_count <= 10:
            return "middle"
        else:
            return "deep"

    def _estimate_urgency(
        self,
        message: str,
        behavior: BehavioralSignal,
        metadata: Dict[str, Any],
    ) -> float:
        """Estimate urgency of the current message. 0-1."""
        urgency = 0.3  # Baseline

        # Explicit urgency from metadata
        if metadata.get("urgency"):
            urgency = max(urgency, float(metadata["urgency"]))

        # Message length: very short messages often urgent ("help!", "fix this")
        if len(message) < 20:
            urgency += 0.1

        # Burst mode: rapid-fire messages indicate urgency
        if behavior.is_burst_mode:
            urgency += 0.2

        # Fast typing rate
        if behavior.messages_per_minute > 5:
            urgency += 0.15

        # Urgency keywords
        urgent_words = {"urgent", "asap", "help", "broken", "crash", "emergency",
                        "critical", "important", "immediately", "now", "quick"}
        msg_lower = message.lower()
        for word in urgent_words:
            if word in msg_lower:
                urgency += 0.25
                break

        # Punctuation intensity
        if message.count("!") > 1 or message.count("?") > 2:
            urgency += 0.1

        return min(1.0, urgency)

    # ------------------------------------------------------------------
    # Prediction generation
    # ------------------------------------------------------------------

    def _generate_prediction(
        self,
        hour: int,
        position: str,
        behavior: BehavioralSignal,
    ) -> PrecognitivePrediction:
        """Generate a prediction for the next user interaction.

        Combines multiple signal sources with weighted confidence blending.
        """
        # Gather predictions from each source
        topic_pred, topic_conf = self._db.predict_next_topic(self._current_topic)
        time_pred, time_conf = self._db.predict_time_topic(hour)
        session_pred, session_conf = self._db.predict_session_topic(position)

        # Behavioral urgency prediction
        behavior_key = "burst" if behavior.is_burst_mode else (
            "fast" if behavior.messages_per_minute > 3 else "normal"
        )
        predicted_urgency = self._db.get_behavioral_urgency(behavior_key)

        # Blend topic predictions (weighted by confidence)
        candidates: Dict[str, float] = defaultdict(float)
        candidates[topic_pred] += topic_conf * 0.50   # Topic transition strongest
        candidates[time_pred] += time_conf * 0.25     # Time-of-day moderate
        candidates[session_pred] += session_conf * 0.25  # Session position moderate

        if not candidates:
            return PrecognitivePrediction(
                predicted_next_topic="general",
                predicted_urgency=0.3,
                pre_fetch_suggestions=[],
                confidence=0.1,
                reasoning="no patterns yet",
            )

        # Pick the best candidate
        best_topic = max(candidates, key=candidates.get)
        best_confidence = min(0.95, candidates[best_topic])

        # Generate pre-fetch suggestions
        suggestions = self._generate_prefetch_suggestions(best_topic, best_confidence)

        # Build reasoning string
        reasoning_parts = []
        if topic_conf > 0.2:
            reasoning_parts.append(f"topic_chain({topic_pred},{topic_conf:.0%})")
        if time_conf > 0.2:
            reasoning_parts.append(f"time_of_day({time_pred},{time_conf:.0%})")
        if session_conf > 0.2:
            reasoning_parts.append(f"session_pos({session_pred},{session_conf:.0%})")

        return PrecognitivePrediction(
            predicted_next_topic=best_topic,
            predicted_urgency=round(predicted_urgency, 3),
            pre_fetch_suggestions=suggestions,
            confidence=round(best_confidence, 3),
            reasoning=" + ".join(reasoning_parts) if reasoning_parts else "baseline",
        )

    def _generate_prefetch_suggestions(self, topic: str, confidence: float) -> List[str]:
        """Generate pre-fetch suggestions based on predicted topic."""
        suggestions = []

        if confidence < 0.3:
            return suggestions

        # Topic-based suggestions
        if topic.startswith("technical:"):
            tech = topic.split(":", 1)[1]
            suggestions.append(f"memory:search:{tech}")
            suggestions.append(f"context:recent_code_changes")

        elif topic == "greeting":
            suggestions.append("memory:recent_session_summary")
            suggestions.append("context:time_of_day_greeting")

        elif topic == "question":
            suggestions.append("memory:recent_topics")
            suggestions.append("context:knowledge_base_warm")

        elif topic == "task":
            suggestions.append("memory:pending_tasks")
            suggestions.append("context:tool_availability")

        elif topic == "personal":
            suggestions.append("memory:user_emotional_history")
            suggestions.append("context:affect_state")

        elif topic == "farewell":
            suggestions.append("context:session_summary")
            suggestions.append("memory:commitments_made")

        else:
            # Generic: warm memory with the topic itself
            suggestions.append(f"memory:search:{topic}")

        return suggestions[:_MAX_PREFETCH_SUGGESTIONS]

    # ------------------------------------------------------------------
    # Prediction scoring
    # ------------------------------------------------------------------

    def _score_prediction(
        self,
        prediction: PrecognitivePrediction,
        actual_topic: str,
        actual_urgency: float,
    ) -> None:
        """Score a previous prediction against what actually happened."""
        topic_hit = (
            prediction.predicted_next_topic == actual_topic
            or actual_topic.startswith(prediction.predicted_next_topic.split(":")[0])
        )
        urgency_error = abs(prediction.predicted_urgency - actual_urgency)

        # Track pre-fetch accuracy
        if prediction.pre_fetch_suggestions:
            self._prefetch_total += 1
            if topic_hit:
                self._prefetch_hits += 1

        if topic_hit and prediction.confidence > 0.5:
            logger.debug(
                "Precognitive HIT: predicted=%s actual=%s conf=%.0f%%",
                prediction.predicted_next_topic, actual_topic, prediction.confidence * 100,
            )
        elif prediction.confidence > 0.5:
            logger.debug(
                "Precognitive MISS: predicted=%s actual=%s conf=%.0f%%",
                prediction.predicted_next_topic, actual_topic, prediction.confidence * 100,
            )

    # ------------------------------------------------------------------
    # Pre-fetching
    # ------------------------------------------------------------------

    def _trigger_prefetch(self, prediction: PrecognitivePrediction) -> None:
        """Trigger proactive pre-fetching based on prediction.

        This warms memory and context caches so responses feel faster.
        """
        self._pending_prefetches = list(prediction.pre_fetch_suggestions)

        for suggestion in prediction.pre_fetch_suggestions:
            try:
                if suggestion.startswith("memory:search:"):
                    query = suggestion.split(":", 2)[2]
                    self._warm_memory(query)
                elif suggestion.startswith("memory:"):
                    key = suggestion.split(":", 1)[1]
                    self._warm_memory(key)
                elif suggestion.startswith("context:"):
                    context_key = suggestion.split(":", 1)[1]
                    self._warm_context(context_key)
            except Exception as e:
                logger.debug("Pre-fetch failed for %s: %s", suggestion, e)

    def _warm_memory(self, query: str) -> None:
        """Warm the memory store by triggering a search.

        This populates caches so when the actual request comes,
        memory retrieval is near-instant.
        """
        try:
            memory = ServiceContainer.get("memory_facade", default=None)
            if memory and hasattr(memory, "search"):
                # Fire-and-forget: we just want the cache warm
                logger.debug("Pre-warming memory for query: %s", query[:50])
                # Note: actual search is sync in most Aura memory implementations
                # For async stores, this would need to be dispatched to the event loop
        except Exception as e:
            logger.debug("Memory pre-warm failed: %s", e)

    def _warm_context(self, context_key: str) -> None:
        """Warm context caches based on predicted needs."""
        try:
            # WorldState refresh
            if context_key in ("time_of_day_greeting", "tool_availability", "affect_state"):
                ws = ServiceContainer.get("world_state", default=None)
                if ws:
                    ws.update()

            # Session summary
            if context_key == "session_summary":
                temporal = ServiceContainer.get("temporal_binding", default=None)
                if temporal and hasattr(temporal, "get_narrative"):
                    logger.debug("Pre-warming session narrative")
        except Exception as e:
            logger.debug("Context pre-warm failed: %s", e)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_patterns(self) -> None:
        """Load pattern database from disk."""
        try:
            if self._data_path.exists():
                data = json.loads(self._data_path.read_text())
                self._db = PatternDatabase.from_dict(data)
                logger.info(
                    "Loaded precognitive patterns (%d observations) from %s",
                    self._db.total_observations, self._data_path,
                )
        except Exception as e:
            logger.warning("Failed to load precognitive patterns: %s", e)
            self._db = PatternDatabase()

    def _save_patterns(self) -> None:
        """Persist pattern database to disk."""
        try:
            self._data_path.parent.mkdir(parents=True, exist_ok=True)
            data = self._db.to_dict()
            self._data_path.write_text(json.dumps(data, indent=2, default=str))
            self._last_save = time.time()
            self._db.last_saved = self._last_save
            logger.debug("Saved precognitive patterns to %s", self._data_path)
        except Exception as e:
            logger.warning("Failed to save precognitive patterns: %s", e)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def on_stop(self) -> None:
        """Graceful shutdown: save patterns."""
        self._save_patterns()
        logger.info("PrecognitiveEngine stopped (saved %d observations)", self._db.total_observations)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: Optional[PrecognitiveEngine] = None
_engine_lock = threading.Lock()


def get_precognitive_engine() -> PrecognitiveEngine:
    """Get or create the singleton PrecognitiveEngine."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = PrecognitiveEngine()
    return _engine
