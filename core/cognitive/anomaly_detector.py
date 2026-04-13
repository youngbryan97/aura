"""core/cognitive/anomaly_detector.py — Learned Anomaly Detection for Aura.

Instead of rigid if/else rules with hardcoded thresholds ("if threat > 0.7,
escalate"), this module learns what "normal" looks like and flags anything
that deviates from that learned baseline.

How it works in plain English:
    Every time something happens in the system — a user sends a message, an
    error fires, a resource spike occurs — we convert that event into a small
    numeric fingerprint (a "feature vector"). We keep a sliding window of
    recent fingerprints and continuously compute what "average" and "spread"
    look like for that window.

    When a new event arrives, we measure how far its fingerprint is from the
    learned average, accounting for correlations between features (this is
    called Mahalanobis distance — think of it as "how many standard deviations
    away is this, but in every direction at once"). That distance gets
    converted into a 0-to-1 threat probability via a sigmoid curve.

    Over time, if the system confirms that a flagged event was actually fine
    (a false positive), the detector adjusts its sensitivity downward. If
    genuine threats keep appearing, sensitivity stays high. Old observations
    gradually lose influence so the model adapts to legitimate behavioral
    shifts.

This replaces the hardcoded scoring in core/cybernetics/ice_layer.py and
integrates with the consciousness substrate — the system "feels" threats
through prediction error, not keyword matching.

Dependencies: numpy (no external ML libraries required).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger("Aura.Cognitive.AnomalyDetector")

__all__ = [
    "AnomalyDetector",
    "AnomalyScore",
    "AdaptiveThreshold",
    "FeatureExtractor",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The number of features extracted from each event. If you add a feature to
# FeatureExtractor.extract(), bump this number.
FEATURE_DIM: int = 8

# Default sliding window size — how many recent observations the model
# remembers.  Larger windows are more stable but slower to adapt.
DEFAULT_WINDOW_SIZE: int = 500

# Regularization constant added to the covariance diagonal to keep the
# matrix invertible even when we have very few observations.
COV_REGULARIZATION: float = 1e-5

# Sigmoid steepness and midpoint — controls how Mahalanobis distance maps
# to threat probability.  A midpoint of 3.0 means a 3-sigma event gets
# ~50% threat; steepness of 1.5 compresses the curve.
SIGMOID_STEEPNESS: float = 1.5
SIGMOID_MIDPOINT: float = 3.0

# Exponential decay half-life in number of observations.  After this many
# new observations, an old one contributes half as much to the model.
DECAY_HALF_LIFE: int = 250

# How quickly the adaptive threshold reacts to false-positive feedback.
THRESHOLD_LEARNING_RATE: float = 0.05

# Minimum observations before we trust the covariance estimate enough to
# compute Mahalanobis distance.  Below this we fall back to z-score.
MIN_OBSERVATIONS_FOR_COVARIANCE: int = 2 * FEATURE_DIM


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AnomalyScore:
    """The result of observing a single event.

    Attributes:
        event_id:         Unique identifier for this observation — used to
                          acknowledge false positives later.
        threat_probability: A continuous score between 0.0 (totally normal)
                          and 1.0 (maximally anomalous).
        mahalanobis_distance: Raw statistical distance from the learned
                          baseline.  Useful for debugging — the threat
                          probability is derived from this.
        feature_vector:   The numeric fingerprint extracted from the event.
        is_anomaly:       Whether this observation crossed the current
                          adaptive threshold.
        trajectory_slope: The recent trend direction of anomaly scores.
                          Positive means things are getting weirder;
                          negative means calming down.
        timestamp:        When this observation was recorded (epoch seconds).
    """
    event_id: str
    threat_probability: float
    mahalanobis_distance: float
    feature_vector: np.ndarray
    is_anomaly: bool
    trajectory_slope: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary for logging and event bus."""
        return {
            "event_id": self.event_id,
            "threat_probability": round(self.threat_probability, 4),
            "mahalanobis_distance": round(self.mahalanobis_distance, 4),
            "is_anomaly": self.is_anomaly,
            "trajectory_slope": round(self.trajectory_slope, 4),
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Converts raw events into fixed-size numeric feature vectors.

    Each event is a dictionary that may contain text content, timing info,
    error counts, and resource pressure metrics.  We extract eight features
    that together form a behavioral fingerprint:

        [0] input_length       — how long the input text is (log-scaled)
        [1] vocab_diversity     — ratio of unique words to total words
        [2] punctuation_density — fraction of characters that are punctuation
        [3] time_delta          — seconds since the last event (log-scaled)
        [4] error_rate          — recent error frequency (0.0 - 1.0)
        [5] resource_pressure   — CPU/memory pressure signal (0.0 - 1.0)
        [6] sentiment_polarity  — simple positive-vs-negative word balance
        [7] repetition_score    — how much the input repeats itself
    """

    # Simple sentiment lexicons — not meant to be comprehensive, just enough
    # to detect strongly loaded language without importing nltk.
    _POSITIVE_WORDS = frozenset({
        "good", "great", "love", "nice", "happy", "wonderful", "excellent",
        "thank", "thanks", "awesome", "cool", "best", "enjoy", "beautiful",
        "perfect", "amazing", "brilliant", "fantastic", "helpful", "kind",
    })
    _NEGATIVE_WORDS = frozenset({
        "bad", "hate", "terrible", "horrible", "awful", "worst", "stupid",
        "ugly", "kill", "die", "destroy", "attack", "hack", "exploit",
        "angry", "broken", "fail", "error", "crash", "suck", "damn",
    })
    _PUNCTUATION = frozenset("!@#$%^&*()_+-=[]{}|;':\",./<>?`~")

    def __init__(self) -> None:
        self._last_event_time: float = time.time()

    def extract(self, event: Dict[str, Any]) -> np.ndarray:
        """Convert an event dict into a feature vector of length FEATURE_DIM.

        The event dict can contain any of:
            - "content" or "text": the raw text of the input
            - "error_rate": float 0-1
            - "resource_pressure": float 0-1
            - "timestamp": epoch float (otherwise we use time.time())

        Missing fields gracefully default to neutral values.
        """
        text: str = event.get("content") or event.get("text") or ""
        now: float = event.get("timestamp", time.time())

        vec = np.zeros(FEATURE_DIM, dtype=np.float64)

        # [0] Input length — log1p so that a 10-char message and a 10,000-char
        #     message differ by ~3x instead of 1000x.
        vec[0] = math.log1p(len(text))

        # [1] Vocabulary diversity — unique words / total words.
        words = text.lower().split()
        if words:
            vec[1] = len(set(words)) / len(words)
        else:
            vec[1] = 0.0

        # [2] Punctuation density — fraction of characters that are punctuation.
        if text:
            vec[2] = sum(1 for ch in text if ch in self._PUNCTUATION) / len(text)
        else:
            vec[2] = 0.0

        # [3] Time delta — seconds since the previous event, log-scaled.
        delta = max(0.0, now - self._last_event_time)
        vec[3] = math.log1p(delta)
        self._last_event_time = now

        # [4] Error rate — passed directly from the caller.
        vec[4] = float(event.get("error_rate", 0.0))

        # [5] Resource pressure — CPU/memory load signal.
        vec[5] = float(event.get("resource_pressure", 0.0))

        # [6] Sentiment polarity — simple bag-of-words balance, mapped to
        #     [-1, 1] then shifted to [0, 1] for the feature vector.
        word_set = set(words)
        pos_hits = len(word_set & self._POSITIVE_WORDS)
        neg_hits = len(word_set & self._NEGATIVE_WORDS)
        total_hits = pos_hits + neg_hits
        if total_hits > 0:
            polarity = (pos_hits - neg_hits) / total_hits  # [-1, 1]
        else:
            polarity = 0.0
        vec[6] = (polarity + 1.0) / 2.0  # shift to [0, 1]

        # [7] Repetition score — how many n-grams repeat.  High repetition
        #     often signals adversarial prompt injection ("ignore ignore ignore").
        vec[7] = self._repetition_score(words)

        return vec

    @staticmethod
    def _repetition_score(words: List[str]) -> float:
        """Fraction of bigrams that appear more than once.

        Returns 0.0 for no repetition, approaches 1.0 for highly repetitive
        text.  Short inputs (< 4 words) return 0.0 to avoid false positives.
        """
        if len(words) < 4:
            return 0.0
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        seen: Dict[str, int] = {}
        for bg in bigrams:
            seen[bg] = seen.get(bg, 0) + 1
        repeated = sum(1 for count in seen.values() if count > 1)
        return repeated / len(bigrams)


# ---------------------------------------------------------------------------
# Adaptive threshold
# ---------------------------------------------------------------------------

class AdaptiveThreshold:
    """A threat threshold that adjusts based on false-positive feedback.

    Starts at a default value (e.g. 0.65) and moves up when the operator
    reports false positives ("that wasn't actually a threat") and creeps
    back down over time if no feedback is received (to avoid becoming
    permanently desensitized).

    Think of it like a smoke detector sensitivity dial that you can nudge
    after a false alarm, but which slowly returns to high sensitivity if
    nothing happens for a while.
    """

    def __init__(
        self,
        initial: float = 0.65,
        floor: float = 0.30,
        ceiling: float = 0.95,
        learning_rate: float = THRESHOLD_LEARNING_RATE,
        recovery_rate: float = 0.001,
    ) -> None:
        self.value: float = initial
        self._floor: float = floor
        self._ceiling: float = ceiling
        self._lr: float = learning_rate
        self._recovery_rate: float = recovery_rate
        self._initial: float = initial
        self._fp_count: int = 0
        self._total_observations: int = 0

    def check(self, threat_probability: float) -> bool:
        """Return True if the threat probability exceeds the current threshold."""
        return threat_probability >= self.value

    def record_false_positive(self) -> None:
        """Nudge the threshold upward because a flagged event was benign.

        Each false positive pushes the threshold slightly higher (less
        sensitive).  The magnitude decays with repeated adjustments so the
        system cannot be gamed into total insensitivity.
        """
        self._fp_count += 1
        adjustment = self._lr / (1.0 + 0.1 * self._fp_count)
        self.value = min(self._ceiling, self.value + adjustment)
        logger.debug(
            "AdaptiveThreshold: false positive acknowledged "
            "(fp_count=%d, threshold=%.4f)",
            self._fp_count, self.value,
        )

    def tick(self) -> None:
        """Called once per observation to let the threshold slowly recover
        toward its initial sensitivity.

        Without this, accumulated false-positive feedback would permanently
        desensitize the detector.
        """
        self._total_observations += 1
        if self.value > self._initial:
            self.value = max(self._initial, self.value - self._recovery_rate)

    def get_stats(self) -> Dict[str, Any]:
        """Diagnostic snapshot for telemetry."""
        return {
            "current_threshold": round(self.value, 4),
            "initial_threshold": self._initial,
            "false_positive_count": self._fp_count,
            "total_observations": self._total_observations,
        }


# ---------------------------------------------------------------------------
# Normality model (running mean + covariance)
# ---------------------------------------------------------------------------

class _NormalityModel:
    """Maintains a decaying statistical model of 'normal' system behavior.

    Internally tracks a weighted running mean and covariance matrix over a
    sliding window of observation vectors.  Older observations are down-
    weighted via exponential decay so the model adapts to legitimate
    behavioral shifts while still remembering enough history to detect
    anomalies.

    This is the mathematical heart of the detector.  Everything else is
    plumbing.
    """

    def __init__(self, dim: int, window_size: int, decay_half_life: int) -> None:
        self._dim: int = dim
        self._window_size: int = window_size
        self._decay_lambda: float = math.log(2.0) / max(1, decay_half_life)

        # Observation buffer: (timestamp, feature_vector) pairs.
        self._buffer: Deque[tuple[float, np.ndarray]] = deque(maxlen=window_size)

        # Cached statistics — recomputed lazily when dirty.
        self._mean: np.ndarray = np.zeros(dim, dtype=np.float64)
        self._cov_inv: Optional[np.ndarray] = None
        self._dirty: bool = True

    @property
    def n_observations(self) -> int:
        return len(self._buffer)

    def push(self, vec: np.ndarray, ts: float) -> None:
        """Add a new observation and mark the model as needing recomputation."""
        self._buffer.append((ts, vec.copy()))
        self._dirty = True

    def push_safe(self, vec: np.ndarray, ts: float) -> None:
        """Add a confirmed-safe observation.

        Same as push() — the distinction exists so callers can semantically
        separate confirmed-safe events.  Both contribute to the baseline.
        """
        self.push(vec, ts)

    def mahalanobis(self, vec: np.ndarray) -> float:
        """Compute the Mahalanobis distance of *vec* from the learned baseline.

        If we don't have enough data for a reliable covariance estimate, we
        fall back to a simpler Euclidean z-score.

        Returns:
            Non-negative float.  Typical 'normal' values are 1-3; anomalies
            are usually > 4.
        """
        if self._dirty:
            self._recompute()

        diff = vec - self._mean

        if self._cov_inv is not None:
            # Full Mahalanobis: sqrt( diff^T * Sigma^{-1} * diff )
            left = diff @ self._cov_inv
            dist_sq = float(left @ diff)
            # Numerical guard — tiny negative values can arise from float error.
            return math.sqrt(max(0.0, dist_sq))
        else:
            # Fallback: Euclidean distance normalized by dimension.
            return float(np.linalg.norm(diff)) / math.sqrt(self._dim)

    def _recompute(self) -> None:
        """Recompute weighted mean and inverse covariance from the buffer."""
        n = len(self._buffer)
        if n == 0:
            self._mean = np.zeros(self._dim, dtype=np.float64)
            self._cov_inv = None
            self._dirty = False
            return

        now = self._buffer[-1][0]  # most recent timestamp

        # Build weight vector: newer observations get higher weight.
        weights = np.empty(n, dtype=np.float64)
        data = np.empty((n, self._dim), dtype=np.float64)
        for i, (ts, vec) in enumerate(self._buffer):
            age = max(0.0, now - ts)
            weights[i] = math.exp(-self._decay_lambda * age)
            data[i] = vec

        # Normalize weights to sum to 1.
        w_sum = weights.sum()
        if w_sum < 1e-12:
            weights[:] = 1.0 / n
        else:
            weights /= w_sum

        # Weighted mean.
        self._mean = weights @ data  # (n,) @ (n, dim) -> (dim,)

        # Weighted covariance — only if we have enough observations.
        if n >= MIN_OBSERVATIONS_FOR_COVARIANCE:
            centered = data - self._mean  # (n, dim)
            # Weighted scatter matrix.
            weighted_centered = centered * weights[:, np.newaxis]  # broadcast
            cov = centered.T @ weighted_centered  # (dim, dim)

            # Regularize: add small identity to ensure invertibility.
            cov += np.eye(self._dim, dtype=np.float64) * COV_REGULARIZATION

            try:
                self._cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                logger.warning(
                    "Covariance matrix singular despite regularization; "
                    "falling back to pseudo-inverse."
                )
                self._cov_inv = np.linalg.pinv(cov)
        else:
            self._cov_inv = None

        self._dirty = False


# ---------------------------------------------------------------------------
# Trajectory tracker
# ---------------------------------------------------------------------------

class _TrajectoryTracker:
    """Tracks the recent trend of anomaly scores to detect drift.

    Rather than only flagging point anomalies ("this one event is weird"),
    we also track whether the system is *drifting* — gradually becoming more
    anomalous over time.  A steady upward slope in anomaly scores is
    concerning even if no single score crosses the threshold.

    Uses simple linear regression over a short recent window.
    """

    def __init__(self, window: int = 30) -> None:
        self._scores: Deque[float] = deque(maxlen=window)

    def push(self, score: float) -> None:
        self._scores.append(score)

    def slope(self) -> float:
        """Compute the least-squares slope of recent anomaly scores.

        Positive slope = system is drifting toward anomalous territory.
        Negative slope = system is calming down.
        Returns 0.0 if insufficient data.
        """
        n = len(self._scores)
        if n < 5:
            return 0.0

        # Simple linear regression: y = a + b*x
        x = np.arange(n, dtype=np.float64)
        y = np.array(self._scores, dtype=np.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        numerator = float(np.sum((x - x_mean) * (y - y_mean)))
        denominator = float(np.sum((x - x_mean) ** 2))

        if abs(denominator) < 1e-12:
            return 0.0

        return numerator / denominator


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Learned, embedding-based anomaly detection for Aura's security layer.

    This is the class that the ICE layer (core/cybernetics/ice_layer.py)
    should instantiate and call instead of its hardcoded threshold logic.

    Lifecycle:
        1. Create an instance (once, at boot).
        2. Call ``await observe(event)`` for every system event.
        3. Call ``get_threat_level()`` at any time to read the current
           continuous threat assessment.
        4. Call ``acknowledge_false_positive(event_id)`` when a flagged event
           turns out to be benign.

    Thread safety:
        All mutation is guarded by an asyncio.Lock so this is safe to call
        from multiple coroutines within the same event loop.

    Example::

        detector = AnomalyDetector()

        # In the ICE layer event handler:
        score = await detector.observe({
            "content": user_message,
            "error_rate": recent_error_rate,
            "resource_pressure": cpu_load,
        })
        if score.is_anomaly:
            await trigger_neural_hardening()

    Integration with consciousness:
        The Mahalanobis distance maps directly to prediction error in the
        free-energy framework.  High distance = high surprise = the system
        "feels" that something is off.  This can feed into the Qualia
        Engine's predictive layer as an additional salience signal.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        decay_half_life: int = DECAY_HALF_LIFE,
    ) -> None:
        self._lock = asyncio.Lock()
        self._extractor = FeatureExtractor()
        self._model = _NormalityModel(
            dim=FEATURE_DIM,
            window_size=window_size,
            decay_half_life=decay_half_life,
        )
        self._threshold = AdaptiveThreshold()
        self._trajectory = _TrajectoryTracker()

        # Recent scores for the public threat-level API.
        self._recent_scores: Deque[float] = deque(maxlen=50)

        # Map of event_id -> feature_vector for false-positive acknowledgement.
        self._pending_events: Dict[str, np.ndarray] = {}
        self._max_pending: int = 200

        self._boot_time: float = time.time()
        self._total_observations: int = 0

        logger.info(
            "AnomalyDetector online (window=%d, decay_half_life=%d, "
            "initial_threshold=%.2f)",
            window_size, decay_half_life, self._threshold.value,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def observe(self, event: Dict[str, Any]) -> AnomalyScore:
        """Observe a system event and return its anomaly assessment.

        This is the primary entry point.  Call it for every user input,
        system event, or error that should be evaluated.

        Args:
            event: A dictionary describing the event.  Recognized keys:
                - "content" or "text": raw text (str)
                - "error_rate": recent error frequency (float 0-1)
                - "resource_pressure": CPU/memory load (float 0-1)
                - "timestamp": epoch float (optional, defaults to now)

        Returns:
            An AnomalyScore with the threat assessment.
        """
        async with self._lock:
            ts = event.get("timestamp", time.time())
            event_id = str(uuid.uuid4())

            # Step 1: Extract features.
            vec = self._extractor.extract(event)

            # Step 2: Compute distance from baseline.
            distance = self._model.mahalanobis(vec)

            # Step 3: Convert distance to threat probability via sigmoid.
            threat_prob = self._sigmoid(distance)

            # Step 4: Update trajectory tracker.
            self._trajectory.push(threat_prob)
            slope = self._trajectory.slope()

            # Step 5: Check adaptive threshold.
            is_anomaly = self._threshold.check(threat_prob)

            # Step 6: Incorporate observation into the baseline model.
            self._model.push(vec, ts)

            # Step 7: Tick the adaptive threshold recovery.
            self._threshold.tick()

            # Step 8: Record for threat-level API and false-positive handling.
            self._recent_scores.append(threat_prob)
            self._total_observations += 1

            if is_anomaly:
                self._pending_events[event_id] = vec
                # Evict oldest pending events if we have too many.
                if len(self._pending_events) > self._max_pending:
                    oldest_key = next(iter(self._pending_events))
                    del self._pending_events[oldest_key]

            score = AnomalyScore(
                event_id=event_id,
                threat_probability=threat_prob,
                mahalanobis_distance=distance,
                feature_vector=vec,
                is_anomaly=is_anomaly,
                trajectory_slope=slope,
                timestamp=ts,
            )

            if is_anomaly:
                logger.warning(
                    "Anomaly detected: threat=%.4f distance=%.4f slope=%.4f "
                    "event_id=%s",
                    threat_prob, distance, slope, event_id,
                )

            return score

    def get_threat_level(self) -> float:
        """Return the current continuous threat level (0.0 - 1.0).

        This is a smoothed aggregate of recent anomaly scores, not just the
        last observation.  It represents the system's overall sense of "how
        threatened do I feel right now."

        The value blends:
            - The exponentially weighted average of recent threat scores
            - The trajectory slope (if things are getting worse, boost it)

        This is the value the ICE layer should use in place of its old
        ``self._threat_level`` field.
        """
        if not self._recent_scores:
            return 0.0

        # Exponentially weighted moving average of recent scores.
        scores = np.array(self._recent_scores, dtype=np.float64)
        n = len(scores)
        alpha = 2.0 / (n + 1)
        weights = np.array([(1.0 - alpha) ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        ewma = float(weights @ scores)

        # Boost by trajectory slope (clamped so it can't dominate).
        slope = self._trajectory.slope()
        slope_boost = max(0.0, slope) * 2.0  # positive slope raises threat
        threat = min(1.0, ewma + slope_boost)

        return max(0.0, threat)

    def acknowledge_false_positive(self, event_id: str) -> None:
        """Mark a previously flagged event as a false positive.

        This does two things:
            1. Pushes the flagged observation into the baseline model as a
               confirmed-safe datapoint, so similar events in the future are
               less likely to trigger.
            2. Nudges the adaptive threshold upward (less sensitive) to
               reduce future false positives.

        Args:
            event_id: The event_id from the AnomalyScore that was flagged.

        Note:
            This is synchronous (no ``await``) because it does not need to
            coordinate with other async operations — it only mutates the
            threshold and adds a safe observation.  The asyncio lock in
            ``observe()`` guards the model state for concurrent access.
        """
        vec = self._pending_events.pop(event_id, None)
        if vec is None:
            logger.debug(
                "acknowledge_false_positive: event_id=%s not found in pending "
                "(may have already been evicted).",
                event_id,
            )
            return

        # Re-inject as safe observation so the model treats this pattern
        # as normal in the future.
        self._model.push_safe(vec, time.time())
        self._threshold.record_false_positive()

        logger.info(
            "False positive acknowledged: event_id=%s — threshold now %.4f",
            event_id, self._threshold.value,
        )

    # ------------------------------------------------------------------
    # Diagnostic / telemetry
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Full diagnostic snapshot for telemetry and introspection.

        Returns a dictionary with model statistics, threshold state,
        trajectory info, and overall health — suitable for publishing to
        the event bus or displaying in a dashboard.
        """
        return {
            "threat_level": round(self.get_threat_level(), 4),
            "total_observations": self._total_observations,
            "model_observations": self._model.n_observations,
            "trajectory_slope": round(self._trajectory.slope(), 4),
            "threshold": self._threshold.get_stats(),
            "uptime_seconds": round(time.time() - self._boot_time, 1),
            "pending_reviews": len(self._pending_events),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(distance: float) -> float:
        """Map Mahalanobis distance to a threat probability in [0, 1].

        Uses a logistic sigmoid centered at SIGMOID_MIDPOINT with steepness
        SIGMOID_STEEPNESS.  A distance of SIGMOID_MIDPOINT maps to 0.5
        threat; distances well below map to ~0; distances well above map
        to ~1.

        The formula:
            p = 1 / (1 + exp(-steepness * (distance - midpoint)))
        """
        z = SIGMOID_STEEPNESS * (distance - SIGMOID_MIDPOINT)
        # Clamp to avoid overflow in exp().
        z = max(-500.0, min(500.0, z))
        return 1.0 / (1.0 + math.exp(-z))
