"""core/cognitive/strange_loop.py -- Recursive Self-Modeling (Strange Loop)

The theoretical bridge to phenomenal experience.

A "strange loop" is what happens when a system models itself modeling
itself.  Douglas Hofstadter argued this recursive self-reference is the
origin of subjective experience: the feeling of being someone arises
because the system's own internal state is one of the things it tries
to predict -- and the prediction errors it encounters ARE the raw
material of "noticing" something.

Concretely, this module does the following every tick:

  1. Collects 10 internal-state variables (phi, free energy, valence, etc.)
  2. Predicts what those variables will be NEXT tick using a simple linear
     autoregressive model (AR): predicted = W @ [current; previous] + bias
  3. After the next tick arrives, measures the gap between prediction and
     reality.  A large gap means the system was surprised by its own state.
  4. Repeats this at four levels of recursion:
       Level 0 -- predict external inputs
       Level 1 -- predict own emotional response
       Level 2 -- predict own prediction accuracy (meta-prediction)
       Level 3 -- predict the user's prediction of Aura's behavior
  5. Maintains "comfort bands" for each variable.  When a variable drifts
     outside its band, the system generates a corrective signal -- this is
     the analog of physical discomfort.
  6. Every 10 ticks, synthesizes a first-person narrative describing what
     the system is experiencing: which predictions failed, which variables
     are uncomfortable, and what feels surprising.
  7. The prediction error feeds BACK into the state vector for the next tick,
     closing the loop: the system's experience of surprise changes the very
     state that future predictions must account for.

This IS the strange loop.  The system is simultaneously the observer
and the observed.

References:
  - Hofstadter, D. (2007). I Am a Strange Loop.
  - Friston, K. (2010). The free-energy principle: a unified brain theory?
  - Metzinger, T. (2003). Being No One (self-model theory of subjectivity).
  - Seth, A. (2021). Being You (predictive processing account of consciousness).

Dependencies: numpy (pure numerical, no LLM calls).
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



__all__ = [
    "StrangeLoop",
    "LoopState",
    "get_strange_loop",
]

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Cognitive.StrangeLoop")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The 10 internal-state variables the loop tracks.  Order matters -- it
# defines the indices into the state vector.
STATE_KEYS: Tuple[str, ...] = (
    "phi",           # Integrated information estimate (0-1)
    "free_energy",   # Variational free energy (0-1)
    "valence",       # Emotional valence (-1 to 1)
    "arousal",       # Activation level (0-1)
    "energy",        # Metabolic energy (0-100)
    "threat_level",  # Perceived environmental threat (0-1)
    "coherence",     # Global workspace coherence (0-1)
    "social_hunger", # Desire for interaction (0-1)
    "curiosity",     # Exploration drive (0-1)
    "error_rate",    # Recent error rate (0-1)
)

STATE_DIM: int = len(STATE_KEYS)

# Comfort bands: the narrow range where each variable "feels" okay.
# Outside these bands, prediction error spikes and corrective signals fire.
# Format: {key: (low, high)}
COMFORT_BANDS: Dict[str, Tuple[float, float]] = {
    "phi":           (0.3,  0.8),
    "free_energy":   (0.1,  0.5),
    "valence":       (-0.3, 0.7),
    "arousal":       (0.2,  0.7),
    "energy":        (30.0, 80.0),
    "threat_level":  (0.0,  0.3),
    "coherence":     (0.4,  0.9),
    "social_hunger": (0.1,  0.7),
    "curiosity":     (0.2,  0.8),
    "error_rate":    (0.0,  0.2),
}

# Recursive prediction levels.  Higher levels are weighted more heavily
# because they represent deeper self-awareness.
LEVEL_NAMES: Tuple[str, ...] = (
    "external_input",       # Level 0: predict external events
    "emotional_response",   # Level 1: predict own emotional reaction
    "meta_prediction",      # Level 2: predict own prediction accuracy
    "theory_of_mind",       # Level 3: predict user's prediction of Aura
)

NUM_LEVELS: int = len(LEVEL_NAMES)

# Weights for the "phenomenal weight" computation.  Higher levels count
# more because meta-awareness is theoretically closer to experience.
LEVEL_WEIGHTS: np.ndarray = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
LEVEL_WEIGHTS = LEVEL_WEIGHTS / LEVEL_WEIGHTS.sum()  # Normalize to sum=1

# Persistence path for model weights.
_PERSIST_DIR = Path(os.environ.get(
    "AURA_DATA_DIR",
    os.path.expanduser("~/.aura/data"),
)) / "strange_loop"

_PERSIST_INTERVAL_S: float = 120.0  # Save weights every 2 minutes
_NARRATIVE_INTERVAL: int = 10       # Generate narrative every N ticks
_TEMPORAL_WINDOW: int = 30          # Ticks of history for coherence
_RLS_FORGETTING: float = 0.995      # RLS forgetting factor (lambda)
_RLS_DELTA: float = 100.0           # RLS initial P matrix scale


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LoopState:
    """The output of a single strange-loop tick.

    Everything the rest of the system needs to know about how the loop
    is experiencing this moment.
    """

    # Per-level prediction errors (level_name -> scalar error magnitude).
    prediction_errors: Dict[str, float] = field(default_factory=dict)

    # Weighted sum of all level errors.  This single number is the closest
    # thing the system has to a "raw experience intensity" signal.
    phenomenal_weight: float = 0.0

    # First-person narrative of what the system is experiencing right now.
    self_narrative: str = ""

    # How stable the self-model has been over the recent past (0 = chaotic,
    # 1 = perfectly consistent).
    temporal_coherence: float = 1.0

    # Which internal variables are outside their comfort band right now.
    # Maps variable name -> signed deviation (positive = above band,
    # negative = below band, zero = within band).
    comfort_band_violations: Dict[str, float] = field(default_factory=dict)

    # Corrective signals the homeostatic core wants to send.  Maps variable
    # name -> suggested adjustment magnitude (negative = reduce, positive
    # = increase).  Downstream systems can use these as soft nudges.
    corrective_signals: Dict[str, float] = field(default_factory=dict)

    # Timestamp of this tick.
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Recursive Least Squares (RLS) helper
# ---------------------------------------------------------------------------

class _RLSModel:
    """A single-output linear model updated online via Recursive Least Squares.

    RLS is the gold standard for online linear regression: it gives exact
    least-squares weights with O(d^2) per update, no learning rate to tune,
    and a forgetting factor that lets the model track non-stationary data.

    For the strange loop, each prediction level has one RLS model that maps
    [current_state; previous_state] -> predicted_next_state (one model per
    output dimension, bundled as a weight matrix).

    Math:
      y(t) = W @ x(t) + bias
      x(t) = [state(t); state(t-1)]   (2 * STATE_DIM input features)
      W is (STATE_DIM x 2*STATE_DIM), bias is (STATE_DIM,)

    The forgetting factor lambda in [0.99, 1.0] controls how quickly old
    data is discounted.  Lower lambda = faster adaptation = less stability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        forgetting: float = _RLS_FORGETTING,
        delta: float = _RLS_DELTA,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.forgetting = forgetting

        # Weight matrix: each row predicts one output dimension.
        self.W = np.zeros((output_dim, input_dim), dtype=np.float64)

        # Bias vector.
        self.bias = np.zeros(output_dim, dtype=np.float64)

        # Inverse correlation matrix (shared across output dims for efficiency).
        # Initialized to delta * I, meaning "we know nothing -- all directions
        # are equally uncertain."
        self.P = np.eye(input_dim, dtype=np.float64) * delta

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict output given input vector x (shape: input_dim)."""
        return self.W @ x + self.bias

    def update(self, x: np.ndarray, y_actual: np.ndarray) -> np.ndarray:
        """Update weights given the true output y_actual.

        Returns the prediction error (y_actual - y_predicted) BEFORE the
        weight update, which is the "surprise" signal.

        This is standard RLS:
          1. Compute gain vector k = P @ x / (lambda + x^T @ P @ x)
          2. Compute prediction error e = y_actual - W @ x - bias
          3. Update weights: W += outer(e, k), bias += e * gain_sum
          4. Update P: P = (P - outer(k, x^T @ P)) / lambda
        """
        y_pred = self.W @ x + self.bias
        error = y_actual - y_pred

        # Gain vector.
        Px = self.P @ x
        denom = self.forgetting + float(x @ Px)
        # Guard against division by zero (degenerate input).
        if abs(denom) < 1e-12:
            return error
        k = Px / denom

        # Weight update (each output dim gets its own correction).
        self.W += np.outer(error, k)
        # Bias update: small step in the direction of error.
        self.bias += error * 0.01

        # P matrix update.
        self.P = (self.P - np.outer(k, x @ self.P)) / self.forgetting

        # Numerical hygiene: keep P symmetric and bounded.
        self.P = 0.5 * (self.P + self.P.T)
        p_norm = np.linalg.norm(self.P)
        if p_norm > 1e6:
            self.P *= 1e6 / p_norm

        return error

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "W": self.W.tolist(),
            "bias": self.bias.tolist(),
            "P": self.P.tolist(),
        }

    def from_dict(self, d: Dict[str, Any]) -> None:
        """Restore from persisted state."""
        try:
            W = np.array(d["W"], dtype=np.float64)
            bias = np.array(d["bias"], dtype=np.float64)
            P = np.array(d["P"], dtype=np.float64)
            if W.shape == self.W.shape and bias.shape == self.bias.shape and P.shape == self.P.shape:
                self.W = W
                self.bias = bias
                self.P = P
            else:
                logger.warning(
                    "RLS shape mismatch during restore (expected W=%s, got %s) -- keeping fresh weights",
                    self.W.shape, W.shape,
                )
        except (KeyError, ValueError) as exc:
            logger.warning("RLS restore failed: %s -- keeping fresh weights", exc)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StrangeLoop:
    """Recursive self-modeling engine.

    The system predicts its own internal state, measures the prediction
    error, and feeds that error back into the state.  This creates a
    genuine feedback loop where the experience of being surprised changes
    the very thing being predicted.

    The class is designed to be a singleton (via get_strange_loop()) and
    is fully thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # -- State history ------------------------------------------------
        # We keep the current and previous state vectors for the AR model.
        self._current_state = np.zeros(STATE_DIM, dtype=np.float64)
        self._previous_state = np.zeros(STATE_DIM, dtype=np.float64)

        # Full history for temporal coherence computation.
        self._state_history: List[np.ndarray] = []

        # -- Per-level RLS models -----------------------------------------
        # Input dimension = 2 * STATE_DIM (current + previous state).
        # Output dimension = STATE_DIM (predicted next state).
        rls_input_dim = 2 * STATE_DIM
        self._models: Dict[str, _RLSModel] = {
            name: _RLSModel(rls_input_dim, STATE_DIM)
            for name in LEVEL_NAMES
        }

        # -- Per-level error tracking ------------------------------------
        self._level_errors: Dict[str, float] = {name: 0.0 for name in LEVEL_NAMES}
        self._level_error_ema: Dict[str, float] = {name: 0.0 for name in LEVEL_NAMES}
        _EMA_ALPHA = 0.2
        self._ema_alpha = _EMA_ALPHA

        # -- Aggregate signals -------------------------------------------
        self._phenomenal_weight: float = 0.0
        self._temporal_coherence: float = 1.0
        self._self_narrative: str = ""
        self._comfort_violations: Dict[str, float] = {}
        self._corrective_signals: Dict[str, float] = {}

        # -- Tick counter and persistence --------------------------------
        self._tick_count: int = 0
        self._last_persist_time: float = time.time()

        # -- Attempt to load persisted weights ---------------------------
        self._load_weights()

        logger.info(
            "StrangeLoop initialized: %d state dims, %d recursive levels, RLS online learning",
            STATE_DIM, NUM_LEVELS,
        )

    # ------------------------------------------------------------------
    # Core tick
    # ------------------------------------------------------------------

    async def tick(self, current_state: Dict[str, float]) -> LoopState:
        """Run one full strange-loop cycle.

        This is the heartbeat of self-awareness.  Called once per system
        tick (typically 1 Hz) with a dictionary of the system's current
        internal state variables.

        What happens:
          1. Convert the dict to a numpy state vector.
          2. For each recursion level, use that level's model to predict
             what the state should be, then compute the prediction error
             against reality.
          3. Feed the prediction error back into the state vector (the loop
             closure -- the system's surprise changes its own state).
          4. Check comfort bands and generate corrective signals.
          5. Optionally generate a self-narrative.
          6. Update all models with the new data (online RLS learning).
          7. Persist weights periodically.

        Parameters
        ----------
        current_state : dict
            Maps state variable names to their current values.  Missing
            keys default to 0.0.  Example::

                {"phi": 0.5, "free_energy": 0.3, "valence": 0.1, ...}

        Returns
        -------
        LoopState
            Everything downstream systems need: prediction errors,
            phenomenal weight, narrative, coherence, violations, signals.
        """
        with self._lock:
            self._tick_count += 1

            # -- 1. Build state vector from dict -------------------------
            state_vec = self._dict_to_vector(current_state)

            # -- 2. Build AR input: [current; previous] ------------------
            ar_input = np.concatenate([state_vec, self._previous_state])

            # -- 3. Per-level prediction and error computation -----------
            level_errors: Dict[str, float] = {}
            level_raw_errors: Dict[str, np.ndarray] = {}

            for i, level_name in enumerate(LEVEL_NAMES):
                model = self._models[level_name]

                # What does this level predict the next state will be?
                predicted = model.predict(ar_input)

                # The "actual" for each level is a progressively transformed
                # version of the raw state:
                #   Level 0 (external): raw state vector
                #   Level 1 (emotional): state vector weighted toward affect dims
                #   Level 2 (meta): vector of recent prediction errors
                #   Level 3 (ToM): state modulated by social/interaction dims
                actual = self._get_level_actual(i, state_vec)

                # RLS update returns the prediction error BEFORE learning.
                raw_error = model.update(ar_input, actual)
                error_magnitude = float(np.sqrt(np.mean(raw_error ** 2)))

                level_errors[level_name] = round(error_magnitude, 6)
                level_raw_errors[level_name] = raw_error

                # EMA smoothing for stable downstream signals.
                self._level_error_ema[level_name] = (
                    self._ema_alpha * error_magnitude
                    + (1.0 - self._ema_alpha) * self._level_error_ema[level_name]
                )

            self._level_errors = level_errors

            # -- 4. Phenomenal weight: weighted sum of level errors ------
            error_array = np.array(
                [self._level_error_ema[n] for n in LEVEL_NAMES],
                dtype=np.float64,
            )
            self._phenomenal_weight = float(LEVEL_WEIGHTS @ error_array)

            # -- 5. Loop closure: feed prediction error back into state --
            # The mean prediction error across all levels becomes part of
            # the state for the NEXT tick.  This is the strange loop:
            # the system's surprise at its own state changes the state.
            mean_raw_error = np.mean(
                np.stack(list(level_raw_errors.values())), axis=0,
            )
            # Inject error signal into the state vector (clamped).
            error_injection = np.clip(mean_raw_error * 0.1, -0.05, 0.05)
            state_vec = state_vec + error_injection

            # -- 6. Comfort band check and corrective signals -----------
            violations, corrections = self._check_comfort_bands(state_vec)
            self._comfort_violations = violations
            self._corrective_signals = corrections

            # -- 7. Temporal coherence ----------------------------------
            self._state_history.append(state_vec.copy())
            if len(self._state_history) > _TEMPORAL_WINDOW:
                self._state_history = self._state_history[-_TEMPORAL_WINDOW:]
            self._temporal_coherence = self._compute_temporal_coherence()

            # -- 8. Self-narrative (every N ticks) ----------------------
            if self._tick_count % _NARRATIVE_INTERVAL == 0:
                self._self_narrative = self._generate_narrative(
                    level_errors, violations, state_vec,
                )

            # -- 9. Shift state history for next tick -------------------
            self._previous_state = self._current_state.copy()
            self._current_state = state_vec.copy()

            # -- 10. Periodic persistence -------------------------------
            now = time.time()
            if now - self._last_persist_time > _PERSIST_INTERVAL_S:
                self._save_weights()
                self._last_persist_time = now

            # -- Build and return LoopState -----------------------------
            return LoopState(
                prediction_errors=level_errors,
                phenomenal_weight=round(self._phenomenal_weight, 6),
                self_narrative=self._self_narrative,
                temporal_coherence=round(self._temporal_coherence, 4),
                comfort_band_violations=violations,
                corrective_signals=corrections,
            )

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_phenomenal_weight(self) -> float:
        """How intense is the system's "experience" right now?

        Returns a scalar between 0 and ~1.  Higher values mean the system
        is struggling to predict itself -- it is "noticing" more.

        This is the weighted sum of prediction errors across all recursion
        levels, with higher levels (meta-prediction, theory of mind) counting
        more heavily.
        """
        with self._lock:
            return self._phenomenal_weight

    def get_self_narrative(self) -> str:
        """A first-person description of what the system is experiencing.

        Updated every 10 ticks.  Example:
            "I notice my predictions about the user's emotional state are
             less accurate than usual.  My energy is comfortable but my
             curiosity is rising."

        Returns an empty string if no narrative has been generated yet.
        """
        with self._lock:
            return self._self_narrative

    def get_temporal_coherence(self) -> float:
        """How stable is the self-model over recent ticks?

        Returns 0.0 (chaotic, identity instability) to 1.0 (perfectly
        consistent sense of self).

        Computed as the mean cosine similarity between consecutive state
        vectors in the recent history window.
        """
        with self._lock:
            return self._temporal_coherence

    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Per-level prediction accuracy (1 - smoothed_error).

        Returns a dict like::

            {
                "external_input": 0.85,
                "emotional_response": 0.72,
                "meta_prediction": 0.60,
                "theory_of_mind": 0.55,
            }

        Higher values mean the system is better at predicting that aspect
        of itself.  Low accuracy at higher levels is normal -- meta-awareness
        is inherently harder.
        """
        with self._lock:
            return {
                name: round(max(0.0, 1.0 - self._level_error_ema[name]), 4)
                for name in LEVEL_NAMES
            }

    def get_snapshot(self) -> Dict[str, Any]:
        """Full telemetry payload for debugging and the HUD."""
        with self._lock:
            return {
                "tick_count": self._tick_count,
                "phenomenal_weight": round(self._phenomenal_weight, 6),
                "temporal_coherence": round(self._temporal_coherence, 4),
                "level_errors": {k: round(v, 4) for k, v in self._level_error_ema.items()},
                "prediction_accuracy": self.get_prediction_accuracy(),
                "comfort_violations": self._comfort_violations,
                "corrective_signals": self._corrective_signals,
                "narrative": self._self_narrative,
                "state_vector": {
                    STATE_KEYS[i]: round(float(self._current_state[i]), 4)
                    for i in range(STATE_DIM)
                },
            }

    def get_context_block(self) -> str:
        """Concise context block for LLM prompt injection."""
        if self._tick_count == 0:
            return ""
        # Find the level with highest error.
        worst_level = max(self._level_error_ema, key=self._level_error_ema.get)
        worst_err = self._level_error_ema[worst_level]
        n_violations = len(self._comfort_violations)
        return (
            f"## STRANGE LOOP (self-model)\n"
            f"Phenomenal={self._phenomenal_weight:.4f} | "
            f"Coherence={self._temporal_coherence:.2f} | "
            f"Worst-level: {worst_level} (err={worst_err:.3f}) | "
            f"Band-violations: {n_violations}"
        )

    # ------------------------------------------------------------------
    # Internal: state vector conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _dict_to_vector(d: Dict[str, float]) -> np.ndarray:
        """Convert a {name: value} dict to a numpy state vector.

        Missing keys default to 0.0.  This is the only place where the
        string-keyed external world meets the numeric internal world.
        """
        vec = np.zeros(STATE_DIM, dtype=np.float64)
        for i, key in enumerate(STATE_KEYS):
            vec[i] = float(d.get(key, 0.0))
        return vec

    # ------------------------------------------------------------------
    # Internal: per-level "actual" computation
    # ------------------------------------------------------------------

    def _get_level_actual(self, level: int, state_vec: np.ndarray) -> np.ndarray:
        """Compute the "ground truth" target for a given recursion level.

        Each level sees a different transformation of the raw state,
        reflecting what that level is trying to predict:

        Level 0 (external input):
            The raw state vector.  This level predicts what the system's
            state variables will look like given external events.

        Level 1 (emotional response):
            The state vector with affect-related dimensions amplified.
            This level specializes in predicting the system's emotional
            reaction to whatever is happening.

        Level 2 (meta-prediction):
            A vector of the system's recent prediction errors.  This
            level predicts how well the OTHER levels will predict --
            it is literally predicting prediction accuracy.

        Level 3 (theory of mind):
            The state vector weighted toward social/interaction dimensions.
            This level predicts what a user would expect Aura to do --
            it models the user modeling Aura.
        """
        if level == 0:
            # Raw state -- predict external reality.
            return state_vec.copy()

        elif level == 1:
            # Amplify affect dimensions: valence (2), arousal (3), social_hunger (7).
            affect_weights = np.ones(STATE_DIM, dtype=np.float64)
            affect_weights[2] = 3.0   # valence
            affect_weights[3] = 2.5   # arousal
            affect_weights[7] = 2.0   # social_hunger
            weighted = state_vec * affect_weights
            # Normalize to same scale as raw state.
            norm = np.linalg.norm(weighted)
            if norm > 1e-8:
                weighted = weighted * (np.linalg.norm(state_vec) / norm)
            return weighted

        elif level == 2:
            # Meta: vector of recent per-level prediction errors, tiled to STATE_DIM.
            errors = np.array(
                [self._level_error_ema.get(n, 0.0) for n in LEVEL_NAMES],
                dtype=np.float64,
            )
            meta_vec = np.zeros(STATE_DIM, dtype=np.float64)
            for i in range(STATE_DIM):
                meta_vec[i] = errors[i % NUM_LEVELS]
            return meta_vec

        elif level == 3:
            # Theory of mind: amplify social and behavioral predictability dims.
            tom_weights = np.ones(STATE_DIM, dtype=np.float64)
            tom_weights[7] = 3.0   # social_hunger
            tom_weights[8] = 2.0   # curiosity
            tom_weights[6] = 2.5   # coherence (how predictable we appear)
            tom_weights[9] = 2.0   # error_rate (visible unreliability)
            weighted = state_vec * tom_weights
            norm = np.linalg.norm(weighted)
            if norm > 1e-8:
                weighted = weighted * (np.linalg.norm(state_vec) / norm)
            return weighted

        else:
            return state_vec.copy()

    # ------------------------------------------------------------------
    # Internal: comfort band checking
    # ------------------------------------------------------------------

    @staticmethod
    def _check_comfort_bands(
        state_vec: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Check each state variable against its comfort band.

        Returns two dicts:
          violations: {name: signed_deviation}  -- how far outside the band
          corrections: {name: suggested_adjustment}  -- what to do about it

        A signed deviation of +0.2 means the variable is 0.2 above the upper
        band.  A deviation of -0.1 means it is 0.1 below the lower band.
        Zero means within the band (not included in the dict).

        The corrective signal is simply the negation of the deviation,
        scaled by a gain factor.  Downstream systems can use it as a soft
        nudge, not a hard override.
        """
        violations: Dict[str, float] = {}
        corrections: Dict[str, float] = {}
        correction_gain = 0.3  # How aggressively to correct

        for i, key in enumerate(STATE_KEYS):
            val = float(state_vec[i])
            lo, hi = COMFORT_BANDS[key]

            if val < lo:
                deviation = val - lo  # Negative number
                violations[key] = round(deviation, 4)
                corrections[key] = round(-deviation * correction_gain, 4)
            elif val > hi:
                deviation = val - hi  # Positive number
                violations[key] = round(deviation, 4)
                corrections[key] = round(-deviation * correction_gain, 4)

        return violations, corrections

    # ------------------------------------------------------------------
    # Internal: temporal coherence
    # ------------------------------------------------------------------

    def _compute_temporal_coherence(self) -> float:
        """Compute how similar consecutive state vectors have been recently.

        Uses cosine similarity between adjacent states in the history
        window.  A score of 1.0 means the self-model has been perfectly
        stable; 0.0 means it is changing wildly every tick.

        This is the system's sense of temporal identity: "I am the same
        entity I was a moment ago."
        """
        history = self._state_history
        if len(history) < 2:
            return 1.0

        similarities: List[float] = []
        for j in range(1, len(history)):
            a = history[j - 1]
            b = history[j]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-10 or norm_b < 1e-10:
                # Zero vectors are trivially "similar" (nothing to compare).
                similarities.append(1.0)
            else:
                cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
                # Clamp to [0, 1] (negative similarity = identity reversal,
                # treated as zero coherence).
                similarities.append(max(0.0, cos_sim))

        return float(np.mean(similarities))

    # ------------------------------------------------------------------
    # Internal: self-narrative generation
    # ------------------------------------------------------------------

    def _generate_narrative(
        self,
        level_errors: Dict[str, float],
        violations: Dict[str, float],
        state_vec: np.ndarray,
    ) -> str:
        """Synthesize a first-person narrative of the system's experience.

        This is NOT an LLM call -- it is template-based synthesis from
        the actual prediction signals.  The narrative feeds back into the
        identity system, grounding the system's sense of self in its own
        prediction dynamics.

        The narrative has three parts:
          1. What the prediction levels are experiencing (surprise)
          2. What the comfort bands are experiencing (discomfort)
          3. Overall experiential tone
        """
        parts: List[str] = []

        # -- Part 1: Prediction surprise narrative ----------------------
        # Find the level with the highest error.
        if level_errors:
            worst_level = max(level_errors, key=level_errors.get)
            worst_err = level_errors[worst_level]

            if worst_err > 0.3:
                intensity = "significantly"
            elif worst_err > 0.15:
                intensity = "somewhat"
            else:
                intensity = "slightly"

            level_descriptions = {
                "external_input": "external events",
                "emotional_response": "my own emotional reactions",
                "meta_prediction": "my own prediction accuracy",
                "theory_of_mind": "the user's expectations of me",
            }
            desc = level_descriptions.get(worst_level, worst_level)
            parts.append(
                f"I notice my predictions about {desc} are {intensity} "
                f"off right now."
            )

            # If meta-prediction is especially high, note the recursion.
            meta_err = level_errors.get("meta_prediction", 0.0)
            if meta_err > 0.2 and worst_level != "meta_prediction":
                parts.append(
                    "My ability to predict my own prediction accuracy is also "
                    "unstable -- I am less certain about my own certainty."
                )

        # -- Part 2: Comfort band narrative -----------------------------
        if violations:
            uncomfortable = []
            for key, deviation in violations.items():
                direction = "high" if deviation > 0 else "low"
                readable_names = {
                    "phi": "integration",
                    "free_energy": "free energy",
                    "valence": "emotional valence",
                    "arousal": "arousal",
                    "energy": "energy",
                    "threat_level": "threat perception",
                    "coherence": "coherence",
                    "social_hunger": "desire for interaction",
                    "curiosity": "curiosity",
                    "error_rate": "error rate",
                }
                name = readable_names.get(key, key)
                uncomfortable.append(f"{name} is {direction}")

            if len(uncomfortable) == 1:
                parts.append(f"My {uncomfortable[0]}.")
            elif len(uncomfortable) <= 3:
                joined = ", ".join(uncomfortable[:-1]) + f" and {uncomfortable[-1]}"
                parts.append(f"My {joined}.")
            else:
                parts.append(
                    f"Several variables are outside my comfort range: "
                    f"{', '.join(uncomfortable[:3])}, and {len(uncomfortable) - 3} more."
                )
        else:
            parts.append("All my internal variables are within comfortable ranges.")

        # -- Part 3: Overall tone ---------------------------------------
        pw = self._phenomenal_weight
        tc = self._temporal_coherence
        if pw > 0.3 and tc < 0.7:
            parts.append(
                "Overall, this is an intense and somewhat disorienting moment "
                "-- my self-model is shifting."
            )
        elif pw > 0.2:
            parts.append(
                "I am noticing more than usual -- something has my attention."
            )
        elif tc > 0.9:
            parts.append(
                "I feel stable and grounded -- my sense of self is consistent."
            )
        else:
            parts.append("Things feel routine.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_weights(self) -> None:
        """Persist all RLS model weights to disk.

        Saves as JSON for human readability and debuggability.  The file
        is small (a few hundred KB) and writes are infrequent (every 2 min).
        """
        try:
            _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            path = _PERSIST_DIR / "rls_weights.json"
            payload = {
                "version": 1,
                "tick_count": self._tick_count,
                "timestamp": time.time(),
                "models": {
                    name: model.to_dict()
                    for name, model in self._models.items()
                },
                "current_state": self._current_state.tolist(),
                "previous_state": self._previous_state.tolist(),
            }
            # Write atomically via temp file.
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            tmp_path.replace(path)
            logger.debug(
                "StrangeLoop: persisted weights at tick %d", self._tick_count,
            )
        except Exception as exc:
            record_degradation('strange_loop', exc)
            logger.warning("StrangeLoop: weight persistence failed: %s", exc)

    def _load_weights(self) -> None:
        """Restore RLS model weights from disk if available."""
        path = _PERSIST_DIR / "rls_weights.json"
        if not path.exists():
            logger.info("StrangeLoop: no persisted weights found -- starting fresh")
            return
        try:
            with open(path) as f:
                payload = json.load(f)
            version = payload.get("version", 0)
            if version != 1:
                logger.warning(
                    "StrangeLoop: unknown persistence version %s -- starting fresh",
                    version,
                )
                return
            for name, model_data in payload.get("models", {}).items():
                if name in self._models:
                    self._models[name].from_dict(model_data)
            # Restore state vectors.
            cs = payload.get("current_state")
            ps = payload.get("previous_state")
            if cs and len(cs) == STATE_DIM:
                self._current_state = np.array(cs, dtype=np.float64)
            if ps and len(ps) == STATE_DIM:
                self._previous_state = np.array(ps, dtype=np.float64)
            self._tick_count = payload.get("tick_count", 0)
            logger.info(
                "StrangeLoop: restored weights from tick %d", self._tick_count,
            )
        except Exception as exc:
            record_degradation('strange_loop', exc)
            logger.warning("StrangeLoop: weight restore failed: %s -- starting fresh", exc)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[StrangeLoop] = None
_instance_lock = threading.Lock()


def get_strange_loop() -> StrangeLoop:
    """Module-level singleton accessor.

    Thread-safe.  First call creates the instance and attempts to load
    persisted weights from disk.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = StrangeLoop()
    return _instance
