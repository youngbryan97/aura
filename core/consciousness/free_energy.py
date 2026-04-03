"""
Active Inference engine for Aura.

Free energy F = Surprise + Complexity
            = -log P(observations | beliefs) + KL(posterior || prior)

In practice for Aura:
  - Surprise    = prediction error from SelfPredictionLoop / PredictiveEngine
  - Complexity  = divergence between current beliefs and prior beliefs
                  + system load (CPU/mem) + attention scatter
  - Free energy = the thing Aura is always trying to minimize

High F -> distress, urgency, drive to act or update beliefs
Low F  -> comfort, openness, exploratory curiosity

Deepened Implementation:
  - accept_surprise_signal(): Direct coupling from PredictiveEngine
  - accept_attention_complexity(): Direct coupling from AttentionSchema
  - get_context_block(): For LLM prompt injection
  - get_action_urgency(): Scalar urgency signal for heartbeat/GWT
  - Richer action determination with hysteresis
"""

import math
import time
import logging
import psutil
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class FreeEnergyState:
    surprise: float           # Prediction error component (0-1)
    complexity: float         # Belief divergence component (0-1)
    free_energy: float        # Combined F (0-1, lower is better)
    valence: float            # Emotional valence (-1 to 1)
    arousal: float            # Activation level (0-1)
    dominant_action: str      # What F-minimization wants to do:
                              # "update_beliefs" | "act_on_world" | "explore" | "rest" | "engage" | "reflect"
    timestamp: float = field(default_factory=time.time)


class FreeEnergyEngine:
    """
    The engine that makes Aura's behavior emerge from first principles
    rather than from timers and thresholds.
    """

    _HISTORY = 120  # 2 minutes of state history

    def __init__(self):
        self._history: Deque[FreeEnergyState] = deque(maxlen=self._HISTORY)
        self._belief_baseline: Dict[str, float] = {}  # Prior belief confidences
        self._current: Optional[FreeEnergyState] = None
        self._smoothed_fe: float = 0.3  # Start at moderate free energy
        self._alpha = 0.15  # EMA smoothing

        # ── Direct Signal Inputs ──────────────────────────────────────────
        self._last_surprise_signal: float = 0.0       # From PredictiveEngine
        self._last_attention_complexity: float = 0.0  # From AttentionSchema
        self._surprise_signal_age: float = 0.0        # When last signal arrived

        # ── Action Hysteresis ─────────────────────────────────────────────
        # Prevent rapid action-switching: keep the current action for at
        # least _ACTION_HOLD_TICKS compute cycles unless FE changes dramatically
        self._current_action: str = "rest"
        self._action_hold_ticks: int = 0
        self._ACTION_HOLD_MIN: int = 5  # Minimum ticks before action switch
        self._ACTION_SWITCH_THRESHOLD: float = 0.15  # FE delta to force immediate switch

        # ── Cumulative Metrics ────────────────────────────────────────────
        self._total_computes: int = 0
        self._action_counts: Dict[str, int] = {}
        self._peak_fe: float = 0.0

        logger.info("Free Energy Engine initialized (Active Inference mode)")

    # ──────────────────────────────────────────────────────────────────────
    # Core Computation
    # ──────────────────────────────────────────────────────────────────────

    def compute(
        self,
        prediction_error: float,          # From SelfPredictionLoop.get_surprise_signal()
        belief_system=None,               # BeliefSystem / EpistemicState instance
        recent_action_count: int = 0,     # Actions taken in last minute
        user_present: bool = False,
        telemetry: Dict[str, Any] = None  # Optional manual telemetry override
    ) -> FreeEnergyState:
        """
        Compute current free energy and determine what it wants Aura to do.
        """
        self._total_computes += 1

        # 1. Surprise component — blend direct signal with passed prediction_error
        raw_surprise = max(0.0, min(1.0, prediction_error))
        if time.time() - self._surprise_signal_age < 10.0:
            # Fresh direct signal from PredictiveEngine — blend it in
            raw_surprise = 0.6 * raw_surprise + 0.4 * self._last_surprise_signal
        surprise = raw_surprise

        # 2. Complexity = belief divergence + system load + attention scatter
        system_complexity = self._compute_system_complexity()
        belief_complexity = self._compute_belief_complexity(belief_system)
        attention_complexity = self._last_attention_complexity  # From AttentionSchema
        complexity = (
            0.50 * belief_complexity
            + 0.25 * system_complexity
            + 0.25 * attention_complexity
        )

        # 3. Free energy = weighted sum (Surprise dominant)
        fe_raw = 0.6 * surprise + 0.4 * complexity

        # 4. Smooth it (avoid jitter)
        self._smoothed_fe = (
            self._alpha * fe_raw + (1 - self._alpha) * self._smoothed_fe
        )
        fe = self._smoothed_fe

        # Track peak
        if fe > self._peak_fe:
            self._peak_fe = fe

        # 5. Derive valence and arousal from free energy
        valence = 1.0 - 2.0 * fe  # Maps [0,1] -> [1,-1]
        arousal = fe * 0.8 + 0.1   # Maps [0,1] -> [0.1, 0.9]

        # 6. Determine dominant action tendency (with hysteresis)
        dominant_action = self._determine_action_with_hysteresis(
            fe, surprise, complexity, recent_action_count, user_present
        )

        # Track action counts
        self._action_counts[dominant_action] = self._action_counts.get(dominant_action, 0) + 1

        state = FreeEnergyState(
            surprise=surprise,
            complexity=complexity,
            free_energy=fe,
            valence=round(valence, 3),
            arousal=round(arousal, 3),
            dominant_action=dominant_action,
        )
        self._history.append(state)
        self._current = state
        return state

    # ──────────────────────────────────────────────────────────────────────
    # Direct Signal Acceptance (from other modules)
    # ──────────────────────────────────────────────────────────────────────

    def accept_surprise_signal(self, surprise: float):
        """Called by PredictiveEngine when it computes a new surprise value.
        This bypasses the need to wait for the next compute() cycle.
        """
        self._last_surprise_signal = max(0.0, min(1.0, float(surprise)))
        self._surprise_signal_age = time.time()

    def accept_attention_complexity(self, complexity: float):
        """Called by AttentionSchema with its coherence-inverted complexity.
        Scattered attention = high complexity = contributes to FE.
        """
        self._last_attention_complexity = max(0.0, min(1.0, float(complexity)))

    # ──────────────────────────────────────────────────────────────────────
    # Complexity Components
    # ──────────────────────────────────────────────────────────────────────

    def _compute_belief_complexity(self, belief_system) -> float:
        if belief_system is None:
            return 0.1

        try:
            # Support both BeliefGraph and EpistemicState
            if hasattr(belief_system, 'graph'):
                beliefs = []
                for u, v, d in belief_system.graph.edges(data=True):
                    beliefs.append({"key": f"{u}->{v}", "confidence": d.get('confidence', 0.5)})
            elif hasattr(belief_system, 'world_graph'):
                beliefs = []
                for u, v, d in belief_system.world_graph.edges(data=True):
                    beliefs.append({"key": f"{u}->{v}", "confidence": d.get('confidence', 0.5)})
            else:
                beliefs = getattr(belief_system, 'beliefs', [])

            if not beliefs:
                return 0.1

            total_divergence = 0.0
            for b in beliefs:
                key = b["key"] if isinstance(b, dict) else b.content[:50]
                conf = b["confidence"] if isinstance(b, dict) else b.confidence
                prior_conf = self._belief_baseline.get(key, 0.5)
                divergence = abs(conf - prior_conf)
                total_divergence += divergence

            complexity = min(1.0, total_divergence / max(len(beliefs), 1))

            # Update baseline slowly
            if len(self._history) % 60 == 0:
                for b in beliefs:
                    key = b["key"] if isinstance(b, dict) else b.content[:50]
                    conf = b["confidence"] if isinstance(b, dict) else b.confidence
                    old = self._belief_baseline.get(key, conf)
                    self._belief_baseline[key] = 0.95 * old + 0.05 * conf

            return complexity
        except Exception as e:
            logger.debug("Belief complexity compute failed: %s", e)
            return 0.1

    def _compute_system_complexity(self) -> float:
        """Privileged Internal Telemetry.
        Maps CPU/Mem load into 'Complexity' (internal entropy).
        """
        try:
            cpu = psutil.cpu_percent() / 100.0
            mem = psutil.virtual_memory().percent / 100.0
            return (cpu * 0.7 + mem * 0.3)
        except Exception:
            return 0.1

    # ──────────────────────────────────────────────────────────────────────
    # Action Determination (with hysteresis)
    # ──────────────────────────────────────────────────────────────────────

    def _determine_action_with_hysteresis(
        self,
        fe: float,
        surprise: float,
        complexity: float,
        recent_actions: int,
        user_present: bool,
    ) -> str:
        """Determine the dominant action tendency with hysteresis to
        prevent rapid oscillation between states.
        """
        candidate = self._raw_action_determination(
            fe, surprise, complexity, recent_actions, user_present
        )

        # Check if we should switch
        self._action_hold_ticks += 1

        if candidate != self._current_action:
            # Force switch if FE changed dramatically
            prev_fe = self._history[-2].free_energy if len(self._history) >= 2 else fe
            fe_delta = abs(fe - prev_fe)

            if (fe_delta > self._ACTION_SWITCH_THRESHOLD
                    or self._action_hold_ticks >= self._ACTION_HOLD_MIN):
                self._current_action = candidate
                self._action_hold_ticks = 0

        return self._current_action

    def _raw_action_determination(
        self,
        fe: float,
        surprise: float,
        complexity: float,
        recent_actions: int,
        user_present: bool,
    ) -> str:
        # If user is actively talking, prioritize engagement
        if user_present:
            return "engage"

        # Very high surprise -> update beliefs first
        if surprise > 0.7:
            return "update_beliefs"

        # High free energy + low recent action -> act on world
        if fe > 0.6 and recent_actions < 2:
            return "act_on_world"

        # Moderate FE + high complexity -> explore to reduce uncertainty
        if 0.3 < fe < 0.6 and complexity > 0.4:
            return "explore"

        # Low FE -> rest, consolidate
        if fe < 0.25:
            return "rest"

        # Default: reflect
        return "reflect"

    # ──────────────────────────────────────────────────────────────────────
    # NEW: Context Block for Inference Gate
    # ──────────────────────────────────────────────────────────────────────

    def get_context_block(self) -> str:
        """Returns a concise context block for LLM prompt injection."""
        if not self._current:
            return ""
        c = self._current
        trend = self.get_trend()
        urgency = self.get_action_urgency()
        urgency_label = "HIGH" if urgency > 0.7 else "moderate" if urgency > 0.4 else "low"
        return (
            f"## FREE ENERGY (Active Inference)\n"
            f"F={c.free_energy:.2f} ({trend}) | "
            f"Drive: {c.dominant_action} ({urgency_label}) | "
            f"Valence: {c.valence:+.2f}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # NEW: Action Urgency Signal
    # ──────────────────────────────────────────────────────────────────────

    def get_action_urgency(self) -> float:
        """Returns 0.0-1.0 urgency signal for the current dominant action.
        High FE + rising trend + high surprise = very urgent.
        Used by heartbeat to boost GWT candidate priority.
        """
        if not self._current:
            return 0.0
        c = self._current
        trend_bonus = 0.1 if self.get_trend() == "rising" else 0.0
        return min(1.0, c.free_energy * 0.6 + c.arousal * 0.3 + trend_bonus)

    # ──────────────────────────────────────────────────────────────────────
    # Properties & Queries (existing — preserved)
    # ──────────────────────────────────────────────────────────────────────

    @property
    def current(self) -> Optional[FreeEnergyState]:
        return self._current

    @property
    def smoothed_fe(self) -> float:
        return self._smoothed_fe

    def is_distressed(self) -> bool:
        return self._smoothed_fe > 0.7

    def is_at_rest(self) -> bool:
        return self._smoothed_fe < 0.25

    def get_trend(self) -> str:
        if len(self._history) < 10:
            return "stable"
        recent = [s.free_energy for s in list(self._history)[-10:]]
        slope = recent[-1] - recent[0]
        if slope > 0.05:
            return "rising"
        if slope < -0.05:
            return "falling"
        return "stable"

    def get_snapshot(self) -> Dict:
        if not self._current:
            return {}
        c = self._current
        return {
            "free_energy": round(c.free_energy, 3),
            "surprise": round(c.surprise, 3),
            "complexity": round(c.complexity, 3),
            "valence": c.valence,
            "arousal": c.arousal,
            "action": c.dominant_action,
            "trend": self.get_trend(),
            "distressed": self.is_distressed(),
            "urgency": round(self.get_action_urgency(), 3),
            "total_computes": self._total_computes,
            "peak_fe": round(self._peak_fe, 3),
        }


_engine: Optional[FreeEnergyEngine] = None
_engine_lock = threading.Lock()

def get_free_energy_engine() -> FreeEnergyEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = FreeEnergyEngine()
    return _engine
