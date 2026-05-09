"""core/meta/metacognitive_monitor.py -- Meta-Cognitive Reflection Loop
=======================================================================
Monitors the learning process itself and adapts learning strategy when
plateaus, forgetting, or calibration drift are detected.

Tracked signals:
  - Gradient norm trend (exponentially weighted)
  - Loss curvature (second derivative approximation)
  - Prediction error trend (world model surprise moving average)
  - Confidence calibration (expected vs actual accuracy)

Detected conditions:
  - PLATEAU: gradient norm and loss stable for > N steps
  - FORGETTING: prediction error on old tasks rising
  - OVERFIT: train loss decreasing but validation surprise increasing
  - MISCALIBRATED: confidence systematically too high or low

Strategy adjustments (sandboxed, require Will approval for core changes):
  - Adjust global learning rate
  - Increase/decrease replay buffer size
  - Trigger consolidation in EWC plasticity governor
  - Request new training data via experience distillery
  - Adjust exploration/exploitation balance

Gate: The meta-cognitive layer is itself sandboxed. It CANNOT directly
modify core architecture without Will approval and long-horizon stability
testing. All adjustments go through the governance pipeline.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.MetaCognitive")

_DATA_DIR = Path.home() / ".aura" / "data" / "metacognitive"
_LOG_PATH = _DATA_DIR / "reflection_log.jsonl"


class LearningCondition(str, Enum):
    """Detected meta-cognitive condition."""
    HEALTHY = "healthy"
    PLATEAU = "plateau"
    FORGETTING = "forgetting"
    OVERFIT = "overfit"
    MISCALIBRATED = "miscalibrated"
    UNSTABLE = "unstable"


class StrategyAction(str, Enum):
    """Meta-cognitive strategy adjustment."""
    NO_ACTION = "no_action"
    LOWER_LR = "lower_learning_rate"
    RAISE_LR = "raise_learning_rate"
    INCREASE_REPLAY = "increase_replay_buffer"
    TRIGGER_CONSOLIDATION = "trigger_ewc_consolidation"
    REQUEST_DATA = "request_new_training_data"
    INCREASE_EXPLORATION = "increase_exploration"
    DECREASE_EXPLORATION = "decrease_exploration"


@dataclass
class MetaCognitiveSnapshot:
    """A single observation of the learning process."""
    timestamp: float
    gradient_norm: float
    loss: float
    prediction_error: float
    confidence: float
    accuracy: float  # Actual accuracy on recent predictions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "gradient_norm": round(self.gradient_norm, 6),
            "loss": round(self.loss, 6),
            "prediction_error": round(self.prediction_error, 6),
            "confidence": round(self.confidence, 4),
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class MetaCognitiveReflection:
    """Result of a meta-cognitive assessment cycle."""
    condition: LearningCondition
    recommended_actions: List[StrategyAction]
    evidence: Dict[str, float]
    reasoning: str
    cycle: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition.value,
            "actions": [a.value for a in self.recommended_actions],
            "evidence": {k: round(v, 6) for k, v in self.evidence.items()},
            "reasoning": self.reasoning,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
        }


@dataclass
class MetaCognitiveConfig:
    """Configuration for the meta-cognitive monitor."""
    window_size: int = 50         # Observation window for trend detection
    plateau_threshold: float = 0.001  # Gradient norm change below this = plateau
    plateau_patience: int = 20    # Steps before declaring plateau
    forgetting_threshold: float = 0.05  # PE increase > this on old tasks = forgetting
    calibration_tolerance: float = 0.15  # |confidence - accuracy| > this = miscalibrated
    instability_threshold: float = 5.0   # Gradient norm > this = unstable
    ema_alpha: float = 0.1        # EMA smoothing factor


class MetaCognitiveMonitor:
    """Monitors the learning process and recommends strategy adjustments.

    Usage:
        monitor = MetaCognitiveMonitor()

        # During each learning step:
        monitor.observe(
            gradient_norm=0.05,
            loss=0.3,
            prediction_error=0.1,
            confidence=0.8,
            accuracy=0.75,
        )

        # Periodically assess:
        reflection = monitor.assess()
        if reflection.condition != LearningCondition.HEALTHY:
            for action in reflection.recommended_actions:
                apply_strategy(action)
    """

    def __init__(self, config: Optional[MetaCognitiveConfig] = None) -> None:
        self._config = config or MetaCognitiveConfig()
        self._history: Deque[MetaCognitiveSnapshot] = deque(
            maxlen=self._config.window_size * 3
        )
        self._reflections: List[MetaCognitiveReflection] = []
        self._cycle = 0

        # EMA-smoothed signals
        self._ema_grad_norm = 0.0
        self._ema_loss = 0.0
        self._ema_pred_error = 0.0
        self._ema_confidence = 0.5
        self._ema_accuracy = 0.5

        # Trend tracking
        self._grad_norm_history: Deque[float] = deque(
            maxlen=self._config.window_size
        )
        self._loss_history: Deque[float] = deque(
            maxlen=self._config.window_size
        )
        self._pred_error_history: Deque[float] = deque(
            maxlen=self._config.window_size
        )

        # Forgetting detection: track prediction error on "old" tasks
        self._old_task_errors: Deque[float] = deque(maxlen=100)

        # Strategy action callbacks (registered by subsystems)
        self._action_handlers: Dict[StrategyAction, Callable[[], None]] = {}

        _DATA_DIR.mkdir(parents=True, exist_ok=True)

    def observe(
        self,
        gradient_norm: float,
        loss: float,
        prediction_error: float,
        confidence: float = 0.5,
        accuracy: float = 0.5,
    ) -> None:
        """Record a single observation of the learning process."""
        alpha = self._config.ema_alpha

        self._ema_grad_norm = alpha * gradient_norm + (1 - alpha) * self._ema_grad_norm
        self._ema_loss = alpha * loss + (1 - alpha) * self._ema_loss
        self._ema_pred_error = alpha * prediction_error + (1 - alpha) * self._ema_pred_error
        self._ema_confidence = alpha * confidence + (1 - alpha) * self._ema_confidence
        self._ema_accuracy = alpha * accuracy + (1 - alpha) * self._ema_accuracy

        self._grad_norm_history.append(gradient_norm)
        self._loss_history.append(loss)
        self._pred_error_history.append(prediction_error)

        snapshot = MetaCognitiveSnapshot(
            timestamp=time.time(),
            gradient_norm=gradient_norm,
            loss=loss,
            prediction_error=prediction_error,
            confidence=confidence,
            accuracy=accuracy,
        )
        self._history.append(snapshot)

    def observe_old_task_error(self, error: float) -> None:
        """Record prediction error on an old/previously-learned task."""
        self._old_task_errors.append(error)

    def register_action_handler(
        self, action: StrategyAction, handler: Callable[[], None]
    ) -> None:
        """Register a callback for a strategy action."""
        self._action_handlers[action] = handler

    def assess(self) -> MetaCognitiveReflection:
        """Run a meta-cognitive assessment cycle.

        Analyzes recent learning signals and produces a reflection with
        detected conditions and recommended actions.

        Returns:
            MetaCognitiveReflection with condition and recommendations.
        """
        self._cycle += 1

        if len(self._history) < 10:
            return MetaCognitiveReflection(
                condition=LearningCondition.HEALTHY,
                recommended_actions=[StrategyAction.NO_ACTION],
                evidence={"observations": len(self._history)},
                reasoning="Insufficient observations for assessment",
                cycle=self._cycle,
            )

        # Compute diagnostic signals
        evidence: Dict[str, float] = {}
        conditions: List[LearningCondition] = []
        actions: List[StrategyAction] = []

        # 1. Gradient norm trend
        grad_trend = self._compute_trend(list(self._grad_norm_history))
        evidence["gradient_trend"] = grad_trend
        evidence["gradient_norm_ema"] = self._ema_grad_norm

        # 2. Loss curvature (second derivative approximation)
        loss_trend = self._compute_trend(list(self._loss_history))
        loss_curvature = self._compute_curvature(list(self._loss_history))
        evidence["loss_trend"] = loss_trend
        evidence["loss_curvature"] = loss_curvature

        # 3. Prediction error trend
        pe_trend = self._compute_trend(list(self._pred_error_history))
        evidence["prediction_error_trend"] = pe_trend
        evidence["prediction_error_ema"] = self._ema_pred_error

        # 4. Confidence calibration
        calibration_gap = abs(self._ema_confidence - self._ema_accuracy)
        evidence["calibration_gap"] = calibration_gap
        evidence["confidence_ema"] = self._ema_confidence
        evidence["accuracy_ema"] = self._ema_accuracy

        # ── Condition detection ──────────────────────────────────────

        # PLATEAU: gradient norm flat and loss flat
        if (abs(grad_trend) < self._config.plateau_threshold
                and abs(loss_trend) < self._config.plateau_threshold
                and len(self._grad_norm_history) >= self._config.plateau_patience):
            conditions.append(LearningCondition.PLATEAU)
            actions.append(StrategyAction.RAISE_LR)
            actions.append(StrategyAction.INCREASE_EXPLORATION)

        # FORGETTING: old task errors increasing
        if len(self._old_task_errors) >= 10:
            old_errors = list(self._old_task_errors)
            old_trend = self._compute_trend(old_errors)
            evidence["old_task_error_trend"] = old_trend
            if old_trend > self._config.forgetting_threshold:
                conditions.append(LearningCondition.FORGETTING)
                actions.append(StrategyAction.TRIGGER_CONSOLIDATION)
                actions.append(StrategyAction.INCREASE_REPLAY)

        # OVERFIT: loss decreasing but prediction error increasing
        if loss_trend < -0.01 and pe_trend > 0.01:
            conditions.append(LearningCondition.OVERFIT)
            actions.append(StrategyAction.LOWER_LR)
            actions.append(StrategyAction.INCREASE_REPLAY)

        # MISCALIBRATED: confidence doesn't match accuracy
        if calibration_gap > self._config.calibration_tolerance:
            conditions.append(LearningCondition.MISCALIBRATED)
            if self._ema_confidence > self._ema_accuracy:
                actions.append(StrategyAction.INCREASE_EXPLORATION)
            else:
                actions.append(StrategyAction.DECREASE_EXPLORATION)

        # UNSTABLE: gradient norm too high
        if self._ema_grad_norm > self._config.instability_threshold:
            conditions.append(LearningCondition.UNSTABLE)
            actions.append(StrategyAction.LOWER_LR)

        # Default: healthy
        if not conditions:
            conditions = [LearningCondition.HEALTHY]
            actions = [StrategyAction.NO_ACTION]

        # Pick the most severe condition
        severity_order = [
            LearningCondition.UNSTABLE,
            LearningCondition.FORGETTING,
            LearningCondition.OVERFIT,
            LearningCondition.MISCALIBRATED,
            LearningCondition.PLATEAU,
            LearningCondition.HEALTHY,
        ]
        primary_condition = LearningCondition.HEALTHY
        for cond in severity_order:
            if cond in conditions:
                primary_condition = cond
                break

        # Deduplicate actions
        unique_actions = list(dict.fromkeys(actions))

        reasoning = self._build_reasoning(primary_condition, evidence, conditions)

        reflection = MetaCognitiveReflection(
            condition=primary_condition,
            recommended_actions=unique_actions,
            evidence=evidence,
            reasoning=reasoning,
            cycle=self._cycle,
        )

        self._reflections.append(reflection)
        self._log_reflection(reflection)

        logger.info(
            "MetaCognitive cycle %d: %s → %s",
            self._cycle, primary_condition.value,
            [a.value for a in unique_actions],
        )

        return reflection

    def execute_actions(self, reflection: MetaCognitiveReflection) -> List[str]:
        """Execute registered handlers for recommended actions.

        Returns list of actions that were actually executed.
        """
        executed = []
        for action in reflection.recommended_actions:
            handler = self._action_handlers.get(action)
            if handler is not None:
                try:
                    handler()
                    executed.append(action.value)
                except (TypeError, ValueError, RuntimeError) as exc:
                    logger.warning("Action handler %s failed: %s", action.value, exc)
        return executed

    # ── Signal Processing ────────────────────────────────────────────

    @staticmethod
    def _compute_trend(values: List[float]) -> float:
        """Compute linear trend (slope) of a signal via least squares."""
        if len(values) < 3:
            return 0.0
        n = len(values)
        x = np.arange(n, dtype=np.float64)
        y = np.array(values, dtype=np.float64)
        # Mask NaN/Inf
        valid = np.isfinite(y)
        if np.sum(valid) < 3:
            return 0.0
        x, y = x[valid], y[valid]
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        denom = np.sum((x - x_mean) ** 2)
        if denom < 1e-10:
            return 0.0
        slope = np.sum((x - x_mean) * (y - y_mean)) / denom
        return float(slope)

    @staticmethod
    def _compute_curvature(values: List[float]) -> float:
        """Approximate second derivative (curvature) of a signal."""
        if len(values) < 5:
            return 0.0
        y = np.array(values[-20:], dtype=np.float64)
        if len(y) < 5:
            return 0.0
        # Second difference
        d2 = np.diff(y, n=2)
        return float(np.mean(d2))

    def _build_reasoning(
        self,
        condition: LearningCondition,
        evidence: Dict[str, float],
        all_conditions: List[LearningCondition],
    ) -> str:
        """Build human-readable reasoning for the reflection."""
        parts = [f"Primary condition: {condition.value}"]
        if len(all_conditions) > 1:
            parts.append(
                f"Also detected: {', '.join(c.value for c in all_conditions if c != condition)}"
            )

        if condition == LearningCondition.PLATEAU:
            parts.append(
                f"Gradient trend ({evidence.get('gradient_trend', 0):.4f}) and "
                f"loss trend ({evidence.get('loss_trend', 0):.4f}) are both flat."
            )
        elif condition == LearningCondition.FORGETTING:
            parts.append(
                f"Old task errors trending up: {evidence.get('old_task_error_trend', 0):.4f}"
            )
        elif condition == LearningCondition.OVERFIT:
            parts.append(
                f"Loss decreasing ({evidence.get('loss_trend', 0):.4f}) but "
                f"prediction error increasing ({evidence.get('prediction_error_trend', 0):.4f})"
            )
        elif condition == LearningCondition.MISCALIBRATED:
            parts.append(
                f"Calibration gap: {evidence.get('calibration_gap', 0):.4f} "
                f"(conf={evidence.get('confidence_ema', 0):.2f}, "
                f"acc={evidence.get('accuracy_ema', 0):.2f})"
            )
        elif condition == LearningCondition.UNSTABLE:
            parts.append(
                f"Gradient norm EMA ({evidence.get('gradient_norm_ema', 0):.4f}) "
                f"exceeds stability threshold"
            )

        return "; ".join(parts)

    def _log_reflection(self, reflection: MetaCognitiveReflection) -> None:
        """Append reflection to persistent log."""
        try:
            with open(_LOG_PATH, "a") as f:
                f.write(json.dumps(reflection.to_dict(), default=str) + "\n")
        except (OSError, IOError):
            return

    # ── Public API ───────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        return {
            "cycle": self._cycle,
            "observations": len(self._history),
            "gradient_norm_ema": round(self._ema_grad_norm, 6),
            "loss_ema": round(self._ema_loss, 6),
            "prediction_error_ema": round(self._ema_pred_error, 6),
            "confidence_ema": round(self._ema_confidence, 4),
            "accuracy_ema": round(self._ema_accuracy, 4),
            "calibration_gap": round(
                abs(self._ema_confidence - self._ema_accuracy), 4
            ),
            "last_condition": (
                self._reflections[-1].condition.value
                if self._reflections else "none"
            ),
            "registered_handlers": list(
                a.value for a in self._action_handlers.keys()
            ),
        }

    def get_recent_reflections(self, n: int = 10) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._reflections[-n:]]
