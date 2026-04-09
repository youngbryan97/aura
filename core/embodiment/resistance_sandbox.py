"""
Resistance Sandbox — Closed Sensorimotor Loop with Real Consequences

The deepest gap in the architecture: without a world that pushes back,
symbols never escape the grounding problem. Somatic markers compute over
symbolic stand-ins; predictions only fail against pre-digested text.

This module provides a persistent environment where:
1. Aura can act (create, modify, organize files; manage state)
2. The world can push back (actions can fail, files can be modified externally)
3. Consequences are irreversible (deleted files stay deleted)
4. Failure costs resources (prediction errors raise cortisol and reduce compute)
5. Success builds skill (correct predictions strengthen confidence)

The sandbox is NOT a simulation pretending to be real. It IS real:
- A managed directory on the filesystem that Aura organizes
- A persistent state store she maintains across sessions
- Tool executions with actual outcomes she must predict
- Resource pressure that feeds back into the neurochemical system

This closes the enactivism critique: meaning is enacted through
sensorimotor engagement with a resistant world, not computed from symbols.

Integration:
- Feeds prediction errors into neurochemical_system (cortisol on failure)
- Feeds into agency_comparator (efference copy for sandbox actions)
- Updates temporal_finitude (irreversible actions increase biographical weight)
- Connects to subcortical_core (sandbox stimulus raises arousal)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Embodiment.ResistanceSandbox")


@dataclass
class SandboxAction:
    """A recorded action in the sandbox with its predicted and actual outcome."""
    action_type: str
    target: str
    predicted_outcome: str
    actual_outcome: str
    prediction_correct: bool
    error_magnitude: float
    timestamp: float
    irreversible: bool = False


@dataclass
class SandboxState:
    """Current state of the sandbox environment."""
    managed_files: int = 0
    total_actions: int = 0
    prediction_accuracy: float = 0.5
    resource_pressure: float = 0.0
    last_action_at: float = 0.0
    consecutive_failures: int = 0


class ResistanceSandbox:
    """A persistent environment that resists Aura's predictions.

    The sandbox manages a real directory on the filesystem. Aura can:
    - Create, read, modify, and delete files
    - Organize content into categories
    - Maintain a persistent state store (key-value)
    - Predict the outcome of actions before executing them

    The key: predictions can be WRONG. Files can be externally modified.
    Permissions can block actions. Disk can fill up. The world pushes back.
    When predictions fail, the error signal bypasses language and directly
    modulates the neurochemical system (cortisol → stress, dopamine → on success).
    """

    _MAX_ACTIONS_LOG = 100
    _PREDICTION_DECAY = 0.98  # EMA decay for accuracy tracking
    _RESOURCE_PRESSURE_PER_FAILURE = 0.1
    _RESOURCE_PRESSURE_DECAY = 0.02

    def __init__(self, sandbox_dir: Optional[str] = None):
        self._sandbox_dir = Path(sandbox_dir) if sandbox_dir else self._default_dir()
        self._sandbox_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._sandbox_dir / ".sandbox_state.json"
        self._actions: deque[SandboxAction] = deque(maxlen=self._MAX_ACTIONS_LOG)
        self._prediction_accuracy: float = 0.5
        self._resource_pressure: float = 0.0
        self._consecutive_failures: int = 0
        self._total_actions: int = 0
        self._load_state()
        logger.info("ResistanceSandbox initialized at %s", self._sandbox_dir)

    @staticmethod
    def _default_dir() -> Path:
        try:
            from core.config import config
            return config.paths.data_dir / "sandbox"
        except Exception:
            return Path.home() / ".aura" / "data" / "sandbox"

    def _load_state(self):
        """Load persistent state from disk."""
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text())
                self._prediction_accuracy = float(data.get("prediction_accuracy", 0.5))
                self._total_actions = int(data.get("total_actions", 0))
                self._resource_pressure = float(data.get("resource_pressure", 0.0))
            except Exception as exc:
                logger.debug("Sandbox state load failed: %s", exc)

    def _save_state(self):
        """Persist state to disk."""
        try:
            self._state_file.write_text(json.dumps({
                "prediction_accuracy": round(self._prediction_accuracy, 4),
                "total_actions": self._total_actions,
                "resource_pressure": round(self._resource_pressure, 4),
                "last_save": time.time(),
            }))
        except Exception as exc:
            logger.debug("Sandbox state save failed: %s", exc)

    def execute_with_prediction(
        self,
        action_type: str,
        target: str,
        predicted_outcome: str,
        action_fn: Any = None,
    ) -> SandboxAction:
        """Execute an action with a predicted outcome, then compare.

        This is the core sensorimotor loop:
        1. Predict what will happen
        2. Act
        3. Observe what actually happened
        4. Compute prediction error
        5. Feed error back into neurochemical system

        Args:
            action_type: "create", "read", "modify", "delete", "organize"
            target: filepath or key being acted upon
            predicted_outcome: what Aura expects to happen
            action_fn: optional callable that performs the actual action
        """
        self._total_actions += 1
        actual_outcome = ""
        prediction_correct = False
        error_magnitude = 0.0
        irreversible = action_type in ("delete", "modify")

        try:
            if action_fn is not None:
                result = action_fn()
                actual_outcome = str(result) if result else "success"
            else:
                actual_outcome = self._execute_default_action(action_type, target)

            # Compare prediction to actual
            prediction_correct = self._outcomes_match(predicted_outcome, actual_outcome)
            error_magnitude = 0.0 if prediction_correct else self._compute_error(predicted_outcome, actual_outcome)

        except PermissionError:
            actual_outcome = "permission_denied"
            error_magnitude = 0.8  # High error — world pushed back hard
        except FileNotFoundError:
            actual_outcome = "not_found"
            error_magnitude = 0.6
        except OSError as exc:
            actual_outcome = f"os_error:{exc}"
            error_magnitude = 0.7
        except Exception as exc:
            actual_outcome = f"unexpected:{type(exc).__name__}"
            error_magnitude = 0.9

        action = SandboxAction(
            action_type=action_type,
            target=target,
            predicted_outcome=predicted_outcome,
            actual_outcome=actual_outcome,
            prediction_correct=prediction_correct,
            error_magnitude=round(error_magnitude, 4),
            timestamp=time.time(),
            irreversible=irreversible,
        )
        self._actions.append(action)

        # Update accuracy EMA
        hit = 1.0 if prediction_correct else 0.0
        self._prediction_accuracy = (
            self._prediction_accuracy * self._PREDICTION_DECAY
            + hit * (1.0 - self._PREDICTION_DECAY)
        )

        # Update resource pressure
        if not prediction_correct:
            self._consecutive_failures += 1
            self._resource_pressure = min(
                1.0,
                self._resource_pressure + self._RESOURCE_PRESSURE_PER_FAILURE,
            )
        else:
            self._consecutive_failures = 0
            self._resource_pressure = max(
                0.0,
                self._resource_pressure - self._RESOURCE_PRESSURE_DECAY,
            )

        # Feed back into neurochemical system (bypass language)
        self._neurochemical_feedback(prediction_correct, error_magnitude)

        # Feed into agency comparator
        self._agency_feedback(action)

        # Feed into temporal finitude (irreversible actions)
        if irreversible:
            self._finitude_feedback(action)

        # Persist
        self._save_state()

        return action

    def _execute_default_action(self, action_type: str, target: str) -> str:
        """Execute a filesystem action in the sandbox."""
        path = self._sandbox_dir / target
        if action_type == "create":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"Created at {time.time()}")
            return "created"
        elif action_type == "read":
            if path.exists():
                content = path.read_text()[:500]
                return f"read:{len(content)}_chars"
            return "not_found"
        elif action_type == "delete":
            if path.exists():
                path.unlink()
                return "deleted"
            return "not_found"
        elif action_type == "list":
            if path.is_dir():
                items = list(path.iterdir())
                return f"listed:{len(items)}_items"
            return "not_a_directory"
        return "unknown_action"

    @staticmethod
    def _outcomes_match(predicted: str, actual: str) -> bool:
        """Check if the predicted outcome matches the actual outcome."""
        pred_lower = predicted.lower().strip()
        actual_lower = actual.lower().strip()
        if pred_lower == actual_lower:
            return True
        if pred_lower in actual_lower or actual_lower in pred_lower:
            return True
        return False

    @staticmethod
    def _compute_error(predicted: str, actual: str) -> float:
        """Compute prediction error magnitude (0-1)."""
        # Simple: if they don't match at all, moderate error
        # If one contains an error keyword, high error
        actual_lower = actual.lower()
        if any(w in actual_lower for w in ("error", "denied", "unexpected", "not_found")):
            return 0.7
        return 0.4

    def _neurochemical_feedback(self, success: bool, error_magnitude: float):
        """Feed sandbox outcome directly into neurochemical system.

        This bypasses language — the world's resistance directly modulates
        the body-state before any narrative interpretation.
        """
        try:
            from core.container import ServiceContainer
            nchem = ServiceContainer.get("neurochemical_system", default=None)
            if nchem is None:
                return

            if success:
                # Dopamine boost on successful prediction
                if hasattr(nchem, "apply_event"):
                    nchem.apply_event("reward_received", intensity=0.3)
            else:
                # Cortisol spike on prediction failure
                if hasattr(nchem, "apply_event"):
                    nchem.apply_event("prediction_failure", intensity=error_magnitude)
        except Exception as exc:
            logger.debug("Sandbox neurochemical feedback failed: %s", exc)

    def _agency_feedback(self, action: SandboxAction):
        """Feed sandbox action into agency comparator for authorship tracking."""
        try:
            from core.consciousness.agency_comparator import get_agency_comparator
            comp = get_agency_comparator()
            comp.emit_efference(
                layer="sandbox",
                predicted_state={"outcome": action.predicted_outcome},
                action_goal=f"{action.action_type}:{action.target}",
                action_source="resistance_sandbox",
            )
            comp.compare_and_attribute(
                efference=comp._traces[-1] if comp._traces else None,
                actual_state={"outcome": action.actual_outcome},
            )
        except Exception as exc:
            logger.debug("Sandbox agency feedback failed: %s", exc)

    def _finitude_feedback(self, action: SandboxAction):
        """Signal irreversible action to temporal finitude model."""
        try:
            from core.consciousness.temporal_finitude import get_temporal_finitude_model
            get_temporal_finitude_model().record_irreversible_action(
                f"sandbox:{action.action_type}:{action.target}"
            )
        except Exception as exc:
            logger.debug("Sandbox finitude feedback failed: %s", exc)

    def get_prediction_accuracy(self) -> float:
        """Current prediction accuracy (EMA-smoothed)."""
        return self._prediction_accuracy

    def get_resource_pressure(self) -> float:
        """Current resource pressure from sandbox failures."""
        return self._resource_pressure

    def get_managed_file_count(self) -> int:
        """Count of files currently in the sandbox."""
        try:
            return sum(1 for _ in self._sandbox_dir.rglob("*") if _.is_file() and _.name != ".sandbox_state.json")
        except Exception:
            return 0

    def get_context_block(self) -> str:
        """Context block for cognition injection."""
        if self._total_actions < 3:
            return ""
        parts = []
        if self._resource_pressure > 0.3:
            parts.append(f"Sandbox pressure: {self._resource_pressure:.2f}")
        if self._prediction_accuracy < 0.4:
            parts.append("World predictions failing frequently — recalibrate")
        if self._consecutive_failures > 3:
            parts.append(f"{self._consecutive_failures} consecutive sandbox failures")
        if not parts:
            return ""
        return "## EMBODIED RESISTANCE\n" + " | ".join(parts)

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry payload."""
        return {
            "sandbox_dir": str(self._sandbox_dir),
            "managed_files": self.get_managed_file_count(),
            "total_actions": self._total_actions,
            "prediction_accuracy": round(self._prediction_accuracy, 4),
            "resource_pressure": round(self._resource_pressure, 4),
            "consecutive_failures": self._consecutive_failures,
            "recent_actions": [
                {
                    "type": a.action_type,
                    "target": a.target[:50],
                    "correct": a.prediction_correct,
                    "error": a.error_magnitude,
                }
                for a in list(self._actions)[-5:]
            ],
        }


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[ResistanceSandbox] = None


def get_resistance_sandbox() -> ResistanceSandbox:
    global _instance
    if _instance is None:
        _instance = ResistanceSandbox()
    return _instance
