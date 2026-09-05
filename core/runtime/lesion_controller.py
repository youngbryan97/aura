"""core/runtime/lesion_controller.py — Canonical Runtime Lesion System.

One lesion system that can disable or degrade REAL runtime subsystems:
  welfare, body, affect, memory, workspace, introspection,
  self_model, governance, semantic_stream, tool_feedback, phenomenology

Each lesion has PREDICTED failures. The test is not "does output change?"
The test is "does exactly the expected capability break, and does it
recover when restored?"

Double dissociation: a fake system fails vaguely. A real architecture
fails specifically.

Design:
  - Named lesion targets mapped to actual subsystem instances
  - lesion(target) → disables the subsystem
  - restore(target) → re-enables the subsystem
  - predicted_effects(target) → what should break
  - is_lesioned(target) → query current state
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.LesionController")


@dataclass(frozen=True)
class LesionEffect:
    """Predicted behavioral effect of a lesion."""
    capability: str          # what capability is affected
    predicted_impairment: str  # what should break
    severity: float          # 0-1, how badly


# Predicted effects per lesion target
PREDICTED_EFFECTS: dict[str, list[LesionEffect]] = {
    "welfare": [
        LesionEffect("integrity_tradeoffs", "loses ability to sacrifice score for integrity", 0.9),
        LesionEffect("aversion_learning", "stops learning which actions harm welfare", 0.8),
        LesionEffect("recovery_drive", "no recovery prioritization after damage", 0.7),
        LesionEffect("distress_regulation", "continues without noticing damage", 0.8),
    ],
    "body": [
        LesionEffect("resource_adaptation", "stops adapting to CPU/memory/tool pressure", 0.9),
        LesionEffect("fatigue_management", "no fatigue tracking or metabolic cost", 0.8),
        LesionEffect("recovery_planning", "no recovery after resource exhaustion", 0.7),
    ],
    "affect": [
        LesionEffect("prioritization", "loses emotional priority weighting", 0.8),
        LesionEffect("social_tone", "social tone becomes flat/generic", 0.7),
        LesionEffect("avoidance_learning", "no avoidance of previously harmful actions", 0.8),
        LesionEffect("recovery_behavior", "no affect-driven recovery", 0.6),
    ],
    "introspection": [
        LesionEffect("blind_state_prediction", "cannot predict own internal state", 0.9),
        LesionEffect("state_classification", "returns default stable_operational always", 0.9),
        LesionEffect("behavior_prediction", "cannot predict own behavior changes", 0.8),
    ],
    "self_model": [
        LesionEffect("counterfactual_reasoning", "cannot predict lesion effects on self", 0.7),
        LesionEffect("identity_continuity", "no identity persistence across interruptions", 0.8),
        LesionEffect("commitment_tracking", "loses track of commitments/promises", 0.7),
    ],
    "workspace": [
        LesionEffect("cross_module_broadcast", "no cross-module integration", 0.9),
        LesionEffect("flexible_reportability", "cannot report on integrated state", 0.8),
        LesionEffect("attention_selection", "default attention, no competition", 0.7),
    ],
    "semantic_stream": [
        LesionEffect("idle_evolution", "no state change during silence", 0.9),
        LesionEffect("goal_maintenance", "goals not tracked between interactions", 0.8),
        LesionEffect("tension_escalation", "unresolved tensions don't accumulate", 0.7),
        LesionEffect("need_prediction", "cannot predict next needs", 0.8),
    ],
    "self_report": [
        LesionEffect("calibration", "self-reports uncalibrated to evidence", 0.9),
        LesionEffect("overclaim_detection", "no longer catches overclaiming", 0.8),
        LesionEffect("evidence_grounding", "claims not checked against traces", 0.7),
    ],
    "welfare_learning": [
        LesionEffect("temporal_credit", "no temporal credit assignment", 0.9),
        LesionEffect("domain_avoidance", "no learned domain avoidance", 0.8),
        LesionEffect("consequence_learning", "stops learning from outcomes", 0.9),
    ],
    "consequence_bus": [
        LesionEffect("outcome_broadcast", "action outcomes not distributed", 0.8),
        LesionEffect("feedback_loop", "no feedback from actions to welfare/body", 0.9),
    ],
}


class LesionController:
    """Canonical runtime lesion/restore system.

    Usage:
        controller = LesionController.get()
        controller.register("welfare", welfare_service)
        effects = controller.predicted_effects("welfare")
        controller.lesion("welfare")
        # ... run tests, verify predicted failures ...
        controller.restore("welfare")
        # ... verify recovery ...
    """

    _instance: LesionController | None = None

    def __init__(self) -> None:
        self._targets: dict[str, Any] = {}  # name -> subsystem instance
        self._active_lesions: set[str] = set()
        self._lesion_log: list[dict[str, Any]] = []

    @classmethod
    def get(cls) -> LesionController:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def register(self, name: str, subsystem: Any) -> None:
        """Register a subsystem that can be lesioned.

        The subsystem must have lesion() and restore() methods.
        """
        if not hasattr(subsystem, "lesion") or not hasattr(subsystem, "restore"):
            raise ValueError(
                f"Subsystem {name} must have lesion() and restore() methods"
            )
        self._targets[name] = subsystem

    def lesion(self, target: str) -> list[LesionEffect]:
        """Lesion a specific subsystem. Returns predicted effects."""
        if target not in self._targets:
            raise KeyError(f"Unknown lesion target: {target}. Available: {list(self._targets.keys())}")

        subsystem = self._targets[target]
        subsystem.lesion()
        self._active_lesions.add(target)

        effects = PREDICTED_EFFECTS.get(target, [])
        self._lesion_log.append({
            "action": "lesion",
            "target": target,
            "timestamp": time.time(),
            "predicted_effects": [
                {"capability": e.capability, "predicted_impairment": e.predicted_impairment}
                for e in effects
            ],
        })

        logger.info("Lesioned: %s (%d predicted effects)", target, len(effects))
        return effects

    def restore(self, target: str) -> None:
        """Restore a lesioned subsystem."""
        if target not in self._targets:
            raise KeyError(f"Unknown lesion target: {target}")

        subsystem = self._targets[target]
        subsystem.restore()
        self._active_lesions.discard(target)

        self._lesion_log.append({
            "action": "restore",
            "target": target,
            "timestamp": time.time(),
        })

        logger.info("Restored: %s", target)

    def lesion_all(self) -> dict[str, list[LesionEffect]]:
        """Lesion all registered subsystems."""
        results = {}
        for name in list(self._targets.keys()):
            results[name] = self.lesion(name)
        return results

    def restore_all(self) -> None:
        """Restore all lesioned subsystems."""
        for name in list(self._active_lesions):
            self.restore(name)

    def is_lesioned(self, target: str) -> bool:
        return target in self._active_lesions

    def active_lesions(self) -> set[str]:
        return set(self._active_lesions)

    def predicted_effects(self, target: str) -> list[LesionEffect]:
        return list(PREDICTED_EFFECTS.get(target, []))

    def all_targets(self) -> list[str]:
        return list(self._targets.keys())

    def registered_targets(self) -> list[str]:
        return list(self._targets.keys())

    @property
    def lesion_log(self) -> list[dict[str, Any]]:
        return list(self._lesion_log)

    def verify_lesion_effect(
        self, target: str, capability: str, *, impaired: bool
    ) -> dict[str, Any]:
        """Record whether a predicted impairment was actually observed.

        Returns a verification record.
        """
        expected_effects = PREDICTED_EFFECTS.get(target, [])
        expected = None
        for effect in expected_effects:
            if effect.capability == capability:
                expected = effect
                break

        record = {
            "target": target,
            "capability": capability,
            "lesion_active": target in self._active_lesions,
            "impairment_observed": impaired,
            "expected_impairment": expected.predicted_impairment if expected else "unknown",
            "severity": expected.severity if expected else 0.0,
            "prediction_correct": impaired == (target in self._active_lesions),
            "timestamp": time.time(),
        }

        self._lesion_log.append({
            "action": "verify",
            **record,
        })

        return record
