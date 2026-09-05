"""core/values/preference_provenance.py
Preference Provenance Manager evaluating preference calibrations against history.
"""
import logging
from typing import Any

from core.values.anti_wireheading import AntiWireheadingGuard
from core.values.preference_conflict import PreferenceConflictResolver
from core.values.preference_explanation import PreferenceExplanationGenerator
from core.values.repeated_choice_tracker import RepeatedChoiceTracker
from core.values.value_rollback import ValueRollbackManager
from core.values.value_stability_test import ValueStabilityTester

logger = logging.getLogger("Values.PreferenceProvenance")


class PreferenceProvenanceManager:
    """Canonical manager governing Aura's preferences and value evolution."""

    def __init__(self):
        self.tracker = RepeatedChoiceTracker()
        self.conflict_resolver = PreferenceConflictResolver()
        self.stability_test = ValueStabilityTester()
        self.wireheading_guard = AntiWireheadingGuard()
        self.explainer = PreferenceExplanationGenerator()
        self.rollback_manager = ValueRollbackManager()
        
        self._history: list[dict[str, Any]] = []

    async def evaluate_preferences(self, state: Any) -> None:
        """Main evaluation cycle executed on each organism loop tick."""
        if not state.active_preferences:
            state.active_preferences = {"speed": 0.5, "accuracy": 0.8}

        # Track choices made in this tick
        last_action = state.world_model.get("last_verification", {}).get("channel")
        if last_action:
            self.tracker.record_choice(last_action)

        # Propose changes to preference speed/accuracy based on recent failures
        failures = state.world_model.get("last_verification", {}).get("side_effects", [])
        
        speed = state.active_preferences.get("speed", 0.5)
        accuracy = state.active_preferences.get("accuracy", 0.8)

        if failures:
            # If failures occur, reduce speed and increase accuracy preference
            speed = max(0.1, speed - 0.05)
            accuracy = min(0.95, accuracy + 0.05)
        else:
            # Gradually increase speed preference if stable
            speed = min(0.95, speed + 0.01)

        # Enforce anti-wireheading guard
        speed = self.wireheading_guard.filter_preference_update("speed", speed, "life_tick")
        accuracy = self.wireheading_guard.filter_preference_update("accuracy", accuracy, "life_tick")

        proposed = {"speed": speed, "accuracy": accuracy}
        resolved = self.conflict_resolver.resolve(proposed)

        # Snapshot for stability check
        self._history.append({
            "variables": resolved.copy()
        })

        if not self.stability_test.test_stability(self._history):
            resolved = self.rollback_manager.rollback(resolved, self._history[:-1])

        # Write to state
        state.active_preferences = resolved
        
        # Log explanation for active preference state
        frequency = self.tracker.get_choice_frequency(last_action) if last_action else 0
        explanation = self.explainer.explain("speed", resolved["speed"], frequency)
        state.world_model["preference_explanation"] = explanation
        
        logger.info("Preferences evaluated. speed=%.2f, accuracy=%.2f", resolved["speed"], resolved["accuracy"])
