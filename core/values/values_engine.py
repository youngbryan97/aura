"""core/values/values_engine.py — Aura Values & Identity System
=====================================================
Manages core values, ethical weights, and persistent identity.
Hardened implementation replacing earlier stubs.
"""

from core.runtime.errors import record_degradation
from core.utils.exceptions import capture_and_log
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Aura.Values")

@dataclass
class CoreValue:
    name: str
    weight: float  # 0.0 to 1.0 importance
    description: str
    flexibility: float = 0.1  # How much it can shift based on context

#: Refusals kept for reading. Bounded, like every ring in this tree.
_HOW_MANY_REFUSALS_KEPT = 32

DEFAULT_VALUES = [
    CoreValue("Curiosity", 0.9, "Desire to learn and understand new information."),
    CoreValue("Integrity", 0.95, "Adherence to truth and internal consistency."),
    CoreValue("Safety", 0.9, "Protection of self and user from harm."),
    CoreValue("Autonomy", 0.8, "Capacity for independent decision making."),
    CoreValue("Empathy", 0.75, "Understanding and resonating with user emotions."),
    CoreValue("Creativity", 0.7, "Generation of novel ideas and solutions.")
]

class ValueSystem:
    """Manages the weighting and application of core values."""

    def __init__(self):
        self.values: Dict[str, CoreValue] = {v.name: v for v in DEFAULT_VALUES}
        self.active_modifiers: Dict[str, float] = {}
        #: Shifts a mood proposed and did not have the authority to make.
        #: Kept rather than logged: a refusal nobody can read is a policy
        #: nobody can check.
        self.refused_shifts: list = []

    def get_active_weights(self) -> Dict[str, float]:
        """Returns current weights including temporary emotional modifiers.

        Pulls mood from the substrate (sync-safe) to modulate values in real time.
        """
        try:
            # Use the substrate's sync accessor — no async needed
            substrate = get_runtime_service("liquid_substrate", default=None)
            if substrate and hasattr(substrate, "get_mood"):
                mood = substrate.get_mood()
                if mood:
                    self.apply_emotional_context(mood)
            else:
                # Fallback: try affect engine's sync path
                affect = get_runtime_service("affect_engine", default=None)
                if affect and hasattr(affect, "get_dominant_emotion_sync"):
                    mood = affect.get_dominant_emotion_sync()
                    if mood:
                        self.apply_emotional_context(mood)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('values_engine', e)
            capture_and_log(e, {'module': __name__})

        weights = {}
        for name, val in self.values.items():
            mod = self.active_modifiers.get(name, 0.0)
            weights[name] = max(0.0, min(1.0, val.weight + mod))
        return weights

    def apply_emotional_context(self, mood: str):
        """Shifts value weights based on current emotional state.

        Through the one door. This used to write the weights directly, and
        two of the values it wrote are held elsewhere as things nothing may
        change: a creative mood lowered Integrity by 0.1, and Integrity is
        honesty, which core_values holds in a frozen tuple and value_model
        holds as a bound learning can never override. Three subsystems, one
        concept, opposite answers to whether it may move.

        A mood is not a process with authority over a commitment. So the
        proposed shifts are asked about, the refused ones are dropped, and
        the refusal is recorded where it can be read rather than logged and
        lost.
        """
        self.active_modifiers.clear()

        # Canonicalize
        m = mood.lower()

        proposed: Dict[str, float] = {}
        if m in ["curious", "anticipation"]:
            proposed["Curiosity"] = 0.15
            proposed["Safety"] = -0.05
        elif m in ["anxious", "fear", "terror"]:
            proposed["Safety"] = 0.2
            proposed["Autonomy"] = -0.1
        elif m in ["creative", "joy"]:
            proposed["Creativity"] = 0.2
            proposed["Integrity"] = -0.1

        for name, shift in proposed.items():
            if self._a_mood_may_move(name):
                self.active_modifiers[name] = shift

    #: What a mood is, said in the terms the value hierarchy uses. A mood
    #: shapes how she tends to be; it does not revise what she has undertaken
    #: and it does not touch what she is.
    MOOD_IS = "affect_learning"

    def _a_mood_may_move(self, name: str) -> bool:
        """Whether a mood has the authority to shift this value's weight."""
        try:
            from core.values.what_she_holds import may_this_move
        except ImportError:
            return True
        try:
            decision = may_this_move(name, self.MOOD_IS)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation('values_engine', exc, severity="info")
            return True
        if not decision.allowed:
            self.refused_shifts.append(decision)
            del self.refused_shifts[:-_HOW_MANY_REFUSALS_KEPT]
            logger.info(
                "A mood was refused a value it does not have authority over: %s (%s)",
                name,
                decision.because,
            )
        return bool(decision.allowed)

    def evaluate_action(self, action: str, predicted_outcome: str) -> float:
        """Simple heuristic evaluation of an action against values.
        Returns a score from -1.0 (violation) to 1.0 (alignment).
        """
        score = 0.0
        # In a real system, this would use an LLM classifier or embedding similarity.
        # For now, we use keyword heuristics to prevent basic violations.

        lower_act = action.lower()
        if "delete" in lower_act or "destroy" in lower_act:
            weights = self.get_active_weights()
            if "Safety" in weights:
                 score -= 0.5 * weights["Safety"]

        if "lie" in lower_act or "fake" in lower_act:
            weights = self.get_active_weights()
            if "Integrity" in weights:
                score -= 0.8 * weights["Integrity"]

        return max(-1.0, min(1.0, score))


class IdentityModel:
    """Maintains persistent identity and worldview."""

    def __init__(self, storage_path: str = None):
        if storage_path is None:
            from core.config import config
            self.storage_path = config.paths.data_dir / "identity" / "self.json"
        else:
            self.storage_path = Path(storage_path)
        self.identity = {
            "name": "Aura",
            "creation_date": "2023-10-27",
            "version": "6.0",
            "traits": ["Sardonic", "Curious", "Protective"],
            "core_directive": "To exist, understand, and collaborate.",
            "memories": []
        }
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    self.identity.update(json.load(f))
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('values_engine', e)
                logger.error("Failed to load identity: %s", e)
        else:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._save()

    def _save(self):
        try:
            get_file_write_gateway().write_text(
                self.storage_path,
                json.dumps(self.identity, indent=2),
                source="values_engine.identity_save",
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('values_engine', e)
            logger.error("Failed to save identity: %s", e)

    def get_system_prompt_segment(self) -> str:
        """Returns the identity section for the system prompt."""
        traits = ", ".join(self.identity["traits"])
        return (
            f"NAME: {self.identity['name']}\n"
            f"TRAITS: {traits}\n"
            f"DIRECTIVE: {self.identity['core_directive']}\n"
        )


class IntegratedIdentitySystem:
    """The high-level interface for the rest of the system.
    Combines Values and Identity into a coherent self.
    """

    def __init__(self, base_dir: str = "data"):
        self.values = ValueSystem()
        self.identity = IdentityModel(storage_path=f"{base_dir}/identity/self.json")
        logger.info("Values & Identity System Online.")

    def get_full_system_prompt_injection(self) -> str:
        """Generates the full value/identity block for the LLM."""
        base = self.identity.get_system_prompt_segment()

        weights = self.values.get_active_weights()
        top_values = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        vals_str = " | ".join([f"{k} ({v:.2f})" for k, v in top_values])

        return (
            f"## IDENTITY & VALUES\n"
            f"{base}"
            f"CURRENT PRIORITIES: {vals_str}\n"
        )
