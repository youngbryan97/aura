"""core/values_engine.py — Aura Values & Identity System
=====================================================
Manages core values, ethical weights, and persistent identity.
Hardened implementation replacing earlier stubs.
"""

import asyncio
from core.utils.exceptions import capture_and_log
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.Values")

@dataclass
class CoreValue:
    name: str
    weight: float  # 0.0 to 1.0 importance
    description: str
    flexibility: float = 0.1  # How much it can shift based on context

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

    def get_active_weights(self) -> Dict[str, float]:
        """Returns current weights including temporary emotional modifiers.

        Pulls mood from the substrate (sync-safe) to modulate values in real time.
        """
        try:
            from core.container import ServiceContainer
            # Use the substrate's sync accessor — no async needed
            substrate = ServiceContainer.get("liquid_substrate", default=None)
            if substrate and hasattr(substrate, "get_mood"):
                mood = substrate.get_mood()
                if mood:
                    self.apply_emotional_context(mood)
            else:
                # Fallback: try affect engine's sync path
                affect = ServiceContainer.get("affect_engine", default=None)
                if affect and hasattr(affect, "get_dominant_emotion_sync"):
                    mood = affect.get_dominant_emotion_sync()
                    if mood:
                        self.apply_emotional_context(mood)
        except Exception as e:
            capture_and_log(e, {'module': __name__})

        weights = {}
        for name, val in self.values.items():
            mod = self.active_modifiers.get(name, 0.0)
            weights[name] = max(0.0, min(1.0, val.weight + mod))
        return weights

    def apply_emotional_context(self, mood: str):
        """Shifts value weights based on current emotional state."""
        self.active_modifiers.clear()
        
        # Canonicalize
        m = mood.lower()
        
        if m in ["curious", "anticipation"]:
            self.active_modifiers["Curiosity"] = 0.15
            self.active_modifiers["Safety"] = -0.05
        elif m in ["anxious", "fear", "terror"]:
            self.active_modifiers["Safety"] = 0.2
            self.active_modifiers["Autonomy"] = -0.1
        elif m in ["creative", "joy"]:
            self.active_modifiers["Creativity"] = 0.2
            self.active_modifiers["Integrity"] = -0.1 
        
    def evaluate_action(self, action: str, predicted_outcome: str) -> float:
        """Simple heuristic evaluation of an action against values.
        Returns a score from -1.0 (violation) to 1.0 (alignment).
        """
        score = 0.0
        # In a real system, this would use an LLM classifier or embedding similarity.
        # For now, we use keyword heuristics to prevent basic violations.
        
        lower_act = action.lower()
        lower_out = predicted_outcome.lower()
        
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
            except Exception as e:
                logger.error("Failed to load identity: %s", e)
        else:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._save()

    def _save(self):
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self.identity, f, indent=2)
        except Exception as e:
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
