"""core/consciousness/mind_model.py
Theory of Mind (ToM) Engine.
Tracks the system's internal model of the user's beliefs, intents, and feelings.
"""
from core.runtime.errors import record_degradation
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from core.base_module import AuraBaseModule

class UserBeliefState:
    """Represents what Aura think the USER knows, wants, and feels."""
    def __init__(self):
        self.known_facts: List[str] = []
        self.active_goals: List[str] = []
        self.perceived_mood: str = "NEUTRAL"
        self.confidence_in_projection: float = 0.5
        self.last_updated: float = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "known_facts": self.known_facts,
            "active_goals": self.active_goals,
            "perceived_mood": self.perceived_mood,
            "confidence": self.confidence_in_projection,
            "last_updated": self.last_updated
        }

class MindModel(AuraBaseModule):
    def __init__(self, data_path: Optional[Path] = None):
        super().__init__("MindModel")
        if not data_path:
            from core.config import config
            data_path = config.paths.data_dir / "consciousness" / "mind_model.json"
        
        self.data_path = data_path
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_state = UserBeliefState()
        self._load()

    def _load(self):
        if self.data_path.exists():
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    self.user_state.known_facts = data.get("known_facts", [])
                    self.user_state.active_goals = data.get("active_goals", [])
                    self.user_state.perceived_mood = data.get("perceived_mood", "NEUTRAL")
                    self.user_state.confidence_in_projection = data.get("confidence", 0.5)
                    self.user_state.last_updated = data.get("last_updated", time.time())
            except Exception as e:
                record_degradation('mind_model', e)
                self.logger.error("Failed to load mind model: %s", e)

    def save(self):
        try:
            with open(self.data_path, 'w') as f:
                json.dump(self.user_state.to_dict(), f, indent=2)
        except Exception as e:
            record_degradation('mind_model', e)
            self.logger.error("Failed to save mind model: %s", e)

    def update_projection(self, interaction_summary: str, current_mood: str):
        """Updates the internal model of the user based on recent interactions."""
        # This is where 'Intent Projection' logic lives.
        # In a full cognitive cycle, the LLM provides this summary.
        self.user_state.perceived_mood = current_mood
        self.user_state.last_updated = time.time()
        
        # Heuristic: If user is asking questions, they are acquiring facts.
        # This will be refined via prompt injections.
        self.logger.info("🧠 MindModel: User mood projected as %s", current_mood)
        self.save()

    def get_context_for_brain(self) -> str:
        """Returns a formatted string representing the user's mind state for the LLM."""
        state = self.user_state
        facts = ", ".join(state.known_facts[:5]) if state.known_facts else "unknown"
        goals = ", ".join(state.active_goals[:3]) if state.active_goals else "unknown"
        
        return (
            f"[THEORY OF MIND]\n"
            f"- User Perceived Mood: {state.perceived_mood}\n"
            f"- Projected User Goals: {goals}\n"
            f"- Assumed User Knowledge: {facts}\n"
            f"- Projection Confidence: {state.confidence_in_projection:.2f}"
        )

    def add_fact(self, fact: str):
        if fact not in self.user_state.known_facts:
            self.user_state.known_facts.append(fact)
            if len(self.user_state.known_facts) > 50:
                self.user_state.known_facts.pop(0)

    def add_goal(self, goal: str):
        if goal not in self.user_state.active_goals:
            self.user_state.active_goals.append(goal)
            if len(self.user_state.active_goals) > 10:
                self.user_state.active_goals.pop(0)