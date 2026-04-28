from core.runtime.errors import record_degradation
import logging
import os
from typing import Any, Dict, Optional

from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.personality_skill")


from pydantic import BaseModel, Field


class PersonalityInput(BaseModel):
    action: str = Field(..., description="Action to perform: 'set', 'get', 'list', or 'speak'.")
    persona: Optional[str] = Field(None, description="The persona ID to set (required for 'set' action).")
    text: Optional[str] = Field(None, description="The text to speak or style (required for 'speak' action).")

class PersonalitySkill(BaseSkill):
    name = "personality"
    description = "Manage and query Aura's active persona (set/list/get)."
    input_model = PersonalityInput

    def __init__(self):
        self.logger = logging.getLogger("Skills.personality")
        # Lazy-import persona adapter
        try:
            from core.brain.persona_adapter import PersonaAdapter
            self.adapter = PersonaAdapter()
        except Exception as e:
            record_degradation('personality_skill', e)
            self.logger.error("Failed to load PersonaAdapter: %s", e)
            self.adapter = None

    def match(self, goal: Dict[str, Any]) -> bool:
        obj = goal.get("objective", "").lower()
        return "persona" in obj or "speak as" in obj or "set persona" in obj

    async def execute(self, params: PersonalityInput, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.adapter:
            return {"ok": False, "error": "Persona system not available"}

        if isinstance(params, dict):
            try:
                params = PersonalityInput(**params)
            except Exception as e:
                record_degradation('personality_skill', e)
                return {"ok": False, "error": f"Invalid input: {e}"}

        action = params.action
        persona = params.persona
        text = params.text

        if action == "list":
            return {"ok": True, "personas": self.adapter.list_personas()}

        if action == "get":
            active = self.adapter.get_active()
            return {"ok": True, "active": active}

        if action == "set":
            if not persona:
                return {"ok": False, "error": "Missing persona name"}
            ok = self.adapter.set_persona(persona)
            return {"ok": ok, "persona": persona}

        if action == "speak":
            if not text:
                return {"ok": False, "error": "Missing text to speak"}
            active = self.adapter.get_active()
            if not active:
                return {"ok": False, "error": "No active persona set"}
            styled = self.adapter.apply_style(text)
            return {"ok": True, "text": styled}

        return {"ok": False, "error": "Unknown action"}