import logging
from typing import Any

from pydantic import BaseModel, Field

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.runtime.errors import record_degradation
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.personality_skill")


class PersonalityInput(BaseModel):
    action: str = Field(..., description="Action to perform: 'set', 'get', 'list', or 'speak'.")
    persona: str | None = Field(None, description="The persona ID to set (required for 'set' action).")
    text: str | None = Field(None, description="The text to speak or style (required for 'speak' action).")

class PersonalitySkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "personality"
    description = "Manage and query Aura's active persona (set/list/get)."
    input_model = PersonalityInput

    def __init__(self):
        self.logger = logging.getLogger("Skills.personality")
        # Lazy-import persona adapter
        try:
            from core.brain.persona_adapter import PersonaAdapter
            self.adapter = PersonaAdapter()
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('personality_skill', e)
            self.logger.error("Failed to load PersonaAdapter: %s", e)
            self.adapter = None

    def match(self, goal: dict[str, Any]) -> bool:
        obj = goal.get("objective", "").lower()
        return "persona" in obj or "speak as" in obj or "set persona" in obj

    async def execute(
        self,
        params: PersonalityInput,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.adapter:
            return {"ok": False, "error": "Persona system not available"}

        if isinstance(params, dict):
            try:
                params = PersonalityInput(**params)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
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
