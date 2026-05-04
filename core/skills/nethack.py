from pydantic import BaseModel, Field
from typing import Any, Dict
from core.skills.base_skill import BaseSkill
from core.container import ServiceContainer
import logging

logger = logging.getLogger("Skills.NetHack")

class NetHackParams(BaseModel):
    action: str = Field(
        ...,
        description="The exact key or sequence to send to the NetHack terminal. Examples: 'y', 'n', 'ESC', 'SPACE', 'ENTER', 'i', 'h', 'j', 'k', 'l'",
    )

class NetHackSkill(BaseSkill):
    name = "execute_nethack_action"
    description = "Send physical keystrokes to the active NetHack game session. Use 'ESC' to cancel menus or 'SPACE' to advance prompts."
    input_model = NetHackParams
    metabolic_cost = 1

    async def execute(self, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            params = NetHackParams(**params)

        adapter = ServiceContainer.get("nethack_adapter", default=None)
        if not adapter:
            return {"ok": False, "error": "No active NetHack session found."}

        action = params.action.upper() if len(params.action) > 1 else params.action
        
        # Translate special keys
        physical_key = action
        if action == "ESC" or action == "ESCAPE":
            physical_key = '\x1b'
        elif action == "SPACE":
            physical_key = ' '
        elif action == "ENTER" or action == "RETURN":
            physical_key = '\n'

        logger.info(f"NetHackTool executing action: '{action}'")
        try:
            adapter.send_action(physical_key)
            return {
                "ok": True, 
                "message": f"Action '{action}' sent successfully to NetHack.",
                "action": action
            }
        except Exception as e:
            logger.error(f"Failed to send NetHack action: {e}")
            return {"ok": False, "error": str(e)}
