import logging
from typing import Any, Dict

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.senses.notifications import DesktopNotifier

logger = logging.getLogger("Skills.NotifyUser")


class NotifyUserInput(BaseModel):
    title: str = Field(description="The bold title of the notification (default: Aura)", default="Aura")
    message: str = Field(description="The body text of the notification")
    sound: str = Field(description="macOS sound to play (Tink, Glass, Basso, Purr)", default="Glass")


class NotifyUserSkill(BaseSkill):
    """Proactively alerts the user via a native OS desktop notification."""

    name = "notify_user"
    description = (
        "Pushes a native OS desktop notification to the user. "
        "Use this when completing a long-running background task, or when "
        "you encounter an urgent insight that shouldn't wait for the user to open the dashboard."
    )

    async def execute(self, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            params = NotifyUserInput(**params)

        DesktopNotifier.send(
            title=params.title,
            message=params.message,
            sound=params.sound
        )

        return {
            "ok": True,
            "status": "success",
            "message": f"Successfully pushed notification: '{params.message}'",
            "delivered": True
        }
