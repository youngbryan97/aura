import logging
from typing import Any

from pydantic import BaseModel, Field

from core.senses.notifications import DeliveryResult, DesktopNotifier
from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT

logger = logging.getLogger("Skills.NotifyUser")


class NotifyUserInput(BaseModel):
    title: str = Field(description="The bold title of the notification (default: Aura)", default="Aura")
    message: str = Field(description="The body text of the notification")
    sound: str = Field(description="macOS sound to play (Tink, Glass, Basso, Purr)", default="Glass")


class NotifyUserSkill(BaseSkill):
    """Proactively alerts the user via a native OS desktop notification."""
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT


    name = "notify_user"
    retry_safe = False  # external send/act — never double-fire on retry
    description = (
        "Pushes a native OS desktop notification to the user. "
        "Use this when completing a long-running background task, or when "
        "you encounter an urgent insight that shouldn't wait for the user to open the dashboard. "
        "The result reports whether the user was ACTUALLY reached — delivery can be "
        "suppressed by quiet hours or disabled notifications, and you must not "
        "assume the user saw anything unless delivered=True."
    )

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            params = NotifyUserInput(**params)

        outcome = DesktopNotifier.send(
            title=params.title,
            message=params.message,
            sound=params.sound
        )

        if outcome is None:
            # Legacy/mocked notifier without a delivery contract: the OS call
            # was made and did not raise, but delivery is unconfirmed.
            outcome = DeliveryResult(
                delivered=True,
                status="delivered",
                detail="Legacy notifier returned no delivery receipt.",
            )

        if outcome.delivered:
            return {
                "ok": True,
                "status": "success",
                "delivered": True,
                "message": f"Notification delivered to the user's desktop: '{params.message}'",
            }

        if outcome.status in ("disabled", "suppressed_quiet_hours"):
            # Not a system fault — the user configured this. But Aura must
            # know the user was NOT reached so she can choose another channel
            # (or wait) instead of believing the message landed.
            return {
                "ok": True,
                "status": outcome.status,
                "delivered": False,
                "message": (
                    f"Notification NOT delivered ({outcome.status}): {outcome.detail} "
                    "The user has not seen this message — use the chat channel or wait "
                    "for quiet hours to end if it matters."
                ),
            }

        return {
            "ok": False,
            "status": "failed",
            "delivered": False,
            "error": f"Notification delivery failed: {outcome.detail or 'unknown OS error'}",
        }
