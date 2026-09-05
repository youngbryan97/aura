"""core/actuation/email_actuator.py — Email Draft and Send Actuator."""
from __future__ import annotations

from typing import Any, Dict
from core.actuation.world_actuator import get_world_actuator


class EmailActuator:
    """Wrapper for drafting and sending emails."""

    @classmethod
    async def create_draft(cls, to: str, subject: str, body: str, source: str = "email_actuator") -> Dict[str, Any]:
        return await get_world_actuator().actuate(
            category="email_drafts",
            action_name="create_draft",
            params={"to": to, "subject": subject, "body": body},
            source=source,
        )

    @classmethod
    async def send_email(cls, to: str, subject: str, body: str, source: str = "email_actuator") -> Dict[str, Any]:
        # High risk: actual sending
        return await get_world_actuator().actuate(
            category="email_drafts",
            action_name="send_message",
            params={"to": to, "subject": subject, "body": body},
            source=source,
            high_risk_flag=True,
        )
