"""core/actuation/calendar_actuator.py — Calendar Actuator."""
from __future__ import annotations

from typing import Any, Dict
from core.actuation.world_actuator import get_world_actuator


class CalendarActuator:
    """Wrapper for creating calendar events and drafts."""

    @classmethod
    async def create_event_draft(cls, title: str, start_time: str, end_time: str, source: str = "calendar_actuator") -> Dict[str, Any]:
        return await get_world_actuator().actuate(
            category="calendar_drafts",
            action_name="create_event_draft",
            params={"title": title, "start_time": start_time, "end_time": end_time},
            source=source,
        )
