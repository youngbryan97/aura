"""core/actuation/browser_actuator.py — Browser and Network Actuator."""
from __future__ import annotations

from typing import Any, Dict
from core.actuation.world_actuator import get_world_actuator


class BrowserActuator:
    """Wrapper for browser automation and remote navigation actions."""

    @classmethod
    async def navigate(cls, url: str, source: str = "browser_actuator") -> Dict[str, Any]:
        return await get_world_actuator().actuate(
            category="browser",
            action_name="navigate",
            params={"url": url, "method": "GET"},
            source=source,
        )

    @classmethod
    async def perform_click(cls, selector: str, source: str = "browser_actuator") -> Dict[str, Any]:
        return await get_world_actuator().actuate(
            category="browser",
            action_name="click",
            params={"selector": selector},
            source=source,
        )

    @classmethod
    async def post_form(cls, url: str, data: Dict[str, Any], source: str = "browser_actuator") -> Dict[str, Any]:
        # High risk check for posting data publicly
        return await get_world_actuator().actuate(
            category="browser",
            action_name="post_publicly" if "post" in url else "submit_form",
            params={"url": url, "method": "POST", "data": data},
            source=source,
            high_risk_flag=True,
        )
