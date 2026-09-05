"""core/body/browser_motor.py
Browser interaction motor channel.
"""
import logging
from typing import Any, Dict

from core.body.motor_controller import BaseMotor
from core.runtime.errors import record_degradation

logger = logging.getLogger("Body.BrowserMotor")

_BROWSER_MOTOR_ERRORS = (AttributeError, ImportError, RuntimeError, TimeoutError, TypeError, ValueError)


class BrowserMotor(BaseMotor):
    """Executes actions on a browser session (navigating, clicking DOM nodes)."""

    @property
    def name(self) -> str:
        return "browser"

    async def actuate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "load_url")
        url = params.get("url", "about:blank")
        
        logger.info("Actuating browser motor: %s -> %s", action, url)
        try:
            from core.capabilities.browser_controller import get_browser_controller

            controller = get_browser_controller()
            if action in {"load_url", "open_url", "navigate"}:
                receipt = await controller.open_url(str(url), new_tab=bool(params.get("new_tab", True)))
            elif action in {"search", "search_and_open"}:
                receipt = await controller.search_and_open(
                    str(params.get("query") or url),
                    count=int(params.get("count", 3)),
                )
            else:
                return {"status": "error", "action": action, "message": f"Unsupported browser action: {action}"}
            return {
                "status": "success" if getattr(receipt, "success", False) else "failed",
                "action": action,
                "url": url,
                "receipt_id": getattr(receipt, "receipt_id", ""),
                "result": getattr(receipt, "result", ""),
                "error": getattr(receipt, "error", ""),
            }
        except _BROWSER_MOTOR_ERRORS as exc:
            record_degradation("body.browser_motor", exc)
            logger.error("Browser motor failed: %s", exc)
            return {"status": "failed", "action": action, "url": url, "error": str(exc)}
