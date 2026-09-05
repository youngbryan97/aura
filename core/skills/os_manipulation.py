import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from core.runtime.errors import record_degradation
from core.skills._pyautogui_runtime import get_pyautogui
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.Hands")

class OSManipulationInput(BaseModel):
    action: str = Field(..., description="Action to perform: 'type', 'click', 'scroll', 'open_app', 'press', 'hotkey'")
    text: str | None = Field(None, description="Text to type.")
    x: int | None = Field(None, description="X coordinate for clicking.")
    y: int | None = Field(None, description="Y coordinate for clicking.")
    button: str | None = Field("left", description="Mouse button: 'left', 'right', 'middle'.")
    clicks: int | None = Field(1, description="Number of clicks.")
    amount: int | None = Field(0, description="Amount to scroll.")
    app_name: str | None = Field(None, description="Name of the app to open.")
    key: str | None = Field(None, description="Key to press.")
    keys: list[str] | None = Field(None, description="List of keys for a hotkey combination.")
    speed: float | None = Field(0.05, description="Typing speed (interval between keys).")

class DesktopControlSkill(BaseSkill):
    """The 'Hands' of the machine.
    Allows Aura to click, type, and scroll.
    """

    name = "os_manipulation"
    description = "Manipulate the mouse and keyboard to interact with the OS using PyAutoGUI."
    input_model = OSManipulationInput

    async def _require_accessibility(self, capability: str) -> dict[str, Any] | None:
        try:
            from core.container import ServiceContainer
            from core.security.permission_guard import PermissionType
        except (ImportError, AttributeError, RuntimeError):
            return None

        guard = ServiceContainer.get("permission_guard", default=None)
        if guard is None:
            return None

        check = await guard.check_permission(PermissionType.ACCESSIBILITY, force=True)
        if check.get("granted"):
            return None
        return {
            "ok": False,
            "status": check.get("status", "denied"),
            "error": f"Accessibility permission is required for {capability}.",
            "permission": "accessibility",
            "guidance": check.get("guidance", ""),
            "detail": check.get("detail", ""),
        }
    
    async def execute(self, params: OSManipulationInput, context: dict[str, Any]) -> dict[str, Any]:
        """Router for physical actions."""
        if isinstance(params, dict):
            try:
                params = OSManipulationInput(**params)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('os_manipulation', e)
                return {"ok": False, "error": f"Invalid input: {e}"}

        blocked = await self._require_accessibility("desktop mouse and keyboard control")
        if blocked:
            return blocked

        action = params.action
        from core.container import ServiceContainer
        host_auto = ServiceContainer.get("host_automation", default=None)
        if host_auto:
            logger.warning("🖐️ OS MANIPULATION (Governed HostAutomation): %s %s", action, params)
            if action == "type":
                text = params.text
                if not text:
                    return {"ok": False, "error": "No text provided to type."}
                res = await host_auto.type_text(text)
                return {"ok": res.success, "result": res.result or res.error, "receipt_id": res.receipt_id}
            elif action == "click":
                x = params.x
                y = params.y
                button = params.button or "left"
                if x is not None and y is not None:
                    res = await host_auto.click_at(x, y, button=button)
                else:
                    pyautogui, pyautogui_error = get_pyautogui()
                    if pyautogui:
                        cx, cy = pyautogui.position()
                        res = await host_auto.click_at(cx, cy, button=button)
                    else:
                        detail = f": {pyautogui_error}" if pyautogui_error else ""
                        return {
                            "ok": False,
                            "error": (
                                "Coordinates not specified and cursor position "
                                f"unavailable{detail}"
                            ),
                            "status": "unavailable",
                        }
                return {"ok": res.success, "result": res.result or res.error, "receipt_id": res.receipt_id}
            elif action == "scroll":
                amount = params.amount or 0
                res = await host_auto.scroll(dy=amount)
                return {"ok": res.success, "result": res.result or res.error, "receipt_id": res.receipt_id}
            elif action == "open_app":
                app_name = params.app_name
                if not app_name:
                    return {"ok": False, "error": "No app name provided."}
                res = await host_auto.launch_app(app_name)
                return {"ok": res.success, "result": res.result or res.error, "receipt_id": res.receipt_id}
            elif action == "press":
                key = params.key
                if not key:
                    return {"ok": False, "error": "No key provided."}
                res = await host_auto.hotkey(key)
                return {"ok": res.success, "result": res.result or res.error, "receipt_id": res.receipt_id}
            elif action == "hotkey":
                keys = params.keys or []
                if not keys:
                    return {"ok": False, "error": "No keys provided for hotkey."}
                res = await host_auto.hotkey(*keys)
                return {"ok": res.success, "result": res.result or res.error, "receipt_id": res.receipt_id}
            return {"ok": False, "error": f"Action '{action}' not recognized."}

        pyautogui, pyautogui_error = get_pyautogui()
        if pyautogui is None:
            detail = f": {pyautogui_error}" if pyautogui_error else ""
            return {
                "ok": False,
                "error": f"PyAutoGUI unavailable{detail}",
                "status": "unavailable",
            }

        logger.warning("🖐️ OS MANIPULATION (Direct PyAutoGUI): %s %s", action, params)

        if action == "type":
            text = params.text
            interval = params.speed or 0.05
            if not text:
                return {"ok": False, "error": "No text provided to type."}
            await asyncio.to_thread(pyautogui.write, text, interval=interval)
            return {"ok": True, "result": f"Typed: {text[:20]}..."}
            
        elif action == "click":
            x = params.x
            y = params.y
            button = params.button or "left"
            clicks = params.clicks or 1
            
            if x is not None and y is not None:
                await asyncio.to_thread(pyautogui.click, x, y, button=button, clicks=clicks)
            else:
                await asyncio.to_thread(pyautogui.click, button=button, clicks=clicks)
            return {"ok": True, "result": f"Clicked {button} at ({x or 'current'}, {y or 'current'})"}

        elif action == "scroll":
            amount = params.amount or 0
            await asyncio.to_thread(pyautogui.scroll, amount)
            return {"ok": True, "result": f"Scrolled {amount}"}

        elif action == "open_app":
            app_name = params.app_name
            if not app_name:
                return {"ok": False, "error": "No app name provided."}
            
            # macOS Spotlight trick
            await asyncio.to_thread(pyautogui.hotkey, 'command', 'space')
            await asyncio.sleep(0.5)
            await asyncio.to_thread(pyautogui.write, app_name)
            await asyncio.sleep(0.5)
            await asyncio.to_thread(pyautogui.press, 'enter')
            return {"ok": True, "result": f"Launched signal for {app_name}"}

        elif action == "press":
            key = params.key
            if not key:
                return {"ok": False, "error": "No key provided."}
            await asyncio.to_thread(pyautogui.press, key)
            return {"ok": True, "result": f"Pressed {key}"}

        elif action == "hotkey":
            keys = params.keys or []
            if not keys:
                return {"ok": False, "error": "No keys provided for hotkey."}
            await asyncio.to_thread(pyautogui.hotkey, *keys)
            return {"ok": True, "result": f"Pressed hotkey: {'+'.join(keys)}"}

        return {"ok": False, "error": f"Action '{action}' not recognized."}
