import asyncio
import logging
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
import pyautogui

from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.Hands")

class OSManipulationInput(BaseModel):
    action: str = Field(..., description="Action to perform: 'type', 'click', 'scroll', 'open_app', 'press', 'hotkey'")
    text: Optional[str] = Field(None, description="Text to type.")
    x: Optional[int] = Field(None, description="X coordinate for clicking.")
    y: Optional[int] = Field(None, description="Y coordinate for clicking.")
    button: Optional[str] = Field("left", description="Mouse button: 'left', 'right', 'middle'.")
    clicks: Optional[int] = Field(1, description="Number of clicks.")
    amount: Optional[int] = Field(0, description="Amount to scroll.")
    app_name: Optional[str] = Field(None, description="Name of the app to open.")
    key: Optional[str] = Field(None, description="Key to press.")
    keys: Optional[List[str]] = Field(None, description="List of keys for a hotkey combination.")
    speed: Optional[float] = Field(0.05, description="Typing speed (interval between keys).")

class DesktopControlSkill(BaseSkill):
    """The 'Hands' of the machine.
    Allows Aura to click, type, and scroll.
    """

    name = "os_manipulation"
    description = "Manipulate the mouse and keyboard to interact with the OS using PyAutoGUI."
    input_model = OSManipulationInput
    
    async def execute(self, params: OSManipulationInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Router for physical actions."""
        if isinstance(params, dict):
            try:
                params = OSManipulationInput(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input: {e}"}

        action = params.action
        
        logger.warning("🖐️ OS MANIPULATION: %s %s", action, params)

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