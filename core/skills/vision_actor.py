"""core/skills/vision_actor.py - Vision to Action Skill

This module implements the See -> Think -> Act loop using PyAutoGUI.
It allows the LLM to control the local UI by asking for visual bounding boxes.
"""

import asyncio
import logging
from typing import Dict, Any, Tuple, Optional

from pydantic import BaseModel, Field
from core.skills.base_skill import BaseSkill
from core.senses.screen_vision import LocalVision
from core.container import ServiceContainer
import re
import time

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.5
except ImportError:
    pyautogui = None

logger = logging.getLogger("Skills.VisionActor")

class VisionActorInput(BaseModel):
    action: str = Field("look", description="Action to perform: 'click', 'type', or 'look'.")
    target_desc: Optional[str] = Field(None, description="Visual description of the target on screen (e.g. 'Submit button', 'Search bar'). Required for 'click' or 'look'.")
    text_to_type: Optional[str] = Field(None, description="Text string to type. Required if action is 'type'.")
    press_enter: bool = Field(False, description="Press Enter after typing.")

class VisionActorSkill(BaseSkill):
    name = "sovereign_vision"
    description = "Control the computer UI. Can 'look' to find coordinates, 'click' visually described elements, or 'type' text."
    input_model = VisionActorInput
    metabolic_cost = 4  # Heavy due to Vision model + UI latency

    def __init__(self):
        super().__init__()
        self.vision_model = "vision-optimized"
        self._vision_engine = None

    def _lazy_init(self):
        """Fetch services from container at runtime to avoid import loops."""
        if not self._vision_engine:
            self._vision_engine = LocalVision(model=self.vision_model)

    async def execute(self, params: VisionActorInput, context: Dict[str, Any]) -> Dict[str, Any]:
        self._lazy_init()
        
        if not pyautogui:
            return {"ok": False, "summary": "Physical execution skipped: PyAutoGUI is not installed.", "action": params.action}

        action = params.action.lower()
        if action == "click":
            if not params.target_desc:
                return {"ok": False, "message": "Target description required for click.", "action": action}
            return await self._execute_look_and_click(params.target_desc)
        elif action == "look":
            if not params.target_desc:
                return {"ok": False, "message": "Target description required for look.", "action": action}
            return await self._execute_look(params.target_desc)
        elif action == "type":
            if not params.text_to_type:
                return {"ok": False, "message": "Text required for type action.", "action": action}
            return await self._execute_type(params.text_to_type, params.press_enter)
        else:
            return {"ok": False, "message": f"Unknown action: {action}", "action": action}

    async def _capture_screen(self) -> Optional[str]:
        if not self._vision_engine:
            return None
        try:
            # LocalVision.capture_screen returns a PIL image
            image = await self._vision_engine.capture_screen()
            if not image:
                return None
            
            from core.senses.screen_vision import _process_image_for_vlm
            img_b64 = _process_image_for_vlm(image)
            return img_b64
        except Exception as e:
            logger.error("Screen capture failed: %s", e)
            return None

    async def _locate_target(self, image_b64: str, target_desc: str) -> Optional[Tuple[int, int]]:
        prompt = (
            f"You are a UI automation parser. Look at this screen and find the center coordinates of: '{target_desc}'. "
            "Respond ONLY with a valid JSON strictly matching this schema: {\"x\": integer, \"y\": integer, \"found\": boolean}. "
            "Do not include any other text."
        )
        try:
            brain = ServiceContainer.get("cognitive_engine", default=None)
            if not brain:
                return None
            
            # Use Brain's unified think with images
            from core.brain.cognitive_engine import ThinkingMode
            response = await brain.think(prompt=f"ACTOR_IMAGE_PARSE: {prompt}", images=[image_b64], mode=ThinkingMode.QUICK)
            text = response.content if hasattr(response, 'content') else str(response)
            
            # Simple JSON extraction
            import json
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if data.get("found"):
                    return (data.get("x", 0), data.get("y", 0))
        except Exception as e:
            logger.error("Vision location failed: %s", e)
        return None

    async def _execute_look(self, target_desc: str) -> Dict[str, Any]:
        """Look but don't click."""
        img = await self._capture_screen()
        if not img:
            return {"ok": False, "summary": "Screen capture failed."}
            
        coords = await self._locate_target(img, target_desc)
        if not coords:
            return {"ok": False, "summary": f"Could not visually locate '{target_desc}' on screen."}
            
        return {
            "ok": True, 
            "summary": f"Target '{target_desc}' located at {coords}.",
            "coordinates": {"x": coords[0], "y": coords[1]}
        }

    async def _execute_look_and_click(self, target_desc: str) -> Dict[str, Any]:
        coords_payload = await self._execute_look(target_desc)
        if not coords_payload["ok"]:
            return coords_payload
            
        x = coords_payload["coordinates"]["x"]
        y = coords_payload["coordinates"]["y"]
        
        try:
            await asyncio.to_thread(self._physical_click, x, y)
            return {"ok": True, "summary": f"Successfully clicked '{target_desc}' at ({x}, {y})."}
        except pyautogui.FailSafeException:
            return {"ok": False, "summary": "Failsafe triggered. Mouse moving to corner aborted action."}
        except Exception as e:
            return {"ok": False, "summary": f"Click execution error: {e}"}

    def _physical_click(self, x: int, y: int):
        pyautogui.moveTo(x, y, duration=0.5, tween=pyautogui.easeInOutQuad)
        pyautogui.click()

    async def _execute_type(self, text: str, enter: bool) -> Dict[str, Any]:
        try:
            await asyncio.to_thread(pyautogui.write, text, interval=0.05)
            if enter:
                await asyncio.to_thread(pyautogui.press, "enter")
            return {"ok": True, "summary": f"Successfully typed text (enter={enter})."}
        except Exception as e:
            return {"ok": False, "summary": f"Typing failed: {e}"}



# Meta-registration for discovery
METADATA = {
    "name": VisionActorSkill.name,
    "description": VisionActorSkill.description,
    "skill_class": VisionActorSkill,
    "enabled": True
}