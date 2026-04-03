from core.utils.exceptions import capture_and_log
import asyncio
import logging
import shlex
import subprocess
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from core.skills.base_skill import BaseSkill
import pyautogui

logger = logging.getLogger("Skills.ComputerUse")


class ComputerUseParams(BaseModel):
    action: str = Field(..., description="click|type|hotkey|scroll|read_screen_text|open_app|run_command")
    target: str = Field("", description="Element description, text to type, key combo, or command")
    x: int = Field(0, description="Screen x coordinate for click/scroll")
    y: int = Field(0, description="Screen y coordinate for click/scroll")


class ComputerUseSkill(BaseSkill):
    name = "computer_use"
    description = "Directly control the computer: click, type, read screen text, run commands, open apps."
    input_model = ComputerUseParams
    metabolic_cost = 2
    
    # SK-01: Restricted command set for autonomous use
    ALLOWED_COMMANDS = frozenset([
        "ls", "pwd", "echo", "cat", "find", "grep", 
        "python3", "pip", "git", "mkdir", "touch"
    ])

    async def execute(self, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            params = ComputerUseParams(**params)

        action = params.action

        # Mycelial root pulse: Agent executing computer control
        try:
            from core.container import ServiceContainer
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                hypha = mycelium.get_hypha("skill", "os")
                if hypha: hypha.pulse(success=True)
        except Exception as e:
            capture_and_log(e, {'module': __name__})

        try:
            if action == "read_screen_text":
                result = await asyncio.to_thread(self._read_screen_text_macos)
                return {"ok": True, "text": result}

            elif action == "click":
                # Issue 88: Use x/y correctly
                await asyncio.to_thread(pyautogui.click, x=params.x, y=params.y)
                return {"ok": True, "action": f"clicked ({params.x},{params.y})"}

            elif action == "type":
                await asyncio.to_thread(pyautogui.typewrite, params.target, interval=0.05)
                return {"ok": True, "typed": params.target[:50]}

            elif action == "hotkey":
                keys = params.target.split("+")
                await asyncio.to_thread(pyautogui.hotkey, *keys)
                return {"ok": True, "hotkey": params.target}

            elif action == "scroll":
                # Issue 88: Use x/y correctly
                clicks = int(params.target or "3")
                await asyncio.to_thread(pyautogui.scroll, clicks, x=params.x, y=params.y)
                return {"ok": True, "scrolled": clicks}

            elif action == "run_command":
                try:
                    args = shlex.split(params.target)
                except ValueError as e:
                    return {"ok": False, "error": f"Invalid command syntax: {e}"}

                if not args:
                    return {"ok": False, "error": "No command provided."}

                cmd = args[0]
                if cmd not in self.ALLOWED_COMMANDS:
                    logger.warning("🛡️ SK-01 Blocked: Command '%s' not in allowlist.", cmd)
                    return {"ok": False, "error": f"Security Violation: Command '{cmd}' is restricted."}

                result = await asyncio.to_thread(
                    subprocess.run,
                    args,
                    capture_output=True, text=True, timeout=30
                )
                output = (result.stdout or result.stderr or "").strip()[:3000]
                return {"ok": True, "output": output, "exit_code": result.returncode}

            elif action == "open_app":
                await asyncio.to_thread(subprocess.run, ["open", "-a", params.target])
                return {"ok": True, "opened": params.target}

            else:
                return {"ok": False, "error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error("ComputerUse action '%s' failed: %s", action, e)
            return {"ok": False, "error": str(e)}

    def read_screen_text(self) -> str:
        """Helper for AgencyCore to read screen text directly."""
        return self._read_screen_text_macos()

    def _read_screen_text_macos(self) -> str:
        """Use macOS Accessibility API to extract text from the frontmost app."""
        script = '''
tell application "System Events"
    set frontApp to first application process whose frontmost is true
    set appName to name of frontApp
    set allText to ""
    try
        set allText to entire contents of frontApp as string
    end try
    return appName & ": " & allText
end tell
'''
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip()[:3000]
        except Exception as e:
            return f"[read_screen_text failed: {e}]"
