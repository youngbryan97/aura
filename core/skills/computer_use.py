from core.utils.exceptions import capture_and_log
import asyncio
import logging
import shutil
import shlex
import subprocess
import urllib.parse
import webbrowser
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from core.skills.base_skill import BaseSkill
from core.skills._pyautogui_runtime import get_pyautogui

logger = logging.getLogger("Skills.ComputerUse")


class ComputerUseParams(BaseModel):
    action: str = Field(
        ...,
        description="click|type|hotkey|scroll|read_screen_text|read_menu_clock|open_app|open_url|run_command",
    )
    target: str = Field("", description="Element description, text to type, key combo, command, app name, or URL")
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

    async def _require_permissions(self, capability: str, *permission_names: str) -> Optional[Dict[str, Any]]:
        try:
            from core.container import ServiceContainer
            from core.security.permission_guard import PermissionType
        except Exception:
            return None

        guard = ServiceContainer.get("permission_guard", default=None)
        if guard is None:
            return None

        for permission_name in permission_names:
            permission_type = getattr(PermissionType, permission_name, None)
            if permission_type is None:
                continue
            check = await guard.check_permission(permission_type, force=True)
            if check.get("granted"):
                continue
            human_name = permission_name.replace("_", " ").title()
            return {
                "ok": False,
                "status": check.get("status", "denied"),
                "error": f"{human_name} permission is required for {capability}.",
                "permission": permission_name.lower(),
                "guidance": check.get("guidance", ""),
                "detail": check.get("detail", ""),
            }
        return None

    @staticmethod
    def _normalize_script_error(stderr: str) -> str:
        message = (stderr or "").strip()
        lowered = message.lower()
        if "not authorized to send apple events" in lowered or "(-1743)" in lowered:
            return "Automation permission is blocked for System Events."
        if "not allowed assistive access" in lowered or "(-1719)" in lowered:
            return "Accessibility permission is blocked for desktop UI inspection."
        return message or "AppleScript execution failed."

    def _run_applescript(self, script: str, *, timeout: int = 10) -> str:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(self._normalize_script_error(result.stderr or result.stdout))
        return (result.stdout or "").strip()

    @staticmethod
    def _normalize_open_url_target(target: str) -> str:
        text = str(target or "").strip()
        if not text:
            return ""
        if text.startswith(("http://", "https://")):
            return text
        return f"https://duckduckgo.com/?q={urllib.parse.quote_plus(text)}"

    @staticmethod
    def _runtime_permission_payload(message: str) -> Optional[Dict[str, Any]]:
        try:
            from core.security.permission_guard import PermissionType, get_permission_guard
        except Exception:
            return None

        guard = get_permission_guard()
        if "Accessibility permission is blocked" in message:
            return {
                "ok": False,
                "status": "denied",
                "error": message,
                "permission": "accessibility",
                "guidance": guard.get_guidance(PermissionType.ACCESSIBILITY),
            }
        if "Automation permission is blocked" in message:
            return {
                "ok": False,
                "status": "denied",
                "error": message,
                "permission": "automation",
                "guidance": guard.get_guidance(PermissionType.AUTOMATION),
            }
        return None

    async def execute(self, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            params = ComputerUseParams(**params)

        action = params.action
        pyautogui = None
        pyautogui_error = None
        if action in {"click", "type", "hotkey", "scroll"}:
            pyautogui, pyautogui_error = get_pyautogui()
            if pyautogui is None:
                detail = f": {pyautogui_error}" if pyautogui_error else ""
                return {
                    "ok": False,
                    "error": f"PyAutoGUI unavailable{detail}",
                    "status": "unavailable",
                }
            blocked = await self._require_permissions(
                "desktop mouse and keyboard control",
                "ACCESSIBILITY",
            )
            if blocked:
                return blocked

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
                blocked = await self._require_permissions(
                    "reading text from the frontmost macOS app",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    return blocked
                result = await asyncio.to_thread(self._read_screen_text_macos)
                return {"ok": True, "text": result}

            elif action == "read_menu_clock":
                blocked = await self._require_permissions(
                    "reading the macOS menu bar clock",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    return blocked
                result = await asyncio.to_thread(self._read_menu_clock_macos)
                return {"ok": True, "clock_text": result, "text": result}

            elif action == "click":
                pre_state_text = ""
                post_state_text = ""
                # Optional pre-verification
                try:
                    pre_state_text = await asyncio.to_thread(self._read_screen_text_macos)
                except Exception:
                    pass

                await asyncio.to_thread(pyautogui.click, x=params.x, y=params.y)
                
                # Verify state shift
                try:
                    await asyncio.sleep(0.5)
                    post_state_text = await asyncio.to_thread(self._read_screen_text_macos)
                except Exception:
                    pass
                
                verification = "State shifted." if pre_state_text != post_state_text else "No obvious state shift detected."
                return {"ok": True, "action": f"clicked ({params.x},{params.y})", "verification": verification}

            elif action == "type":
                await asyncio.to_thread(pyautogui.typewrite, params.target, interval=0.03)
                
                # Check what was typed roughly
                try:
                    post_state = await asyncio.to_thread(self._read_screen_text_macos)
                    if params.target[:10] in post_state:
                        pass # Typed text is visible
                except Exception:
                    pass
                    
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

            elif action == "open_url":
                target_url = self._normalize_open_url_target(params.target)
                if not target_url:
                    return {"ok": False, "error": "No URL or search query provided."}
                if target_url.startswith("file:"):
                    return {"ok": False, "error": "Refusing to open local file URLs from chat."}
                if shutil.which("open"):
                    result = await asyncio.to_thread(
                        subprocess.run,
                        ["open", target_url],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode != 0:
                        error = (result.stderr or result.stdout or "open command failed").strip()
                        return {"ok": False, "error": error}
                else:
                    opened = await asyncio.to_thread(webbrowser.open, target_url, 2)
                    if not opened:
                        return {"ok": False, "error": "The default browser did not accept the URL."}
                return {
                    "ok": True,
                    "action": "open_url",
                    "url": target_url,
                    "summary": f"I opened a browser tab for {target_url}.",
                }

            else:
                return {"ok": False, "error": f"Unknown action: {action}"}

        except Exception as e:
            runtime_permission_error = self._runtime_permission_payload(str(e))
            if runtime_permission_error:
                return runtime_permission_error
            logger.error("ComputerUse action '%s' failed: %s", action, e)
            return {"ok": False, "error": str(e)}

    def read_screen_text(self) -> str:
        """Helper for AgencyCore to read screen text directly."""
        try:
            return self._read_screen_text_macos()
        except Exception as e:
            return f"[read_screen_text failed: {e}]"

    def read_menu_clock(self) -> str:
        """Helper for reading the macOS menu bar clock."""
        try:
            return self._read_menu_clock_macos()
        except Exception as e:
            return f"[read_menu_clock failed: {e}]"

    def _read_screen_text_macos(self) -> str:
        """Use macOS Accessibility API to extract text from the frontmost app with anti-hang limits."""
        script = '''
tell application "System Events"
    try
        set frontApp to first application process whose frontmost is true
        set appName to name of frontApp
        set allText to entire contents of frontApp as string
        return appName & ": " & allText
    on error
        return "[Accessibility error or UI unresponsive]"
    end try
end tell
'''
        raw = self._run_applescript(script, timeout=6)
        if len(raw) > 3000:
            return raw[:1500] + "\n... [TRUNCATED] ...\n" + raw[-1500:]
        return raw

    def _read_menu_clock_macos(self) -> str:
        """Read the live menu bar clock through System Events."""
        script = '''
tell application "System Events"
    tell process "SystemUIServer"
        return name of first menu bar item of menu bar 1 whose description is "Clock"
    end tell
end tell
'''
        return self._run_applescript(script, timeout=10)[:240]
