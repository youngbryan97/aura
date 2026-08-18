"""core/capabilities/host_automation.py — Generalized OS Automation Provider
=============================================================================
The single abstraction that gives Aura arbitrary OS manipulation through
governed primitives — NO hardcoded app-specific logic.

Routes through the most reliable adapter automatically:
    1. Direct API (file ops, system settings) — safest
    2. AppleScript / System Events — reliable for app control
    3. Accessibility API — for UI element interaction
    4. PyAutoGUI — fallback for generic screen control

Every call produces a ToolExecutionReceipt and goes through
CapabilityEngine + UnifiedWill. No action bypasses governance.

Usage:
    provider = get_host_automation()
    result = await provider.launch_app("Notes")
    result = await provider.get_frontmost_app()
    result = await provider.execute_applescript(script)  # AST-guarded
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.governance.will import ActionDomain
from core.governance_context import local_internal_governed_scope
from core.runtime.action_executor import ActionExecutor
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.security.execution_authority import (
    KIND_SHELL,
    authorize_execution,
    release_execution,
)

logger = logging.getLogger("Aura.HostAutomation")


def _as_applescript_string(value: Any) -> str:
    """Encode an arbitrary value as a safe quoted AppleScript string literal.

    App names and menu-path components are caller-controlled and were
    interpolated raw inside "..." in AppleScript source — a quote/backslash
    could break out of the literal and inject script. This escapes backslashes
    and quotes, strips control characters, and bounds the length so the value
    can only ever be data inside its own string literal.
    """
    text = "".join(ch for ch in str(value or "") if ch == " " or ord(ch) >= 32)[:256]
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _resolve_screenshot_path_policy(save_path: str) -> tuple[Path | None, tuple[Path, ...]]:
    """Resolve a requested capture path and its allowed roots off the event loop."""
    try:
        resolved = Path(save_path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        resolved = None
    roots = (
        (Path(state_root()) / "data").resolve(),
        (Path.home() / "Desktop" / "Aura").resolve(),
        (Path.home() / "Documents" / "Aura").resolve(),
    )
    return resolved, roots


_HOST_AUTOMATION_ERRORS = (
    ImportError,
    OSError,
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
    asyncio.TimeoutError,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AutomationReceipt:
    """Immutable receipt for every host automation action."""
    action: str
    target: str
    adapter: str                    # "applescript", "accessibility", "pyautogui", "direct_api"
    success: bool
    result: Any = None
    error: str = ""
    duration_ms: float = 0.0
    script_hash: str = ""           # SHA256 of any executed script
    timestamp: float = field(default_factory=time.time)
    receipt_id: str = ""
    #: Where the recognized text actually sat, for perception actions that
    #: read a screen. `result` is the words in reading order; this is the same
    #: reading with its geometry intact, which is the difference between
    #: knowing a screen says "2048" and knowing where on the screen it says it.
    layout: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.receipt_id:
            payload = f"{self.timestamp}|{self.action}|{self.target}|{self.success}"
            self.receipt_id = hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Script Safety Guard
# ---------------------------------------------------------------------------

class ScriptASTGuard:
    """Validates AppleScript and shell commands before execution.

    Blocks destructive patterns while allowing standard app control.
    This is the safety boundary for dynamic script compilation.
    """

    # Patterns that are ALWAYS blocked in AppleScript
    BLOCKED_APPLESCRIPT_PATTERNS = [
        r'\bdo\s+shell\s+script\s+.*\brm\s+(-[rRf]+\s+)?/',   # rm -rf from AppleScript
        r'\bdo\s+shell\s+script\s+.*\bsudo\b',                  # sudo from AppleScript
        r'\bdo\s+shell\s+script\s+.*\bcurl\s+.*\|\s*sh\b',      # curl | sh
        r'\bdo\s+shell\s+script\s+.*\bwget\s+.*\|\s*sh\b',      # wget | sh
        r'\bdo\s+shell\s+script\s+.*\bmkfs\b',                  # filesystem format
        r'\bdo\s+shell\s+script\s+.*\bdd\s+if=',                # disk duplicate
        r'\bdo\s+shell\s+script\s+.*\bformat\b',                # disk format
        r'\bdo\s+shell\s+script\s+.*\bshutdown\b',              # system shutdown
        r'\bdo\s+shell\s+script\s+.*\breboot\b',                # system reboot
        r'\bdo\s+shell\s+script\s+.*\blaunchctl\s+unload\b',    # service unload
        r'\bdo\s+shell\s+script\s+.*\bkillall\b',               # mass kill
        r'\bdelete\s+every\s+',                                  # mass delete in apps
        r'\bdo\s+shell\s+script\s+.*\bsecurity\s+delete\b',     # keychain delete
    ]

    # Patterns that are ALWAYS allowed
    ALLOWED_APPLESCRIPT_COMMANDS = {
        "tell application", "activate", "set", "get", "click",
        "keystroke", "key code", "delay", "return", "end tell",
        "name of", "title of", "window", "menu item", "menu bar",
        "open location", "make new", "set value", "frontmost",
        "bounds of", "size of", "position of", "count",
        "properties of", "exists", "close", "save",
    }

    @classmethod
    def validate_applescript(cls, script: str) -> tuple[bool, str]:
        """Validate an AppleScript for safety.

        Returns (is_safe, reason).
        """
        if not script or not script.strip():
            return False, "Empty script"

        script_lower = script.lower()

        # Check blocked patterns
        for pattern in cls.BLOCKED_APPLESCRIPT_PATTERNS:
            if re.search(pattern, script_lower, re.IGNORECASE):
                return False, f"Blocked pattern detected: {pattern[:60]}"

        # Check for `do shell script` — allowed only with safe commands
        if "do shell script" in script_lower:
            # Extract the shell command
            shell_match = re.findall(
                r'do\s+shell\s+script\s+["\'](.+?)["\']',
                script, re.IGNORECASE | re.DOTALL,
            )
            for cmd in shell_match:
                if not cls._is_safe_shell_command(cmd):
                    return False, f"Unsafe shell command: {cmd[:100]}"

        # Length limit
        if len(script) > 10000:
            return False, f"Script too long ({len(script)} chars, max 10000)"

        return True, "safe"

    @classmethod
    def _is_safe_shell_command(cls, cmd: str) -> bool:
        """Check if a shell command embedded in AppleScript is safe."""
        cmd_lower = cmd.strip().lower()
        # Whitelist of safe shell commands
        safe_prefixes = [
            "open ", "pbcopy", "pbpaste", "screencapture",
            "osascript", "defaults read", "echo ", "cat ",
            "ls ", "pwd", "whoami", "date", "sw_vers",
            "system_profiler", "pmset -g", "ioreg",
            "desktoppr",  # wallpaper tool
        ]
        return any(cmd_lower.startswith(prefix) for prefix in safe_prefixes)

    @classmethod
    def validate_shell_command(cls, command: str) -> tuple[bool, str]:
        """Validate a direct shell command for safety."""
        if not command or not command.strip():
            return False, "Empty command"

        cmd_lower = command.strip().lower()

        # Block destructive commands
        blocked = [
            "rm -rf /", "rm -rf ~", "rm -rf /*", "sudo rm",
            "mkfs", "dd if=", "format", "shutdown", "reboot",
            "killall", "launchctl unload", "> /dev/sd",
            "chmod -R 777 /", "chown -R",
        ]
        for pattern in blocked:
            if pattern in cmd_lower:
                return False, f"Blocked: {pattern}"

        return True, "safe"


# ---------------------------------------------------------------------------
# AppleScript Runner
# ---------------------------------------------------------------------------

class AppleScriptRunner:
    """Runs validated AppleScript with proper error handling and receipts."""

    @staticmethod
    async def run(  # noqa: ASYNC109 - public API accepts an explicit bounded timeout.
        script: str,
        timeout: float = 10.0,  # noqa: ASYNC109
        *,
        read_only: bool = False,
        source: str = "host_automation.applescript",
    ) -> AutomationReceipt:
        """Execute an AppleScript after AST validation."""
        start = time.time()

        # Validate
        is_safe, reason = ScriptASTGuard.validate_applescript(script)
        if not is_safe:
            return AutomationReceipt(
                action="execute_applescript",
                target=script[:200],
                adapter="applescript",
                success=False,
                error=f"Script blocked by ASTGuard: {reason}",
                duration_ms=(time.time() - start) * 1000,
                script_hash=hashlib.sha256(script.encode()).hexdigest()[:16],
            )

        # Execute
        try:
            from core.governance_context import local_internal_governed_scope
            with local_internal_governed_scope(source, domain="tool_execution"):
                completed = await get_subprocess_gateway().run_async(
                    ["osascript", "-e", script],
                    timeout=timeout,
                    read_only=read_only,
                    capture_output=True,
                    source=source,
                    accelerator_capability="none",
                )
            success = completed.returncode == 0
            result = str(completed.stdout or "").strip()
            error = str(completed.stderr or "").strip() if not success else ""

            return AutomationReceipt(
                action="execute_applescript",
                target=script[:200],
                adapter="applescript",
                success=success,
                result=result,
                error=error,
                duration_ms=(time.time() - start) * 1000,
                script_hash=hashlib.sha256(script.encode()).hexdigest()[:16],
            )
        except (TimeoutError, subprocess.TimeoutExpired):
            return AutomationReceipt(
                action="execute_applescript",
                target=script[:200],
                adapter="applescript",
                success=False,
                error=f"AppleScript timed out after {timeout}s",
                duration_ms=(time.time() - start) * 1000,
                script_hash=hashlib.sha256(script.encode()).hexdigest()[:16],
            )
        except (OSError, RuntimeError, ValueError) as e:
            return AutomationReceipt(
                action="execute_applescript",
                target=script[:200],
                adapter="applescript",
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
                script_hash=hashlib.sha256(script.encode()).hexdigest()[:16],
            )


# ---------------------------------------------------------------------------
# The Provider
# ---------------------------------------------------------------------------

#: Where macOS keeps applications. Bounded and explicit rather than a
#: filesystem walk: an app that is not in one of these is not something a
#: person means when they say "open X".
_APPLICATION_DIRECTORIES: tuple[str, ...] = (
    "/Applications",
    "/Applications/Utilities",
    "/System/Applications",
    "/System/Applications/Utilities",
    str(Path.home() / "Applications"),
)

_INSTALLED_APPS_CACHE: tuple[float, tuple[str, ...]] | None = None
_INSTALLED_APPS_TTL_S = 60.0


def _normalize_app_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def installed_application_names(*, refresh: bool = False) -> tuple[str, ...]:
    """Applications actually present on this machine.

    Cached briefly — an app list does not change between two steps of one
    task, and stat-ing five directories per launch attempt is waste.
    """
    global _INSTALLED_APPS_CACHE
    now = time.monotonic()
    if (
        not refresh
        and _INSTALLED_APPS_CACHE is not None
        and now - _INSTALLED_APPS_CACHE[0] < _INSTALLED_APPS_TTL_S
    ):
        return _INSTALLED_APPS_CACHE[1]

    found: list[str] = []
    for directory in _APPLICATION_DIRECTORIES:
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        for entry in entries:
            if entry.endswith(".app"):
                found.append(entry[: -len(".app")])
    names = tuple(sorted(set(found)))
    _INSTALLED_APPS_CACHE = (now, names)
    return names


@dataclass(frozen=True)
class AppNameResolution:
    """What the requested app name actually refers to, if anything."""

    requested: str
    resolved: str | None
    basis: str
    candidates: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.resolved)

    def failure_detail(self) -> str:
        """A refusal a person can act on.

        The live 2026-07-30 demo failed with "Unable to find application named
        'Note'" while Notes.app sat in /System/Applications. A name that does
        not resolve should say what IS there, not just that this is not.
        """
        if self.candidates:
            return (
                f"no application named {self.requested!r}; "
                f"closest installed: {', '.join(self.candidates[:5])}"
            )
        return f"no application named {self.requested!r} is installed"


def resolve_application_name(app_name: str) -> AppNameResolution:
    """Map what a person said onto an app that exists.

    "Note app" means Notes. "chrome" means Google Chrome. Passing the raw
    string to AppleScript makes the difference between doing the task and
    refusing it a matter of whether the person typed the exact bundle name,
    which is not a thing anyone knows.

    Resolution is tiered and only auto-selects when the tier yields exactly
    ONE candidate. Two plausible matches is an ambiguity to report, not a
    coin to flip — opening the wrong application is worse than asking.
    """
    requested = str(app_name or "").strip()
    if not requested:
        return AppNameResolution(requested, None, "empty")

    installed = installed_application_names()
    if not installed:
        # Cannot enumerate: let AppleScript try the literal name rather than
        # refusing on the strength of a directory listing that did not work.
        return AppNameResolution(requested, requested, "unverified_passthrough")

    # "the Notes app" is how people say it. Strip a TRAILING app/application
    # only, so "App Store" is untouched.
    spoken = re.sub(r"^\s*the\s+", "", requested, flags=re.IGNORECASE).strip()
    spoken = (
        re.sub(
            r"^(.*?)\s*(?:app|application)\s*$", r"\1", spoken, flags=re.IGNORECASE
        ).strip()
        or spoken
        or requested
    )

    wanted = _normalize_app_name(spoken)
    by_normal: dict[str, list[str]] = {}
    for name in installed:
        by_normal.setdefault(_normalize_app_name(name), []).append(name)

    if wanted in by_normal and len(by_normal[wanted]) == 1:
        return AppNameResolution(requested, by_normal[wanted][0], "exact")

    # "note" -> "notes", "notes" -> "note"
    for variant in (wanted + "s", wanted.rstrip("s")):
        if variant and variant != wanted and len(by_normal.get(variant, ())) == 1:
            return AppNameResolution(requested, by_normal[variant][0], "plural_form")

    # "chrome" -> "Google Chrome"; unique substring only.
    contains = [name for name in installed if wanted and wanted in _normalize_app_name(name)]
    if len(contains) == 1:
        return AppNameResolution(requested, contains[0], "substring")

    prefixed = [
        name for name in installed if wanted and _normalize_app_name(name).startswith(wanted)
    ]
    if len(prefixed) == 1:
        return AppNameResolution(requested, prefixed[0], "prefix")

    candidates = tuple(sorted(set(contains + prefixed)))
    return AppNameResolution(requested, None, "ambiguous" if candidates else "absent", candidates)



class HostAutomationProvider:
    """Generalized OS automation through governed primitives.

    Every method:
    1. Validates input through ScriptASTGuard
    2. Executes through the most reliable adapter
    3. Produces an AutomationReceipt
    4. Logs to LifeTrace

    This is how Aura manipulates the OS without hardcoded app-specific code.
    The LLM + TaskDecomposer decides WHAT to do; this layer executes HOW.
    """

    def __init__(self) -> None:
        self._receipts: list[AutomationReceipt] = []
        self._max_receipts = 500
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("host_automation", self, required=False)
        self._started = True
        logger.info("HostAutomationProvider ONLINE — generalized OS automation ready")

    def _log_receipt(self, receipt: AutomationReceipt) -> None:
        """Log receipt to internal buffer and LifeTrace."""
        self._receipts.append(receipt)
        if len(self._receipts) > self._max_receipts:
            self._receipts = self._receipts[-self._max_receipts:]

        # Log to LifeTrace
        try:
            from core.runtime.life_trace import get_life_trace
            get_life_trace().record(
                event_type="action_executed",
                origin="host_automation",
                action_taken={
                    "action": receipt.action,
                    "target": str(receipt.target)[:200],
                    "adapter": receipt.adapter,
                    "success": receipt.success,
                },
                result={
                    "result": str(receipt.result)[:500] if receipt.result else "",
                    "error": receipt.error[:200] if receipt.error else "",
                    "duration_ms": receipt.duration_ms,
                    "receipt_id": receipt.receipt_id,
                },
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("host_automation.life_trace", e)

    @staticmethod
    def _action_completed(result: dict[str, Any]) -> bool:
        return bool(
            result.get("ok")
            and result.get("effect_verified")
            and result.get("receipt_persisted")
            and result.get("post_action_receipt_id")
        )

    # ------------------------------------------------------------------
    # App control primitives
    # ------------------------------------------------------------------

    async def launch_app(self, app_name: str) -> AutomationReceipt:
        """Launch an application by name. Uses AppleScript 'activate'.

        The name is resolved against what is installed first (CP: the
        2026-07-30 demo refused "Note" while Notes.app was present), so a
        person can say what they mean instead of the exact bundle name.
        """
        resolution = await asyncio.to_thread(resolve_application_name, app_name)
        resolved_name = str(resolution.resolved or "").strip()
        if not resolution.ok or not resolved_name:
            receipt = AutomationReceipt(
                action="launch_app",
                target=app_name,
                adapter="applescript",
                success=False,
                error=resolution.failure_detail(),
            )
            self._log_receipt(receipt)
            return receipt
        if resolved_name != app_name:
            logger.info(
                "🖥️ Resolved requested app %r to %r (%s).",
                app_name,
                resolved_name,
                resolution.basis,
            )
        app_name = resolved_name
        script = f'tell application {_as_applescript_string(app_name)} to activate'
        receipt = await AppleScriptRunner.run(script, timeout=10.0)
        receipt.action = "launch_app"
        receipt.target = app_name

        # Verify it actually launched
        if receipt.success:
            await asyncio.sleep(0.5)
            frontmost = await self.get_frontmost_app()
            if frontmost.result and app_name.lower() in str(frontmost.result).lower():
                receipt.result = f"{app_name} is now frontmost"
            else:
                receipt.result = f"{app_name} launched (may not be frontmost yet)"

        self._log_receipt(receipt)
        return receipt

    async def focus_app(self, app_name: str) -> AutomationReceipt:
        """Bring an already-running application to front."""
        script = f'''
            tell application "System Events"
                set frontProcess to first process whose name is {_as_applescript_string(app_name)}
                set frontmost of frontProcess to true
            end tell
        '''
        receipt = await AppleScriptRunner.run(script, timeout=5.0)
        receipt.action = "focus_app"
        receipt.target = app_name
        self._log_receipt(receipt)
        return receipt

    async def get_frontmost_app(self) -> AutomationReceipt:
        """Get the name of the currently frontmost application."""
        script = 'tell application "System Events" to get name of first application process whose frontmost is true'
        receipt = await AppleScriptRunner.run(
            script,
            timeout=3.0,
            read_only=True,
            source="host_automation.frontmost_app",
        )
        receipt.action = "get_frontmost_app"
        receipt.target = ""
        # Don't log this one to LifeTrace (it's a read, not an action)
        return receipt

    async def get_window_title(self, app_name: str = "") -> AutomationReceipt:
        """Get the title of the frontmost window of an app (or the frontmost app)."""
        if app_name:
            script = f'tell application "System Events" to get name of front window of process {_as_applescript_string(app_name)}'
        else:
            script = '''
                tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    set winTitle to name of front window of process frontApp
                end tell
                return winTitle
            '''
        receipt = await AppleScriptRunner.run(
            script,
            timeout=3.0,
            read_only=True,
            source="host_automation.window_title",
        )
        receipt.action = "get_window_title"
        receipt.target = app_name
        return receipt

    async def get_frontmost_window_context(self) -> AutomationReceipt:
        """Read frontmost app and window title in one atomic OS observation.

        Reading them in separate AppleScript calls allows focus to change
        between the two results. That is particularly unsafe for privacy
        admission: a public app name paired with a newly private title can make
        the capture decision describe no window that ever existed.
        """
        script = '''
            tell application "System Events"
                set frontApp to name of first application process whose frontmost is true
                set winTitle to ""
                try
                    set winTitle to name of front window of process frontApp
                end try
            end tell
            return frontApp & "|" & winTitle
        '''
        receipt = await AppleScriptRunner.run(
            script,
            timeout=3.0,
            read_only=True,
            source="host_automation.frontmost_window_context",
        )
        receipt.action = "get_frontmost_window_context"
        receipt.target = "frontmost_window"
        if receipt.success and str(receipt.result or "").strip("|").strip():
            return receipt

        # System Events can be unavailable to the Python child even while the
        # signed resident Aura.app has the correct desktop identity. Use the
        # resident bridge's NSWorkspace + CoreGraphics observation before
        # declaring the foreground unknowable. This is metadata only; pixels
        # remain behind the independent capture-admission gate.
        try:
            from core.security.native_desktop_bridge import (
                invoke_native_desktop_bridge,
            )

            native = await asyncio.to_thread(
                invoke_native_desktop_bridge,
                "frontmost_window_context",
                read_only=True,
                timeout=1.0,
                allow_one_shot=False,
            )
        except (ImportError, OSError, RuntimeError, TimeoutError, TypeError, ValueError):
            native = {}
        if native.get("ok") and native.get("bridge_transport") == "resident_ipc":
            app = str(native.get("app") or "").strip()
            title = str(native.get("title") or "").strip()
            if app or title:
                return AutomationReceipt(
                    action="get_frontmost_window_context",
                    target="frontmost_window",
                    adapter="resident_native_bridge",
                    success=True,
                    result=f"{app}|{title}",
                    duration_ms=receipt.duration_ms,
                )
        admission = native.get("capture_admission")
        if (
            native.get("error") == "screen_capture_refused"
            and isinstance(admission, dict)
            and not bool(admission.get("allowed", True))
        ):
            # A private foreground is a successful privacy decision. Return no
            # metadata and let ambient perception treat the surface as absent.
            return AutomationReceipt(
                action="get_frontmost_window_context",
                target="frontmost_window",
                adapter="resident_native_bridge",
                success=True,
                result="",
                duration_ms=receipt.duration_ms,
            )
        return receipt

    @staticmethod
    def _main_screen_visible_frame() -> tuple[int, int, int, int]:
        """Return the primary usable display in System Events coordinates."""
        from AppKit import NSScreen

        screen = NSScreen.mainScreen()
        if screen is None:
            raise RuntimeError("macOS did not report a primary screen")
        full = screen.frame()
        visible = screen.visibleFrame()
        x = int(round(float(visible.origin.x)))
        y = int(
            round(
                float(full.origin.y + full.size.height)
                - float(visible.origin.y + visible.size.height)
            )
        )
        width = int(round(float(visible.size.width)))
        height = int(round(float(visible.size.height)))
        if width <= 0 or height <= 0:
            raise RuntimeError(f"invalid primary screen frame: {x},{y},{width},{height}")
        return x, y, width, height

    async def get_desktop_frame(self) -> AutomationReceipt:
        """Read the primary usable desktop frame without launching Finder."""
        start = time.time()
        try:
            frame = await asyncio.to_thread(self._main_screen_visible_frame)
            return AutomationReceipt(
                action="get_desktop_frame",
                target="primary_screen",
                adapter="appkit",
                success=True,
                result=frame,
                duration_ms=(time.time() - start) * 1000,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return AutomationReceipt(
                action="get_desktop_frame",
                target="primary_screen",
                adapter="appkit",
                success=False,
                error=str(exc),
                duration_ms=(time.time() - start) * 1000,
            )

    async def close_app(self, app_name: str) -> AutomationReceipt:
        """Quit an application gracefully."""
        script = f'tell application {_as_applescript_string(app_name)} to quit'
        receipt = await AppleScriptRunner.run(script, timeout=5.0)
        receipt.action = "close_app"
        receipt.target = app_name
        self._log_receipt(receipt)
        return receipt

    async def get_running_apps(self) -> AutomationReceipt:
        """List all running GUI applications."""
        script = 'tell application "System Events" to get name of every application process whose background only is false'
        receipt = await AppleScriptRunner.run(
            script,
            timeout=5.0,
            read_only=True,
            source="host_automation.running_apps",
        )
        receipt.action = "get_running_apps"
        if receipt.success and receipt.result:
            # Parse comma-separated list
            apps = [a.strip() for a in str(receipt.result).split(",") if a.strip()]
            receipt.result = apps
        return receipt

    # ------------------------------------------------------------------
    # UI interaction primitives
    # ------------------------------------------------------------------

    async def menu_select(self, app_name: str, menu_path: list[str]) -> AutomationReceipt:
        """Click a menu item by path. E.g., menu_path=["File", "Export as PDF..."]."""
        if not menu_path:
            return AutomationReceipt(
                action="menu_select", target=app_name,
                adapter="applescript", success=False, error="Empty menu path",
            )

        # Actually need to navigate the menu hierarchy
        if len(menu_path) == 1:
            script = f'''
                tell application "System Events"
                    tell process {_as_applescript_string(app_name)}
                        click menu item {_as_applescript_string(menu_path[0])} of menu bar 1
                    end tell
                end tell
            '''
        elif len(menu_path) == 2:
            script = f'''
                tell application "System Events"
                    tell process {_as_applescript_string(app_name)}
                        click menu item {_as_applescript_string(menu_path[1])} of menu 1 of menu bar item {_as_applescript_string(menu_path[0])} of menu bar 1
                    end tell
                end tell
            '''
        elif len(menu_path) == 3:
            script = f'''
                tell application "System Events"
                    tell process {_as_applescript_string(app_name)}
                        click menu item {_as_applescript_string(menu_path[2])} of menu 1 of menu item {_as_applescript_string(menu_path[1])} of menu 1 of menu bar item {_as_applescript_string(menu_path[0])} of menu bar 1
                    end tell
                end tell
            '''
        else:
            return AutomationReceipt(
                action="menu_select", target=f"{app_name}: {' > '.join(menu_path)}",
                adapter="applescript", success=False, error="Menu path too deep (max 3 levels)",
            )

        receipt = await AppleScriptRunner.run(script, timeout=5.0)
        receipt.action = "menu_select"
        receipt.target = f"{app_name}: {' > '.join(menu_path)}"
        self._log_receipt(receipt)
        return receipt

    async def type_text(
        self, text: str, use_clipboard: bool = True, *, expect_app: str = ""
    ) -> AutomationReceipt:
        """Type text into the currently focused application.

        For text longer than 50 chars, uses clipboard paste (faster, more reliable).
        For short text, uses keystroke (more natural).

        `expect_app` names the application the text is FOR, and the same focus
        guard applies as for hotkey. It matters more here: an arrow key sent to
        the wrong window is noise, while a sentence typed into the wrong window
        is content in someone's document, chat or terminal. "Currently focused"
        is a description of where it will land, never a check that it is the
        right place.
        """
        refusal = await self._refuse_if_not_frontmost(expect_app, "type_text")
        if refusal is not None:
            self._log_receipt(refusal)
            return refusal
        start = time.time()
        if use_clipboard and len(text) > 50:
            # Clipboard paste method — faster and more reliable
            try:
                # Save current clipboard
                save_proc = await get_subprocess_gateway().spawn_async(
                    ["pbpaste"],
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    read_only=True,
                    source="host_automation.clipboard_read",
                    accelerator_capability="none",
                )
                old_clipboard, _ = await asyncio.wait_for(save_proc.communicate(), timeout=2.0)

                # Set new clipboard content
                set_proc = await get_subprocess_gateway().spawn_async(
                    ["pbcopy"],
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    source="host_automation.clipboard_write",
                    accelerator_capability="none",
                )
                await asyncio.wait_for(
                    set_proc.communicate(input=text.encode("utf-8")),
                    timeout=2.0,
                )
                # From here the outbound text is on the clipboard and MUST be
                # cleared/restored on every exit path — including paste failure.
                clipboard_dirtied = True

                try:
                    # Paste
                    paste_script = '''
                        tell application "System Events"
                            keystroke "v" using command down
                        end tell
                    '''
                    receipt = await AppleScriptRunner.run(paste_script, timeout=3.0)
                    receipt.action = "type_text"
                    receipt.target = f"[clipboard paste, {len(text)} chars]"
                    receipt.adapter = "clipboard+applescript"
                    await asyncio.sleep(0.3)
                    self._log_receipt(receipt)
                    return receipt
                finally:
                    if clipboard_dirtied:
                        # Restore the prior clipboard — or CLEAR it when the
                        # user's clipboard was empty, so the outbound text never
                        # lingers for other apps / clipboard history.
                        try:
                            restore_proc = await get_subprocess_gateway().spawn_async(
                                ["pbcopy"],
                                stdin=asyncio.subprocess.PIPE,
                                source="host_automation.clipboard_restore",
                                accelerator_capability="none",
                            )
                            await asyncio.wait_for(
                                restore_proc.communicate(input=old_clipboard or b""),
                                timeout=2.0,
                            )
                        except (TimeoutError, OSError) as restore_exc:
                            record_degradation("host_automation", restore_exc)
                            logger.warning("Clipboard restore failed; outbound text may remain: %s", restore_exc)

            except (TimeoutError, OSError) as e:
                logger.debug("Clipboard paste failed, falling back to keystroke: %s", e)
                # Fall through to keystroke method

        # Keystroke method — for short text or when clipboard fails. Chunk the
        # RAW text; _as_applescript_string() below does the escaping (a manual
        # pre-escape here would double-escape backslashes/quotes).
        chunk_size = 200
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        success = True
        errors = []
        for chunk in chunks:
            script = f'''
                tell application "System Events"
                    keystroke {_as_applescript_string(chunk)}
                end tell
            '''
            result = await AppleScriptRunner.run(script, timeout=5.0)
            if not result.success:
                success = False
                errors.append(result.error)
                break
            await asyncio.sleep(0.05)  # Small delay between chunks

        receipt = AutomationReceipt(
            action="type_text",
            target=f"[keystroke, {len(text)} chars]",
            adapter="applescript",
            success=success,
            error="; ".join(errors) if errors else "",
            duration_ms=(time.time() - start) * 1000,
        )
        self._log_receipt(receipt)
        return receipt

    async def _refuse_if_not_frontmost(
        self, expect_app: str, action: str
    ) -> AutomationReceipt | None:
        """None when `expect_app` is in front, a refusal receipt when it is not.

        LIVE DEFECT, 2026-08-18. Playing 2048 in a browser: her own browser
        controller opened play2048.co, her own screen read found the board, and
        `hotkey("left")` returned success=True while nothing on the board moved.
        The frontmost application was Claude. The arrow key went there.

        A keystroke has no address. AppleScript delivers it to whatever is in
        front at that instant, so every keyboard-driven task is silently
        aimed at whichever window the person last touched — and the receipt
        says success, because the key WAS delivered. The loop then sees no
        change and concludes the task failed, which is the wrong lesson from
        the wrong evidence.

        Perception already knows what is in front. Actuation never asked. This
        is that question, asked at the moment of the keystroke rather than
        earlier, because focus can change between deciding and acting.

        Substring match, case-insensitive, because the frontmost reading is
        "Google Chrome|<page title>" and a caller means the application.
        """
        wanted = " ".join(str(expect_app or "").split()).lower()
        if not wanted:
            return None
        context = await self.get_frontmost_window_context()
        observed = str(getattr(context, "result", "") or "")
        # Keep the name as the OS spelled it for the message, and compare on a
        # folded copy. A refusal that says 'claude' when the app is called
        # Claude reads like a different program.
        app_as_named = observed.split("|", 1)[0].strip()
        app = app_as_named.lower()
        if not getattr(context, "success", False) or not app:
            # Unable to tell is not permission to fire blind: a keystroke aimed
            # at an unknown window is exactly what this exists to prevent.
            return AutomationReceipt(
                action=action, target=expect_app, adapter="focus_guard",
                success=False,
                error=(
                    "refused: could not read the frontmost window, so the "
                    "target application could not be confirmed"
                ),
            )
        if wanted in app or app in wanted:
            return None
        return AutomationReceipt(
            action=action, target=expect_app, adapter="focus_guard",
            success=False,
            error=(
                f"refused: {expect_app!r} is not frontmost ({app_as_named!r} is), "
                f"so the keystroke would have gone to the wrong application"
            ),
        )

    async def hotkey(self, *keys: str, expect_app: str = "") -> AutomationReceipt:
        """Press a keyboard shortcut. E.g., hotkey("command", "s").

        `expect_app` names the application the keystroke is FOR. Given, the
        press is refused unless that application is frontmost at the moment of
        sending — see _refuse_if_not_frontmost for the live defect this closes.
        """
        refusal = await self._refuse_if_not_frontmost(expect_app, "hotkey")
        if refusal is not None:
            self._log_receipt(refusal)
            return refusal
        modifiers = {
            "command": "command down",
            "cmd": "command down",
            "shift": "shift down",
            "option": "option down",
            "alt": "option down",
            "control": "control down",
            "ctrl": "control down",
        }

        key_parts = list(keys)
        if not key_parts:
            return AutomationReceipt(
                action="hotkey", target="", adapter="applescript",
                success=False, error="No keys specified",
            )

        # Separate modifiers from the main key
        mods = []
        main_key = ""
        for k in key_parts:
            k_lower = k.lower().strip()
            if k_lower in modifiers:
                mods.append(modifiers[k_lower])
            else:
                main_key = k_lower

        if not main_key:
            return AutomationReceipt(
                action="hotkey", target="+".join(keys),
                adapter="applescript", success=False, error="No main key specified",
            )

        mod_str = " using {" + ", ".join(mods) + "}" if mods else ""
        # Handle special keys
        special_keys = {
            "return": 'key code 36', "enter": 'key code 36',
            "tab": 'key code 48', "escape": 'key code 53', "esc": 'key code 53',
            "delete": 'key code 51', "backspace": 'key code 51',
            "space": 'key code 49',
            "up": 'key code 126', "down": 'key code 125',
            "left": 'key code 123', "right": 'key code 124',
        }

        if main_key in special_keys:
            script = f'''
                tell application "System Events"
                    {special_keys[main_key]}{mod_str}
                end tell
            '''
        else:
            script = f'''
                tell application "System Events"
                    keystroke "{main_key}"{mod_str}
                end tell
            '''

        receipt = await AppleScriptRunner.run(script, timeout=3.0)
        receipt.action = "hotkey"
        receipt.target = "+".join(keys)
        self._log_receipt(receipt)
        return receipt

    async def click_element(
        self,
        reference: str,
        *,
        app: str = "",
        button: str = "left",
    ) -> AutomationReceipt:
        """Click a NAMED control, or refuse — never a guessed coordinate.

        `click_at` takes two numbers and nothing checks that they mean
        anything. This reads the frontmost window's controls into an inventory
        (core/perception/element_inventory.py), resolves the reference against
        it, and clicks the centre of the element it actually found.

        Refusing is the feature. An agent that always produces a coordinate
        will click SOMETHING when it recognised nothing, and a wrong click is
        not a smaller version of the right one — it is a different action taken
        on the person's machine. The refusal names the candidates so the next
        turn can disambiguate instead of guessing again.
        """
        start = time.time()
        try:
            from core.perception.element_inventory import (
                build_inventory,
                resolve_action_target,
            )
        except ImportError as exc:
            return AutomationReceipt(
                action="click_element", target=str(reference), adapter="element_inventory",
                success=False, error=f"element inventory unavailable: {exc}",
                duration_ms=(time.time() - start) * 1000,
            )

        target_app = str(app or "").strip()
        if not target_app:
            try:
                from core.perception.frontmost_app import frontmost_app_name_fast

                target_app = str(frontmost_app_name_fast() or "").strip()
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                target_app = ""
        if not target_app:
            return AutomationReceipt(
                action="click_element", target=str(reference), adapter="element_inventory",
                success=False, error="no frontmost app to read controls from",
                duration_ms=(time.time() - start) * 1000,
            )

        inventory = build_inventory(target_app)
        resolution = resolve_action_target(inventory, reference)
        if not resolution.resolved or resolution.element is None:
            return AutomationReceipt(
                action="click_element", target=str(reference), adapter="element_inventory",
                success=False, error=resolution.reason,
                duration_ms=(time.time() - start) * 1000,
            )

        element = resolution.element
        centre_x, centre_y = element.centre
        receipt = await self.click_at(int(centre_x), int(centre_y), button)
        # The receipt names WHAT was clicked, not only where. A coordinate in a
        # log cannot be audited after the screen has moved on.
        receipt.action = "click_element"
        receipt.target = f"{element.element_id} ({element.function()})"
        return receipt

    async def click_at(self, x: int, y: int, button: str = "left") -> AutomationReceipt:
        """Click at screen coordinates using cliclick (fast) or PyAutoGUI (fallback)."""
        start = time.time()
        try:
            # Try cliclick first (faster, no Python dependency)
            click_type = "c" if button == "left" else "rc"
            proc = await get_subprocess_gateway().spawn_async(
                ["cliclick", f"{click_type}:{x},{y}"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="host_automation.click",
                accelerator_capability="none",
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            success = proc.returncode == 0
            receipt = AutomationReceipt(
                action="click", target=f"{x},{y}",
                adapter="cliclick", success=success,
                error=stderr.decode().strip() if stderr and not success else "",
                duration_ms=(time.time() - start) * 1000,
            )
        except (FileNotFoundError, OSError):
            # Fallback to PyAutoGUI
            try:
                import pyautogui
                if button == "right":
                    pyautogui.rightClick(x, y)
                else:
                    pyautogui.click(x, y)
                receipt = AutomationReceipt(
                    action="click", target=f"{x},{y}",
                    adapter="pyautogui", success=True,
                    duration_ms=(time.time() - start) * 1000,
                )
            except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
                receipt = AutomationReceipt(
                    action="click", target=f"{x},{y}",
                    adapter="pyautogui", success=False,
                    error=str(e),
                    duration_ms=(time.time() - start) * 1000,
                )
        except TimeoutError:
            receipt = AutomationReceipt(
                action="click", target=f"{x},{y}",
                adapter="cliclick", success=False,
                error="Click timed out",
                duration_ms=(time.time() - start) * 1000,
            )

        self._log_receipt(receipt)
        return receipt

    async def scroll(self, dx: int = 0, dy: int = 0) -> AutomationReceipt:
        """Scroll by delta amounts."""
        start = time.time()
        try:
            import pyautogui
            pyautogui.scroll(dy, _pause=False)
            if dx:
                pyautogui.hscroll(dx, _pause=False)
            receipt = AutomationReceipt(
                action="scroll", target=f"dx={dx},dy={dy}",
                adapter="pyautogui", success=True,
                duration_ms=(time.time() - start) * 1000,
            )
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
            receipt = AutomationReceipt(
                action="scroll", target=f"dx={dx},dy={dy}",
                adapter="pyautogui", success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
        self._log_receipt(receipt)
        return receipt

    # ------------------------------------------------------------------
    # Screen capture primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _retention_limit(name: str, default: int, minimum: int) -> int:
        try:
            return max(minimum, int(os.getenv(name, str(default)) or str(default)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _screenshot_retention_candidates(
        directory: Path,
        keep_path: Path | None,
    ) -> list[tuple[Path, os.stat_result, bool]]:
        candidates: list[tuple[Path, os.stat_result, bool]] = []
        resolved_keep = keep_path.resolve() if keep_path is not None else None
        try:
            for path in directory.iterdir():
                if path.suffix.lower() != ".png" or not path.is_file():
                    continue
                try:
                    candidates.append(
                        (
                            path,
                            path.stat(),
                            resolved_keep is not None and path.resolve() == resolved_keep,
                        )
                    )
                except OSError:
                    continue
        except OSError:
            return []
        candidates.sort(key=lambda item: item[1].st_mtime, reverse=True)
        return candidates

    @classmethod
    async def _enforce_screenshot_retention(
        cls,
        directory: Path,
        *,
        keep_path: Path | None = None,
    ) -> dict[str, int]:
        """Bound retained captures by age, count, and total bytes.

        The ephemeral directory gets its own, far tighter budget. A retained
        screenshot is something Aura is meant to still have; an ephemeral one
        is deleted immediately after its OCR, so ANY file at rest there is the
        residue of a failure. Sharing the 200-file retained budget meant 67
        orphans from a few hours of refused cleanups — 112MB of full-screen
        captures — sat inside the limit and were never reclaimed.
        """
        ephemeral = directory.name == "ephemeral"
        max_files = cls._retention_limit(
            "AURA_SCREENSHOT_RETENTION_MAX_FILES", 4 if ephemeral else 200, 1
        )
        if ephemeral:
            max_age_seconds = cls._retention_limit(
                "AURA_EPHEMERAL_SCREENSHOT_RETENTION_MAX_SECONDS", 300, 30
            )
        else:
            max_age_seconds = (
                cls._retention_limit("AURA_SCREENSHOT_RETENTION_MAX_DAYS", 14, 1)
                * 86400
            )
        max_bytes = cls._retention_limit(
            "AURA_SCREENSHOT_RETENTION_MAX_BYTES",
            (16 if ephemeral else 512) * 1024 * 1024,
            8 * 1024 * 1024,
        )
        cutoff = time.time() - max_age_seconds
        candidates = await asyncio.to_thread(
            cls._screenshot_retention_candidates,
            directory,
            keep_path,
        )

        kept = 0
        kept_bytes = 0
        deleted = 0
        bytes_deleted = 0
        for path, stat_result, is_keep in candidates:
            expired = stat_result.st_mtime < cutoff
            over_count = kept >= max_files
            over_bytes = kept_bytes + stat_result.st_size > max_bytes
            if not is_keep and (expired or over_count or over_bytes):
                try:
                    # Same missing-scope defect as the capture directory: with
                    # no declared scope the Will refused every deletion, so
                    # retention silently kept everything and the capture
                    # directory grew without bound.
                    with local_internal_governed_scope(
                        "host_automation.screenshot_retention",
                        domain=ActionDomain.FILE_WRITE.value,
                        constraints={"path": str(path), "op": "delete"},
                    ):
                        deletion = await ActionExecutor.execute(
                            domain=ActionDomain.FILE_WRITE,
                            action_name="host_automation.screenshot_retention_delete",
                            params={"path": str(path), "op": "delete"},
                            source="host_automation.screenshot_retention",
                        )
                    if cls._action_completed(deletion):
                        deleted += 1
                        bytes_deleted += stat_result.st_size
                        continue
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    record_degradation(
                        "host_automation.screenshot_retention",
                        exc,
                        action="retained a screenshot after governed retention cleanup failed",
                        severity="warning",
                    )
            kept += 1
            kept_bytes += stat_result.st_size
        return {"kept": kept, "deleted": deleted, "bytes_deleted": bytes_deleted}

    async def take_screenshot(
        self,
        save_path: str = "",
        region: tuple[int, int, int, int] | None = None,
        *,
        retain_capture: bool = True,
    ) -> AutomationReceipt:
        """Take a screenshot and optionally save to path.

        Args:
            save_path: Where to save (auto-generated if empty).
            region: Optional (x, y, w, h) to capture a region.

        Returns:
            Receipt with save_path as result.
        """
        start = time.time()
        from core.security.screen_capture_policy import (
            evaluate_screen_capture_admission_async,
        )

        admission = await evaluate_screen_capture_admission_async()
        if not admission.allowed:
            return AutomationReceipt(
                action="take_screenshot",
                target="",
                adapter="screen_capture_policy",
                success=False,
                error=admission.public_error,
                duration_ms=(time.time() - start) * 1000,
            )
        if not save_path:
            ts = time.strftime("%Y%m%d_%H%M%S")
            unique = f"{time.time_ns() % 1_000_000_000:09d}"
            folder = "screenshots" if retain_capture else "ephemeral"
            save_dir = state_root() / "data" / folder
            # Creating Aura's OWN capture directory under her state root is
            # internal maintenance, not a user-directed write to the host.
            # Without a declared scope the Will refused this on every ambient
            # perception tick and take_screenshot failed before it ever
            # reached screencapture — screen perception was dead behind a
            # warning card. The scope is narrow on purpose: one domain, one
            # path, released as soon as the directory exists.
            with local_internal_governed_scope(
                "host_automation.screenshot_directory",
                domain=ActionDomain.FILE_WRITE.value,
                constraints={"path": str(save_dir), "op": "ensure_directory"},
            ):
                directory_result = await ActionExecutor.execute(
                    domain=ActionDomain.FILE_WRITE,
                    action_name="host_automation.ensure_screenshot_directory",
                    params={"path": str(save_dir), "op": "ensure_directory"},
                    source="host_automation.screenshot_directory",
                )
            if not self._action_completed(directory_result):
                return AutomationReceipt(
                    action="take_screenshot",
                    target=str(save_dir),
                    adapter="screencapture",
                    success=False,
                    error=(
                        "Screenshot directory could not be created through the governed "
                        "file transaction lane."
                    ),
                    duration_ms=(time.time() - start) * 1000,
                )
            save_path = str(save_dir / f"screenshot_{ts}_{unique}.png")
        else:
            # A caller-supplied path must resolve inside an allowed screenshot
            # root and be an image file — otherwise screencapture would write an
            # arbitrary host path (symlink/traversal) with no boundary.
            resolved, allowed_roots = await asyncio.to_thread(
                _resolve_screenshot_path_policy, save_path
            )
            in_root = resolved is not None and any(
                resolved == r or str(resolved).startswith(str(r) + os.sep) for r in allowed_roots
            )
            # `in_root` already requires resolved is not None, but that is a
            # correlation mypy cannot follow — and a reader cannot either.
            # Test the thing directly rather than relying on a prior clause.
            if (
                resolved is None
                or not in_root
                or resolved.suffix.lower() not in {".png", ".jpg", ".jpeg"}
            ):
                return AutomationReceipt(
                    action="take_screenshot",
                    target=str(save_path),
                    adapter="screencapture",
                    success=False,
                    error="Screenshot save_path is outside the allowed roots or not an image file.",
                    duration_ms=(time.time() - start) * 1000,
                )
            save_path = str(resolved)

        try:
            capture_started_at = time.time()
            cmd = ["screencapture", "-x"]  # -x = no sound
            if region:
                x, y, w, h = region
                cmd.extend(["-R", f"{x},{y},{w},{h}"])
            cmd.append(save_path)

            proc = await get_subprocess_gateway().spawn_async(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="host_automation.screenshot",
                accelerator_capability="auto",
            )
            await asyncio.wait_for(proc.communicate(), timeout=5.0)

            def _fresh_capture() -> bool:
                p = Path(save_path)
                if not p.exists():
                    return False
                st = p.stat()
                # A pre-existing file satisfies mere existence — require a
                # nonempty file written at/after this capture began.
                return st.st_size > 0 and st.st_mtime >= (capture_started_at - 1.0)

            fresh = await asyncio.to_thread(_fresh_capture)
            success = proc.returncode == 0 and fresh

            receipt = AutomationReceipt(
                action="take_screenshot", target=save_path,
                adapter="screencapture", success=success,
                result=save_path if success else "",
                duration_ms=(time.time() - start) * 1000,
            )
        except (TimeoutError, OSError) as e:
            receipt = AutomationReceipt(
                action="take_screenshot", target=save_path,
                adapter="screencapture", success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

        self._log_receipt(receipt)
        # Retention runs for the ephemeral directory too.
        #
        # It used to be gated on retain_capture, so the ONLY thing bounding
        # ~/.aura/data/ephemeral was the per-call cleanup in get_screen_text —
        # a single point of failure guarding the directory with by far the
        # highest churn, roughly one full-screen capture every 7 seconds. When
        # that cleanup was refused for hours on 2026-08-10 nothing noticed and
        # nothing pruned. A backstop is the difference between a bug that
        # leaves stale files and one that fills a disk with pictures of the
        # person's screen.
        if receipt.success:
            try:
                retention = await self._enforce_screenshot_retention(
                    Path(save_path).parent,
                    keep_path=Path(save_path),
                )
                if retention["deleted"]:
                    logger.info(
                        "Screenshot retention removed %d captures (%d bytes)",
                        retention["deleted"],
                        retention["bytes_deleted"],
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "host_automation.screenshot_retention",
                    exc,
                    action="returned screenshot after retention enforcement failed",
                    severity="warning",
                )
        return receipt

    @staticmethod
    def _ocr_image_regions(image_path: str) -> list[dict[str, Any]]:
        """Recognized text WITH the position of each run.

        macOS Vision already computes this. Every VNRecognizedTextObservation
        carries a boundingBox, and _ocr_image_text read `.string()` off the top
        candidate and discarded the geometry, returning "\n".join(lines).

        So the flat text a caller received was not what the OS produced — it
        was what survived. Anything laid out in two dimensions became a column
        of strings in reading order: a table lost its columns, a form lost
        which label went with which field, and a grid lost the grid. Tasks
        that need to know WHERE something is were unreachable, and the reason
        was invisible because OCR appeared to work.

        Boxes are normalized 0..1 with a TOP-left origin, matching how screen
        coordinates are expressed everywhere else in this codebase. Vision's
        own origin is bottom-left, and leaving that mismatch for each caller to
        remember is how a click lands at the wrong end of the screen.

        Returns [] rather than raising: a caller that wants flat text still has
        _ocr_image_text, and losing layout is not a reason to lose the words.
        """
        try:
            from Foundation import NSURL
            from Quartz import (
                CGImageSourceCreateImageAtIndex,
                CGImageSourceCreateWithURL,
            )
            from Vision import (
                VNImageRequestHandler,
                VNRecognizeTextRequest,
                VNRequestTextRecognitionLevelAccurate,
            )

            image_url = NSURL.fileURLWithPath_(str(image_path))
            image_source = CGImageSourceCreateWithURL(image_url, None)
            if image_source is None:
                return []
            image = CGImageSourceCreateImageAtIndex(image_source, 0, None)
            if image is None:
                return []
            request = VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(True)
            handler = VNImageRequestHandler.alloc().initWithCGImage_options_(image, {})
            succeeded, _error = handler.performRequests_error_([request], None)
            if not succeeded:
                return []
            regions: list[dict[str, Any]] = []
            for observation in list(request.results() or []):
                candidates = list(observation.topCandidates_(1) or [])
                if not candidates:
                    continue
                value = str(candidates[0].string() or "").strip()
                if not value:
                    continue
                try:
                    box = observation.boundingBox()
                    x = float(box.origin.x)
                    y = float(box.origin.y)
                    w = float(box.size.width)
                    h = float(box.size.height)
                    confidence = float(candidates[0].confidence())
                except (AttributeError, TypeError, ValueError):
                    continue
                regions.append(
                    {
                        "text": value,
                        # Vision measures up from the bottom; everything else
                        # here measures down from the top.
                        "x": round(x, 5),
                        "y": round(1.0 - (y + h), 5),
                        "width": round(w, 5),
                        "height": round(h, 5),
                        "center_x": round(x + (w / 2.0), 5),
                        "center_y": round(1.0 - (y + (h / 2.0)), 5),
                        "confidence": round(confidence, 4),
                    }
                )
            return regions
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return []

    @staticmethod
    def _ocr_image_text(image_path: str) -> str:
        """Recognize text with native macOS Vision, then optional Tesseract."""
        native_error = ""
        try:
            from Foundation import NSURL
            from Quartz import (
                CGImageSourceCreateImageAtIndex,
                CGImageSourceCreateWithURL,
            )
            from Vision import (
                VNImageRequestHandler,
                VNRecognizeTextRequest,
                VNRequestTextRecognitionLevelAccurate,
            )

            image_url = NSURL.fileURLWithPath_(str(image_path))
            image_source = CGImageSourceCreateWithURL(image_url, None)
            if image_source is None:
                raise ValueError("Vision could not open the screenshot image source")
            image = CGImageSourceCreateImageAtIndex(image_source, 0, None)
            if image is None:
                raise ValueError("Vision could not decode the screenshot image")
            request = VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(True)
            handler = VNImageRequestHandler.alloc().initWithCGImage_options_(image, {})
            succeeded, error = handler.performRequests_error_([request], None)
            if not succeeded:
                raise RuntimeError(f"Vision OCR request failed: {error}")
            lines: list[str] = []
            for observation in list(request.results() or []):
                candidates = list(observation.topCandidates_(1) or [])
                if not candidates:
                    continue
                value = str(candidates[0].string() or "").strip()
                if value:
                    lines.append(value)
            if lines:
                return "\n".join(lines)
            native_error = "Vision returned no recognized text"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            native_error = f"{type(exc).__name__}:{exc}"

        try:
            import pytesseract
            from PIL import Image

            return str(pytesseract.image_to_string(Image.open(str(image_path))) or "").strip()
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            fallback_error = f"{type(exc).__name__}:{exc}"
            raise RuntimeError(
                f"OCR unavailable (native={native_error or 'unavailable'}; "
                f"fallback={fallback_error})"
            ) from exc

    async def get_screen_text(
        self,
        region: tuple[int, int, int, int] | None = None,
        *,
        retain_screenshot: bool = True,
    ) -> AutomationReceipt:
        """Take a screenshot and extract text via OCR.

        Verification callers set ``retain_screenshot=False`` so repeated
        desktop actions do not accumulate private screen captures indefinitely.
        """
        start = time.time()
        # Take screenshot first
        ss = await self.take_screenshot(
            region=region,
            retain_capture=retain_screenshot,
        )
        if not ss.success or not ss.result:
            return AutomationReceipt(
                action="get_screen_text", target="",
                adapter="ocr", success=False,
                error=f"Screenshot failed: {ss.error}",
                duration_ms=(time.time() - start) * 1000,
            )

        text = ""
        ocr_error = ""
        regions: list[dict[str, Any]] = []
        try:
            text = await asyncio.to_thread(self._ocr_image_text, str(ss.result))
            # Read the layout in the SAME pass. Vision computes the position of
            # every text run whether or not anyone asks, so taking it here
            # costs one more traversal of a result set already in memory —
            # against a second full screenshot and a second OCR, which is
            # the difference between a loop that can watch something
            # change and one that cannot.
            regions = await asyncio.to_thread(self._ocr_image_regions, str(ss.result))
        except _HOST_AUTOMATION_ERRORS as e:
            ocr_error = str(e)

        if not retain_screenshot:
            try:
                # THIRD instance of the missing-scope defect, after the capture
                # directory and screenshot retention. Both of those were fixed
                # by declaring the scope; this path was not, so every ephemeral
                # capture survived its own deletion.
                #
                # LIVE, 2026-08-10, immediately after the capture fix went live:
                # ~/.aura/data/ephemeral grew to 10 files / 15MB in the first
                # minutes, one roughly every 7 seconds — about 770MB an hour of
                # full-screen captures that nothing could ever remove. The
                # refusal reads "permission_model_blocked: Modality
                # 'file_delete' is disabled", and that rule is correct: Aura
                # may not delete a person's files, and file_delete is
                # deliberately ungrantable. This file is one she created
                # herself, seconds ago, under her own state root,
                # explicitly as ephemeral — and leaving it there is the privacy
                # harm the rule exists to prevent, not a way of avoiding one.
                with local_internal_governed_scope(
                    "host_automation.ephemeral_ocr_cleanup",
                    domain=ActionDomain.FILE_WRITE.value,
                    constraints={"path": str(ss.result), "op": "delete"},
                ):
                    cleanup = await ActionExecutor.execute(
                        domain=ActionDomain.FILE_WRITE,
                        action_name="host_automation.ephemeral_ocr_cleanup",
                        params={"path": str(ss.result), "op": "delete"},
                        source="host_automation.ephemeral_ocr_cleanup",
                    )
                if not self._action_completed(cleanup):
                    raise RuntimeError(
                        str(cleanup.get("error") or "ephemeral OCR cleanup was not verified")
                    )
            except _HOST_AUTOMATION_ERRORS as exc:
                record_degradation(
                    "host_automation.ocr_cleanup",
                    exc,
                    action="retained an ephemeral OCR screenshot after governed cleanup failed",
                    severity="warning",
                )
                logger.warning("Ephemeral OCR screenshot cleanup failed: %s", exc)

        receipt = AutomationReceipt(
            action="get_screen_text",
            target=str(ss.result) if retain_screenshot else "ephemeral_verification_capture",
            adapter="ocr", success=bool(text),
            result=text[:2000],
            error=ocr_error[:500],
            duration_ms=(time.time() - start) * 1000,
            layout=regions,
        )
        return receipt

    # ------------------------------------------------------------------
    # AppleScript execution (the general-purpose primitive)
    # ------------------------------------------------------------------

    async def execute_applescript(self, script: str) -> AutomationReceipt:
        """Execute arbitrary AppleScript after safety validation.

        This is the general-purpose primitive. The LLM/TaskDecomposer can
        compile any macOS automation into AppleScript, and this method
        executes it safely.

        The script passes through ScriptASTGuard before execution.
        """
        receipt = await AppleScriptRunner.run(script, timeout=15.0)
        self._log_receipt(receipt)
        return receipt

    async def inspect_applescript(
        self,
        script: str,
        *,
        timeout_s: float = 5.0,
        source: str = "host_automation.applescript_inspection",
    ) -> AutomationReceipt:
        """Run a read-only AppleScript probe without recording an action event.

        Desktop verification needs current UI state, but those observations are
        not host mutations and must not be mislabeled as executed actions. The
        subprocess gateway receives ``read_only=True`` so policy can enforce the
        same distinction below this provider boundary.
        """
        receipt = await AppleScriptRunner.run(
            script,
            timeout=max(0.5, min(float(timeout_s), 15.0)),
            read_only=True,
            source=source,
        )
        receipt.action = "inspect_applescript"
        return receipt

    # ------------------------------------------------------------------
    # Shell command execution (governed)
    # ------------------------------------------------------------------

    async def run_command(  # noqa: ASYNC109 - public API accepts an explicit bounded timeout.
        self,
        command: str,
        timeout: float = 15.0,  # noqa: ASYNC109
    ) -> AutomationReceipt:
        """Run a shell command after authorization and safety validation.

        The section header above said "(governed)" while the only check was
        `ScriptASTGuard` — a syntactic guard, not a governance decision. This
        method is reachable from `mission_state` with a plan-supplied string,
        so it is Aura executing an arbitrary command of her own devising, and
        it must ask the Will exactly like the terminal skill does.
        """
        start = time.time()

        verdict = await authorize_execution(
            KIND_SHELL,
            command,
            source="tool_execution:host_automation.shell_command",
            extra={"timeout_s": float(timeout)},
        )
        if not verdict.approved:
            receipt = AutomationReceipt(
                action="run_command", target=str(command)[:200],
                adapter="shell", success=False,
                error=verdict.reason,
                duration_ms=(time.time() - start) * 1000,
            )
            self._log_receipt(receipt)
            return receipt

        is_safe, reason = ScriptASTGuard.validate_shell_command(command)
        if not is_safe:
            release_execution(
                verdict,
                source="host_automation.shell_command",
                success=False,
                error=reason,
            )
            receipt = AutomationReceipt(
                action="run_command", target=command[:200],
                adapter="shell", success=False,
                error=f"Command blocked: {reason}",
                duration_ms=(time.time() - start) * 1000,
            )
            self._log_receipt(receipt)
            return receipt

        try:
            proc = await get_subprocess_gateway().spawn_shell_async(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path.home()),
                source="host_automation.shell_command",
                accelerator_capability="auto",
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            success = proc.returncode == 0
            receipt = AutomationReceipt(
                action="run_command", target=command[:200],
                adapter="shell", success=success,
                result=stdout.decode("utf-8", errors="replace").strip()[:2000] if stdout else "",
                error=stderr.decode("utf-8", errors="replace").strip()[:500] if stderr and not success else "",
                duration_ms=(time.time() - start) * 1000,
            )
        except TimeoutError:
            receipt = AutomationReceipt(
                action="run_command", target=command[:200],
                adapter="shell", success=False,
                error=f"Command timed out after {timeout}s",
                duration_ms=(time.time() - start) * 1000,
            )
        except OSError as e:
            receipt = AutomationReceipt(
                action="run_command", target=command[:200],
                adapter="shell", success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

        release_execution(
            verdict,
            source="host_automation.shell_command",
            success=bool(receipt.success),
            error=str(receipt.error or ""),
        )
        self._log_receipt(receipt)
        return receipt

    # ------------------------------------------------------------------
    # Wait / condition primitives
    # ------------------------------------------------------------------

    async def wait_for_condition(  # noqa: ASYNC109 - bounded polling API.
        self,
        predicate_name: str,
        predicate_args: dict[str, Any],
        timeout: float = 10.0,  # noqa: ASYNC109
        poll_interval: float = 0.5,
    ) -> AutomationReceipt:
        """Wait until a condition is true or timeout.

        Supported predicates:
            app_is_frontmost(name) — check if app is the frontmost
            file_exists(path) — check if file exists
            window_title_contains(text) — check window title
        """
        start = time.time()
        while (time.time() - start) < timeout:
            try:
                result = await self._check_predicate(predicate_name, predicate_args)
                if result:
                    return AutomationReceipt(
                        action="wait_for_condition",
                        target=f"{predicate_name}({predicate_args})",
                        adapter="poll", success=True,
                        result=f"Condition met after {(time.time()-start)*1000:.0f}ms",
                        duration_ms=(time.time() - start) * 1000,
                    )
            except (OSError, RuntimeError) as e:
                logger.debug("Predicate check failed: %s", e)
            await asyncio.sleep(poll_interval)

        return AutomationReceipt(
            action="wait_for_condition",
            target=f"{predicate_name}({predicate_args})",
            adapter="poll", success=False,
            error=f"Condition not met within {timeout}s",
            duration_ms=(time.time() - start) * 1000,
        )

    async def _check_predicate(self, name: str, args: dict[str, Any]) -> bool:
        """Evaluate a single predicate."""
        if name == "app_is_frontmost":
            receipt = await self.get_frontmost_app()
            return bool(
                receipt.success and receipt.result
                and str(args.get("name", "")).lower() in str(receipt.result).lower()
            )
        elif name == "file_exists":
            return await asyncio.to_thread(Path(str(args.get("path", ""))).exists)
        elif name == "window_title_contains":
            receipt = await self.get_window_title(args.get("app", ""))
            return bool(
                receipt.success and receipt.result
                and str(args.get("text", "")).lower() in str(receipt.result).lower()
            )
        return False

    # ------------------------------------------------------------------
    # Status / Audit
    # ------------------------------------------------------------------

    def get_recent_receipts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent automation receipts for audit."""
        return [
            {
                "action": r.action,
                "target": r.target[:100],
                "adapter": r.adapter,
                "success": r.success,
                "error": r.error[:100] if r.error else "",
                "duration_ms": round(r.duration_ms, 1),
                "receipt_id": r.receipt_id,
                "timestamp": r.timestamp,
            }
            for r in self._receipts[-limit:]
        ]

    def get_status(self) -> dict[str, Any]:
        """Provider status for dashboards."""
        total = len(self._receipts)
        successes = sum(1 for r in self._receipts if r.success)
        return {
            "started": self._started,
            "total_actions": total,
            "success_rate": round(successes / max(1, total), 3),
            "recent_actions": self.get_recent_receipts(5),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: HostAutomationProvider | None = None


def get_host_automation() -> HostAutomationProvider:
    global _instance
    if _instance is None:
        _instance = HostAutomationProvider()
    return _instance


__all__ = [
    "HostAutomationProvider",
    "AutomationReceipt",
    "ScriptASTGuard",
    "AppleScriptRunner",
    "get_host_automation",
]
