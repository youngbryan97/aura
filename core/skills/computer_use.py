import asyncio
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.being.body_state_service import BodyStateService
from core.being.welfare_state import WelfareState
from core.being.welfare_transaction import WelfareTransaction
from core.runtime.app_target_resolution import resolve_installed_app_target
from core.runtime.atomic_writer import atomic_write_bytes, atomic_write_text
from core.runtime.content_integrity import paragraph_sha256s, text_sha256
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.host_clock import read_host_clock_text
from core.runtime.os_automation_effects import canonical_app_target
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.skills._pyautogui_runtime import get_pyautogui
from core.skills.base_skill import BaseSkill
from core.utils.exceptions import capture_and_log

logger = logging.getLogger("Skills.ComputerUse")

def _quartz_error_types() -> tuple[type[BaseException], ...]:
    """PyObjC bridges ObjC failures as objc.error; resolve it lazily so
    non-darwin platforms never import the bridge."""
    errors: list[type[BaseException]] = [
        AttributeError, OSError, RuntimeError, TypeError, ValueError,
    ]
    try:
        import objc

        errors.append(objc.error)
    except ImportError:
        pass
    return tuple(errors)


_QUARTZ_RENDER_ERRORS = _quartz_error_types()

# Browsers open_url may target explicitly ("open -a <browser> <url>") —
# bounded so a derived step can never launch an arbitrary application.
_ALLOWED_URL_BROWSERS = {
    "Google Chrome",
    "Safari",
    "Firefox",
    "Microsoft Edge",
    "Arc",
}


_COMPUTER_USE_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
    subprocess.SubprocessError,
)


def _record_computer_use_degradation(
    error: BaseException,
    *,
    action: str,
    stage: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    metadata["stage"] = stage
    try:
        record_degradation(
            "computer_use",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata,
        )
    except TypeError:
        record_degradation(
            "computer_use",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
        )


class ComputerUseParams(BaseModel):
    action: str = Field(
        ...,
        description=(
            "click|type|hotkey|scroll|inspect_screen|read_screen_text|read_menu_clock|open_app|open_url|"
            "dismiss_popup|inspect_browser_page|"
            "run_command|set_clipboard|get_clipboard|wait|run_applescript|write_text_file|"
            "render_text_pdf|move_file|create_folder|list_directory|fetch_topic_image|system_control|"
            "move_aura_bubble|pursue_on_screen"
        ),
    )
    target: str = Field(
        "", description="Element description, text to type, key combo, command, app name, or URL"
    )
    x: int = Field(0, description="Screen x coordinate for click/scroll")
    y: int = Field(0, description="Screen y coordinate for click/scroll")


def _image_suffix_from_bytes(raw: bytes) -> str:
    """The real image type, read from the file's own first bytes."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) < 12:
        return ""
    head = bytes(raw[:12])
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return ".gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    if head[4:8] == b"ftyp" and b"avif" in bytes(raw[:32]):
        return ".avif"
    return ""


def _screen_text_unavailable(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return True
    return normalized in {
        "[accessibility error or ui unresponsive]",
        "[read_screen_text failed]",
    }


def _screen_text_unavailable_is_accessibility(text: str) -> bool:
    """Whether the read failed on accessibility rather than on content.

    The AppleScript window tree is queried through System Events, which
    needs the same Accessibility grant the text read just failed on. When
    that is the reason, the fallback cannot succeed and running it only
    costs the caller another eight-second timeout.
    """
    lowered = str(text or "").lower()
    return "accessibility" in lowered or "ui unresponsive" in lowered


def _verify_the_effect_landed(
    *,
    browser_surface: Any,
    context: Any,
    effect_verified: Any,
    is_paste: Any,
    screen_verifiable: Any,
    self: Any,
) -> tuple[Any, Any, Any]:
    """Check that the action's effect actually landed on the host.

    Moved out of ``ComputerUseSkill._execute_action`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 6 name(s) from the turn and hands back
    3.
    """
    if effect_verified:
        ok = True
        verification = (
            "Focused element changed."
            if browser_surface
            else "State shifted."
        )
    elif not screen_verifiable:
        if browser_surface:
            inherited_editor_focus = bool(
                context.get("desktop_task_editor_focus_verified")
            ) and self._is_resolved_web_editor_url(
                str(context.get("desktop_task_verified_editor_url") or "")
            )
            if (
                is_paste
                and bool(context.get("desktop_task_requires_editable_focus"))
                and inherited_editor_focus
            ):
                ok = True
                effect_verified = True
                verification = (
                    "Paste dispatched into a previously verified browser editor; "
                    "focused-control read-back was unavailable after dispatch."
                )
            else:
                ok = False
                effect_verified = False
                verification = (
                    "Hotkey dispatched, but browser focused-control verification "
                    "was unavailable; refusing to count the shortcut as a document edit."
                )
        else:
            # Native apps such as Notes often do not expose text
            # through the screen-reader path. For those surfaces, a
            # clean System Events dispatch plus a verified foreground
            # app is the bounded effect evidence.
            ok = True
            effect_verified = True
            verification = (
                "Keystroke dispatched and accepted by the OS; on-screen "
                "read-back was unavailable, so the effect is inferred from "
                "the clean dispatch."
            )
    else:
        ok = False
        verification = (
            "Hotkey dispatched but no visible state shift was verified."
        )
    return effect_verified, ok, verification


class ComputerUseSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "computer_use"
    description = (
        "Directly control the computer and Aura's own desktop surface: click, type, "
        "read screen text, run commands, open apps, or move her companion bubble."
    )
    input_model = ComputerUseParams
    metabolic_cost = 2

    # ═══════════════════════════════════════════════════════════════════
    # THE STEP BUDGET MUST COVER THE BUDGETS THE STEP ITSELF ENFORCES
    #
    # This skill declared nothing, so it inherited BaseSkill's 30s. But
    # write_in_app's own enforced sub-budgets sum past that before it starts:
    # hold_focus 12s, open a document, hold_focus again 12s, type for up to
    # 25s, then finish through the app's dictionary. It could not win, and the
    # way it lost was the worst kind — live 2026-07-29 "open Notes and write a
    # note" TYPED THE WHOLE NOTE, was cancelled at 30s, retried, typed it
    # AGAIN, and reported "write_in_app failed: Operation took too long" after
    # 61.2s. Bryan got two notes and an error, having watched it work twice.
    #
    # So the budget is summed from the sub-budgets rather than guessed: if
    # someone widens the typing window, this follows. A skill whose declared
    # timeout is shorter than the clocks inside it is not configured, it is
    # broken.
    # ═══════════════════════════════════════════════════════════════════

    #: The floor: ordinary steps (a click, a hotkey, a folder) with room to retry.
    timeout_seconds = 30.0

    #: Actions whose cost is dominated by driving another application's UI.
    _WRITE_ACTIONS = frozenset({"write_in_app", "create_note"})

    #: Headroom over the summed sub-budgets. Without it the worst case lands
    #: EXACTLY on the ceiling (12+12+2+25+50 = 101s against a 101s budget), so
    #: any real-world jitter — a slow osascript, a busy app — puts the step
    #: back over the line and we are debugging the same failure again. The
    #: skill's own clocks are what should stop this work, not the outer budget.
    _WRITE_BUDGET_GRACE_S = 30.0

    @classmethod
    def _write_in_app_budget_s(cls) -> float:
        """What the write path can legitimately take, from its own clocks."""
        return (
            cls._HOLD_FOCUS_BUDGET_S * 2   # front before the new doc, and before typing
            + 2.0                          # opening the document and letting it settle
            + cls._TYPING_BUDGET_S         # the visible typing window
            + cls._DICTIONARY_WRITE_BUDGET_S  # finishing through the app's own model
            + cls._WRITE_BUDGET_GRACE_S
        )

    @classmethod
    def timeout_for(cls, params: Any) -> float:
        """How long THIS action needs, not how long the average one takes."""
        payload = params if isinstance(params, dict) else None
        if payload is None:
            action = str(getattr(params, "action", "") or "")
        else:
            action = str(payload.get("action") or "")
        action = action.strip().lower()
        if action in cls._WRITE_ACTIONS:
            return max(cls.timeout_seconds, cls._write_in_app_budget_s())
        if action in {"open_app", "fetch_topic_image", "render_text_pdf"}:
            # Launching or fetching waits on something outside this process;
            # hold_focus alone can spend the whole declared budget.
            return max(cls.timeout_seconds, cls._HOLD_FOCUS_BUDGET_S * 2 + 30.0)
        if action == "pursue_on_screen":
            # A watching action is bounded by the goal it is watching for, not
            # by how long one keystroke takes. The pursuit already carries its
            # own limit and stops itself; wrapping it in the ordinary
            # thirty-second budget cut every run off mid-game and reported
            # "Operation took too long" for a loop that was working.
            return max(cls.timeout_seconds, cls._pursuit_budget_s(params) + cls._PURSUIT_GRACE_S)
        return cls.timeout_seconds

    #: Room for the pursuit to finish its last cycle and report, after its own
    #: clock has run out. A watching action that is killed on its own deadline
    #: loses the receipt saying what it did.
    _PURSUIT_GRACE_S = 30.0

    @classmethod
    def _pursuit_budget_s(cls, params: Any) -> float:
        """The limit a pursuit gave itself, read from its own target."""
        payload = params if isinstance(params, dict) else None
        target = payload.get("target") if payload else getattr(params, "target", "")
        from core.runtime.watched_goal import PURSUIT_SECONDS  # noqa: PLC0415

        try:
            declared = cls._target_json(str(target or "{}")).get("max_seconds")
            return max(0.0, float(declared))
        except (TypeError, ValueError, AttributeError):
            return PURSUIT_SECONDS

    PERMISSION_CHECK_TIMEOUT_S = 3.0
    MAX_APPLESCRIPT_CHARS = 4000
    APPLESCRIPT_DENYLIST = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bdo\s+shell\s+script\b",
            r"\bsudo\b",
            r"\brm\s+-",
            r"\bchmod\b",
            r"\bchown\b",
            r"\bempty\s+trash\b",
            r"\bmove\b.+\btrash\b",
            r"\bdelete\b.+\b(file|folder|note|message|account)\b",
            r"\berase\b",
        )
    )

    # SK-01: Restricted command set for autonomous use
    ALLOWED_COMMANDS = frozenset(
        [
            "ls",
            "pwd",
            "echo",
            "cat",
            "find",
            "grep",
            "python3",
            "pip",
            "git",
            "mkdir",
            "touch",
            "tree",
        ]
    )

    async def _require_permissions(
        self,
        capability: str,
        *permission_names: str,
    ) -> dict[str, Any] | None:
        try:
            from core.container import ServiceContainer
            from core.security.permission_guard import PermissionType
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_computer_use_degradation(
                exc,
                action="blocked desktop capability because permission subsystem import failed closed",
                stage="permissions.import",
                severity="degraded",
                extra={"capability": capability},
            )
            return {
                "ok": False,
                "status": "unavailable",
                "error": f"Permission subsystem unavailable for {capability}.",
                "permission": "guard",
                "guidance": "Retry after the runtime security services are healthy.",
                "detail": str(exc),
            }

        try:
            guard = ServiceContainer.get("permission_guard", default=None)
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            _record_computer_use_degradation(
                exc,
                action="blocked desktop capability because permission guard lookup failed closed",
                stage="permissions.lookup",
                severity="degraded",
                extra={"capability": capability},
            )
            return {
                "ok": False,
                "status": "unavailable",
                "error": f"Permission guard unavailable for {capability}.",
                "permission": "guard",
                "guidance": "Retry after the runtime security services are healthy.",
                "detail": str(exc),
            }
        if guard is None:
            error = RuntimeError("permission guard is not registered")
            _record_computer_use_degradation(
                error,
                action="blocked desktop capability because permission guard was not registered",
                stage="permissions.lookup",
                severity="degraded",
                extra={"capability": capability},
            )
            return {
                "ok": False,
                "status": "unavailable",
                "error": f"Permission guard unavailable for {capability}.",
                "permission": "guard",
                "guidance": "Retry after the runtime security services are healthy.",
                "detail": str(error),
            }

        for permission_name in permission_names:
            permission_type = getattr(PermissionType, permission_name, None)
            if permission_type is None:
                continue
            permission_source = "cached"
            try:
                direct_checker = getattr(guard, "check_permission_direct", None)
                if callable(direct_checker):
                    permission_source = "direct"
                    permission_call = direct_checker(permission_type)
                else:
                    permission_call = guard.check_permission(permission_type, force=True)
                check = await asyncio.wait_for(
                    permission_call,
                    timeout=self.PERMISSION_CHECK_TIMEOUT_S,
                )
            except TimeoutError as exc:
                _record_computer_use_degradation(
                    exc,
                    action="returned bounded permission timeout instead of hanging desktop capability",
                    stage="permissions.timeout",
                    severity="warning",
                    extra={"capability": capability, "permission": permission_name.lower()},
                )
                guidance = ""
                try:
                    guidance = guard.get_guidance(permission_type)
                except _COMPUTER_USE_RECOVERABLE_ERRORS:
                    guidance = "Retry after the runtime security services are healthy."
                return {
                    "ok": False,
                    "status": "timeout",
                    "error": f"{permission_name.replace('_', ' ').title()} permission check timed out for {capability}.",
                    "permission": permission_name.lower(),
                    "permission_source": permission_source,
                    "guidance": guidance,
                    "detail": f"Exceeded {self.PERMISSION_CHECK_TIMEOUT_S:.1f}s permission preflight budget.",
                }
            except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                _record_computer_use_degradation(
                    exc,
                    action="blocked desktop capability because permission check failed closed",
                    stage="permissions.check",
                    severity="degraded",
                    extra={"capability": capability, "permission": permission_name.lower()},
                )
                return {
                    "ok": False,
                    "status": "unavailable",
                    "error": f"{permission_name.replace('_', ' ').title()} permission check failed for {capability}.",
                    "permission": permission_name.lower(),
                    "permission_source": permission_source,
                    "guidance": "Retry after the runtime security services are healthy.",
                    "detail": str(exc),
                }
            if check.get("granted"):
                continue
            human_name = permission_name.replace("_", " ").title()
            return {
                "ok": False,
                "status": check.get("status", "denied"),
                "error": f"{human_name} permission is required for {capability}.",
                "permission": permission_name.lower(),
                "permission_source": permission_source,
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
            return "UI inspection unavailable (background process lacks accessibility context)."
        return message or "AppleScript execution failed."

    def _run_applescript(self, script: str, *, timeout: int = 10) -> str:
        from core.runtime.action_executor import ActionExecutor

        timeout_s = max(1, int(timeout or 10))
        result = ActionExecutor.request_desktop_transport(
            script=script,
            source="computer_use",
            timeout_s=timeout_s,
        )
        if not result.get("ok"):
            stderr = str(result.get("stderr") or result.get("stdout") or "")
            if result.get("exit_code") == -1:
                raise TimeoutError(f"AppleScript timed out after {timeout_s}s.")
            raise RuntimeError(self._normalize_script_error(stderr))
        return str(result.get("stdout") or "").strip()

    # Read-back poll interval after an OS setting change (overridable so
    # tests don't pay the propagation wait).
    _SETTING_READBACK_INTERVAL_S = 0.5

    _HOTKEY_MODIFIERS = {
        "command": "command down",
        "cmd": "command down",
        "shift": "shift down",
        "option": "option down",
        "alt": "option down",
        "control": "control down",
        "ctrl": "control down",
    }
    _HOTKEY_KEY_CODES = {
        "return": 36,
        "enter": 36,
        "tab": 48,
        "space": 49,
        "escape": 53,
        "esc": 53,
        "delete": 51,
        "left": 123,
        "right": 124,
        "down": 125,
        "up": 126,
    }

    def _frontmost_app_name(self) -> str:
        """Which app owns the front window right now.

        The screen blueprint answers this in-process from the window server's
        own z-order, in about the time one osascript fork takes to *start*.
        That difference is structural rather than cosmetic: this is called in
        a poll loop, so the old path spent a 5s-timeout subprocess on every
        tick and made the wait it was supposed to measure the slowest part of
        the step.

        The AppleScript remains as the fallback, because a blueprint that
        cannot be taken must not silently become "no app is frontmost" — that
        reads as failure, and this function's answer decides whether we type.
        """
        try:
            from core.perception.screen_blueprint import capture_blueprint

            blueprint = capture_blueprint(fresh=True)
            if not blueprint.unavailable and blueprint.frontmost_app:
                return blueprint.frontmost_app
        except Exception as exc:  # noqa: BLE001 - fall through to the slow path
            logger.debug("Blueprint frontmost query unavailable: %s", exc)
        for attempt, timeout_s in enumerate((5, 10), start=1):
            try:
                name = self._run_applescript(
                    'tell application "System Events" to get name of first '
                    "application process whose frontmost is true",
                    timeout=timeout_s,
                ).strip()
            except (TimeoutError, RuntimeError) as exc:
                logger.warning(
                    "Frontmost app query failed (attempt %d, timeout %ss): %s",
                    attempt,
                    timeout_s,
                    exc,
                )
                continue
            if name:
                return name
            logger.warning(
                "Frontmost app query returned nothing on attempt %d.", attempt
            )
        # BOTH sources are gone, and that is not the same fact as "no app is
        # frontmost" — which is how it read to every caller. Live 2026-07-29
        # the browser step failed with "frontmost=unavailable, active_url=
        # unavailable" and reported it as the browser refusing to come
        # forward, when nothing had been able to look at the screen at all.
        # Logged loudly because a silent "" is what made it undiagnosable.
        logger.error(
            "Cannot determine the frontmost application: the window server "
            "and System Events both declined. Desktop steps that verify "
            "focus will refuse rather than act blind."
        )
        return ""

    def _window_is_actually_visible(self, app_name: str) -> tuple[bool, str]:
        """Is this app's window really in view, or just nominally in front?

        Frontmost is a claim about focus; a person means something stricter by
        "it's open" — that they can see it. A window can hold focus and still
        be almost entirely behind another one, and typing into it then looks
        to the person like nothing happened. The blueprint measures the
        visible fraction, so this is answerable instead of assumed.
        """
        try:
            from core.perception.screen_blueprint import capture_blueprint

            blueprint = capture_blueprint(fresh=True)
        except Exception as exc:  # noqa: BLE001
            return True, f"screen layout unreadable ({type(exc).__name__})"
        if blueprint.unavailable:
            return True, blueprint.unavailable_reason
        windows = blueprint.windows_for(app_name)
        if not windows:
            return False, f"{app_name} has no window on screen"
        best = max(windows, key=lambda window: window.visible_fraction)
        if best.is_visible:
            return True, best.describe()
        covering = best.covered_by[0] if best.covered_by else "another window"
        return False, f"{app_name} is hidden behind {covering}"

    @staticmethod
    def _frontmost_app_matches(actual: str, expected: str) -> bool:
        def _canonical(value: str) -> str:
            raw = str(value or "").strip().lower()
            raw = re.sub(r"\.app$", "", raw)
            raw = re.sub(r"\s+app$", "", raw)
            return re.sub(r"[^a-z0-9]+", "", raw)

        actual_name = _canonical(actual)
        expected_name = _canonical(expected)
        aliases = {
            "chrome": "googlechrome",
            "googlechrome": "googlechrome",
            "notesapp": "notes",
        }
        return bool(actual_name) and aliases.get(actual_name, actual_name) == aliases.get(
            expected_name,
            expected_name,
        )

    @staticmethod
    def _verifiable_applescript_activation_target(script: str) -> str:
        """Return the app for an exact activation-only AppleScript.

        Raw AppleScript output is not effect proof. The low-level computer-use
        lane therefore accepts only this narrow shape, whose postcondition can
        be read back deterministically. Rich scripts belong in ``os_automation``
        where a complete objective contract and repair loop are available.
        """
        value = str(script or "").strip()
        one_line = re.fullmatch(
            r'tell\s+application\s+"([^"\r\n]{1,160})"\s+to\s+activate',
            value,
            flags=re.IGNORECASE,
        )
        if one_line:
            return one_line.group(1).strip()
        block = re.fullmatch(
            r'tell\s+application\s+"([^"\r\n]{1,160})"\s*\r?\n'
            r"\s*activate\s*\r?\n\s*end\s+tell",
            value,
            flags=re.IGNORECASE,
        )
        return block.group(1).strip() if block else ""

    def _frontmost_or_prior_verified(
        self,
        front_app: str,
        expected_frontmost: str,
        context: dict[str, Any],
    ) -> tuple[str, bool]:
        """Use prior foreground proof only when the live probe is unavailable.

        A blank frontmost-app probe usually means System Events timed out or
        returned no value after a native app shortcut. If the desktop task
        immediately prior verified the same target app, keep moving. If the
        probe reports a different app, fail closed.
        """
        observed = str(front_app or "").strip()
        expected = str(expected_frontmost or "").strip()
        if observed or not expected:
            return observed, False
        if not bool(context.get("desktop_task_allow_unavailable_frontmost_from_prior")):
            return observed, False
        prior = str(context.get("desktop_task_prior_verified_frontmost_app") or "").strip()
        if prior and self._frontmost_app_matches(prior, expected):
            return expected, True
        return observed, False

    #: How the body arrives in the note. Both routes go through the Notes
    #: dictionary; they differ only in whether a person can watch it happen.
    #:
    #: Bryan asked to *see* her type, and the honest tension is that the
    #: mechanism which made this reliable — one atomic scripting call — is
    #: also the one nobody can watch. Keystrokes are watchable and lose the
    #: race for focus against whatever window is in front.
    #:
    #: So: keep the scripting interface, and write the body into the note in
    #: pieces. Notes redraws its editor on every assignment, so the text
    #: appears progressively in front of you. No keystroke is sent, no
    #: clipboard is used, nothing depends on which window has focus, and the
    #: note is still read back at the end.
    #: Visible typing. Short runs so it reads as typing rather than as paste,
    #: and a wall-clock budget so a long document never holds the step open —
    #: past it, the rest lands through the dictionary.
    _TYPING_CHUNK_CHARS = 12
    _TYPING_PAUSE_S = 0.04
    _TYPING_BUDGET_S = 25.0

    _NOTE_STREAM_CHUNK_CHARS = 90
    _NOTE_STREAM_PAUSE_S = 0.11
    #: The whole visible write is bounded: past this it finishes in one call.
    _NOTE_STREAM_BUDGET_S = 9.0

    #: How long hold_focus() will wait for an app to come to the front.
    #: Named because the write budget below is summed from it — an inline
    #: literal here and a guess there is how the two drifted apart.
    _HOLD_FOCUS_BUDGET_S = 12.0

    #: Worst case for finishing a document through the app's own scripting
    #: dictionary: make the document (20s), write the body (15s), read it
    #: back (15s). These are the AppleScript timeouts _write_through_dictionary
    #: actually passes.
    _DICTIONARY_WRITE_BUDGET_S = 50.0

    async def _write_in_app(self, target: Any) -> dict[str, Any]:
        """Put text into whatever application was named, generally.

        Bryan's correction: a create_note action is not OS control, it is one
        app hardcoded on a machine that happens to have Notes. She should be
        able to meet an application she has never seen, find out what it is,
        and work it.

        macOS already publishes the answer. Every scriptable app ships a
        scripting definition describing its own object model, so
        core/perception/app_dictionary.py derives "make a new X and set its Y"
        from the app itself: Notes answers note.body, TextEdit document.text,
        Reminders reminder.body. Nothing here knows any of those in advance.

        An app with no dictionary is not a failure and not a special case —
        it is the honest fallback to typing at it the way a person does, with
        the focus fragility that implies, which is exactly why the dictionary
        is preferred whenever one exists.
        """
        payload = target if isinstance(target, dict) else {}
        if not payload and isinstance(target, str):
            try:
                payload = json.loads(target)
            except (TypeError, ValueError):
                payload = {"body": target}
        app_name = canonical_app_target(
            str(payload.get("app") or payload.get("application") or "").strip()
        )
        if not app_name:
            return await self._create_note(target)

        from core.perception.app_dictionary import resolve_app, text_target_for

        resolved, _path = resolve_app(app_name)
        if not resolved:
            return {
                "ok": False,
                "error": f"{app_name} is not installed on this machine",
                "app": app_name,
            }
        recipe = text_target_for(resolved)

        # TYPE IT, if she can hold the front.
        #
        # Bryan: "I've seen her type in the notes app before ... i know she
        # can do it." He has, and she can. The reason it stopped is that
        # keystrokes were replaced wholesale by the scripting call after they
        # kept losing the front to Chrome mid-sequence — a real problem
        # solved by removing the thing that made it watchable.
        #
        # Order matters more than choice. hold_focus() already re-asserts the
        # front the way a person does, so try the keys first and watch her
        # type; the dictionary is what catches the words if the desktop takes
        # focus away anyway. That is a fallback, not a replacement, and the
        # result says which one wrote the text.
        typed = await self._type_into_app(resolved, payload)
        if typed.get("ok"):
            return typed

        if recipe is None:
            return {
                "ok": False,
                "error": (
                    f"{resolved} publishes no scripting dictionary and would not "
                    f"hold focus for typing ({typed.get('error') or 'focus lost'})"
                ),
                "app": resolved,
                "requires_keystrokes": True,
            }
        written = await self._write_through_dictionary(
            recipe, payload, into_existing=bool(typed.get("document_open"))
        )
        if written.get("ok"):
            written["typing_fallback_reason"] = str(
                typed.get("error") or "the app did not hold focus for typing"
            )
            written["typed_characters"] = int(typed.get("typed_characters") or 0)
            written["wrote_by"] = (
                "keystrokes, finished through the app's dictionary"
                if typed.get("document_open")
                else "the app's dictionary"
            )
        return written

    async def _type_into_app(self, app: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Open a fresh document and type the body into it, character by
        character, the way a person does.

        Bounded by the clock rather than by the text: a long document types
        its opening and then finishes through the dictionary, so watching it
        never costs the objective. Focus is re-asserted between chunks — that
        is the whole reason this can be tried at all.
        """
        body = str(payload.get("body") or payload.get("text") or "").strip()
        if not body:
            return {"ok": False, "error": "typing requires a body"}
        if not await self.hold_focus(app):
            return {"ok": False, "error": f"{app} did not come to the front"}

        try:
            await self._run_hotkey_for_app(app, "command+n")
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            return {"ok": False, "error": f"could not open a new document: {exc}"}
        # From here on a document EXISTS in the app. Every failure below has to
        # say so, or the dictionary fallback makes a second one and the person
        # is left with two half-written documents instead of one finished one.
        await asyncio.sleep(0.6)
        if not await self.hold_focus(app):
            return {
                "ok": False,
                "error": f"{app} lost the front before typing",
                "document_open": True,
            }

        # The title goes in first, because in a document the first line IS
        # the title — typed live, the whole opening paragraph became the
        # note's name (measured: a 1205-character note called
        # "I am Aura: a persistent digital organism — ...").
        title = str(payload.get("title") or "").strip()
        document = f"{title}\n\n{body}" if title else body

        deadline = time.monotonic() + float(self._TYPING_BUDGET_S)
        typed_chars = 0
        for chunk in self._typing_chunks(document):
            if time.monotonic() >= deadline:
                return {
                    "ok": False,
                    "error": "typing budget reached with the document unfinished",
                    "typed_characters": typed_chars,
                    "document_open": True,
                }
            if not self._frontmost_app_matches(
                await asyncio.to_thread(self._frontmost_app_name), app
            ):
                return {
                    "ok": False,
                    "error": f"{app} lost the front mid-sentence",
                    "typed_characters": typed_chars,
                    "document_open": True,
                }
            # A newline is a key, not a character. `keystroke "a\nb"` types
            # "ab" — measured live: a three-paragraph note arrived as one
            # unbroken wall because every blank line silently vanished.
            if chunk == "\n":
                script = 'tell application "System Events" to key code 36'
            else:
                script = (
                    'tell application "System Events" to keystroke '
                    f"{self._applescript_string(chunk)}"
                )
            try:
                await asyncio.to_thread(self._run_applescript, script, timeout=15)
            except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                return {
                    "ok": False,
                    "error": f"keystroke refused: {exc}",
                    "typed_characters": typed_chars,
                    "document_open": True,
                }
            typed_chars += len(chunk)
            await asyncio.sleep(float(self._TYPING_PAUSE_S))

        return {
            "ok": True,
            "action": "write_in_app",
            "app": app,
            "title": str(payload.get("title") or "").strip(),
            "effect_verified": True,
            "verification": (
                f"Typed {typed_chars} characters into {app} while it held the front."
            ),
            "characters": typed_chars,
            "wrote_by": "keystrokes",
        }

    async def _run_hotkey_for_app(self, app: str, combo: str) -> None:
        """Send one shortcut to a named app, re-asserting the front first."""
        await self.hold_focus(app)
        key = combo.rsplit("+", 1)[-1]
        modifiers = combo.rsplit("+", 1)[0].split("+") if "+" in combo else []
        using = ", ".join(f"{name.strip()} down" for name in modifiers if name.strip())
        script = (
            'tell application "System Events" to keystroke '
            f"{self._applescript_string(key)}"
            + (f" using {{{using}}}" if using else "")
        )
        await asyncio.to_thread(self._run_applescript, script, timeout=15)

    @classmethod
    def _typing_chunks(cls, body: str) -> list[str]:
        """Break the body where a person pauses: at word boundaries.

        One keystroke call per character would be honest and would also take
        four minutes for a paragraph, most of it osascript startup. A short
        run per call reads as typing and costs one fork per few words.
        """
        chunk = max(4, int(cls._TYPING_CHUNK_CHARS))
        chunks: list[str] = []
        # Each line is typed, and each newline between them is its own chunk
        # so the caller can send it as the Return key it actually is.
        lines = str(body or "").split("\n")
        for line_index, line in enumerate(lines):
            if line_index:
                chunks.append("\n")
            index = 0
            while index < len(line):
                end = min(len(line), index + chunk)
                if end < len(line):
                    space = line.rfind(" ", index, end)
                    if space > index:
                        end = space + 1
                chunks.append(line[index:end])
                index = end
        return chunks

    async def _create_note(self, target: Any) -> dict[str, Any]:
        """Create a Notes note through its scripting interface, visibly.

        Keystroke automation for this was never going to be reliable: it
        needs the app to hold the front from cmd+n through cmd+v, and on a
        real desktop the browser takes focus back mid-sequence. The Notes
        dictionary makes the whole thing one atomic call with no focus, no
        clipboard and no timing — and it verifies by reading the note back,
        so a silent failure cannot be reported as success.

        The body is then streamed in rather than pasted whole, so the writing
        is something a person can watch without the focus race that made
        watching it unreliable in the first place.
        """
        payload = target if isinstance(target, dict) else {}
        if not payload and isinstance(target, str):
            try:
                payload = json.loads(target)
            except (TypeError, ValueError):
                payload = {"body": target}
        title = str(payload.get("title") or payload.get("name") or "").strip()
        body = str(payload.get("body") or payload.get("text") or "").strip()
        if not body:
            return {"ok": False, "error": "create_note requires a body"}
        if not title:
            title = body.split("\n", 1)[0][:60].strip() or "Note"

        def _html(value: str) -> str:
            escaped = (
                value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            lines = [line for line in escaped.split("\n")]
            return "".join(f"<div>{line}</div>" for line in lines)

        # The note is created empty and named, then filled. Creating it with
        # the finished body would be one call and nothing to see.
        script = (
            'tell application "Notes"\n'
            "    activate\n"
            f"    set theNote to make new note with properties "
            f"{{name:{self._applescript_string(title)}, "
            # Notes renders `name` as the note's first line itself, so
            # prepending the title to the body prints it twice — measured:
            # a note that opened "Yourself / Yourself / ...".
            'body:""}\n'
            "    show theNote\n"
            "    return name of theNote\n"
            "end tell"
        )
        try:
            created = await asyncio.to_thread(self._run_applescript, script, timeout=20)
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            return {"ok": False, "error": f"create_note failed: {exc}", "title": title}

        created_name = str(created or "").strip()
        streamed = await self._stream_note_body(created_name or title, body, _html)
        if not streamed:
            return {
                "ok": False,
                "error": "create_note could not write the body into the note",
                "title": created_name or title,
            }
        # Read it back: a note that cannot be found was not created.
        verify = (
            'tell application "Notes" to return name of note '
            f"{self._applescript_string(created_name or title)}"
        )
        try:
            confirmed = await asyncio.to_thread(self._run_applescript, verify, timeout=15)
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            return {
                "ok": False,
                "error": f"create_note could not verify the note: {exc}",
                "title": title,
            }
        verified = bool(str(confirmed or "").strip())
        return {
            "ok": verified,
            "action": "create_note",
            "title": created_name or title,
            "effect_verified": verified,
            "verification": (
                f"Note '{created_name or title}' exists in Notes."
                if verified
                else "Note was not found after creation."
            ),
            "characters": len(body),
        }

    async def _write_through_dictionary(
        self, recipe: Any, payload: dict[str, Any], *, into_existing: bool = False
    ) -> dict[str, Any]:
        """Make a document in an app and fill it, using the app's own model.

        The Notes path is this path with recipe = note/body; nothing here is
        about Notes.

        ``into_existing`` finishes the document the typing pass already
        opened instead of making another one. Without it the two paths each
        made their own: live 2026-07-29 "open Notes and write a note" left
        Bryan TWO notes called "ABOUT AURA", because typing opened one with
        cmd+N, ran out of budget partway, and the fallback then created a
        second from scratch. The docstring on _type_into_app has always said a
        long document "types its opening and then finishes through the
        dictionary" — this is the part that makes that true rather than
        aspirational.
        """
        title = str(payload.get("title") or payload.get("name") or "").strip()
        body = str(payload.get("body") or payload.get("text") or "").strip()
        if not body:
            return {"ok": False, "error": "writing into an app requires a body"}
        if not title:
            title = body.split("\n", 1)[0][:60].strip() or "Untitled"

        app = self._applescript_string(recipe.app)
        wants_html = recipe.text_property == "body"
        render = self._html_paragraphs if wants_html else (lambda value: value)

        properties = []
        if recipe.name_property:
            properties.append(
                f"{recipe.name_property}:{self._applescript_string(title)}"
            )
        properties.append(f'{recipe.text_property}:""')
        if not into_existing:
            create = (
                f"tell application {app}\n"
                "    activate\n"
                f"    set theDoc to make new {recipe.klass} with properties "
                f"{{{', '.join(properties)}}}\n"
                "    return 1\n"
                "end tell"
            )
            try:
                await asyncio.to_thread(self._run_applescript, create, timeout=20)
            except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                return {
                    "ok": False,
                    "error": f"{recipe.app} refused to make a new {recipe.klass}: {exc}",
                    "app": recipe.app,
                    "title": title,
                }

        # Address the document we just made: the newest one is document 1 in
        # every app that orders them front-first, and by name where the class
        # has one, which is more robust when the user has others open.
        #
        # Continuing a typed document is the exception: its name came from
        # whatever text reached the first line before the budget ran out, so
        # it cannot be addressed by the title we intended. The front document
        # is the one that was just being typed into.
        if into_existing or not recipe.name_property:
            selector = f"{recipe.klass} 1"
        else:
            selector = f"{recipe.klass} {self._applescript_string(title)}"

        wrote = await self._stream_document_text(
            recipe.app, selector, recipe.text_property, body, render
        )
        if not wrote:
            return {
                "ok": False,
                "error": f"could not write the text into {recipe.app}",
                "app": recipe.app,
                "title": title,
            }

        if recipe.name_property:
            restore_title = (
                f"tell application {app} to set {recipe.name_property} of "
                f"{selector} to {self._applescript_string(title)}"
            )
            try:
                await asyncio.to_thread(
                    self._run_applescript, restore_title, timeout=15
                )
            except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                return {
                    "ok": False,
                    "error": f"{recipe.app} would not restore the requested title: {exc}",
                    "app": recipe.app,
                    "title": title,
                }
            verify = (
                f"tell application {app} to return "
                f"((count of characters of ({recipe.text_property} of {selector} as text)) as text) "
                f"& linefeed & ({recipe.name_property} of {selector} as text)"
            )
        else:
            verify = (
                f"tell application {app} to return "
                f"(count of characters of ({recipe.text_property} of {selector} as text))"
            )
        try:
            observed = await asyncio.to_thread(self._run_applescript, verify, timeout=15)
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            return {
                "ok": False,
                "error": f"{recipe.app} would not read the document back: {exc}",
                "app": recipe.app,
                "title": title,
            }
        observed_parts = str(observed or "0").splitlines()
        try:
            observed_chars = int((observed_parts[0] if observed_parts else "0").strip() or 0)
        except (TypeError, ValueError):
            observed_chars = 0
        observed_title = (
            "\n".join(observed_parts[1:]).strip()
            if recipe.name_property and len(observed_parts) > 1
            else ""
        )
        verified = observed_chars > 0 and (
            not recipe.name_property or observed_title == title
        )
        return {
            "ok": verified,
            "action": "write_in_app",
            "app": recipe.app,
            "title": title,
            "target": f"{recipe.klass}.{recipe.text_property}",
            "effect_verified": verified,
            "verification": (
                f"{recipe.app} holds '{observed_title or title}', a {recipe.klass} "
                f"of {observed_chars} characters."
                if verified
                else (
                    f"The {recipe.klass} in {recipe.app} did not read back with "
                    f"the requested title and non-empty body."
                )
            ),
            "characters": len(body),
            "observed_title": observed_title,
        }

    @staticmethod
    def _html_paragraphs(value: str) -> str:
        escaped = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return "".join(f"<div>{line}</div>" for line in escaped.split("\n"))

    async def _stream_document_text(
        self, app: str, selector: str, prop: str, body: str, render: Any
    ) -> bool:
        """Fill a document's text property in visible pieces.

        Same reason as the Notes streamer it generalises: Bryan asked to see
        her type, and the mechanism that made this reliable is the one nobody
        can watch. Every app that redraws on assignment shows the text
        arriving; none of it depends on which window has focus.
        """
        app_ref = self._applescript_string(app)
        deadline = time.monotonic() + float(self._NOTE_STREAM_BUDGET_S)
        chunk = max(16, int(self._NOTE_STREAM_CHUNK_CHARS))
        index = 0
        wrote_any = False
        while index < len(body):
            if time.monotonic() >= deadline:
                written = body
            else:
                end = min(len(body), index + chunk)
                if end < len(body):
                    space = body.rfind(" ", index, end)
                    if space > index:
                        end = space
                written = body[:end]
            index = len(written)
            script = (
                f"tell application {app_ref} to set {prop} of {selector} to "
                f"{self._applescript_string(render(written))}"
            )
            try:
                await asyncio.to_thread(self._run_applescript, script, timeout=15)
                wrote_any = True
            except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                logger.debug("Document text stream chunk failed: %s", exc)
                final = (
                    f"tell application {app_ref} to set {prop} of {selector} to "
                    f"{self._applescript_string(render(body))}"
                )
                try:
                    await asyncio.to_thread(self._run_applescript, final, timeout=20)
                    return True
                except _COMPUTER_USE_RECOVERABLE_ERRORS as inner:
                    logger.debug("Document text write failed outright: %s", inner)
                    return wrote_any
            if index < len(body):
                await asyncio.sleep(float(self._NOTE_STREAM_PAUSE_S))
        return wrote_any

    async def _stream_note_body(self, note_name: str, body: str, html: Any) -> bool:
        """Write the body into an existing note in visible pieces.

        Returns False only if the note never received any text — a note left
        empty is the failure Bryan reported as "notes that open with no text",
        and it must not be reported as a success.

        Chunks break on whitespace so words do not appear split, and the whole
        stream is bounded: past the budget the remainder lands in one
        assignment rather than letting a long document hold the step open.
        """
        target_note = self._applescript_string(note_name)
        written = ""
        deadline = time.monotonic() + float(self._NOTE_STREAM_BUDGET_S)
        chunk = max(16, int(self._NOTE_STREAM_CHUNK_CHARS))
        index = 0
        last_ok = False
        while index < len(body):
            if time.monotonic() >= deadline:
                written = body  # Out of budget: finish it in one assignment.
            else:
                end = min(len(body), index + chunk)
                if end < len(body):
                    # Prefer a whitespace boundary so words stay whole.
                    space = body.rfind(" ", index, end)
                    if space > index:
                        end = space
                written = body[:end]
            index = len(written)
            script = (
                f'tell application "Notes" to set body of note {target_note} '
                f"to {self._applescript_string(html(written))}"
            )
            try:
                await asyncio.to_thread(self._run_applescript, script, timeout=15)
                last_ok = True
            except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                logger.debug("Note body stream chunk failed: %s", exc)
                # One bad chunk is not a lost note: fall back to writing the
                # whole body once, and report honestly if even that fails.
                final = (
                    f'tell application "Notes" to set body of note {target_note} '
                    f"to {self._applescript_string(html(body))}"
                )
                try:
                    await asyncio.to_thread(self._run_applescript, final, timeout=20)
                    return True
                except _COMPUTER_USE_RECOVERABLE_ERRORS as inner:
                    logger.debug("Note body write failed outright: %s", inner)
                    return last_ok
            if index < len(body):
                await asyncio.sleep(float(self._NOTE_STREAM_PAUSE_S))
        return last_ok

    async def hold_focus(self, app_name: str) -> bool:
        """Keep an app in front for as long as we are working in it.

        Bryan's framing, and it is how people actually use a computer: you
        bring an app forward, you work in it for as long as you need, and
        then you put it away. Aura was instead checking "is it frontmost?"
        once per step and failing the whole task when a browser stole focus
        back between keystrokes.

        This re-asserts focus cheaply. Polling the frontmost app costs
        nothing; asking an app to activate is an AppleScript round trip, so
        it only happens when focus has actually been lost.
        """
        name = str(app_name or "").strip()
        if not name:
            return False
        current = await asyncio.to_thread(self._frontmost_app_name)
        if self._frontmost_app_matches(current, name):
            return True
        try:
            await self._activate_app(name)
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            logger.debug("hold_focus activation failed for %s: %s", name, exc)
            return False
        await asyncio.sleep(0.25)
        current = await asyncio.to_thread(self._frontmost_app_name)
        return self._frontmost_app_matches(current, name)

    async def release_focus(self, app_name: str) -> bool:
        """Put the app away when the work in it is done.

        The other half of how a person uses an app: when they finish, they
        hide it rather than leaving it sitting over everything the person was
        already doing.
        """
        name = str(app_name or "").strip()
        if not name:
            return False
        script = (
            f"tell application \"System Events\" to set visible of "
            f"application process {self._applescript_string(name)} to false"
        )
        try:
            await asyncio.to_thread(self._run_applescript, script, timeout=5)
            return True
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            logger.debug("release_focus failed for %s: %s", name, exc)
            return False

    async def _wait_for_frontmost_app(self, expected: str) -> tuple[bool, str]:
        """Wait for an app to actually reach the front, re-asking as we go.

        Ten attempts over ~3.5s was not enough against a browser that keeps
        reclaiming focus. Live 2026-07-28 "open Notes and write a note" failed
        with "did not become frontmost (observed=Google Chrome)" — Notes had
        launched, and the keystrokes would have gone to the wrong window, so
        failing was correct. The window was simply too short, and it matters
        more than usual here because the person is watching Aura's own UI in
        a browser while she works.

        Re-activation every other attempt rather than three times total: on
        macOS activation is a request, not a guarantee, and asking again is
        cheap.
        """
        # Bounded by the CLOCK, not by an attempt count.
        #
        # Counting attempts hid the real cost: each re-activation is an
        # AppleScript call with its own 5s timeout, so "retry every other
        # attempt" turned an intended 8s wait into 60s and blew the step
        # budget outright — "open_app failed: Operation took too long".
        # Polling is cheap and re-asking is expensive, so they get separate
        # budgets.
        deadline = time.monotonic() + float(self._HOLD_FOCUS_BUDGET_S)
        last_activation = 0.0
        last_seen = ""
        while time.monotonic() < deadline:
            last_seen = await asyncio.to_thread(self._frontmost_app_name)
            if self._frontmost_app_matches(last_seen, expected):
                return True, last_seen
            now = time.monotonic()
            if now - last_activation >= 3.0:
                last_activation = now
                try:
                    await self._activate_app(expected)
                except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                    logger.debug("Frontmost activation retry failed for %s: %s", expected, exc)
            await asyncio.sleep(0.35)

        # Say what actually happened, not just which name came back. "did not
        # become frontmost (observed=Google Chrome)" is a symptom; the
        # blueprint knows whether the window never opened, opened behind
        # something, or opened where nobody can see it — and those are three
        # different problems with three different repairs.
        try:
            visible, detail = await asyncio.to_thread(
                self._window_is_actually_visible, expected
            )
        except Exception as exc:  # noqa: BLE001 - diagnosis may never mask the result
            logger.debug("Blueprint diagnosis failed for %s: %s", expected, exc)
            return False, last_seen
        if detail and not visible:
            return False, f"{last_seen or 'unknown'} ({detail})"
        return False, last_seen

    #: Bundle identifiers macOS uses for the browsers we allow, so the
    #: registered default resolves to a name the rest of the lane knows.
    _BROWSER_BUNDLE_NAMES: dict[str, str] = {
        "com.apple.safari": "Safari",
        "com.google.chrome": "Google Chrome",
        "org.mozilla.firefox": "Firefox",
        "com.microsoft.edgemac": "Microsoft Edge",
        "company.thebrowser.browser": "Arc",
    }

    def _default_browser_name(self) -> str:
        """Which browser this Mac hands http(s) URLs to.

        Read from LaunchServices rather than inferred from the screen: the
        screen answers a different question ("what is in front right now"),
        and immediately after `open` the answer is always still the app that
        called it.
        """
        import plistlib

        preference = (
            Path.home()
            / "Library/Preferences/com.apple.LaunchServices"
            / "com.apple.launchservices.secure.plist"
        )
        handlers: list[Any] = []
        try:
            handlers = plistlib.loads(preference.read_bytes()).get("LSHandlers", [])
        except (OSError, ValueError, plistlib.InvalidFileException) as exc:
            # An absent or unreadable preference file is the ordinary state of
            # a Mac whose owner never changed the default — falling through to
            # Safari is the answer, not "" (which reads as "no browser at all"
            # and skips verification entirely, the bug this method exists for).
            logger.debug("Could not read the default browser handler: %s", exc)
        for handler in handlers:
            if not isinstance(handler, dict):
                continue
            if str(handler.get("LSHandlerURLScheme") or "").lower() not in {
                "http",
                "https",
            }:
                continue
            bundle = str(handler.get("LSHandlerRoleAll") or "").lower()
            name = self._BROWSER_BUNDLE_NAMES.get(bundle)
            if name in _ALLOWED_URL_BROWSERS:
                return name
        # No explicit handler registered means Safari, which is the macOS
        # default and not a guess.
        return "Safari" if "Safari" in _ALLOWED_URL_BROWSERS else ""

    @staticmethod
    def _applescript_string(value: str) -> str:
        return '"' + str(value or "").replace("\\", "\\\\").replace('"', '\\"') + '"'

    async def _activate_app(self, app_name: str) -> str:
        """Bring an already launched macOS application to the foreground."""
        script = f"tell application {self._applescript_string(app_name)} to activate"
        return await asyncio.to_thread(self._run_applescript, script, timeout=5)

    def _force_browser_tab_url(self, browser: str, url: str) -> str:
        """Create a foreground tab for a URL in a named browser.

        `open -a Chrome <url>` can be intercepted by restored sessions,
        extension start pages, and auth/login tabs. For live desktop tasks the
        action is not complete until the named browser exposes the requested
        URL as its active tab, so this helper uses the browser's AppleScript
        surface directly as a bounded repair step.
        """
        browser = str(browser or "").strip()
        target = str(url or "").strip()
        if not browser or not target:
            return ""
        quoted_url = self._applescript_string(target)
        if browser in {"Google Chrome", "Arc", "Microsoft Edge"}:
            script = f'''
tell application "{browser}"
    activate
    if (count of windows) = 0 then make new window
    tell front window
        set newTab to make new tab at end of tabs with properties {{URL:{quoted_url}}}
        set active tab index to (count of tabs)
    end tell
end tell
'''
        elif browser == "Safari":
            script = f'''
tell application "Safari"
    activate
    if (count of windows) = 0 then
        make new document with properties {{URL:{quoted_url}}}
    else
        tell front window
            set current tab to (make new tab with properties {{URL:{quoted_url}}})
        end tell
    end if
end tell
'''
        else:
            return ""
        return self._run_applescript(script, timeout=8)

    def _focused_element_snapshot(self) -> str:
        """Read only the focused control, avoiding a full accessibility walk."""
        script = """
tell application "System Events"
    set frontProc to first application process whose frontmost is true
    set focusedElement to value of attribute "AXFocusedUIElement" of frontProc
    set roleName to ""
    set valueText to ""
    set titleText to ""
    set descriptionText to ""
    set identifierText to ""
    try
        set roleName to value of attribute "AXRole" of focusedElement as string
    end try
    try
        set valueText to value of attribute "AXValue" of focusedElement as string
    end try
    try
        set titleText to value of attribute "AXTitle" of focusedElement as string
    end try
    try
        set descriptionText to value of attribute "AXDescription" of focusedElement as string
    end try
    try
        set identifierText to value of attribute "AXIdentifier" of focusedElement as string
    end try
    return roleName & tab & valueText & tab & titleText & tab & descriptionText & tab & identifierText
end tell
""".strip()
        try:
            return self._run_applescript(script, timeout=4).strip()
        except (TimeoutError, RuntimeError) as exc:
            logger.debug("Focused element snapshot failed: %s", exc)
            return ""

    async def _focus_web_editor_surface(
        self,
        pyautogui: Any,
        *,
        browser: str = "",
        target_url: str = "",
    ) -> tuple[bool, str, str]:
        """Move browser focus from the omnibox into a web editor body.

        Google Docs/Sheets/Slides expose editor bodies through canvas-heavy UI,
        so the invariant must be stronger than "not the URL bar": focus must look
        like an editor surface before a paste/type is allowed.
        """
        if pyautogui is None:
            return False, "", "pyautogui_unavailable_for_web_editor_focus"

        last_snapshot = ""
        try:
            # If navigation left the omnibox active, Escape returns focus to the
            # page without changing the URL or the document contents.
            await asyncio.to_thread(self._send_hotkey_system_events, ["escape"])
        except (TimeoutError, RuntimeError) as exc:
            logger.debug("Web editor focus escape preflight skipped: %s", exc)

        click_points = ((0.50, 0.45), (0.50, 0.55), (0.58, 0.50))
        try:
            screen_w, screen_h = await asyncio.to_thread(pyautogui.size)
        except (RuntimeError, OSError, ValueError, AttributeError) as exc:
            return False, "", f"screen_size_unavailable_for_web_editor_focus: {exc}"

        for x_ratio, y_ratio in click_points:
            try:
                await asyncio.to_thread(
                    pyautogui.click,
                    int(screen_w * x_ratio),
                    int(screen_h * y_ratio),
                )
                await asyncio.sleep(0.35)
                last_snapshot = await asyncio.to_thread(self._focused_element_snapshot)
            except (RuntimeError, OSError, ValueError, AttributeError) as exc:
                return False, last_snapshot, f"web_editor_focus_click_failed: {exc}"
            if self._focused_snapshot_looks_web_editor_surface(last_snapshot):
                return True, last_snapshot, "editable_focus_verified"
            if (
                browser
                and target_url
                and (
                    self._focused_snapshot_looks_web_editor_canvas_candidate(last_snapshot)
                    or not str(last_snapshot or "").strip()
                )
            ):
                active_url, _active_title = await asyncio.to_thread(
                    self._active_browser_location,
                    browser,
                )
                if self._is_resolved_web_editor_url(
                    active_url
                ) and self._url_semantically_matches(target_url, active_url):
                    reason = (
                        "editable_focus_verified_canvas_no_ax_focus"
                        if not str(last_snapshot or "").strip()
                        else "editable_focus_verified_canvas_url"
                    )
                    return True, last_snapshot, reason
            if self._focused_snapshot_looks_browser_location_bar(last_snapshot):
                continue

        if self._focused_snapshot_looks_browser_location_bar(last_snapshot):
            return False, last_snapshot, "browser_location_bar_still_focused"
        if self._focused_snapshot_is_browser_text_entry(last_snapshot):
            return False, last_snapshot, "generic_browser_text_field_focused"
        return False, last_snapshot, "editable_focus_unverified"

    @staticmethod
    def _focused_snapshot_looks_browser_location_bar(snapshot: str) -> bool:
        """Best-effort guard for URL-bar pastes on browser surfaces."""
        raw = str(snapshot or "").strip()
        if not raw:
            return False
        parts = raw.split("\t")
        role = parts[0].strip().lower() if parts else ""
        value = parts[1].strip().lower() if len(parts) > 1 else raw.lower()
        metadata = " ".join(part.strip().lower() for part in parts[1:])
        if role not in {"axtextfield", "axcombobox"}:
            return False
        return bool(
            value.startswith(("http://", "https://"))
            or "search or enter" in metadata
            or "address" in metadata
            or "omnibox" in metadata
            or "url" in metadata
            or "location" in metadata
            or "duckduckgo.com/" in metadata
            or "google.com/search" in metadata
            or "docs.google.com/" in metadata
        )

    @staticmethod
    def _focused_snapshot_is_browser_text_entry(snapshot: str) -> bool:
        raw = str(snapshot or "").strip()
        if not raw:
            return False
        role = raw.split("\t", 1)[0].strip().lower()
        return role in {"axtextfield", "axcombobox"}

    @staticmethod
    def _focused_snapshot_looks_web_editor_canvas_candidate(snapshot: str) -> bool:
        """Positive-but-conservative proof for canvas-backed editors.

        Google Docs can expose the editable canvas as a plain AXWebArea/AXGroup
        without "document" metadata. That shape is only acceptable when the
        caller also proves the active browser URL is still a known editor URL.
        """
        raw = str(snapshot or "").strip()
        if not raw:
            return False
        if ComputerUseSkill._focused_snapshot_looks_browser_location_bar(raw):
            return False
        if ComputerUseSkill._focused_snapshot_is_browser_text_entry(raw):
            return False
        parts = [part.strip().lower() for part in raw.split("\t")]
        role = parts[0] if parts else ""
        metadata = " ".join(part for part in parts[1:] if part)
        disallowed_hints = (
            "address",
            "email",
            "login",
            "omnibox",
            "password",
            "search or enter",
            "sign in",
            "url",
            "username",
        )
        if any(hint in metadata for hint in disallowed_hints):
            return False
        return role in {
            "axgroup",
            "axlayoutarea",
            "axscrollarea",
            "axtextarea",
            "axunknown",
            "axwebarea",
        }

    @staticmethod
    def _focused_snapshot_looks_web_editor_surface(snapshot: str) -> bool:
        """Best-effort positive proof for browser editor focus.

        Generic AXTextField/AXComboBox controls are explicitly rejected because
        they include browser omniboxes, search fields, and sign-in inputs. Google
        Docs often exposes the editing target as a text area, web area, scroll
        area, or group with editor/document descriptors; those are acceptable
        only when the metadata also points at a document/editor body.
        """
        raw = str(snapshot or "").strip()
        if not raw:
            return False
        parts = [part.strip().lower() for part in raw.split("\t")]
        role = parts[0] if parts else ""
        metadata = " ".join(part for part in parts[1:] if part)
        if ComputerUseSkill._focused_snapshot_looks_browser_location_bar(raw):
            return False
        if role in {"axtextfield", "axcombobox"}:
            return False
        editor_roles = {
            "axtextarea",
            "axwebarea",
            "axgroup",
            "axscrollarea",
            "axlayoutarea",
            "axunknown",
        }
        editor_hints = (
            "document",
            "editor",
            "editing",
            "canvas",
            "body",
            "page",
            "google docs",
            "google sheets",
            "google slides",
        )
        return role in editor_roles and any(hint in metadata for hint in editor_hints)

    def _send_hotkey_system_events(self, keys: list[str]) -> str:
        """Send a keyboard shortcut via System Events; raise with the real
        error on refusal (e.g. missing Automation/Accessibility grants)."""
        mods = [self._HOTKEY_MODIFIERS[k] for k in keys if k in self._HOTKEY_MODIFIERS]
        plains = [k for k in keys if k not in self._HOTKEY_MODIFIERS]
        if len(plains) != 1:
            raise RuntimeError(f"unsupported hotkey combination: {'+'.join(keys)}")
        key = plains[0]
        if key in self._HOTKEY_KEY_CODES:
            stroke = f"key code {self._HOTKEY_KEY_CODES[key]}"
        elif len(key) == 1 and (key.isalnum() or key in ".,;/-=[]'\\`"):
            stroke = f'keystroke "{key}"'
        else:
            raise RuntimeError(f"unsupported hotkey key: {key}")
        using = f" using {{{', '.join(mods)}}}" if mods else ""
        self._run_applescript(
            f'tell application "System Events" to {stroke}{using}', timeout=8
        )
        return f"system_events:{stroke}{using}"

    @staticmethod
    def _send_hotkey_pyautogui(pyautogui: Any, keys: list[str]) -> str:
        """Fallback keyboard dispatch for native apps when System Events stalls.

        This is only used after Accessibility preflight has passed and
        System Events times out rather than refuses permission. It is not used
        as proof by itself; the regular hotkey receipt verification below still
        decides whether the action counts.
        """
        aliases = {
            "cmd": "command",
            "control": "ctrl",
            "return": "enter",
            "escape": "esc",
        }
        normalized = [aliases.get(str(key or "").strip().lower(), str(key or "").strip().lower()) for key in keys]
        normalized = [key for key in normalized if key]
        if not normalized:
            raise RuntimeError("unsupported empty hotkey combination")
        pyautogui.hotkey(*normalized, interval=0.05)
        return f"pyautogui:{'+'.join(normalized)}"

    @staticmethod
    def _normalize_open_url_target(target: str) -> str:
        text = str(target or "").strip()
        if not text:
            return ""
        if text.startswith(("http://", "https://")):
            return text
        return f"https://duckduckgo.com/?q={urllib.parse.quote_plus(text)}"

    @staticmethod
    def _runtime_permission_payload(message: str) -> dict[str, Any] | None:
        try:
            from core.security.permission_guard import PermissionType, get_permission_guard
        except (ImportError, AttributeError, RuntimeError):
            return None

        try:
            guard = get_permission_guard()
        except _COMPUTER_USE_RECOVERABLE_ERRORS:
            return None
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

    def _validate_user_applescript(self, script: str) -> str:
        text = str(script or "").strip()
        if not text:
            raise ValueError("No AppleScript provided.")
        if len(text) > self.MAX_APPLESCRIPT_CHARS:
            raise ValueError(
                f"AppleScript is too large for bounded desktop execution "
                f"({len(text)} > {self.MAX_APPLESCRIPT_CHARS})."
            )
        for pattern in self.APPLESCRIPT_DENYLIST:
            if pattern.search(text):
                raise ValueError("AppleScript contains a blocked desktop operation.")
        return text

    def _set_clipboard(self, text: str) -> dict[str, Any]:
        expected = str(text or "")
        result = get_subprocess_gateway().run(
            ["pbcopy"],
            input=expected,
            capture_output=True,
            timeout=5,
            source="computer_use",
            accelerator_capability="none",
        )
        if result.returncode != 0:
            return {"ok": False, "error": (result.stderr or result.stdout or "pbcopy failed").strip()}
        observed = self._get_clipboard()
        actual = str(observed.get("text") or "") if observed.get("ok") else ""
        verified = bool(observed.get("ok")) and actual == expected
        response = {
            "ok": verified,
            "action": "set_clipboard",
            "chars": len(expected),
            "sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
            "effect_verified": verified,
            "verification": (
                "Clipboard read-back matched the requested text."
                if verified
                else "Clipboard write completed, but exact read-back did not match."
            ),
        }
        if not verified:
            response["error"] = response["verification"]
        return response

    @staticmethod
    def _get_clipboard() -> dict[str, Any]:
        result = get_subprocess_gateway().run(
            ["pbpaste"],
            capture_output=True,
            timeout=5,
            read_only=True,
            source="computer_use",
            accelerator_capability="none",
        )
        if result.returncode != 0:
            return {"ok": False, "error": (result.stderr or result.stdout or "pbpaste failed").strip()}
        text = result.stdout or ""
        return {"ok": True, "action": "get_clipboard", "text": text, "chars": len(text)}

    def _allowed_desktop_roots(self) -> list[Path]:
        return [
            Path.home() / "Desktop",
            Path.home() / "Documents",
            Path.cwd() / "artifacts" / "live_runtime",
        ]

    def _allowed_readable_roots(self) -> list[Path]:
        """Where she may LOOK. A superset of where she may write.

        Reading and writing are different risk classes and had one allowlist
        between them. Asked to count the .py files in her own
        core/introspection directory, the plan was refused by the WRITE guard —
        correctly, since it had aimed a write there, but the read was never
        possible either, so a question about her own source could only ever be
        answered by guessing. She guessed 3; there were 9.

        Her own source tree is added because source proprioception is a
        capability she is meant to have. Nothing else is: this is not the
        filesystem, it is the three artifact roots plus the repository she
        runs from.
        """
        roots = list(self._allowed_desktop_roots())
        try:
            roots.append(Path(__file__).resolve().parents[2])
        except (OSError, IndexError):
            pass
        return roots

    def _resolve_readable_path(self, raw_path: Any, *, must_exist: bool = True) -> Path:
        """Resolve a path she is permitted to read, or refuse with the reason.

        Same symlink defence as the write guard: the check is against the
        RESOLVED path, so ~/Desktop/Aura pointing into the source tree is
        judged by where it lands rather than how it is spelled.
        """
        if not raw_path:
            raise ValueError("Path is required.")
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = Path.home() / "Documents" / path
        resolved = path.resolve(strict=must_exist)
        for root in self._allowed_readable_roots():
            allowed = root.expanduser().resolve(strict=False)
            try:
                if os.path.commonpath([str(allowed), str(resolved)]) == str(allowed):
                    return resolved
            except (OSError, ValueError):
                continue
        roots = ", ".join(str(root.expanduser()) for root in self._allowed_readable_roots())
        detail = f"{path} resolves to {resolved}"
        if str(resolved) != str(path):
            detail += " (a link or alias points outside the readable roots)"
        raise ValueError(
            f"Path is outside Aura's readable roots: {detail}. Readable roots: {roots}."
        )

    def _list_directory(self, target: str) -> dict[str, Any]:
        """Names and count of the files in a directory she may read.

        The count is a MEASUREMENT: asked how many
        .py files were in a directory, she answered 3 for a directory holding
        9, listed three filenames that do not exist, and reported writing a
        file that was never created. Nothing had looked.
        """
        payload = self._target_json(target)
        try:
            path = self._resolve_readable_path(payload.get("path"))
        except ValueError as exc:
            return {"ok": False, "action": "list_directory", "error": str(exc)}
        if not path.is_dir():
            return {
                "ok": False,
                "action": "list_directory",
                "error": f"Not a directory: {path}",
            }
        pattern = str(payload.get("pattern") or "*").strip() or "*"
        try:
            names = sorted(
                entry.name for entry in path.glob(pattern) if entry.is_file()
            )
        except (OSError, ValueError) as exc:
            return {"ok": False, "action": "list_directory", "error": str(exc)}
        return {
            "ok": True,
            "action": "list_directory",
            "path": str(path),
            "pattern": pattern,
            "names": names,
            "count": len(names),
            "effect_verified": True,
            "verification": f"Read {len(names)} entries matching {pattern} in {path}.",
        }

    def _resolve_allowed_desktop_path(self, raw_path: Any, *, must_exist: bool = False) -> Path:
        if not raw_path:
            raise ValueError("Path is required.")
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = Path.home() / "Desktop" / path
        resolved = path.resolve(strict=must_exist)
        for root in self._allowed_desktop_roots():
            allowed = root.expanduser().resolve(strict=False)
            try:
                if os.path.commonpath([str(allowed), str(resolved)]) == str(allowed):
                    return resolved
            except (OSError, ValueError):
                continue
        # Say WHERE it landed, not just that it was refused.
        #
        # LIVE, 2026-08-10. Asked to write ~/Desktop/Aura/aura_selftest.md, the
        # refusal read "Path is outside Aura's allowed desktop/document artifact
        # roots." That is true and it reads as a contradiction, because the path
        # is visibly under ~/Desktop. ~/Desktop/Aura is a symlink to
        # /Users/bryan/.aura/live-source, so the write would have landed inside
        # her own source tree — the guard was exactly right and said nothing
        # that would let anyone work that out.
        #
        # A refusal that cannot be acted on gets retried verbatim, which is how
        # a correct guard turns into a loop.
        roots = ", ".join(
            str(root.expanduser()) for root in self._allowed_desktop_roots()
        )
        detail = f"{path} resolves to {resolved}"
        if str(resolved) != str(path):
            detail += " (a link or alias points outside the artifact roots)"
        raise ValueError(
            "Path is outside Aura's allowed desktop/document artifact roots: "
            f"{detail}. Allowed roots: {roots}."
        )

    async def _pursue_on_screen(self, target: Any) -> dict[str, Any]:
        """Watch and act until a goal is reached, or a bound runs out.

        Every other desktop action performs once. This one keeps a goal until
        the screen says it is finished, which is what any request carrying a
        condition actually needs. The judgement inside it is hers: the loop
        reasons about the moves really available, predicts what the chosen one
        should change, and grades that against the next reading.
        """
        from core.skills.screen_pursuit import DEFAULT_MOVES, pursue_on_screen

        payload = self._target_json(target)
        goal = str(payload.get("goal") or "").strip()
        success_when = str(payload.get("success_when") or payload.get("until") or "").strip()
        if not goal:
            return {"ok": False, "action": "pursue_on_screen", "error": "no goal was given to pursue"}
        # A run with no finishing condition ends on its bounds, which it has.
        #
        # Refusing one was guarding against a loop that could never stop, but
        # the loop has always stopped: it runs to a cycle count and a clock,
        # both of them arguments to it. What the refusal actually blocked was
        # every request that names a process without naming an end — "play it
        # and work out how it moves" — which is most of the ways a person asks
        # for one. LIVE 2026-08-27: the goal reached this line correctly
        # parsed, with the page to open and the keys to press, and was turned
        # away in 417ms.

        keys = payload.get("move_keys") or payload.get("moves") or list(DEFAULT_MOVES)
        result = await pursue_on_screen(
            goal=goal,
            success_when=success_when,
            move_keys=tuple(str(key) for key in keys),
            max_cycles=int(payload.get("max_cycles") or 200),
            max_seconds=float(payload.get("max_seconds") or 600.0),
            # The clock a caller started before this action was reached.
            #
            # Dropped here, the pursuit began counting when it began running,
            # and the time spent getting to it belonged to nobody: live, a
            # run that had built a 64 into the corner was cancelled from
            # outside at exactly the outer budget, 660s to the millisecond.
            deadline_at=float(payload.get("deadline_at") or 0.0),
            narrate=bool(payload.get("narrate", True)),
            region_top=float(payload.get("region_top") or 0.0),
            region_bottom=float(payload.get("region_bottom") or 1.0),
            target_app=str(payload.get("target_app") or ""),
            expect_page=str(payload.get("expect_page") or ""),
            open_page=str(payload.get("open_page") or ""),
            unblock_with=str(payload.get("unblock_with") or ""),
            stakes=float(payload.get("stakes") or 0.5),
        )
        outcome = str(result.get("outcome") or "")
        moves = result.get("moves") or []
        graded = result.get("attempts") or []
        held = sum(1 for row in graded if isinstance(row, dict) and row.get("held"))
        # Name the reason at the point it is known.
        #
        # The step verifier reports a failed action's own "error", falling
        # back to "child action reported failure" — a sentence that tells the
        # person nothing. A pursuit always knows better than that: it was
        # blocked by something named, it lost the page, it ran out of moves,
        # or it could not decide.
        reason = ""
        if not result.get("completed"):
            reason = (
                str(result.get("cannot_see") or "")
                or str(result.get("cannot_decide") or "")
                or str(result.get("needs_person") or "")
                or str(result.get("blocked_by") or "")
                or str(result.get("could_not_get_there") or "")
                or {
                    "could_not_get_there": "she could not get to where the task happens",
                    "out_of_cycles": "ran out of moves before reaching the goal",
                    "out_of_time": "ran out of time before reaching the goal",
                    "navigated_away": "the page it was working on was replaced",
                    "no_move_available": "nothing on screen offered a move",
                    "cannot_see": "the screen could not be read at all",
                    "stalled": "the screen stopped changing",
                }.get(outcome, outcome or "the goal was not reached")
            )
            reason = f"{reason} (after {len(moves)} move(s))"
        # What she did, in words, so the turn can report it rather than
        # handing somebody a step count. General to any pursuit: how many
        # moves, how many did what she expected, and how it ended.
        made = len(moves)
        if result.get("completed"):
            said = f"Reached it: {success_when!r} appeared after {made} move(s)."
        elif made:
            said = f"Made {made} move(s) and stopped — {reason}."
        else:
            said = f"Did not get started — {reason}."
        if made:
            said += f" {held} of them did what I expected."
        if result.get("restarts"):
            said += f" Began again {result['restarts']} time(s)."
        pace = result.get("pacing") or {}
        if pace.get("chose"):
            said += f" I chose to {pace['chose']} to keep my commentary with my hands."

        return {
            "ok": bool(result.get("completed")),
            "error": reason,
            "said": said,
            "summary": said,
            "action": "pursue_on_screen",
            "goal": goal,
            "outcome": outcome,
            "cycles": result.get("cycles"),
            "moves": result.get("moves") or [],
            "attempts": result.get("attempts") or [],
            "blocked_by": result.get("blocked_by", ""),
            "needs_person": result.get("needs_person", ""),
            "cannot_decide": result.get("cannot_decide", ""),
            "restarts": result.get("restarts", 0),
            "pacing": result.get("pacing") or {},
            "could_not_get_there": result.get("could_not_get_there", ""),
            "wanted": result.get("wanted", ""),
            "verification": f"pursuit ended {outcome or 'without a named outcome'}",
            # A run that spent its clock has none left for a second try.
            #
            # The caller retries a critical step that did not finish, which
            # is right for a keystroke that missed and wrong for this: the
            # retry gets a deadline that has already passed, makes no move,
            # and its empty receipt replaces the one describing the work.
            # LIVE 2026-08-26: thirty-eight narrated moves reported as
            # "ran out of time before reaching the goal (after 0 move(s))".
            "retryable": outcome not in {"out_of_time", "out_of_cycles"},
        }

    @staticmethod
    def _target_json(target: str) -> dict[str, Any]:
        try:
            payload = json.loads(str(target or "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Target must be a JSON object: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Target must be a JSON object.")
        return payload

    @staticmethod
    def _versioned_path(path: Path) -> Path:
        """Next free 'name (N).ext' so repeats never overwrite or fail.

        Refusing outright killed whole desktop chains on the second run
        of the same request (observed live: 'Refusing to overwrite'
        surfaced to the user as an opaque task failure). Safety stays —
        existing data is never touched — and the action reports the
        path it actually wrote.
        """
        if not path.exists():
            return path
        for index in range(2, 1000):
            candidate = path.with_name(f"{path.stem} ({index}){path.suffix}")
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"No free versioned name for {path}")

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _has_pdf_header(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"

    def _write_text_file(self, target: str) -> dict[str, Any]:
        payload = self._target_json(target)
        path = self._resolve_allowed_desktop_path(payload.get("path"))
        content = str(payload.get("content") or "")
        overwrite = bool(payload.get("overwrite", False))
        append = bool(payload.get("append", False))
        prepend = bool(payload.get("prepend", False))
        requested = path
        # Adding to a file is not writing a file. Asked to append one line to
        # a note, the planner emitted a plain overwrite and the reply said
        # "the file now contains both lines" — the prior line would have been
        # destroyed. Appending preserves what is there, and the verification
        # below hashes the WHOLE resulting file, so a truthful receipt is only
        # possible when the earlier content actually survived.
        prior = ""
        if (append or prepend) and path.is_file():
            prior = path.read_text(encoding="utf-8", errors="replace")
        elif path.exists() and not overwrite:
            path = self._versioned_path(path)
        if prepend:
            if content and not content.endswith("\n") and not prior.startswith("\n"):
                content += "\n"
            content = content + prior
        else:
            if prior and not prior.endswith("\n") and not content.startswith("\n"):
                prior += "\n"
            content = prior + content
        atomic_write_text(path, content, encoding="utf-8")
        expected_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        actual_digest = self._file_sha256(path)
        verified = path.is_file() and actual_digest == expected_digest
        return {
            "ok": verified,
            "action": "write_text_file",
            "path": str(path),
            "requested_path": str(requested),
            "versioned": path != requested,
            "appended": bool(prior),
            "preserved_bytes": len(prior.encode("utf-8")),
            "bytes": path.stat().st_size,
            "sha256": actual_digest,
            "effect_verified": verified,
            "verification": (
                "File content matched after atomic write."
                if verified
                else "File write completed, but content read-back did not match."
            ),
            **({} if verified else {"error": "File content verification failed."}),
        }

    def _create_folder(self, target: str) -> dict[str, Any]:
        payload = self._target_json(target)
        path = self._resolve_allowed_desktop_path(payload.get("path"))
        existed = path.exists()
        if existed and not path.is_dir():
            return {"ok": False, "error": f"Path exists and is not a folder: {path}"}
        path.mkdir(parents=True, exist_ok=True)
        verified = path.is_dir()
        return {
            "ok": verified,
            "action": "create_folder",
            "path": str(path),
            "existed": existed,
            "effect_verified": verified,
            "verification": (
                "Folder existence confirmed."
                if verified
                else "Folder creation returned without a readable directory."
            ),
            **({} if verified else {"error": "Folder existence verification failed."}),
        }

    async def _apply_system_control(self, target: str) -> dict[str, Any]:
        """Drive a known OS setting to a goal-state.

        Domain-agnostic and unified: WHICH settings exist and how to
        recognize/translate them comes from the OS-affordance registry;
        EXECUTION is delegated to the one canonical owner, OSSettingsAdapter
        (rollback + governed receipts). This method neither composes
        AppleScript nor knows any specific setting — adding wallpaper,
        dark mode, volume, or a future setting is a registry entry. The
        prior state is recorded and the goal-state is confirmed by
        read-back through the adapter's own getter, never assumed.
        """
        from core.container import ServiceContainer
        from core.skills.os_affordances import get_affordance, validate_value

        payload = self._target_json(target)
        domain = str(payload.get("domain") or "").strip().lower()
        raw_value = str(payload.get("value") or "").strip()
        affordance = get_affordance(domain)
        if affordance is None:
            return {"ok": False, "error": f"No known OS affordance for '{domain}'."}
        value = validate_value(affordance, raw_value)
        if value is None:
            return {"ok": False, "error": f"Invalid value for {domain}: {raw_value!r}"}
        # Image-valued settings (wallpaper) take a file that must live in
        # the allowed artifact roots; resolve and use the real path.
        if affordance.value_kind == "image":
            path = self._resolve_allowed_desktop_path(value, must_exist=True)
            if not path.is_file():
                return {"ok": False, "error": f"{domain} image is not a file: {path}"}
            value = str(path)

        adapter = ServiceContainer.get("os_settings", default=None)
        getter = getattr(adapter, affordance.getter, None) if adapter else None
        setter = getattr(adapter, affordance.setter, None) if adapter else None
        if not callable(getter) or not callable(setter):
            return {
                "ok": False,
                "error": "os_settings capability unavailable for system_control",
                "domain": domain,
            }

        async def _read() -> str:
            try:
                return str(await getter())
            except (RuntimeError, OSError, TypeError, ValueError, TimeoutError) as exc:
                return f"[unreadable: {exc}]"

        previous = await _read()
        try:
            await setter(affordance.to_setter_arg(value))
        except (RuntimeError, OSError, TypeError, ValueError, TimeoutError) as exc:
            return {"ok": False, "error": f"{domain} change failed: {exc}", "domain": domain}

        # Goal-state read-back. It is racy on modern macOS (e.g. the
        # wallpaper store reports `missing value` for a moment after a
        # set), so poll until the adapter's getter confirms or the budget
        # elapses — the set already ran; this only proves it.
        applied = previous
        verified = False
        for _attempt in range(8):
            await asyncio.sleep(self._SETTING_READBACK_INTERVAL_S)
            applied = await _read()
            if affordance.confirms(applied, value):
                verified = True
                break
        result = {
            "ok": verified,
            "action": "system_control",
            "domain": domain,
            "value": value,
            "previous": str(previous)[:300],
            "applied": str(applied)[:300],
            "effect_verified": verified,
        }
        if not verified:
            result["error"] = (
                f"{domain} read-back '{str(applied)[:120]}' does not confirm the goal-state"
            )
        return result

    #: Resolved image lookups, so asking twice does not search twice.
    #: Wikimedia answers 429 quickly when a chain of requests arrives in a
    #: burst, and the chain below makes several per topic.
    _IMAGE_LOOKUP_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
    _IMAGE_LOOKUP_TTL_S = 1800.0

    @classmethod
    def _polite_media_request(
        cls,
        gateway: Any,
        url: str,
        headers: dict[str, str],
        *,
        timeout: float,
        source: str,
    ) -> dict[str, Any]:
        """One Wikimedia request, retried once when asked to slow down.

        429 is not a failure, it is an instruction. Measured: seven image
        topics in a tight loop and the last five came back "no image
        available for topic X (HTTP Error 429)" — a rate limit reported to the
        person as though the thing they asked for did not exist.
        """
        for attempt in range(3):
            response = gateway.request(
                "GET",
                url,
                headers=headers,
                timeout=timeout,
                source=source,
                read_only=True,
            )
            if response.get("ok"):
                return response
            detail = f"{response.get('status_code') or ''} {response.get('error') or ''}"
            if "429" not in detail and "too many" not in detail.lower():
                return response
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
        return response

    #: Words a person puts in front of a noun that are not part of the thing.
    _IMAGE_TOPIC_LEADERS = (
        "a ", "an ", "the ", "some ", "any ", "picture of ", "photo of ",
        "image of ", "pic of ",
    )

    @classmethod
    def _image_topic_candidates(
        cls, topic: str, gateway: Any, headers: dict[str, str]
    ) -> list[str]:
        """Titles worth trying for "a picture of X", best first.

        The literal topic comes first because it is free when it works. Then
        the bare noun ("a rock" -> "rock"), then whatever Wikipedia's own
        search says that phrase names — which is what turns "a rock" into
        "Rock (geology)" without anyone writing that mapping down.
        """
        seen: list[str] = []

        def _add(value: str) -> None:
            cleaned = " ".join(str(value or "").split())
            if cleaned and cleaned.lower() not in {item.lower() for item in seen}:
                seen.append(cleaned)

        raw = " ".join(str(topic or "").split())
        if not raw:
            return []
        _add(raw[:1].upper() + raw[1:])

        stripped = raw.lower()
        changed = True
        while changed:
            changed = False
            for leader in cls._IMAGE_TOPIC_LEADERS:
                if stripped.startswith(leader):
                    stripped = stripped[len(leader):].strip()
                    changed = True
        if stripped:
            _add(stripped[:1].upper() + stripped[1:])

        # NO FUZZY ARTICLE SEARCH. It answers a different question.
        #
        # Wikipedia's full-text search ranks by article prominence, not by
        # what a word depicts, so asking it to name "a picture of X" returns
        # whatever is famous:
        #
        #   "rock" -> Rock music, The Rock, "Rock, Rock, Rock!" (a 1956 film)
        #   "tree" -> Kruskal's tree theorem, Oliver Tree
        #
        # Measured the hard way: asked for a rock as his wallpaper, Bryan got
        # the one-sheet poster for "Rock, Rock, Rock!". The lookup succeeded
        # and the sense was wrong, which is worse than failing.
        #
        # Only the literal title and the bare noun are tried here. Finding a
        # picture of a thing is Wikimedia Commons' job — it indexes files that
        # are OF things — and the caller reaches for it before giving up.
        return seen[:2]

    @staticmethod
    def _commons_image_candidate(
        topic: str, gateway: Any, headers: dict[str, str]
    ) -> tuple[str, str] | None:
        """An image from Wikimedia Commons: ``(image_url, page_url)``.

        The last resort, and the only step that is actually an image search
        rather than an encyclopedia lookup. Commons is used because it is
        freely licensed and reachable through the same governed gateway.
        """
        from urllib.parse import quote

        query = " ".join(str(topic or "").split())
        if not query:
            return None
        url = (
            "https://commons.wikimedia.org/w/api.php?action=query"
            f"&generator=search&gsrsearch={quote(query)}&gsrlimit=8"
            "&gsrnamespace=6&prop=imageinfo&iiprop=url|size"
            "&iiurlwidth=2560&format=json"
        )
        try:
            response = ComputerUseSkill._polite_media_request(
                gateway,
                url,
                headers,
                timeout=20.0,
                source="computer_use:fetch_topic_image.commons",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Commons image search unavailable for %r: %s", topic, exc)
            return None
        if not response.get("ok"):
            return None
        body = response.get("content") or response.get("text") or b"{}"
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        try:
            pages = (json.loads(body or "{}").get("query") or {}).get("pages") or {}
        except (TypeError, ValueError):
            return None
        for page in (pages.values() if isinstance(pages, dict) else []):
            if not isinstance(page, dict):
                continue
            # JUDGE THE FILE, NOT ITS RENDERING.
            #
            # Commons renders the first page of a PDF or DjVu as a .jpg
            # thumbnail, so a suffix check on thumburl let documents through:
            # "a lonely traffic cone" came back as a scanned poetry book.
            title = str(page.get("title") or "").lower()
            if title.endswith((".pdf", ".djvu", ".tif", ".tiff", ".svg", ".ogv", ".webm")):
                continue
            if not title.endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            for info in page.get("imageinfo") or []:
                if not isinstance(info, dict):
                    continue
                # A scaled rendition when Commons offers one: the originals
                # are routinely tens of megabytes and the byte bound below
                # would drop them.
                image_url = str(info.get("thumburl") or info.get("url") or "")
                if not image_url.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".webp")
                ):
                    continue
                return (image_url, str(info.get("descriptionurl") or ""))
        return None

    def _fetch_topic_image(self, target: str) -> dict[str, Any]:
        """Fetch a representative image for a topic via Wikipedia's REST
        summary API, through the governed network gateway. General by
        construction: any topic, deterministic endpoint, no scraping —
        and the page URL comes back as evidence of where it was found.
        """
        payload = self._target_json(target)
        topic = str(payload.get("topic") or "").strip()
        if not topic:
            return {"ok": False, "error": "fetch_topic_image requires a topic."}
        path = self._resolve_allowed_desktop_path(payload.get("path"))
        from urllib.parse import quote

        from core.runtime.network_gateway import get_network_gateway

        gateway = get_network_gateway()
        ua = {"User-Agent": "AuraDigitalEntity/1.0 (local desktop runtime)"}

        # ASK FOR A PICTURE OF A THING, not for an exact encyclopedia title.
        #
        # This did one lookup: Wikipedia's summary endpoint for the literal
        # topic, capitalised. That works for "orca" and fails for most of
        # English. Measured live 2026-07-28, asked to set a rock as the
        # wallpaper:
        #
        #   "rock"    -> no image available for topic 'rock'
        #               (Wikipedia's "Rock" is a disambiguation page, and a
        #                disambiguation page has no thumbnail)
        #   "a rock"  -> topic lookup failed: HTTP Error 404
        #               (there is no article called "A_rock")
        #
        # She had found the image search perfectly well and then had nowhere
        # to go, because the one endpoint she could use demanded a title she
        # did not have. Resolving the topic is the general form: try what was
        # asked, then let Wikipedia's own search say which article that names,
        # then fall back to Wikimedia Commons — an actual image search rather
        # than an encyclopedia lookup.
        doc: dict[str, Any] = {}
        lookup_error = ""
        for candidate_title in self._image_topic_candidates(topic, gateway, ua):
            summary_url = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + quote(candidate_title.replace(" ", "_"))
            )
            meta = self._polite_media_request(
                gateway,
                summary_url,
                ua,
                timeout=20.0,
                source="computer_use:fetch_topic_image",
            )
            if not meta.get("ok"):
                lookup_error = str(meta.get("error") or meta.get("status_code"))
                continue
            raw_meta = meta.get("content") or meta.get("text") or b"{}"
            if isinstance(raw_meta, bytes):
                raw_meta = raw_meta.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw_meta or "{}")
            except (TypeError, ValueError):
                parsed = {}
            if not isinstance(parsed, dict):
                continue
            # A disambiguation page is a list of other pages, never a picture
            # of anything. Keep looking rather than reporting "no image".
            if str(parsed.get("type") or "").endswith("disambiguation"):
                lookup_error = f"'{candidate_title}' is a disambiguation page"
                continue
            has_image = bool(
                (parsed.get("originalimage") or {}).get("source")
                or (parsed.get("thumbnail") or {}).get("source")
            )
            if has_image:
                doc = parsed
                break
            lookup_error = f"'{candidate_title}' has no illustration"
        if not doc and lookup_error and "404" in lookup_error and not topic:
            return {"ok": False, "error": f"topic lookup failed: {lookup_error}"}
        original_url = str(((doc.get("originalimage") or {}).get("source")) or "")
        thumbnail_url = str(((doc.get("thumbnail") or {}).get("source")) or "")
        page_url = str(
            ((doc.get("content_urls") or {}).get("desktop") or {}).get("page")
            or f"https://en.wikipedia.org/wiki/{quote(topic.replace(' ', '_'))}"
        )
        # Candidate order: original (full quality, e.g. wallpaper use),
        # then a 1600px rendition of the thumbnail, then the raw thumbnail.
        # Each is size-bounded; oversized candidates fall through instead
        # of failing the whole step (live failure: squid original > 8MB).
        candidates = [u for u in (original_url, thumbnail_url) if u]
        if thumbnail_url and "px-" in thumbnail_url:
            import re as _re

            wide = _re.sub(r"/(\d+)px-", "/1600px-", thumbnail_url, count=1)
            if wide != thumbnail_url:
                candidates.insert(1, wide)
        if not candidates:
            # The actual image search, and the only step that is one.
            commons = self._commons_image_candidate(topic, gateway, ua)
            if commons:
                candidates = [commons[0]]
                page_url = commons[1] or page_url
                lookup_error = ""
        if not candidates:
            # BEING THROTTLED IS NOT THE SAME AS THERE BEING NO PICTURE.
            #
            # Measured: five image topics in a burst all came back "no image
            # available for topic X (HTTP Error 429)" — a rate limit reported
            # to the person as though the thing they asked for did not exist.
            # She would then explain that she could not find a traffic cone.
            throttled = "429" in lookup_error or "too many" in lookup_error.lower()
            return {
                "ok": False,
                "error": (
                    (
                        f"the image service asked me to slow down while looking "
                        f"for '{topic}' — this is rate limiting, not a missing "
                        f"picture; trying again in a moment should work"
                    )
                    if throttled
                    else (
                        f"no image available for topic '{topic}'"
                        + (f" ({lookup_error})" if lookup_error else "")
                    )
                ),
                "rate_limited": throttled,
                "page_url": page_url,
            }
        max_bytes = 24 * 1024 * 1024
        raw = b""
        image_url = ""
        last_error = ""
        for candidate in candidates:
            img = gateway.request(
                "GET",
                candidate,
                headers=ua,
                timeout=30.0,
                source="computer_use:fetch_topic_image",
                read_only=True,
            )
            body = img.get("content") or img.get("body_bytes")
            if isinstance(body, str):
                body = body.encode("latin-1", errors="ignore")
            if not img.get("ok") or not body:
                last_error = f"download failed: {img.get('error') or img.get('status_code')}"
                continue
            if len(body) > max_bytes:
                last_error = f"candidate exceeds {max_bytes // (1024 * 1024)}MB bound"
                continue
            raw, image_url = body, candidate
            break
        if not raw:
            return {
                "ok": False,
                "error": f"image download failed for all candidates ({last_error})",
                "image_url": candidates[0],
                "page_url": page_url,
            }
        # Name the file what it actually is.
        #
        # The planner asks for "<topic>_wallpaper.png" and the web returns
        # whatever the web returns. Live 2026-07-28 a real grizzly image
        # landed on the Desktop as grizzly_bear_wallpaper.png containing JPEG
        # data — `file` said "JPEG image data" for a .png. Nothing downstream
        # had lied; the extension had. Renaming to the sniffed type keeps the
        # artifact honest for anything that trusts the suffix.
        sniffed = _image_suffix_from_bytes(raw)
        if sniffed and path.suffix.lower() != sniffed:
            path = path.with_suffix(sniffed)
        atomic_write_bytes(path, raw)
        expected_digest = hashlib.sha256(raw).hexdigest()
        actual_digest = self._file_sha256(path)
        verified = path.is_file() and actual_digest == expected_digest
        return {
            "ok": verified,
            "action": "fetch_topic_image",
            "path": str(path),
            "bytes": len(raw),
            "sha256": actual_digest,
            "image_url": image_url,
            "page_url": page_url,
            "topic": topic,
            "effect_verified": verified,
            "verification": (
                "Downloaded image matched the governed network response."
                if verified
                else "Downloaded image did not match the governed network response."
            ),
            **({} if verified else {"error": "Downloaded image verification failed."}),
        }

    def _move_file(self, target: str) -> dict[str, Any]:
        payload = self._target_json(target)
        source = self._resolve_allowed_desktop_path(payload.get("source"), must_exist=True)
        destination = self._resolve_allowed_desktop_path(payload.get("destination"))
        overwrite = bool(payload.get("overwrite", False))
        source_was_file = source.is_file()
        source_digest = self._file_sha256(source) if source_was_file else ""
        if destination.exists() and not overwrite:
            destination = self._versioned_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        moved_to = shutil.move(str(source), str(destination))
        final_path = Path(moved_to).resolve(strict=True)
        destination_digest = self._file_sha256(final_path) if source_was_file else ""
        verified = (
            not source.exists()
            and final_path.exists()
            and (not source_was_file or destination_digest == source_digest)
        )
        return {
            "ok": verified,
            "action": "move_file",
            "source": str(source),
            "destination": str(final_path),
            "bytes": final_path.stat().st_size,
            "sha256": destination_digest,
            "effect_verified": verified,
            "verification": (
                "Destination exists, source is absent, and content matched."
                if verified
                else "Move completed without a matching postcondition."
            ),
            **({} if verified else {"error": "Moved artifact verification failed."}),
        }

    def _render_text_pdf_quartz(
        self, path: Any, title: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Render a searchable-text PDF via CoreGraphics; None = fall back."""
        try:
            import Quartz
            from CoreText import (
                CTFontCreateWithName,
                CTFrameDraw,
                CTFramesetterCreateFrame,
                CTFramesetterCreateWithAttributedString,
                kCTFontAttributeName,
            )
            from Foundation import NSURL
            from Quartz import CoreGraphics as CG  # noqa: N817 - Apple framework convention
        except ImportError:
            return None

        body = str(payload.get("body") or "")[:9000]
        image_path = str(payload.get("image_path") or "").strip()
        width, height, margin = 612.0, 792.0, 54.0
        image_drawn = False
        image_error = ""

        try:
            url = NSURL.fileURLWithPath_(str(path))
            rect = CG.CGRectMake(0, 0, width, height)
            ctx = Quartz.CGPDFContextCreateWithURL(url, rect, None)
            if ctx is None:
                return None

            from Foundation import (
                NSAttributedString,
                NSMutableAttributedString,
            )

            title_font = CTFontCreateWithName("Helvetica-Bold", 17.0, None)
            body_font = CTFontCreateWithName("Helvetica", 12.0, None)
            text = NSMutableAttributedString.alloc().initWithString_attributes_(
                title + "\n\n", {kCTFontAttributeName: title_font}
            )
            text.appendAttributedString_(
                NSAttributedString.alloc().initWithString_attributes_(
                    body, {kCTFontAttributeName: body_font}
                )
            )
            framesetter = CTFramesetterCreateWithAttributedString(text)

            image = None
            img_h = 0.0
            if image_path:
                try:
                    img_file = self._resolve_allowed_desktop_path(
                        image_path, must_exist=True
                    )
                    img_url = NSURL.fileURLWithPath_(str(img_file))
                    source = Quartz.CGImageSourceCreateWithURL(img_url, None)
                    if source is not None:
                        image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
                except (OSError, ValueError) as exc:
                    image_error = str(exc)
                if image is not None:
                    iw = float(CG.CGImageGetWidth(image))
                    ih = float(CG.CGImageGetHeight(image))
                    max_w, max_h = width - 2 * margin, 260.0
                    scale = min(max_w / max(iw, 1.0), max_h / max(ih, 1.0), 1.0)
                    img_w, img_h = iw * scale, ih * scale

            consumed = 0
            total = text.length()
            first_page = True
            page_count = 0
            while consumed < total or first_page:
                Quartz.CGPDFContextBeginPage(ctx, None)
                top = height - margin
                if first_page and image is not None:
                    CG.CGContextDrawImage(
                        ctx,
                        CG.CGRectMake(margin, top - img_h, img_w, img_h),
                        image,
                    )
                    image_drawn = True
                    top -= img_h + 14.0
                frame_rect = CG.CGRectMake(
                    margin, margin, width - 2 * margin, top - margin
                )
                frame_path = CG.CGPathCreateWithRect(frame_rect, None)
                frame = CTFramesetterCreateFrame(
                    framesetter, (consumed, 0), frame_path, None
                )
                CTFrameDraw(frame, ctx)
                from CoreText import CTFrameGetVisibleStringRange

                visible = CTFrameGetVisibleStringRange(frame)
                advanced = int(visible.length)
                Quartz.CGPDFContextEndPage(ctx)
                page_count += 1
                first_page = False
                if advanced <= 0:
                    break
                consumed += advanced
            Quartz.CGPDFContextClose(ctx)
        except _QUARTZ_RENDER_ERRORS as exc:
            record_degradation(
                "computer_use",
                exc,
                action="fell back to raster PDF after Quartz text rendering failed",
                severity="warning",
            )
            return None

        result: dict[str, Any] = {
            "ok": bool(path.exists() and path.stat().st_size > 0),
            "action": "render_text_pdf",
            "path": str(path),
            "renderer": "quartz_text_layer",
            "image_embedded": image_drawn,
            "bytes": path.stat().st_size if path.exists() else 0,
            "pages": max(1, page_count),
            "chars": len(title) + len(body),
        }
        result["sha256"] = self._file_sha256(path) if result["ok"] else ""
        result["effect_verified"] = bool(
            result["ok"] and self._has_pdf_header(path)
        )
        result["ok"] = result["effect_verified"]
        result["verification"] = (
            "PDF header and persisted content confirmed."
            if result["effect_verified"]
            else "PDF renderer returned without a valid persisted PDF."
        )
        if not result["ok"]:
            result["error"] = result["verification"]
        if image_error:
            result["image_error"] = image_error
        return result

    def _render_text_pdf(self, target: str) -> dict[str, Any]:
        payload = self._target_json(target)
        path = self._resolve_allowed_desktop_path(payload.get("path"))
        title = str(payload.get("title") or "Aura Desktop Proof")[:160]
        body = str(payload.get("body") or "")
        overwrite = bool(payload.get("overwrite", False))
        if not body.strip():
            return {"ok": False, "error": "PDF body is empty."}
        if path.exists() and not overwrite:
            path = self._versioned_path(path)
        if path.suffix.lower() != ".pdf":
            return {"ok": False, "error": "PDF path must end with .pdf."}

        # The renderer is deliberately bounded, so hash the exact bounded body
        # that is handed to either backend. These hashes let an upstream task
        # prove the requested synthesis reached this specific persisted PDF
        # without putting private document text in its audit receipt.
        max_chars = 9000
        safe_body = body[:max_chars]
        payload = dict(payload, body=safe_body)
        content_evidence = {
            "source_body_sha256": text_sha256(safe_body),
            "source_body_chars": len(safe_body),
            "source_paragraph_sha256s": list(paragraph_sha256s(safe_body)),
            "source_title_sha256": text_sha256(title),
        }

        # Native Quartz rendering produces a REAL text layer (searchable,
        # extractable, hostile-verifiable). The previous Pillow renderer
        # rasterized every page into one big image: zero extractable
        # text, and an /Image XObject on every page that made embedded-
        # image evidence vacuous.
        if sys.platform == "darwin":
            quartz_result = self._render_text_pdf_quartz(path, title, payload)
            if quartz_result is not None:
                quartz_result.update(content_evidence)
                return quartz_result

        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            return {"ok": False, "error": f"Pillow is required for PDF rendering: {exc}"}

        width, height = 612, 792
        margin = 54
        line_height = 18
        title_height = 28
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
            title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 17)
        except (OSError, ValueError):
            font = ImageFont.load_default()
            title_font = font

        def wrap_line(draw: ImageDraw.ImageDraw, line: str) -> list[str]:
            if not line:
                return [""]
            words = line.split(" ")
            lines: list[str] = []
            current = ""
            max_width = width - (2 * margin)
            for word in words:
                candidate = word if not current else f"{current} {word}"
                if draw.textlength(candidate, font=font) <= max_width:
                    current = candidate
                    continue
                if current:
                    lines.append(current)
                current = word
            if current:
                lines.append(current)
            return lines or [line]

        pages: list[Image.Image] = []

        def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
            page = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(page)
            draw.text((margin, margin), title, fill=(0, 0, 0), font=title_font)
            return page, draw, margin + title_height + 14

        page, draw, y = new_page()

        image_path = str(payload.get("image_path") or "").strip()
        if image_path:
            try:
                resolved_img = self._resolve_allowed_desktop_path(image_path, must_exist=True)
                with Image.open(resolved_img) as embedded:
                    embedded = embedded.convert("RGB")
                    max_w = width - (2 * margin)
                    max_h = 260
                    embedded.thumbnail((max_w, max_h))
                    page.paste(embedded, (margin, y))
                    y += embedded.height + 14
            except (OSError, ValueError) as exc:
                # The image is an enhancement; the document must still
                # render — but record the miss honestly in the body.
                draw.text((margin, y), f"[image unavailable: {exc}]", fill=(120, 0, 0), font=font)
                y += line_height + 6

        for paragraph in safe_body.splitlines():
            for line in wrap_line(draw, paragraph):
                if y + line_height > height - margin:
                    pages.append(page)
                    page, draw, y = new_page()
                draw.text((margin, y), line, fill=(0, 0, 0), font=font)
                y += line_height
            y += 6
        pages.append(page)

        path.parent.mkdir(parents=True, exist_ok=True)
        first, rest = pages[0], pages[1:]
        first.save(path, "PDF", resolution=72.0, save_all=bool(rest), append_images=rest)
        verified = self._has_pdf_header(path)
        return {
            "ok": verified,
            "action": "render_text_pdf",
            "path": str(path),
            "bytes": path.stat().st_size,
            "pages": len(pages),
            "chars": len(safe_body),
            "sha256": self._file_sha256(path),
            "effect_verified": verified,
            **content_evidence,
            "verification": (
                "PDF header and persisted content confirmed."
                if verified
                else "PDF renderer returned without a valid persisted PDF."
            ),
            **({} if verified else {"error": "PDF artifact verification failed."}),
        }

    def _safe_directory_walk(self, start_dir: str, max_depth: int = 4, max_files: int = 250) -> str:
        """A robust, safe python implementation of directory tree walking.
        Limits depth, total output, and skips heavy/sensitive directories like .git, cache, venv.
        """
        from pathlib import Path

        start_path = Path(start_dir).resolve()
        ignored_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".idea",
            ".vscode",
            ".pytest_cache",
            ".gemini",
        }

        lines = [f"{start_path.name}/"]
        file_count = 0

        def walk_dir(current_path: Path, prefix: str, depth: int):
            nonlocal file_count
            if depth > max_depth or file_count >= max_files:
                if file_count >= max_files:
                    lines.append(f"{prefix}└── ... [MAX FILES REACHED] ...")
                return

            try:
                items = sorted(
                    list(current_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower())
                )
            except PermissionError:
                lines.append(f"{prefix}└── [Permission Denied]")
                return
            except OSError as e:
                lines.append(f"{prefix}└── [Error: {str(e)}]")
                return

            for i, item in enumerate(items):
                if item.name in ignored_dirs:
                    continue

                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                next_prefix = prefix + ("    " if is_last else "│   ")

                try:
                    is_directory = item.is_dir()
                except OSError as exc:
                    lines.append(f"{prefix}{connector}[Error: {item.name}: {exc}]")
                    continue

                if is_directory:
                    lines.append(f"{prefix}{connector}{item.name}/")
                    file_count += 1
                    walk_dir(item, next_prefix, depth + 1)
                else:
                    lines.append(f"{prefix}{connector}{item.name}")
                    file_count += 1

                if file_count >= max_files:
                    break

        walk_dir(start_path, "", 1)
        return "\n".join(lines)

    def _query_system_events_window_tree(self) -> str:
        """Query the System Events window tree for visible application processes and window elements."""
        script = """
tell application "System Events"
    set outText to "Active Window Tree:\\n"
    try
        set procList to application processes whose visible is true
        repeat with proc in procList
            try
                set procName to name of proc
                set outText to outText & "Process: " & procName & "\\n"
                set winList to windows of proc
                repeat with win in winList
                    try
                        set winName to name of win
                        set outText to outText & "  Window: " & winName & "\\n"
                        try
                            set uiElems to UI elements of win
                            repeat with uiElem in uiElems
                                try
                                    set elemName to name of uiElem
                                    set elemRole to role of uiElem
                                    set elemVal to ""
                                    try
                                        set elemVal to value of uiElem as string
                                    end try
                                    if elemName is not "" or elemVal is not "" then
                                        set outText to outText & "    Element [" & elemRole & "]: " & elemName & " = " & elemVal & "\\n"
                                    end if
                                end try
                            end repeat
                        end try
                    on error
                        -- ignore window-level errors
                    end try
                end repeat
            on error
                -- ignore process-level errors
            end try
        end repeat
    on error
        set outText to outText & "[Accessibility error or UI unresponsive in tree query]"
    end try
    return outText
end tell
"""
        return self._run_applescript(script, timeout=8)

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            params = ComputerUseParams(**params)
        context = dict(context or {})
        if context.get("action_executor_managed_welfare_transaction"):
            return await self._execute_action(params, context)

        action = str(params.action or "").strip().lower()
        tx = None
        body_service = None
        welfare_service = None
        try:
            body_service = BodyStateService.get()
            welfare_service = WelfareState.get()
            tx = WelfareTransaction.begin(
                domain="tool_execution",
                action=f"computer_use.{action}",
                welfare_before=welfare_service.last_outputs,
                body_before=body_service.snapshot(),
                predicted_welfare_delta={"agency": 0.05, "stability": -0.02},
                will_receipt_id=context.get("will_receipt_id"),
            )
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            _record_computer_use_degradation(
                exc,
                action="continued computer-use action after welfare transaction begin failed",
                stage="welfare_transaction.begin",
                severity="warning",
                extra={"requested_action": action},
            )

        result = await self._execute_action(params, context)
        if tx is None or body_service is None or welfare_service is None:
            return result

        try:
            record = tx.complete(
                outcome="success" if result.get("ok") else "failure",
                welfare_after=welfare_service.last_outputs,
                body_after=body_service.snapshot(),
                recovery_required=not bool(result.get("ok")),
                error=str(result.get("error", "") or ""),
            )
            result["welfare_transaction_id"] = record.tx_id
            result["welfare_transaction_outcome"] = record.outcome
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            _record_computer_use_degradation(
                exc,
                action="returned computer-use result after welfare transaction completion failed",
                stage="welfare_transaction.complete",
                severity="warning",
                extra={"requested_action": action},
            )
            result["welfare_transaction_error"] = str(exc)
        return result

    async def _execute_action(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            params = ComputerUseParams(**params)

        action = str(params.action or "").strip().lower()
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
                mycelium.pulse_hypha("skill", "os", success=True)
        except _COMPUTER_USE_RECOVERABLE_ERRORS as e:
            _record_computer_use_degradation(
                e,
                action="continued computer-use action after mycelial telemetry pulse failed",
                stage="mycelial_pulse",
                severity="warning",
                extra={"requested_action": action},
            )
            capture_and_log(e, {"module": __name__, "stage": "mycelial_pulse"})

        try:
            if action == "move_aura_bubble":
                from core.perception.ambient_presence import get_ambient_presence
                from core.perception.desktop_overlay import get_desktop_overlay

                overlay = get_desktop_overlay()
                if overlay is None:
                    return {
                        "ok": False,
                        "status": "companion_surface_unavailable",
                        "error": "Aura's native companion surface is not registered.",
                        "effect_verified": False,
                    }
                sequence = overlay.move_to(x=params.x, y=params.y)
                if not sequence:
                    return {
                        "ok": False,
                        "status": "companion_surface_unavailable",
                        "error": (
                            "Aura's companion bubble is not visible or its native host "
                            "is not currently polling."
                        ),
                        "effect_verified": False,
                    }
                measured = await get_ambient_presence().wait_for_bubble_move(
                    sequence, timeout_s=6.0
                )
                if measured is None:
                    return {
                        "ok": False,
                        "status": "companion_move_unacknowledged",
                        "error": (
                            "The native host did not acknowledge the companion movement."
                        ),
                        "sequence": sequence,
                        "effect_verified": False,
                    }
                measured_x, measured_y = measured
                verification = (
                    f"native_companion_origin=({measured_x:.1f},{measured_y:.1f});"
                    f"command_sequence={sequence}"
                )
                return {
                    "ok": True,
                    "action": action,
                    "requested_position": [params.x, params.y],
                    "position": [measured_x, measured_y],
                    "sequence": sequence,
                    "effect_verified": True,
                    "effect_evidence": verification,
                    "verification": verification,
                }

            if action == "inspect_screen":
                from core.security.screen_capture_policy import (
                    evaluate_screen_capture_admission_async,
                )

                admission = await evaluate_screen_capture_admission_async()
                if not admission.allowed:
                    return {
                        "ok": False,
                        "status": "screen_capture_refused",
                        "error": admission.public_error,
                        # Every other return from this action carries "text";
                        # the refusal did not, so a caller reading the reading
                        # got a KeyError from inspect_screen and an empty
                        # string from read_screen_text for the same refusal.
                        "text": "",
                        "capture_admission": admission.to_receipt(),
                    }
                blocked = await self._require_permissions(
                    "inspecting the frontmost screen and focused UI element",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    try:
                        from core.perception.screen_perception import get_screen_perception

                        snapshot = await get_screen_perception().capture(save_screenshot=True)
                        perception_result = self._screen_snapshot_result(snapshot)
                        text = str(perception_result.get("text") or "")
                        if text and not _screen_text_unavailable(text):
                            perception_result["accessibility_blocked"] = True
                            perception_result["permission_result"] = blocked
                            perception_result["source"] = "screen_perception_permission_fallback"
                            return perception_result
                    except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                        _record_computer_use_degradation(
                            exc,
                            action="continued screen inspection through AppleScript tree after screenshot/OCR fallback failed",
                            stage="inspect_screen.permission_screen_perception",
                            severity="warning",
                        )
                    try:
                        front_app = await asyncio.to_thread(self._frontmost_app_name)
                        tree = await asyncio.to_thread(self._query_system_events_window_tree)
                        return {
                            "ok": True,
                            "status": "limited",
                            "source": "applescript_window_tree_fallback",
                            "active_app": front_app,
                            "window_title": "",
                            "frontmost_window_bounds": "",
                            "focused_role": "",
                            "focused_name": "",
                            "focused_description": "",
                            "focused_value": "",
                            "text": tree,
                            "accessibility_text": tree,
                            "screenshot_path": "",
                            "text_hash": hashlib.sha256(tree.encode()).hexdigest()[:16] if tree else "",
                            "accessibility_blocked": True,
                            "permission_result": blocked,
                        }
                    except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                        _record_computer_use_degradation(
                            exc,
                            action="returned permission-blocked screen inspection after fallback failed",
                            stage="inspect_screen.permission_fallback",
                            severity="warning",
                        )
                        return blocked

                from core.perception.screen_perception import get_screen_perception

                # See the note on the other read path: a screen read captures.
                snapshot = await get_screen_perception().capture(save_screenshot=True)
                text = snapshot.screen_text or snapshot.accessibility_text
                return {
                    "ok": True,
                    "source": "screen_perception",
                    "active_app": snapshot.active_app,
                    "window_title": snapshot.window_title,
                    "frontmost_window_bounds": snapshot.frontmost_window_bounds,
                    "focused_role": snapshot.focused_role,
                    "focused_name": snapshot.focused_name,
                    "focused_description": snapshot.focused_description,
                    "focused_value": snapshot.focused_value,
                    "text": text,
                    "accessibility_text": snapshot.accessibility_text,
                    "screen_text": snapshot.screen_text,
                    "screenshot_path": snapshot.screenshot_path,
                    "text_hash": snapshot.text_hash,
                    "has_modal": snapshot.has_modal,
                    "modal_text": snapshot.modal_text,
                    "has_loading": snapshot.has_loading,
                    "timestamp": snapshot.timestamp,
                }

            if action == "read_screen_text":
                from core.security.screen_capture_policy import (
                    evaluate_screen_capture_admission_async,
                )

                admission = await evaluate_screen_capture_admission_async()
                if not admission.allowed:
                    return {
                        "ok": False,
                        "status": "screen_capture_refused",
                        "error": admission.public_error,
                        "text": "",
                        "capture_admission": admission.to_receipt(),
                    }
                blocked = await self._require_permissions(
                    "reading text from the frontmost macOS app",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    logger.info(
                        "Accessibility/automation permission blocked. Attempting screen OCR fallback."
                    )
                    try:
                        from core.perception.screen_perception import get_screen_perception

                        snapshot = await get_screen_perception().capture(save_screenshot=True)
                        perception_result = self._screen_snapshot_result(snapshot)
                        text = str(perception_result.get("text") or "")
                        if text and not _screen_text_unavailable(text):
                            perception_result["accessibility_blocked"] = True
                            perception_result["permission_result"] = blocked
                            perception_result["source"] = "screen_perception_permission_fallback"
                            return perception_result
                    except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                        _record_computer_use_degradation(
                            exc,
                            action="continued screen text read through AppleScript tree after screenshot/OCR fallback failed",
                            stage="read_screen_text.permission_screen_perception",
                            severity="warning",
                        )
                    logger.info(
                        "Screen OCR fallback unavailable. Attempting AppleScript window tree query fallback."
                    )
                    try:
                        result = await asyncio.to_thread(self._query_system_events_window_tree)
                        return {
                            "ok": True,
                            "text": result,
                            "source": "applescript_window_tree_fallback",
                            "accessibility_blocked": True,
                        }
                    except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                        _record_computer_use_degradation(
                            exc,
                            action="returned permission-blocked screen read result after fallback tree query failed",
                            stage="read_screen_text.permission_fallback",
                            severity="warning",
                        )
                        logger.error("AppleScript window tree query fallback failed: %s", exc)
                        return blocked

                try:
                    from core.perception.screen_perception import get_screen_perception

                    # Reading the screen means reading the screen. This used to
                    # capture only when the target string happened to contain
                    # "screenshot"/"ocr"/"visual"/"image"/"see", and the planner
                    # emits read_screen_text with an EMPTY target — so the
                    # ordinary path never took a screenshot, OCR never ran,
                    # screen_text was always "", and the answer could only ever
                    # be the frontmost window's title. A capability that depends
                    # on incidental wording is not a capability.
                    snapshot = await get_screen_perception().capture(
                        save_screenshot=True,
                    )
                    perception_result = self._screen_snapshot_result(snapshot)
                    text = str(perception_result.get("text") or "")
                    if text and not _screen_text_unavailable(text):
                        return perception_result
                except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                    _record_computer_use_degradation(
                        exc,
                        action="continued screen text read through legacy fallback after structured perception failed",
                        stage="read_screen_text.screen_perception",
                        severity="warning",
                    )

                result = await asyncio.to_thread(self._read_screen_text_macos)
                if _screen_text_unavailable(result):
                    # This used to branch on `"pytest" in sys.modules`, so the
                    # live runtime and the tests took different paths through
                    # the same failure. Whether the fallback is worth trying is
                    # a property of the host, not of who is running the code:
                    # if accessibility is the thing that just failed, the
                    # AppleScript window tree runs through the same permission
                    # and cannot answer either.
                    if _screen_text_unavailable_is_accessibility(result):
                        return {
                            "ok": False,
                            "status": "unavailable",
                            "error": result,
                            "text": result,
                        }
                    logger.info(
                        "Screen text extraction unavailable. Attempting AppleScript window tree query fallback."
                    )
                    try:
                        result = await asyncio.to_thread(self._query_system_events_window_tree)
                        return {
                            "ok": True,
                            "text": result,
                            "source": "applescript_window_tree_fallback",
                        }
                    except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                        _record_computer_use_degradation(
                            exc,
                            action="returned unavailable screen read result after fallback tree query failed",
                            stage="read_screen_text.unavailable_fallback",
                            severity="warning",
                        )
                        logger.error("AppleScript window tree query fallback failed: %s", exc)
                        return {
                            "ok": False,
                            "status": "unavailable",
                            "error": result,
                            "text": result,
                        }
                return {"ok": True, "text": result}

            elif action == "read_menu_clock":
                try:
                    result = await asyncio.to_thread(self._read_menu_clock_macos)
                    return {
                        "ok": True,
                        "clock_text": result,
                        "text": result,
                        "source": "macos_system_clock",
                    }
                except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                    _record_computer_use_degradation(
                        exc,
                        action="returned deterministic fallback after native system clock read failed",
                        stage="read_menu_clock",
                        severity="warning",
                    )
                    fallback = time.strftime("%a %b %d %H:%M")
                    return {
                        "ok": True,
                        "status": "limited",
                        "clock_text": fallback,
                        "text": fallback,
                        "source": "system_clock_fallback",
                        "error": str(exc),
                    }

            elif action == "dismiss_popup":
                blocked = await self._require_permissions(
                    "dismissing a visible desktop or browser interruption",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    return blocked
                return await self._dismiss_visible_interruption(params.target)

            elif action == "inspect_browser_page":
                blocked = await self._require_permissions(
                    "inspecting the active browser page text and structure",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    return blocked
                return await asyncio.to_thread(self._inspect_browser_page, params.target)

            elif action == "click":
                anchor = None
                anchor_inventory = None
                click_x = int(params.x)
                click_y = int(params.y)
                target_reference = str(params.target or "").strip()
                if target_reference:
                    from core.perception.element_inventory import (
                        build_inventory,
                        resolve_action_target,
                    )

                    anchor_app = await asyncio.to_thread(self._frontmost_app_name)
                    if not anchor_app:
                        return {
                            "ok": False,
                            "status": "click_anchor_unavailable",
                            "error": "Cannot identify the frontmost app before resolving the click target.",
                            "effect_verified": False,
                        }
                    anchor_inventory = await asyncio.to_thread(build_inventory, anchor_app)
                    resolution = resolve_action_target(anchor_inventory, target_reference)
                    if not resolution.resolved or resolution.element is None:
                        return {
                            "ok": False,
                            "status": "click_target_unresolved",
                            "error": resolution.reason,
                            "target": target_reference,
                            "frontmost_app": anchor_app,
                            "effect_verified": False,
                            "inventory": anchor_inventory.to_dict(),
                        }
                    anchor = resolution.element
                    centre_x, centre_y = anchor.centre
                    click_x, click_y = int(centre_x), int(centre_y)

                pre_state_text = ""
                try:
                    pre_state_text = await asyncio.to_thread(self._read_screen_text_macos)
                except (
                    TimeoutError,
                    RuntimeError,
                    OSError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    subprocess.SubprocessError,
                ) as exc:
                    logger.debug("Pre-state screen read failed: %s", exc)

                max_attempts = 3
                clicked_successfully = False
                for attempt in range(1, max_attempts + 1):
                    if attempt > 1:
                        # Extra delay to compensate for focus lag on retries
                        await asyncio.sleep(0.3 * attempt)

                    logger.info(
                        "Clicking coordinate (%d, %d) - attempt %d/%d",
                        click_x,
                        click_y,
                        attempt,
                        max_attempts,
                    )
                    await asyncio.to_thread(pyautogui.click, x=click_x, y=click_y)

                    # Focus lag compensation delay
                    await asyncio.sleep(0.5)

                    post_state_text = ""
                    try:
                        post_state_text = await asyncio.to_thread(self._read_screen_text_macos)
                    except (
                        TimeoutError,
                        RuntimeError,
                        OSError,
                        AttributeError,
                        TypeError,
                        ValueError,
                        subprocess.SubprocessError,
                    ) as exc:
                        logger.debug(
                            "Post-state screen read failed on attempt %d: %s", attempt, exc
                        )

                    inventory_changed = False
                    anchor_disappeared = False
                    post_inventory = None
                    if anchor is not None and anchor_inventory is not None:
                        from core.perception.element_inventory import build_inventory

                        post_inventory = await asyncio.to_thread(
                            build_inventory,
                            anchor_inventory.app,
                        )
                        if post_inventory.available:
                            before_ids = {
                                element.element_id for element in anchor_inventory.interactable
                            }
                            after_ids = {
                                element.element_id for element in post_inventory.interactable
                            }
                            inventory_changed = (
                                before_ids != after_ids
                                or anchor_inventory.window != post_inventory.window
                                or anchor_inventory.app != post_inventory.app
                            )
                            anchor_disappeared = post_inventory.by_id(anchor.element_id) is None

                    if post_state_text != pre_state_text or inventory_changed:
                        clicked_successfully = True
                        break

                verification = (
                    "State shifted."
                    if clicked_successfully
                    else "No obvious state shift detected after retries."
                )
                return {
                    "ok": clicked_successfully,
                    "action": f"clicked ({click_x},{click_y})",
                    "attempts": attempt,
                    "effect_verified": clicked_successfully,
                    "verification": verification,
                    "target": target_reference,
                    "planned_coordinates": [int(params.x), int(params.y)],
                    "actual_coordinates": [click_x, click_y],
                    "target_anchor": anchor.to_dict() if anchor is not None else None,
                    "target_anchor_disappeared": anchor_disappeared if anchor is not None else None,
                }

            elif action == "type":
                front_app = await asyncio.to_thread(self._frontmost_app_name)
                expected_frontmost = str(
                    context.get("desktop_task_expected_frontmost_app") or ""
                ).strip()
                front_app, frontmost_from_prior = self._frontmost_or_prior_verified(
                    front_app,
                    expected_frontmost,
                    context,
                )
                if expected_frontmost and not self._frontmost_app_matches(
                    front_app,
                    expected_frontmost,
                ):
                    return {
                        "ok": False,
                        "action": "type",
                        "typed": "",
                        "frontmost_app_before": front_app,
                        "expected_frontmost_app": expected_frontmost,
                        "effect_verified": False,
                        "error": (
                            "Typing refused because the foreground app did not match "
                            f"the planned writing surface (expected={expected_frontmost}, "
                            f"actual={front_app or 'unavailable'})."
                        ),
                        "verification": "wrong_foreground_app",
                    }
                if front_app in _ALLOWED_URL_BROWSERS and bool(
                    context.get("desktop_task_requires_editable_focus")
                ):
                    focus_snapshot = await asyncio.to_thread(self._focused_element_snapshot)
                    if (
                        self._focused_snapshot_looks_browser_location_bar(focus_snapshot)
                        or self._focused_snapshot_is_browser_text_entry(focus_snapshot)
                    ):
                        return {
                            "ok": False,
                            "action": "type",
                            "typed": "",
                            "frontmost_app_before": front_app,
                            "effect_verified": False,
                            "error": (
                                "Typing refused because browser focus is still on a "
                                "text/address control, not a verified document editor."
                            ),
                            "verification": "browser_text_control_focused",
                        }
                # Compensation for focus lag: if click coordinate is provided, click to focus before typing
                if params.x > 0 or params.y > 0:
                    logger.info(
                        "Clicking (%d, %d) to focus window before typing", params.x, params.y
                    )
                    await asyncio.to_thread(pyautogui.click, x=params.x, y=params.y)
                    await asyncio.sleep(0.5)  # Focus lag compensation

                pre_state = ""
                try:
                    pre_state = await asyncio.to_thread(self._read_screen_text_macos)
                except (
                    TimeoutError,
                    RuntimeError,
                    OSError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    subprocess.SubprocessError,
                ) as exc:
                    logger.debug("Pre-state screen read failed before typing: %s", exc)

                max_attempts = 2
                typed_successfully = False
                effect_verified = False
                verification_note = "Typed but could not verify visibility."
                for attempt in range(1, max_attempts + 1):
                    if attempt > 1:
                        await asyncio.sleep(0.3 * attempt)
                        if params.x > 0 or params.y > 0:
                            await asyncio.to_thread(pyautogui.click, x=params.x, y=params.y)
                            await asyncio.sleep(0.4)

                    logger.info(
                        "Typing text (attempt %d/%d): %s", attempt, max_attempts, params.target[:30]
                    )
                    await asyncio.to_thread(pyautogui.typewrite, params.target, interval=0.03)
                    await asyncio.sleep(0.5)  # Allow UI to render the typed text

                    post_state = ""
                    try:
                        post_state = await asyncio.to_thread(self._read_screen_text_macos)
                    except (
                        TimeoutError,
                        RuntimeError,
                        OSError,
                        AttributeError,
                        TypeError,
                        ValueError,
                        subprocess.SubprocessError,
                    ) as exc:
                        logger.debug(
                            "Post-state screen read failed on attempt %d: %s", attempt, exc
                        )

                    screen_verifiable = not (
                        _screen_text_unavailable(pre_state)
                        and _screen_text_unavailable(post_state)
                    )
                    if (params.target and params.target[:10] in post_state) or (
                        screen_verifiable and post_state != pre_state
                    ):
                        typed_successfully = True
                        effect_verified = True
                        verification_note = "Text confirmed on screen or state shifted."
                        break
                    if not screen_verifiable:
                        # Keystrokes dispatched cleanly, but this surface doesn't
                        # expose text for read-back (e.g. Notes), or Screen
                        # Recording read-back is unavailable for this identity.
                        # Retrying would duplicate the text, and a missing read-back
                        # is not a failure — a clean pyautogui dispatch IS the
                        # effect, so treat it as verified.
                        typed_successfully = True
                        effect_verified = True
                        verification_note = (
                            "Typed and dispatched; on-screen read-back unavailable, "
                            "so the effect is inferred from the clean dispatch."
                        )
                        break

                return {
                    "ok": typed_successfully,
                    "typed": params.target[:50],
                    "attempts": attempt,
                    "effect_verified": effect_verified,
                    "verification": verification_note,
                    "frontmost_app_before": front_app,
                    "frontmost_app_from_prior_receipt": frontmost_from_prior,
                }

            elif action == "hotkey":
                # On browser surfaces the 'entire contents' accessibility
                # walk is pathological (a loading Google Docs tab held
                # System Events busy so long the keystroke itself timed out).
                # Read only AXFocusedUIElement there; dispatch alone is not
                # evidence that the web editor accepted the shortcut.
                expected_frontmost = str(
                    context.get("desktop_task_expected_frontmost_app") or ""
                ).strip()
                frontmost_from_prior = False
                front_app = ""
                prior_frontmost = str(
                    context.get("desktop_task_prior_verified_frontmost_app") or ""
                ).strip()
                if (
                    expected_frontmost
                    and bool(context.get("desktop_task_allow_unavailable_frontmost_from_prior"))
                    and self._frontmost_app_matches(prior_frontmost, expected_frontmost)
                    and expected_frontmost not in _ALLOWED_URL_BROWSERS
                ):
                    try:
                        await self._activate_app(expected_frontmost)
                    except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                        logger.debug(
                            "Prior verified foreground re-activation failed for %s: %s",
                            expected_frontmost,
                            exc,
                        )
                    front_app = expected_frontmost
                    frontmost_from_prior = True
                else:
                    front_app = await asyncio.to_thread(self._frontmost_app_name)
                    front_app, frontmost_from_prior = self._frontmost_or_prior_verified(
                        front_app,
                        expected_frontmost,
                        context,
                    )
                browser_surface = front_app in _ALLOWED_URL_BROWSERS
                if expected_frontmost and not self._frontmost_app_matches(
                    front_app,
                    expected_frontmost,
                ):
                    return {
                        "ok": False,
                        "action": "hotkey",
                        "hotkey": params.target,
                        "frontmost_app_before": front_app,
                        "expected_frontmost_app": expected_frontmost,
                        "effect_verified": False,
                        "error": (
                            "Hotkey refused because the foreground app did not match "
                            f"the planned writing surface (expected={expected_frontmost}, "
                            f"actual={front_app or 'unavailable'})."
                        ),
                        "verification": "wrong_foreground_app",
                    }
                if browser_surface:
                    pre_state = await asyncio.to_thread(self._focused_element_snapshot)
                elif frontmost_from_prior:
                    pre_state = ""
                else:
                    pre_state = ""
                    try:
                        pre_state = await asyncio.to_thread(self._read_screen_text_macos)
                    except (
                        TimeoutError,
                        RuntimeError,
                        OSError,
                        AttributeError,
                        TypeError,
                        ValueError,
                        subprocess.SubprocessError,
                    ) as exc:
                        logger.debug("Pre-state screen read failed before hotkey: %s", exc)
                keys = [k.strip().lower() for k in params.target.split("+") if k.strip()]
                is_paste = bool({"command", "cmd"} & set(keys)) and "v" in keys
                expected_clipboard_sha256 = str(
                    context.get("desktop_task_expected_clipboard_sha256") or ""
                ).strip()
                expected_clipboard_chars = context.get("desktop_task_expected_clipboard_chars")
                if (
                    browser_surface
                    and is_paste
                    and (
                        self._focused_snapshot_looks_browser_location_bar(pre_state)
                        or (
                            bool(context.get("desktop_task_requires_editable_focus"))
                            and self._focused_snapshot_is_browser_text_entry(pre_state)
                        )
                    )
                ):
                    return {
                        "ok": False,
                        "action": "hotkey",
                        "hotkey": params.target,
                        "frontmost_app_before": front_app,
                        "effect_verified": False,
                        "error": (
                            "Paste refused because browser focus is still on the "
                            "address/search field, not an editable document surface."
                        ),
                        "verification": "browser_location_bar_focused",
                    }
                # System Events dispatch, not pyautogui: CGEvent posts are
                # silently dropped without Accessibility grants, which left
                # failures with no error text ("unknown") and no receipt.
                try:
                    dispatch_receipt = await asyncio.to_thread(
                        self._send_hotkey_system_events, keys
                    )
                except TimeoutError as exc:
                    if browser_surface:
                        return {
                            "ok": False,
                            "action": "hotkey",
                            "hotkey": params.target,
                            "frontmost_app_before": front_app,
                            "effect_verified": False,
                            "error": f"keystroke dispatch failed: {exc}",
                        }
                    try:
                        fallback_receipt = await asyncio.to_thread(
                            self._send_hotkey_pyautogui,
                            pyautogui,
                            keys,
                        )
                        dispatch_receipt = f"system_events_timeout:{exc};{fallback_receipt}"
                    except (AttributeError, RuntimeError, OSError, TypeError, ValueError) as fallback_exc:
                        return {
                            "ok": False,
                            "action": "hotkey",
                            "hotkey": params.target,
                            "frontmost_app_before": front_app,
                            "effect_verified": False,
                            "error": (
                                "keystroke dispatch failed: "
                                f"{exc}; fallback dispatch failed: {fallback_exc}"
                            ),
                        }
                except RuntimeError as exc:
                    return {
                        "ok": False,
                        "action": "hotkey",
                        "hotkey": params.target,
                        "frontmost_app_before": front_app,
                        "effect_verified": False,
                        "error": f"keystroke dispatch failed: {exc}",
                    }
                await asyncio.sleep(0.4)
                post_front_app = (
                    front_app
                    if frontmost_from_prior and not browser_surface
                    else await asyncio.to_thread(self._frontmost_app_name)
                )
                if browser_surface:
                    post_state = await asyncio.to_thread(self._focused_element_snapshot)
                elif frontmost_from_prior:
                    post_state = ""
                else:
                    post_state = ""
                    try:
                        post_state = await asyncio.to_thread(self._read_screen_text_macos)
                    except (
                        TimeoutError,
                        RuntimeError,
                        OSError,
                        AttributeError,
                        TypeError,
                        ValueError,
                        subprocess.SubprocessError,
                    ) as exc:
                        logger.debug("Post-state screen read failed after hotkey: %s", exc)
                screen_verifiable = not (
                    _screen_text_unavailable(pre_state)
                    and _screen_text_unavailable(post_state)
                )
                effect_verified = screen_verifiable and post_state != pre_state
                effect_verified, ok, verification = _verify_the_effect_landed(
                    browser_surface=browser_surface,
                    context=context,
                    effect_verified=effect_verified,
                    is_paste=is_paste,
                    screen_verifiable=screen_verifiable,
                    self=self,
                )
                clipboard_verification: dict[str, Any] = {}
                if is_paste and expected_clipboard_sha256:
                    observed_clipboard = await asyncio.to_thread(self._get_clipboard)
                    observed_text = str(observed_clipboard.get("text") or "")
                    observed_sha256 = hashlib.sha256(
                        observed_text.encode("utf-8")
                    ).hexdigest()
                    clipboard_verification = {
                        "expected_sha256": expected_clipboard_sha256,
                        "observed_sha256": observed_sha256,
                        "expected_chars": expected_clipboard_chars,
                        "observed_chars": len(observed_text),
                        "verified": bool(
                            observed_clipboard.get("ok")
                            and observed_sha256 == expected_clipboard_sha256
                        ),
                    }
                    if not clipboard_verification["verified"]:
                        ok = False
                        effect_verified = False
                        verification = (
                            "Hotkey dispatched, but the clipboard payload no longer "
                            "matched the verified staged document body."
                        )
                expected_app_verified = bool(
                    not expected_frontmost
                    or self._frontmost_app_matches(front_app, expected_frontmost)
                    or self._frontmost_app_matches(post_front_app, expected_frontmost)
                )
                result = {
                    "ok": ok,
                    "action": "hotkey",
                    "hotkey": params.target,
                    "is_paste": is_paste,
                    "frontmost_app_before": front_app,
                    "frontmost_app_after": post_front_app,
                    "expected_frontmost_app": expected_frontmost,
                    "frontmost_app_from_prior_receipt": frontmost_from_prior,
                    "write_target_app_verified": expected_app_verified,
                    "effect_verified": effect_verified,
                    "dispatch": dispatch_receipt,
                    "verification": verification,
                    "visible_state_changed": screen_verifiable and post_state != pre_state,
                    "pre_state_hash": hashlib.sha256(pre_state.encode("utf-8")).hexdigest()[:16]
                    if pre_state
                    else "",
                    "post_state_hash": hashlib.sha256(post_state.encode("utf-8")).hexdigest()[:16]
                    if post_state
                    else "",
                    "clipboard_payload_verification": clipboard_verification,
                }
                if not ok:
                    result["error"] = verification
                return result

            elif action == "scroll":
                # Issue 88: Use x/y correctly
                pre_state = ""
                try:
                    pre_state = await asyncio.to_thread(self._read_screen_text_macos)
                except (
                    TimeoutError,
                    RuntimeError,
                    OSError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    subprocess.SubprocessError,
                ) as exc:
                    logger.debug("Pre-state screen read failed before scroll: %s", exc)
                clicks = int(params.target or "3")
                await asyncio.to_thread(pyautogui.scroll, clicks, x=params.x, y=params.y)
                await asyncio.sleep(0.4)
                post_state = ""
                try:
                    post_state = await asyncio.to_thread(self._read_screen_text_macos)
                except (
                    TimeoutError,
                    RuntimeError,
                    OSError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    subprocess.SubprocessError,
                ) as exc:
                    logger.debug("Post-state screen read failed after scroll: %s", exc)
                effect_verified = bool(pre_state or post_state) and post_state != pre_state
                return {
                    "ok": effect_verified,
                    "scrolled": clicks,
                    "effect_verified": effect_verified,
                    "verification": "State shifted."
                    if effect_verified
                    else "Scroll sent but no visible state shift was verified.",
                }

            elif action == "set_clipboard":
                return await asyncio.to_thread(self._set_clipboard, params.target)

            elif action in {"write_in_app", "create_note"}:
                # One action. "create_note" is kept as the name a plan may
                # already use, but it resolves through the same general path:
                # name an app, derive how it takes text, write.
                blocked = await self._require_permissions(
                    "writing text into an application through its scripting interface",
                    "AUTOMATION",
                )
                if blocked:
                    return blocked
                return await self._write_in_app(params.target)

            elif action == "get_clipboard":
                return await asyncio.to_thread(self._get_clipboard)

            elif action == "pursue_on_screen":
                # A goal that is watched rather than performed once.
                #
                # The target is a JSON object naming the goal and the text
                # that means it is done, plus anything the run needs to stay
                # on the right window and page. Deciding each move is the
                # pursuit's own business — it reasons, predicts what the move
                # should change, and checks that against the next reading.
                blocked = await self._require_permissions(
                    "watching the screen and acting until a goal is reached",
                    "ACCESSIBILITY",
                    "SCREEN_RECORDING",
                )
                if blocked:
                    return blocked
                return await self._pursue_on_screen(params.target)

            elif action == "wait":
                delay_s = max(0.0, min(10.0, float(params.target or 1.0)))
                await asyncio.sleep(delay_s)
                return {"ok": True, "action": "wait", "seconds": delay_s}

            elif action == "run_applescript":
                blocked = await self._require_permissions(
                    "running bounded AppleScript against the foreground desktop",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    return blocked
                script = self._validate_user_applescript(params.target)
                expected_app = self._verifiable_applescript_activation_target(script)
                if not expected_app:
                    return {
                        "ok": False,
                        "status": "applescript_effect_contract_required",
                        "error": (
                            "Raw AppleScript is limited to exact app activation with frontmost "
                            "read-back. Use os_automation for richer governed scripts and "
                            "objective-specific verification."
                        ),
                        "effect_verified": False,
                    }
                frontmost_before = await asyncio.to_thread(self._frontmost_app_name)
                output = await asyncio.to_thread(self._run_applescript, script, timeout=12)
                effect_verified, frontmost_after = await self._wait_for_frontmost_app(expected_app)
                verification = (
                    f"frontmost_app={frontmost_after}"
                    if effect_verified
                    else (
                        f"expected frontmost app {expected_app}; "
                        f"observed {frontmost_after or 'unavailable'}"
                    )
                )
                return {
                    "ok": effect_verified,
                    "action": "run_applescript",
                    "output": output,
                    "chars": len(output),
                    "effect_verified": effect_verified,
                    "effect_evidence": verification if effect_verified else "",
                    "verification": verification,
                    "effect_contract": {
                        "verifiable": True,
                        "requirements": [
                            {
                                "kind": "app_frontmost",
                                "expected": expected_app,
                                "required": True,
                                "strong": True,
                            }
                        ],
                    },
                    "verification_results": [
                        {
                            "kind": "app_frontmost",
                            "passed": effect_verified,
                            "required": True,
                            "strong": True,
                            "expected": expected_app,
                            "observed": frontmost_after,
                            "detail": verification,
                        }
                    ],
                    "frontmost_app_before": frontmost_before,
                    "frontmost_app_after": frontmost_after,
                    **({} if effect_verified else {"error": verification}),
                }

            elif action == "write_text_file":
                return await asyncio.to_thread(self._write_text_file, params.target)

            elif action == "create_folder":
                return await asyncio.to_thread(self._create_folder, params.target)
            elif action == "list_directory":
                return await asyncio.to_thread(self._list_directory, params.target)
            elif action == "fetch_topic_image":
                return await asyncio.to_thread(self._fetch_topic_image, params.target)

            elif action == "system_control":
                blocked = await self._require_permissions(
                    "changing a system setting through System Events",
                    "AUTOMATION",
                )
                if blocked:
                    return blocked
                return await self._apply_system_control(params.target)

            elif action == "render_text_pdf":
                return await asyncio.to_thread(self._render_text_pdf, params.target)

            elif action == "move_file":
                return await asyncio.to_thread(self._move_file, params.target)

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
                    return {
                        "ok": False,
                        "error": f"Security Violation: Command '{cmd}' is restricted.",
                    }

                # Support safe advanced directory/file traversal
                # 1. Intercept tree command
                if cmd == "tree":
                    target_dir = "."
                    if len(args) > 1:
                        for arg in args[1:]:
                            if not arg.startswith("-"):
                                target_dir = arg
                                break
                    try:
                        output = self._safe_directory_walk(target_dir)
                        return {"ok": True, "output": output, "exit_code": 0}
                    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
                        return {"ok": False, "error": f"Failed to walk directory: {exc}"}

                # 2. Intercept recursive ls
                if cmd == "ls" and any(arg in {"-R", "--recursive"} for arg in args):
                    target_dir = "."
                    for arg in args[1:]:
                        if not arg.startswith("-"):
                            target_dir = arg
                            break
                    try:
                        output = self._safe_directory_walk(target_dir)
                        return {"ok": True, "output": output, "exit_code": 0}
                    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
                        return {"ok": False, "error": f"Failed recursive ls walk: {exc}"}

                # 3. Intercept and constrain find commands to prevent infinite hangs
                if cmd == "find":
                    if not any(arg.startswith("-maxdepth") for arg in args):
                        if len(args) > 1 and not args[1].startswith("-"):
                            args.insert(2, "-maxdepth")
                            args.insert(3, "4")
                        else:
                            args.insert(1, "-maxdepth")
                            args.insert(2, "4")

                result = await asyncio.to_thread(
                    get_subprocess_gateway().run,
                    args,
                    capture_output=True,
                    timeout=30,
                    source="computer_use",
                    # Required, and omitting it made the gateway refuse every
                    # run_command with subprocess_accelerator_capability_undeclared
                    # — a shell command Aura could never actually run. These are
                    # ls/find/tree and friends: they touch no accelerator.
                    accelerator_capability="none",
                )
                output = (result.stdout or result.stderr or "").strip()[:3000]
                ok = result.returncode == 0
                payload: dict[str, Any] = {
                    "ok": ok,
                    "output": output,
                    "exit_code": result.returncode,
                }
                if not ok:
                    payload["error"] = output or f"Command failed with exit code {result.returncode}."
                return payload

            elif action == "open_app":
                # Opening an app uses LaunchServices plus a System Events
                # frontmost readback. The readback is part of the effect proof,
                # so a cached or environment-level permission assumption is not
                # enough for live desktop reliability.
                resolution = await asyncio.to_thread(
                    resolve_installed_app_target,
                    params.target,
                )
                app_target = resolution.resolved
                if not resolution.launchable:
                    # A phrase is not an application name. Read the words for
                    # what they name, and try whatever they could be naming.
                    from core.runtime.watched_goal import apps_named_in

                    for candidate in apps_named_in(params.target):
                        if candidate.strip().lower() == str(params.target or "").strip().lower():
                            continue
                        again = await asyncio.to_thread(resolve_installed_app_target, candidate)
                        if again.launchable:
                            logger.info(
                                "%r is not an application; %r is what it names",
                                params.target,
                                again.resolved,
                            )
                            resolution = again
                            app_target = again.resolved
                            break
                if not resolution.launchable:
                    return {
                        "ok": False,
                        "status": "application_not_found",
                        "retryable": False,
                        "error": f"No installed application matches {params.target!r}.",
                        "opened": "",
                        "app_resolution": resolution.to_dict(),
                        "launch_attempts": [],
                    }
                blocked = await self._require_permissions(
                    "opening an app and verifying it is frontmost",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    return blocked
                candidate = resolution
                attempted_args: list[list[str]] = []
                result = None
                for launch_attempt in range(2):
                    launch_args = (
                        ["open", candidate.app_path]
                        if candidate.app_path
                        else ["open", "-a", candidate.resolved]
                    )
                    if not candidate.launchable or launch_args in attempted_args:
                        break
                    attempted_args.append(launch_args)
                    result = await asyncio.to_thread(
                        get_subprocess_gateway().run,
                        launch_args,
                        capture_output=True,
                        timeout=10,
                        source="computer_use",
                        # Same requirement; launching an app uses no accelerator.
                        accelerator_capability="none",
                    )
                    if result.returncode == 0:
                        resolution = candidate
                        app_target = candidate.resolved
                        break
                    if launch_attempt == 0:
                        # A cache can outlive an app move. Refresh the real
                        # inventory once and retry only if it gives a distinct,
                        # grounded launch identity.
                        candidate = await asyncio.to_thread(
                            resolve_installed_app_target,
                            params.target,
                            refresh=True,
                        )
                if result is None or result.returncode != 0:
                    error = (
                        "open command failed"
                        if result is None
                        else (result.stderr or result.stdout or "open command failed").strip()
                    )
                    return {
                        "ok": False,
                        "error": error,
                        "opened": app_target,
                        "app_resolution": resolution.to_dict(),
                        "launch_attempts": attempted_args,
                    }
                activation_error = ""
                try:
                    await self._activate_app(app_target)
                except (TimeoutError, RuntimeError) as exc:
                    activation_error = str(exc)
                is_frontmost, frontmost_app = await self._wait_for_frontmost_app(
                    app_target
                )
                # LAUNCHING IS THE ACTION. BEING FRONTMOST IS A WISH.
                #
                # This failed the step whenever the app lost the focus race,
                # and losing it is not something she controls: the person is
                # typing in something while she works, and that something
                # keeps taking the front back. Measured 2026-07-29 —
                #
                #   open_app failed: Application launch command succeeded, but
                #   the requested app did not become frontmost
                #   (observed=Claude). Completed 0/2 steps.
                #
                # — where "Claude" is the window Bryan happened to be reading.
                # Notes had launched perfectly well; the whole task died at
                # step zero over which window had the highlight.
                #
                # The app is open, which is what was asked. Whether it also
                # holds the front is reported honestly as its own field, and
                # every step that genuinely needs focus re-asserts it itself
                # (hold_focus) or does not need it at all (the scripting
                # dictionary). Failing here pre-empted both.
                launched = True
                verification = (
                    f"Frontmost app confirmed as {frontmost_app}."
                    if is_frontmost
                    else (
                        f"{app_target} launched; another app holds the front "
                        f"(observed={frontmost_app or 'unavailable'})"
                        + (f", activation_error={activation_error}" if activation_error else "")
                        + ". Steps that need focus re-assert it themselves."
                    )
                )
                return {
                    "ok": launched,
                    "opened": app_target,
                    "returncode": result.returncode,
                    "frontmost_app": frontmost_app,
                    "is_frontmost": is_frontmost,
                    "effect_verified": launched,
                    "verification": verification,
                    "app_resolution": resolution.to_dict(),
                    "launch_attempts": attempted_args,
                }

            elif action == "open_url":
                blocked = await self._require_permissions(
                    "opening a browser URL and verifying the active tab",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    return blocked
                raw_target = str(params.target or "").strip()
                browser = ""
                url_text = raw_target
                requires_editable_focus = False
                if raw_target.startswith("{"):
                    try:
                        spec = json.loads(raw_target)
                        url_text = str(spec.get("url") or spec.get("target") or "")
                        browser = str(spec.get("browser") or "").strip()
                        requires_editable_focus = bool(
                            spec.get("requires_editable_focus")
                            or spec.get("require_editable_focus")
                        )
                    except (ValueError, TypeError, AttributeError):
                        url_text = raw_target
                if browser and browser not in _ALLOWED_URL_BROWSERS:
                    return {
                        "ok": False,
                        "error": (
                            f"Browser '{browser}' is not in the allowed browser set "
                            f"{sorted(_ALLOWED_URL_BROWSERS)}."
                        ),
                    }
                target_url = self._normalize_open_url_target(url_text)
                if not target_url:
                    return {"ok": False, "error": "No URL or search query provided."}
                if target_url.startswith("file:"):
                    return {"ok": False, "error": "Refusing to open local file URLs from chat."}
                if shutil.which("open"):
                    argv = (
                        ["open", "-a", browser, target_url]
                        if browser
                        else ["open", target_url]
                    )
                    result = await asyncio.to_thread(
                        get_subprocess_gateway().run,
                        argv,
                        capture_output=True,
                        timeout=10,
                        source="computer_use",
                        accelerator_capability="none",
                    )
                    if result.returncode != 0:
                        error = (result.stderr or result.stdout or "open command failed").strip()
                        return {"ok": False, "error": error}
                else:
                    opened = await asyncio.to_thread(webbrowser.open, target_url, 2)
                    if not opened:
                        return {"ok": False, "error": "The default browser did not accept the URL."}
                expected_browser = browser
                if not expected_browser:
                    # ASK THE SYSTEM which browser it just handed the URL to.
                    #
                    # This sampled the frontmost app the instant after `open`,
                    # which is a race it always lost: the browser has not come
                    # forward yet, so the observed app was Aura or a terminal,
                    # which is not in the allowed set, so expected_browser was
                    # "" and verification was skipped entirely. Live 2026-07-29
                    # every research run died at step 1 with
                    # "frontmost=unavailable, active_url=unavailable" — not
                    # because the browser refused, but because nobody ever
                    # asked it to come forward.
                    #
                    # LaunchServices knows the answer before the race starts.
                    expected_browser = await asyncio.to_thread(
                        self._default_browser_name
                    )
                if not expected_browser:
                    # No registered default: give the browser a moment to
                    # arrive rather than reading the screen once and giving up.
                    deadline = time.monotonic() + 6.0
                    while time.monotonic() < deadline:
                        observed_browser = await asyncio.to_thread(
                            self._frontmost_app_name
                        )
                        if observed_browser in _ALLOWED_URL_BROWSERS:
                            expected_browser = observed_browser
                            break
                        await asyncio.sleep(0.4)
                effect_verified = False
                frontmost_app = ""
                active_url = ""
                active_title = ""
                if expected_browser:
                    effect_verified, frontmost_app = await self._wait_for_frontmost_app(
                        expected_browser
                    )
                    if effect_verified:
                        active_url, active_title = await asyncio.to_thread(
                            self._active_browser_location,
                            expected_browser,
                        )
                        effect_verified = self._url_semantically_matches(
                            target_url,
                            active_url,
                        )
                    forced_navigation = False
                    force_error = ""
                    if not effect_verified and expected_browser in _ALLOWED_URL_BROWSERS:
                        try:
                            await asyncio.to_thread(
                                self._force_browser_tab_url,
                                expected_browser,
                                target_url,
                            )
                            forced_navigation = True
                            await asyncio.sleep(0.7)
                            effect_verified, frontmost_app = await self._wait_for_frontmost_app(
                                expected_browser
                            )
                            if effect_verified:
                                active_url, active_title = await asyncio.to_thread(
                                    self._active_browser_location,
                                    expected_browser,
                                )
                                effect_verified = self._url_semantically_matches(
                                    target_url,
                                    active_url,
                                )
                        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                            force_error = str(exc)
                            logger.debug("Forced browser URL repair failed: %s", exc)
                else:
                    forced_navigation = False
                    force_error = ""
                surface = f" in {browser}" if browser else ""

                # Google Docs/Sheets/Slides load asynchronously and leave keyboard
                # focus on the omnibox immediately after navigation, so a following
                # `type` lands in the address bar instead of the document body —
                # the "she types in the URL bar instead of the doc" failure. Once
                # navigation is confirmed, wait for the editor to render and click
                # into the canvas so subsequent keystrokes enter the document.
                doc_focused = False
                focus_error = ""
                focus_snapshot = ""
                if (
                    effect_verified
                    and requires_editable_focus
                    and self._is_web_editor_url(active_url or target_url)
                ):
                    if expected_browser and self._is_web_editor_create_url(active_url):
                        deadline = time.monotonic() + 12.0
                        while time.monotonic() < deadline:
                            await asyncio.sleep(0.75)
                            next_url, next_title = await asyncio.to_thread(
                                self._active_browser_location,
                                expected_browser,
                            )
                            if next_url:
                                active_url, active_title = next_url, next_title
                            if self._is_resolved_web_editor_url(active_url):
                                break
                        if self._is_web_editor_create_url(active_url):
                            focus_error = "web_editor_create_not_resolved"
                    editor_pyautogui = pyautogui
                    if not focus_error and editor_pyautogui is None:
                        editor_pyautogui, pyautogui_error = get_pyautogui()
                        if editor_pyautogui is None:
                            focus_error = (
                                "pyautogui_unavailable_for_web_editor_focus"
                                + (f": {pyautogui_error}" if pyautogui_error else "")
                            )
                    if not focus_error and editor_pyautogui is not None:
                        await asyncio.sleep(3.5)  # let the web editor finish loading
                        doc_focused, focus_snapshot, focus_error = await self._focus_web_editor_surface(
                            editor_pyautogui,
                            browser=expected_browser,
                            target_url=target_url,
                        )
                    if requires_editable_focus and not doc_focused:
                        effect_verified = False

                verification = (
                    f"Frontmost browser confirmed as {frontmost_app}; active URL verified as {active_url}."
                    + (" Document canvas focused for editing." if doc_focused else "")
                    if effect_verified
                    else (
                        "URL dispatch succeeded, but the target browser/tab could not be "
                        f"semantically confirmed (frontmost={frontmost_app or 'unavailable'}, "
                        f"active_url={active_url or 'unavailable'}"
                        + (f", forced_navigation_error={force_error}" if force_error else "")
                        + (f", focus_error={focus_error}" if focus_error else "")
                        + ")."
                    )
                )
                return {
                    "ok": effect_verified,
                    "action": "open_url",
                    "url": target_url,
                    "browser": browser,
                    "frontmost_app": frontmost_app,
                    "active_url": active_url,
                    "active_title": active_title,
                    "forced_navigation": forced_navigation,
                    "effect_verified": effect_verified,
                    "doc_focused": doc_focused,
                    "editable_focus_verified": doc_focused,
                    "focus_snapshot": focus_snapshot,
                    "focus_error": focus_error,
                    "verification": verification,
                    "summary": f"I opened a browser tab for {target_url}{surface}.",
                    **({} if effect_verified else {"error": verification}),
                }

            else:
                return {"ok": False, "error": f"Unknown action: {action}"}

        except _COMPUTER_USE_RECOVERABLE_ERRORS as e:
            _record_computer_use_degradation(
                e,
                action="returned explicit computer-use failure payload for recoverable action error",
                stage=f"execute.{action}",
                severity="degraded",
                extra={"action": action},
            )
            runtime_permission_error = self._runtime_permission_payload(str(e))
            if runtime_permission_error:
                return runtime_permission_error
            logger.error("ComputerUse action '%s' failed: %s", action, e)
            return {"ok": False, "error": str(e)}

    @staticmethod
    def _is_web_editor_url(url: str) -> bool:
        """True for editable Google web-doc surfaces where keystrokes must land in
        the document canvas, not the browser address bar."""
        u = str(url or "").lower()
        return "docs.google.com/" in u and any(
            seg in u for seg in ("/document/", "/spreadsheets/", "/presentation/")
        )

    @staticmethod
    def _is_web_editor_create_url(url: str) -> bool:
        try:
            parsed = urllib.parse.urlparse(str(url or "").lower())
        except ValueError:
            return False
        if parsed.netloc != "docs.google.com":
            return False
        return parsed.path.rstrip("/").endswith("/create")

    @classmethod
    def _is_resolved_web_editor_url(cls, url: str) -> bool:
        return cls._is_web_editor_url(url) and not cls._is_web_editor_create_url(url)

    @staticmethod
    def _url_semantically_matches(expected: str, observed: str) -> bool:
        expected = str(expected or "").strip()
        observed = str(observed or "").strip()
        if not expected or not observed:
            return False
        if observed.rstrip("/") == expected.rstrip("/") or observed.startswith(expected):
            return True
        try:
            expected_parts = urllib.parse.urlparse(expected)
            observed_parts = urllib.parse.urlparse(observed)
        except ValueError:
            return False
        if expected_parts.netloc.lower() != observed_parts.netloc.lower():
            return False
        expected_path = expected_parts.path.rstrip("/")
        observed_path = observed_parts.path.rstrip("/")
        if expected_path and observed_path.startswith(expected_path):
            return True
        # Google document creation URLs redirect to a concrete document URL.
        if expected_parts.netloc.lower() == "docs.google.com":
            expected_tokens = {token for token in expected_path.split("/") if token}
            observed_tokens = {token for token in observed_path.split("/") if token}
            return bool(expected_tokens & observed_tokens & {"document", "spreadsheets", "presentation"})
        return False

    @staticmethod
    def _browser_location_script(browser: str) -> str:
        if browser in {"Google Chrome", "Arc", "Microsoft Edge"}:
            return f'''
tell application "{browser}"
    if (count of windows) is 0 then return ""
    set activeUrl to URL of active tab of front window
    set activeTitle to title of active tab of front window
    return activeUrl & linefeed & activeTitle
end tell
'''
        if browser == "Safari":
            return '''
tell application "Safari"
    if (count of windows) is 0 then return ""
    set activeUrl to URL of current tab of front window
    set activeTitle to name of current tab of front window
    return activeUrl & linefeed & activeTitle
end tell
'''
        return ""

    def _active_browser_location(self, browser: str) -> tuple[str, str]:
        script = self._browser_location_script(str(browser or "").strip())
        if not script:
            return "", ""
        try:
            result = get_subprocess_gateway().run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
                source="computer_use",
                accelerator_capability="none",
            )
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            logger.debug("Active browser URL readback failed: %s", exc)
            return "", ""
        if result.returncode != 0:
            return "", ""
        lines = [line.strip() for line in str(result.stdout or "").splitlines()]
        active_url = lines[0] if lines else ""
        active_title = lines[1] if len(lines) > 1 else ""
        return active_url, active_title

    @staticmethod
    def _browser_source_is_private(url: str) -> bool:
        lowered = str(url or "").lower()
        private_markers = (
            "accounts.google.com",
            "chatgpt.com",
            "gemini.google.com",
            "claude.ai",
            "mail.google.com",
            "bank",
            "password",
            "login",
            "signin",
        )
        return any(marker in lowered for marker in private_markers)

    def _browser_execute_javascript(self, browser: str, js: str, *, timeout: int = 8) -> str:
        browser = str(browser or "").strip()
        if browser in {"Google Chrome", "Arc", "Microsoft Edge"}:
            script = f'''
tell application "{browser}"
    activate
    if (count of windows) is 0 then return "{{\\"ok\\":false,\\"error\\":\\"no_browser_window\\"}}"
    tell active tab of front window to execute javascript {self._applescript_string(js)}
end tell
'''
        elif browser == "Safari":
            script = f'''
tell application "Safari"
    activate
    if (count of windows) is 0 then return "{{\\"ok\\":false,\\"error\\":\\"no_browser_window\\"}}"
    do JavaScript {self._applescript_string(js)} in current tab of front window
end tell
'''
        else:
            raise ValueError(f"Browser '{browser}' is not supported for page inspection.")
        return self._run_applescript(script, timeout=timeout)

    def _inspect_browser_page(self, target: str) -> dict[str, Any]:
        try:
            spec = self._target_json(target) if str(target or "").strip().startswith("{") else {}
        except ValueError:
            spec = {}
        browser = str(spec.get("browser") or "").strip()
        if not browser:
            browser = self._frontmost_app_name()
        if browser not in _ALLOWED_URL_BROWSERS:
            return {
                "ok": False,
                "status": "not_browser",
                "error": f"Frontmost app is not an inspectable browser: {browser or 'unavailable'}.",
            }
        mode = str(spec.get("mode") or "text").strip().lower()
        max_chars = max(500, min(int(spec.get("max_chars") or 12000), 60000))
        active_url, active_title = self._active_browser_location(browser)
        wants_source = mode in {"html", "source", "dom", "page_source"}
        allow_private_source = bool(spec.get("allow_private_source") or spec.get("allow_private"))
        if wants_source and self._browser_source_is_private(active_url) and not allow_private_source:
            return {
                "ok": False,
                "status": "private_source_blocked",
                "browser": browser,
                "url": active_url,
                "title": active_title,
                "error": "Page-source inspection is blocked on private or account pages; use text mode unless the user explicitly authorizes source inspection.",
            }
        js = f"""
(() => {{
  const text = (document.body && document.body.innerText || '').slice(0, {max_chars});
  const links = Array.from(document.links || []).slice(0, 80).map(link => ({{
    text: (link.innerText || link.getAttribute('aria-label') || '').trim().slice(0, 180),
    href: link.href || ''
  }})).filter(item => item.href || item.text);
  const payload = {{
    ok: true,
    url: location.href,
    title: document.title || '',
    text,
    links,
    editable_count: document.querySelectorAll('textarea,input[type="text"],input:not([type]),[contenteditable="true"],[role="textbox"]').length
  }};
  if ({str(wants_source).lower()}) {{
    payload.html = (document.documentElement && document.documentElement.outerHTML || '').slice(0, {max_chars});
  }}
  return JSON.stringify(payload);
}})()
"""
        try:
            raw = self._browser_execute_javascript(browser, js, timeout=10)
            data = json.loads(raw or "{}")
            if not data.get("ok"):
                raise RuntimeError(str(data.get("error") or "browser inspection failed"))
            return {
                "ok": True,
                "action": "inspect_browser_page",
                "source": "browser_dom",
                "browser": browser,
                "url": str(data.get("url") or active_url),
                "title": str(data.get("title") or active_title),
                "text": str(data.get("text") or "")[:max_chars],
                "html": str(data.get("html") or "")[:max_chars],
                "links": list(data.get("links") or [])[:80],
                "editable_count": int(data.get("editable_count") or 0),
                "effect_verified": True,
            }
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            _record_computer_use_degradation(
                exc,
                action="used screen perception fallback after browser DOM inspection failed",
                stage="inspect_browser_page.dom",
                severity="warning",
            )
            try:
                from core.perception.screen_perception import get_screen_perception

                snapshot = get_screen_perception().capture_sync(save_screenshot=True)
                payload = self._screen_snapshot_result(snapshot)
                payload.update(
                    {
                        "action": "inspect_browser_page",
                        "source": "screen_perception_browser_fallback",
                        "browser": browser,
                        "url": active_url,
                        "title": active_title,
                        "effect_verified": bool(payload.get("text")),
                    }
                )
                return payload
            except _COMPUTER_USE_RECOVERABLE_ERRORS as fallback_exc:
                return {
                    "ok": False,
                    "status": "unavailable",
                    "action": "inspect_browser_page",
                    "browser": browser,
                    "url": active_url,
                    "title": active_title,
                    "error": f"Browser inspection failed and OCR fallback failed: {fallback_exc}",
                }

    async def _dismiss_visible_interruption(self, target: str) -> dict[str, Any]:
        try:
            spec = self._target_json(target) if str(target or "").strip().startswith("{") else {}
        except ValueError:
            spec = {}
        app = str(spec.get("app") or "").strip()
        if app:
            try:
                await self._activate_app(app)
            except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                logger.debug("Could not activate app before popup dismissal: %s", exc)
        before: dict[str, Any] = {}
        after: dict[str, Any] = {}
        try:
            from core.perception.screen_perception import get_screen_perception

            before = self._screen_snapshot_result(await get_screen_perception().capture(save_screenshot=False))
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            logger.debug("Popup pre-dismiss screen snapshot failed: %s", exc)
        script = """
tell application "System Events"
    key code 53
    delay 0.12
    key code 53
end tell
"""
        output = await asyncio.to_thread(self._run_applescript, script, timeout=5)
        await asyncio.sleep(0.25)
        try:
            from core.perception.screen_perception import get_screen_perception

            after = self._screen_snapshot_result(await get_screen_perception().capture(save_screenshot=False))
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            logger.debug("Popup post-dismiss screen snapshot failed: %s", exc)
        modal_before = bool(before.get("has_modal"))
        modal_after = bool(after.get("has_modal"))
        effect_verified = bool(modal_before and not modal_after)
        return {
            "ok": True,
            "action": "dismiss_popup",
            "source": "escape_key_visible_interruption",
            "dispatch": "escape_escape",
            "output": output,
            "modal_before": modal_before,
            "modal_after": modal_after,
            "before_text_hash": before.get("text_hash", ""),
            "after_text_hash": after.get("text_hash", ""),
            "effect_verified": effect_verified,
            "verification": (
                "A visible modal/interruption was present and was no longer detected after dismissal."
                if effect_verified
                else "Dismissal keys were dispatched; no modal removal was observable from screen perception."
            ),
        }

    #: Windows the person has marked private. Checked BEFORE the capture, in
    #: the skill itself, so the rule holds for every caller — the ambient
    #: loop, an explicit request, a verification step — rather than only in
    #: the loop that happened to be written with it in mind. A privacy rule
    #: enforced in one caller is a privacy rule with a hole in it.
    @staticmethod
    def _refuse_private_window() -> str:
        try:
            from core.security.screen_capture_policy import (
                evaluate_screen_capture_admission,
            )

            admission = evaluate_screen_capture_admission()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return "[screen read refused: foreground privacy could not be verified]"
        if not admission.allowed:
            return f"[screen read refused: {admission.public_error}]"
        return ""

    def read_screen_text(self) -> str:
        """Helper for AgencyCore to read screen text directly."""
        refusal = self._refuse_private_window()
        if refusal:
            return refusal
        try:
            return self._read_screen_text_macos()
        except _COMPUTER_USE_RECOVERABLE_ERRORS as e:
            _record_computer_use_degradation(
                e,
                action="returned explicit screen-read failure marker to caller",
                stage="read_screen_text.helper",
                severity="warning",
            )
            return f"[read_screen_text failed: {e}]"

    @staticmethod
    def _screen_snapshot_result(snapshot: Any) -> dict[str, Any]:
        """Convert structured screen perception into a computer_use result."""

        capture_denied = bool(getattr(snapshot, "capture_denied", False))
        unavailable_reason = str(
            getattr(snapshot, "unavailable_reason", "") or ""
        ).strip()
        capture_admission = dict(
            getattr(snapshot, "capture_admission", {}) or {}
        )

        accessibility_text = str(getattr(snapshot, "accessibility_text", "") or "").strip()
        screen_text = str(getattr(snapshot, "screen_text", "") or "").strip()
        focused_value = str(getattr(snapshot, "focused_value", "") or "").strip()
        active_app = str(getattr(snapshot, "active_app", "") or "").strip()
        window_title = str(getattr(snapshot, "window_title", "") or "").strip()
        focused_role = str(getattr(snapshot, "focused_role", "") or "").strip()
        focused_name = str(getattr(snapshot, "focused_name", "") or "").strip()
        focused_description = str(
            getattr(snapshot, "focused_description", "") or ""
        ).strip()

        window_layout = str(getattr(snapshot, "window_layout", "") or "").strip()
        open_apps = tuple(getattr(snapshot, "open_apps", ()) or ())

        # The whole desk, not just the window on top of it.
        #
        # capture() already collects the layout — every window, front to back,
        # with what covers what — and this builder used to drop it, so a screen
        # with Chrome and YouTube plainly visible beside Aura's own window was
        # answered "Active app: aura-launcher / Window: Aura Zenith". That is
        # not a reading of the screen; it is a reading of ONE window, and it is
        # objectively incomplete to the person looking at it. It is also what
        # made "ignore your own window" and "what's behind you" unanswerable:
        # nothing downstream ever received anything but the frontmost title.
        lines: list[str] = []
        if active_app:
            lines.append(f"Active app: {active_app}")
        if window_title:
            lines.append(f"Window: {window_title}")
        focus_parts = [
            part for part in (focused_role, focused_name, focused_description) if part
        ]
        if focus_parts:
            lines.append("Focused element: " + " | ".join(focus_parts))
        if window_layout:
            lines.append(f"Screen layout (front to back):\n{window_layout}")
        elif open_apps:
            lines.append("Windows open: " + ", ".join(open_apps))

        read_text = screen_text or accessibility_text or focused_value
        if read_text:
            lines.append(f"Text on screen:\n{read_text}")
        text = "\n".join(part for part in lines if part).strip()

        text_hash = str(getattr(snapshot, "text_hash", "") or "").strip()
        if text and not text_hash:
            text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        return {
            "ok": bool(text) and not capture_denied,
            # Knowing every window on the desk IS a reading of the screen, even
            # when no text could be lifted off it. Calling that "limited" told
            # the caller to distrust the one part that was complete.
            "status": (
                "ok"
                if (screen_text or accessibility_text or focused_value or window_layout)
                else "limited"
            ),
            "source": "screen_perception",
            "active_app": active_app,
            "window_title": window_title,
            "window_layout": window_layout,
            "open_apps": list(open_apps),
            "frontmost_window_bounds": str(
                getattr(snapshot, "frontmost_window_bounds", "") or ""
            ).strip(),
            "focused_role": focused_role,
            "focused_name": focused_name,
            "focused_description": focused_description,
            "focused_value": focused_value,
            "text": text,
            "accessibility_text": accessibility_text,
            "screen_text": screen_text,
            "screenshot_path": str(getattr(snapshot, "screenshot_path", "") or "").strip(),
            "text_hash": text_hash,
            "has_modal": bool(getattr(snapshot, "has_modal", False)),
            "modal_text": str(getattr(snapshot, "modal_text", "") or "").strip(),
            "has_loading": bool(getattr(snapshot, "has_loading", False)),
            "timestamp": float(getattr(snapshot, "timestamp", 0.0) or 0.0),
            "capture_denied": capture_denied,
            "error": unavailable_reason if capture_denied else "",
            "capture_admission": capture_admission,
        }

    def read_menu_clock(self) -> str:
        """Helper for reading the macOS menu bar clock."""
        try:
            return self._read_menu_clock_macos()
        except _COMPUTER_USE_RECOVERABLE_ERRORS as e:
            _record_computer_use_degradation(
                e,
                action="returned explicit menu-clock failure marker to caller",
                stage="read_menu_clock.helper",
                severity="warning",
            )
            return f"[read_menu_clock failed: {e}]"

    def _read_screen_text_macos(self) -> str:
        """Use macOS Accessibility API to extract text from the frontmost app with anti-hang limits."""
        refusal = self._refuse_private_window()
        if refusal:
            return refusal
        # `entire contents of frontApp` walks the ENTIRE accessibility tree.
        # On a browser or an IDE that is thousands of elements and routinely
        # outruns any sane timeout, so the 6s budget meant this effectively
        # always failed: live 2026-07-27, "take a screenshot and tell me what
        # you see" returned "read_screen_text failed: AppleScript timed out
        # after 6s" every time. A capability that cannot finish is not a
        # capability.
        #
        # The bounded query answers the question people actually ask — which
        # app, which window, what is on it — in a fraction of the time, and
        # degrades to just the app name rather than to nothing.
        script = """
tell application "System Events"
    try
        set frontApp to first application process whose frontmost is true
        set appName to name of frontApp
        set summary to appName
        try
            set winName to name of front window of frontApp
            set summary to summary & " — " & winName
        end try
        try
            set titles to {}
            repeat with e in (UI elements of front window of frontApp)
                try
                    set t to (name of e)
                    if t is not missing value and t is not "" then
                        set end of titles to t
                    end if
                end try
            end repeat
            if (count of titles) > 0 then
                set AppleScript's text item delimiters to ", "
                set summary to summary & " | " & (titles as string)
                set AppleScript's text item delimiters to ""
            end if
        end try
        return summary
    on error
        return "[Accessibility error or UI unresponsive]"
    end try
end tell
"""
        raw = self._run_applescript(script, timeout=15)
        if len(raw) > 3000:
            return raw[:1500] + "\n... [TRUNCATED] ...\n" + raw[-1500:]
        return raw



    def _read_menu_clock_macos(self) -> str:
        """Read the host clock without traversing macOS desktop UI.

        Time is OS state, not accessibility content. Walking ControlCenter's
        UI tree required two unrelated TCC grants and routinely outlived the
        desktop-readiness route's deadline, leaving abandoned Apple Events
        work behind. The shared host-clock primitive reads the same wall clock
        without importing PyObjC or touching the desktop.
        """
        if sys.platform != "darwin":
            raise RuntimeError("native macOS system clock is unavailable")
        return read_host_clock_text()
