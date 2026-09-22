"""core/runtime/desktop_action_gateway.py — Canonical Desktop Action Gateway.

All desktop/computer-use actions should flow through this module to ensure correct governance, logging, and audit.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from typing import Any

from core.governance_context import (
    governance_runtime_active,
    require_governance,
)
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.DesktopActionGateway")
_DESKTOP_ACTION_RECOVERABLE_ERRORS = (
    FileNotFoundError,
    OSError,
    subprocess.SubprocessError,
    subprocess.TimeoutExpired,
    TypeError,
    ValueError,
)
_DESKTOP_ACTION_DOMAINS = (
    "environment_action",
    "external_action",
    "tool_execution",
)
_MAX_APPLESCRIPT_CHARS = 50_000



def _refuse_if_untrusted_context(operation: str, source: str) -> dict[str, Any] | None:
    """Refuse a desktop action when this turn read something untrusted.

    Returns a refusal payload, or None to proceed. Shaped like every other
    failure this gateway returns so callers need no new branch — a refusal
    that arrives in an unfamiliar shape gets mishandled into a crash, and a
    security control whose failure mode is a traceback gets turned off.

    The check fails closed. A broken provenance service cannot establish that
    the current turn is trusted, so lookup failure cannot be treated as a clean
    turn.
    """
    try:
        from core.security.content_provenance import describe_untrusted_context
        from core.security.rule_of_two import (
            get_rule_of_two_registry,
            install_known_handlers,
        )

        handler = get_rule_of_two_registry().get("desktop_automation")
        if handler is None:
            # These declarations are immutable security metadata, not a
            # service that should depend on orchestrator boot order. The API
            # becomes reachable before engineering foundations finish, so an
            # early desktop-access poll can reach this gate first. Install the
            # canonical declarations here, then check the registry again. A
            # failed declaration still fails closed below.
            install_known_handlers()
            handler = get_rule_of_two_registry().get("desktop_automation")
            if handler is None:
                if governance_runtime_active():
                    raise RuntimeError(
                        "desktop_automation rule-of-two handler is not installed"
                    )
                return None
        if not handler.violates_now():
            return None
        why = describe_untrusted_context() or "this turn read untrusted content"
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "desktop_action_gateway",
            exc,
            severity="degraded",
            action="refused desktop control because the untrusted-context check was unavailable",
            enforce_failure_policy=False,
        )
        return {
            "ok": False,
            "stdout": "",
            "stderr": (
                "Desktop control is unavailable because the content-provenance "
                f"guard could not establish a trusted context: {type(exc).__name__}."
            ),
            "exit_code": 125,
            "refused": "provenance_guard_unavailable",
            "operation": operation,
            "source": source,
        }

    record_degradation(
        "desktop_action_gateway",
        PermissionError(f"desktop action refused under untrusted context: {why}"),
        severity="degraded",
        action="refused a desktop action taken during a turn that ingested untrusted content",
    )
    return {
        "ok": False,
        "stdout": "",
        "stderr": (
            f"Desktop control is not available on this turn: {why}. Acting on the "
            "desktop with untrusted content in context would put this surface at "
            "three of the Rule of Two's three legs. Ask again in a turn that has "
            "not read external content, or run the action through an isolated lane."
        ),
        "exit_code": 126,
        "refused": "untrusted_context",
        "operation": operation,
        "source": source,
    }

class DesktopActionGateway:
    """Single canonical owner for desktop control and AppleScript actions."""

    def __init__(self) -> None:
        self._allowed_domains = _DESKTOP_ACTION_DOMAINS

    def run_applescript(
        self,
        script: str,
        *,
        source: str = "unknown",
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Execute a desktop command via AppleScript."""
        script = _coerce_script(script)
        timeout_s = _coerce_timeout(timeout)

        # Rule of Two, asked of THIS turn rather than of the declaration.
        # This surface declares TRUSTED input because it drives the desktop
        # from internally-formed intent — true until the turn has read a web
        # page, at which point "internally formed" means "formed from
        # something a stranger wrote". Untrusted input + executes + in-process
        # is three legs, and the rule's whole value is that it does not ask
        # anyone to estimate exploitability.
        refusal = _refuse_if_untrusted_context("run_applescript", source)
        if refusal is not None:
            return refusal

        if governance_runtime_active():
            require_governance(
                f"desktop_action_gateway.run_applescript:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )

        native_result = _run_native_applescript(script) if _native_enabled() else None
        if native_result is not None:
            return native_result

        if shutil.which("osascript") is None:
            return {
                "ok": False,
                "stdout": "",
                "stderr": "osascript is unavailable on this host",
                "exit_code": 127,
            }

        try:
            proc = get_subprocess_gateway().run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=timeout_s,
                source=f"desktop_action_gateway:{source}",
                accelerator_capability="none",
            )
            return {
                "ok": proc.returncode == 0,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"AppleScript timed out: {exc}",
                "exit_code": -1,
            }
        except _DESKTOP_ACTION_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "desktop_action_gateway",
                exc,
                action="returned failed desktop action receipt",
            )
            logger.warning("AppleScript execution failed: %s", exc)
            return {
                "ok": False,
                "stdout": "",
                "stderr": str(exc),
                "exit_code": -2,
            }

    async def run_applescript_async(
        self,
        script: str,
        *,
        source: str = "unknown",
        timeout: float = 15.0,  # noqa: ASYNC109 - public gateway contract
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.run_applescript,
            script,
            source=source,
            timeout=timeout,
        )


def _native_enabled() -> bool:
    return os.environ.get("AURA_COMPUTER_USE_NATIVE_APPLESCRIPT") == "1"


def _native_applescript_error_types() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = list(_DESKTOP_ACTION_RECOVERABLE_ERRORS)
    try:
        import objc

        errors.append(objc.error)
    # not a failure: without PyObjC there is no objc.error to add to the
    # recoverable set, and the osascript transport carries the work.
    except ImportError:
        pass
    return tuple(errors)


def _run_native_applescript(script: str) -> dict[str, Any] | None:
    """Use Aura.app's native Apple-event identity when PyObjC is available.

    Bridge absence falls back to the bounded ``osascript`` transport. An
    execution error is terminal because replaying a stateful script through a
    second transport could apply the effect twice.
    """
    try:
        from Foundation import NSAppleScript
    # not a failure: no PyObjC bridge means the bounded osascript transport
    # runs the script instead, which is the documented fallback above.
    except (ImportError, AttributeError):
        return None

    try:
        apple_script = NSAppleScript.alloc().initWithSource_(script)
        descriptor, error_info = apple_script.executeAndReturnError_(None)
        if descriptor is not None:
            return {
                "ok": True,
                "stdout": str(descriptor.stringValue() or "").strip(),
                "stderr": "",
                "exit_code": 0,
                "transport": "native_nsapplescript",
            }
        message = ""
        error_number = ""
        if error_info:
            message = str(error_info.get("NSAppleScriptErrorMessage") or "")
            error_number = str(error_info.get("NSAppleScriptErrorNumber") or "")
        detail = f"{message} ({error_number})" if error_number else message
        return {
            "ok": False,
            "stdout": "",
            "stderr": detail or "AppleScript native execution failed.",
            "exit_code": int(error_number) if error_number.lstrip("-").isdigit() else -2,
            "transport": "native_nsapplescript",
        }
    except _native_applescript_error_types() as exc:
        record_degradation(
            "desktop_action_gateway.native_applescript",
            exc,
            action="returned failed native desktop action receipt without replaying the effect",
        )
        return {
            "ok": False,
            "stdout": "",
            "stderr": str(exc),
            "exit_code": -2,
            "transport": "native_nsapplescript",
        }


def _coerce_script(script: str) -> str:
    if not isinstance(script, str):
        raise TypeError("AppleScript payload must be a string")
    script = script.strip()
    if not script:
        raise ValueError("AppleScript payload must not be empty")
    if len(script) > _MAX_APPLESCRIPT_CHARS:
        raise ValueError(f"AppleScript payload exceeds {_MAX_APPLESCRIPT_CHARS} characters")
    return script


def _coerce_timeout(timeout: float) -> float:
    try:
        timeout_s = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("desktop action timeout must be numeric") from exc
    if timeout_s <= 0:
        raise ValueError("desktop action timeout must be positive")
    return min(timeout_s, 120.0)


_gateway: DesktopActionGateway | None = None


def get_desktop_action_gateway() -> DesktopActionGateway:
    global _gateway
    if _gateway is None:
        _gateway = DesktopActionGateway()
    return _gateway


__all__ = ["DesktopActionGateway", "get_desktop_action_gateway"]
