"""Computer-use realism shell.

The audit calls for a bounded, governed, verifiable computer-use
surface: screen perception, window detection, OCR, UI grounding,
cursor/keyboard control, app state tracking, undo/rollback, and
approval before destructive actions.

Every action call routes through a sandbox policy + capability token +
optional verifier. Platform-specific default drivers provide real macOS
screen/control attempts and fail honestly when the host does not grant the
required permissions.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import logging
import os
import subprocess
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.permission_gates import screen_allowed
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.runtime.who_gets_it_next import GaveUp

logger = logging.getLogger(__name__)

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


def _record_computer_use_tool_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "computer_use",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        extra=extra,
    )


def _driver_failure_reason(output: Any) -> str | None:
    if not isinstance(output, dict) or output.get("ok") is not False:
        return None
    reason = output.get("error") or output.get("failure_reason") or output.get("detail")
    return str(reason or "driver reported failure")


def _read_png_base64(path: str) -> str | None:
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return None
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@dataclass
class ComputerUseAction:
    kind: str  # screenshot, click, type, ocr, detect_windows
    target: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComputerUseResult:
    ok: bool
    action: ComputerUseAction
    output: Any = None
    failure_reason: str | None = None
    receipt_id: str | None = None
    verification_evidence: dict[str, Any] = field(default_factory=dict)


DriverFn = Callable[[ComputerUseAction], Awaitable[Any]]
VerifierFn = Callable[[ComputerUseAction, Any], Awaitable[tuple[bool, dict[str, Any]]]]


class ComputerUseSkill:
    """Bounded computer-use skill.

    All actions are denied unless:
      - the sandbox policy allows them
      - a capability token has been issued
      - a driver is registered for the action kind
      - destructive actions hold an explicit user approval flag

    A registered verifier may confirm the action (e.g. screenshot diff,
    expected text appeared) before returning success_verified.
    """

    READ_ACTIONS = frozenset({"screenshot", "ocr", "detect_windows"})
    DESTRUCTIVE_ACTIONS = frozenset({"click", "type", "drag"})

    def __init__(self):
        self._drivers: dict[str, DriverFn] = {}
        self._verifiers: dict[str, VerifierFn] = {}
        
        # Register default realism drivers
        self.register_driver("screenshot", self._default_screenshot)
        self.register_driver("click", self._default_click)
        self.register_driver("type", self._default_type)
        self.register_driver("ocr", self._default_ocr)
        self.register_driver("detect_windows", self._default_detect_windows)

    async def _default_screenshot(self, action: ComputerUseAction) -> str:
        # Gate BEFORE either backend (screencapture or the pyautogui fallback) so
        # disabling permissions.screen denies screen capture entirely, not just
        # the primary path. (docs/SETTINGS_WIRING_AUDIT.md)
        if not screen_allowed():
            raise PermissionError("screen_permission_denied: permissions.screen is disabled")
        from core.security.screen_capture_policy import (
            require_screen_capture_admission_async,
        )

        # Re-check foreground privacy immediately before either backend.  The
        # setting check above keeps the historical error contract; this gate
        # additionally covers private and unverifiable foreground contexts.
        await require_screen_capture_admission_async()
        errors: list[str] = []
        temp_path: str | None = None
        try:
            fd, temp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            proc = await get_subprocess_gateway().spawn_async(
                ["screencapture", "-x", temp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                source="tool_execution:computer_use.screenshot",
                accelerator_capability="none",
            )
            await asyncio.wait_for(proc.wait(), timeout=10.0)
            if proc.returncode != 0:
                raise RuntimeError(f"screencapture exited with {proc.returncode}")
            encoded = await asyncio.to_thread(_read_png_base64, temp_path)
            if encoded:
                return encoded
            raise RuntimeError("screencapture produced no image")
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            errors.append(f"screencapture:{type(exc).__qualname__}:{exc}")
            _record_computer_use_tool_degradation(
                exc,
                action="fell back from macOS screencapture to pyautogui screenshot",
                severity="warning",
                extra={"target": action.target},
            )
        finally:
            if temp_path:
                with contextlib.suppress(FileNotFoundError, PermissionError, OSError):
                    os.unlink(temp_path)

        try:
            from core.skills._pyautogui_runtime import get_pyautogui

            pyautogui, _ = get_pyautogui()
            if pyautogui:
                img = pyautogui.screenshot()
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            errors.append("pyautogui:unavailable")
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            errors.append(f"pyautogui:{type(exc).__qualname__}:{exc}")
            _record_computer_use_tool_degradation(
                exc,
                action="screen capture failed after pyautogui fallback attempt",
                severity="warning",
                extra={"target": action.target},
            )

        raise RuntimeError(f"screen capture unavailable: {'; '.join(errors) or 'no backend'}")

    async def _default_click(self, action: ComputerUseAction) -> dict[str, Any]:
        try:
            from core.skills.computer_use import ComputerUseSkill as CoreSkill

            skill = CoreSkill()
            x = action.payload.get("x", 0)
            y = action.payload.get("y", 0)
            return await skill.safe_execute({"action": "click", "x": x, "y": y}, {})
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            _record_computer_use_tool_degradation(
                exc,
                action="click driver failed before a verified desktop action could complete",
                severity="degraded",
                extra={"target": action.target, "payload": dict(action.payload)},
            )
            return {"ok": False, "error": f"click fallback error: {exc!r}"}

    async def _default_type(self, action: ComputerUseAction) -> dict[str, Any]:
        try:
            from core.skills.computer_use import ComputerUseSkill as CoreSkill

            skill = CoreSkill()
            x = action.payload.get("x", 0)
            y = action.payload.get("y", 0)
            return await skill.safe_execute({"action": "type", "target": action.target, "x": x, "y": y}, {})
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            _record_computer_use_tool_degradation(
                exc,
                action="type driver failed before a verified desktop action could complete",
                severity="degraded",
                extra={"target": action.target, "payload": dict(action.payload)},
            )
            return {"ok": False, "error": f"type fallback error: {exc!r}"}

    async def _default_ocr(self, action: ComputerUseAction) -> dict[str, Any]:
        try:
            from core.skills.computer_use import ComputerUseSkill as CoreSkill

            skill = CoreSkill()
            return await skill.safe_execute({"action": "read_screen_text", "target": action.target}, {})
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            _record_computer_use_tool_degradation(
                exc,
                action="ocr driver failed before screen text could be read",
                severity="degraded",
                extra={"target": action.target},
            )
            return {"ok": False, "error": f"ocr fallback error: {exc!r}"}

    async def _default_detect_windows(self, action: ComputerUseAction) -> dict[str, Any]:
        try:
            from core.skills.computer_use import ComputerUseSkill as CoreSkill

            skill = CoreSkill()
            tree = await asyncio.to_thread(skill._query_system_events_window_tree)
            return {"ok": True, "window_tree": tree}
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            _record_computer_use_tool_degradation(
                exc,
                action="window detection driver failed before UI tree could be read",
                severity="degraded",
                extra={"target": action.target},
            )
            return {"ok": False, "error": f"detect_windows fallback error: {exc!r}"}

    def register_driver(self, kind: str, driver: DriverFn) -> None:
        if not str(kind or "").strip():
            raise ValueError("computer-use driver kind must be non-empty")
        if not callable(driver):
            raise TypeError("computer-use driver must be callable")
        self._drivers[kind] = driver

    def register_verifier(self, kind: str, verifier: VerifierFn) -> None:
        if not str(kind or "").strip():
            raise ValueError("computer-use verifier kind must be non-empty")
        if not callable(verifier):
            raise TypeError("computer-use verifier must be callable")
        self._verifiers[kind] = verifier

    async def perform(
        self,
        action: ComputerUseAction,
        *,
        sandbox_check: Callable[[str, str], tuple[bool, str]],
        capability_grant: bool,
        approval_for_destructive: bool = False,
        receipt_id: str | None = None,
    ) -> ComputerUseResult:
        if not capability_grant:
            return ComputerUseResult(
                ok=False, action=action, failure_reason="no capability token"
            )
        cap_kind = "browser.read" if action.kind in self.READ_ACTIONS else "self.modify"
        # destructive UI events use file.write-style sandbox decision
        if action.kind in self.DESTRUCTIVE_ACTIONS:
            if not approval_for_destructive:
                return ComputerUseResult(
                    ok=False,
                    action=action,
                    failure_reason="destructive action requires explicit approval",
                )
            cap_kind = "file.write"
        ok, reason = sandbox_check(cap_kind, action.target)
        if not ok:
            return ComputerUseResult(ok=False, action=action, failure_reason=reason)
        driver = self._drivers.get(action.kind)
        if driver is None:
            return ComputerUseResult(
                ok=False,
                action=action,
                failure_reason=f"no driver registered for '{action.kind}'",
            )
        try:
            # An action that moves the pointer or types holds the screen while
            # it does. Two of these interleaving is why a run once spent 35
            # moves in the wrong window: each move was correct and neither had
            # the screen the other thought it had. Reads overlap freely.
            if action.kind in self.READ_ACTIONS:
                output = await driver(action)
            else:
                from core.runtime.who_gets_it_next import claim

                async with claim("screen", f"computer_use.{action.kind}"):
                    output = await driver(action)
        except GaveUp as exc:
            # Not reaching the screen is a failed action, not a crash: every
            # caller of perform() reads a result, and an exception escaping
            # here would land somewhere that has no idea what a claim is.
            return ComputerUseResult(
                ok=False,
                action=action,
                failure_reason=f"could not take the screen: {exc}",
            )
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            _record_computer_use_tool_degradation(
                exc,
                action="driver raised before action could be verified",
                severity="degraded",
                extra={"kind": action.kind, "target": action.target},
            )
            logger.debug("Computer-use driver failed for %s: %s", action.kind, exc)
            return ComputerUseResult(
                ok=False, action=action, failure_reason=f"driver failed: {exc!r}"
            )
        if failure_reason := _driver_failure_reason(output):
            return ComputerUseResult(
                ok=False,
                action=action,
                output=output,
                failure_reason=f"driver rejected action: {failure_reason}",
                receipt_id=receipt_id,
            )
        verifier = self._verifiers.get(action.kind)
        evidence: dict[str, Any] = {}
        verified = True
        if verifier is not None:
            try:
                verified, evidence = await verifier(action, output)
            except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                _record_computer_use_tool_degradation(
                    exc,
                    action="verifier raised after driver output; action was not accepted",
                    severity="warning",
                    extra={"kind": action.kind, "target": action.target},
                )
                logger.debug("Computer-use verifier failed for %s: %s", action.kind, exc)
                return ComputerUseResult(
                    ok=False,
                    action=action,
                    output=output,
                    failure_reason=f"verifier raised: {exc!r}",
                )
            if not isinstance(evidence, dict):
                evidence = {"raw_evidence": evidence}
        return ComputerUseResult(
            ok=verified,
            action=action,
            output=output,
            receipt_id=receipt_id,
            verification_evidence=evidence,
            failure_reason=None if verified else "verifier rejected output",
        )
