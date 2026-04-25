"""Computer-use realism shell.

The audit calls for a bounded, governed, verifiable computer-use
surface: screen perception, window detection, OCR, UI grounding,
cursor/keyboard control, app state tracking, undo/rollback, and
approval before destructive actions.

This module provides the *contract* without bundling a real screen
driver. Every action call routes through a sandbox policy + capability
token + verifier. Real platform-specific drivers register themselves
via ``register_driver`` once they exist.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


@dataclass
class ComputerUseAction:
    kind: str  # screenshot, click, type, ocr, detect_windows
    target: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComputerUseResult:
    ok: bool
    action: ComputerUseAction
    output: Any = None
    failure_reason: Optional[str] = None
    receipt_id: Optional[str] = None
    verification_evidence: Dict[str, Any] = field(default_factory=dict)


DriverFn = Callable[[ComputerUseAction], Awaitable[Any]]
VerifierFn = Callable[[ComputerUseAction, Any], Awaitable[Tuple[bool, Dict[str, Any]]]]


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

    DESTRUCTIVE_ACTIONS = frozenset({"click", "type", "drag"})

    def __init__(self):
        self._drivers: Dict[str, DriverFn] = {}
        self._verifiers: Dict[str, VerifierFn] = {}

    def register_driver(self, kind: str, driver: DriverFn) -> None:
        self._drivers[kind] = driver

    def register_verifier(self, kind: str, verifier: VerifierFn) -> None:
        self._verifiers[kind] = verifier

    async def perform(
        self,
        action: ComputerUseAction,
        *,
        sandbox_check: Callable[[str, str], Tuple[bool, str]],
        capability_grant: bool,
        approval_for_destructive: bool = False,
        receipt_id: Optional[str] = None,
    ) -> ComputerUseResult:
        if not capability_grant:
            return ComputerUseResult(
                ok=False, action=action, failure_reason="no capability token"
            )
        cap_kind = "browser.read" if action.kind in {"screenshot", "ocr"} else "self.modify"
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
            output = await driver(action)
        except BaseException as exc:
            return ComputerUseResult(
                ok=False, action=action, failure_reason=f"driver failed: {exc!r}"
            )
        verifier = self._verifiers.get(action.kind)
        evidence: Dict[str, Any] = {}
        verified = True
        if verifier is not None:
            try:
                verified, evidence = await verifier(action, output)
            except BaseException as exc:
                return ComputerUseResult(
                    ok=False,
                    action=action,
                    output=output,
                    failure_reason=f"verifier raised: {exc!r}",
                )
        return ComputerUseResult(
            ok=verified,
            action=action,
            output=output,
            receipt_id=receipt_id,
            verification_evidence=evidence,
            failure_reason=None if verified else "verifier rejected output",
        )
