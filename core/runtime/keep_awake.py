"""macOS keep-awake controller for continuous Aura operation.

Aura can keep thinking while the display sleeps by holding system-idle, disk,
and AC-power sleep assertions through `caffeinate`.  Closed-lid operation on a
Mac still depends on Apple's hardware rules: power connected, thermal safety,
and clamshell/external-display support.  This module does the software side
reliably and reports the remaining hardware constraints explicitly.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional

_ENABLED_VALUES = {"1", "true", "yes", "on", "enabled"}
_DISABLED_VALUES = {"0", "false", "no", "off", "disabled"}


@dataclass
class KeepAwakeStatus:
    supported: bool
    active: bool
    pid: Optional[int] = None
    command: tuple[str, ...] = ()
    reason: str = ""
    started_at: float | None = None
    constraints: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "active": self.active,
            "pid": self.pid,
            "command": list(self.command),
            "reason": self.reason,
            "started_at": self.started_at,
            "constraints": list(self.constraints),
        }


class MacKeepAwakeController:
    """Owns a caffeinate assertion process."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._reason = ""
        self._started_at: float | None = None

    def supported(self) -> bool:
        return platform.system() == "Darwin" and shutil.which("caffeinate") is not None

    def build_command(self, *, keep_display_awake: bool = False, require_ac_power: bool = True) -> tuple[str, ...]:
        flags = ["-i", "-m"]
        if require_ac_power:
            flags.append("-s")
        if keep_display_awake:
            flags.append("-d")
        return tuple(["caffeinate", *flags])

    def start(
        self,
        *,
        reason: str = "aura_continuous_runtime",
        keep_display_awake: bool = False,
        require_ac_power: bool = True,
    ) -> KeepAwakeStatus:
        if self.is_active():
            return self.status()
        if not self.supported():
            return KeepAwakeStatus(
                supported=False,
                active=False,
                reason="caffeinate unavailable on this platform",
                constraints=self.constraints(),
            )
        cmd = self.build_command(keep_display_awake=keep_display_awake, require_ac_power=require_ac_power)
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        self._reason = reason
        self._started_at = time.time()
        return self.status()

    def stop(self) -> KeepAwakeStatus:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        self._process = None
        return self.status()

    def is_active(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def status(self) -> KeepAwakeStatus:
        return KeepAwakeStatus(
            supported=self.supported(),
            active=self.is_active(),
            pid=self._process.pid if self.is_active() and self._process else None,
            command=tuple(self._process.args) if self.is_active() and self._process else (),
            reason=self._reason,
            started_at=self._started_at,
            constraints=self.constraints(),
        )

    @staticmethod
    def constraints() -> tuple[str, ...]:
        return (
            "Display sleep is allowed by default; use keep_display_awake=True only when needed.",
            "Closed-lid execution on Mac notebooks requires AC power and Apple-supported clamshell conditions.",
            "Thermal pressure or battery policy can still force sleep; Aura records this as an operational constraint.",
        )


_controller: MacKeepAwakeController | None = None


def get_keep_awake_controller() -> MacKeepAwakeController:
    global _controller
    if _controller is None:
        _controller = MacKeepAwakeController()
    return _controller


def keep_awake_enabled_from_environment() -> bool:
    raw = os.environ.get("AURA_KEEP_AWAKE")
    if raw is not None:
        return raw.strip().lower() in _ENABLED_VALUES
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return True


def require_ac_power_from_environment() -> bool:
    raw = os.environ.get("AURA_KEEP_AWAKE_REQUIRE_AC")
    if raw is not None:
        normalized = raw.strip().lower()
        if normalized in _DISABLED_VALUES:
            return False
        if normalized in _ENABLED_VALUES:
            return True
    if os.environ.get("AURA_KEEP_AWAKE_ON_BATTERY", "").strip().lower() in _ENABLED_VALUES:
        return False
    return True


def start_from_environment() -> KeepAwakeStatus:
    controller = get_keep_awake_controller()
    if not keep_awake_enabled_from_environment():
        return controller.status()
    keep_display = os.environ.get("AURA_KEEP_DISPLAY_AWAKE", "").strip().lower() in {"1", "true", "yes", "on"}
    return controller.start(
        keep_display_awake=keep_display,
        require_ac_power=require_ac_power_from_environment(),
    )


__all__ = [
    "KeepAwakeStatus",
    "MacKeepAwakeController",
    "get_keep_awake_controller",
    "keep_awake_enabled_from_environment",
    "require_ac_power_from_environment",
    "start_from_environment",
]
