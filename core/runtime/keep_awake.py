"""macOS keep-awake controller for continuous Aura operation.

Aura can keep thinking while the display sleeps by holding system-idle, disk,
and AC-power sleep assertions through `caffeinate`.  Closed-lid operation on a
Mac still depends on Apple's hardware rules: power connected, thermal safety,
and clamshell/external-display support.  This module does the software side
reliably and reports the remaining hardware constraints explicitly.
"""
from __future__ import annotations

import atexit
import logging
import os
import platform
import shutil
import signal
import time
from dataclasses import dataclass, field
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger(__name__)

_ENABLED_VALUES = {"1", "true", "yes", "on", "enabled"}
_DISABLED_VALUES = {"0", "false", "no", "off", "disabled"}
_KEEP_AWAKE_RECOVERABLE_ERRORS = (
    ChildProcessError,
    FileNotFoundError,
    OSError,
    ProcessLookupError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


@dataclass
class KeepAwakeStatus:
    supported: bool
    active: bool
    pid: int | None = None
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


@dataclass
class AssertionProcess:
    pid: int
    args: tuple[str, ...]
    returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        try:
            waited_pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            try:
                os.kill(self.pid, 0)
            except ProcessLookupError:
                self.returncode = 0
                return self.returncode
            # not a failure: a live pid we may not signal is still alive, and
            # None is this method's word for "still running".
            except PermissionError:
                return None
            return None
        if waited_pid == 0:
            return None
        self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode

    def terminate(self) -> None:
        os.kill(self.pid, signal.SIGTERM)

    def kill(self) -> None:
        os.kill(self.pid, signal.SIGKILL)

    def wait(self, timeout: float) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            code = self.poll()
            if code is not None:
                return code
            time.sleep(0.05)
        code = self.poll()
        if code is not None:
            return code
        raise TimeoutError(f"process {self.pid} did not exit within {timeout:.1f}s")


def _spawn_assertion_process(command: tuple[str, ...]):
    with local_internal_governed_scope(
        "core.runtime.keep_awake.caffeinate_assertion",
        domain="environment_action",
        constraints={
            "maintenance_surface": "keep_awake",
            "effect": "prevent_idle_sleep",
            "user_visible": False,
        },
    ):
        return get_subprocess_gateway().spawn(
            command,
            stdout_path=os.devnull,
            stderr_path=os.devnull,
            start_new_session=True,
            source="core.runtime.keep_awake.caffeinate_assertion",
            accelerator_capability="auto",
        )


def _register_assertion_with_runtime_hygiene(process: Any, command: tuple[str, ...]) -> None:
    try:
        from core.runtime.runtime_hygiene import get_runtime_hygiene

        get_runtime_hygiene().register_process_handle(
            process,
            kind="subprocess",
            name="keep_awake.caffeinate",
            source="core.runtime.keep_awake",
            command=" ".join(command),
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "keep_awake",
            exc,
            severity="warning",
            action="keep-awake assertion process could not register with runtime hygiene",
        )


class MacKeepAwakeController:
    """Owns a caffeinate assertion process."""

    def __init__(
        self,
        *,
        process_launcher=None,
        platform_name: str | None = None,
        path_resolver=None,
    ) -> None:
        self._process: AssertionProcess | None = None
        self._process_launcher = process_launcher or _spawn_assertion_process
        self._platform_name = platform_name
        self._path_resolver = path_resolver or shutil.which
        self._reason = ""
        self._started_at: float | None = None

    def supported(self) -> bool:
        system = self._platform_name or platform.system()
        return system == "Darwin" and self._path_resolver("caffeinate") is not None

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
        try:
            self._process = self._process_launcher(cmd)
            _register_assertion_with_runtime_hygiene(self._process, cmd)
            self._reason = reason
            self._started_at = time.time()
        except _KEEP_AWAKE_RECOVERABLE_ERRORS as exc:
            self._process = None
            self._reason = f"caffeinate start failed: {exc}"
            self._started_at = None
            return KeepAwakeStatus(
                supported=True,
                active=False,
                reason=self._reason,
                constraints=self.constraints(),
            )
        return self.status()

    def stop(self) -> KeepAwakeStatus:
        if self._process is not None and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except TimeoutError:
                self._process.kill()
                self._process.wait(timeout=3)
            except _KEEP_AWAKE_RECOVERABLE_ERRORS as exc:
                self._reason = f"caffeinate stop failed: {exc}"
        self._process = None
        return self.status()

    def on_stop(self) -> None:
        self.stop()

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
_shutdown_hooks_registered = False


def get_keep_awake_controller() -> MacKeepAwakeController:
    global _controller
    if _controller is None:
        _controller = MacKeepAwakeController()
    return _controller


def _register_shutdown_hooks(controller: MacKeepAwakeController) -> None:
    global _shutdown_hooks_registered
    if _shutdown_hooks_registered:
        return
    try:
        from core.container import ServiceContainer

        ServiceContainer.register_instance("keep_awake_controller", controller, required=False)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "keep_awake",
            exc,
            severity="warning",
            action="keep-awake controller could not register in ServiceContainer",
        )
    try:
        from core.runtime.shutdown_coordinator import get_shutdown_coordinator

        get_shutdown_coordinator().register(
            controller.stop,
            phase="actors",
            name="keep_awake_controller",
            timeout=4.0,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "keep_awake",
            exc,
            severity="warning",
            action="keep-awake controller could not register shutdown hook",
        )
    atexit.register(controller.stop)
    _shutdown_hooks_registered = True


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
    status = controller.start(
        keep_display_awake=keep_display,
        require_ac_power=require_ac_power_from_environment(),
    )
    if status.active:
        _register_shutdown_hooks(controller)
    return status


__all__ = [
    "KeepAwakeStatus",
    "MacKeepAwakeController",
    "get_keep_awake_controller",
    "keep_awake_enabled_from_environment",
    "require_ac_power_from_environment",
    "start_from_environment",
]
