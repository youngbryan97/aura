"""Native macOS computer-use bridge owned by the signed Aura.app identity.

macOS TCC evaluates the executable performing an action.  Aura's cognitive
runtime is Python, while the user grants Screen Recording and Accessibility to
``Aura.app``.  This module routes primitive desktop effects through the native
launcher executable so the visible app's grant and the acting process are the
same identity.  Planning, authority, receipts, and verification remain in the
canonical Python runtime.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from collections import namedtuple
from pathlib import Path
from typing import Any

from core import governance_context as _governance_context
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.subprocess_gateway import get_subprocess_gateway

_BRIDGE_FLAG = "--native-desktop-bridge"
_PROBE_READY_TTL_S = 30.0
_PROBE_DEGRADED_TTL_S = 2.0
_PROBE_JOIN_TIMEOUT_S = 0.35
_PROBE_LOCK = threading.Condition()
_PROBE_CACHE: tuple[float, dict[str, Any]] = (0.0, {})
_PROBE_IN_FLIGHT = False
_PROBE_STARTED_AT = 0.0
_CODE_SIGNATURE_CACHE_TTL_S = 300.0
_CODE_SIGNATURE_CACHE_LOCK = threading.Lock()
_CODE_SIGNATURE_CACHE: dict[str, tuple[int, float, dict[str, Any]]] = {}
_EFFECT_DOMAINS = (
    "environment_action",
    "external_action",
    "tool_execution",
    "state_mutation",
    "file_write",
    "self_modification",
)
_LONG_RESIDENT_COMMANDS = frozenset(
    {
        # ScreenCaptureKit performs an asynchronous shareable-content lookup
        # before the actual window capture.  Its bounded cold path can exceed
        # the three-second budget used by cheap metadata probes.
        "observe_foreground_frame",
    }
)
_Size = namedtuple("Size", "width height")
_Point = namedtuple("Point", "x y")


def _code_signature_summary(executable: Path | None) -> dict[str, Any]:
    """Return signing facts that explain macOS TCC retention behavior.

    Screen Recording and Accessibility grants are tied to the app identity
    macOS sees.  Ad-hoc signed rebuilds can leave System Settings showing an
    old "Aura" row while the current executable is still denied.  The desktop
    access panel needs those facts so it can report identity drift instead of a
    vague blocked state.
    """
    if executable is None or not executable.exists():
        return {"available": False, "reason": "bridge_executable_missing"}
    if sys.platform != "darwin":
        return {"available": False, "reason": "not_macos"}
    try:
        completed = get_subprocess_gateway().run(
            ["codesign", "-dv", "--verbose=4", str(executable)],
            timeout=1.5,
            read_only=True,
            capture_output=True,
            source="native_desktop_bridge.codesign_identity",
            accelerator_capability="none",
        )
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    text = "\n".join(
        str(part or "") for part in (completed.stdout, completed.stderr)
    )
    summary: dict[str, Any] = {
        "available": completed.returncode == 0,
        "returncode": int(completed.returncode),
        "adhoc": "Signature=adhoc" in text,
        "signature": "",
        "team_identifier": "",
        "authorities": [],
        "identifier": "",
        "cdhash": "",
        "stable_tcc_identity": True,
    }
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "Signature":
            summary["signature"] = value
        elif key == "TeamIdentifier":
            summary["team_identifier"] = value
        elif key == "Identifier":
            summary["identifier"] = value
        elif key == "CDHash":
            summary["cdhash"] = value
        elif key == "Authority":
            summary["authorities"].append(value)
    team = str(summary.get("team_identifier") or "").strip().lower()
    signature = str(summary.get("signature") or "").strip().lower()
    authorities = [
        str(value or "").strip()
        for value in summary.get("authorities", [])
        if str(value or "").strip()
    ]
    summary["stable_tcc_identity"] = bool(
        completed.returncode == 0
        and not summary["adhoc"]
        and (
            authorities
            or (signature and signature != "adhoc")
            or (team and team not in {"not set", "none"})
        )
    )
    if not summary["stable_tcc_identity"]:
        if summary["adhoc"]:
            summary["tcc_repair_hint"] = (
                "This Aura.app build is ad-hoc signed. macOS may show a stale "
                "permission row after rebuilds; install a stable signing identity "
                "before granting Screen Recording and Accessibility again."
            )
        else:
            summary["tcc_repair_hint"] = (
                "Aura.app does not expose a stable signing authority. Rebuild and "
                "install it with a persistent local or Developer ID certificate."
            )
    cert_name = os.getenv("AURA_CODESIGN_IDENTITY", "").strip()
    if cert_name:
        summary["configured_codesign_identity"] = cert_name
    if re.search(r"\badhoc\b", text, flags=re.IGNORECASE):
        summary["adhoc"] = True
    return summary


def _cached_code_signature_summary(executable: Path | None) -> dict[str, Any]:
    """Cache immutable signing evidence without extending the probe critical path."""
    if executable is None:
        return _code_signature_summary(None)
    try:
        identity_revision = int(executable.stat().st_mtime_ns)
    # not a failure: a file with no readable mtime has no identity revision, and 0 is
    # the cache key for that.
    except OSError:
        identity_revision = 0
    cache_key = str(executable)
    now = time.monotonic()
    with _CODE_SIGNATURE_CACHE_LOCK:
        cached = _CODE_SIGNATURE_CACHE.get(cache_key)
        if (
            cached is not None
            and cached[0] == identity_revision
            and now - cached[1] < _CODE_SIGNATURE_CACHE_TTL_S
        ):
            return dict(cached[2])
    summary = _code_signature_summary(executable)
    with _CODE_SIGNATURE_CACHE_LOCK:
        _CODE_SIGNATURE_CACHE[cache_key] = (
            identity_revision,
            time.monotonic(),
            dict(summary),
        )
    return summary


def _candidate_executables() -> tuple[Path, ...]:
    configured = str(os.getenv("AURA_NATIVE_DESKTOP_BRIDGE", "") or "").strip()
    project_root = Path(__file__).absolute().parents[2]
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path("/Applications/Aura.app/Contents/MacOS/aura-launcher"),
        project_root / "dist" / "Aura.app" / "Contents" / "MacOS" / "aura-launcher",
    ]
    return tuple(
        Path(os.path.abspath(os.path.expanduser(str(candidate))))
        for candidate in candidates
        if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK)
    )


def bridge_executable() -> Path | None:
    candidates = _candidate_executables()
    return candidates[0] if candidates else None


def native_desktop_bridge_identity(*, executable: Path | None = None) -> dict[str, Any]:
    """Return independently observed resident-process and signing evidence."""

    candidate = executable or bridge_executable()
    if candidate is not None:
        candidate = Path(os.path.abspath(os.path.expanduser(str(candidate))))
    return {
        "bridge_executable": str(candidate or ""),
        "resident_running": _resident_bridge_process_running(candidate),
        "code_signature": _cached_code_signature_summary(candidate),
        "captured_at_unix": time.time(),
    }


def _probe_cache_ttl(result: dict[str, Any]) -> float:
    """Cache good bridge evidence longer than denials or launch races.

    The signed Aura.app bridge is the durable macOS TCC identity.  During app
    launch, however, the bridge can temporarily miss, time out, or report a
    partial grant while System Settings has already been toggled.  A long-lived
    negative cache makes the desktop UI look blocked after the bridge is ready.
    """
    if not result or not result.get("ok"):
        return _PROBE_DEGRADED_TTL_S
    required = ("screen_recording", "accessibility", "automation")
    if all(bool(result.get(key)) for key in required):
        return _PROBE_READY_TTL_S
    return _PROBE_DEGRADED_TTL_S


def _bridge_ipc_dirs() -> tuple[Path, Path]:
    base = Path(os.getenv("AURA_NATIVE_BRIDGE_DIR", "~/.aura/native_bridge")).expanduser()
    return base / "requests", base / "responses"


def _resident_bridge_process_running(executable: Path | None = None) -> bool:
    """Return True only when the signed Aura launcher bridge is actually alive.

    The IPC directories are persistent by design.  Treating their presence as a
    live resident bridge makes direct-python launches wait on stale request
    folders until the proof/readiness probe times out.  If the resident process
    is not observable, the caller should use the one-shot signed bridge path.
    """
    executable = executable or bridge_executable()
    if executable is None:
        return False
    expected = os.path.normcase(
        os.path.normpath(os.path.abspath(os.path.expanduser(str(executable))))
    )
    from core.runtime.flags import FlagKind, declare

    bridge_pid = str(
        declare(
            "AURA_NATIVE_BRIDGE_PID",
            kind=FlagKind.STRING,
            default="",
            description="PID of the native desktop bridge process (set by the launcher)",
            owner="core.security.native_desktop_bridge",
        ).value()
    )
    candidate_pids: list[int] = []
    for raw_pid in (bridge_pid, os.getppid()):
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            continue
        if pid > 1 and pid not in candidate_pids:
            candidate_pids.append(pid)
    launcher_lock = Path(
        str(
            declare(
                "AURA_NATIVE_BRIDGE_PID_FILE",
                kind=FlagKind.STRING,
                default="~/.aura/locks/desktop-app-instance.lock",
                description="Launcher single-instance lock file path",
                owner="core.security.native_desktop_bridge",
            ).value()
        )
    ).expanduser()
    try:
        lock_pid = int(launcher_lock.read_text(encoding="utf-8").strip().splitlines()[0])
        if lock_pid > 1 and lock_pid not in candidate_pids:
            candidate_pids.append(lock_pid)
    # not a failure: a reading that cannot be taken, or that is not a
    # number when it is, leaves this at the value below.
    except (OSError, IndexError, TypeError, ValueError):
        pass

    def _matches(value: Any) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        observed = os.path.normcase(
            os.path.normpath(os.path.abspath(os.path.expanduser(text)))
        )
        return observed == expected

    try:
        from core.runtime.resource_observation import get_resource_observer

        observer = get_resource_observer()
        for pid in candidate_pids:
            try:
                process = observer.process(pid)
                if process is None:
                    continue
                if _matches(process.exe):
                    return True
                cmdline = process.cmdline
                if cmdline and _matches(cmdline[0]):
                    return True
            except (OSError, TypeError, ValueError):
                continue
    # not a failure: a process table this cannot read shows no matching process.
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return False


def _require_effect_governance(command: str) -> None:
    should_fail_closed = _governance_context.governance_runtime_active()
    token = _governance_context.require_governance(
        f"native_desktop_bridge.{command}",
        strict=True,
        allowed_domains=_EFFECT_DOMAINS,
    )
    if should_fail_closed and (
        token is None or getattr(token, "domain", "") == "degraded"
    ):
        raise _governance_context.GovernanceViolation(
            f"native_desktop_bridge.{command} called outside governed context"
        )


def _invoke_resident_bridge(
    command: str,
    *,
    timeout: float,
    **payload: Any,
) -> dict[str, Any] | None:
    request_dir, response_dir = _bridge_ipc_dirs()
    if not request_dir.is_dir() or not response_dir.is_dir():
        return None
    if not _resident_bridge_process_running():
        return None

    request_id = uuid.uuid4().hex
    request_path = request_dir / f"{request_id}.json"
    response_path = response_dir / f"{request_id}.json"
    request = {"command": str(command or "probe"), **payload}
    try:
        atomic_write_text(
            request_path,
            json.dumps(request, separators=(",", ":")),
            encoding="utf-8",
        )
    except OSError:
        request_path.unlink(missing_ok=True)
        return None

    deadline = time.monotonic() + max(0.05, float(timeout))
    try:
        while time.monotonic() < deadline:
            if response_path.is_file():
                text = response_path.read_text(encoding="utf-8")
                result = json.loads(text or "{}")
                if isinstance(result, dict):
                    result.setdefault("returncode", int(result.get("returncode", 0) or 0))
                    return result
                return {"ok": False, "error": "resident_bridge_non_object_response"}
            time.sleep(0.02)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"ok": False, "error": f"resident_bridge_error:{type(exc).__name__}: {exc}"}
    finally:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)
    return None


def invoke_native_desktop_bridge(
    command: str,
    *,
    read_only: bool = False,
    timeout: float = 12.0,
    prefer_one_shot: bool = False,
    allow_one_shot: bool = True,
    **payload: Any,
) -> dict[str, Any]:
    if not read_only:
        _require_effect_governance(command)

    command_name = str(command or "probe").lower()
    if command_name == "screenshot":
        from core.security.screen_capture_policy import (
            evaluate_screen_capture_admission,
        )

        admission = evaluate_screen_capture_admission()
        if not admission.allowed:
            return {
                "ok": False,
                "error": admission.public_error,
                "capture_admission": admission.to_receipt(),
                "bridge_transport": "policy_refusal",
            }
    resident_timeout = max(0.25, float(timeout))
    if not (
        command_name.startswith("request_")
        or command_name in _LONG_RESIDENT_COMMANDS
    ):
        resident_timeout = min(resident_timeout, 3.0)
    resident = None if prefer_one_shot else _invoke_resident_bridge(command, timeout=resident_timeout, **payload)
    if resident is not None:
        resident.setdefault("bridge_transport", "resident_ipc")
        return resident

    if not allow_one_shot:
        return {
            "ok": False,
            "error": "resident_bridge_unavailable",
            "bridge_transport": "unavailable",
            "resident_bridge_running": False,
            "retryable": True,
        }

    return _invoke_one_shot_bridge(
        command,
        read_only=read_only,
        timeout=timeout,
        **payload,
    )


def _invoke_one_shot_bridge(
    command: str,
    *,
    read_only: bool,
    timeout: float,
    **payload: Any,
) -> dict[str, Any]:
    executable = bridge_executable()
    if executable is None:
        return {"ok": False, "error": "native_desktop_bridge_unavailable"}

    request = {"command": str(command or "probe"), **payload}
    completed = get_subprocess_gateway().run(
        [str(executable), _BRIDGE_FLAG, json.dumps(request, separators=(",", ":"))],
        timeout=max(1.0, float(timeout)),
        read_only=bool(read_only),
        capture_output=True,
        source=f"native_desktop_bridge.{command}",
        accelerator_capability="none",
    )
    text = str(completed.stdout or "").strip()
    try:
        result = json.loads(text or "{}")
    # not a failure: text that is not the JSON this expects is not a record it can
    # read back.
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "native_desktop_bridge_invalid_response",
            "returncode": completed.returncode,
            "stderr": str(completed.stderr or "")[:240],
        }
    if not isinstance(result, dict):
        return {"ok": False, "error": "native_desktop_bridge_non_object_response"}
    result.setdefault("returncode", completed.returncode)
    result.setdefault("bridge_transport", "one_shot_subprocess")
    return result


def probe_native_desktop_bridge(*, force: bool = False, prefer_one_shot: bool = False) -> dict[str, Any]:
    """Return bounded bridge evidence without serializing callers behind bridge I/O.

    Production readiness is resident-only. A one-shot launcher is available only
    when explicitly requested for diagnosis; it must never make the live desktop
    surface look resident or occupy the production cache.
    """
    global _PROBE_CACHE, _PROBE_IN_FLIGHT, _PROBE_STARTED_AT

    now = time.monotonic()
    if not prefer_one_shot:
        with _PROBE_LOCK:
            captured_at, cached = _PROBE_CACHE
            if not force and cached and (now - captured_at) < _probe_cache_ttl(cached):
                cached_result = dict(cached)
                cached_result["cache_hit"] = True
                cached_result["cache_age_s"] = round(max(0.0, now - captured_at), 3)
                return cached_result

            if _PROBE_IN_FLIGHT:
                owner_started_at = _PROBE_STARTED_AT
                _PROBE_LOCK.wait(timeout=_PROBE_JOIN_TIMEOUT_S)
                captured_at, cached = _PROBE_CACHE
                if cached and captured_at >= owner_started_at:
                    shared_result = dict(cached)
                    shared_result["cache_hit"] = True
                    shared_result["singleflight_shared"] = True
                    shared_result["cache_age_s"] = round(
                        max(0.0, time.monotonic() - captured_at),
                        3,
                    )
                    return shared_result
                return {
                    "ok": False,
                    "error": "native_desktop_bridge_probe_in_flight",
                    "bridge_transport": "pending",
                    "probe_state": "in_flight",
                    "retryable": True,
                    "cache_hit": False,
                    "in_flight_age_s": round(
                        max(0.0, time.monotonic() - owner_started_at),
                        3,
                    ),
                }

            _PROBE_IN_FLIGHT = True
            _PROBE_STARTED_AT = now

    try:
        try:
            result = invoke_native_desktop_bridge(
                "probe",
                read_only=True,
                timeout=5.0,
                prefer_one_shot=prefer_one_shot,
                allow_one_shot=prefer_one_shot,
            )
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            result = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        executable = bridge_executable()
        result["bridge_executable"] = str(executable or "")
        result["code_signature"] = _cached_code_signature_summary(executable)
        result["cache_hit"] = False
        result["captured_at_unix"] = time.time()
        result["cache_ttl_s"] = _probe_cache_ttl(result)
        if not prefer_one_shot:
            with _PROBE_LOCK:
                _PROBE_CACHE = (time.monotonic(), dict(result))
        return result
    finally:
        if not prefer_one_shot:
            with _PROBE_LOCK:
                _PROBE_IN_FLIGHT = False
                _PROBE_STARTED_AT = 0.0
                _PROBE_LOCK.notify_all()


class NativePyAutoGUI:
    """Small PyAutoGUI-compatible facade backed by Aura.app CoreGraphics."""

    FAILSAFE = True
    PAUSE = 0.1

    @staticmethod
    def easeInOutQuad(value: float) -> float:  # noqa: N802 - pyautogui API compatibility
        value = max(0.0, min(1.0, float(value)))
        return 2 * value * value if value < 0.5 else 1 - ((-2 * value + 2) ** 2) / 2

    def _invoke(self, command: str, **payload: Any) -> dict[str, Any]:
        result = invoke_native_desktop_bridge(command, **payload)
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or f"native {command} failed"))
        pause = max(0.0, float(self.PAUSE or 0.0))
        if pause:
            time.sleep(pause)
        return result

    def size(self) -> _Size:
        result = invoke_native_desktop_bridge("size", read_only=True)
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "native size failed"))
        return _Size(int(result.get("width", 0)), int(result.get("height", 0)))

    def position(self) -> _Point:
        result = invoke_native_desktop_bridge("position", read_only=True)
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "native position failed"))
        return _Point(int(result.get("x", 0)), int(result.get("y", 0)))

    def screenshot(self) -> Any:
        from core.security.screen_capture_policy import require_screen_capture_admission

        require_screen_capture_admission()
        from PIL import Image

        fd, raw_path = tempfile.mkstemp(prefix="aura-screen-", suffix=".png")
        os.close(fd)
        target = Path(raw_path)
        try:
            self._invoke("screenshot", path=str(target))
            with Image.open(target) as image:
                return image.copy()
        finally:
            target.unlink(missing_ok=True)

    def moveTo(  # noqa: N802 - pyautogui API compatibility
        self,
        x: float,
        y: float,
        duration: float = 0.0,
        tween: Any = None,
    ) -> None:
        del duration, tween
        self._invoke("move", x=float(x), y=float(y))

    def click(
        self,
        x: float | None = None,
        y: float | None = None,
        clicks: int = 1,
        interval: float = 0.0,
        button: str = "left",
    ) -> None:
        payload: dict[str, Any] = {
            "clicks": max(1, int(clicks)),
            "interval": max(0.0, float(interval)),
            "button": str(button),
        }
        if x is not None:
            payload["x"] = float(x)
        if y is not None:
            payload["y"] = float(y)
        self._invoke("click", **payload)

    def write(self, text: str, interval: float = 0.0) -> None:
        self._invoke("write", text=str(text), interval=max(0.0, float(interval)))

    def press(self, key: str, presses: int = 1, interval: float = 0.0) -> None:
        self._invoke(
            "press",
            key=str(key),
            presses=max(1, int(presses)),
            interval=max(0.0, float(interval)),
        )

    def hotkey(self, *keys: str, interval: float = 0.0) -> None:
        del interval
        self._invoke("hotkey", keys=[str(key) for key in keys])

    def scroll(self, amount: int) -> None:
        self._invoke("scroll", amount=int(amount))


_NATIVE_PYAUTOGUI = NativePyAutoGUI()


def get_native_pyautogui() -> NativePyAutoGUI | None:
    probe = probe_native_desktop_bridge()
    if probe.get("ok") and probe.get("accessibility"):
        return _NATIVE_PYAUTOGUI
    return None


__all__ = [
    "NativePyAutoGUI",
    "bridge_executable",
    "get_native_pyautogui",
    "invoke_native_desktop_bridge",
    "native_desktop_bridge_identity",
    "probe_native_desktop_bridge",
]
