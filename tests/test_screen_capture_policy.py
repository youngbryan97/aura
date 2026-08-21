from __future__ import annotations

import base64
import hashlib
import json
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


def _denied(reason):
    from core.security.screen_capture_policy import ScreenCaptureAdmission

    return ScreenCaptureAdmission(allowed=False, reason=reason)


def test_runtime_setting_denies_before_foreground_probe(monkeypatch):
    from core.security import screen_capture_policy as policy

    probed = False

    def _probe():
        nonlocal probed
        probed = True
        return "Google Chrome", "Private material - Incognito"

    monkeypatch.setattr(policy, "screen_allowed", lambda: False)
    monkeypatch.setattr("core.senses.screen_context.frontmost_window_hint", _probe)

    admission = policy.evaluate_screen_capture_admission()

    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.RUNTIME_SETTING_DISABLED
    assert probed is False


@pytest.mark.parametrize(
    ("app", "title"),
    [
        ("Google Chrome", "Private material - Incognito"),
        ("Safari", "Private Browsing"),
        ("1Password", "Vault"),
        ("Terminal", "banking credentials"),
    ],
)
def test_private_foreground_is_denied_without_metadata_leak(monkeypatch, app, title):
    from core.security import screen_capture_policy as policy

    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    admission = policy.evaluate_screen_capture_admission(context=(app, title))

    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.PRIVATE_FOREGROUND
    rendered = str(admission.to_receipt()) + admission.public_error
    assert app not in rendered
    assert title not in rendered


def test_unknown_foreground_fails_closed(monkeypatch):
    from core.security import screen_capture_policy as policy

    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    admission = policy.evaluate_screen_capture_admission(context=("", ""))
    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.FOREGROUND_UNKNOWN


def test_browser_with_unreadable_title_fails_closed(monkeypatch):
    from core.security import screen_capture_policy as policy

    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    admission = policy.evaluate_screen_capture_admission(context=("Google Chrome", ""))
    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.BROWSER_TITLE_UNKNOWN


def test_ordinary_foreground_is_admitted(monkeypatch):
    from core.security import screen_capture_policy as policy

    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    admission = policy.evaluate_screen_capture_admission(context=("Terminal", "pytest"))
    assert admission.allowed is True
    assert admission.to_receipt()["reason"] == "none"


def test_shared_privacy_policy_is_valid_and_bundled_once():
    from core.security import screen_capture_policy as policy

    policy._load_privacy_policy.cache_clear()
    loaded = policy._load_privacy_policy()
    assert loaded is not None
    assert "incognito" in loaded.private_window_markers
    assert "1password" in loaded.private_apps
    assert "google chrome" in loaded.private_browsing_apps

    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "config" / "screen_capture_privacy_policy.json").read_text(encoding="utf-8")
    )
    assert payload["schema"] == "aura.security.screen_capture_privacy_policy.v1"

    swift = (root / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")
    bundle = (root / "scripts" / "bundle_app.sh").read_text(encoding="utf-8")
    assert '"incognito"' not in swift
    assert '"1password"' not in swift
    assert "screen_capture_privacy_policy.json" in bundle
    assert 'cp "${SCREEN_CAPTURE_POLICY_SOURCE}" "${SCREEN_CAPTURE_POLICY_RESOURCE}"' in bundle
    assert "kCGWindowOwnerName" in swift
    assert '"private_visible"' in swift
    assert 'session["CGSSessionScreenIsLocked"]' in swift
    assert 'session["kCGSSessionOnConsoleKey"]' in swift
    assert 'session["kCGSessionLoginDoneKey"]' in swift
    assert "onConsole.boolValue" in swift
    assert "loginDone.boolValue" in swift
    assert "ownerApplication.activationPolicy == .regular" in swift
    assert 'reason: "session_locked"' in swift
    assert "return bridgeScreenCaptureRefusal(bridgeScreenCaptureAdmission())" in swift


def test_resident_bridge_is_authoritative_and_receipt_does_not_leak_metadata(
    monkeypatch,
):
    from core.security import native_desktop_bridge as bridge
    from core.security import screen_capture_policy as policy

    calls: list[tuple[str, dict[str, object]]] = []

    def _invoke(command, **kwargs):
        calls.append((command, kwargs))
        return {
            "ok": True,
            "bridge_transport": "resident_ipc",
            "frontmost_app": "1Password",
            "window_title": "Protected material",
            "capture_admission": {
                "schema": "aura.security.screen_capture_admission.v1",
                "allowed": False,
                "reason": "private_foreground",
                "context_known": True,
                "authority": "resident_bridge",
            },
        }

    monkeypatch.setattr(policy.sys, "platform", "darwin")
    monkeypatch.setattr(bridge, "invoke_native_desktop_bridge", _invoke)
    admission = policy._resident_bridge_capture_admission()

    assert admission is not None
    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.PRIVATE_FOREGROUND
    assert admission.authority == "resident_bridge"
    assert calls == [
        (
            "foreground_capture_admission",
            {
                "read_only": True,
                "timeout": 0.75,
                "allow_one_shot": False,
            },
        )
    ]
    rendered = str(admission.to_receipt()) + admission.public_error
    assert "1Password" not in rendered
    assert "Protected material" not in rendered


def test_production_admission_prefers_resident_bridge_over_python_probe(monkeypatch):
    from core.security import screen_capture_policy as policy

    resident = policy.ScreenCaptureAdmission(
        allowed=True,
        context_known=True,
        authority="resident_bridge",
    )
    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    monkeypatch.setattr(policy, "_resident_bridge_capture_admission", lambda: resident)
    monkeypatch.setattr(
        "core.senses.screen_context.frontmost_window_hint",
        lambda: (_ for _ in ()).throw(
            AssertionError("Python foreground probe should not override resident bridge")
        ),
    )

    assert policy.evaluate_screen_capture_admission() is resident


def test_unavailable_resident_bridge_uses_complete_visible_window_authority(
    monkeypatch,
):
    from core.security import screen_capture_policy as policy

    visible = policy.ScreenCaptureAdmission(
        allowed=True,
        context_known=True,
        authority="python_visible_windows",
    )
    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    monkeypatch.setattr(policy.sys, "platform", "darwin")
    monkeypatch.setattr(policy, "_resident_bridge_capture_admission", lambda: None)
    monkeypatch.setattr(policy, "_python_macos_capture_admission", lambda: visible)
    monkeypatch.setattr(
        "core.senses.screen_context.frontmost_window_hint",
        lambda: (_ for _ in ()).throw(
            AssertionError("complete visible-window authority must avoid frontmost-only probe")
        ),
    )

    admission = policy.evaluate_screen_capture_admission()

    assert admission is visible


def test_all_visible_windows_are_checked_when_bridge_is_unavailable():
    from core.security import screen_capture_policy as policy

    admission = policy._admission_from_visible_windows(
        [
            {
                "kCGWindowLayer": 0,
                "kCGWindowOwnerPID": 41,
                "kCGWindowOwnerName": "Terminal",
                "kCGWindowName": "pytest",
            },
            {
                "kCGWindowLayer": 0,
                "kCGWindowOwnerPID": 42,
                "kCGWindowOwnerName": "Google Chrome",
                "kCGWindowName": "Private material - Incognito",
            },
        ],
        foreground_pid=41,
        authority="python_visible_windows",
    )

    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.PRIVATE_VISIBLE
    assert "Chrome" not in str(admission.to_receipt()) + admission.public_error


def test_visible_browser_without_title_fails_closed():
    from core.security import screen_capture_policy as policy

    admission = policy._admission_from_visible_windows(
        [
            {
                "kCGWindowLayer": 0,
                "kCGWindowOwnerPID": 42,
                "kCGWindowOwnerName": "Safari",
                "kCGWindowName": "",
            }
        ],
        foreground_pid=42,
        authority="python_visible_windows",
    )

    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.BROWSER_TITLE_UNKNOWN


@pytest.mark.parametrize(
    "windows",
    [
        [object()],
        [{"kCGWindowLayer": "not-an-integer"}],
        [
            {
                "kCGWindowLayer": 0,
                "kCGWindowOwnerPID": 42,
                "kCGWindowOwnerName": "",
                "kCGWindowName": "unattributed",
            }
        ],
    ],
)
def test_incomplete_visible_window_inventory_fails_closed(windows):
    from core.security import screen_capture_policy as policy

    admission = policy._admission_from_visible_windows(
        windows,
        foreground_pid=42,
        authority="python_visible_windows",
    )

    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.FOREGROUND_UNKNOWN


def test_missing_complete_window_authority_still_fails_closed(monkeypatch):
    from core.security import screen_capture_policy as policy

    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    monkeypatch.setattr(policy.sys, "platform", "darwin")
    monkeypatch.setattr(policy, "_resident_bridge_capture_admission", lambda: None)
    monkeypatch.setattr(policy, "_python_macos_capture_admission", lambda: None)

    admission = policy.evaluate_screen_capture_admission()

    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.FOREGROUND_UNKNOWN
    assert admission.authority == "visible_window_authority_unavailable"


def test_resident_bridge_can_refuse_private_content_visible_off_foreground(
    monkeypatch,
):
    from core.security import native_desktop_bridge as bridge
    from core.security import screen_capture_policy as policy

    monkeypatch.setattr(policy.sys, "platform", "darwin")
    monkeypatch.setattr(
        bridge,
        "invoke_native_desktop_bridge",
        lambda *_args, **_kwargs: {
            "ok": True,
            "bridge_transport": "resident_ipc",
            "capture_admission": {
                "schema": "aura.security.screen_capture_admission.v1",
                "allowed": False,
                "reason": "private_visible",
                "context_known": True,
                "authority": "resident_bridge",
            },
        },
    )

    admission = policy._resident_bridge_capture_admission()

    assert admission is not None
    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.PRIVATE_VISIBLE
    assert "private content is visible" in admission.public_error


def test_malformed_resident_receipt_is_not_authoritative(monkeypatch):
    from core.security import native_desktop_bridge as bridge
    from core.security import screen_capture_policy as policy

    monkeypatch.setattr(policy.sys, "platform", "darwin")
    monkeypatch.setattr(
        bridge,
        "invoke_native_desktop_bridge",
        lambda *_args, **_kwargs: {
            "ok": True,
            "bridge_transport": "resident_ipc",
            "capture_admission": {
                "schema": "aura.security.screen_capture_admission.v0",
                "allowed": True,
                "reason": "none",
                "context_known": True,
                "authority": "resident_bridge",
            },
        },
    )

    assert policy._resident_bridge_capture_admission() is None


def test_missing_shared_policy_fails_closed(monkeypatch):
    from core.security import screen_capture_policy as policy

    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    monkeypatch.setattr(policy, "_load_privacy_policy", lambda: None)

    admission = policy.evaluate_screen_capture_admission(context=("Terminal", "Public work"))

    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.POLICY_UNAVAILABLE
    assert "Terminal" not in str(admission.to_receipt()) + admission.public_error


def test_complete_native_foreground_avoids_subprocess(monkeypatch):
    from core.senses import screen_context

    called = False

    class _Gateway:
        def run(self, *_args, **_kwargs):
            nonlocal called
            called = True
            raise AssertionError("subprocess should not run for complete native metadata")

    monkeypatch.setattr(
        screen_context,
        "_native_frontmost_window_hint",
        lambda: ("Terminal", "pytest"),
    )
    monkeypatch.setattr(screen_context, "get_subprocess_gateway", lambda: _Gateway())

    assert screen_context.frontmost_window_hint() == ("Terminal", "pytest")
    assert called is False


def test_subprocess_can_complete_native_app_without_title(monkeypatch):
    from core.senses import screen_context

    class _Gateway:
        @staticmethod
        def run(*_args, **_kwargs):
            return SimpleNamespace(
                returncode=0,
                stdout="Google Chrome|Public documentation",
            )

    monkeypatch.setattr(
        screen_context,
        "_native_frontmost_window_hint",
        lambda: ("Google Chrome", ""),
    )
    monkeypatch.setattr(screen_context, "get_subprocess_gateway", lambda: _Gateway())

    assert screen_context.frontmost_window_hint() == (
        "Google Chrome",
        "Public documentation",
    )


@pytest.mark.asyncio
async def test_host_automation_denies_before_creating_capture_path(monkeypatch):
    from core.capabilities.host_automation import HostAutomationProvider
    from core.security import screen_capture_policy as policy

    denial = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)

    async def _evaluate():
        return denial

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", _evaluate)
    provider = HostAutomationProvider()

    receipt = await provider.take_screenshot()

    assert receipt.success is False
    assert receipt.adapter == "screen_capture_policy"
    assert "private" in receipt.error
    assert receipt.evidence["capture_admission"] == denial.to_receipt()


@pytest.mark.asyncio
async def test_ocr_composition_preserves_typed_capture_denial(monkeypatch):
    from core.capabilities.host_automation import (
        AutomationReceipt,
        HostAutomationProvider,
    )
    from core.security import screen_capture_policy as policy

    denial = _denied(policy.ScreenCaptureDenial.SESSION_LOCKED)
    provider = HostAutomationProvider()

    async def _screenshot(*_args, **_kwargs):
        return AutomationReceipt(
            action="take_screenshot",
            target="",
            adapter="screen_capture_policy",
            success=False,
            error=denial.public_error,
            evidence={"capture_admission": denial.to_receipt()},
        )

    monkeypatch.setattr(provider, "take_screenshot", _screenshot)

    receipt = await provider.get_screen_text(retain_screenshot=False)

    assert receipt.success is False
    assert receipt.adapter == "screen_capture_policy"
    assert receipt.evidence["capture_admission"] == denial.to_receipt()


@pytest.mark.asyncio
async def test_screen_perception_denies_before_accessibility_or_pixels(monkeypatch):
    from core.perception.screen_perception import ScreenPerception
    from core.security import screen_capture_policy as policy

    denial = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)

    async def _evaluate():
        return denial

    async def _must_not_read(*_args, **_kwargs):
        raise AssertionError("accessibility content was read before privacy admission")

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", _evaluate)
    perception = ScreenPerception()
    monkeypatch.setattr(perception, "_frontmost_accessibility_summary", _must_not_read)
    monkeypatch.setattr(perception, "_take_screenshot", _must_not_read)

    snapshot = await perception.capture(save_screenshot=True)

    assert snapshot.capture_denied is True
    assert snapshot.screen_text == ""
    assert snapshot.accessibility_text == ""
    assert "private" in snapshot.unavailable_reason


@pytest.mark.asyncio
async def test_local_vision_denies_before_screenshot_backend(monkeypatch):
    from core.security import screen_capture_policy as policy
    from core.senses.screen_vision import LocalVision

    denial = _denied(policy.ScreenCaptureDenial.FOREGROUND_UNKNOWN)

    async def _evaluate():
        return denial

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", _evaluate)

    assert await LocalVision().capture_screen() is None


@pytest.mark.asyncio
async def test_continuous_vision_does_not_initialize_backend_while_denied(monkeypatch):
    from core.security import screen_capture_policy as policy
    from core.senses.continuous_vision import ContinuousSensoryBuffer

    denial = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)

    async def _evaluate():
        return denial

    class _MSS:
        called = False

        @classmethod
        def mss(cls):
            cls.called = True
            raise AssertionError("capture backend initialized while privacy denied")

    monkeypatch.setattr(
        "core.senses.continuous_vision.evaluate_screen_capture_admission_async",
        _evaluate,
    )
    buffer = ContinuousSensoryBuffer.__new__(ContinuousSensoryBuffer)
    buffer.sct = None
    buffer.monitor = None
    buffer._mss_module = _MSS
    buffer._screen_probe_cooldown_until = 0.0
    buffer._screen_permission_notice_at = 0.0
    buffer._screen_permission_notice_interval_s = 300.0

    assert await buffer._ensure_screen_backend() is False
    assert _MSS.called is False
    assert buffer._screen_backend_state.value == "privacy_deferred"
    assert buffer._screen_backend_reason == "private_foreground"
    assert buffer._screen_retry_delay_s == 2.0


@pytest.mark.asyncio
async def test_continuous_vision_retries_transient_unknown_context_quickly(monkeypatch):
    from core.container import ServiceContainer
    from core.security import screen_capture_policy as policy
    from core.senses.continuous_vision import ContinuousSensoryBuffer

    unknown = _denied(policy.ScreenCaptureDenial.FOREGROUND_UNKNOWN)
    admitted = policy.ScreenCaptureAdmission(
        allowed=True,
        context_known=True,
        authority="resident_bridge",
    )
    decisions = iter((unknown, admitted))

    async def _evaluate():
        return next(decisions)

    class _Guard:
        @staticmethod
        async def check_permission(_permission):
            return {"granted": True, "status": "active_native_bridge"}

    class _Capture:
        monitors = [{}, {"width": 1728, "height": 1117}]

        def close(self):
            return None

    class _MSS:
        calls = 0

        @classmethod
        def mss(cls):
            cls.calls += 1
            return _Capture()

    monkeypatch.setattr(
        "core.senses.continuous_vision.evaluate_screen_capture_admission_async",
        _evaluate,
    )
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(
            lambda cls, name, default=None: _Guard() if name == "permission_guard" else default
        ),
    )
    buffer = ContinuousSensoryBuffer.__new__(ContinuousSensoryBuffer)
    buffer.sct = None
    buffer.monitor = None
    buffer._mss_module = _MSS
    buffer._vision_executor = ThreadPoolExecutor(max_workers=1)
    buffer._screen_probe_cooldown_until = 0.0
    buffer._screen_permission_notice_at = 0.0
    buffer._screen_permission_notice_interval_s = 300.0
    try:
        assert await buffer._ensure_screen_backend() is False
        assert buffer._screen_backend_state.value == "privacy_deferred"
        assert buffer._screen_backend_reason == "foreground_unknown"
        assert buffer._screen_retry_delay_s == 0.75
        assert _MSS.calls == 0

        buffer._screen_probe_cooldown_until = 0.0
        assert await buffer._ensure_screen_backend() is True
        assert buffer._screen_backend_state.value == "ready"
        assert _MSS.calls == 1
    finally:
        buffer._vision_executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.asyncio
async def test_continuous_vision_reports_true_monitor_enumeration_failure(monkeypatch):
    from core.senses.continuous_vision import ContinuousSensoryBuffer

    class _Capture:
        monitors = []
        closed = False

        def close(self):
            self.closed = True

    candidate = _Capture()

    class _MSS:
        @staticmethod
        def mss():
            return candidate

    buffer = ContinuousSensoryBuffer.__new__(ContinuousSensoryBuffer)
    buffer.sct = None
    buffer.monitor = None
    buffer._mss_module = _MSS
    buffer._vision_executor = ThreadPoolExecutor(max_workers=1)
    buffer._screen_probe_cooldown_until = 0.0

    async def _permission_active():
        return True

    monkeypatch.setattr(buffer, "_screen_permission_active", _permission_active)
    monkeypatch.setattr(buffer, "_native_bridge_size", lambda: None)
    try:
        assert await buffer._ensure_screen_backend() is False
        assert buffer._screen_backend_state.value == "no_monitors"
        assert buffer._screen_backend_reason == "no_usable_screen_backend"
        assert buffer._screen_retry_delay_s == 30.0
        assert candidate.closed is True
    finally:
        buffer._vision_executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.asyncio
async def test_continuous_vision_falls_through_zero_sized_mss_to_native_bridge(
    monkeypatch,
):
    from core.senses.continuous_vision import ContinuousSensoryBuffer

    class _Capture:
        monitors = [{"left": 0, "top": 0, "width": 0, "height": 0}]
        closed = False

        def close(self):
            self.closed = True

    candidate = _Capture()

    class _MSS:
        @staticmethod
        def mss():
            return candidate

    buffer = ContinuousSensoryBuffer.__new__(ContinuousSensoryBuffer)
    buffer.sct = None
    buffer.monitor = None
    buffer._screen_backend_kind = ""
    buffer._mss_module = _MSS
    buffer._vision_executor = ThreadPoolExecutor(max_workers=1)
    buffer._screen_probe_cooldown_until = 0.0

    async def _permission_active():
        return True

    monkeypatch.setattr(buffer, "_screen_permission_active", _permission_active)
    monkeypatch.setattr(
        buffer,
        "_native_bridge_size",
        lambda: {"left": 0, "top": 0, "width": 1728, "height": 1117},
    )
    try:
        assert await buffer._ensure_screen_backend() is True
    finally:
        buffer._vision_executor.shutdown(wait=True, cancel_futures=True)

    assert candidate.closed is True
    assert buffer.sct is None
    assert buffer.monitor["width"] == 1728
    assert buffer._screen_backend_kind == "native_bridge"
    assert buffer._screen_backend_state.value == "ready"
    assert buffer._screen_backend_reason == "native_bridge_capture_ready"


def test_native_continuous_capture_validates_atomic_in_memory_receipt(monkeypatch):
    from core.senses.continuous_vision import ContinuousSensoryBuffer

    png = b"\x89PNG\r\n\x1a\nframe"

    def _native(command, **payload):
        assert command == "observe_foreground_frame"
        assert "path" not in payload
        return {
            "ok": True,
            "bridge_transport": "resident_ipc",
            "schema": "aura.perception.foreground_frame.v1",
            "sequence": 17,
            "captured_monotonic_ns": time.monotonic_ns(),
            "context_revision": "100:22:0:0:1728:1117:Aura",
            "app": "Google Chrome",
            "title": "Aura",
            "window_id": 22,
            "bounds": {"x": 0, "y": 0, "width": 1728, "height": 1117},
            "width": 1728,
            "height": 1117,
            "byte_length": len(png),
            "frame_sha256": hashlib.sha256(png).hexdigest(),
            "frame_base64": base64.b64encode(png).decode("ascii"),
            "capture_admission": {"allowed": True, "authority": "resident_bridge"},
        }

    monkeypatch.setattr(
        "core.security.native_desktop_bridge.invoke_native_desktop_bridge",
        _native,
    )

    buffer = ContinuousSensoryBuffer.__new__(ContinuousSensoryBuffer)
    buffer._last_native_frame_receipt = {}
    payload = buffer._capture_native_bridge_png()

    assert payload.startswith(b"\x89PNG")
    assert buffer._last_native_frame_receipt["sequence"] == 17
    assert "frame_base64" not in buffer._last_native_frame_receipt


def test_native_continuous_capture_rejects_contextless_frame(monkeypatch):
    from core.senses.continuous_vision import ContinuousSensoryBuffer

    png = b"\x89PNG\r\n\x1a\nframe"
    monkeypatch.setattr(
        "core.security.native_desktop_bridge.invoke_native_desktop_bridge",
        lambda *_args, **_kwargs: {
            "ok": True,
            "bridge_transport": "resident_ipc",
            "schema": "aura.perception.foreground_frame.v1",
            "sequence": 1,
            "captured_monotonic_ns": time.monotonic_ns(),
            "context_revision": "",
            "width": 100,
            "height": 100,
            "byte_length": len(png),
            "frame_sha256": hashlib.sha256(png).hexdigest(),
            "frame_base64": base64.b64encode(png).decode("ascii"),
        },
    )
    buffer = ContinuousSensoryBuffer.__new__(ContinuousSensoryBuffer)
    buffer._last_native_frame_receipt = {}

    with pytest.raises(RuntimeError, match="receipt is incomplete"):
        buffer._capture_native_bridge_png()


def test_native_continuous_capture_rejects_replayed_frame(monkeypatch):
    from core.senses.continuous_vision import ContinuousSensoryBuffer

    png = b"\x89PNG\r\n\x1a\nframe"
    result = {
        "ok": True,
        "bridge_transport": "resident_ipc",
        "schema": "aura.perception.foreground_frame.v1",
        "sequence": 8,
        "captured_monotonic_ns": time.monotonic_ns(),
        "context_revision": "100:22:0:0:100:100:Aura",
        "width": 100,
        "height": 100,
        "byte_length": len(png),
        "frame_sha256": hashlib.sha256(png).hexdigest(),
        "frame_base64": base64.b64encode(png).decode("ascii"),
        "capture_admission": {"allowed": True, "authority": "resident_bridge"},
    }
    monkeypatch.setattr(
        "core.security.native_desktop_bridge.invoke_native_desktop_bridge",
        lambda *_args, **_kwargs: result,
    )
    buffer = ContinuousSensoryBuffer.__new__(ContinuousSensoryBuffer)
    buffer._last_native_frame_receipt = {"sequence": 8}

    with pytest.raises(RuntimeError, match="receipt is incomplete"):
        buffer._capture_native_bridge_png()


@pytest.mark.asyncio
async def test_continuous_vision_reopens_backend_after_live_grab_failure(monkeypatch):
    from core.senses.continuous_vision import (
        ContinuousSensoryBuffer,
        ScreenBackendState,
    )

    class _Capture:
        closed = False

        def close(self):
            self.closed = True

    capture = _Capture()
    buffer = ContinuousSensoryBuffer.__new__(ContinuousSensoryBuffer)
    buffer.sct = capture
    buffer.monitor = {"width": 100, "height": 100}
    buffer._screen_backend_state = ScreenBackendState.READY
    buffer._screen_backend_reason = "capture_ready"
    buffer._screen_retry_delay_s = 0.75
    buffer._screen_probe_cooldown_until = 0.0
    buffer._vision_executor = ThreadPoolExecutor(max_workers=1)
    try:
        await buffer._invalidate_screen_backend("OSError")
    finally:
        buffer._vision_executor.shutdown(wait=True, cancel_futures=True)

    assert capture.closed is True
    assert buffer.sct is None
    assert buffer.monitor is None
    assert buffer._screen_backend_state is ScreenBackendState.BACKEND_ERROR
    assert buffer._screen_backend_reason == "OSError"
    assert buffer._screen_probe_cooldown_until > 0.0


@pytest.mark.asyncio
async def test_continuous_vision_final_recheck_blocks_grab_and_clears_frames(
    monkeypatch,
):
    from core.security import screen_capture_policy as policy
    from core.senses.continuous_vision import ContinuousSensoryBuffer

    denied = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)
    grabbed = False

    class _Capture:
        def grab(self, _monitor):
            nonlocal grabbed
            grabbed = True
            raise AssertionError("capture ran after the final privacy denial")

    async def _evaluate():
        return denied

    async def _stop_after_tick(_delay):
        buffer._is_active = False

    monkeypatch.setattr(
        "core.senses.continuous_vision.evaluate_screen_capture_admission_async",
        _evaluate,
    )
    monkeypatch.setattr(
        "core.senses.continuous_vision.asyncio.sleep",
        _stop_after_tick,
    )
    buffer = ContinuousSensoryBuffer.__new__(ContinuousSensoryBuffer)
    buffer.sct = _Capture()
    buffer.monitor = {"width": 1, "height": 1}
    buffer._capture_lock = __import__("asyncio").Lock()
    buffer._vision_executor = ThreadPoolExecutor(max_workers=1)
    buffer.frame_buffer = __import__("collections").deque(
        [("image/png", b"stale-public-frame")],
        maxlen=6,
    )
    buffer.camera_capture_enabled = False
    buffer._camera_lease = None
    buffer._is_active = True
    buffer._last_backend_fail_log = 0.0
    buffer._compute_budget = lambda: SimpleNamespace(
        interval_s=0.1,
        foreground_active=False,
        effective_hz=0.1,
    )
    try:
        await buffer._capture_loop()
    finally:
        buffer._vision_executor.shutdown(wait=True, cancel_futures=True)

    assert grabbed is False
    assert not buffer.frame_buffer
    assert buffer._screen_backend_state.value == "privacy_deferred"


def test_swift_screenshot_rechecks_privacy_immediately_before_capture():
    root = Path(__file__).resolve().parents[1]
    swift = (root / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")
    screenshot = swift[swift.index('case "screenshot":') : swift.index('case "move":')]

    assert screenshot.count("bridgeScreenCaptureAdmission()") == 2
    assert screenshot.index("let finalAdmission") < screenshot.index("try capture.run()")


def test_sensory_sidecar_denies_before_importing_capture_backend(monkeypatch):
    from core.security import screen_capture_policy as policy
    from core.senses import sensory_worker

    denial = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)
    monkeypatch.setattr(
        sensory_worker,
        "evaluate_screen_capture_admission",
        lambda: denial,
    )
    requests: queue.Queue[dict[str, str]] = queue.Queue()
    responses: queue.Queue[dict[str, object]] = queue.Queue()
    requests.put({"command": "init_vision"})
    requests.put({"command": "exit"})

    sensory_worker.sensory_worker_loop(requests, responses)

    response = responses.get(timeout=0.1)
    assert response["status"] == "error"
    assert response["msg"] == "private_foreground"
    assert "title" not in str(response)


def test_sensory_sidecar_rechecks_foreground_before_each_frame(monkeypatch):
    import sys

    from core.security import screen_capture_policy as policy
    from core.senses import sensory_worker

    admitted = policy.ScreenCaptureAdmission(allowed=True, context_known=True)
    denied = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)
    decisions = iter((admitted, denied))
    captured = False

    class _MSSContext:
        monitors = [{}, {"width": 1, "height": 1}]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def grab(self, _monitor):
            nonlocal captured
            captured = True
            raise AssertionError("private foreground was captured")

    fake_mss = SimpleNamespace(mss=lambda: _MSSContext())
    fake_cv2 = SimpleNamespace(__version__="test")
    monkeypatch.setitem(sys.modules, "mss", fake_mss)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(
        sensory_worker,
        "evaluate_screen_capture_admission",
        lambda: next(decisions),
    )
    monkeypatch.setattr(sensory_worker, "_screen_capture_preflight_allowed", lambda: True)
    requests: queue.Queue[dict[str, str]] = queue.Queue()
    responses: queue.Queue[dict[str, object]] = queue.Queue()
    requests.put({"command": "init_vision"})
    requests.put({"command": "capture_screen"})
    requests.put({"command": "exit"})

    sensory_worker.sensory_worker_loop(requests, responses)

    assert responses.get(timeout=0.1) == {"status": "ok"}
    refusal = responses.get(timeout=0.1)
    assert refusal["status"] == "error"
    assert refusal["msg"] == "private_foreground"
    assert captured is False


def test_native_bridge_refuses_before_transport(monkeypatch):
    from core.security import native_desktop_bridge as bridge
    from core.security import screen_capture_policy as policy

    denial = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)
    transported = False

    def _transport(*_args, **_kwargs):
        nonlocal transported
        transported = True
        return {"ok": True}

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission", lambda: denial)
    monkeypatch.setattr(bridge, "_invoke_resident_bridge", _transport)

    result = bridge.invoke_native_desktop_bridge("screenshot", read_only=True)

    assert result["ok"] is False
    assert result["bridge_transport"] == "policy_refusal"
    assert transported is False


@pytest.mark.asyncio
async def test_computer_use_screenshot_denies_before_backend(monkeypatch):
    from core.security import screen_capture_policy as policy
    from core.tools.computer_use import ComputerUseSkill

    denial = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)

    async def _require():
        raise policy.ScreenCaptureDeniedError(denial)

    monkeypatch.setattr("core.tools.computer_use.screen_allowed", lambda: True)
    monkeypatch.setattr(policy, "require_screen_capture_admission_async", _require)
    skill = ComputerUseSkill.__new__(ComputerUseSkill)

    with pytest.raises(policy.ScreenCaptureDeniedError):
        await skill._default_screenshot(SimpleNamespace(target="screen", payload={}))


@pytest.mark.asyncio
async def test_screen_sensor_returns_privacy_safe_denial(monkeypatch):
    from core.body import screen_sensor
    from core.body.screen_sensor import ScreenSensor
    from core.security import screen_capture_policy as policy

    denial = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)

    async def _evaluate():
        return denial

    monkeypatch.setattr(screen_sensor, "screen_allowed", lambda: True)
    monkeypatch.setattr(
        screen_sensor,
        "evaluate_screen_capture_admission_async",
        _evaluate,
    )

    result = await ScreenSensor.__new__(ScreenSensor).read()

    assert result["available"] is False
    assert result["capture_admission"]["reason"] == "private_foreground"
    assert "title" not in str(result)


def test_computer_use_helper_refuses_unknown_foreground(monkeypatch):
    from core.security import screen_capture_policy as policy
    from core.skills.computer_use import ComputerUseSkill

    monkeypatch.setattr(
        policy,
        "evaluate_screen_capture_admission",
        lambda: policy.ScreenCaptureAdmission(
            allowed=False,
            reason=policy.ScreenCaptureDenial.FOREGROUND_UNKNOWN,
            authority="test_visible_windows",
        ),
    )
    called = False

    def _read(_self):
        nonlocal called
        called = True
        return "screen content"

    monkeypatch.setattr(ComputerUseSkill, "_read_screen_text_macos", _read)
    result = ComputerUseSkill.__new__(ComputerUseSkill).read_screen_text()

    assert "refused" in result
    assert called is False
