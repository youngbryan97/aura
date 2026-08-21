from __future__ import annotations

import asyncio

import pytest

from core.capabilities.host_automation import AutomationReceipt
from core.perception.ambient_presence import (
    _BROKEN_SKIP_ESCALATION,
    AmbientPresence,
    PresenceMode,
    ScreenContext,
    SkipReason,
)
from core.security.screen_capture_policy import (
    ScreenCaptureAdmission,
    ScreenCaptureDenial,
)


@pytest.fixture
def presence(monkeypatch):
    instance = AmbientPresence()
    instance.set_mode(PresenceMode.BUBBLE)
    monkeypatch.setattr(
        "core.perception.ambient_presence._proactivity_suppressed", lambda: False
    )

    async def _context():
        return ScreenContext(app="Safari", title="Public page")

    instance._current_context = _context
    return instance


def _denied(reason: ScreenCaptureDenial, *, authority: str = "resident_bridge"):
    admission = ScreenCaptureAdmission(
        allowed=False,
        reason=reason,
        context_known=reason
        in {ScreenCaptureDenial.PRIVATE_FOREGROUND, ScreenCaptureDenial.PRIVATE_VISIBLE},
        authority=authority,
    )
    return AutomationReceipt(
        action="get_screen_text",
        target="",
        adapter="screen_capture_policy",
        success=False,
        error=admission.public_error,
        evidence={"capture_admission": admission.to_receipt()},
    )


@pytest.mark.parametrize(
    ("denial", "skip"),
    [
        (ScreenCaptureDenial.PRIVATE_FOREGROUND, SkipReason.PRIVACY_DEFERRED),
        (ScreenCaptureDenial.PRIVATE_VISIBLE, SkipReason.PRIVACY_DEFERRED),
        (ScreenCaptureDenial.BROWSER_TITLE_UNKNOWN, SkipReason.PRIVACY_DEFERRED),
        (ScreenCaptureDenial.SESSION_LOCKED, SkipReason.SESSION_LOCKED),
        (ScreenCaptureDenial.RUNTIME_SETTING_DISABLED, SkipReason.SCREEN_DISABLED),
    ],
)
def test_typed_capture_deferrals_never_become_blindness(
    presence, monkeypatch, denial, skip
):
    recorded = []
    monkeypatch.setattr(
        "core.perception.ambient_presence.record_degradation",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )

    async def _read():
        return _denied(denial)

    presence._read_screen_text = _read
    for _ in range(_BROKEN_SKIP_ESCALATION * 2):
        result = asyncio.run(presence.tick())

    state = presence.state()
    assert result.skip_reason is skip
    assert state["blind"] is False
    assert state["observation_deferred"] is True
    assert state["observation_deferred_reason"] == skip.value
    assert recorded == []
    assert "Public page" not in result.detail


@pytest.mark.parametrize(
    "reading",
    [
        _denied(ScreenCaptureDenial.POLICY_UNAVAILABLE),
        _denied(
            ScreenCaptureDenial.FOREGROUND_UNKNOWN,
            authority="resident_bridge_unavailable",
        ),
        AutomationReceipt(
            action="get_screen_text",
            target="",
            adapter="ocr",
            success=False,
            error="OCR backend failed",
        ),
    ],
)
def test_security_or_backend_failures_remain_broken(presence, reading):
    async def _read():
        return reading

    presence._read_screen_text = _read
    result = asyncio.run(presence.tick())

    assert result.skip_reason is SkipReason.CAPTURE_FAILED
    assert presence.state()["consecutive_broken_skips"] == 1
    assert presence.state()["observation_deferred"] is False


def test_success_after_session_unlock_clears_deferred_state(presence):
    readings = iter(
        (
            _denied(ScreenCaptureDenial.SESSION_LOCKED),
            AutomationReceipt(
                action="get_screen_text",
                target="ephemeral_verification_capture",
                adapter="ocr",
                success=True,
                result="Visible public content",
            ),
        )
    )

    async def _read():
        return next(readings)

    presence._read_screen_text = _read
    assert asyncio.run(presence.tick()).skip_reason is SkipReason.SESSION_LOCKED
    assert presence.state()["observation_deferred"] is True

    result = asyncio.run(presence.tick())

    assert result.observed is True
    assert presence.state()["observation_deferred"] is False
    assert presence.state()["observation_deferred_reason"] == ""
