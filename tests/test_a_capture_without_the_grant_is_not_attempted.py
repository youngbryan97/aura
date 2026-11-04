"""A screen capture macOS would refuse is not attempted, so macOS does not ask again.

On 3 November her screen-recording grant named an older signing certificate than her app's. Every ambient tick
spawned screencapture, each spawn was refused, and macOS showed Bryan its permission dialog 43 times in ten minutes.
Admission now asks the grant without a prompt first (CGPreflightScreenCaptureAccess).
"""
from __future__ import annotations

import pytest

from core.security import screen_capture_policy as policy

pytestmark = pytest.mark.unit


def test_no_grant_means_no_capture(monkeypatch):
    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    monkeypatch.setattr(policy, "screen_recording_granted", lambda: False)
    admission = policy.evaluate_screen_capture_admission()
    assert not admission.allowed
    assert admission.reason is policy.ScreenCaptureDenial.NOT_GRANTED
    assert "not granted" in admission.public_error


def test_an_unaskable_grant_leaves_the_policy_as_it_was(monkeypatch):
    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    monkeypatch.setattr(policy, "screen_recording_granted", lambda: None)
    monkeypatch.setattr(policy, "_resident_bridge_capture_admission",
                        lambda: policy.ScreenCaptureAdmission(allowed=True, authority="test"))
    assert policy.evaluate_screen_capture_admission().allowed


def test_ambient_presence_reads_no_grant_as_no_permission():
    from types import SimpleNamespace

    from core.perception.ambient_presence import AmbientPresence, SkipReason

    admission = policy.ScreenCaptureAdmission(
        allowed=False, reason=policy.ScreenCaptureDenial.NOT_GRANTED, authority="tcc_preflight"
    )
    reading = SimpleNamespace(evidence={"capture_admission": admission.to_receipt()})
    mapped = AmbientPresence._capture_deferral(reading)
    assert mapped is not None and mapped[0] is SkipReason.NO_PERMISSION
