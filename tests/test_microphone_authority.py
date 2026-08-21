"""One global microphone switch, one physical owner, one admitted ingress."""

from __future__ import annotations

import ast
import asyncio
import time
from pathlib import Path

import pytest

from core.voice import microphone_authority as microphone_module
from core.voice.microphone_authority import (
    STALE_MICROPHONE_LEASE_S,
    AudioIngressBroker,
    MicrophoneAuthority,
    MicrophoneDenial,
    MicrophoneLease,
    record_sounddevice_array,
)


def _authority(monkeypatch, *, allowed: bool = True) -> MicrophoneAuthority:
    monkeypatch.setattr(microphone_module, "microphone_allowed", lambda: allowed)
    return MicrophoneAuthority()


def test_owner_switch_is_checked_before_any_capture_owner_is_admitted(monkeypatch):
    authority = _authority(monkeypatch, allowed=False)

    result = authority.acquire(
        "browser",
        principal="owner:local",
        source="browser_duplex",
        mode="focused",
    )

    assert isinstance(result, MicrophoneDenial)
    assert result.reason == "owner_disabled"
    assert authority.state()["lease_active"] is False


def test_host_microphone_has_one_owner(monkeypatch):
    authority = _authority(monkeypatch)
    first = authority.acquire(
        "resident",
        principal="owner:local",
        source="sounddevice",
        mode="ambient",
    )
    second = authority.acquire(
        "other",
        principal="owner:local",
        source="browser_duplex",
        mode="ambient",
    )

    assert isinstance(first, MicrophoneLease)
    assert isinstance(second, MicrophoneDenial)
    assert second.reason == "device_busy"


def test_focused_conversation_preempts_passive_capture(monkeypatch):
    authority = _authority(monkeypatch)
    revoked: list[str] = []
    passive = authority.acquire(
        "resident",
        principal="owner:local",
        source="sounddevice",
        mode="passive",
        revoke_callback=revoked.append,
    )

    focused = authority.acquire(
        "browser",
        principal="owner:local",
        source="browser_duplex",
        mode="focused",
    )

    assert isinstance(passive, MicrophoneLease)
    assert isinstance(focused, MicrophoneLease)
    assert passive.active is False
    assert revoked == ["preempted_by:browser:focused"]
    assert authority.state()["preemptions"] == 1


def test_displaced_passive_owner_wakes_only_after_focused_handle_releases(monkeypatch):
    authority = _authority(monkeypatch)
    passive = authority.acquire(
        "resident",
        principal="owner:local",
        source="sounddevice",
        mode="passive",
    )
    focused = authority.acquire(
        "browser",
        principal="owner:local",
        source="browser_duplex",
        mode="focused",
    )
    assert isinstance(passive, MicrophoneLease)
    assert isinstance(focused, MicrophoneLease)
    available: list[str] = []
    authority.register_availability_waiter(
        "resident",
        principal="owner:local",
        source="sounddevice",
        callback=available.append,
    )

    assert available == []
    assert authority.state()["availability_waiters"] == {
        "host_microphone": ["resident"]
    }
    authority.release(focused)

    assert available == ["host_microphone"]
    assert authority.state()["availability_waiters"] == {}


def test_remote_microphone_does_not_contend_with_host_microphone(monkeypatch):
    authority = _authority(monkeypatch)
    host = authority.acquire(
        "desktop",
        principal="owner:local",
        source="browser_duplex",
        mode="focused",
    )
    remote = authority.acquire(
        "phone",
        principal="paired:device-1",
        source="browser_duplex",
        mode="focused",
    )

    assert isinstance(host, MicrophoneLease)
    assert isinstance(remote, MicrophoneLease)
    assert host.group != remote.group
    assert len(authority.state()["holders"]) == 2


def test_stale_holder_is_reclaimed(monkeypatch):
    authority = _authority(monkeypatch)
    first = authority.acquire(
        "dead",
        principal="owner:local",
        source="sounddevice",
        mode="focused",
    )
    assert isinstance(first, MicrophoneLease)
    first.last_seen = time.monotonic() - STALE_MICROPHONE_LEASE_S - 1

    second = authority.acquire(
        "replacement",
        principal="owner:local",
        source="sounddevice",
        mode="focused",
    )

    assert isinstance(second, MicrophoneLease)
    assert first.revoked_reason == "stale_holder_reclaimed"


def test_global_revocation_invalidates_every_transport(monkeypatch):
    authority = _authority(monkeypatch)
    host = authority.acquire(
        "desktop",
        principal="owner:local",
        source="browser_duplex",
        mode="focused",
    )
    remote = authority.acquire(
        "phone",
        principal="paired:device-1",
        source="browser_duplex",
        mode="focused",
    )

    receipt = authority.revoke_all(reason="runtime_setting_disabled")

    assert receipt["revoked"] == 2
    assert isinstance(host, MicrophoneLease) and not host.active
    assert isinstance(remote, MicrophoneLease) and not remote.active
    assert authority.state()["lease_active"] is False


def test_ingress_broker_rejects_audio_without_current_lease(monkeypatch):
    authority = _authority(monkeypatch)
    broker = AudioIngressBroker(authority)
    lease = authority.acquire(
        "desktop",
        principal="owner:local",
        source="browser_duplex",
        mode="focused",
    )
    assert isinstance(lease, MicrophoneLease)

    assert broker.admit(lease, 640) is True
    authority.release(lease)
    assert broker.admit(lease, 640) is False
    status = broker.state()
    assert status["frames_by_lease"][lease.lease_id] == 1
    assert status["bytes_by_lease"][lease.lease_id] == 640
    assert status["rejected_frames"] == 1


def test_frame_validation_never_waits_on_the_global_authority_lock(monkeypatch):
    authority = _authority(monkeypatch)
    broker = AudioIngressBroker(authority)
    lease = authority.acquire(
        "desktop",
        principal="owner:local",
        source="browser_duplex",
        mode="focused",
    )
    assert isinstance(lease, MicrophoneLease)

    class ForbiddenLock:
        def __enter__(self):
            raise AssertionError("frame admission acquired the global authority lock")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(authority, "_lock", ForbiddenLock())

    assert authority.validate(lease) == (True, "active")
    assert broker.admit(lease, 640) is True


def test_published_lease_snapshot_tracks_preemption_release_and_revocation(monkeypatch):
    authority = _authority(monkeypatch)
    passive = authority.acquire(
        "resident",
        principal="owner:local",
        source="sounddevice",
        mode="passive",
    )
    focused = authority.acquire(
        "desktop",
        principal="owner:local",
        source="browser_duplex",
        mode="focused",
    )
    assert isinstance(passive, MicrophoneLease)
    assert isinstance(focused, MicrophoneLease)
    assert authority.validate(passive) == (False, "preempted_by:desktop:focused")
    assert authority.validate(focused) == (True, "active")

    assert authority.release(focused, reason="conversation_complete") is True
    assert authority.validate(focused) == (False, "conversation_complete")

    remote = authority.acquire(
        "phone",
        principal="paired:device-1",
        source="browser_duplex",
        mode="focused",
    )
    assert isinstance(remote, MicrophoneLease)
    assert authority.revoke(remote, reason="owner_request") is True
    assert authority.validate(remote) == (False, "owner_request")


def test_native_voice_engine_never_touches_sounddevice_when_lease_is_denied(
    monkeypatch,
    tmp_path,
):
    from core.senses import voice_engine as voice_module

    opened: list[bool] = []

    class SoundDevice:
        def InputStream(self, **_kwargs):  # noqa: N802 - sounddevice API
            opened.append(True)
            raise AssertionError("sounddevice opened without microphone authority")

    class Authority:
        def acquire(self, *_args, **_kwargs):
            return MicrophoneDenial("device_busy", "focused voice owns it", "wait")

    monkeypatch.setattr(voice_module, "sd", SoundDevice())
    monkeypatch.setattr(voice_module, "get_microphone_authority", lambda: Authority())
    engine = voice_module.SovereignVoiceEngine(data_dir=str(tmp_path / "voice"))
    engine._stt_initialized = True

    assert asyncio.run(engine.start_listening()) is False
    assert opened == []


def test_native_audio_enters_asr_only_through_the_ingress_broker(
    monkeypatch,
    tmp_path,
):
    from core.senses import voice_engine as voice_module

    authority = _authority(monkeypatch)
    broker = AudioIngressBroker(authority)
    monkeypatch.setattr(voice_module, "get_microphone_authority", lambda: authority)
    monkeypatch.setattr(voice_module, "get_audio_ingress_broker", lambda: broker)

    engine = voice_module.SovereignVoiceEngine(data_dir=str(tmp_path / "voice"))
    engine.microphone_enabled = True
    lease = authority.acquire(
        engine._voice_owner_generation,
        principal="owner:local",
        source="sounddevice",
        mode="passive",
    )
    assert isinstance(lease, MicrophoneLease)
    engine._mic_lease = lease
    engine._mic_listening = True

    engine._mic_callback(b"\x01\x02" * 320, 320, None, None)

    assert engine._audio_buffer.get_nowait() == b"\x01\x02" * 320
    assert broker.state()["frames_by_lease"][lease.lease_id] == 1

    authority.release(lease)
    engine._mic_callback(b"\x03\x04" * 320, 320, None, None)
    assert engine._audio_buffer.empty()


def test_bounded_sounddevice_capture_never_opens_without_a_lease(monkeypatch):
    authority = _authority(monkeypatch)
    broker = AudioIngressBroker(authority)
    monkeypatch.setattr(microphone_module, "_authority", authority)
    monkeypatch.setattr(microphone_module, "_broker", broker)

    class Recording:
        nbytes = 640

    class SoundDevice:
        def __init__(self):
            self.calls = 0
            self.waited = False
            self.stopped = 0

        def rec(self, _frames, **_kwargs):
            self.calls += 1
            return Recording()

        def wait(self):
            self.waited = True

        def stop(self):
            self.stopped += 1

    current = authority.acquire(
        "focused",
        principal="owner:local",
        source="browser_duplex",
        mode="focused",
        preemptible=False,
    )
    assert isinstance(current, MicrophoneLease)
    device = SoundDevice()

    with pytest.raises(RuntimeError, match="device_busy"):
        record_sounddevice_array(
            device,
            holder="snapshot",
            source="test",
            mode="snapshot",
            frames=320,
            samplerate=16_000,
            channels=1,
            dtype="int16",
        )
    assert device.calls == 0

    authority.release(current)
    recording = record_sounddevice_array(
        device,
        holder="snapshot",
        source="test",
        mode="snapshot",
        frames=320,
        samplerate=16_000,
        channels=1,
        dtype="int16",
    )
    assert recording.nbytes == 640
    assert device.calls == 1
    assert device.waited is True
    assert authority.state()["lease_active"] is False
    assert sum(broker.state()["bytes_by_lease"].values()) == 640


def test_bounded_capture_closes_hardware_before_focused_preemption_returns(monkeypatch):
    authority = _authority(monkeypatch)
    broker = AudioIngressBroker(authority)
    monkeypatch.setattr(microphone_module, "_authority", authority)
    monkeypatch.setattr(microphone_module, "_broker", broker)

    class Recording:
        nbytes = 640

    class SoundDevice:
        stopped = 0
        focused: MicrophoneLease | None = None

        def rec(self, _frames, **_kwargs):
            return Recording()

        def wait(self):
            result = authority.acquire(
                "focused",
                principal="owner:local",
                source="browser_duplex",
                mode="focused",
            )
            assert isinstance(result, MicrophoneLease)
            self.focused = result

        def stop(self):
            self.stopped += 1

    device = SoundDevice()
    with pytest.raises(RuntimeError, match="preempted_by:focused:focused"):
        record_sounddevice_array(
            device,
            holder="ambient",
            source="proactive_perception",
            mode="ambient",
            frames=320,
            samplerate=16_000,
            channels=1,
            dtype="int16",
        )

    assert device.stopped == 1
    assert device.focused is not None
    assert authority.state()["holders"]["host_microphone"]["holder"] == "focused"
    authority.release(device.focused)


def test_every_direct_microphone_api_is_fenced_by_canonical_authority():
    root = Path(__file__).resolve().parents[1]
    found: set[tuple[str, str]] = set()
    for path in (root / "core").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in {"rec", "playrec", "InputStream"}:
                found.add((str(path.relative_to(root)), node.func.attr))
                continue
            if node.func.attr != "open":
                continue
            if any(
                keyword.arg == "input"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            ):
                found.add((str(path.relative_to(root)), "open(input=True)"))

    assert found == {
        ("core/senses/voice_engine.py", "InputStream"),
        ("core/voice/local_voice_cortex.py", "open(input=True)"),
        ("core/voice/microphone_authority.py", "playrec"),
        ("core/voice/microphone_authority.py", "rec"),
    }

    aura_js = (root / "interface/static/aura.js").read_text(encoding="utf-8")
    duplex_js = (root / "interface/static/voice_mode.js").read_text(encoding="utf-8")
    assert "getUserMedia({ audio: true" not in aura_js
    assert "getUserMedia({" in duplex_js
    assert "legacy_voice_transport_retired" in (
        root / "interface/server.py"
    ).read_text(encoding="utf-8")
