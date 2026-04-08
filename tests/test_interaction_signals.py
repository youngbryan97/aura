import asyncio
import base64
import time
from types import SimpleNamespace

import pytest

from core.senses.interaction_signals import InteractionSignalsEngine, decode_data_url_image
from interface.routes.interaction_signals import _camera_signal_allowed
from interface.routes import privacy as privacy_routes


def test_decode_data_url_image_round_trip():
    raw = b"fake-jpeg-payload"
    data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    assert decode_data_url_image(data_url) == raw


def test_interaction_signals_typing_hesitation_biases_concise_guidance():
    engine = InteractionSignalsEngine()
    engine._typing = engine._update_typing_state(
        {
            "timestamp": time.time(),
            "active": False,
            "session_ms": 5400,
            "key_count": 18,
            "correction_count": 5,
            "max_pause_ms": 2100,
            "pause_before_submit_ms": 1900,
            "message_chars": 24,
            "submitted": True,
        }
    )

    fused = engine._compute_fused_state()
    guidance = engine.get_prompt_guidance()

    assert fused.hesitation > 0.5
    assert fused.verbosity_bias == "concise"
    assert "LIVE HUMAN SIGNALS" in guidance
    assert "question pressure" in guidance.lower()


def test_interaction_signals_voice_and_vision_raise_attention_and_engagement():
    engine = InteractionSignalsEngine()
    now = time.time()
    engine._voice = engine._update_voice_state(
        {
            "timestamp": now,
            "speech_ratio": 0.92,
            "rms_avg": 0.065,
            "rms_std": 0.009,
            "peak_avg": 0.34,
            "zcr_avg": 0.13,
            "clipping_ratio": 0.0,
        }
    )
    engine._vision = engine._update_vision_state(
        {
            "updated_at": now,
            "face_present": True,
            "face_count": 1,
            "face_area_ratio": 0.18,
            "gaze_direction": "center",
            "head_pose": "center",
            "attention_available": 0.86,
            "eyes_detected": 2,
        }
    )

    fused = engine._compute_fused_state()

    assert engine._voice.label in {"activated", "steady"}
    assert fused.engagement > 0.45
    assert fused.attention_available > 0.75
    assert "voice" in fused.active_modalities
    assert "vision" in fused.active_modalities


@pytest.mark.asyncio
async def test_interaction_signals_async_publish_updates_queue_consumers():
    engine = InteractionSignalsEngine()
    await engine.publish_typing(
        {
            "timestamp": time.time(),
            "active": True,
            "session_ms": 1200,
            "key_count": 14,
            "correction_count": 1,
            "max_pause_ms": 240,
            "pause_before_submit_ms": 0,
            "message_chars": 19,
            "submitted": False,
        }
    )

    for _ in range(20):
        await asyncio.sleep(0.01)
        status = engine.get_status()
        if status["typing"]["message_chars"] == 19:
            break
    else:
        status = engine.get_status()

    await engine.stop()

    assert status["typing"]["message_chars"] == 19
    assert status["typing"]["label"] in {"flowing", "rapid", "considered"}
    assert status["queues"]["typing_depth"] == 0


@pytest.mark.asyncio
async def test_browser_only_camera_privacy_keeps_vision_signals_available(monkeypatch):
    original_state = privacy_routes.get_browser_camera_privacy()
    smc = SimpleNamespace(camera_enabled=False)
    vision_buffer = SimpleNamespace(camera_enabled=False)

    def fake_get(name, default=None):
        if name == "sensory_motor_cortex":
            return smc
        if name == "continuous_vision":
            return vision_buffer
        return default

    monkeypatch.setattr(privacy_routes.ServiceContainer, "get", fake_get)

    import core.runtime.boot_safety as boot_safety

    monkeypatch.setattr(boot_safety, "main_process_camera_policy", lambda enabled: (False, "denied for tests"))

    try:
        response = await privacy_routes.api_privacy_camera(privacy_routes.PrivacyPayload(enabled=True), None)

        assert response["ok"] is True
        assert response["enabled"] is True
        assert response["mode"] == "browser_only"
        assert smc.camera_enabled is False
        assert vision_buffer.camera_enabled is False
        assert _camera_signal_allowed() is True
    finally:
        privacy_routes.set_browser_camera_privacy(
            enabled=bool(original_state.get("enabled", False)),
            mode=str(original_state.get("mode", "off")),
            reason=original_state.get("reason"),
        )
