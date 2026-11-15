"""Her cortex's admission can evict a model that lives inside her own process, by that model's own release.

LIVE 14 November 2026: her memory watchdog killed her cortex, and it could not come back for 27 minutes. Room for it
meant evicting her voice synthesiser, an in-process model that registered its own release with the lane controller.
Her MLX client's evictor knew only MLX workers and sent every other owner to the process evictor, which cannot unload
a model inside her own process, so the eviction was refused without the synthesiser ever being asked.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import core.brain.llm.mlx_client as mlx
from core.runtime.model_lane_control import register_model_lane_owner_adapter, unregister_model_lane_owner_adapter

OWNER = "in-process-sync:4242:voice-duplex-tts:1"


def test_an_in_process_owner_is_asked_to_release(monkeypatch):
    asked = []

    async def release(owner, reason):
        asked.append((owner.owner_id, reason))
        return True

    monkeypatch.setattr(mlx, "_clients_snapshot", lambda: [])
    register_model_lane_owner_adapter(OWNER, evict=release)
    try:
        owner = SimpleNamespace(owner_id=OWNER, model_path="voice-duplex/tts/pocket-kokoro-piper")
        assert asyncio.run(mlx._evict_model_lane_owner(owner, "room for her cortex")) is True
    finally:
        unregister_model_lane_owner_adapter(OWNER)
    assert asked == [(OWNER, "room for her cortex")]


def test_its_refusal_is_its_own(monkeypatch):
    async def busy(owner, reason):
        return False  # mid-synthesis

    monkeypatch.setattr(mlx, "_clients_snapshot", lambda: [])
    register_model_lane_owner_adapter(OWNER, evict=busy)
    try:
        owner = SimpleNamespace(owner_id=OWNER, model_path="voice-duplex/tts/pocket-kokoro-piper")
        assert asyncio.run(mlx._evict_model_lane_owner(owner, "room for her cortex")) is False
    finally:
        unregister_model_lane_owner_adapter(OWNER)


def test_an_owner_with_no_release_of_its_own_goes_to_the_process_evictor(monkeypatch):
    import core.runtime.model_lane_control as lanes

    sent = []

    async def process_evictor(owner, reason):
        sent.append(owner.owner_id)
        return True

    monkeypatch.setattr(mlx, "_clients_snapshot", lambda: [])
    monkeypatch.setattr(lanes, "evict_managed_process_owner", process_evictor)
    owner = SimpleNamespace(owner_id="pid:777:some-worker", model_path="/models/x")
    assert asyncio.run(mlx._evict_model_lane_owner(owner, "room")) is True
    assert sent == ["pid:777:some-worker"]
