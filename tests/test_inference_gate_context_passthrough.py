from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_inference_gate_think_forwards_live_state_context():
    from core.brain.inference_gate import InferenceGate

    gate = InferenceGate()
    captured = {}
    live_state = SimpleNamespace(cognition=SimpleNamespace(modifiers={}))

    async def fake_generate(prompt, *, context=None, timeout=None):
        captured["prompt"] = prompt
        captured["context"] = context or {}
        captured["timeout"] = timeout
        return "ok"

    gate.generate = fake_generate
    gate._post_inference_update = lambda _result: None

    result = await gate.think(
        "huh",
        origin="api",
        foreground_request=True,
        state=live_state,
        skip_runtime_payload=True,
    )

    assert result == "ok"
    assert captured["context"]["state"] is live_state
    assert captured["context"]["skip_runtime_payload"] is True
