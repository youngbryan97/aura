from __future__ import annotations

import pytest

import core.cognitive_integration_layer as cil_module
from core.cognitive_integration_layer import (
    CognitiveIntegrationLayer,
    _run_inline_inference,
)
from core.cognitive_integration_patch import patch_cognitive_integration


def test_retired_cognitive_integration_patch_is_noop():
    original = CognitiveIntegrationLayer.process_turn

    patch_cognitive_integration()

    assert CognitiveIntegrationLayer.process_turn is original


@pytest.mark.asyncio
async def test_inline_inference_normalizes_router_json(monkeypatch):
    class Router:
        async def think(self, *_args, **_kwargs):
            return (
                '{"implicit_intent":"needs help","user_subtext":"anxious",'
                '"momentum":"LOUD","conversation_hooks":"runtime health"}'
            )

    def service_get(name, default=None):
        return Router() if name == "llm_router" else default

    monkeypatch.setattr(cil_module.ServiceContainer, "get", staticmethod(service_get))

    data = await _run_inline_inference("Can you check this?", [])

    assert data == {
        "implicit_intent": "needs help",
        "user_subtext": "anxious",
        "momentum": "flowing",
        "conversation_hooks": ["runtime health"],
    }


@pytest.mark.asyncio
async def test_process_turn_failure_returns_bounded_response(monkeypatch):
    class Reflex:
        def process(self, _message):
            return None

    class Kernel:
        async def evaluate(self, message, **_kwargs):
            if message:
                raise RuntimeError("kernel unavailable")
            return None

    monkeypatch.setattr(cil_module, "get_reflex", lambda: Reflex())
    monkeypatch.setattr(
        cil_module.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    layer = CognitiveIntegrationLayer()
    layer._initialized = True
    layer.kernel = Kernel()

    response = await layer.process_turn("hello")

    assert "stable conversation path" in response
    assert layer._processing_turn is False
