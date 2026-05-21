from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest

from core.brain.homeostatic_modulator import InferenceModulation
from core.brain.local_llm import LocalBrain
from core.brain.unified_inference import UnifiedInferenceEngine
from core.runtime.errors import get_degradation_tracker


@pytest.mark.asyncio
async def test_ensure_identity_anchor():
    engine = UnifiedInferenceEngine()

    # Case 1: Empty messages
    messages = []
    engine._ensure_identity_anchor(messages)
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert "You are Aura Luna" in messages[0]["content"]

    # Case 2: System message exists but lacks identity anchor
    messages = [{"role": "system", "content": "Keep it technical."}]
    engine._ensure_identity_anchor(messages)
    assert len(messages) == 1
    assert "You are Aura Luna" in messages[0]["content"]
    assert "Keep it technical" in messages[0]["content"]

    # Case 3: System message already has the anchor
    existing = "You are Aura Luna. Speak with direct first-person continuity... Keep it technical."
    messages = [{"role": "system", "content": existing}]
    engine._ensure_identity_anchor(messages)
    assert len(messages) == 1
    assert messages[0]["content"] == existing


@pytest.mark.asyncio
async def test_run_ollama_fallback():
    engine = UnifiedInferenceEngine()
    modulation = InferenceModulation(
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1,
        logit_bias={123: 1.0},
        head_weights=np.ones(32),
        urgency=0.5,
    )

    mock_brain_instance = MagicMock()
    mock_brain_instance.chat = AsyncMock(
        return_value={"response": "This is a response from Ollama fallback", "thought": "thought"}
    )
    mock_brain_instance.__aenter__ = AsyncMock(return_value=mock_brain_instance)
    mock_brain_instance.__aexit__ = AsyncMock(return_value=None)

    with patch("core.brain.local_llm.LocalBrain", return_value=mock_brain_instance):
        result = await engine._run_ollama_fallback(
            messages=[{"role": "user", "content": "hello"}],
            model_name="default_model",
            modulation=modulation,
            options=None,
        )

        assert result["response"] == "This is a response from Ollama fallback"
        assert result["thought"] == "thought"

        # Verify chat call options had modulated params
        called_options = mock_brain_instance.chat.call_args[1]["options"]
        assert called_options["temperature"] == 0.8
        assert called_options["top_p"] == 0.95
        assert called_options["repeat_penalty"] == 1.1


@pytest.mark.asyncio
async def test_generate_unified_routing():
    engine = UnifiedInferenceEngine()

    # Mock all model registry functions
    mock_registry = {
        "get_local_backend": lambda: "llama_cpp",
        "get_lane_model_name": lambda name: "model_name",
        "get_lane_runtime_model_path": lambda name: "model_path.gguf",
        "get_lane_context_window": lambda name: 8192,
    }

    # Helper for running direct vs fallback
    with patch.multiple("core.brain.llm.model_registry", **mock_registry):
        # Case 1: llama_cpp backend available, gguf loaded
        mock_llama = MagicMock()
        mock_llama.create_chat_completion = MagicMock(
            return_value={
                "choices": [
                    {
                        "message": {"content": "<think>thinking process</think>final answer"},
                        "logprobs": {"content": []},
                    }
                ]
            }
        )

        with patch.object(engine, "_get_llama_instance", return_value=mock_llama):
            res = await engine.generate_unified(prompt="test prompt")
            assert res["response"] == "final answer"
            assert res["thought"] == "thinking process"

        # Case 2: llama_cpp backend, but model instance load fails -> falls back to Ollama
        mock_fallback = AsyncMock(return_value={"response": "fallback answer", "thought": ""})
        with patch.object(engine, "_get_llama_instance", return_value=None):
            with patch.object(engine, "_run_ollama_fallback", mock_fallback):
                res = await engine.generate_unified(prompt="test prompt")
                assert res["response"] == "fallback answer"
                mock_fallback.assert_called_once()


@pytest.mark.asyncio
async def test_local_brain_records_unified_generate_fallback(monkeypatch):
    import core.brain.unified_inference as unified_inference

    get_degradation_tracker().reset()

    class FailingUnifiedInference:
        async def generate_unified(self, **_kwargs):
            self.was_called = True
            raise RuntimeError("unified unavailable")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "<think>private scratch</think>visible answer"}

    class FakeClient:
        async def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(
        unified_inference,
        "UnifiedInferenceEngine",
        FailingUnifiedInference,
    )

    brain = LocalBrain(model_name="test-model")
    brain._client = FakeClient()

    result = await brain.generate("hello")

    assert result == {"response": "visible answer", "thought": "private scratch"}
    assert any(
        "fell back to raw Ollama generation" in record.action
        for record in get_degradation_tracker().recent(
            subsystem="local_llm_unified_fallback", limit=5
        )
    )


@pytest.mark.asyncio
async def test_local_brain_stream_failure_yields_visible_marker():
    get_degradation_tracker().reset()

    class FailingStream:
        async def __aenter__(self):
            self.entered = True
            raise httpx.ConnectError("stream offline")

        async def __aexit__(self, *_args):
            return None

    class FakeClient:
        def stream(self, *_args, **_kwargs):
            return FailingStream()

    brain = LocalBrain(model_name="test-model")
    brain._client = FakeClient()

    chunks = [chunk async for chunk in brain.generate_text_stream_async("hello")]

    assert chunks and "Sovereign stream interrupted" in chunks[-1]
    assert any(
        "yielded stream interruption marker" in record.action
        for record in get_degradation_tracker().recent(subsystem="local_llm", limit=5)
    )
