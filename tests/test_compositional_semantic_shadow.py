from types import SimpleNamespace

import numpy as np
import pytest

from core.brain.llm import compositional_semantic_shadow as shadow
from core.learning.semantic_program_runtime import (
    SemanticProgramDecodeRejectedError,
    SemanticProgramObservationError,
)


class _Client:
    model_path = "/models/aura-27b"

    def __init__(self, observation):
        self.observation = observation
        self.calls = []

    async def encode_hidden_sequence(self, text, *, timeout_s, representation):
        self.calls.append((text, timeout_s, representation))
        return self.observation


def _status():
    return {
        "available": True,
        "mode": "shadow",
        "serving_authority": False,
        "representation_basis_sha256": "a" * 64,
        "transducer_receipt_sha256": "b" * 64,
        "receipt_sha256": "c" * 64,
    }


def _observation(token_ids=None):
    tokens = [10, 11, 12] if token_ids is None else token_ids
    return {
        "token_ids": tokens,
        "hidden_states": np.ones((len(tokens), 2), dtype=np.float32),
        "receipt": {
            "model_basis": {"worker_model_path": "/models/aura-27b"},
            "hidden_state_sha256": "d" * 64,
        },
    }


def _install_runtime_stubs(monkeypatch, *, token_ids=None):
    tokens = [10, 11, 12] if token_ids is None else token_ids
    monkeypatch.setattr(shadow, "compositional_semantic_shadow_status", lambda _path: _status())
    monkeypatch.setattr(shadow, "_load_offset_tokenizer", lambda _path: object())
    monkeypatch.setattr(
        shadow,
        "tokenize_with_offsets",
        lambda _tokenizer, _text: (tokens, [(0, 3), (4, 5), (6, 7)]),
    )
    monkeypatch.setattr(
        shadow,
        "_load_transducer",
        lambda *_args: SimpleNamespace(max_inputs=4),
    )


@pytest.mark.asyncio
async def test_shadow_does_not_touch_the_model_without_public_inputs(monkeypatch):
    client = _Client(_observation())
    monkeypatch.setattr(
        shadow,
        "compositional_semantic_shadow_status",
        lambda _path: pytest.fail("status must not be read for an ineligible prompt"),
    )

    result = await shadow.execute_compositional_semantic_shadow(
        client=client,
        prompt="Explain the shortest path invariant.",
    )

    assert result == {
        "eligible": False,
        "attempted": False,
        "ok": False,
        "reason": "compositional_semantic_no_public_inputs",
    }
    assert client.calls == []


@pytest.mark.asyncio
async def test_shadow_executes_on_the_existing_worker_without_answer_authority(monkeypatch):
    client = _Client(_observation())
    _install_runtime_stubs(monkeypatch)
    outcome = SimpleNamespace(
        execution=SimpleNamespace(result=7),
        receipt={"receipt_sha256": "e" * 64},
    )
    monkeypatch.setattr(
        shadow,
        "execute_compositional_semantic_observation",
        lambda **_kwargs: outcome,
    )

    result = await shadow.execute_compositional_semantic_shadow(
        client=client,
        prompt="Add 3 and 4.",
        timeout_s=9.0,
    )

    assert client.calls == [("Add 3 and 4.", 9.0, "lexical_mid_final_v1")]
    assert result["ok"] is True
    assert result["result"] == 7
    assert result["text"] == "7"
    assert result["mode"] == "shadow"
    assert result["serving_authority"] is False


@pytest.mark.asyncio
async def test_shadow_treats_resident_lane_contention_as_backpressure(monkeypatch):
    client = _Client(None)
    _install_runtime_stubs(monkeypatch)

    result = await shadow.execute_compositional_semantic_shadow(
        client=client,
        prompt="Add 3 and 4.",
    )

    assert result["eligible"] is True
    assert result["attempted"] is False
    assert result["reason"] == "compositional_semantic_resident_lane_busy"


@pytest.mark.asyncio
async def test_shadow_rejects_local_worker_token_disagreement(monkeypatch):
    client = _Client(_observation([10, 99, 12]))
    _install_runtime_stubs(monkeypatch, token_ids=[10, 11, 12])

    with pytest.raises(RuntimeError, match="local and worker tokens differ"):
        await shadow.execute_compositional_semantic_shadow(
            client=client,
            prompt="Add 3 and 4.",
        )


@pytest.mark.asyncio
async def test_shadow_preserves_a_neural_decode_rejection(monkeypatch):
    client = _Client(_observation())
    _install_runtime_stubs(monkeypatch)

    def _reject(**_kwargs):
        raise SemanticProgramDecodeRejectedError("typed_argument_chart_empty")

    monkeypatch.setattr(shadow, "execute_compositional_semantic_observation", _reject)

    result = await shadow.execute_compositional_semantic_shadow(
        client=client,
        prompt="Add 3 and 4.",
    )

    assert result["eligible"] is True
    assert result["attempted"] is True
    assert result["ok"] is False
    assert result["reason"] == "typed_argument_chart_empty"
    assert result["activation_receipt"]["serving_authority"] is False


@pytest.mark.asyncio
async def test_shadow_reports_neural_basis_drift_without_raising(monkeypatch):
    client = _Client(_observation())
    _install_runtime_stubs(monkeypatch)

    def _reject(**_kwargs):
        raise SemanticProgramObservationError(
            "compositional semantic representation basis differs"
        )

    monkeypatch.setattr(shadow, "execute_compositional_semantic_observation", _reject)

    result = await shadow.execute_compositional_semantic_shadow(
        client=client,
        prompt="Add 3 and 4.",
    )

    assert result["eligible"] is True
    assert result["attempted"] is True
    assert result["ok"] is False
    assert result["reason"] == "compositional semantic representation basis differs"
    assert result["observed_representation_basis"] == {
        "worker_model_path": "/models/aura-27b"
    }
    assert len(result["observed_representation_basis_sha256"]) == 64
    assert result["expected_representation_basis_sha256"] == "a" * 64
    assert result["activation_receipt"]["serving_authority"] is False


@pytest.mark.asyncio
async def test_resident_shadow_never_constructs_a_missing_model_client(monkeypatch, tmp_path):
    from core.brain.llm import mlx_client, model_registry

    monkeypatch.setattr(model_registry, "get_runtime_model_path", lambda: str(tmp_path))
    monkeypatch.setattr(mlx_client, "clients_snapshot", lambda: [])
    monkeypatch.setattr(
        shadow,
        "compositional_semantic_shadow_status",
        lambda _path: _status(),
    )

    result = await shadow.observe_resident_compositional_semantics("Add 3 and 4.")

    assert result["attempted"] is False
    assert result["reason"] == "compositional_semantic_resident_client_missing"
    assert shadow.compositional_semantic_shadow_observations()[-1]["reason"] == result[
        "reason"
    ]


def test_shadow_can_be_disabled_without_reading_artifacts(monkeypatch):
    monkeypatch.setenv("AURA_COMPOSITIONAL_SEMANTIC_SHADOW", "off")

    assert shadow.compositional_semantic_shadow_status("/missing") == {
        "available": False,
        "reason": "compositional_semantic_shadow_disabled",
    }
