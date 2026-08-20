"""SPARK-051 resident first-action continuation contracts."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.action_continuation import (  # noqa: E402
    ActionContinuationError,
    PortableStateComponent,
)
from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    CortexConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)


def _model():
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=32,
        num_hidden_layers=8,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _engine(model):
    return LatentCortexEngine(
        model,
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=2, seed=3),
            recurrence=RecurrenceConfig(max_steps=2, min_steps=1),
            branches=BranchConfig(n_branches=1, isolation_steps=1),
            prelude_frac=0.25,
            coda_frac=0.25,
            decode_max_tokens=2,
            allow_vanilla_fallback=False,
        ),
    )


def _runner_state(tag: str = "A") -> dict:
    return {
        "durable_state": {"conversation_epoch": 19, "tag": tag},
        "rng_state": {"root_seed": 104_729, "stream": tag},
    }


def test_portable_codec_is_canonical_chunked_and_tensor_exact():
    value = {
        "mlx": mx.array([[1.25, -2.5]], dtype=mx.float16),
        "mlx_bfloat16": mx.array([[0.5, -7.25]], dtype=mx.bfloat16),
        "numpy": np.array([3, 5, 8], dtype=np.int32),
        "nested": ({"z", "a"}, [None, True, -7, float("inf")]),
        "payload": b"resident-state" * 31,
    }
    component = PortableStateComponent.from_value(value)
    chunks = list(component.iter_encoded_chunks(chunk_bytes=37))
    raw = b"".join(chunks)

    assert chunks
    assert max(map(len, chunks)) <= 37
    assert hashlib.sha256(raw).hexdigest() == component.sha256()
    restored = PortableStateComponent.from_bytes(raw).decode()
    np.testing.assert_array_equal(np.asarray(restored["mlx"]), np.asarray(value["mlx"]))
    assert restored["mlx_bfloat16"].dtype == mx.bfloat16
    np.testing.assert_array_equal(
        np.asarray(restored["mlx_bfloat16"].astype(mx.float32)),
        np.asarray(value["mlx_bfloat16"].astype(mx.float32)),
    )
    np.testing.assert_array_equal(restored["numpy"], value["numpy"])
    assert restored["nested"][0] == {"a", "z"}
    assert PortableStateComponent.from_value(restored).sha256() == component.sha256()

    with pytest.raises(ActionContinuationError, match="magic_invalid"):
        PortableStateComponent.from_bytes(b"not-a-continuation")
    with pytest.raises(ActionContinuationError, match="trailing_bytes"):
        PortableStateComponent.from_bytes(raw + b"x").decode()


def test_real_qwen_capture_stops_before_action_and_decode_then_restores_exactly():
    model = _model()
    engine = _engine(model)
    captured = []
    first = engine.reason(
        token_ids=[1, 2, 3, 4],
        action_continuation_capture=captured.append,
        action_continuation_runner_state=_runner_state(),
        action_continuation_capture_only=True,
    )

    assert first.ok is True
    assert first.reason == "action_state_captured"
    assert first.receipt.cognitive_action_trace == []
    assert first.receipt.decode_generated_tokens == 0
    assert first.receipt.last_stage == "action_state_captured"
    assert len(captured) == 1
    continuation = captured[0]
    assert continuation.episode_step == 0
    assert continuation.schedule_step == 0
    assert continuation.branch_id == "branch-0"
    assert set(continuation.state_components) == {
        "branch_state_sha256",
        "durable_state_sha256",
        "evidence_state_sha256",
        "kv_cache_sha256",
        "latent_slots_sha256",
        "memory_state_sha256",
        "public_action_state_sha256",
        "rng_state_sha256",
    }

    restored = []
    second = engine.reason(
        token_ids=[1, 2, 3, 4],
        action_continuation_capture=restored.append,
        action_continuation_restore=continuation,
        action_continuation_runner_state=_runner_state(),
        action_continuation_capture_only=True,
    )
    assert second.ok is True
    assert second.reason == "action_state_captured"
    assert second.receipt.cognitive_action_trace == []
    assert second.receipt.decode_generated_tokens == 0
    assert len(restored) == 1
    assert restored[0].state_components == continuation.state_components
    for name in continuation.private_state:
        assert (
            restored[0].private_state[name].to_bytes()
            == continuation.private_state[name].to_bytes()
        )


def test_runner_state_drift_fails_before_action_or_decode():
    model = _model()
    engine = _engine(model)
    captured = []
    engine.reason(
        token_ids=[1, 2, 3, 4],
        action_continuation_capture=captured.append,
        action_continuation_runner_state=_runner_state(),
        action_continuation_capture_only=True,
    )

    result = engine.reason(
        token_ids=[1, 2, 3, 4],
        action_continuation_capture=lambda _continuation: None,
        action_continuation_restore=captured[0],
        action_continuation_runner_state=_runner_state("DRIFT"),
        action_continuation_capture_only=True,
    )
    assert result.ok is False
    # The reason the caller sees, not the exception message: reasons are
    # published with messages stripped because they can carry local paths and
    # processed text, so what must identify the fault is the class.
    assert result.reason == "latent_phase_failed:ActionContinuationDrift"
    assert result.receipt.cognitive_action_trace == []
    assert result.receipt.decode_generated_tokens == 0

    clean_frame = []
    clean = engine.reason(
        token_ids=[1, 2, 3, 4],
        action_continuation_capture=clean_frame.append,
        action_continuation_runner_state=_runner_state(),
        action_continuation_capture_only=True,
    )
    assert clean.ok is True
    assert len(clean_frame) == 1
