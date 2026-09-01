from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest


def _resident_client(monkeypatch):
    from core.brain.llm import mlx_client

    client = object.__new__(mlx_client.MLXLocalClient)
    client.model_path = "/tmp/resident-model"
    client._worker_identity = {
        "schema": "test.worker.identity.v1",
        "worker_boot_id": "1" * 32,
        "worker_pid": 1234,
        "worker_model_path": "/tmp/resident-model",
        "worker_source_sha256": "2" * 64,
    }
    client._shutting_down = False
    client._init_done = True
    client._process = SimpleNamespace(pid=1234, is_alive=lambda: True)
    client._active_generations = 0
    client._active_generation_started_at = 0.0
    client._warmup_in_flight = False
    client._request_lock = threading.Lock()
    client._request_lock_state_lock = threading.RLock()
    client._request_lock_owner_label = ""
    client._request_lock_owner_token = ""
    client._request_lock_acquired_at = 0.0
    client._detached_worker_requests = {}
    client._pending_generations = {}
    client._current_gen_future = None
    client._current_request_id = ""
    client._foreground_generation_watchdog = None
    client._job_seq_counter = 0
    client._model_lane_fencing_token = 0
    client._model_lane_owner_id = ""
    client._durable_lane_release_owed = False
    client._authorize_job = lambda request, **_kwargs: request
    monkeypatch.setattr(mlx_client, "_foreground_owner_active", lambda: False)
    return client


def test_client_exposes_exact_hash_bound_model_lane_ownership(monkeypatch) -> None:
    client = _resident_client(monkeypatch)
    client._model_lane_state_lock = threading.RLock()
    client._model_lane_owner_id = "mlx:test:resident"
    client._model_lane_fencing_token = 17
    client._model_lane_terminal_receipt_id = "terminal-17"

    receipt = client.get_model_lane_ownership_snapshot()

    assert receipt["schema"] == "aura.mlx_model_lane_ownership.v1"
    assert receipt["exclusive"] is True
    assert receipt["owner_id"] == "mlx:test:resident"
    assert receipt["fencing_token"] == 17
    assert receipt["terminal_receipt_id"] == "terminal-17"
    assert receipt["campaign_pid"] > 0
    assert receipt["worker_pid"] == 1234
    body = dict(receipt)
    receipt_sha256 = body.pop("receipt_sha256")
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert receipt_sha256 == hashlib.sha256(encoded).hexdigest()


def test_client_refuses_partial_or_stale_model_lane_ownership(monkeypatch) -> None:
    client = _resident_client(monkeypatch)
    client._model_lane_state_lock = threading.RLock()
    client._model_lane_owner_id = "mlx:test:resident"
    client._model_lane_fencing_token = 17
    client._model_lane_terminal_receipt_id = "terminal-17"
    client._process = SimpleNamespace(pid=9999, is_alive=lambda: True)

    assert client.get_model_lane_ownership_snapshot() == {}


def _valid_worker_response(client, request, *, hidden_states=None):
    from core.brain.llm.latent_cortex.runtime_identity import worker_model_basis
    from core.brain.llm.mlx_worker import (
        _HIDDEN_SEQUENCE_MAX_INPUT_CHARS,
        _HIDDEN_SEQUENCE_MAX_TOKENS,
        _HIDDEN_SEQUENCE_MAX_WIDTH,
    )

    token_ids = [11, 12]
    states = np.asarray(
        hidden_states if hidden_states is not None else [[1.0, 0.0], [0.0, 1.0]],
        dtype="<f4",
    )
    hidden_state_bytes = states.tobytes(order="C")
    return {
        "id": request["id"],
        "action": "encode_hidden_sequence",
        "status": "ok",
        "token_ids": token_ids,
        "hidden_state_bytes": hidden_state_bytes,
        "hidden_shape": [2, 2],
        "hidden_dtype": "float32_le",
        "receipt": {
            "schema": "aura.hidden_sequence_encoding.v1",
            "request_id": request["id"],
            "action": "encode_hidden_sequence",
            "input_char_count": len(request["text"]),
            "token_count": len(token_ids),
            "hidden_size": 2,
            "hidden_state_bytes": len(hidden_state_bytes),
            "hidden_state_sha256": hashlib.sha256(hidden_state_bytes).hexdigest(),
            "transport": "packed_float32_le",
            "limits": {
                "max_input_chars": _HIDDEN_SEQUENCE_MAX_INPUT_CHARS,
                "max_tokens": _HIDDEN_SEQUENCE_MAX_TOKENS,
                "max_hidden_size": _HIDDEN_SEQUENCE_MAX_WIDTH,
            },
            "model_basis": worker_model_basis(client.get_worker_identity_snapshot()),
            "forward_passes": 1,
            "causal_full_sequence": True,
            "sampling": False,
            "generated_tokens": 0,
            "generated_text": False,
        },
    }


def test_worker_encodes_every_token_with_one_non_generative_forward(monkeypatch) -> None:
    from core.brain import nonparametric_generation
    from core.brain.llm.mlx_worker import _encode_hidden_sequence_response

    calls: list[list[int]] = []

    class Encoder:
        def __init__(self, model, tokenizer) -> None:
            assert model == "model"
            assert tokenizer is tokenization

        def encode_hidden_sequence_ids(self, token_ids):
            calls.append(list(token_ids))
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    tokenization = SimpleNamespace(encode=lambda text: [7, 9])
    monkeypatch.setattr(nonparametric_generation, "MLXEncoder", Encoder)
    response = _encode_hidden_sequence_response(
        model="model",
        tokenizer=tokenization,
        text="add the values",
        request_id="request-7",
        encoder_cache={},
        worker_identity={"worker_boot_id": "basis-1"},
        metal_semaphore=contextlib.nullcontext(),
    )

    assert calls == [[7, 9]]
    assert response["token_ids"] == [7, 9]
    assert response["hidden_shape"] == [2, 2]
    assert response["hidden_dtype"] == "float32_le"
    assert np.frombuffer(response["hidden_state_bytes"], dtype="<f4").reshape(2, 2).tolist() == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert response["receipt"]["forward_passes"] == 1
    assert response["receipt"]["sampling"] is False
    assert response["receipt"]["generated_tokens"] == 0
    assert response["receipt"]["generated_text"] is False
    assert response["receipt"]["model_basis"] == {"worker_boot_id": "basis-1"}


def test_worker_refuses_token_overflow_before_model_forward(monkeypatch) -> None:
    from core.brain import nonparametric_generation
    from core.brain.llm.mlx_worker import (
        _HIDDEN_SEQUENCE_MAX_TOKENS,
        _encode_hidden_sequence_response,
    )

    class Encoder:
        def __init__(self, *_args) -> None:
            raise AssertionError("token overflow must not construct the encoder")

    monkeypatch.setattr(nonparametric_generation, "MLXEncoder", Encoder)
    tokenizer = SimpleNamespace(
        encode=lambda _text: list(range(_HIDDEN_SEQUENCE_MAX_TOKENS + 1))
    )
    with pytest.raises(ValueError, match="exceeds 512 tokens"):
        _encode_hidden_sequence_response(
            model="model",
            tokenizer=tokenizer,
            text="bounded characters, excessive tokens",
            request_id="request-overflow",
            encoder_cache={},
            worker_identity={},
            metal_semaphore=contextlib.nullcontext(),
        )


def test_client_returns_validated_hidden_sequence_and_exact_receipt(monkeypatch) -> None:
    from core.brain.llm.latent_cortex.runtime_identity import worker_model_basis

    client = _resident_client(monkeypatch)
    captured: dict[str, object] = {}

    def authorize(request, *, principal):
        captured["principal"] = principal
        return request

    class ReplyingQueue:
        def put(self, request, *_args):
            captured["request"] = dict(request)
            client._pending_generations[request["id"]].set_result(
                _valid_worker_response(client, request)
            )

    client._authorize_job = authorize
    client._req_q = ReplyingQueue()
    result = asyncio.run(client.encode_hidden_sequence("compose this operation"))

    assert result is not None
    assert result["token_ids"] == [11, 12]
    np.testing.assert_array_equal(
        result["hidden_states"],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    assert result["receipt"]["model_basis"] == worker_model_basis(
        client._worker_identity
    )
    assert captured["principal"] == "mlx_client.encode_hidden_sequence"
    request = captured["request"]
    assert isinstance(request, dict)
    assert request["action"] == "encode_hidden_sequence"
    assert request["text"] == "compose this operation"
    assert isinstance(request["id"], str) and request["id"]
    assert client._active_generations == 0
    assert client._pending_generations == {}
    assert not client._request_lock.locked()


def test_parent_only_capture_attestation_does_not_change_model_basis(monkeypatch) -> None:
    client = _resident_client(monkeypatch)
    child_identity = dict(client._worker_identity)
    client._worker_identity["worker_action_capture_origin_binding"] = {
        "schema": "parent-only-attestation"
    }

    class ReplyingQueue:
        def put(self, request, *_args):
            response = _valid_worker_response(client, request)
            response["receipt"]["model_basis"] = child_identity
            client._pending_generations[request["id"]].set_result(response)

    client._req_q = ReplyingQueue()
    result = asyncio.run(client.encode_hidden_sequence("same neural basis"))

    assert result is not None
    assert "worker_action_capture_origin_binding" not in result["receipt"]["model_basis"]


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (
            lambda response: response.update(
                hidden_state_bytes=np.asarray(
                    [[float("nan"), 0.0], [0.0, 1.0]], dtype="<f4"
                ).tobytes()
            ),
            "not finite",
        ),
        (
            lambda response: response.update(hidden_shape=[1, 2]),
            "shape or dtype",
        ),
        (
            lambda response: response["receipt"].update(forward_passes=2),
            "receipt does not match",
        ),
    ],
)
def test_client_rejects_malformed_worker_response(
    monkeypatch,
    mutate,
    expected,
) -> None:
    client = _resident_client(monkeypatch)

    class ReplyingQueue:
        def put(self, request, *_args):
            response = _valid_worker_response(client, request)
            mutate(response)
            client._pending_generations[request["id"]].set_result(response)

    client._req_q = ReplyingQueue()
    with pytest.raises(RuntimeError, match=expected):
        asyncio.run(client.encode_hidden_sequence("inspect the sequence"))


def test_client_refuses_immediately_while_worker_lane_is_busy(monkeypatch) -> None:
    client = _resident_client(monkeypatch)
    client._req_q = queue.Queue()
    assert client._request_lock.acquire(False)
    client._request_lock_owner_label = "foreground_generation"

    assert asyncio.run(client.encode_hidden_sequence("do not queue this")) is None
    assert client._req_q.empty()
    assert client._active_generations == 0
    assert client._request_lock_owner_label == "foreground_generation"


@pytest.mark.parametrize("field, wrong", [("id", "another-request"), ("action", "generate")])
def test_client_rejects_response_identity_mismatch(monkeypatch, field, wrong) -> None:
    client = _resident_client(monkeypatch)

    class ReplyingQueue:
        def put(self, request, *_args):
            response = _valid_worker_response(client, request)
            response[field] = wrong
            client._pending_generations[request["id"]].set_result(response)

    client._req_q = ReplyingQueue()
    with pytest.raises(RuntimeError, match="response identity mismatch"):
        asyncio.run(client.encode_hidden_sequence("bind this response"))


def test_client_enforces_character_bound_before_touching_worker(monkeypatch) -> None:
    from core.brain.llm.mlx_worker import _HIDDEN_SEQUENCE_MAX_INPUT_CHARS

    client = _resident_client(monkeypatch)
    client._req_q = queue.Queue()
    with pytest.raises(ValueError, match="exceeds 4096 characters"):
        asyncio.run(
            client.encode_hidden_sequence("x" * (_HIDDEN_SEQUENCE_MAX_INPUT_CHARS + 1))
        )
    assert client._req_q.empty()
    assert client._active_generations == 0


def test_hidden_sequence_terminal_action_is_routed() -> None:
    from core.brain.llm.mlx_client import _TERMINAL_WORKER_ACTIONS

    assert "encode_hidden_sequence" in _TERMINAL_WORKER_ACTIONS
