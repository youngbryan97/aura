"""A forward that reads her model is not steered and does not teach her affect."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")


class _Block:
    def __call__(self, x):
        return x


def _steered_block(alpha: float = 0.3):
    from core.consciousness.affective_steering import AffectiveSteeringHook

    block = _Block()
    hook = AffectiveSteeringHook(block=block, layer_idx=3, vectors={}, alpha=alpha)
    hook.install()
    hook.override_composite_vector(np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
    hook._last_substrate_sync_monotonic = time.monotonic()
    return block, hook


def _states():
    return mx.array(np.asarray([[[1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]]], dtype=np.float32))


def test_her_steering_moves_the_last_position_of_an_ordinary_forward() -> None:
    block, _hook = _steered_block()
    before = np.asarray(_states())
    after = np.asarray(block(_states()))
    assert np.array_equal(after[0, :2], before[0, :2])
    assert not np.array_equal(after[0, 2], before[0, 2])


def test_inside_an_observation_pass_the_same_forward_is_bit_identical() -> None:
    from core.runtime.observation_pass import in_observation_pass, observation_pass

    block, hook = _steered_block()
    with observation_pass():
        assert in_observation_pass()
        assert hook._effective_alpha() == 0.0
        after = np.asarray(block(_states()))
    assert not in_observation_pass()
    assert np.array_equal(after, np.asarray(_states()))
    assert hook._effective_alpha() == pytest.approx(0.3)


def test_an_observation_pass_on_one_thread_leaves_a_generation_on_another_steered() -> None:
    from core.runtime.observation_pass import observation_pass

    block, hook = _steered_block()
    seen: dict[str, float] = {}

    def generation() -> None:
        seen["alpha"] = hook._effective_alpha()

    with observation_pass():
        thread = threading.Thread(target=generation)
        thread.start()
        thread.join()
    assert seen["alpha"] == pytest.approx(0.3)


def test_latent_readouts_do_not_learn_from_an_observation_pass() -> None:
    from core.consciousness.latent_bridge import LatentReadoutHook
    from core.runtime.observation_pass import observation_pass

    block = _Block()
    vector = SimpleNamespace(substrate_idx=0,
                             get_mx_array=lambda dtype=None: mx.array(np.ones(4, dtype=np.float32)))
    hook = LatentReadoutHook(block=block, layer_idx=3, steering_vectors={"valence": vector})
    hook.install()
    with observation_pass():
        block(_states())
    assert hook._readout_count == 0
    block(_states())
    assert hook._readout_count == 1


def test_the_worker_encodes_inside_an_observation_pass_and_names_that_basis(monkeypatch) -> None:
    from core.brain import nonparametric_generation
    from core.brain.llm.mlx_worker import _encode_hidden_sequence_response
    from core.runtime.observation_pass import in_observation_pass

    during: list[bool] = []

    class Encoder:
        def __init__(self, model, tokenizer) -> None:
            pass

        def encode_hidden_sequence_ids(self, token_ids):
            during.append(in_observation_pass())
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(nonparametric_generation, "MLXEncoder", Encoder)
    response = _encode_hidden_sequence_response(
        model="model", tokenizer=SimpleNamespace(encode=lambda text: [7, 9]),
        text="add the values", request_id="request-7", encoder_cache={},
        worker_identity={"worker_boot_id": "basis-1", "worker_affective_steering_active": True,
                         "worker_affective_steering_alpha": 0.3},
        metal_semaphore=contextlib.nullcontext(),
    )
    assert during == [True]
    assert not in_observation_pass()
    assert response["receipt"]["model_basis"] == {
        "worker_boot_id": "basis-1", "worker_affective_steering_active": False,
        "worker_affective_steering_alpha": 0.0}


def test_the_client_accepts_an_observation_while_her_steering_is_up_and_refuses_a_steered_one(
    monkeypatch,
) -> None:
    from core.brain.llm.latent_cortex.runtime_identity import worker_model_basis
    from tests.test_mlx_hidden_sequence_ipc import _resident_client, _valid_worker_response

    client = _resident_client(monkeypatch)
    client._worker_identity.update(worker_affective_steering_active=True,
                                   worker_affective_steering_alpha=0.3)

    def run(steered: bool):
        class ReplyingQueue:
            def put(self, request, *_args):
                response = _valid_worker_response(client, request)
                if steered:
                    response["receipt"]["model_basis"] = worker_model_basis(client._worker_identity)
                client._pending_generations[request["id"]].set_result(response)

        client._req_q = ReplyingQueue()
        return asyncio.run(client.encode_hidden_sequence("compose this operation"))

    result = run(steered=False)
    assert result["receipt"]["model_basis"]["worker_affective_steering_active"] is False
    with pytest.raises(RuntimeError, match="receipt does not match"):
        run(steered=True)
