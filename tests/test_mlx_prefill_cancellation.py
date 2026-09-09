"""Stop reaches the model before its first decoded token."""

from contextlib import contextmanager
from multiprocessing import RawValue
from types import SimpleNamespace

import pytest

from core.brain.llm.mlx_client import MLXLocalClient
from core.brain.llm.mlx_worker import (
    _PrefillCancelledError,
    _build_prefill_progress_callback,
    _generation_stream_with_activity,
    soft_cancel_requested,
)


@pytest.mark.parametrize("action", ["generate", "stream"])
def test_prefill_stop_unwinds_before_a_request_bound_ack(action):
    events = []
    cancel = RawValue("q", 7)
    callback = _build_prefill_progress_callback(
        SimpleNamespace(activity=lambda: events.append("activity")),
        SimpleNamespace(put=lambda message: events.append(message)),
        request_id="the-cancelled-turn", action=action,
        cancel_check=lambda: soft_cancel_requested(cancel, 7),
        snapshot_at=4, keep_prefix=lambda _count: events.append("snapshot"),
    )

    @contextmanager
    def tap():
        try:
            yield
        finally:
            events.append("restored")

    def generate(*_args, **_kwargs):
        try:
            callback(4, 64)
            yield "must not decode"
        finally:
            events.append("closed")

    stream = _generation_stream_with_activity(
        generate, None, None, prompt="input", generation_kwargs={},
        watchdog=SimpleNamespace(activity=lambda: None), tap=tap(),
    )
    with pytest.raises(_PrefillCancelledError) as caught:
        next(stream)
    assert events == ["activity", "closed", "restored"]
    frame = caught.value.terminal_frame()
    assert frame["id"] == "the-cancelled-turn"
    assert frame["action"] == ("stream_done" if action == "stream" else action)
    assert frame["text"] == ""
    assert frame["soft_cancelled"] is True
    assert frame["tokens_used"] == 0
    client = object.__new__(MLXLocalClient)
    client._soft_cancel_target = {"req_id": "the-cancelled-turn", "requested_monotonic": 0}
    client._soft_cancel_ack = None
    client._note_soft_cancel_acknowledgement(frame)
    assert client._soft_cancel_ack_matches(client._soft_cancel_target)
    assert not client._soft_cancel_ack_matches({"req_id": "next-turn"})


@pytest.mark.parametrize("cancelled_seq", [0, 6, 8])
def test_another_requests_cancel_does_not_interrupt_prefill(cancelled_seq):
    messages = []
    callback = _build_prefill_progress_callback(
        SimpleNamespace(activity=lambda: None), SimpleNamespace(put=messages.append),
        request_id="current", action="generate",
        cancel_check=lambda: soft_cancel_requested(RawValue("q", cancelled_seq), 7),
    )
    callback(4, 64)
    assert messages[0]["prompt_tokens_processed"] == 4


@pytest.mark.hardware
@pytest.mark.parametrize("cancel_at", [0, 4, 8])
def test_installed_mlx_stops_between_materialized_chunks_and_reuses_the_model(cancel_at):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.qwen2 import Model, ModelArgs

    model = Model(ModelArgs(
        model_type="qwen2", hidden_size=16, num_hidden_layers=1,
        intermediate_size=32, num_attention_heads=2, rms_norm_eps=1e-6,
        vocab_size=32, num_key_value_heads=1, max_position_embeddings=64,
    ))
    cache = make_prompt_cache(model)
    cancel = RawValue("q", 0)
    processed = []
    callback = _build_prefill_progress_callback(
        SimpleNamespace(activity=lambda: None), SimpleNamespace(put=lambda _message: None),
        request_id="real-prefill", action="generate",
        cancel_check=lambda: soft_cancel_requested(cancel, 7),
    )

    def observe(count, total):
        processed.append(count)
        if count == cancel_at:
            cancel.value = 7
        callback(count, total)

    generator = generate_step(
        mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), model,
        prompt_cache=cache, max_tokens=1, prefill_step_size=4,
        prompt_progress_callback=observe,
    )
    with pytest.raises(_PrefillCancelledError) as caught:
        next(generator)
    generator.close()
    assert processed[-1] == cancel_at
    assert cache[0].offset == cancel_at
    assert caught.value.processed == cancel_at
    # The loaded parameters survive; a fresh cache can serve the next request.
    followup = generate_step(mx.array([1, 2]), model, max_tokens=1)
    token, _logprobs = next(followup)
    followup.close()
    assert 0 <= token < 32
