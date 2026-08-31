"""Continued generation shares its consumer, cache ledger and measured cost."""

from types import SimpleNamespace

import pytest

from core.brain.llm.mlx_worker import _GenerationPasses, _generation_pass_performance


def test_continuation_closes_previous_generator_before_starting_next():
    events = []

    def generate(tap, prompt, kwargs):
        events.append(("start", prompt, tap))
        try:
            for token in kwargs["tokens"]:
                yield SimpleNamespace(token=token)
        finally:
            events.append(("close", prompt))

    passes = _GenerationPasses(generate, "capture", "question", {"tokens": [1, 2, 3]})
    consumed = []
    for response in passes:
        consumed.append(response.token)
        if response.token == 2:
            passes.continue_with("boundary", {"tokens": [4, 5]})
    assert consumed == [1, 2, 4, 5]
    assert events == [
        ("start", "question", "capture"), ("close", "question"),
        ("start", "boundary", None), ("close", "boundary"),
    ]
    assert [response.token for response in passes.final_responses] == [2, 5]


def test_closing_consumer_closes_active_continuation():
    closed = []

    def generate(tap, prompt, kwargs):
        try:
            yield prompt
            yield "unconsumed"
        finally:
            closed.append(prompt)

    passes = _GenerationPasses(generate, None, "first", {})
    stream = iter(passes)
    assert next(stream) == "first"
    passes.continue_with("second", {})
    assert next(stream) == "second"
    stream.close()
    assert closed == ["first", "second"]


def test_all_passes_contribute_to_performance():
    responses = [
        SimpleNamespace(prompt_tokens=100, prompt_tps=50, generation_tokens=10,
                        generation_tps=5, peak_memory=3, finish_reason=""),
        SimpleNamespace(prompt_tokens=2, prompt_tps=10, generation_tokens=20,
                        generation_tps=4, peak_memory=4, finish_reason="stop"),
    ]
    result = _generation_pass_performance(
        responses, prompt_tokens=100, generation_tokens=30,
        first_token_seconds=2, stream_seconds=9.2,
    )
    assert result["prompt_tokens"] == 102
    assert result["generation_tokens"] == 30
    assert result["prefill_seconds"] == pytest.approx(2.2)
    assert result["decode_seconds"] == 7
    assert result["generation_tps"] == pytest.approx(30 / 7)
    assert result["peak_memory_gb"] == 4
    assert result["finish_reason"] == "stop"
    assert result["generation_passes"] == 2


def test_hybrid_continuation_cache_matches_full_token_ledger():
    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    config = {
        "model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6, "vocab_size": 128, "max_position_embeddings": 256,
        "linear_num_value_heads": 4, "linear_num_key_heads": 2,
        "linear_key_head_dim": 16, "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 2, "full_attention_interval": 2,
        "head_dim": 16, "tie_word_embeddings": False,
    }
    model = Model(ModelArgs(model_type="qwen3_5", text_config=config))
    cache = model.make_cache()
    ledger = [3, 5, 7]

    def generate(tap, prompt, kwargs):
        mx.eval(model(mx.array([prompt]), cache=cache))
        for token in kwargs["tokens"]:
            mx.eval(model(mx.array([[token]]), cache=cache))
            yield SimpleNamespace(token=token)

    passes = _GenerationPasses(generate, None, ledger[:], {"tokens": [11, 13, 15]})
    for response in passes:
        ledger.append(response.token)
        if response.token == 13:
            ledger.extend([17, 19])
            passes.continue_with([17, 19], {"tokens": [23, 29]})
    assert ledger == [3, 5, 7, 11, 13, 17, 19, 23, 29]
    reference = model.make_cache()
    # Identical step boundaries exclude reduction-order differences.
    for group in ([3, 5, 7], [11], [13], [17, 19], [23], [29]):
        mx.eval(model(mx.array([group]), cache=reference))
    continued = model(mx.array([[31]]), cache=cache)
    expected = model(mx.array([[31]]), cache=reference)
    mx.eval(continued, expected)
    assert mx.array_equal(continued, expected).item()
