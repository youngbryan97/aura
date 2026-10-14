"""Speculation with her model's next-next-token head keeps exactly the tokens greedy decoding keeps."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")


@pytest.fixture
def on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _tiny(vocab: int, seed: int):
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    mx.random.seed(seed)
    text = {"model_type": "qwen3_5_text", "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 8,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "vocab_size": vocab,
            "linear_num_value_heads": 4, "linear_num_key_heads": 2, "linear_key_head_dim": 8,
            "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.5}}
    model = Model(ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": text}))
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_speculation_keeps_exactly_the_greedy_tokens_through_accepted_and_rejected_drafts(on_cpu, seed) -> None:
    from core.brain.llm.two_tokens_a_step import MTPHead, plain_greedy, speculative_greedy

    model = _tiny(vocab=5, seed=seed)
    head = MTPHead(model.language_model.args)
    mx.eval(head.parameters())
    prompt = [1, 3, 0, 2, 4, 1, 1, 3]
    stats: dict = {}
    spoken = list(speculative_greedy(model, head, prompt, 200, stats=stats))
    assert spoken == plain_greedy(model, prompt, 200)
    # Both paths ran: some drafts accepted (two tokens a pass), some rejected and unread.
    assert 0 < stats["accepted"] < stats["passes"] and stats["tokens"] == 200


def test_a_stop_token_ends_speculation_where_greedy_decoding_ends(on_cpu) -> None:
    from core.brain.llm.two_tokens_a_step import MTPHead, plain_greedy, speculative_greedy

    model = _tiny(vocab=5, seed=4)
    head = MTPHead(model.language_model.args)
    mx.eval(head.parameters())
    prompt = [2, 2, 1]
    for stop in ([0], [3], [4]):
        assert list(speculative_greedy(model, head, prompt, 120, stop=stop)) == plain_greedy(model, prompt, 120, stop)
