"""Checking several drafted tokens a pass keeps exactly what she writes greedily, and exactly her sampling distribution."""

from __future__ import annotations

from collections import Counter

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
@pytest.mark.parametrize("with_head", [False, True])
def test_greedy_tokens_are_exactly_greedy_decodings_with_drafts_from_her_text_and_her_head(on_cpu, seed,
                                                                                           with_head) -> None:
    from core.brain.llm.several_tokens_a_pass import Drafts, several_tokens_a_pass
    from core.brain.llm.two_tokens_a_step import MTPHead, plain_greedy

    model = _tiny(vocab=7, seed=seed)
    head = None
    if with_head:
        head = MTPHead(model.language_model.args)
        mx.eval(head.parameters())
    prompt = [1, 3, 0, 2, 4, 1, 3, 0, 2, 5, 6, 1, 3]
    drafts = Drafts(longest=3, depth=5, head=head)
    spoken = list(several_tokens_a_pass(model, prompt, 160, drafts=drafts))
    assert spoken == plain_greedy(model, prompt, 160)
    stats = drafts.stats
    assert stats["from_text"] > 0 and 0 < stats["accepted"] < stats["drafted"]
    assert stats["passes"] < 160, stats


def test_a_stop_token_ends_where_greedy_decoding_ends(on_cpu) -> None:
    from core.brain.llm.several_tokens_a_pass import Drafts, several_tokens_a_pass
    from core.brain.llm.two_tokens_a_step import plain_greedy

    model = _tiny(vocab=7, seed=5)
    prompt = [2, 2, 1, 2, 2, 1]
    for stop in ([0], [3], [4], [6]):
        assert list(several_tokens_a_pass(model, prompt, 100, drafts=Drafts(depth=6), stop=stop)) == \
            plain_greedy(model, prompt, 100, stop)


def _plain_sample(model, prompt, count, sampling):
    from core.brain.llm.several_tokens_a_pass import _logits, _text

    text = _text(model)
    cache = text.make_cache()
    hidden = text.model(mx.array([list(prompt)]), cache)
    out = []
    for _ in range(count):
        probs = sampling.probabilities(_logits(text, hidden[:, -1, :])[0])
        token = int(mx.random.categorical(mx.log(probs)).item())
        out.append(token)
        hidden = text.model(mx.array([[token]]), cache)
    return out


def test_sampled_tokens_follow_her_distribution_exactly(on_cpu) -> None:
    from core.brain.llm.several_tokens_a_pass import Drafts, Sampling, several_tokens_a_pass

    model = _tiny(vocab=5, seed=7)
    prompt = [1, 2, 3, 1, 2, 3, 1, 2]
    sampling = Sampling(temperature=1.0, top_p=0.95)
    runs = 1200
    mx.random.seed(11)
    plain = [_plain_sample(model, prompt, 3, sampling) for _ in range(runs)]
    mx.random.seed(12)
    fast = [list(several_tokens_a_pass(model, prompt, 3, drafts=Drafts(depth=3), sampling=sampling))
            for _ in range(runs)]
    for position in (1, 2):
        a = Counter(seq[position] for seq in plain)
        b = Counter(seq[position] for seq in fast)
        cells = sorted(set(a) | set(b))
        # Two-sample chi-square on the position's marginal; the critical value is chi-square's at p = 0.001.
        chi = sum((a[c] - b[c]) ** 2 / (a[c] + b[c]) for c in cells if a[c] + b[c])
        critical = {1: 10.83, 2: 13.82, 3: 16.27, 4: 18.47}[max(1, len(cells) - 1)]
        assert chi < critical, (position, a, b, chi)


def test_a_steering_hook_on_her_blocks_acts_on_the_checking_pass_as_on_any_other(on_cpu) -> None:
    from core.brain.llm.several_tokens_a_pass import Drafts, several_tokens_a_pass
    from core.brain.llm.two_tokens_a_step import plain_greedy

    model = _tiny(vocab=7, seed=9)
    prompt = [1, 3, 0, 2, 4, 1, 3, 0, 2, 5, 6, 1, 3]
    unsteered = plain_greedy(model, prompt, 80)
    # Affective steering swaps a block's class for one whose call adds a vector to its output.
    block = model.language_model.model.layers[5]
    vector = mx.random.normal((32,)) * 3.0

    class Steered(block.__class__):
        def __call__(self, x, *args, **kwargs):
            return super().__call__(x, *args, **kwargs) + vector

    block.__class__ = Steered
    steered = plain_greedy(model, prompt, 80)
    assert steered != unsteered
    assert list(several_tokens_a_pass(model, prompt, 80, drafts=Drafts(depth=5))) == steered
