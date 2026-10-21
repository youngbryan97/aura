"""Her head drafting in her worker's stream keeps what mlx_lm's own step gives: the same tokens, the same history to
each logits processor, the same reading on from a prompt cache, and the tap at the position being chosen after."""
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


def _head(model):
    from core.brain.llm.two_tokens_a_step import MTPHead

    head = MTPHead(model.language_model.args)
    mx.eval(head.parameters())
    return head


class _Recording:
    """A logits processor with state, as hers have: it counts, remembers each history, and leans on what it saw."""

    def __init__(self):
        self.histories = []

    def __call__(self, tokens, logits):
        self.histories.append([int(t) for t in tokens.tolist()])
        lean = (mx.arange(logits.shape[-1]) == len(self.histories) % logits.shape[-1]).astype(logits.dtype)
        return logits + 0.25 * (len(self.histories) % 3) * lean


PROMPT = [1, 3, 0, 2, 4, 1, 3, 0, 2, 5, 6, 1, 3]


def _plain(model, prompt, count, *, processors=None, cache=None):
    from mlx_lm.generate import generate_step

    return [int(t) for t, _ in generate_step(mx.array(prompt), model, max_tokens=count, logits_processors=processors,
                                             prompt_cache=cache, prefill_step_size=4)]


def _with_head(model, head, prompt, count, *, processors=None, cache=None, tap=None):
    from core.brain.llm.her_head_in_the_stream import head_generate_step

    out = list(head_generate_step(mx.array(prompt), model, head, max_tokens=count, logits_processors=processors,
                                  prompt_cache=cache, prefill_step_size=4, tap=tap))
    return [int(t) for t, _, _ in out], sum(1 for *_, drafted in out if drafted)


def test_the_same_tokens_as_her_worker_decodes_one_at_a_time(on_cpu):
    kept_from_her_head = 0
    for seed in (0, 1, 2, 3):
        model = _tiny(vocab=7, seed=seed)
        tokens, drafted = _with_head(model, _head(model), PROMPT, 120)
        assert tokens == _plain(model, PROMPT, 120), seed
        kept_from_her_head += drafted
    assert kept_from_her_head > 0  # some of them were her head's drafts, kept


@pytest.mark.parametrize("seed", [0, 4])
def test_each_logits_processor_sees_what_it_sees_one_token_at_a_time(on_cpu, seed):
    model = _tiny(vocab=7, seed=seed)
    one_at_a_time, with_head = _Recording(), _Recording()
    plain = _plain(model, PROMPT, 80, processors=[one_at_a_time])
    tokens, _ = _with_head(model, _head(model), PROMPT, 80, processors=[with_head])
    assert tokens == plain
    # mlx_lm's step works one token ahead and has processed the token after its last; the positions spoken are alike.
    assert with_head.histories == one_at_a_time.histories[:len(with_head.histories)]
    assert len(one_at_a_time.histories) - len(with_head.histories) <= 1


def test_reading_on_from_a_prompt_cache(on_cpu):
    from mlx_lm.models.cache import make_prompt_cache

    model = _tiny(vocab=7, seed=6)
    plain_cache, head_cache = make_prompt_cache(model), make_prompt_cache(model)
    for cache in (plain_cache, head_cache):
        model.language_model.model(mx.array([PROMPT[:7]]), cache)
    tokens, _ = _with_head(model, _head(model), PROMPT[7:], 60, cache=head_cache)
    assert tokens == _plain(model, PROMPT[7:], 60, cache=plain_cache)


def test_the_tap_holds_the_hidden_state_of_the_position_being_chosen_after(on_cpu):
    from types import SimpleNamespace

    from core.brain.llm.several_tokens_a_pass import _text

    model = _tiny(vocab=7, seed=7)
    tap = SimpleNamespace(last_hidden=None)
    seen = []

    def reads_the_tap(tokens, logits):
        seen.append((list(tokens.tolist()), tap.last_hidden))
        return logits

    tokens, _ = _with_head(model, _head(model), PROMPT, 40, processors=[reads_the_tap], tap=tap)
    text = _text(model)
    for history, hidden in seen[:12]:
        whole = list(PROMPT[:-1]) + history  # the prompt, then the history mlx_lm hands its processors
        expected = text.model(mx.array([whole]))[0, -1]
        assert mx.allclose(hidden, expected, atol=1e-4).item()


def test_the_stream_says_what_mlx_lms_stream_says():  # on the GPU: mlx_lm's stream reads the device's working set
    from mlx_lm import stream_generate
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from core.brain.llm.her_head_in_the_stream import stream_generate_with_her_head

    class _Tokenizer:
        eos_token_id = 6
        bos_token = None
        chat_template = None

        def encode(self, text, add_special_tokens=False):
            return [int(c) for c in text]

        def decode(self, ids, **_kw):
            return "".join(str(int(i)) for i in ids)

        def convert_ids_to_tokens(self, ids):
            return [str(int(i)) for i in ids]

        def get_vocab(self):
            return {str(i): i for i in range(7)}

    # On the GPU whatever an earlier test left as the default: in a full run one left the CPU, and mlx_lm's stream
    # asked the CPU for the GPU's working set (KeyError: 'max_recommended_working_set_size').
    was = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        model = _tiny(vocab=7, seed=8)
        prompt = mx.array(PROMPT)
        plain = [(r.token, r.finish_reason) for r in stream_generate(model, TokenizerWrapper(_Tokenizer()), prompt, max_tokens=50)]
        with_head = [(r.token, r.finish_reason) for r in stream_generate_with_her_head(
            model, TokenizerWrapper(_Tokenizer()), prompt, max_tokens=50, draft_model=_head(model))]
    finally:
        mx.set_default_device(was)
    assert with_head == plain


def test_her_worker_lets_her_head_draft_with_everything_her_generations_carry(monkeypatch):
    from types import SimpleNamespace

    from core.brain.llm import mlx_worker

    model = _tiny(vocab=7, seed=9)
    head = _head(model)
    carried = {"logits_processors": [lambda tokens, logits: logits], "prompt_cache": [], "max_tokens": 8}
    assert mlx_worker._speculative_eligible(head, carried, {}) is True  # her processors and her cache, kept
    assert mlx_worker._speculative_eligible(head, {**carried, "kv_bits": 4}, {}) is False
    assert mlx_worker._speculative_eligible(head, carried, {"schema": {"type": "object"}}) is False
    assert mlx_worker._speculative_eligible(None, carried, {}) is False
    assert mlx_worker._speculative_eligible(SimpleNamespace(), carried, {}) is False  # another draft: as before
    monkeypatch.setenv("AURA_HER_HEAD_DRAFTS", "0")
    from core.brain.llm.her_head_in_the_stream import her_own_head

    assert her_own_head(model) is None  # off until switched on


def test_in_her_heads_pass_every_position_is_steered_and_sampled():
    """MEASURED 20 October 2026: steered at its last position only, the first of her head's two tokens went unsteered
    and her words parted from her one-at-a-time decoding at token 59."""
    from core.consciousness.affective_steering import AffectiveSteeringHook
    from core.consciousness.spoken_positions import speaking_several

    hook = AffectiveSteeringHook.__new__(AffectiveSteeringHook)
    hook._layer_idx = 0
    pass_of_two = mx.zeros((1, 2, 8))
    assert hook._completion_position_mask(pass_of_two) is not None  # a prompt: its last position alone
    with speaking_several():
        assert hook._completion_position_mask(pass_of_two) is None  # two spoken tokens: both
