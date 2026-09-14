"""Private mentions must not stop or grade an ordinary reasoning baseline."""

import hashlib
import importlib
from types import SimpleNamespace

import pytest

from core.brain.llm import public_channel_decode as channel
from core.brain.llm.latent_cortex.answer_contract import (
    ContractDecodeDisposition,
    contract_decode_disposition,
)


class Tokenizer:
    eos_token_id = 0

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(t) for t in tokens)

    def encode(self, text, add_special_tokens=False):
        return tuple(ord(c) for c in text)


def _decoder(monkeypatch, text, seen):
    def decode(model, prompt, *, eos_token_id, max_tokens, prefill_tokens, completion_check, progress):
        seen["prefill"] = prefill_tokens
        generated = list(prefill_tokens)
        stopped = False
        for i, token in enumerate(map(ord, text)):
            if i == max_tokens:
                break
            generated.append(token)
            if progress:
                progress(i + 1)
            if token == eos_token_id or completion_check(tuple(generated)):
                stopped = True
                break
        return tuple(generated), stopped, 7
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_transfer_decode.decode_base_greedy_tokens", decode,
    )


def _complete(text):
    return contract_decode_disposition(text) in {
        ContractDecodeDisposition.COMPLETE, ContractDecodeDisposition.INVALID,
    }


def test_private_contract_mentions_do_not_stop_or_grade_generation(monkeypatch):
    text = 'Need FINAL_ANSWER: later. FINAL_ANSWER: still private.</think>\nFINAL_ANSWER: {"x":2}'
    seen = {}
    _decoder(monkeypatch, text, seen)
    tokenizer = Tokenizer()
    result = channel.decode_public_greedy(object(), tokenizer, tokenizer.encode("assistant\n<think>\n"),
        max_tokens=1000, completion_check=_complete)
    assert result.text == 'FINAL_ANSWER: {"x":2}'
    assert result.stop_reason == "public_contract" and result.boundary_closed
    assert result.reasoning_chars > 0 and result.native_thinking
    assert "still private" not in repr(result.receipt())


def test_budget_exhausted_private_thought_is_empty_public_not_wrong_json(monkeypatch):
    _decoder(monkeypatch, "thinking without a public answer", {})
    tokenizer = Tokenizer()
    result = channel.decode_public_greedy(object(), tokenizer, tokenizer.encode("<think>\n"),
        max_tokens=8, completion_check=_complete)
    assert result.text == "" and result.stop_reason == "token_limit"
    assert not result.stopped and not result.boundary_closed
    assert result.reasoning_chars == 8


def test_wire_prefix_is_public_and_boundary_tokens_are_counted(monkeypatch):
    seen = {}
    _decoder(monkeypatch, "2}", seen)
    tokenizer = Tokenizer()
    wire = tokenizer.encode('FINAL_ANSWER: {"x":')
    result = channel.decode_public_greedy(object(), tokenizer, tokenizer.encode("<think>\n"),
        max_tokens=20, public_prefill=wire, completion_check=_complete)
    assert tokenizer.decode(seen["prefill"]).startswith("</think>\n\nFINAL_ANSWER:")
    assert result.text == 'FINAL_ANSWER: {"x":2}'
    assert result.generated_tokens == 2 and result.prefill_tokens == len(wire)
    assert result.boundary_tokens == len("</think>\n\n")
    assert result.reasoning_chars == 0


def test_non_thinking_model_keeps_public_output_and_eos_reason(monkeypatch):
    _decoder(monkeypatch, "hello\0", {})
    tokenizer = Tokenizer()
    result = channel.decode_public_greedy(object(), tokenizer, tokenizer.encode("assistant:"), max_tokens=40)
    assert result.text == "hello" and result.stop_reason == "eos"
    assert not result.native_thinking and result.boundary_closed
    assert result.reasoning_sha256 == hashlib.sha256(b"").hexdigest()


def test_explicit_generated_thinking_is_also_private(monkeypatch):
    _decoder(monkeypatch, '<think>FINAL_ANSWER: hidden.</think>FINAL_ANSWER: {"x":5}', {})
    tokenizer = Tokenizer()
    result = channel.decode_public_greedy(object(), tokenizer, tokenizer.encode("assistant:"),
        max_tokens=1000, completion_check=_complete)
    assert result.text == 'FINAL_ANSWER: {"x":5}'


def test_public_invalid_contract_still_stops(monkeypatch):
    _decoder(monkeypatch, "FINAL_ANSWER: wrong", {})
    tokenizer = Tokenizer()
    result = channel.decode_public_greedy(object(), tokenizer, tokenizer.encode("assistant:"),
        max_tokens=1000, completion_check=_complete)
    assert result.stop_reason == "public_contract"
    assert result.generated_tokens < len("FINAL_ANSWER: wrong")


def test_model_exhaustion_is_not_misreported_as_completion(monkeypatch):
    _decoder(monkeypatch, "partial", {})
    tokenizer = Tokenizer()
    result = channel.decode_public_greedy(object(), tokenizer, tokenizer.encode("assistant:"), max_tokens=40)
    assert result.stop_reason == "generator_exhausted" and not result.stopped


def test_sampled_decode_uses_the_same_channel_boundary_without_changing_prompt(monkeypatch):
    seen = {}
    def stream(model, tokenizer, **kwargs):
        seen.update(kwargs)
        yield SimpleNamespace(text="happy curious private", token=1, generation_tokens=1, finish_reason=None)
        yield SimpleNamespace(text="</think>\n\nThe report is ready.", token=2,
                              generation_tokens=2, finish_reason="stop")
    monkeypatch.setattr(importlib.import_module("mlx_lm.generate"), "stream_generate", stream)
    sampler = object()
    result = channel.decode_public_sample(object(), Tokenizer(), "user text\n<think>\n",
                                          max_tokens=256, sampler=sampler)
    assert seen == {"prompt": "user text\n<think>\n", "max_tokens": 256, "sampler": sampler}
    assert result.text == "The report is ready."
    assert result.stopped and result.boundary_closed and result.generated_tokens == 2
    assert result.receipt()["policy"] == channel.PUBLIC_CHANNEL_SAMPLE_POLICY
    assert "happy" not in repr(result.receipt())


@pytest.mark.parametrize("text, public, closed", [
    ("happy private", "", False),
    ("private</think>Partial", "Partial", True),
])
def test_sampled_truncation_cannot_be_reported_as_completion(monkeypatch, text, public, closed):
    def stream(*args, **kwargs):
        yield SimpleNamespace(text=text, token=1, generation_tokens=8, finish_reason="length")
    monkeypatch.setattr(importlib.import_module("mlx_lm.generate"), "stream_generate", stream)
    result = channel.decode_public_sample(object(), Tokenizer(), "<think>\n", max_tokens=8, sampler=object())
    assert result.text == public and result.boundary_closed is closed
    assert not result.stopped and result.stop_reason == "token_limit"


def test_sampled_decode_observes_real_mlx_stream_terminal_metadata(monkeypatch):
    import mlx.core as mx
    import mlx.nn as nn
    from tokenizers import Tokenizer as FastTokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    generator = importlib.import_module("mlx_lm.generate")
    vocab = {"<unk>": 0, "<eos>": 1, "hello": 2}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=FastTokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
        eos_token="<eos>", unk_token="<unk>",
    )
    def steps(*args, **kwargs):
        yield 2, mx.array([0.0, 0.0, 0.0])
        yield 1, mx.array([0.0, 0.0, 0.0])
    monkeypatch.setattr(generator, "generate_step", steps)
    result = channel.decode_public_sample(nn.Linear(2, 2), tokenizer, [2], max_tokens=10, sampler=object())
    assert result.text == "hello" and result.generated_tokens == 2
    assert result.stop_reason == "eos" and result.stopped


def test_sampled_decode_preserves_callers_logits_processors(monkeypatch):
    processor = object()
    def stream(*args, **kwargs):
        assert kwargs["logits_processors"] == [processor]
        yield SimpleNamespace(text="{}", token=1, generation_tokens=1, finish_reason="stop")
    monkeypatch.setattr(importlib.import_module("mlx_lm.generate"), "stream_generate", stream)
    result = channel.decode_public_sample(object(), Tokenizer(), "question", max_tokens=8,
                                          sampler=object(), logits_processors=(processor,))
    assert result.text == "{}" and result.stopped
