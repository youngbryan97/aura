"""Private mentions must not stop or grade an ordinary reasoning baseline."""

import hashlib

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
