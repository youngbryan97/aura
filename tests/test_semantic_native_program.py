"""Native semantic supervision preserves source identity and token boundaries."""

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_native_program import (
    native_program_sequence,
    native_program_text,
    parse_native_program,
    source_text_from_tokens,
)


class Tokenizer:
    def encode(self, text, **_kwargs):
        return list(text.encode())

    def decode(self, tokens, **_kwargs):
        return bytes(tokens).decode()

    def __call__(self, text, **_kwargs):
        return {"input_ids": self.encode(text),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=True):
        text = "".join(f"<{message['role']}>" + message["content"] for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text) if tokenize else text


def _program():
    return Program(2, (Instruction("sub", (0, 1)),))


def test_program_codec_round_trips_with_no_expected_value_or_family():
    text = native_program_text(_program())
    assert text == '{"inputs":2,"steps":[["sub",[0,1]]]}'
    assert parse_native_program(text) == _program()


@pytest.mark.parametrize("text", [
    '{"inputs":2,"steps":[["unknown",[0,1]]]}',
    '{"inputs":2,"steps":[["sub",[0,2]]]}',
    '{"inputs":2,"steps":[["sub",[true,1]]]}',
    '{"inputs":2,"steps":[]}',
    '{"inputs":2,"inputs":3,"steps":[["sub",[0,1]]]}',
    '{"inputs":2,"steps":[["sub",[0,1]]],"answer":9}',
])
def test_malformed_or_answer_bearing_output_is_not_a_program(text):
    with pytest.raises((ValueError, TypeError)):
        parse_native_program(text)


def test_supervision_contains_the_unchanged_request_and_masked_prefix():
    tokenizer = Tokenizer()
    source = "Subtract 8 from 13."
    row = native_program_sequence(source, _program(), tokenizer)
    prefix = tokenizer.decode(list(row.tokens[:row.continuation_start]))
    target = tokenizer.decode(list(row.tokens[row.continuation_start:]))
    assert prefix == "<user>" + source + "<assistant>"
    assert target == native_program_text(_program())
    with pytest.raises(ValueError, match="sequence length"):
        native_program_sequence(source, _program(), tokenizer, max_tokens=5)


def test_tokenizer_template_boundary_cannot_be_repaired_by_guessing():
    class Drifting(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            tokens = super().apply_chat_template(messages, **kwargs)
            return tokens[1:] if len(messages) == 2 else tokens
    with pytest.raises(ValueError, match="continuation"):
        native_program_sequence("Same source", _program(), Drifting())


def test_request_reconstruction_requires_both_text_and_token_identity():
    tokenizer = Tokenizer()
    text = "Given 4 and 6, return their sum."
    item = SimpleNamespace(ir=SimpleNamespace(source_token_ids=tuple(tokenizer.encode(text)),
        source_text_sha256=hashlib.sha256(text.encode()).hexdigest()))
    assert source_text_from_tokens(item, tokenizer) == text
    item.ir.source_text_sha256 = "f" * 64
    with pytest.raises(ValueError, match="round-trip"):
        source_text_from_tokens(item, tokenizer)


def test_token_crossing_the_template_boundary_is_supervised_without_text_changes():
    class Merging(Tokenizer):
        def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=True):
            text = "<user>" + messages[0]["content"] + "<assistant>\n"
            if len(messages) == 2:
                text += "\n" + messages[1]["content"]
            return self(text)["input_ids"] if tokenize else text

        def __call__(self, text, **_kwargs):
            tokens, offsets, index = [], [], 0
            while index < len(text):
                end = index + (2 if text[index:index + 2] == "\n\n" else 1)
                tokens.append(256 if end - index == 2 else ord(text[index]))
                offsets.append((index, end))
                index = end
            return {"input_ids": tokens, "offset_mapping": offsets}

    tokenizer = Merging()
    row = native_program_sequence("same source", _program(), tokenizer)
    assert row.tokens[row.continuation_start] == 256
    assert row.tokens[:row.continuation_start] == tuple(Tokenizer().encode(
        "<user>same source<assistant>"))


def test_current_resident_tokenizer_preserves_the_native_private_boundary():
    checkpoint = os.environ.get("AURA_NATIVE_TOKENIZER_CHECKPOINT")
    if not checkpoint:
        pytest.skip("local resident tokenizer checkpoint not supplied")
    from mlx_lm.utils import load_tokenizer
    tokenizer = load_tokenizer(Path(checkpoint))
    source = "Subtract the second input from the first."
    row = native_program_sequence(source, _program(), tokenizer)
    rendered = tokenizer.decode(list(row.tokens))
    assert source in rendered
    assert native_program_text(_program()) in rendered
    assert "</think>" in rendered
    assert 0 < row.continuation_start < len(row.tokens)
