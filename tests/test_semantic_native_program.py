"""Native semantic supervision preserves source identity and token boundaries."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_native_program import (
    native_choice_loss,
    native_program_sequence,
    native_program_surface,
    native_program_text,
    native_text_decision_sequence,
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


def test_graph_decision_spans_cover_atoms_and_semantic_continue_stop_choices():
    program = Program(2, (Instruction("sub", (0, 1)), Instruction("mul", (2, 0))))
    text, spans = native_program_surface(program)
    assert text == json.dumps({"inputs": 2, "steps": [["sub", [0, 1]], ["mul", [2, 0]]]},
                              separators=(",", ":"))
    assert [text[start:end] for start, end in spans] == ["sub", "0", "1", ",", "mul", "2", "0", "]"]


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
    assert bytes(row.tokens[index] for index in row.semantic_positions).decode() == "sub01]"
    with pytest.raises(ValueError, match="sequence length"):
        native_program_sequence(source, _program(), tokenizer, max_tokens=5)


def test_partial_grammar_prefix_scores_only_the_declared_decision():
    text = '{"inputs":2,"steps":[["sub"'
    span = (text.index("sub"), text.index("sub") + 3)
    sequence = native_text_decision_sequence("Unchanged request", text, (span,), Tokenizer())
    assert bytes(sequence.tokens[index] for index in sequence.semantic_positions).decode() == "sub"
    with pytest.raises(ValueError, match="target spans"):
        native_text_decision_sequence("Unchanged request", text, ((0, len(text) + 1),), Tokenizer())
    with pytest.raises(ValueError):
        parse_native_program(text)


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
    assert all(index >= row.continuation_start for index in row.semantic_positions)
    decision_tokens = tokenizer.decode([row.tokens[index] for index in row.semantic_positions])
    assert "sub" in decision_tokens
    assert "think" not in decision_tokens


def test_choice_objective_has_the_exact_complete_graph_gradient():
    import mlx.core as mx
    scores = mx.array([.2, -.7, 1.1])
    loss = native_choice_loss(scores, (1,))
    expected = mx.logsumexp(scores) - scores[1]
    assert mx.allclose(loss, expected).item()
    gradient = mx.grad(lambda value: native_choice_loss(value, (1,)))(scores)
    assert mx.allclose(gradient, mx.softmax(scores) - mx.array([0., 1., 0.]), atol=1e-6).item()
    updated = scores - .1 * gradient
    assert native_choice_loss(updated, (1,)).item() < loss.item()
    assert mx.allclose(native_choice_loss(scores + 10., (1,)), loss, atol=1e-6).item()


def test_known_equivalent_positives_share_probability_mass_without_negative_labels():
    import mlx.core as mx
    scores = mx.array([.2, -.7, 1.1])
    assert mx.allclose(native_choice_loss(scores, (0, 2)),
                       mx.logsumexp(scores) - mx.logsumexp(scores[mx.array([0, 2])])).item()
    assert native_choice_loss(scores, (0, 1, 2)).item() == 0.


@pytest.mark.parametrize("positives", [(), (True,), (-1,), (3,), (0, 0), [0]])
def test_invalid_choice_labels_do_not_enter_the_objective(positives):
    import mlx.core as mx
    with pytest.raises(ValueError, match="source-positive"):
        native_choice_loss(mx.array([0., 1., 2.]), positives)


def test_archived_atom_basis_is_explicit_and_does_not_change_graph_text():
    from core.learning.semantic_native_program import native_program_surface
    program = Program(2, (Instruction("add", (0, 1)), Instruction("sub", (2, 0))))
    text, current = native_program_surface(program)
    old_text, old = native_program_surface(program, decision_basis="program_atoms_v1")
    assert text == old_text
    assert set(old) < set(current)
    assert {text[left:right] for left, right in set(current) - set(old)} == {",", "]"}
    with pytest.raises(ValueError, match="decision basis"):
        native_program_surface(program, decision_basis="unknown")
