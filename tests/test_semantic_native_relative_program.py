"""Role-relative native text remains a reversible view of the existing IR."""

import json
import os
from pathlib import Path

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_native_grammar import decode_native_grammar
from core.learning.semantic_native_program import parse_native_program
from core.learning.semantic_native_relative_program import (
    parse_relative_native_program,
    relative_native_program_sequence,
    relative_native_program_surface,
)


def test_relative_surface_and_decisions_preserve_the_complete_program():
    p = Program(4, (Instruction("add", (0, 1)), Instruction("mul", (4, 2)),
                    Instruction("sub", (5, 3))))
    text, spans = relative_native_program_surface(p)
    assert parse_relative_native_program(text) == p
    assert tuple(text[left:right] for left, right in spans) == (
        "add", "input:0", "input:1", ",", "mul", "result:0", "input:2", ",",
        "sub", "result:1", "input:3", "]")
    with pytest.raises(ValueError):
        parse_native_program(text)


def test_native_grammar_reuses_the_same_relative_register_coordinates():
    pending = iter(("add", "input:0", "input:1", "continue", "mul", "result:0",
                    "input:2", "continue", "sub", "result:1", "input:3", "finish"))
    def score(choices):
        wanted = next(pending)
        assert wanted in [choice.value for choice in choices]
        return tuple(0. if choice.value == wanted else -10. for choice in choices)
    result = decode_native_grammar(("integer",) * 4, score, register_encoding="role_relative_v1")
    assert result.program.run((10, 2, 3, 4)) == 32
    assert result.program.instructions[1].args == (4, 2)


@pytest.mark.parametrize("reference", ["result:0", "input:2", "input:00", 0, True])
def test_relative_parser_rejects_unavailable_or_noncanonical_first_step_refs(reference):
    body = {"inputs": 2, "registers": "role_relative_v1", "steps": [["neg", [reference]]]}
    with pytest.raises(ValueError):
        parse_relative_native_program(json.dumps(body))


def test_encoding_and_field_identity_cannot_be_inferred_from_malformed_text():
    text, _ = relative_native_program_surface(Program(1, (Instruction("neg", (0,)),)))
    for malformed in (text.replace('"role_relative_v1"', '"absolute_v1"'),
                      text.replace('"inputs":1', '"inputs":1,"inputs":1'),
                      text.replace('"inputs":1', '"inputs":true')):
        with pytest.raises(ValueError):
            parse_relative_native_program(malformed)


def test_computed_reference_text_is_invariant_to_unused_input_padding():
    for arity in (3, 4, 8):
        p = Program(arity, (Instruction("add", (0, 1)), Instruction("mul", (arity, 2))))
        text, spans = relative_native_program_surface(p)
        assert "result:0" in tuple(text[a:b] for a, b in spans)
        assert parse_relative_native_program(text).run(tuple(range(arity))) == 2


def test_resident_tokenizer_keeps_relative_decisions_in_the_public_channel():
    checkpoint = os.environ.get("AURA_NATIVE_TOKENIZER_CHECKPOINT")
    if not checkpoint:
        pytest.skip("local resident tokenizer checkpoint not supplied")
    from mlx_lm.utils import load_tokenizer
    tokenizer = load_tokenizer(Path(checkpoint))
    program = Program(4, (Instruction("add", (0, 1)), Instruction("mul", (4, 2)),
                          Instruction("sub", (5, 3))))
    source = "Add the first two inputs, multiply by the third, then subtract the fourth."
    row = relative_native_program_sequence(source, program, tokenizer)
    rendered = tokenizer.decode(list(row.tokens))
    assert source in rendered
    assert relative_native_program_surface(program)[0] in rendered
    assert "</think>" in rendered
    assert all(index >= row.continuation_start for index in row.semantic_positions)
    decisions = tokenizer.decode([row.tokens[index] for index in row.semantic_positions])
    assert "result:0" in decisions and "input:3" in decisions
    assert "think" not in decisions
