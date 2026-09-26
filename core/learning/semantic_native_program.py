"""Source-bound native-language supervision for the existing semantic program IR."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_floor import semantic_program_structural_key


def native_program_text(program: Program) -> str:
    """Serialize a checked graph without its expected value or source family."""
    if not isinstance(program, Program) or semantic_program_structural_key(program) is None:
        raise ValueError("native semantic target is outside the existing program grammar")
    return json.dumps({"inputs": program.n_inputs,
                       "steps": [[step.op, list(step.args)] for step in program.instructions]},
                      separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def parse_native_program(text: str) -> Program:
    """Read the same typed graph; malformed output has no execution authority."""
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("native semantic output repeats a field")
            result[key] = value
        return result

    payload = json.loads(text, object_pairs_hook=unique_fields)
    if (not isinstance(payload, dict) or set(payload) != {"inputs", "steps"}
            or type(payload["inputs"]) is not int or not isinstance(payload["steps"], list)
            or not payload["steps"] or any(
                not isinstance(step, list) or len(step) != 2 or not isinstance(step[0], str)
                or not isinstance(step[1], list) or any(type(ref) is not int for ref in step[1])
                for step in payload["steps"])):
        raise ValueError("native semantic output differs from its program schema")
    program = Program(payload["inputs"], tuple(Instruction(op, tuple(refs))
                                             for op, refs in payload["steps"]))
    if semantic_program_structural_key(program) is None:
        raise ValueError("native semantic output is outside the existing program grammar")
    return program


def source_text_from_tokens(item: Any, tokenizer: Any) -> str:
    """Recover the exact retained request, refusing tokenizer or source drift."""
    tokens = list(item.ir.source_token_ids)
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    if (hashlib.sha256(text.encode("utf-8")).hexdigest() != item.ir.source_text_sha256
            or tokenizer.encode(text, add_special_tokens=False) != tokens):
        raise ValueError("native semantic source does not round-trip through this tokenizer")
    return text


@dataclass(frozen=True)
class NativeProgramSequence:
    """Teacher-forced tokens and the exact beginning of supervised continuation."""

    tokens: tuple[int, ...]
    continuation_start: int


def native_program_sequence(source: str, program: Program, tokenizer: Any,
                            *, max_tokens: int = 1024) -> NativeProgramSequence:
    """Use the model's unchanged chat template and an ordinary, unmodified request.

    The target is supervised tissue data, not a runtime instruction asking the
    model to follow a special reasoning strategy. No system message is added.
    The template's private-channel boundary, if present, is preserved exactly.
    """
    if not isinstance(source, str) or not source or type(max_tokens) is not int or max_tokens < 2:
        raise ValueError("native semantic sequence needs a request and a finite token bound")
    from core.learning.semantic_program_feature_materialization import (
        offset_tokenizer_for_worker,
        tokenize_with_offsets,
    )

    user = {"role": "user", "content": source}
    messages = [user, {"role": "assistant", "content": native_program_text(program)}]
    prefix_text = tokenizer.apply_chat_template([user], add_generation_prompt=True, tokenize=False)
    whole_text = tokenizer.apply_chat_template(messages, tokenize=False)
    if (not isinstance(prefix_text, str) or not prefix_text or not isinstance(whole_text, str)
            or not whole_text.startswith(prefix_text) or len(whole_text) <= len(prefix_text)):
        raise ValueError("native semantic continuation or sequence length differs from its template")
    whole, offsets = tokenize_with_offsets(offset_tokenizer_for_worker(tokenizer), whole_text)
    if whole != tokenizer.apply_chat_template(messages, tokenize=True) or len(whole) > max_tokens:
        raise ValueError("native semantic continuation or sequence length differs from its template")
    # A BPE token may straddle the boundary; supervise that whole token once.
    start = next((index for index, (_begin, end) in enumerate(offsets)
                  if end > len(prefix_text)), None)
    if start is None or start < 1:
        raise ValueError("native semantic continuation has no offset-bound token boundary")
    return NativeProgramSequence(tuple(whole), start)
