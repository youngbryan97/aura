"""Source-bound native-language supervision for the existing semantic program IR."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_floor import semantic_program_structural_key


def native_program_surface(program: Program, *, decision_basis="program_atoms_and_graph_termination_v1"
                           ) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Locate operations, references, and graph continuation/completion decisions."""
    if not isinstance(program, Program) or semantic_program_structural_key(program) is None:
        raise ValueError("native semantic target is outside the existing program grammar")
    if decision_basis not in {"program_atoms_v1", "program_atoms_and_graph_termination_v1"}:
        raise ValueError("unknown native semantic decision basis")
    termination = decision_basis == "program_atoms_and_graph_termination_v1"
    parts, spans, position = [], [], 0
    def append(value: Any, *, atom=False, decision=False):
        nonlocal position
        piece = json.dumps(value, ensure_ascii=True, allow_nan=False) if atom else value
        if atom or decision:
            begin, end = position, position + len(piece)
            if atom and isinstance(value, str):
                begin, end = begin + 1, end - 1
            spans.append((begin, end))
        parts.append(piece)
        position += len(piece)
    append('{"inputs":' + str(program.n_inputs) + ',"steps":[')
    for index, step in enumerate(program.instructions):
        append("," if index else "", decision=bool(index) and termination)
        append("[")
        append(step.op, atom=True)
        append(",[")
        for role, reference in enumerate(step.args):
            append("," if role else "")
            append(reference, atom=True)
        append("]]")
    append("]", decision=termination)
    append("}")
    return "".join(parts), tuple(spans)


def native_choice_loss(scores, positive_indices: tuple[int, ...]):
    """Source-supervised competition over complete graphs, not answer authority.

    The loss is logsumexp(all scores) - logsumexp(known-positive scores).
    With one positive its score gradient is softmax(scores) - one_hot(positive).
    Uncertain equivalence is not a negative label; callers must establish their
    source contrasts before invoking this training-only objective.
    """
    import mlx.core as mx

    if (scores.ndim != 1 or scores.shape[0] < 2
            or not isinstance(positive_indices, tuple) or not positive_indices
            or len(set(positive_indices)) != len(positive_indices)
            or any(type(index) is not int or not 0 <= index < scores.shape[0]
                   for index in positive_indices)):
        raise ValueError("native choice loss needs distinct source-positive graph indices")
    positive = mx.take(scores, mx.array(positive_indices, dtype=mx.int32))
    return mx.logsumexp(scores) - mx.logsumexp(positive)


def native_program_text(program: Program) -> str:
    """Serialize a checked graph without its expected value or source family."""
    return native_program_surface(program)[0]


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
    semantic_positions: tuple[int, ...] = ()


def native_program_sequence(source: str, program: Program, tokenizer: Any,
                            *, max_tokens: int = 1024,
                            decision_basis="program_atoms_and_graph_termination_v1"
                            ) -> NativeProgramSequence:
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
    target, semantic_spans = native_program_surface(program, decision_basis=decision_basis)
    messages = [user, {"role": "assistant", "content": target}]
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
    target_start = whole_text.rfind(target)
    if target_start < len(prefix_text):
        raise ValueError("native semantic graph has no assistant-owned text span")
    decisions = tuple(index for index, (begin, end) in enumerate(offsets)
                      if any(begin < target_start + right and end > target_start + left
                             for left, right in semantic_spans))
    if not decisions or any(index < start for index in decisions):
        raise ValueError("native semantic decisions escaped the continuation")
    return NativeProgramSequence(tuple(whole), start, decisions)
