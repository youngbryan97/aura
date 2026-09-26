"""Native supervision using the shared substrate's role-relative registers."""

from __future__ import annotations

import json

from core.learning.procedure_induction import Program
from core.learning.semantic_native_program import native_text_decision_sequence
from core.learning.semantic_register_identity import (
    RegisterIdentity,
    program_from_register_identities,
    program_register_identities,
)

REGISTER_ENCODING = "role_relative_v1"


def relative_native_program_surface(program: Program) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Serialize the same graph; result coordinates do not shift with input arity."""
    steps = program_register_identities(program)
    parts, spans, position = [], [], 0

    def append(text: str, *, decision=False, quoted=False):
        nonlocal position
        rendered = json.dumps(text, ensure_ascii=True) if quoted else text
        if decision:
            left = position + int(quoted)
            spans.append((left, position + len(rendered) - int(quoted)))
        parts.append(rendered)
        position += len(rendered)

    append('{"inputs":' + str(program.n_inputs) + ',"registers":"' + REGISTER_ENCODING + '","steps":[')
    for ordinal, (operation, references) in enumerate(steps):
        if ordinal:
            append(",", decision=True)
        append("[")
        append(operation, decision=True, quoted=True)
        append(",[")
        for role, reference in enumerate(references):
            if role:
                append(",")
            append(reference.encode(), decision=True, quoted=True)
        append("]]")
    append("]", decision=True)
    append("}")
    return "".join(parts), tuple(spans)


def parse_relative_native_program(text: str) -> Program:
    """Resolve an explicit wire version without repairing missing or forward refs."""
    def unique_fields(pairs):
        values = {}
        for key, value in pairs:
            if key in values:
                raise ValueError("relative native output repeats a field")
            values[key] = value
        return values

    payload = json.loads(text, object_pairs_hook=unique_fields)
    if (not isinstance(payload, dict) or set(payload) != {"inputs", "registers", "steps"}
            or type(payload["inputs"]) is not int or payload["registers"] != REGISTER_ENCODING
            or not isinstance(payload["steps"], list) or not payload["steps"]
            or any(not isinstance(step, list) or len(step) != 2
                   or not isinstance(step[0], str) or not isinstance(step[1], list)
                   for step in payload["steps"])):
        raise ValueError("relative native output differs from its explicit wire version")
    steps = tuple((operation, tuple(RegisterIdentity.parse(ref) for ref in refs))
                  for operation, refs in payload["steps"])
    return program_from_register_identities(payload["inputs"], steps)


def relative_native_program_sequence(source: str, program: Program, tokenizer, *, max_tokens=1024):
    target, spans = relative_native_program_surface(program)
    return native_text_decision_sequence(source, target, spans, tokenizer, max_tokens=max_tokens)
