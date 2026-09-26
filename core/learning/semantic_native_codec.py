"""Bind native training and replay to an explicit register wire contract."""

from __future__ import annotations

from collections.abc import Mapping

from core.learning.semantic_native_program import (
    native_program_sequence,
    native_program_surface,
    parse_native_program,
)
from core.learning.semantic_native_relative_program import (
    REGISTER_ENCODING,
    parse_relative_native_program,
    relative_native_program_sequence,
    relative_native_program_surface,
)

REGISTER_ENCODINGS = ("absolute_v1", REGISTER_ENCODING)
NATIVE_CODEC_IMPLEMENTATION_PATHS = (
    "core/learning/semantic_native_codec.py",
    "core/learning/semantic_native_relative_program.py",
    "core/learning/semantic_register_identity.py",
    "core/learning/semantic_program_shared_transducer.py",
)


def validate_register_encoding(encoding: str) -> str:
    if not isinstance(encoding, str) or encoding not in REGISTER_ENCODINGS:
        raise ValueError("native register encoding is unknown")
    return encoding


def register_encoding_from_plan(plan: Mapping) -> str:
    """Historical plans without the field retain their original absolute wire."""
    if not isinstance(plan, Mapping):
        raise ValueError("native register encoding needs a training plan")
    return validate_register_encoding(plan.get("register_encoding", "absolute_v1"))


def native_sequence_for_encoding(source, program, tokenizer, *, register_encoding="absolute_v1",
                                 max_tokens=1024,
                                 decision_basis="program_atoms_and_graph_termination_v1"):
    encoding = validate_register_encoding(register_encoding)
    if encoding == "absolute_v1":
        return native_program_sequence(source, program, tokenizer, max_tokens=max_tokens,
                                       decision_basis=decision_basis)
    if decision_basis != "program_atoms_and_graph_termination_v1":
        raise ValueError("relative native decision basis is unsupported")
    return relative_native_program_sequence(source, program, tokenizer, max_tokens=max_tokens)


def native_surface_for_encoding(program, *, register_encoding="absolute_v1"):
    encoding = validate_register_encoding(register_encoding)
    surface = native_program_surface if encoding == "absolute_v1" else relative_native_program_surface
    return surface(program)


def parse_native_for_encoding(text, *, register_encoding="absolute_v1"):
    encoding = validate_register_encoding(register_encoding)
    parse = parse_native_program if encoding == "absolute_v1" else parse_relative_native_program
    return parse(text)
