"""No native adapter silently changes the coordinate system it was fitted on."""

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_native_codec import (
    native_surface_for_encoding,
    parse_native_for_encoding,
    register_encoding_from_plan,
)


@pytest.mark.parametrize("encoding", ["absolute_v1", "role_relative_v1"])
def test_declared_codec_round_trips_the_same_executor_graph(encoding):
    program = Program(3, (Instruction("sub", (0, 1)), Instruction("mul", (3, 2))))
    text, spans = native_surface_for_encoding(program, register_encoding=encoding)
    assert spans
    assert parse_native_for_encoding(text, register_encoding=encoding) == program
    other = "role_relative_v1" if encoding == "absolute_v1" else "absolute_v1"
    with pytest.raises(ValueError):
        parse_native_for_encoding(text, register_encoding=other)


def test_historical_absence_is_distinct_from_unknown_explicit_encoding():
    assert register_encoding_from_plan({}) == "absolute_v1"
    assert register_encoding_from_plan({"register_encoding": "role_relative_v1"}) == "role_relative_v1"
    for unsupported in (None, "auto", "relative_v2", True):
        with pytest.raises(ValueError, match="unknown"):
            register_encoding_from_plan({"register_encoding": unsupported})


def test_unknown_codec_has_no_parser_or_scoring_authority():
    with pytest.raises(ValueError, match="unknown"):
        parse_native_for_encoding("{}", register_encoding="auto")
    with pytest.raises(ValueError):
        register_encoding_from_plan(None)
