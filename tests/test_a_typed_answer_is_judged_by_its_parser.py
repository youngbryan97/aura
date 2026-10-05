"""A typed answer is judged by its own parser, not by a reader of prose.

LIVE 2026-10-05: a list of checks was refused as a lexical loop after 418
seconds, and a part of a program as an "escaped control artifact" for the
newlines JSON writes inside its code string.
"""
from __future__ import annotations

from core.brain.a_typed_answer import judged_by_its_parser


def test_a_typed_answer_s_loops_cut_offs_and_escapes_are_its_parser_s_to_judge():
    asked = {"schema": {"type": "object"}, "output_shape": "json_object"}
    assert judged_by_its_parser(asked, {"low_lexical_diversity_loop", "truncated_tail"})
    assert judged_by_its_parser(asked, {"escaped_control_artifact"})


def test_prose_and_leaks_are_still_the_gate_s():
    assert not judged_by_its_parser({}, {"truncated_tail"})
    assert not judged_by_its_parser({"output_shape": "json_object"}, {"prompt_artifact"})
