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


def test_a_typed_answer_keeps_the_length_it_was_asked_for():
    """LIVE 2026-10-05 a part of a program asked for at 2048 tokens went out at 429 after a memory alarm."""
    from core.brain.a_typed_answer import keeps_its_length

    assert keeps_its_length({"schema": {"type": "object"}, "max_tokens": 2048}, 429) == 2048
    assert keeps_its_length({"output_shape": "json_object", "max_tokens": 2048}, 429) == 2048
    assert keeps_its_length({"max_tokens": 2048}, 429) == 429  # a reply in words is the moment's to shorten
    assert keeps_its_length({"schema": {"type": "object"}}, 429) == 429
