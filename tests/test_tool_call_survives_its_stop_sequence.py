"""A tool call is still a tool call when its closing tag was never generated.

LIVE, 2026-08-20. Six calls in one session were logged as "none called" while
the model had emitted a well-formed ``<tool_call>`` for web_search, code_repl
and file_operation. ``</tool_call>`` is a stop sequence, so it terminates
generation instead of appearing in it, and the extractor required it.
"""

from __future__ import annotations

import pytest

from core.brain.llm.mlx_client import MLXLocalClient, _balanced_json_object

ALLOWED = {"web_search", "code_repl", "file_operation"}


def extract(text: str):
    return MLXLocalClient._extract_tool_call_payload(text, allowed_tools=ALLOWED)


def test_the_call_that_was_dropped_live() -> None:
    text = (
        '<tool_call> {"name": "web_search", "arguments": '
        '{"query": "current temperature in Reykjavik using open-meteo api"}}'
    )
    assert extract(text) == {
        "tool": "web_search",
        "args": {"query": "current temperature in Reykjavik using open-meteo api"},
    }


def test_a_closed_tag_still_works() -> None:
    text = '<tool_call>\n{"name": "file_operation", "arguments": {"action": "read"}}\n</tool_call>'
    assert extract(text) == {"tool": "file_operation", "args": {"action": "read"}}


def test_a_brace_inside_a_string_does_not_end_the_object() -> None:
    text = '<tool_call> {"name": "code_repl", "arguments": {"code": "print(\\"}\\")"}}'
    assert extract(text) == {"tool": "code_repl", "args": {"code": 'print("}")'}}


def test_an_unfinished_object_is_not_a_call() -> None:
    assert extract('<tool_call> {"name": "web_search", "arguments":') is None


def test_a_tool_nobody_offered_is_still_refused() -> None:
    assert extract('<tool_call> {"name": "rm_rf", "arguments": {}}') is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("{}", "{}"),
        ('{"a": {"b": 1}} trailing', '{"a": {"b": 1}}'),
        ('prefix {"a": "}"} ', '{"a": "}"}'),
        ('{"a": "\\\\"} ', '{"a": "\\\\"}'),
        ("no object here", None),
        ('{"unclosed": ', None),
    ],
)
def test_brace_balance(text: str, expected: str | None) -> None:
    assert _balanced_json_object(text, 0) == expected


def test_a_program_with_real_newlines_is_still_a_call() -> None:
    """LIVE, 2026-08-20. The model emitted a well-formed call whose `code`
    argument was a program, with literal newlines in it. Strict JSON forbids
    those, and that is how every model writes multi-line code — so a complete
    call with balanced braces and correct arguments was refused, and the turn
    ended in an apology."""
    call = (
        '<tool_call> {"name": "code_repl", "arguments": {"code": '
        '"html = \\"<html>\n<head>\n<title>T</title>\n</head>\n</html>\\"\nprint(html)"}}'
    )
    parsed = extract(call)
    assert parsed is not None
    assert parsed["tool"] == "code_repl"
    assert parsed["args"]["code"].count("\n") == 5


def test_valid_json_is_returned_unchanged() -> None:
    from core.brain.llm.mlx_client import _json_with_control_characters_escaped

    body = '{"name": "web_search", "arguments": {"query": "a b"}}'
    assert _json_with_control_characters_escaped(body) == body


def test_whitespace_between_tokens_is_left_alone() -> None:
    from core.brain.llm.mlx_client import _json_with_control_characters_escaped

    body = '{\n  "a": 1,\n  "b": "two"\n}'
    assert _json_with_control_characters_escaped(body) == body


def test_an_escaped_newline_is_not_escaped_twice() -> None:
    from core.brain.llm.mlx_client import _json_with_control_characters_escaped

    body = '{"code": "line one\\nline two"}'
    assert _json_with_control_characters_escaped(body) == body


def test_a_truncated_call_is_still_refused() -> None:
    assert extract('<tool_call> {"name": "code_repl", "arguments": {"code": ') is None
