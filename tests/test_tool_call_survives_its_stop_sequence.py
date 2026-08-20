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
