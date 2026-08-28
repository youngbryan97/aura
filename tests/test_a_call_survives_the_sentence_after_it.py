"""The envelope ends at </function>; what follows is the model talking on.

LIVE, 2026-08-28: a request to work out what was wrong with a project produced
a correct diagnose_repo call with a sentence behind it. The whole envelope was
refused, the body was empty, and the turn ended with no answer. The tool that
had the answer was called, and the call was thrown away for the prose after it.
"""

from __future__ import annotations

from core.brain.llm.mlx_client import _native_xml_tool_payload

_TOOLS = {
    "diagnose_repo": {
        "function": {
            "name": "diagnose_repo",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }
    }
}

_CALL = (
    "<function=diagnose_repo>\n<parameter=path>\n/tmp/p\n</parameter>\n</function>"
)


def _parse(text: str):
    return _native_xml_tool_payload(text, start=0, tool_definitions=_TOOLS)


def test_a_clean_call_parses() -> None:
    payload, why = _parse(_CALL)
    assert payload == {"name": "diagnose_repo", "arguments": {"path": "/tmp/p"}}, why


def test_prose_after_the_call_does_not_invalidate_it() -> None:
    payload, why = _parse(f"{_CALL}\nNow let me look at what came back.")
    assert payload is not None, why
    assert payload["name"] == "diagnose_repo"


def test_the_closing_tool_call_tag_and_then_prose_is_fine() -> None:
    payload, why = _parse(f"{_CALL}\n</tool_call>\nI will read the result.")
    assert payload is not None, why


def test_a_second_function_lets_the_first_one_run() -> None:
    """The loop takes more than one turn; refusing both loses one to say nothing.

    LIVE, 2026-08-28: turn two of a diagnosis emitted a call with more markup
    behind it. Both were refused, and the turn ended having run one tool and
    said nothing about what it found.
    """

    for tail in (
        "<function=file_operation>\n</function>",
        "<tool_call>\nsomething else\n",
        "</tool_call extra>",
    ):
        payload, why = _parse(f"{_CALL}\n{tail}")
        assert payload is not None, f"{tail}: {why}"
        assert payload["name"] == "diagnose_repo"


def test_a_parameter_after_the_close_is_still_refused() -> None:
    """It may have been meant for THIS call, so taking the call changes it."""

    payload, why = _parse(f"{_CALL}\n<parameter=path>/somewhere/else</parameter>")
    assert payload is None
    assert "parameter after the function" in why
