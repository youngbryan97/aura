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


def test_envelope_markup_in_the_tail_is_still_refused() -> None:
    """Prose is the model talking on; markup is a second thing being asked for.

    Taking the first call and dropping the rest silently would be worse than
    refusing, because nothing downstream would learn what else was emitted. A
    stray parameter is ambiguous the same way: it may have been meant for this
    call.
    """

    for tail in (
        "<function=file_operation>\n</function>",
        "<tool_call>\nsomething else\n",
        "<parameter=path>/somewhere/else</parameter>",
        "</tool_call extra>",
    ):
        payload, why = _parse(f"{_CALL}\n{tail}")
        assert payload is None, tail
        assert "markup after the function" in why


def test_an_unclosed_function_is_still_refused() -> None:
    payload, why = _parse("<function=diagnose_repo>\n<parameter=path>/tmp/p</parameter>")
    assert payload is None
    assert why
