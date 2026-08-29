"""A second complete call is not a stray parameter.

The envelope grammar allows one function per ``<tool_call>``. When the model
asks for two things it emits two, and the loop is built to take the first and
ask again with the result in hand — the code says so.

It never got there. The check for a stray parameter after the function ran
first, and a second complete call contains parameters of its own, so two
well-formed calls were refused for the crime of the second one containing the
word "parameter".

Live on 2026-08-28: asked to read a library's docs and use it, the model
emitted two correct file_operation calls, the first complete and the second
cut off mid-path by the budget. Both were thrown away, the raw envelope leaked
into the user-facing draft, and the turn ended on an apology.
"""

from __future__ import annotations

from core.brain.llm.mlx_client import _native_xml_tool_payload

_DEFINITIONS = {
    "file_operation": {
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "path": {"type": "string"},
            },
        }
    }
}


def _one(action: str, path: str) -> str:
    return (
        "<tool_call>\n<function=file_operation>\n"
        f"<parameter=action>\n{action}\n</parameter>\n"
        f"<parameter=path>\n{path}\n</parameter>\n"
        "</function>\n</tool_call>"
    )


def test_one_call_parses() -> None:
    text = _one("read", "/tmp/ledgerkit/API.md")
    payload, reason = _native_xml_tool_payload(
        text, start=len("<tool_call>"), tool_definitions=_DEFINITIONS
    )
    assert payload is not None, reason
    assert payload["arguments"]["path"] == "/tmp/ledgerkit/API.md"


def test_the_first_of_two_calls_is_taken() -> None:
    """The behaviour the loop was built for, and could not reach."""

    text = _one("read", "/tmp/ledgerkit/API.md") + "\n" + _one("read", "/tmp/other.md")
    payload, reason = _native_xml_tool_payload(
        text, start=len("<tool_call>"), tool_definitions=_DEFINITIONS
    )
    assert payload is not None, reason
    assert payload["arguments"]["path"] == "/tmp/ledgerkit/API.md"


def test_a_second_call_cut_off_mid_path_still_leaves_the_first() -> None:
    """Exactly what happened live: the budget ended inside the second call."""

    text = (
        _one("read", "/tmp/ledgerkit/API.md")
        + "\n<tool_call>\n<function=file_operation>\n"
        "<parameter=action>\nread\n</parameter>\n"
        "<parameter=path>\n/private/tmp/claude-501/-Users-bry"
    )
    payload, reason = _native_xml_tool_payload(
        text, start=len("<tool_call>"), tool_definitions=_DEFINITIONS
    )
    assert payload is not None, reason
    assert payload["arguments"]["path"] == "/tmp/ledgerkit/API.md"


def test_a_genuinely_stray_parameter_is_still_refused() -> None:
    """Nothing after it could own it, so taking the call would change the ask."""

    text = (
        _one("read", "/tmp/ledgerkit/API.md")
        + "\n<parameter=recursive>\ntrue\n</parameter>"
    )
    payload, reason = _native_xml_tool_payload(
        text, start=len("<tool_call>"), tool_definitions=_DEFINITIONS
    )
    assert payload is None
    assert "parameter after the function" in reason
