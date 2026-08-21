"""A tool call carrying a document is not clamped to the size of a phrase.

LIVE, 2026-08-20. Asked to build a single-file web app, the model emitted
exactly the right call — code_repl with the program as its argument — and the
generation stopped mid-string:

    <tool_call> {"name": "code_repl", "arguments": {"code": "#!/usr/bin/env
    python3\\nimport http.server\\nimport socketserver\\nfrom threading import
    Thread, Event\\n\\nclass Handler(http.server.SimpleHTTPReq

An incomplete JSON object is not a call, so the loop reported "none called"
and the turn ended by telling the person a file had been saved to Downloads
that was never written.
"""

from __future__ import annotations

import pytest

from core.brain.llm.mlx_client import _tools_can_carry_a_document as carries


def test_a_code_tool_carries_a_document() -> None:
    assert carries({"code_repl": {"parameters": {"properties": {"code": {"type": "string"}}}}})


def test_a_search_tool_does_not() -> None:
    assert carries({"web_search": {"parameters": {"properties": {"query": {"type": "string"}}}}}) is False


def test_the_openai_shaped_definition_is_read_too() -> None:
    assert carries([{"function": {"parameters": {"properties": {"content": {"type": "string"}}}}}])


@pytest.mark.parametrize("tools", [None, {}, [], "not a map", [{"parameters": {}}]])
def test_nothing_offered_carries_nothing(tools: object) -> None:
    assert carries(tools) is False


def test_a_document_argument_must_be_a_string() -> None:
    assert carries({"t": {"parameters": {"properties": {"code": {"type": "integer"}}}}}) is False
    assert carries({"t": {"parameters": {"properties": {"code": {"type": "string"}}}}}) is True


def test_the_clamp_consults_it() -> None:
    import inspect

    from core.brain.llm import mlx_client

    source = inspect.getsource(mlx_client._apply_memory_pressure_generation_controls)
    assert '_tools_can_carry_a_document(options.get("tools"))' in source
    assert "effective_cap = max(effective_cap, requested_max_tokens)" in source


def test_a_document_call_is_not_clamped_to_the_phrase_floor() -> None:
    """The behaviour, not the spelling: with a document tool offered, the
    budget is the caller's, not the 320-token floor a phrase needs."""
    from core.brain.llm import mlx_client

    document_tools = {"code_repl": {"parameters": {"properties": {"code": {"type": "string"}}}}}
    phrase_tools = {"web_search": {"parameters": {"properties": {"query": {"type": "string"}}}}}

    class _Pressure:
        """Warning level: what a resident 32B produces as a steady state."""

        max_token_cap = 384

    with_document = {"tools": document_tools, "max_tokens": 4096}
    with_phrase = {"tools": phrase_tools, "max_tokens": 4096}
    mlx_client._apply_memory_pressure_generation_controls(
        with_document, _Pressure(), default_max_tokens=4096
    )
    mlx_client._apply_memory_pressure_generation_controls(
        with_phrase, _Pressure(), default_max_tokens=4096
    )

    assert int(with_document["max_tokens"]) >= int(with_phrase["max_tokens"])
    assert int(with_document["max_tokens"]) > mlx_client._TOOL_CALL_TOKEN_FLOOR


def test_a_call_is_not_a_reply() -> None:
    """LIVE: the desktop lane planned 970 tokens for its answer, and the tool
    loop took the same number for a call whose argument was an HTML page."""
    from core.brain.llm.mlx_client import _tool_call_budget

    document = {"code_repl": {"parameters": {"properties": {"code": {"type": "string"}}}}}
    phrase = {"web_search": {"parameters": {"properties": {"query": {"type": "string"}}}}}

    assert _tool_call_budget(970, 8192, document) == 8192
    assert _tool_call_budget(970, 8192, phrase) == 970


def test_the_budget_never_shrinks_what_was_asked_for() -> None:
    from core.brain.llm.mlx_client import _tool_call_budget

    document = {"code_repl": {"parameters": {"properties": {"code": {"type": "string"}}}}}
    assert _tool_call_budget(9000, 8192, document) == 9000
    assert _tool_call_budget(970, None, document) == 970
    assert _tool_call_budget(None, 8192, document) == 8192


def test_the_loop_uses_it() -> None:
    import inspect

    from core.brain.llm.mlx_client import MLXLocalClient

    source = inspect.getsource(MLXLocalClient.think_and_act)
    assert "_tool_call_budget(" in source
