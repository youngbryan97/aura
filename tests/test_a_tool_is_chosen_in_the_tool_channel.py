"""Tool selection asked for JSON in a sentence and parsed whatever arrived.

The state machine pasted the OpenAI schema into a system prompt — "Output ONLY
a JSON object with 'tool' (string) and 'params' (dict). Do not include any
explanation or markdown formatting" — sampled, and ran `json.loads` on the
result behind a comment reading "Robust JSON extraction".

Every part of the alternative already existed. The tokenizer renders tool
definitions natively; the worker logs "Rendering native chat/tool template";
`_extract_tool_call_payload` on the client reads the call back and refuses one
naming a tool nobody offered. This caller went around all of it.
"""

from __future__ import annotations

import inspect

from core.cognitive import state_machine


def test_the_schema_is_not_serialised_into_a_sentence():
    source = inspect.getsource(state_machine)
    assert "Output ONLY a JSON object" not in source
    assert "Available tools (OpenAI Function Schema)" not in source
    assert "You are an action-taking AI" not in source


def test_the_native_channel_is_what_is_called():
    source = inspect.getsource(state_machine)
    at = source.index("SKILL: Formulating tool call for")
    window = source[at : at + 900]
    assert "self.llm.think_and_act(" in window
    assert "tools=_tools_for_the_native_channel(" in window


def test_nothing_called_is_not_an_occasion_to_ask_again():
    """A turn where no tool was called is a conversation, and _handle_chat
    owns it. Sampling a second time with a firmer instruction is the shape
    this change exists to remove."""
    source = inspect.getsource(state_machine)
    at = source.index("SKILL: Formulating tool call for")
    window = source[at : at + 1600]
    assert 'raise KeyError("tool")' in window
    assert "retry" not in window.lower()


def test_the_channel_is_handed_a_mapping_it_can_check_names_against():
    offered = state_machine._tools_for_the_native_channel(
        [
            {"type": "function", "function": {"name": "local_file_read", "parameters": {}}},
            {"name": "web_search", "parameters": {}},
            {"function": {"name": "", "parameters": {}}},
            "not a schema",
            None,
        ]
    )
    assert set(offered) == {"local_file_read", "web_search"}


def test_no_schemas_offers_nothing_rather_than_everything():
    assert state_machine._tools_for_the_native_channel([]) == {}
    assert state_machine._tools_for_the_native_channel(None) == {}
