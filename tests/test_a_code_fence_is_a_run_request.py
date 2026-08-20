"""A model that answers with runnable code has asked to run it.

LIVE, 2026-08-19. Asked to read a file, with code_repl among the offered
tools, the model produced exactly this and nothing else:

    ```python
    with open('/private/tmp/.../README.md') as f: print(f.read())
    ```

That is the right action, expressed the way a model naturally expresses "run
this". Rejecting it for lacking a tool-call envelope threw away a correct
attempt and reported that she had refused to act.

The whole-response rule is what keeps this safe, and it is the rule the JSON
envelope already lives under: a fence EMBEDDED in prose is a worked example
being discussed, and only a response that IS the fence is a request to run it.
That distinction exists because an earlier version accepted fenced JSON
anywhere in prose, and a quotation became an effect request.
"""

from __future__ import annotations

import pytest

from core.brain.llm.mlx_client import MLXLocalClient, _code_execution_tool

_extract = MLXLocalClient._extract_tool_call_payload
OFFERED = {"code_repl", "file_operation", "web_search"}


def test_the_live_answer_is_read_as_a_run_request():
    response = (
        "```python\n"
        "with open('/private/tmp/x/README.md') as f: print(f.read())\n"
        "```"
    )
    call = _extract(response, allowed_tools=OFFERED)
    assert call is not None
    assert call["tool"] == "code_repl"
    assert "open(" in call["args"]["code"]


def test_a_fence_inside_prose_is_a_worked_example():
    """The rule that stopped a quotation becoming an effect request."""
    response = (
        "Here is roughly what that would look like:\n\n"
        "```python\nprint('hello')\n```\n\n"
        "Would you like me to run it?"
    )
    assert _extract(response, allowed_tools=OFFERED) is None


def test_no_code_tool_offered_means_no_run_request():
    """A fence is only a call if something present can execute it."""
    response = "```python\nprint('hello')\n```"
    assert _extract(response, allowed_tools={"web_search", "file_operation"}) is None


def test_an_empty_fence_is_not_a_call():
    assert _extract("```python\n\n```", allowed_tools=OFFERED) is None


def test_a_plain_prose_answer_is_still_an_answer():
    assert _extract("The close() method zeroes the account.", allowed_tools=OFFERED) is None


def test_the_native_envelope_still_wins():
    response = '<tool_call>{"name": "file_operation", "arguments": {"action": "read", "path": "/x"}}</tool_call>'
    call = _extract(response, allowed_tools=OFFERED)
    assert call is not None and call["tool"] == "file_operation"


@pytest.mark.parametrize(
    "offered,expected",
    [
        ({"code_repl", "run_code"}, "code_repl"),
        ({"run_code", "web_search"}, "run_code"),
        ({"internal_sandbox"}, "internal_sandbox"),
        ({"web_search", "file_operation"}, None),
    ],
)
def test_the_runner_is_chosen_by_what_it_is_named_for(offered, expected):
    """No list of names, so a runner registered tomorrow is found."""
    assert _code_execution_tool(offered) == expected
