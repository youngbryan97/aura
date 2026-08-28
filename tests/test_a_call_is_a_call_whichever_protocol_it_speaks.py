"""A tool call served through the JSON contract is still a tool call.

The memory-pressure control shrinks a generation when the host is loaded, and
it already carried an exception for tool calls: a call clamped below the size of
a call cannot be expressed, so nothing runs. That exception read
``options.get("tools")`` — which carries the definitions only when the model's
NATIVE tool template is in use.

A model whose template has no tool support, or one whose native attempt came
back empty, is served the same call through a JSON contract in the prompt, and
that path sets ``tools`` to None. Every protection written for tool calls
stopped applying to the path that needs them most: the whole envelope is
generated there rather than templated, so it needs MORE room, not less.

LIVE, 2026-08-28: "read the docs at <path>, then use it" was granted 2048
tokens by its own clock, given 399 by the pressure cap, and cut off inside
``from ledgerkit imp`` — on a turn whose offered tools included one taking a
program as an argument.
"""

from __future__ import annotations

from core.brain.llm.mlx_client import (
    _apply_memory_pressure_generation_controls,
    _offered_for_budgeting,
)


class _Loaded:
    """A host under enough pressure to cap a generation hard."""

    max_token_cap = 399
    should_gc = False
    refuse_heavy_local_generation = False


_CARRIES_A_PROGRAM = [
    {
        "type": "function",
        "function": {
            "name": "code_repl",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
        },
    }
]


def _granted(options: dict) -> int:
    return int(
        _apply_memory_pressure_generation_controls(
            dict(options), _Loaded(), default_max_tokens=2048
        )["max_tokens"]
    )


def test_both_protocols_get_the_room() -> None:
    native = {
        "max_tokens": 2048,
        "tools": _CARRIES_A_PROGRAM,
        "tool_budget_definitions": _CARRIES_A_PROGRAM,
    }
    json_contract = {
        "max_tokens": 2048,
        "tools": None,
        "tool_budget_definitions": _CARRIES_A_PROGRAM,
    }
    assert _granted(native) == 2048
    assert _granted(json_contract) == 2048


def test_a_turn_that_calls_nothing_is_still_capped() -> None:
    """The exception is for calls. Ordinary prose still yields under pressure."""

    assert _granted({"max_tokens": 2048}) == 399


def test_the_loop_declares_what_it_may_call_either_way() -> None:
    from pathlib import Path

    body = Path("core/brain/llm/mlx_client.py").read_text()
    assert "tool_budget_definitions=template_tools or None" in body
    # And it never reaches the worker.
    assert 'kwargs.pop("tool_budget_definitions", None)' in body


def test_the_budget_reads_either_source() -> None:
    assert _offered_for_budgeting({"tools": _CARRIES_A_PROGRAM})
    assert _offered_for_budgeting({"tool_budget_definitions": _CARRIES_A_PROGRAM})
    assert not _offered_for_budgeting({"tools": None})
    assert not _offered_for_budgeting({})
