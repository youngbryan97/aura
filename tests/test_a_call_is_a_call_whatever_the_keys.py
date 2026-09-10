"""A tool call the model wrote with different key names is still a call.

The parser accepted `tool`/`args` and `name`/`arguments`. A model writing
`function`/`args` produced an object nothing recognised, so its intent became
prose — and the safety gate then correctly refused that prose as a prompt
artifact, leaving a turn with no call, no answer, and no record of either.

LIVE, 2026-09-07: `ResponseGeneration rejected unsafe user-facing draft
(prompt_artifact, len=138): '<tool_call>\\n{"function": "local_file_read",
"args": {"path": "/Users/bryan/.aura/live-source/CLAUDE.md", ...}}'`. The
fallback ladder answered from the 9B, 417 seconds in.

Both halves are still required, so an ordinary JSON object with a `name` field
is not a call. And the allowlist still decides whether the named tool may run:
what changes is that a name nobody offers is now refused BY NAME instead of
disappearing into prose.
"""

from __future__ import annotations

import pytest

from core.brain.llm.mlx_client import (
    _CALL_ARGUMENT_KEYS,
    _CALL_NAME_KEYS,
    _first_present,
)


@pytest.mark.parametrize("name_key", _CALL_NAME_KEYS)
@pytest.mark.parametrize("argument_key", _CALL_ARGUMENT_KEYS)
def test_every_combination_names_a_call(name_key: str, argument_key: str) -> None:
    payload = {name_key: "file_operation", argument_key: {"path": "README.md"}}
    assert _first_present(payload, _CALL_NAME_KEYS) == "file_operation"
    assert _first_present(payload, _CALL_ARGUMENT_KEYS) == {"path": "README.md"}


def test_the_live_payload_is_recognised() -> None:
    payload = {
        "function": "local_file_read",
        "args": {"path": "CLAUDE.md"},
    }
    assert _first_present(payload, _CALL_NAME_KEYS) == "local_file_read"
    assert _first_present(payload, _CALL_ARGUMENT_KEYS)["path"].endswith("CLAUDE.md")


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "a person"},
        {"arguments": {"x": 1}},
        {"title": "something", "body": "else"},
        {},
    ],
)
def test_half_a_call_is_not_a_call(payload: dict) -> None:
    """An object with a name and no arguments is ordinary JSON."""

    name = _first_present(payload, _CALL_NAME_KEYS)
    args = _first_present(payload, _CALL_ARGUMENT_KEYS)
    assert name is None or args is None


def test_a_key_holding_none_is_still_a_key() -> None:
    """Absent and present-but-empty are different facts."""

    assert _first_present({"args": None}, _CALL_ARGUMENT_KEYS) is None
    assert "args" in {"args": None}
    assert _first_present({"name": ""}, _CALL_NAME_KEYS) == ""


def test_the_parser_still_checks_the_allowlist() -> None:
    """A name nobody offered is refused by name, not turned into prose."""

    import inspect

    from core.brain.llm import mlx_client as mc

    body = inspect.getsource(mc)
    marker = body.index("_first_present(payload, _CALL_NAME_KEYS)")
    after = body[marker : marker + 1200]
    assert "tool_not_advertised" in after
