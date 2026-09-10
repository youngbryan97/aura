"""Inside a tool loop the prompt grows at the end, so the front can be reused.

The per-turn block travels with the turn, immediately before the person's
message, because that is where it is read and where it costs no prefix. That
placement is right for a plain turn and wrong for a tool loop: a loop appends
an assistant call and a tool result per step, all of them AFTER the user's
message, so a block anchored before that message sits in front of everything
the loop appends — and it changes every step.

On the resident model that is a full re-prefill each step, because
`Qwen3_5.make_cache` returns `ArraysCache` for its linear layers and the
prompt cache can never be trimmed. Only a stored key that is a strict PREFIX
of the new prompt is reusable at all.

LIVE, 2026-09-07: five steps reading one file, prefilling 3,735 then 5,166
then 6,611 then 8,056 then 9,502 tokens; 40s, 55s, 56s, 83s to first token;
the turn's 178.8-second budget gone and an unfinished answer.

Finding the person's turn also has to use the role as written. `_message_role`
maps "tool" onto "user" for templates with no tool role, so the backward scan
found the last TOOL RESULT and anchored the block between an assistant's call
and the result of that call.
"""

from __future__ import annotations

from core.brain.llm.chat_format import system_first

_STABLE = "## USER-FACING CONVERSATION RELIABILITY CONTRACT\nbe accurate\n"
_VOLATILE = "## LIVE TONE\nmood: bright\n"


def _roles(messages: list[dict]) -> list[str]:
    return [str(message.get("role")) for message in system_first(messages)]


def _base() -> list[dict]:
    return [
        {"role": "system", "content": _STABLE},
        {"role": "system", "content": _VOLATILE},
        {"role": "user", "content": "read the file"},
    ]


def _with_steps(steps: int) -> list[dict]:
    messages = _base()
    for index in range(steps):
        messages.append(
            {"role": "assistant", "content": "", "tool_calls": [{"id": str(index)}]}
        )
        messages.append(
            {"role": "tool", "content": "x" * 40, "tool_call_id": str(index)}
        )
    return messages


def test_a_plain_turn_keeps_the_block_beside_the_question() -> None:
    assert _roles(_base()) == ["system", "runtime_evidence", "user"]


def test_a_tool_loop_puts_the_block_last() -> None:
    assert _roles(_with_steps(2)) == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "runtime_evidence",
    ]


def test_the_block_never_lands_between_a_call_and_its_result() -> None:
    roles = _roles(_with_steps(3))
    for index, role in enumerate(roles):
        if role != "runtime_evidence":
            continue
        assert not (
            index > 0
            and roles[index - 1] == "assistant"
            and index + 1 < len(roles)
            and roles[index + 1] == "tool"
        ), f"the block split a call from its result: {roles}"


def test_every_step_extends_the_step_before_it() -> None:
    """The property the cache needs, stated as the property.

    On a model whose cache cannot be trimmed, reuse happens only when the
    stored prompt is a strict prefix of the new one.
    """

    previous: list[str] | None = None
    for steps in range(0, 5):
        rendered = [
            (str(message.get("role")), str(message.get("content") or ""))
            for message in system_first(_with_steps(steps))
        ]
        # Everything but the trailing per-turn block.
        prefix = [item for item in rendered if item[0] != "runtime_evidence"]
        flat = ["|".join(item) for item in prefix]
        if previous is not None:
            assert flat[: len(previous)] == previous, (
                f"step {steps} rebuilt its prefix instead of extending it"
            )
        previous = flat


def test_the_raw_role_is_what_finds_the_person() -> None:
    from core.brain.llm.chat_format import _message_role, _raw_role

    tool_message = {"role": "tool", "content": "result"}
    assert _message_role(tool_message) == "user"
    assert _raw_role(tool_message) == "tool"
