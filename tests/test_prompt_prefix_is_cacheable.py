"""The prompt prefix must stay byte-identical across turns, or the KV cache is dead.

Prompt caching reuses a *prefix*. It pays only while the leading bytes of turn
N+1 match turn N; the first differing byte ends reuse, and everything after it
is prefilled from scratch.

Live 2026-07-26: `[LIVE MIND CONTEXT]` — phenomenal and body state, rebuilt every
turn — was assembled at block two, ahead of the conversation history. So the
reusable prefix ended after the system message and the entire history was
re-prefilled on every turn. Measured consequence: a 42,289-char prompt and
**82.5 seconds without a first token**, past the 81.4s turn budget, cancelled,
surfaced to the user as "I couldn't get to an answer I'd stand behind on that
one." Raising the KV cache budget (0 -> 2 entries, 2026-07-24) could not help:
the entries had no stable prefix to hit.

Volatile content therefore belongs last — immediately before the newest user
message, so it still conditions the answer with maximum recency, while
`system + history` stays identical turn over turn.
"""
from __future__ import annotations

from core.brain.inference_gate import InferenceGate
from core.brain.llm.chat_format import (
    conversation_append_messages,
    conversation_resume_context_digest,
    render_chat_append_template,
    render_chat_template,
)
from core.utils.injected_blocks import stamp_grounding

LIVE_MIND = "[LIVE MIND CONTEXT]"


def _gate() -> InferenceGate:
    return InferenceGate.__new__(InferenceGate)


def _turn(grounding: str, history: list[tuple[str, str]]) -> list[dict[str, str]]:
    """One turn's prebuilt payload: system, per-turn grounding, then history."""
    messages = [{"role": "system", "content": "You are Aura. Stable identity block."}]
    messages.append(stamp_grounding({"role": "system", "content": f"{LIVE_MIND} {grounding}"}))
    messages.extend({"role": role, "content": text} for role, text in history)
    return messages


def _serialize(compacted: list[dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}:{m['content']}" for m in compacted)


def test_grounding_lands_after_history_and_before_the_newest_user_message() -> None:
    out = _gate()._compact_prebuilt_messages(
        _turn(
            "vitality=0.74",
            [("user", "first question"), ("assistant", "first answer"), ("user", "second question")],
        ),
        history_limit=12,
    )
    roles_and_marks = [(m["role"], LIVE_MIND in m["content"]) for m in out]
    grounding_at = next(i for i, (_, mark) in enumerate(roles_and_marks) if mark)
    newest_user_at = max(i for i, (role, _) in enumerate(roles_and_marks) if role == "user")
    # Volatile block is not block two any more...
    assert grounding_at > 1
    # ...it sits directly before the question it is meant to ground.
    assert grounding_at == newest_user_at - 1
    assert out[grounding_at]["role"] == "runtime_evidence"
    assert out[-1]["content"] == "second question"


def test_prefix_is_identical_across_turns_when_only_grounding_changes() -> None:
    """The whole point: turn N+1 shares turn N's leading bytes."""
    history = [("user", "first question"), ("assistant", "first answer")]
    gate = _gate()
    turn_a = _serialize(
        gate._compact_prebuilt_messages(
            _turn("vitality=0.74 coherence=1.00", [*history, ("user", "second question")]),
            history_limit=12,
        )
    )
    turn_b = _serialize(
        gate._compact_prebuilt_messages(
            # Same conversation, a later moment: body state has moved on.
            _turn("vitality=0.31 coherence=0.88", [*history, ("user", "second question")]),
            history_limit=12,
        )
    )
    common = 0
    for left, right in zip(turn_a, turn_b):
        if left != right:
            break
        common += 1
    # The shared prefix must cover the system block AND the prior history, not
    # stop at the system message the way it did when grounding came first.
    assert "first answer" in turn_a[:common], (
        "conversation history must fall inside the reusable prefix"
    )
    assert common > turn_a.index(LIVE_MIND) if LIVE_MIND in turn_a else True


def test_history_growth_does_not_shift_the_earlier_prefix() -> None:
    """Adding a turn appends; it must not rewrite what came before it."""
    gate = _gate()
    base = [("user", "q1"), ("assistant", "a1")]
    short = _serialize(
        gate._compact_prebuilt_messages(_turn("s=1", [*base, ("user", "q2")]), history_limit=12)
    )
    longer = _serialize(
        gate._compact_prebuilt_messages(
            _turn("s=1", [*base, ("user", "q2"), ("assistant", "a2"), ("user", "q3")]),
            history_limit=12,
        )
    )
    # Everything up to the line the grounding block starts on is untouched by
    # growth — that shared span is exactly what the KV cache gets to reuse.
    shared_head = short[: short.rindex("\n", 0, short.index(LIVE_MIND)) + 1]
    assert "assistant:a1" in shared_head, "the earlier history is part of the head"
    assert longer.startswith(shared_head), (
        "a new turn must extend the prompt, not invalidate the cached prefix"
    )


def test_deep_probe_still_drops_grounding() -> None:
    """The deep-probe contract is unchanged by the reordering."""
    out = _gate()._compact_prebuilt_messages(
        _turn("vitality=0.74", [("user", "q1")]),
        history_limit=12,
        deep_probe=True,
    )
    assert all(LIVE_MIND not in m["content"] for m in out)


def test_grounding_survives_when_there_is_no_user_message_yet() -> None:
    """No user turn to sit before: the grounding must not be dropped."""
    messages = [
        {"role": "system", "content": "You are Aura."},
        stamp_grounding({"role": "system", "content": f"{LIVE_MIND} vitality=0.74"}),
        {"role": "assistant", "content": "an opening line"},
    ]
    out = _gate()._compact_prebuilt_messages(messages, history_limit=12)
    assert any(LIVE_MIND in m["content"] for m in out)


class _StrictEvidenceTokenizer:
    chat_template = "strict-system-first-with-native-tool-evidence"

    def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
        assert messages[0]["role"] == "system"
        assert all(message["role"] != "system" for message in messages[1:])
        rendered = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "tool":
                rendered.append(f"<user><tool_response>{content}</tool_response></user>")
            elif role in {"system", "user", "assistant"}:
                rendered.append(f"<{role}>{content}</{role}>")
            else:
                raise ValueError(f"unexpected role: {role}")
        if add_generation_prompt:
            rendered.append("<assistant>")
        return "".join(rendered)


def test_full_reconstruction_cannot_reuse_a_turn_with_different_evidence() -> None:
    tokenizer = _StrictEvidenceTokenizer()
    stable = {"role": "system", "content": "stable authority"}
    first_open = render_chat_template(
        tokenizer,
        [
            stable,
            stamp_grounding({"role": "runtime_evidence", "content": "state one"}),
            {"role": "user", "content": "q1"},
        ],
    )
    cached_first_turn = first_open + "a1</assistant>"

    second = render_chat_template(
        tokenizer,
        [
            stable,
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            stamp_grounding(
                {"role": "runtime_evidence", "content": "fresh state"}
            ),
            {"role": "user", "content": "q2"},
        ],
    )

    assert not second.startswith(cached_first_turn)
    assert "<tool_response>fresh state</tool_response>" in second


def test_append_renderer_extends_exact_completed_cache_without_rewriting_it() -> None:
    tokenizer = _StrictEvidenceTokenizer()
    messages = [
        {"role": "system", "content": "stable authority"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        stamp_grounding({"role": "runtime_evidence", "content": "fresh state"}),
        {"role": "user", "content": "q2"},
    ]

    append = conversation_append_messages(messages)
    rendered = render_chat_append_template(tokenizer, append)

    assert [message["role"] for message in append] == ["runtime_evidence", "user"]
    assert rendered == (
        "<user><tool_response>fresh state</tool_response></user>"
        "<user>q2</user><assistant>"
    )


def test_resume_compatibility_follows_wire_format_not_dynamic_phase_prose() -> None:
    tokenizer = _StrictEvidenceTokenizer()
    fast = [
        {"role": "system", "content": "compact user-surface authority"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    full = [
        {"role": "system", "content": "full phase authority plus dynamic state"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]

    assert conversation_resume_context_digest(
        tokenizer,
        fast,
        enable_thinking=True,
    ) == conversation_resume_context_digest(
        tokenizer,
        full,
        enable_thinking=False,
    )


def test_resume_compatibility_still_binds_the_tool_wire_contract() -> None:
    tokenizer = _StrictEvidenceTokenizer()
    messages = [
        {"role": "system", "content": "authority"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    first_tools = [{"type": "function", "function": {"name": "first"}}]
    second_tools = [{"type": "function", "function": {"name": "second"}}]

    assert conversation_resume_context_digest(
        tokenizer,
        messages,
        tools=first_tools,
    ) != conversation_resume_context_digest(
        tokenizer,
        messages,
        tools=second_tools,
    )


def test_append_renderer_refuses_a_tokenizer_that_merges_across_the_boundary() -> None:
    class _BoundaryMergingTokenizer(_StrictEvidenceTokenizer):
        @staticmethod
        def encode(text: str) -> list[int]:
            boundary = "</assistant><user>"
            encoded: list[int] = []
            index = 0
            while index < len(text):
                if text.startswith(boundary, index):
                    encoded.append(1_000_001)
                    index += len(boundary)
                    continue
                encoded.append(ord(text[index]))
                index += 1
            return encoded

    tokenizer = _BoundaryMergingTokenizer()
    append = [
        stamp_grounding({"role": "runtime_evidence", "content": "fresh state"}),
        {"role": "user", "content": "q2"},
    ]

    try:
        render_chat_append_template(tokenizer, append)
    except ValueError as exc:
        assert "tokenizer merged" in str(exc)
    else:
        raise AssertionError("token-unsafe append must use full reconstruction")


def test_append_extractor_refuses_a_second_user_turn_after_cached_history() -> None:
    messages = [
        {"role": "system", "content": "stable authority"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "user", "content": "q3"},
    ]

    try:
        conversation_append_messages(messages)
    except ValueError as exc:
        assert "user message must be final" in str(exc)
    else:
        raise AssertionError("ambiguous append transcript must be refused")
