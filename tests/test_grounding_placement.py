"""Where volatile grounding sits in the message list, and why it matters.

Two forces pull in opposite directions and both have already caused a live
failure:

- Put the per-turn grounding FIRST and its churn invalidates the KV prefix for
  everything behind it. Measured: reuse of 21 of 31,718 tokens.
- Put it dead LAST and it stands between the person's question and the model's
  turn, so the model continues the grounding instead of answering. Measured:
  asked to run a sandbox calculation, the whole reply was "Things feel
  unusually settled right now. My attention is on internal monitoring..."

The resolution is one position: immediately BEFORE the final user message.
"""

from __future__ import annotations


def _place(messages, blocks):
    """Mirror of the gate's placement rule, exercised directly."""
    from core.brain import inference_gate  # noqa: F401  (import guard)

    grounding_message = {"role": "system", "content": "\n\n".join(blocks)}
    final_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], dict)
            and str(messages[index].get("role", "")).strip().lower() == "user"
        ),
        None,
    )
    if final_user_index is None:
        return [*messages, grounding_message]
    return [
        *messages[:final_user_index],
        grounding_message,
        *messages[final_user_index:],
    ]


def test_the_persons_question_is_the_last_thing_the_model_sees():
    messages = [
        {"role": "system", "content": "stable identity"},
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "run the sandbox and show me the result"},
    ]
    placed = _place(messages, ["## PRESENT MOMENT\nclock", "## WHAT YOU ACTUALLY JUST DID\nreceipts"])

    assert placed[-1]["role"] == "user", (
        "the model continues the nearest text; it must be the person's question"
    )
    assert placed[-1]["content"] == "run the sandbox and show me the result"
    assert placed[-2]["role"] == "system"
    assert "PRESENT MOMENT" in placed[-2]["content"]


def test_every_stable_token_still_precedes_the_volatile_block():
    """The KV-prefix property volatile-last existed to protect."""
    messages = [
        {"role": "system", "content": "stable identity"},
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "turn two"},
    ]
    placed = _place(messages, ["volatile clock"])

    volatile_at = next(
        index for index, msg in enumerate(placed) if msg["content"] == "volatile clock"
    )
    stable_before = [msg["content"] for msg in placed[:volatile_at]]
    assert stable_before == ["stable identity", "turn one", "reply one"], (
        "history must remain a contiguous reusable prefix ahead of the churn"
    )
    # Only the new turn sits behind the volatile block, and it had to be
    # prefilled anyway.
    assert [msg["content"] for msg in placed[volatile_at + 1 :]] == ["turn two"]


def test_grounding_is_never_dropped_when_there_is_no_user_turn():
    messages = [{"role": "system", "content": "stable"}]
    placed = _place(messages, ["volatile clock"])
    assert any("volatile clock" in msg["content"] for msg in placed), (
        "grounding must survive even with no user message to ride ahead of"
    )


def test_the_gate_actually_implements_this_placement():
    """Pin the production code, not just the rule."""
    from core.brain import inference_gate
    from tests.source_contract import family_text

    # The splice moved into a lifted sibling of the gate (2026-09-20); the
    # family reader sees the parent and its siblings as one text.
    source = family_text(inference_gate)
    assert "final_user_index" in source, "the gate must locate the final user turn"
    assert "*messages[:final_user_index]," in source, (
        "grounding must be spliced in AHEAD of the final user message"
    )
    # The old placement must be gone.
    assert 'messages = [*messages, {\n                "role": "system",' not in source, (
        "grounding must not be appended after the final user message"
    )
