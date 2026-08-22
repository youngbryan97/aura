"""A hallucinated continuation is cut, not flagged and served anyway.

LIVE 2026-08-18, asked to append a line to a file, this reached the person:

    "Would you like to check the file or do something else with it?<tool_call>
     !user yes check it. Read the contents back to me. Keep them on screen as
     you speak. Don't close the file or terminal window. I want to see ever"

The model had begun writing the CONVERSATION rather than a turn in it —
inventing the person's next message and a tool-call token. The artifact was
detected: assess_user_facing_reply adds "prompt_artifact" for exactly this. It
was only ever a REASON, the reason is repairable, and nothing repaired it, so
the draft went out with the invention attached.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    assess_user_facing_reply,
    contains_prompt_artifact,
    strip_prompt_artifacts,
)


def test_the_live_leak_is_cut_at_the_marker() -> None:
    leak = (
        "Would you like to check the file or do something else with it?"
        "<tool_call> !user yes check it. Read the contents back to me."
    )

    kept = strip_prompt_artifacts(leak)

    assert kept == "Would you like to check the file or do something else with it?"
    assert "tool_call" not in kept
    assert "!user" not in kept


@pytest.mark.parametrize(
    "marker",
    [
        "<tool_call>",
        "</tool_call>",
        "<function_call>",
        "<|im_start|>",
        "<|im_end|>",
        "!user",
        "!assistant",
        "<|start_of_turn|>",
    ],
)
def test_every_turn_marker_family_is_cut(marker: str) -> None:
    kept = strip_prompt_artifacts(f"The real answer.{marker} invented continuation")

    assert kept == "The real answer."


def test_a_clean_reply_is_untouched() -> None:
    original = "A normal answer with no leaks."

    assert strip_prompt_artifacts(original) == original


def test_structured_pseudocode_state_is_not_executable_prompt_scaffold() -> None:
    prompt = (
        "Explain Dijkstra with its invariant, numbered pseudocode, a worked "
        "example, both complexities, and the negative-weight alternative."
    )
    answer = """(1) Once the minimum-distance vertex is extracted, its distance is final.

(2) Pseudocode:
```text
state: distance and predecessor maps
1: initialize the source distance to zero
2: extract the minimum and relax each outgoing edge
```

(3) For A-B=1, A-C=4, B-C=2, B-D=5, and C-D=1, the final distances are A=0, B=1, C=3, D=4.

(4) A binary heap costs O((V+E) log V); an array costs O(V^2+E).

(5) Negative weights break the settled-distance invariant; use Bellman-Ford instead."""

    assessment = assess_user_facing_reply(prompt, answer)

    assert assessment.ok
    assert not contains_prompt_artifact(answer)
    assert "prompt_artifact" not in assessment.reasons
    assert strip_prompt_artifacts(answer) == answer


def test_inline_prompt_vocabulary_is_quoted_data() -> None:
    answer = "The label `state:` is part of the schema, not a new instruction."

    assert not contains_prompt_artifact(answer)
    assert strip_prompt_artifacts(answer) == answer


def test_quoted_next_user_turn_remains_a_transcript_continuation() -> None:
    leaked = "The answer is complete.\n\n> User: now invent my next request"

    assert contains_prompt_artifact(leaked)


def test_unrelated_terse_headings_do_not_become_a_scaffold_run() -> None:
    answer = (
        "State: healthy\nThe runtime checks have all passed.\n\n"
        "Mood: steady\nThat reading comes from the current affect state.\n\n"
        "Goals: ship\nThe next step is live verification."
    )

    assert not contains_prompt_artifact(answer)


def test_a_reply_that_is_only_continuation_yields_nothing() -> None:
    """No reply, only continuation — the caller must not serve a fragment."""
    assert strip_prompt_artifacts("<tool_call> only continuation") == ""


@pytest.mark.parametrize("value", [None, "", "   "])
def test_empty_input_is_safe(value) -> None:
    assert strip_prompt_artifacts(value) == ""


def test_the_funnel_applies_it() -> None:
    """Detection without excision is what let this reach a person."""
    import inspect

    from interface.routes import chat

    assert "strip_prompt_artifacts" in inspect.getsource(chat)
