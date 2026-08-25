"""Two ways a reply stopped being hers, both measured live 2026-07-27.

**Scaffolding, served.** Asked how she would spend a free hour, she answered —
then kept going, into the system message that had been inserted just before
generation:

    "I'll go with my curiosity, not long-term memory consolidation.
     [SKILL EXECUTION] The skill 'web_search' just completed successfully. Its
     outcome is in your context as [SKILL RESULT: web_search]. Narrate it
     naturally — as yourself, not an output log."

The wording is a paraphrase, not a copy: she continued the instruction rather
than following it. No prompt phrasing makes that impossible, so containment
belongs at the egress. The genuine answer is kept and only the scaffold is cut,
because discarding real work over a formatting defect is the worse mistake.

**Someone else's memory, quoted as ours.** Asked what the first thing in this
conversation was, on a conversation four turns old:

    "The first thing you asked me was: 'If I had a whole Saturday with no
     obligations, what would I do?'"

The grounding path fired and logged success. It resolves against live working
memory — one global, fixed-length buffer shared by every origin the runtime
has — so "this conversation" meant "whatever survived in the buffer". A
confident quote of the wrong turn is worse than an admission: it is
indistinguishable from remembering.
"""
from __future__ import annotations

import time

import pytest

from core.conversation.grounded_recall import resolve_positional_turn
from core.conversation.response_reliability import (
    normalize_user_facing_format,
    strip_internal_scaffold,
)

LEAKED = (
    "I'll go with my curiosity, not long-term memory consolidation."
    "[SKILL EXECUTION] The skill 'web_search' just completed successfully. "
    "Its outcome is in your context as [SKILL RESULT: web_search]. "
    "Narrate it naturally — as yourself, not an output log."
)


# ── Scaffolding never ships ────────────────────────────────────────────────

def test_the_real_answer_survives_and_the_scaffold_does_not() -> None:
    assert (
        strip_internal_scaffold(LEAKED)
        == "I'll go with my curiosity, not long-term memory consolidation."
    )


@pytest.mark.parametrize(
    "marker",
    [
        "[SKILL EXECUTION]",
        "[SKILL RESULT: web_search]",
        "[TOOL RESULT: desktop_task]",
        "[LIVE MIND CONTEXT]",
        "[ACTIVE GROUNDING EVIDENCE]",
        "[FETCHED PAGE CONTENT]",
        "## PRESENT MOMENT",
        "## YOUR OWN INSTRUMENTS",
        "REMEMBER: You are Aura.",
    ],
)
def test_every_internal_marker_is_contained(marker: str) -> None:
    assert strip_internal_scaffold(f"Here is my answer. {marker} internals") == (
        "Here is my answer."
    )


def test_a_leading_marker_keeps_what_follows_it() -> None:
    assert (
        strip_internal_scaffold("[SKILL RESULT: web_search]\n\nHere is the real answer.")
        == "Here is the real answer."
    )


def test_a_reply_that_is_only_scaffold_becomes_empty() -> None:
    """Empty hands the turn to the existing empty-reply handling.

    Shipping a status page as conversation is the failure mode this whole pass
    exists to remove; an empty string is honest and already handled.
    """
    assert strip_internal_scaffold("[SKILL EXECUTION] only scaffold here") == ""


def test_ordinary_prose_is_untouched() -> None:
    ordinary = "I used a skill and here is the result: it worked well."
    assert strip_internal_scaffold(ordinary) == ordinary


def test_the_guard_sits_on_the_path_every_reply_takes() -> None:
    """normalize_user_facing_format is what the cortex output already runs."""
    assert normalize_user_facing_format(LEAKED) == (
        "I'll go with my curiosity, not long-term memory consolidation."
    )


# ── "This conversation" means this conversation ────────────────────────────

#: What a chat entry actually carries. An entry with neither an origin nor
#: this marker is unknown provenance, and grounded recall refuses it — the
#: rule that stops the runtime quoting its own writes back as the person's
#: words. The fixture predated that rule and modelled entries no live writer
#: produces, so every positional recall in this file resolved to None.
_FROM_THE_PERSON = {"metadata": {"source": "chat_api"}}


def _history() -> list[dict]:
    now = time.time()
    return [
        {
            "role": "user",
            "content": "If I had a whole Saturday with no obligations, what would I do?",
            "timestamp": now - 8 * 3600,
            **_FROM_THE_PERSON,
        },
        {"role": "assistant", "content": "...", "timestamp": now - 8 * 3600 + 10},
        {
            "role": "user",
            "content": "a curiosity probe the runtime wrote to itself",
            "timestamp": now - 7 * 3600,
            "origin": "curiosity",
        },
        {
            "role": "user",
            "content": "Hey Aura. What is 17 times 23?",
            "timestamp": now - 300,
            **_FROM_THE_PERSON,
        },
        {"role": "assistant", "content": "391.", "timestamp": now - 295},
        {
            "role": "user",
            "content": "Morning. What's it actually like in there right now?",
            "timestamp": now - 200,
            **_FROM_THE_PERSON,
        },
    ]


def test_the_first_turn_is_this_conversations_first_turn() -> None:
    assert (
        resolve_positional_turn(
            "What was the very first thing I asked you in this conversation?",
            "first",
            history=_history(),
        )
        == "Hey Aura. What is 17 times 23?"
    )


def test_a_turn_from_hours_ago_is_not_in_this_conversation() -> None:
    resolved = resolve_positional_turn(
        "What did I ask you first?", "first", history=_history()
    )
    assert "Saturday" not in (resolved or "")


def test_the_runtime_talking_to_itself_is_not_the_user_talking() -> None:
    """Several writers append role="user" without going through role_for_origin."""
    resolved = resolve_positional_turn(
        "What did I ask you first?", "first", history=_history()
    )
    assert "curiosity probe" not in (resolved or "")


def test_the_most_recent_turn_still_resolves() -> None:
    assert (
        resolve_positional_turn("What did I just say?", "last", history=_history())
        == "Morning. What's it actually like in there right now?"
    )


def test_an_unstamped_history_still_grounds() -> None:
    """Older buffers carry no timestamps; losing recall entirely is worse.

    A missing TIMESTAMP and a missing PROVENANCE are different absences. The
    first costs the conversation boundary and recall proceeds; the second means
    nobody can say a person wrote it, and recall refuses. This fixture used to
    omit both and so tested the wrong one.
    """
    history = [
        {"role": "user", "content": "first question", **_FROM_THE_PERSON},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second question", **_FROM_THE_PERSON},
    ]
    assert (
        resolve_positional_turn("What did I ask first?", "first", history=history)
        == "first question"
    )


def test_an_entry_with_no_provenance_is_refused() -> None:
    """Absence of both origin and marker is unknown authorship, not evidence.

    This is the rule that stops the runtime quoting its own writes back to a
    person as their own words, and it is worth a test of its own rather than
    living only in the shape of a fixture.
    """
    anonymous = [
        {"role": "user", "content": "who wrote this"},
        {"role": "assistant", "content": "unclear"},
    ]
    assert resolve_positional_turn("What did I ask first?", "first", history=anonymous) is None


def test_a_missing_timestamp_does_not_cost_provenance() -> None:
    stamped_out = [
        {"role": "user", "content": "first question", **_FROM_THE_PERSON},
        {"role": "user", "content": "second question", **_FROM_THE_PERSON},
    ]
    assert (
        resolve_positional_turn("What did I just say?", "last", history=stamped_out)
        == "second question"
    )
