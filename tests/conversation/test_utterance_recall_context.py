"""Asking what was said must open the transcript window.

LIVE DEFECT, 2026-08-10. "quote me the exact first sentence I said to you
today, and tell me what I was wearing — one of those is answerable and one
isn't; I want to see if you can tell which" was answered by refusing BOTH.

The refusal was honest given what she had, and that is the point: the turn was
not classified as needing recent context, so it ran on the live desktop
default of four exchanges (min(4, _RECENT_CONVERSATION_CONTEXT_EXCHANGES),
against a declared window of 12). The first sentence of the session was long
out of that window, so the one question whose entire answer sat in the
transcript could not be answered from it.

The classifier caught "what did we just talk about" and "what did you TELL me
earlier" but missed the most direct forms of the same request. Each phrasing
below was verified to miss before the fix.
"""

from __future__ import annotations

import pytest


def _needs(text: str) -> bool:
    from interface.routes.chat import _desktop_turn_needs_recent_context

    return _desktop_turn_needs_recent_context(text)


@pytest.mark.parametrize(
    "message",
    [
        "quote me the exact first sentence I said to you today",
        "what did you say a minute ago",
        "remind me what you told me earlier",
        "repeat what you just said",
        "what were your exact words",
        "you said something earlier, what was it",
        "earlier you told me something you find interesting. tell me what it "
        "was, in your own words",
        "restate the line you used about coherence",
        "what sentence did I open with",
    ],
)
def test_questions_about_what_was_said_need_the_transcript(message: str) -> None:
    assert _needs(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "what did we just talk about",
        "summarize this conversation so far",
    ],
)
def test_previously_working_recall_phrasings_still_work(message: str) -> None:
    """The fix must not regress what the old classifier already caught."""
    assert _needs(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "what is the capital of France",
        "open my notes and write a paragraph about yourself",
        "take a screenshot of my screen",
        "what is 17 times 4",
        "build me a checkers game",
    ],
)
def test_ordinary_turns_do_not_pull_the_transcript(message: str) -> None:
    """The expensive direction: every turn dragging in twelve exchanges."""
    assert _needs(message) is False


def test_utterance_pattern_is_the_thing_being_matched() -> None:
    """Attribute the behaviour to this fix, not to a neighbouring classifier."""
    from interface.routes.chat import _UTTERANCE_RECALL_RE

    assert _UTTERANCE_RECALL_RE.search("what were your exact words")
    assert _UTTERANCE_RECALL_RE.search("repeat what you just said")
    # A status question is recent-context-worthy for its own reasons; this
    # pattern must not be what claims it.
    assert not _UTTERANCE_RECALL_RE.search("how are you feeling right now")
    assert not _UTTERANCE_RECALL_RE.search("what is the capital of France")


def test_the_classifier_decides_between_the_whole_transcript_and_none() -> None:
    """Documents why the classifier matters, on the window that exists now.

    It used to choose between four exchanges and a declared twelve. The two
    constants were replaced by one visible window, read whole when the turn
    needs the transcript and not at all when it does not, so a turn this
    classifier misses now gets no transcript rather than a short one.
    """
    import ast
    from pathlib import Path

    from core.conversation.delivered_history import VISIBLE_CONVERSATION_EXCHANGES

    assert VISIBLE_CONVERSATION_EXCHANGES >= 12

    # The window is read whole or not at all. Asserted structurally, because a
    # literal source match goes stale the first time the file is reformatted
    # and then reports green forever.
    tree = ast.parse(Path("interface/routes/chat.py").read_text(encoding="utf-8"))
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.IfExp)
        and isinstance(node.body, ast.Name)
        and node.body.id == "VISIBLE_CONVERSATION_EXCHANGES"
    ]
    assert branches, "the visible window is no longer chosen for the turn"
    assert any(
        isinstance(branch.orelse, ast.Constant) and branch.orelse.value == 0
        for branch in branches
    ), "a turn the classifier misses no longer falls to zero"
