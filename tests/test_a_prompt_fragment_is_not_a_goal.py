"""An instruction to a model is not something she wants.

Live 2026-08-18, the integrity monitor reported 22 cognitive_engine
degradations in thirty minutes, every one of them the same shape:

    objective repeatedly unresolved: You are writing Aura
    objective repeatedly unresolved: Summarize the follow

Both are clipped system instructions that were promoted into volitional state.
Neither can ever be resolved — there is no "the following" in an objective and
no task in "you are" — so they generate sustained objective friction forever,
and unresolved obligations are what hold the autonomy gates shut.

The existing scaffold filter could not catch them and structurally could not:
it needs TWO markers from a phrase list, or one marker plus 500+ characters.
Truncation removes both. The shorter and more mangled the fragment, the more
certainly it slipped through — the filter was weakest exactly where the
evidence was worst.
"""

from __future__ import annotations

import pytest

from core.autonomy.research_goal_filter import (
    is_instruction_shaped_goal,
    is_stale_or_prompt_scaffold_goal,
)

#: Verbatim from the live degradation log.
LIVE_FRAGMENTS = ("You are writing Aura", "Summarize the follow")


@pytest.mark.parametrize("fragment", LIVE_FRAGMENTS)
def test_the_fragments_that_ran_forever_are_rejected(fragment):
    assert is_stale_or_prompt_scaffold_goal(fragment), fragment


@pytest.mark.parametrize(
    "scaffold",
    [
        "You are an AI assistant that helps users",
        "Your task is to classify the message",
        "Your instructions are below",
        "Summarize the following conversation",
        "Rewrite the above passage",
        "Given the following context, answer",
        "Based on the above, produce a plan",
        "Analyze the transcript and report",
    ],
)
def test_instruction_shapes_are_rejected_whatever_the_topic(scaffold):
    """Matched on grammatical FORM, so it does not vary with subject."""
    assert is_instruction_shaped_goal(scaffold), scaffold


@pytest.mark.parametrize(
    "goal",
    [
        "Investigate the paradox of xenobiology concepts and its implications",
        "Find the most obscure fact about xenobiology concepts.",
        "Deconstruct and comprehensively research: user privacy",
        "enhance memory retention",
        "Understand why the disk keeps filling up",
        "Your privacy matters to Bryan",
    ],
)
def test_real_goals_survive(goal):
    """Over-filtering silences her autonomy, which is the same failure inverted."""
    assert not is_stale_or_prompt_scaffold_goal(goal), goal


def test_a_goal_about_follow_up_work_is_not_an_instruction():
    """"the follow" is the clipped instruction; "the follow-up" is a real thing.

    The bare `follow` match is what catches the truncation, so it has to stop
    short of a legitimate goal that simply starts the same way.
    """
    assert not is_instruction_shaped_goal("Summarize the follow-up notes from the meeting")
    assert is_instruction_shaped_goal("Summarize the follow")


def test_truncation_does_not_help_a_fragment_survive():
    """The old thresholds got WEAKER as the evidence got worse."""
    whole = "You are writing Aura's reply to the user, using the context below."
    for cut in range(len("You are writing"), len(whole)):
        fragment = whole[:cut]
        assert is_stale_or_prompt_scaffold_goal(fragment), fragment
