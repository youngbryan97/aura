"""A document about her own doing has a source, and it is not her self-model.

LIVE 2026-08-26: asked for "one sentence about what you did tonight" she
wrote "I spent tonight thinking through the architecture of my own
continuity, examining how memory structures persist across sessions". Real
prose, in her own voice, and not what she had done — she had played a game to
a 256 tile and written files. The receipts were there; the prompt never
showed them to her, so the only source left was her self-model.
"""
from __future__ import annotations

import inspect

import pytest

from core.skills.desktop_task import _ABOUT_HER_DOING, _what_she_actually_did


@pytest.mark.parametrize(
    "objective",
    [
        "one sentence about what you did tonight",
        "make a file with a paragraph about your evening",
        "write up what you worked on today",
        "summarise what you got up to this evening",
        "a note about what you have done so far",
    ],
)
def test_a_request_about_her_own_doing_is_recognised(objective):
    assert _ABOUT_HER_DOING.search(objective)


@pytest.mark.parametrize(
    "objective",
    [
        "write a note about whales",
        "a paragraph about sourdough starters",
        "draft a letter to the council",
        "write a haiku about rain",
    ],
)
def test_a_request_about_anything_else_is_left_alone(objective):
    """A note about whales must not carry her receipts."""
    assert not _ABOUT_HER_DOING.search(objective)
    assert _what_she_actually_did(objective) == ""


def test_the_record_is_read_directly_not_through_the_question_gate():
    """`past_actions_answer` answers "what did you just do?". A request to
    WRITE about it is not that shape and never will be — routed through that
    gate, it returned nothing every time."""
    source = inspect.getsource(_what_she_actually_did)
    assert "resolve_past_actions" in source
    assert "render_past_actions" in source
    assert "past_actions_answer" not in source


def test_the_prompt_carries_the_record_when_there_is_one():
    from core.skills.desktop_task import DesktopTaskSkill

    source = inspect.getsource(DesktopTaskSkill._synthesize_requested_writing)
    assert "_what_she_actually_did(objective)" in source
    assert "What you actually did, from your own records" in source
    # And says nothing about records when there are none.
    assert "if recalled else \"\"" in source
