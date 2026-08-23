"""A judgement about meaning belongs to the learned surface.

2026-08-22: "that should aim to do better than heuristics. we have a
lingusitic general learning substrate for anything we may think to default to
heuristics for."

Two readers written this same day gated consequential decisions from word
lists alone: whether the turn asks for something to exist, which sets the
effect ceiling and therefore which capabilities may be offered at all; and
whether the answer has to come from outside her, which decides whether a
search runs.

A list of nouns will always be the nouns somebody thought of. The lists stay
as the floor — they settle the obvious cases and teach the surface as they go
— and the surface is the mechanism.
"""

from __future__ import annotations

import pytest

from core.conversation.asks_about_the_world import _NEEDS_OUTSIDE, wants_outside_evidence
from core.intent.artifact_request import _WANTS_A_THING, asks_for_an_artifact


@pytest.mark.parametrize(
    ("asked", "wants"),
    [
        ("Six slides, no fluff: what you are", True),
        ("put together a short report on the ledger", True),
        ("build me a little web app", True),
        ("what is a deck?", False),
        ("how are you feeling today?", False),
        ("who founded Hugging Face?", False),
    ],
)
def test_the_artifact_question_still_answers_what_it_did(asked: str, wants: bool):
    assert asks_for_an_artifact(asked) is wants, asked


@pytest.mark.parametrize(
    ("asked", "wants"),
    [
        ("who founded Hugging Face?", True),
        ("tell me about Anthropic the company", True),
        ("link your sources", True),
        ("how are you feeling today?", False),
        ("tell me about yourself", False),
        ("what is 7919 * 6367?", False),
    ],
)
def test_the_evidence_question_still_answers_what_it_did(asked: str, wants: bool):
    assert wants_outside_evidence(asked) is wants, asked


def test_the_floor_teaches_the_artifact_surface():
    before = len(_WANTS_A_THING.positives)
    asks_for_an_artifact("put together a short report on the migration risks")
    assert len(_WANTS_A_THING.positives) > before


def test_the_floor_teaches_the_evidence_surface():
    before = (len(_NEEDS_OUTSIDE.positives), len(_NEEDS_OUTSIDE.negatives))
    wants_outside_evidence("who founded Quibbleflax Dynamics?")
    wants_outside_evidence("how are you feeling about your own uptime today?")
    after = (len(_NEEDS_OUTSIDE.positives), len(_NEEDS_OUTSIDE.negatives))
    assert after[0] > before[0]
    assert after[1] > before[1]


def test_both_surfaces_carry_their_own_declaration():
    """Declared examples, so the surface has something to be right about
    before anything has been observed."""
    for surface in (_WANTS_A_THING, _NEEDS_OUTSIDE):
        assert surface.positives and surface.negatives
        assert surface.name
