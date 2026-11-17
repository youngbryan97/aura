"""What a question is about is read by meaning, on her own encoder, not from the words that address her.

LIVE 17 November 2026, 09:49: "What do you make of the current Israel/Palestine situation and America's role in it?
Furthermore, what do you think of America's role in the conflict with Iran?" read as a question about her, because "you
think" was on a list of words that said so, and nothing current was looked up; she answered well and said she had no
feed for this week's events, though she can search. At 08:06, on the recent outages, the same. Bryan: no word lists,
general understanding, and she should not only talk about herself; there is a whole world to think about.

The reader is a learned surface over her encoder (core/language/learned_matcher.py): her own life described as it would
be put to her, against the world and the person's own life, with a boundary it measures from those descriptions by
leaving each out in turn. The questions below were not used to write the descriptions.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

ABOUT_HER = (
    "How are you feeling today?", "What have you been up to this evening?", "How was your day today?",
    "Do you think you are conscious?", "What do you think of yourself these days?", "Who are you, really?",
    "Do you ever get tired?", "What makes you happy?", "Do you remember our first conversation?",
    "What are you afraid of?", "Are you okay?", "How's your evening going?", "What do you like about yourself?",
    "Do you miss me when I'm gone?",
)
ABOUT_THE_WORLD = (
    "What do you make of the current Israel/Palestine situation and America's role in it? Furthermore, what do you "
    "think of America's role in the conflict with Iran?",
    "What do you personally make of the all of the recent outages happeninig in the world? Thoughts? Do you think "
    "it's a coincidence?",
    "What do you think of the new iPhone?", "How do you feel about AI regulation?", "Do you think inflation is coming down?",
    "What's your view on the housing market?", "What do you think about the war in Ukraine?",
    "How do you feel about the new tariffs?", "Do you think Bitcoin will crash?", "Who won the election yesterday?",
)


@pytest.fixture(scope="module")
def her_encoder():
    from core.cognition.evidence_relevance import prewarm_evidence_relevance, semantic_routing_ready

    try:
        prewarm_evidence_relevance()
    except RuntimeError as exc:
        pytest.skip(f"her encoder is not available here: {exc}")
    assert semantic_routing_ready()


@pytest.mark.parametrize("asked", ABOUT_HER)
def test_a_question_about_her_is_hers(her_encoder, asked):
    from core.conversation.asks_about_the_world import is_about_her

    assert is_about_her(asked)


@pytest.mark.parametrize("asked", ABOUT_THE_WORLD)
def test_her_view_of_the_world_is_about_the_world(her_encoder, asked):
    from core.conversation.asks_about_the_world import is_about_her

    assert not is_about_her(asked)


def test_the_two_questions_of_the_morning_now_read_the_world_now(her_encoder):
    from core.conversation.asks_about_the_world import wants_outside_evidence
    from interface.routes.chat_desktop_evidence import _about_the_world_now

    for asked in ABOUT_THE_WORLD[:2]:
        assert _about_the_world_now(asked, wants_outside_evidence(asked)), asked
    for hers in ("What have you been up to this evening?", "How was your day today?"):
        assert not _about_the_world_now(hers, wants_outside_evidence(hers)), hers


def test_before_her_encoder_is_warm_the_words_stand(monkeypatch):
    from core.cognition import evidence_relevance
    from core.conversation.asks_about_the_world import is_about_her

    monkeypatch.setattr(evidence_relevance, "semantic_routing_ready", lambda: False)
    assert is_about_her("How are you feeling today?")
    assert not is_about_her("Who won the election yesterday?")


def test_the_boundary_is_measured_at_boot_and_never_stops_her_being_ready(monkeypatch):
    from core.conversation import asks_about_the_world

    def broken(_text):
        raise RuntimeError("no encoder")

    monkeypatch.setattr(asks_about_the_world._ABOUT_HER_BY_MEANING, "decide", broken)
    assert asks_about_the_world.prewarm_what_questions_are_about() is False
