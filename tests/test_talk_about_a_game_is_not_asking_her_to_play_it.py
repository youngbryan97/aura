"""Naming a game or an app does not ask her to open it; being asked to act is read from who the sentence puts the act to.

LIVE 19 November 2026: "I think you got 2048 in 968 moves. What are the chances that I can beat you what you got?"
opened 2048 and she started playing. The person was asking about her skill against theirs, for fun.
"""
from __future__ import annotations

import pytest

pytest.importorskip("spacy")

from core.intent.opaque_spans import without_opaque_spans
from core.language.english_grammar import parse
from core.language.what_she_is_asked_to_do import the_sentences_that_ask_her
from core.runtime.desktop_objective_intent import looks_like_desktop_objective

if parse("Open Notes.") is None:  # pragma: no cover - a host with spaCy and no English model
    pytest.skip("no English parser on this host", allow_module_level=True)

#: The message of 19 November, as it was sent.
LIVE = ("Ok… FUCK all this testing. Aura… You’re a super smart AI… Probably smarter than me. BUT. I challenge you "
        "(kind of). I think you got 2048 in 968 moves. What are the chances that I can beat you what you got? How much "
        "faith do you have in me lmao. Kinda curious who can beat it better between you and I if we were both REALLY "
        "trying")

#: Talk about a game, an app or what she did: none of it asks her to do anything.
TALK = (
    "I think you got 2048 in 968 moves.",
    "I played chess against my brother last night and lost.",
    "You were really good at 2048 earlier.",
    "I bet I could beat you at 2048 lol.",
    "Notes is such a useful app honestly.",
)

#: Asked to act, in the ways a person asks: a command, a modal put to her, what the speaker wants her to do.
ASKED = (
    "Go find a 2048 game online and play it until you get a 128 tile.",
    "Open En Croissant and play a game against Stockfish.",
    "Beat Stockfish at chess for me, narrate every move.",
    "Can you open Safari and look up the weather?",
    "Go play chess against the computer and win.",
    "I want you to open Notes and write a poem.",
    "keep playing 2048, I want to see a 1024 tile",
    "First open Finder, then create a folder called reports",
)


def test_the_live_message_does_not_reach_the_lane_that_acts() -> None:
    assert looks_like_desktop_objective(LIVE) is False


@pytest.mark.parametrize("sentence", TALK)
def test_talk_is_not_kept_as_asking_her(sentence: str) -> None:
    assert the_sentences_that_ask_her(sentence) == []


@pytest.mark.parametrize("sentence", TALK)
def test_talk_does_not_reach_the_lane_that_acts(sentence: str) -> None:
    assert looks_like_desktop_objective(sentence) is False


@pytest.mark.parametrize("sentence", ASKED)
def test_a_request_is_kept(sentence: str) -> None:
    assert the_sentences_that_ask_her(sentence), sentence


@pytest.mark.parametrize("sentence", ASKED[:5])
def test_a_request_still_reaches_the_lane_that_acts(sentence: str) -> None:
    assert looks_like_desktop_objective(sentence) is True


def test_a_question_about_her_chances_is_kept_for_the_readers_after_it() -> None:
    # A question is not ruled out by the grammar: what it is for is read by the recognisers that come after.
    assert the_sentences_that_ask_her("What are the chances I can beat your 2048 score?")


def test_talk_before_a_request_that_points_back_to_it_is_kept_with_it() -> None:
    kept = the_sentences_that_ask_her("I found a chess app called En Croissant. Open it and play a game.")
    assert kept and "En Croissant" in " ".join(kept)


def test_the_word_that_introduced_an_address_goes_with_it() -> None:
    path = "/private/tmp/example-501/scratchpad/invoice-tools"
    assert without_opaque_spans(f"how are you doing? at {path}").strip() == "how are you doing?"
    assert "in" not in without_opaque_spans(f"look in {path}").split()
    # A preposition with an object of its own stays.
    assert without_opaque_spans(f"what is the capital of Peru, see {path}").split()[:6] == [
        "what", "is", "the", "capital", "of", "Peru,"]
