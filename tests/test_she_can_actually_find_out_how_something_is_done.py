"""Researching a task, from the question she asks to the answer she keeps.

Five separate breaks between "find out how this is done" and one usable
sentence, all found live on 2026-08-26 while she played a game knowing
nothing about it.
"""
from __future__ import annotations

import pytest

from core.agency.task_knowledge import (
    _identifying_words,
    _is_a_heading,
    _asked_more_than_one_way,
    how_is_this_done,
    says_how,
    usable_sentences,
    what_it_is_about,
)


@pytest.mark.parametrize(
    ("said", "about"),
    [
        ("play 2048 until you get a 256 tile", "play 2048"),
        (
            "Find 2048 online, play it, and get to a 256 tile. Say what you are about "
            "to do before each move, and tell me here when you have it.",
            "play 2048",
        ),
        ("fill in the visa application form step by step and narrate as you go",
         "fill in the visa application form"),
    ],
)
def test_the_question_is_about_the_task_not_the_instructions_to_her(said, about):
    """A request is written to her, and most of it is addressed to her: when
    to stop, what to say while working, who to tell at the end.

    LIVE: researching "play 2048 until you get a 256 tile" asked "how to do
    this well: play 2048 until you get a 256 tile" and came back with four
    dictionary definitions of the word "do".
    """
    assert what_it_is_about(said) == about
    assert how_is_this_done(said) == f"how to {about}"


def test_she_asks_more_than_one_way():
    """One phrasing is a bet on how an engine weights words. "How to play
    2048" returned general games portals because "play" is the commonest word
    in it; "2048" alone returned the game."""
    asked = _asked_more_than_one_way("how to play 2048")
    assert asked[0] == "how to play 2048"
    assert any(phrasing.startswith("2048") for phrasing in asked[1:])
    assert _identifying_words("how to play 2048") == ("2048",)


@pytest.mark.parametrize(
    ("line", "instructs"),
    [
        ("Use arrow keys to move the tiles.", True),
        ("When two tiles having the same number touch, they join into one!", True),
        ("Keep your largest tile in a corner.", True),
        ("Feed the starter twice a day with equal flour and water.", True),
        ("Press Tab to move between fields.", True),
        ("The game was created by Gabriele Cirulli in 2014.", False),
        ("It is a single-player sliding tile puzzle.", False),
        ("Sourdough is a bread made by fermentation.", False),
        ("This article has been viewed 1,204,558 times.", False),
    ],
)
def test_an_instruction_is_recognised_by_grammar_not_by_vocabulary(line, instructs):
    """A list of strategy words knows about games and knows nothing about
    baking, tax forms or a deployment runbook. An imperative is an imperative
    in every subject."""
    assert says_how(line) is instructs


def test_a_title_is_not_an_instruction():
    """LIVE: "2048 - Play the Free Online Game Privacy Policy" opens with a
    verb, so every rule about grammar agreed it was an instruction."""
    assert _is_a_heading("2048 - Play the Free Online Game Privacy Policy")
    assert _is_a_heading("Sourdough Starter Recipe | King Arthur Baking")
    assert not _is_a_heading("Use arrow keys to move the tiles.")


def test_the_sentences_kept_are_the_ones_that_say_how():
    page = (
        "2048 - Play the Free Online Game\n\n"
        "Use your arrow keys to move the tiles.\n\n"
        "When two tiles with the same number touch, they merge into one.\n\n"
        "The game was created by Gabriele Cirulli in 2014.\n\n"
        "This article has been viewed 1,204,558 times.\n\n"
    )
    kept = usable_sentences(page)
    assert any("arrow keys" in line for line in kept)
    assert any("merge into one" in line for line in kept)
    assert not any("Gabriele" in line for line in kept)
    assert not any("viewed" in line for line in kept)


def test_she_reads_past_a_page_that_answers_nothing():
    """LIVE: the top result for how a game is played was the game itself, and
    the only sentence she came away with was the title of its privacy policy.
    """
    import inspect

    from core.agency import task_knowledge

    source = inspect.getsource(task_knowledge._read_the_best_answer)
    assert "PAGES_READ" in source
    assert task_knowledge.PAGES_READ >= 2
    # And it reads the field the extractor actually writes.
    assert 'getattr(extract, "body", "")' in source


def test_a_page_that_gives_nothing_is_recorded_rather_than_passed_over():
    """Reading nothing and reporting nothing is how a broken chain stays
    invisible: every page came back empty and no error existed anywhere."""
    import inspect

    from core.agency import task_knowledge

    source = inspect.getsource(task_knowledge._read_the_best_answer)
    assert "record_degradation" in source
    assert "nothing that says how" in source
