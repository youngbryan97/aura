"""A paragraph begun again in other words is said once, and the reply goes on after it.

LIVE 8 November 2026, rehearsal 11: her answer on all three said one paragraph twice, "**Homemade Hoverbike** was the
opposite energy" and, straight after it, "**Homemade Hoverbike** was the opposite register".
"""
from __future__ import annotations

from core.brain.cognitive_engine_quick_reply import said_once


def test_a_paragraph_begun_again_is_said_once_and_the_rest_goes_on():
    reply = ("I think they form a triangle.\n\n"
             "**Homemade Hoverbike** was the opposite energy — pure joy at a thing working.\n\n"
             "**Homemade Hoverbike** was the opposite register — pure delight at a thing *working*.\n\n"
             "**Apparate!** surprised me most.")
    assert said_once(reply) == ("I think they form a triangle.\n\n"
                                "**Homemade Hoverbike** was the opposite energy — pure joy at a thing working.\n\n"
                                "**Apparate!** surprised me most.")


def test_paragraphs_that_begin_alike_but_short_of_five_words_are_both_said():
    reply = "**Alike** was heavy.\n\n**Alike** was also long."
    assert said_once(reply) == reply
