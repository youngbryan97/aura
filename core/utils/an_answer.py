"""What separates an answer from text that arrived where an answer should be.

Two things come back from a model that are not answers, and both pass every
test of shape: the question handed back, and a passage that stopped in the
middle. Measured live on 2026-08-26, both were used as if they were answers —
one was held as her plan for a whole game of 2048, and the other became the
question a deeper pass then went off and answered.

The tests here are structural. They ask what the text does, not what it says,
so they hold for any question in any language a model writes sentences in.
"""

from __future__ import annotations

import re

__all__ = ["adds_nothing_to", "content_words", "was_cut_off"]

#: The least a line can add and still have added something. Below this an
#: answer is a rearrangement of the question.
ENOUGH_NEW_WORDS = 4

#: What a finished sentence ends with.
_ENDS = ".!?:;\"')]}…。！？"

#: Marks that come in pairs, and are proof of a cut when they do not.
_PAIRS = (('"', '"'), ("(", ")"), ("[", "]"), ("{", "}"))


def content_words(said: str) -> set[str]:
    """The words in a line that carry what it is about."""
    return {word for word in re.findall(r"[a-z0-9]+", str(said or "").lower()) if len(word) > 2}


def adds_nothing_to(answer: str, asked: str) -> bool:
    """True when the answer is the question handed back.

    A model that is warming up returns the instruction it was given, and every
    test of shape passes it: it has the length of an answer and the words of
    one. What it does not have is anything that was not already in the
    question.
    """
    if not str(asked or "").strip():
        return False
    return len(content_words(answer) - content_words(asked)) < ENOUGH_NEW_WORDS


def was_cut_off(said: str) -> bool:
    """True when the text stopped in the middle rather than finishing.

    Structural, so it needs no vocabulary and no threshold: a writer who was
    writing sentences and stopped without ending one was interrupted, and so
    was a writer who opened a quote or a bracket and never closed it. A short
    reply that was never a sentence — "left" — ends nothing and is complete.
    """
    text = " ".join(str(said or "").split())
    if not text:
        return False
    for opening, closing in _PAIRS:
        if opening == closing:
            if text.count(opening) % 2:
                return True
        elif text.count(opening) > text.count(closing):
            return True
    if text[-1] in _ENDS:
        return False
    return any(mark in text[:-1] for mark in ".!?。！？")
