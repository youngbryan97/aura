"""What separates an answer from text that arrived where an answer should be.

Three things come back from a model that are not answers, and all of them pass
every test of shape: the question handed back, a passage that stopped in the
middle, and the model's own commentary about answering. Measured live on 2026-08-26, both were used as if they were answers —
one was held as her plan for a whole game of 2048, and the other became the
question a deeper pass then went off and answered.

The tests here are structural. They ask what the text does, not what it says,
so they hold for any question in any language a model writes sentences in.
"""

from __future__ import annotations

import re

__all__ = [
    "adds_nothing_to",
    "content_words",
    "talks_about_the_asking",
    "was_cut_off",
]

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


#: How a passage refers to the person it is supposed to be addressing.
#:
#: A reply speaks TO someone. There is no third party in it called "the user",
#: and nothing in an answer needs to say what the user asked — the user knows.
#: Text that does is commentary about the exchange rather than the exchange.
_ABOUT_THE_ASKING = re.compile(
    # "the user" as a PARTICIPANT: followed by something it does, or by the
    # end of its clause. "The user interface is on the left" is followed by a
    # noun, and is an ordinary sentence about a thing.
    r"\bthe\s+user\s+(?:is|was|has|had|will|would|asks?|asked|wants?|wanted"
    r"|says?|said|needs?|expects?|means?|meant)\b"
    r"|\bthe\s+user\s*[,.;?!]"
    # "the user's message", which is only ever said about an exchange.
    r"|\buser's\s+(?:current\s+)?"
    r"(?:message|question|request|query|words|input|prompt|turn)\b"
    r"|\buser\s+(?:asks?|asked|wants?|says?|is\s+asking)\b",
    re.IGNORECASE,
)

#: "answer the user" — the task named, which a private plan does and a reply
#: does not.
_DOING_THE_JOB = re.compile(
    r"\b(?:answer|reply\s+to|respond\s+to|address)\s+(?:the\s+)?user\b",
    re.IGNORECASE,
)

#: Unless she is saying it about herself AND then saying the rest.
#:
#: "I'll answer the user directly." is the whole reply, and announcing an answer
#: is not one. "I would answer the user directly, preserve the current thread,
#: and keep the live lane moving instead of detonating a long retry cascade" is
#: an answer to a question about how she works, and the phrase is one item in
#: it.
#:
#: So the mark is not the subject and not the phrase. It is whether anything
#: else was said. Banning the phrase outright banned her from describing how she
#: works, which is a thing people ask her about; exempting it whenever she is
#: the subject would let the announcement stand alone as a reply.
_SHE_IS_THE_SUBJECT = re.compile(
    r"\bi(?:'d|'ll|'m)?\s+(?:\w+\s+){0,3}$",
    re.IGNORECASE,
)

#: How much has to follow before an announcement is a clause rather than a
#: reply. Five words is one more clause, which is the smallest thing that can
#: carry content.
_ENOUGH_TO_BE_MORE_THAN_AN_ANNOUNCEMENT = 5

#: And how it refers to its own attempts, which a reader was never shown.
_ABOUT_ITS_OWN_ATTEMPTS = re.compile(
    r"\b(?:previous|earlier|last|first)\s+(?:draft|attempt|response|reply|answer)\b"
    r"|\bdraft\s+(?:failed|was\s+rejected|did\s+not)\b"
    r"|\b(?:the\s+)?contract\s+(?:says|requires|demands)\b"
    r"|\bwe\s+(?:need|must)\s+(?:to\s+)?(?:answer|return|produce|give|reply)\b",
    re.IGNORECASE,
)


def talks_about_the_asking(said: str) -> bool:
    """Whether this is commentary about answering rather than an answer.

    A model that has been handed a scaffold and a failed draft will sometimes
    write about the job instead of doing it: what the user asked, why the last
    attempt was rejected, what the contract requires. Every word of it is
    fluent, on topic and well formed, so nothing that measures shape catches
    it — and it arrives where an answer should be.

    LIVE 2026-08-27, in full, to "What is 17 times 23?":

        We need answer user's current message: "What is 17 times 23?" Need
        direct arithmetic. Previous draft failed for missing numeric answer.
        We must return only requested user-facing content? ...

    The test is grammatical rather than topical. A reply speaks TO a person:
    there is no third party in it called "the user", and nothing in an answer
    needs to report what the user asked, because the user knows. The same goes
    for its own earlier attempts, which the reader was never shown.
    """
    body = str(said or "")
    if not body.strip():
        return False
    if _ABOUT_THE_ASKING.search(body) or _ABOUT_ITS_OWN_ATTEMPTS.search(body):
        return True
    for found in _DOING_THE_JOB.finditer(body):
        if not _SHE_IS_THE_SUBJECT.search(body[: found.start()]):
            return True
        rest = body[found.end() :]
        if len(rest.split()) < _ENOUGH_TO_BE_MORE_THAN_AN_ANNOUNCEMENT:
            # She said she would answer, and stopped. That is the announcement
            # standing where the answer goes.
            return True
    return False
