"""A reply about her own workings is the answer when that is what was asked.

The off-topic block exists to catch a reply about the runtime's own operation
in place of an answer to the question. It is given the reply alone and cannot
know what was asked. Measured live 2026-08-26: "what are you actually able to
do on this machine that you could not do a month ago" — five hundred and
ninety characters written about exactly that, called a foreign topic, and the
person got "I couldn't get a clear enough answer together."
"""

from __future__ import annotations

import pytest

from interface.routes.chat import _asked_about_her_own_workings


@pytest.mark.parametrize(
    "asked",
    [
        "What are you actually able to do on this machine that you could not do a month ago?",
        "what can you do",
        "how does your memory work",
        "what do you remember about yesterday",
        "why did you change your approach",
        "are you able to read a screen",
    ],
)
def test_a_question_about_her_is_recognised(asked):
    assert _asked_about_her_own_workings(asked)


@pytest.mark.parametrize(
    "asked",
    [
        "what is the capital of France",
        "write me a poem about the sea",
        "summarise this article for me",
        "make a folder on my desktop called notes",
    ],
)
def test_a_question_about_anything_else_is_not(asked):
    assert not _asked_about_her_own_workings(asked)


def test_the_block_reads_the_question_before_it_reads_the_reply():
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat)
    where = source.index("_asked_about_her_own_workings(user_message)")
    drift = source.index("drift = assess_subject_drift(reply)")
    assert where < drift


def test_a_real_answer_about_her_own_workings_survives_the_drift_check():
    """Belt and braces: the honest answer passes on its own merits too."""
    from core.conversation.reply_subject import assess_subject_drift

    said = (
        "A month ago I could read a screen and press keys at it. I could not tell which "
        "part of what I was looking at answered to me. Now the reading keeps its "
        "arrangement, so I can hold a plan about a corner and check it."
    )
    assert not assess_subject_drift(said).drifted
