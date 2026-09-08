"""A word that cannot be a name does not make the reply a confabulation.

The confabulation gate catches a real failure: a reply that speaks about a
person nobody introduced. It finds candidates by capturing `[A-Z][a-z]{2,}` in
a relational frame — "X told me", "X and I", "my friend X" — and then asks
whether X is grounded in what was said.

An ordinary sentence opening with a closed-class word lands in that frame.
"Nobody asked me anything" invents a Nobody, fails the grounding check because
no such person was introduced, and the whole reply is thrown out over it. The
answer is not repaired; there is nothing to repair.

LIVE, 2026-09-07: asked "What did I just ask you?", the draft was rejected
with `ungrounded_person_narrative` and the turn went round again. The pronoun
case — "You asked me..." — survived only because the question happened to
contain the word "you", which is grounding by luck.

The stoplist beside this one records words that have been seen going wrong.
This is a fact about English instead: a pronoun, a determiner, an auxiliary or
an indefinite is not somebody's name in any sentence.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    _has_ungrounded_person_narrative as reads_as_confabulation,
)


@pytest.mark.parametrize(
    "reply",
    [
        "Nobody asked me anything.",
        "You asked me what I am made of.",
        "Someone told me the deploy had failed.",
        "Everyone always warns me about that one.",
        "Nothing told me otherwise.",
        "Something asked me to re-read the file first.",
        "Anyone could have told me that.",
        "That usually leaves me with two options to weigh up.",
        "Neither told me which branch to take.",
        "Others told me the same thing.",
    ],
)
def test_a_closed_class_word_is_not_a_person(reply: str) -> None:
    assert not reads_as_confabulation("What did I just ask you?", reply)


@pytest.mark.parametrize(
    "reply",
    [
        "Brenner told me the plan.",
        "Peter and I go way back.",
        "My friend Marcus warned me about it.",
        "Dana usually had the good sense to stay away from me.",
        "I worked with Alina on the first pass.",
    ],
)
def test_a_person_nobody_introduced_is_still_caught(reply: str) -> None:
    """The gate exists for exactly this and must keep doing it."""

    assert reads_as_confabulation("hi", reply)


def test_a_person_the_user_named_is_grounded() -> None:
    assert not reads_as_confabulation("Tell me about Dana", "Dana told me the plan.")


def test_the_closed_class_and_the_stoplist_stay_different_things() -> None:
    """One is a fact about English; the other is a record of what went wrong.

    Merging them would lose the reason either exists, and the stoplist is the
    one that grows.
    """

    from core.conversation.response_reliability import (
        _NEVER_A_PERSON_NAME,
        _PERSON_NAME_STOPLIST,
    )

    assert _NEVER_A_PERSON_NAME
    assert _PERSON_NAME_STOPLIST
    # A weekday or a browser is an open-class word that happens not to be a
    # name here. It does not belong in the closed class.
    for word in ("friday", "python", "safari", "github"):
        assert word not in _NEVER_A_PERSON_NAME
