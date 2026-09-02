"""What somebody meant, learned from what turned out to answer them.

The list of words she had gave seven phrasings of one request back as a
request, a question, and five remarks — and that label goes into what she is
told about the person before she answers.
"""

from __future__ import annotations

from core.cognition.what_kind_of_thing_was_said import WhatSheHasHeard

#: Turns she took, and what turned out to answer each. Nothing here says
#: which words mean what.
A_RECORD = [
    ("open the 2048 app", "did it"),
    ("start the browser for me", "did it"),
    ("get the tile game up on screen", "did it"),
    ("put the file in front of me", "did it"),
    ("fire up the editor", "did it"),
    ("bring the window forward", "did it"),
    ("why did the board stop moving", "said something"),
    ("what does that score mean", "said something"),
    ("how many moves was that", "said something"),
    ("tell me about the last game", "said something"),
    ("explain what happened there", "said something"),
]


def _heard() -> WhatSheHasHeard:
    heard = WhatSheHasHeard()
    for said, doing in A_RECORD:
        heard.it_was_answered_by(said, doing, went_well=True)
    return heard


def test_a_way_of_asking_she_has_not_heard_still_lands() -> None:
    """Not one of these is in the record, and the words that carry them are."""
    heard = _heard()
    for said in (
        "put 2048 in front of me",
        "get the editor up",
        "bring the tile game forward",
    ):
        got = heard.what_kind(said)
        assert got.kind == "did it", f"{said!r} -> {got.describe()}"


def test_asking_and_telling_come_apart() -> None:
    heard = _heard()
    assert heard.what_kind("what does the last score mean").kind == "said something"
    assert heard.what_kind("open the editor for me").kind == "did it"


def test_with_nothing_heard_it_says_so_rather_than_guessing() -> None:
    """A confident wrong label reaches her reasoning, and is worse there than
    none at all."""
    fresh = WhatSheHasHeard()
    got = fresh.what_kind("open the 2048 app")
    assert not got.worked_out
    assert got.kind == ""
    assert "not worked out" in got.describe()


def test_words_she_has_never_heard_say_nothing() -> None:
    """Rather than counting as evidence for the commonest answer, which is how
    a list of words gets rebuilt by accident."""
    heard = _heard()
    got = heard.what_kind("zwoosh flarn quibbet")
    assert not got.worked_out
    assert heard.what_she_has_heard_of("zwoosh flarn quibbet") == 0


def test_a_response_that_did_not_work_teaches_nothing_about_the_person() -> None:
    """It says something about her, and this is not a record of her."""
    heard = WhatSheHasHeard()
    heard.it_was_answered_by("open the app", "said something", went_well=False)
    assert heard.turns == 0
    assert not heard.what_kind("open the app").worked_out


def test_what_she_has_heard_survives_the_process() -> None:
    heard = _heard()
    again = WhatSheHasHeard.from_memory(heard.as_memory())
    assert again.turns == heard.turns
    assert again.kinds_she_knows() == heard.kinds_she_knows()
    assert again.what_kind("put 2048 in front of me").kind == "did it"
    assert WhatSheHasHeard.from_memory("not a memory").turns == 0
