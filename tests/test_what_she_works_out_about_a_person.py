"""What she works out about somebody from watching, without being told.

A question is not free. Somebody asking about a thing is telling her they do
not have it, and somebody asking the same thing three times is telling her the
first two answers did not land.
"""

from __future__ import annotations

from core.consciousness import theory_of_mind as tom


def test_a_question_is_recorded_about_the_asker_not_only_answered() -> None:
    tom._ASKING.asked.clear()
    tom._ASKING.turns = 0
    tom.TheoryOfMindEngine._classify_turn_intent("where is the lattice built?")
    got = tom.what_she_knows_about("the person")
    assert "lattice" in got["does_not_have"]


def test_asking_twice_shows_up_as_being_stuck() -> None:
    tom._ASKING.asked.clear()
    tom._ASKING.turns = 0
    for _ in range(2):
        tom.they_asked("the person", ["why", "does", "it", "fail"])
    tom.they_asked("the person", ["something", "else"])
    assert "fail" in tom.what_she_knows_about("the person")["stuck_on"]


def test_she_can_say_what_somebody_will_probably_do() -> None:
    for _ in range(5):
        tom.they_did("the person", facing="a red build", act="ask about it")
    got = tom.what_she_knows_about("the person", facing="a red build")
    assert got["will_probably"] == "ask about it"
    assert got["how_likely"] > 0.5


def test_she_can_say_what_gives_her_own_kind_away() -> None:
    """The reverse test: not which habits are wrong, which are distinctive."""
    for _ in range(6):
        tom.an_example_of("hers", ["moreover", "delve", "the point is"])
        tom.an_example_of("theirs", ["the point is", "anyway"])
    tells = tom.what_gives_her_away(["moreover", "the point is", "anyway"])
    assert "moreover" in tells
    assert "the point is" not in tells, "shared, so it marks nothing"


def test_none_of_it_needed_their_cooperation() -> None:
    from core.consciousness import theory_of_mind

    with open(theory_of_mind.__file__, encoding="utf-8") as handle:
        text = handle.read()
    for one in ("WhatTheirAskingSays", "WhatTheyTendToDo", "TellingThemApart"):
        assert one in text
