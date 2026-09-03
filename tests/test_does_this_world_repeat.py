"""Whether this is a world worth memorising, or one that must be played.

Ghosts 'n Goblins puts the same enemy in the same place every time, so the way
through is to remember it. The same effort spent on 2048 buys nothing, because
the board is dealt fresh. Both efforts are right and each is a waste in the
other's world.
"""

from __future__ import annotations

import random

from core.cognition.does_this_world_repeat import DoesItRepeat


def test_a_fixed_world_is_worth_memorising() -> None:
    fixed = DoesItRepeat()
    for _ in range(10):
        for place in ("the ledge", "the bridge", "the pit"):
            fixed.she_saw(place, "jump", f"{place} cleared")
    assert fixed.how_much_it_repeats() == 1.0
    assert fixed.worth_remembering_places()
    assert "worth learning the places" in fixed.what_to_spend_it_on()


def test_a_world_dealt_fresh_is_not() -> None:
    """The same effort, and it buys nothing."""
    roll = random.Random(4)
    shuffled = DoesItRepeat()
    for _ in range(40):
        shuffled.she_saw("a board", "left", roll.randrange(1000))
    assert shuffled.how_much_it_repeats() < 0.1
    assert not shuffled.worth_remembering_places()
    assert "dealt fresh" in shuffled.what_to_spend_it_on()


def test_a_question_not_yet_asked_is_answered_as_such() -> None:
    once = DoesItRepeat()
    for at in range(20):
        once.she_saw(f"place {at}", "go", "somewhere")
    assert once.tried_twice == 0
    assert once.how_much_it_repeats() == 0.5
    assert not once.worth_remembering_places()
    assert "nothing done twice" in once.what_to_spend_it_on()


def test_most_worlds_are_partly_both_and_it_says_how_much() -> None:
    """A fixed layout with wandering things in it — a number is more use than
    a verdict."""
    roll = random.Random(9)
    mixed = DoesItRepeat()
    for _ in range(30):
        mixed.she_saw("the corridor", "walk", "through")          # fixed
        mixed.she_saw("the yard", "walk", roll.randrange(4))      # wanders
    got = mixed.how_much_it_repeats()
    assert 0.3 < got < 0.9, got


def test_a_pair_done_thirty_times_counts_for_more_than_one_done_twice() -> None:
    weighted = DoesItRepeat()
    for _ in range(30):
        weighted.she_saw("the reliable one", "act", "same")
    weighted.she_saw("the odd one", "act", "this")
    weighted.she_saw("the odd one", "act", "that")
    assert weighted.how_much_it_repeats() > 0.9


def test_what_it_learned_survives_the_process() -> None:
    fixed = DoesItRepeat()
    for _ in range(6):
        fixed.she_saw("the ledge", "jump", "cleared")
    again = DoesItRepeat.from_memory(fixed.as_memory())
    assert again.worth_remembering_places()
    assert DoesItRepeat.from_memory("no").tried_twice == 0
