"""Whether she is where she should be, answered by looking rather than asking.

Every check for this asked something other than the screen: is the right
application frontmost, what window was the reading scoped to, what address
does the browser report. All of that can be perfectly true while she types
into the wrong thing — a reading scoped to the right application contains
whatever that application is showing, and after a stray click that is a
different page. Live, that was thirty-five moves of a game played into a chat
window with six guards passing.

She can see the screen. The thing she is acting in is the thing whose places
she has been acting on, and she is holding those places.
"""

from __future__ import annotations

from core.perception.the_lattice_she_holds import TheLatticeSheHolds
from core.perception.where_am_i import where_am_i
from core.perception.where_it_responds import what_is_there


def _read(text: str):
    return what_is_there({"ok": True, "text": text, "layout": [], "bounds": []}, None)


def _held() -> TheLatticeSheHolds:
    lattice = TheLatticeSheHolds()
    places = [(x, y) for y in (20, 40, 60, 80) for x in (20, 40, 60, 80)]
    # Offered twice: a grid is built from places that have stopped changing.
    lattice.built_from(places, acts=8)
    lattice.built_from(places, acts=8)
    return lattice


def _glance(lattice: TheLatticeSheHolds, filled: int = 16):
    places = [
        (0.2 + 0.2 * row, 0.2 + 0.2 * column, "2")
        for row in range(4)
        for column in range(4)
    ][:filled]
    return lattice.fit(places)


def test_the_thing_she_has_been_acting_in_is_recognised():
    here = where_am_i(_glance(_held()), lattice=_held())
    assert here.the_thing_is_here
    assert "4 by 4" in here.because


def test_a_sparse_glance_of_it_is_still_it():
    """Places are there when they are empty, which is what holding one is for."""
    lattice = _held()
    assert where_am_i(_glance(lattice, filled=3), lattice=lattice).the_thing_is_here


def test_another_window_is_not_it_and_she_says_what_it_looks_like():
    here = where_am_i(
        _read("Claude\nHow can I help you today?\nSend a message\nnew chat"),
        lattice=_held(),
    )
    assert not here.the_thing_is_here
    assert "does not look like where I should be" in here.said()
    assert "looking at" in here.said()


def test_a_blank_screen_is_not_it():
    here = where_am_i(_read(""), lattice=_held())
    assert not here.the_thing_is_here
    assert "cannot read anything" in here.because


def test_before_she_has_acted_anywhere_she_does_not_claim_to_be_lost():
    """A request names where she is going, not what is in front of her.

    "Get to a 256 tile" is about a tile that does not exist yet, so a check
    that refused every screen not already showing the goal would refuse the
    first keystroke of every task.
    """
    here = where_am_i(_read("Score 0\n2 . . .\n. . . 2"), asked_for="get to a 256 tile")
    assert here.the_thing_is_here


def test_seeing_what_she_was_told_to_look_for_is_worth_saying():
    here = where_am_i(
        _read("2048\nJoin the numbers\n2 4 8 2"), asked_for="play 2048"
    )
    assert here.the_thing_is_here
    assert "2048" in here.because


def test_the_answer_reads_as_something_she_could_say():
    said = where_am_i(_glance(_held()), lattice=_held()).said()
    assert said.startswith("This is the right place")
    assert said.endswith(".")


def test_a_keystroke_needs_something_bound_to_receive_it():
    """The rule was written down and not enforced."""
    from core.skills.screen_pursuit import _bound_to_a_window

    assert not _bound_to_a_window("right", "")
    assert _bound_to_a_window("right", "Google Chrome")
    # Except the one key that declines and commits to nothing.
    assert _bound_to_a_window("escape", "")
