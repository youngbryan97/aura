"""Making the world remember, so she does not have to.

Everybody who plays Minecraft learns it the same way: they walk off in a
straight line, turn round, and cannot get home. What they do about it is not
to remember harder — they put a torch down every fifty blocks, and after that
the way back is not in their head at all.
"""

from __future__ import annotations

from core.cognition.marks_she_leaves_behind import MarksOnTheGround, worth_marking


def _walked() -> MarksOnTheGround:
    marks = MarksOnTheGround()
    for place, why in (
        ("the door", "home"),
        ("the stream", ""),
        ("the ravine", "iron down there"),
        ("the ridge", ""),
    ):
        marks.she_marked(place, saying=why)
    marks.come_back_to("the door")
    return marks


def test_a_place_is_recognised_rather_than_recalled() -> None:
    marks = _walked()
    assert marks.has_been_here("the ravine")
    assert not marks.has_been_here("somewhere new")
    assert marks.what_she_said_here("the ravine") == "iron down there"


def test_the_way_back_is_the_way_it_was_laid() -> None:
    """A shorter way may exist and she has not walked it."""
    marks = _walked()
    assert marks.the_way_back() == ("the ridge", "the ravine", "the stream", "the door")
    assert marks.the_way_back(to="the stream") == ("the ridge", "the ravine", "the stream")


def test_pacing_does_not_make_a_trail() -> None:
    marks = MarksOnTheGround()
    for _ in range(10):
        marks.she_marked("the same spot")
    assert len(marks.trail) == 1


def test_she_says_where_failure_will_put_her() -> None:
    """So the ground she loses is ground she chose to risk."""
    marks = _walked()
    assert marks.starts_again_at == "the door", "handed back as she gave it"
    assert marks.how_far_from_safety(now="the ridge") == 3
    assert marks.how_far_from_safety(now="the door") == 0


def test_a_way_back_to_somewhere_she_has_never_been_is_no_way_back() -> None:
    assert _walked().the_way_back(to="the moon") == ()


def test_it_says_which_places_are_unmarked() -> None:
    marks = _walked()
    assert worth_marking(["the stream", "a new cave"], marks) == ("a new cave",)


def test_what_she_left_behind_survives_the_process() -> None:
    marks = _walked()
    again = MarksOnTheGround.from_memory(marks.as_memory())
    assert again.the_way_back() == marks.the_way_back()
    assert again.what_she_said_here("the ravine") == "iron down there"
    assert MarksOnTheGround.from_memory(7).trail == []
