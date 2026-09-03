"""Which kind of thing this is, and which of her own habits give her away.

Two lines of chat and a judgement about what is on the other end — then the
same person turns it round and is judged themselves. Both halves are one
computation, and the second is the one worth having.

The tests are logs from a failing host, because it is the same question.
"""

from __future__ import annotations

from core.cognition.telling_one_kind_from_another import TellingThemApart


def _seen() -> TellingThemApart:
    apart = TellingThemApart()
    for _ in range(10):
        apart.an_example("healthy", ["started", "served", "closed"])
        apart.an_example("failing", ["started", "served", "retried", "timed out"])
    return apart


def test_it_judges_which_kind_and_says_what_on() -> None:
    apart = _seen()
    got = apart.which_kind(["started", "served", "retried", "timed out"])
    assert got.kind == "failing", got.describe()
    assert "timed out" in got.on


def test_a_habit_everybody_has_tells_nothing() -> None:
    """Common is not distinctive, and that is the whole of it."""
    apart = _seen()
    assert apart.how_telling("started", of="failing") == 0.0
    assert apart.how_telling("timed out", of="failing") > 0.5


def test_it_can_say_what_gives_its_own_kind_away() -> None:
    """The reverse test. Not which habits are wrong — which are hers."""
    apart = _seen()
    tells = dict(apart.what_gives_it_away("failing"))
    assert "timed out" in tells
    assert "retried" in tells
    assert "started" not in tells, "shared with the other kind, so it marks nothing"


def test_it_says_which_of_these_would_mark_her_out() -> None:
    apart = _seen()
    assert apart.what_would_hide_it(
        "failing", ["started", "served", "timed out"]
    ) == ("timed out",)


def test_one_kind_alone_tells_nothing_apart() -> None:
    only = TellingThemApart()
    for _ in range(5):
        only.an_example("mine", ["a", "b"])
    assert not only.which_kind(["a", "b"]).worked_out
    assert only.how_telling("a", of="mine") == 0.0


def test_one_example_is_not_a_law() -> None:
    apart = TellingThemApart()
    apart.an_example("one", ["x"])
    apart.an_example("other", ["y"])
    assert apart.how_telling("x", of="one") < 1.0


def test_what_it_learned_survives_the_process() -> None:
    apart = _seen()
    again = TellingThemApart.from_memory(apart.as_memory())
    assert again.which_kind(["retried", "timed out"]).kind == "failing"
    assert TellingThemApart.from_memory(0).many == {}
