"""What the world does while she is acting, measured by acting.

A Ghosts 'n Goblins jump cannot be called off, and for that second the knight
cannot answer anything — so the decision to jump is a decision to be unable to
respond, taken while knowing what is coming. Pressing a key and starting a
build that runs five minutes were the same kind of move to her, and the
difference between them is the whole of it.
"""

from __future__ import annotations


def test_the_loop_times_its_own_acts() -> None:
    from core.skills import screen_pursuit

    with open(screen_pursuit.__file__, encoding="utf-8") as handle:
        text = handle.read()
    assert "busy.an_act_took(" in text
    at = text.index("busy.an_act_took(")
    near = text[at : at + 300]
    assert "time.monotonic() - started_acting" in near, "measured, not guessed"


def test_the_clock_starts_before_the_keys_go_out() -> None:
    """The whole sequence is the act, not the last key of it."""
    from core.skills import screen_pursuit

    with open(screen_pursuit.__file__, encoding="utf-8") as handle:
        text = handle.read()
    began = text.index("started_acting = time.monotonic()")
    pressed = text.index("arrived = await press_many(")
    assert began < pressed, "the clock has to start before the acting does"


def test_the_world_is_only_counted_as_moving_where_it_has() -> None:
    """A world that has never done anything on its own exposes her to nothing,
    and saying otherwise would invent a hazard."""
    from core.skills import screen_pursuit

    with open(screen_pursuit.__file__, encoding="utf-8") as handle:
        text = handle.read()
    at = text.index("busy.the_world_moved(")
    near = text[at : at + 260]
    assert "world.acts_with_arrivals" in near
