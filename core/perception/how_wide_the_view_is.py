"""How wide her view is, measured by turning all the way round.

She can measure how far the picture slides for a given mouse travel, which
tells her nothing about degrees: a slide is in pixels and a bearing is in
degrees, and nothing in a frame says how many degrees a pixel is worth.

One landmark settles it. Turning steadily, the picture slides by a measured
amount each time, and something named on screen crosses the middle of the
view, goes round the room and crosses it again. Between those two crossings
she has turned once: three hundred and sixty degrees over the slide between
them is how many degrees a pixel is worth, and the frame's own width in
pixels is then the field of view.

The frames themselves cannot answer it. Turning in steps of whole mouse
points does not land back on the starting heading — measured here, 18.09
degrees a step, so a revolution takes 19.9 of them and the view never
repeats. A landmark's crossing is continuous and can be interpolated between
two looks, which is what makes the count exact.

It matters for aiming at something that moves. A thing crossing her view is
caught by heading for where it is going, and the lead is an angle: how far
round the room it will have gone by the time she arrives. Measured 2026-09-26
without this, leading by a drift in pixels aimed past the edge of the view
and she lost the thing — chasing fell from 20 of 30 to 11. The lead needs a
field of view, so the field of view is measured.

Nothing here knows the world. The return is recognised by the picture, the
revolution is counted in her own turns, and a world whose walls repeat or
whose view never comes back reports nothing rather than a guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from core.perception.how_the_view_moves import grey, how_it_moved

logger = logging.getLogger("Aura.Optics")

#: A full turn, in degrees. The one constant here, and it is geometry.
_ALL_THE_WAY_ROUND = 360.0

#: How alike the first view and a later one must be, left where they are, for
#: her to call it the same heading. A wall of texture correlates at nearly one
#: with itself and at nothing with a different part of itself, so this is a
#: floor against a blank wall rather than a fitted threshold.
_THE_SAME_HEADING = 0.5  # unused by the landmark measurement; kept for the frame test below

#: How many turns past the best match she keeps turning before calling it the
#: peak. The match rises and falls over the few turns around a revolution, so
#: the highest is only known once some have passed.
_PATIENCE = 4


@dataclass(frozen=True)
class HowWideTheViewIs:
    """What one turn of the room measured. ``known`` is false where it told her nothing."""

    degrees_across: float = 0.0
    degrees_per_pixel: float = 0.0
    turns_taken: int = 0
    known: bool = False

    def degrees_for(self, across_px: float) -> float:
        """A slide in small-frame pixels, as degrees round the room."""
        return float(across_px) * self.degrees_per_pixel if self.known else 0.0

    def pixels_for(self, degrees: float) -> float:
        """An angle, as a slide in small-frame pixels."""
        if not self.known or self.degrees_per_pixel <= 0.0:
            return 0.0
        return float(degrees) / self.degrees_per_pixel


def _named_on_screen(layout: object) -> dict[str, float]:
    """What is named on screen and where across it sits, as a fraction of the view."""
    seen: dict[str, float] = {}
    for item in layout or ():
        if not isinstance(item, dict):
            continue
        said = " ".join(str(item.get("text") or "").split()).lower()
        across = item.get("center_x")
        if not said or not isinstance(across, (int, float)):
            continue
        # The first mention wins: a thing is named once, and the lines under it
        # ("Press E to use the door") name it again from the bottom of the view.
        seen.setdefault(said, float(across))
    return seen


def _a_landmark(layout: object) -> str:
    """The thing to watch round the room: whatever is named nearest the middle."""
    named = _named_on_screen(layout)
    if not named:
        return ""
    return min(named, key=lambda said: abs(named[said] - 0.5))


async def _the_view_back_where_it_was(
    move_view: Callable[[int], Awaitable[None]],
    settle_view: Callable[[], tuple[object, object]],
    turned_through: int,
) -> None:
    """Turn back by everything she turned through, so the next thing starts here.

    The same courtesy the body learner already pays the mouse probe. Without
    it a measurement is a side effect on every trip that follows.
    """
    if not turned_through:
        return
    await move_view(-int(turned_through))
    settle_view()


async def turn_all_the_way_round(
    world: object,
    body: object,
    *,
    move_view: Callable[[int], Awaitable[None]],
    settle_view: Callable[[], tuple[object, object]],
    most_turns: int = 240,
) -> HowWideTheViewIs:
    """Turn steadily until a landmark has been round the room, and measure the slide.

    The travel is scaled so a turn slides about a sixth of a view: small
    enough that a landmark is seen crossing the middle rather than jumping
    over it, large enough that a room is covered in tens of turns.

    ``most_turns`` bounds a world with nothing named in it, a camera that does
    not turn, or a mouse that does nothing. Such a world reports
    ``known=False``, and every caller then behaves as it did before any of
    this existed.
    """
    first_frame, first_layout = world.look()
    wide = float(grey(first_frame).shape[1])
    travel = body.turn_for(wide / 6.0)
    if not travel:
        logger.info("the view's width was not measured: nothing the mouse does turns the camera")
        return HowWideTheViewIs()
    # Everything she turns through, so the view can be put back where she
    # found it. A measurement that leaves her facing somewhere else is a
    # measurement that costs every trip after it: measured 2026-09-27, the
    # sweep left her looking away from the two things an "ask which one" world
    # puts in front of her, and asking fell from 30 of 30 to 36 of 60.
    turned_through = 0
    landmark = _a_landmark(first_layout)
    if not landmark:
        # Facing a bare wall. She looks round for something to follow, which
        # is what anyone does before they can measure a turn — and where a
        # whole sweep of the room shows nothing named, there is nothing here
        # to measure by.
        for _ in range(int(most_turns)):
            await move_view(travel)
            turned_through += travel
            _frame, layout = settle_view()
            landmark = _a_landmark(layout)
            if landmark:
                break
        if not landmark:
            await _the_view_back_where_it_was(move_view, settle_view, turned_through)
            logger.info(
                "the view's width was not measured: nothing named came into view all the way round"
            )
            return HowWideTheViewIs()
        logger.info("Following the %s round the room to see how wide the view is.", landmark)
    slid = 0.0
    was: tuple[float, float] | None = None
    crossings: list[float] = []
    for turn in range(1, int(most_turns) + 1):
        before, _ = world.look()
        await move_view(travel)
        turned_through += travel
        after, layout = settle_view()
        across, _down, _grew, _sure = how_it_moved(grey(before), grey(after))
        slid += abs(across)
        here = _named_on_screen(layout).get(landmark)
        if here is None:
            was = None
            continue
        if was is not None and (was[1] - 0.5) * (here - 0.5) < 0.0:
            # It crossed the middle between these two looks. Where exactly,
            # by the share of the gap it had left to go — the crossing is what
            # is being counted, not the look that happened to notice it.
            share = abs(was[1] - 0.5) / max(1e-9, abs(here - was[1]))
            crossings.append(was[0] + share * (slid - was[0]))
            if len(crossings) >= 2:
                break
        was = (slid, here)
    await _the_view_back_where_it_was(move_view, settle_view, turned_through)
    if len(crossings) < 2:
        logger.info(
            "the view's width was not measured: %s crossed the middle of the view %d time(s) "
            "in %d turns",
            landmark,
            len(crossings),
            most_turns,
        )
        return HowWideTheViewIs()
    once_round = abs(crossings[1] - crossings[0])
    if once_round <= wide:
        logger.info(
            "the view's width was not measured: two crossings %0.f pixels apart is less than "
            "a view, so that was not a turn of the room",
            once_round,
        )
        return HowWideTheViewIs()
    degrees_per_pixel = _ALL_THE_WAY_ROUND / once_round
    measured = HowWideTheViewIs(
        degrees_across=wide * degrees_per_pixel,
        degrees_per_pixel=degrees_per_pixel,
        turns_taken=turn,
        known=True,
    )
    logger.info(
        "The view is about %.0f degrees across: the %s came round to the middle of it after "
        "%.0f pixels of sliding, over %d turns of the mouse.",
        measured.degrees_across,
        landmark,
        once_round,
        turn,
    )
    return measured
