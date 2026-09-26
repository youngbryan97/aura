"""Practising in a world on her own: set a task, try it, see whether the world answered, keep it.

This is the shape of SIMA 2's self-improvement loop (arXiv 2512.04797, §4.5)
built from parts that already stand on their own: the task comes from
`setting_herself_a_task`, the attempt is a trip in `in_a_world_through_a_camera`,
the grade is what the world did afterwards, and the episode goes into
`what_she_tried`. SIMA 2 grades with a second Gemini model scoring the video;
here the grade is the screen's own answer, which no weighting of hers can
move and which a person can check by looking.

What improves across attempts is what she knows about this world: which
things she can reach and use, and so what is worth trying next. Training a
policy on the kept episodes is a later step, and the episodes are kept in the
form it will need.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from core.agency.setting_herself_a_task import what_to_practise
from core.agency.what_hands_do import Chunk, Slot
from core.agency.what_she_tried import HERSELF, Episode, how_it_goes, keep
from core.perception.how_the_view_moves import grey
from core.skills.in_a_world_through_a_camera import (
    ACameraWorld,
    NotInFront,
    go_to,
    learn_the_body,
    look_settled,
)

__all__ = ["practise"]

#: The one kind of task the setter proposes so far.
GOING = "going to a thing"


async def practise(
    world: ACameraWorld,
    world_name: str,
    *,
    keys: Sequence[str],
    slot_s: float,
    attempts: int,
    most_chunks: int,
    title: str = "",
    tell: Callable[[str], None] | None = None,
    keeping: Callable[..., Any] = keep,
) -> list[Episode]:
    """Up to ``attempts`` tasks she sets herself here, each tried, graded and kept."""
    try:
        body = await learn_the_body(world, keys=keys, slot_s=slot_s)
    except NotInFront as why:
        if tell is not None:
            tell(f"I could not practise here: {why}")
        return []
    lived: list[Episode] = []
    for _ in range(max(0, attempts)):
        frame, layout = look_settled(world)
        record = dict(how_it_goes(world_name))
        for episode in lived:
            tried, worked = record.get(episode.task, (0, 0))
            record[episode.task] = (tried + 1, worked + (1 if episode.succeeded else 0))
        task = what_to_practise(layout, record, title=title)
        if not task:
            # Nothing in view is worth trying: turn half a view and look again.
            travel = body.turn_for(0.5 * grey(frame).shape[1]) if frame is not None else 0
            if not travel:
                break
            await world.play(Chunk((Slot(moved=(travel, 0)),), slot_s))
            continue
        if tell is not None:
            tell(f"I will try this next: {task}")
        named = task.removeprefix("go to the ")
        try:
            trip = await go_to(world, named, body, slot_s=slot_s, most_chunks=most_chunks, tell=tell)
        except NotInFront as why:
            if tell is not None:
                tell(f"I stopped practising: {why}")
            break
        heard = trip.done and bool(trip.answered) and "nothing on screen answered" not in trip.answered
        episode = Episode(
            world=world_name, task=task, set_by=HERSELF, succeeded=heard,
            ended=trip.ended, answered=trip.answered, played=list(trip.played), skill=GOING,
        )
        await asyncio.to_thread(keeping, episode, list(trip.looks))
        lived.append(episode)
    return lived
