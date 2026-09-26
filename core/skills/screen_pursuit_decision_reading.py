"""What the decision reads before it decides.

Five helpers that came out of decide_the_next_move whole and are called
only by it. They took the module past the 2,000-line ceiling again when
the annotation sweep gave them signatures, so they move where the
branches already went.


Lifted whole out of `screen_pursuit_decision`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


async def _decide_the_next_move_how_long_whole(
    anchor: Any,
    costs: Any,
    responds: Any,
    target_app: Any,
) -> tuple[list[Any], Any]:
    from .screen_pursuit import _whats_on_top
    # How long a whole move takes when she does not stop to put it into
    # words. Measured between the starts of two cycles, so it is the
    # reading, the deciding and the act — which is what a pass is being
    # weighed against.
    _began_deciding = time.monotonic()
    if costs["at"] > 0.0:
        costs["cycle_s"] += _began_deciding - costs["at"]
        costs["cycles"] += 1.0
    if costs["at"] > 0.0 and costs.get("was_quiet"):
        costs["quiet_s"] += _began_deciding - costs["at"]
        costs["quiet"] += 1.0
    costs["at"] = _began_deciding
    costs["was_quiet"] = 0.0
    # The moves really on offer, named before anything can ask for them.
    #
    # A caller with its own policy skips the whole deliberation, and the
    # part that decides how far to commit reads this — so hoisting one
    # line out of the branch that used to guard it turned a policy-driven
    # run into UnboundLocalError on its first move.
    available: list[Any] = []
    # Something in front of her work is cleared before anything else.
    #
    # Not once it has cost her four moves discovering that nothing
    # answers: a dialog that owns the keyboard makes every reading and
    # every keystroke of this cycle meaningless, and a person closes it
    # and carries on rather than playing on underneath it.
    # Over the part of the screen she is using, not merely above her
    # window. Before the answering part is worked out this is None and
    # anything on top counts, which is the honest reading while she does
    # not yet know where the task lives.
    in_front = await _whats_on_top(
        target_app or anchor["app"], over=responds["state"].band()
    )
    return available, in_front

async def _decide_the_next_move_what_she_looking(
    anchor: Any,
    drawn: Any,
    narrate: Any,
    observation: dict[str, Any],
    responds: Any,
    target_app: Any,
) -> tuple[Any, bool]:
    from .screen_pursuit_decision import (
        _was_of_that_window,
    )

    from .screen_pursuit import _tell
    # What she is looking at, kept to the part that answers to her.
    #
    # A reading of a screen is everything on it. On the page holding a
    # game that is the board, the score, two advertising rails and a
    # copyright line, so what she recalls about "a situation like this
    # one" is dominated by whichever advertisement was loaded — and two
    # readings of the same board look like different situations because
    # the advertising rotated under her.
    #
    # Which part is the task is not written anywhere on the page, but it
    # is answered by what happens when she acts. Until enough acts have
    # answered it, this is the whole reading, because a guess about where
    # the task is would be worse.
    if not drawn["asked"]:
        drawn["asked"] = True
        from core.perception.what_the_page_says import where_the_drawing_is

        drawn["where"] = await where_the_drawing_is()
        if drawn["where"] and narrate:
            _tell("The page told me where it is drawing — I will look there.")

    # Where the task lives, asked rather than worked out, when it can be.
    #
    # The band is normally learned: act, look, and see which places
    # changed. That is right for anything that cannot be questioned, and
    # slow — it takes many moves, and until it settles she is reading
    # browser tabs and advertising rails as part of the thing. A page that
    # draws its content can say exactly where it draws, and then she knows
    # on the first cycle what would otherwise take twenty.
    #
    # LIVE 2026-08-29: play2048.co draws its board on a canvas. She was
    # reading the whole screen and finding five of the sixteen places on
    # it, and no model ever formed.
    # Not applied twice. When the reading IS the part, every position in
    # it is already a share of that part, and filtering again would cut
    # the thing down by its own outline.
    already = str(observation.get("read_within") or "") == "the part"
    band = None if already else (drawn["where"] or responds["state"].band())
    # Whether this reading is OF the thing rather than of everything.
    #
    # It was read off "is there a band", which stopped meaning that the
    # moment a reading could be scoped by photographing only the part —
    # then the band is None BECAUSE she is already looking at the right
    # place, and every guard that tested for one read it as the opposite.
    # She played on without learning anything from a board she was finally
    # reading properly.
    # And a picture taken through somebody else's window is not a
    # reading of the thing however well it is scoped or cropped.
    # Or when the places of a grid are in the picture itself. Then which part
    # is the thing is seen, not learned, and there is nothing to wait for.
    sees_its_places = bool(observation.get("grids"))
    looking_at_the_thing = (already or band is not None or sees_its_places) and _was_of_that_window(
        observation, target_app or anchor["app"]
    )
    return band, looking_at_the_thing

async def _decide_the_next_move_seen(
    band: Any,
    coming: Any,
    in_the_way: Any,
    observation: dict[str, Any],
    responds: Any,
    target_app: Any,
) -> tuple[Any, Any]:
    from .screen_pursuit_decision import (
        _bring_the_thing_back_to_the_front,
        _put_her_own_window_away,
    )

    from core.cognition.what_nobody_could_show import WhatIsHidden
    from core.perception.where_it_responds import within

    from .screen_pursuit import logger
    seen = within(observation, band, responds["state"])
    # Which places answer to her, not merely their outline. See
    # what_is_there: furniture inside the outline defines columns the
    # board does not have, and a rule about sliding along a row cannot
    # match rows that are not the board's.
    lattice = responds["lattice"]
    # Which of the things that could be wrong is left, from what has been
    # ruled out. Evidence from a failure to produce settles several
    # candidates at once where a sighting settles one.
    why_not = WhatIsHidden(
        candidates=("nothing there", "something over it", "it has ended", "she is early"),
        parties=("the reading", "the band", "the rule"),
    )
    if observation.get("layout"):
        why_not.could_not_produce("the reading", ["nothing there"])
    if not responds["state"].nothing_answers():
        why_not.could_not_produce("the band", ["it has ended"])
    if not responds["lattice"].looks_covered():
        why_not.could_not_produce("the rule", ["something over it"])
    if lattice.looks_covered() and not in_the_way["last"]:
        # Something is over the thing, and reading through it gives an
        # answer that looks well formed and is wrong.
        logger.info(
            "something is sitting over what she is reading (%s)",
            why_not.describe(),
        )
        # Something that has come before, and what it wanted when it did.
        coming.it_came(
            "something over the thing",
            at=time.monotonic(),
            needing=["her own window away", "the thing brought forward"],
        )
        if not observation.get("window_number"):
            await _put_her_own_window_away()
        if target_app:
            await _bring_the_thing_back_to_the_front(target_app)
    return lattice, seen

def _decide_the_next_move_part_4(
    knows: Any,
    lattice: Any,
    move_keys: Any,
    responds: Any,
    *,
    skilled: Any = None,
    world: Any = None,
) -> tuple[Any, Any]:
    from .screen_pursuit_decision import (
        _placed_in,
    )

    from core.perception.the_lattice_she_holds import TheLatticeSheHolds
    from core.perception.what_moves_within_itself import MovesWithinItself
    from core.perception.where_it_responds import the_places_that_answer

    from .screen_pursuit import logger
    if lattice.has_changed():
        # Several readings in a row that will not go into it is the thing
        # having been replaced — a new game, a resized window — rather than
        # a run of poor glances.
        logger.info("what she was looking at has changed shape")
        responds["lattice"] = lattice = TheLatticeSheHolds()
        responds["moving"] = MovesWithinItself()
    answering = the_places_that_answer(responds["state"])
    # And of those, the ones whose contents move about rather than arrive.
    #
    # LIVE 2026-08-31, on the native app with a clean reading of one
    # window: the tiles landed on columns 0, 3, 4 and 6 of a nine-column
    # lattice, because a score and a title answer to her too and their
    # positions were defining columns the board has not got. Five attempts
    # to tell them apart inside one frame all failed, and had to: in one
    # picture a board and a score panel are the same object.
    moving = responds["moving"]
    if moving.settled():
        itself = moving.the_thing_itself()
        if itself:
            # Not intersected with the band. The band is where things
            # happen and a score happens as reliably as a board does, so
            # intersecting throws away places this found and the band
            # missed, for nothing. Measured on a full run of the chain:
            # intersecting gave three places and a two-by-two reading.
            answering = itself
            # And those places, gathered over acts rather than read off
            # one glance, are what the grid is built from.
            was = (responds["lattice"].rows, responds["lattice"].columns)
            # A change is against the shape her counts were read through. A
            # first grid, the shape of every reading so far, once wiped eleven
            # moves of a rule right on all of them (the real app, 26 Sep).
            read_through = tuple(getattr(knows.rules, "read_through", ()) or ())
            if responds["lattice"].built_from(
                itself,
                moving.acts,
                # A grid worked out from what moves cannot be believed
                # until she has moved every way she can.
                tried=responds["state"].tried,
                available=move_keys,
            ):
                now = (responds["lattice"].rows, responds["lattice"].columns)
                if was != now and knows.rules is not None and read_through != now:
                    # Everything counted while the grid was the wrong
                    # shape was counted about a thing that does not exist.
                    knows.rules.learned_through_a_different_reading()
                    # All of it, as when a sitting begins on another grid:
                    # what worked is looked up by the shape of the situation
                    # and what the world does is learned between two
                    # arrangements, and both were read through the grid that
                    # has just turned out to be the wrong shape. Only the rule
                    # was dropped here, so the other two carried the wrong
                    # grid's shapes for the rest of the run.
                    for kept in (skilled, world):
                        forget = getattr(kept, "forget_what_was_read_differently", None)
                        if callable(forget):
                            forget()
        # And which of its places only report, told rather than re-derived.
        #
        # The rule learner works this out for itself, and needs many
        # observations to do it — while every observation it is learning
        # from is scored against a rule that is wrong about a score every
        # single time. So it could not get the evidence until it had the
        # answer. This split is worked out from the same acts and settles
        # far sooner. Measured on a board with a score above it: the right
        # rule scored nought out of five.
        reporting = moving.the_things_that_report()
        if reporting and knows.rules is not None:
            told = knows.rules.told_these_report(
                _placed_in(responds["lattice"], reporting)
            )
            if told:
                logger.info(
                    "%d place(s) only report; they are not hers to move", told
                )
    return answering, lattice

def _decide_the_next_move_laid_out(
    confirmed_here: Any,
    got_to: Any,
    lattice: Any,
    moves: Any,
    pending: Any,
    whole: Any,
) -> Any:
    from .screen_pursuit_decision import (
        _is_a_thing_laid_out,
        _the_thing_she_is_acting_in,
        _worth_holding,
    )

    from .screen_pursuit import logger
    laid_out = _the_thing_she_is_acting_in(
        whole,
        lattice,
        like=_worth_holding(
            pending["arranged"], pending["whole"], pending.setdefault("shapes", {})
        ),
    )
    # A frame to work in while the one she will believe is still settling.
    #
    # The grid she trusts is built from the places that answer to her,
    # gathered over acts, and it is right to make that wait until the set
    # has stopped growing. What it costs is the whole of the beginning:
    # the set takes on about a place an act, so it is rarely still, and
    # until it is there is no frame — so no two readings are comparable,
    # so nothing is learned from the moves she is making meanwhile.
    # Measured on a run through her own reasoning: three of twenty-nine
    # moves reached the rule.
    #
    # The crop is a second source and a better-behaved one. It sees the
    # whole of a thing laid out in one glance rather than accumulating it,
    # so two glances of a steady board offer the same lines and it settles
    # in a couple of cycles. Offered through the same settling rule, and
    # replaced the moment the places that answer settle into something
    # else, because that one is about what she can act on and this one is
    # only about what is drawn.
    if not lattice.held and _is_a_thing_laid_out(laid_out):
        down, across = laid_out.down_at, laid_out.across_at
        if len(down) >= 2 and len(across) >= 2:
            corners = [
                (int(round(x * 100)), int(round(y * 100)))
                for y in down
                for x in across
            ]
            if lattice.built_from(corners, len(moves)):
                logger.info(
                    "holding the frame she can see, for now: %dx%d",
                    lattice.rows, lattice.columns,
                )

    # Is this the thing she was asked to act in.
    #
    # Checked before a key is pressed rather than after, because a
    # keystroke into the wrong window is not something a later cycle can
    # take back.
    # How far the way in got last time, replayed rather than rethought.
    if not confirmed_here["value"] and got_to.frontier:
        logger.info("the way in, as far as it went before: %s", got_to.describe())
    return laid_out

