"""Making the move she chose, and saying whether it landed.

The half of a decision that touches the screen. It is here rather than inside
the decision because the two fail differently: a decision that is wrong picks
the wrong control, and an action that is wrong presses nothing and reports
success. Every path ends in a receipt saying which of those happened.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from .screen_pursuit_bearings import (
    _AS_IT_USUALLY_IS,  # noqa: F401
    A_SCREENFUL_AT_LEAST,  # noqa: F401
    ENOUGH_TO_BE_A_THING,  # noqa: F401
    MOST_OF_A_SCREEN,  # noqa: F401
    SCREENFULS_TO_LOOK,  # noqa: F401
    SETTLE_AFTER_SCROLL_S,  # noqa: F401
    SLOW_DOWN,
    _a_screenful,  # noqa: F401
    _a_step_back,  # noqa: F401
    _by_how_much_room,  # noqa: F401
    _moves_that_leave_her_nothing,  # noqa: F401
    _time_left,  # noqa: F401
    _within_a_move,  # noqa: F401
    let_the_voice_catch_up,
    narration_backlog,
)
from .screen_pursuit_looking import (
    _ANSWERING_TOOK,  # noqa: F401
    ASKING_TO_CONFIRM,  # noqa: F401
    LONGER_THAN_USUAL,  # noqa: F401
    OBSERVE_TIMEOUT_S,  # noqa: F401
    _bring_the_thing_back_to_the_front,
    _expected_of,  # noqa: F401
    _how_full,  # noqa: F401
    _how_long_to_wait,
    _move_her_own_surface_aside,  # noqa: F401
    _narrate,  # noqa: F401
    _say_intent,
    _say_it_did_not_land,
    _settled_after,
    _where,  # noqa: F401
    _where_clicked,  # noqa: F401
    _where_it_asks,  # noqa: F401
)
from .screen_pursuit_surface import (
    LABEL_REACH,  # noqa: F401
    _bound_to_a_window,  # noqa: F401
    _looks_like,
    _matches,  # noqa: F401
    _screen_size,  # noqa: F401
    _value_is_on_screen,  # noqa: F401
    labelled_by,  # noqa: F401
    press,
    press_many,
)


async def carry_out_the_move(

    run: SimpleNamespace,
) -> bool:
    # Reached at call time: screen_pursuit imports the decision that
    # imports this one, so the other direction cannot be a module-level
    # import, and a call-time one still sees a test's patch of the
    # original.
    # These were imports inside the function this was lifted from, and
    # they stay inside it. A module-level import binds once; a test that
    # patches one of these on its own module would then never be seen,
    # which is exactly what happened to `learn_about`.
    from core.agency import what_she_is_doing as doing
    from core.perception.the_lattice_she_holds import TheLatticeSheHolds
    from core.perception.what_moves_within_itself import MovesWithinItself
    from core.perception.where_am_i import where_am_i

    from .screen_pursuit import _tell, logger
    # The closure, named. It assigns none of these, so binding them back
    # to locals of the same name is the same program.
    about_to = run.about_to
    anchor = run.anchor
    at_rest = run.at_rest
    busy = run.busy
    expected = run.expected
    follow_on = run.follow_on
    goal = run.goal
    in_flight = run.in_flight
    key = run.key
    laid_out = run.laid_out
    lattice = run.lattice
    made = run.made
    moves = run.moves
    narrate = run.narrate
    pacing = run.pacing
    pending = run.pending
    responds = run.responds
    target_app = run.target_app
    world = run.world

    # Said after the body did it, never before.
    #
    # The line has to correspond to what actually happened. Announcing
    # a decision announces an intention: a keystroke refused for focus,
    # or sent to the wrong window, would have been described as a move
    # she made. What she says she did is now what her body did, in the
    # order it did it.
    # One call for the whole sequence she settled on.
    #
    # Looking and deciding cost about three seconds; the keystroke
    # itself costs a third of one, and spawning the automation is most
    # of that. A run that re-reads the board between every key of a
    # pattern it has already chosen pays the whole cycle per move —
    # measured live, about one move every three seconds, where a
    # person plays several a second.
    # Saying less is about her mouth, not her hands.
    #
    # Choosing it collapsed every committed sequence to a single key
    # for the rest of the run, so one decision about commentary cost
    # her all multi-move play: measured live 2026-08-26, forty-eight
    # cycles that had committed to two to four moves each produced
    # fifty-three moves between them, one screen reading apiece.
    sequence = [key, *follow_on] if follow_on else [key]
    started_acting = time.monotonic()
    # Before the body moves: is this where she should be.
    #
    # Everything that asked this before asked the machine — is the
    # right application frontmost, what window was the reading scoped
    # to, what address does the browser report. All of that can be
    # perfectly true while she types into the wrong thing, because a
    # reading scoped to the right application contains whatever that
    # application is showing, and after a stray click that is a
    # different page.
    #
    # She can see the screen, so she answers it the way anybody does:
    # the thing she is acting in is the thing whose places she has
    # been acting on, and she is holding those places.
    here = where_am_i(laid_out, lattice=lattice, asked_for=goal)
    if not here.the_thing_is_here:
        if narrate:
            _tell(here.said())
        logger.info("not pressing anything: %s", here.said())
        # And not being able to see it is evidence about the frame.
        #
        # Otherwise a frame worked out from a poor first glance locks
        # her out of the thing for the rest of the run: every reading
        # fails to be it, she declines to press, and nothing ever
        # revises the frame that is doing the declining. Several in a
        # row is the thing having changed rather than one bad look,
        # which is the judgement the lattice already makes.
        lattice.would_not_fit += 1
        if lattice.has_changed():
            logger.info("the frame I was holding was not the thing after all")
            responds["lattice"] = TheLatticeSheHolds()
            responds["moving"] = MovesWithinItself()
        if target_app:
            await _bring_the_thing_back_to_the_front(target_app)
        pending["arranged"] = None
        pending["whole"] = None
        return None
    # Intent, then action. Said before the body moves, because that is
    # the order a person doing something narrates it in.
    #
    # Only the first one carries the reason she gave. The rest are
    # the same decision continuing, and repeating its reason under
    # each of them says something false: live, she committed to
    # left-then-right and narrated "Going right — left has worked."
    for position, step in enumerate(sequence):
        reason = (None if pacing["brief"] else made) if position == 0 else None
        # Under a quiet pace only the first of a sequence is spoken.
        # The moves still happen; what she asked for is fewer words.
        aloud = narrate and (position == 0 or not pacing["brief"])
        _say_intent(step, reason, out_loud=aloud, following_on=position > 0)
    if len(sequence) > 1:
        # Only the keys that really landed are spoken for. Focus can
        # move part-way through a batch, and a commentary describing
        # moves the window never received is the disconnect this
        # whole path exists to avoid.
        arrived = await press_many(sequence, expect_app=target_app or anchor["app"])
    else:
        arrived = 1 if await press(key, expect_app=target_app or anchor["app"]) else 0
    for position, step in enumerate(sequence[:arrived]):
        if position == 0:
            about_to["at"] = time.time()
            moves.append(about_to)
        else:
            moves.append(
                {"key": step, "because": "part of the same plan", "at": time.time()}
            )
        doing.a_step_taken()
    for step in sequence[arrived:]:
        # An intention she stated and did not carry out is corrected
        # out loud. The record of what she did is written only from
        # what landed, so the two can never drift.
        _say_it_did_not_land(step, out_loud=narrate)
    if arrived:
        # Let the world answer before anything reads it again.
        #
        # There was no wait here at all. The keystroke returned, the
        # loop came round, and the next reading was of a board
        # mid-slide — or of one the game had not started moving yet.
        # Compared with the reading before it, that says nothing
        # happened.
        #
        # It is not a small error. Of what she had kept about playing
        # this game on 2026-08-30, ninety-eight of a hundred acts were
        # written down as having changed nothing, and the only rule
        # she ever confirmed was "this does not move", ninety-eight
        # times. She was playing correctly and recording the opposite,
        # and every part of her that learns from what happened —
        # which acts do anything, how the thing moves, where it
        # answers her — was being taught from that.
        came_to_rest, _ = await _settled_after(
            pending["watched"] or {},
            target_app or anchor["app"],
            arrived=_looks_like(
                expected["after"], responds["state"], responds["lattice"]
            ),
        )
        # Only a reading it actually took. Where every capture timed
        # out it hands back the one it was given, and that is the
        # picture from BEFORE the act — which as the next cycle's
        # observation would be a move recorded against the board it
        # was made from.
        at_rest["reading"] = (
            came_to_rest if came_to_rest is not pending["watched"] else None
        )
        # How long she could not do anything else for, measured by
        # doing it rather than guessed at.
        busy.an_act_took(key, time.monotonic() - started_acting)
        # Started and not yet seen to land.
        if expected["after"] is not None:
            in_flight.she_started(
                key,
                at=started_acting,
                lands_at=time.monotonic() + _how_long_to_wait(),
                brings=str(expected["took"]),
            )
        busy.the_world_moved(
            time.monotonic() - started_acting,
            times=1 if world.acts_with_arrivals else 0,
        )
    if arrived and pacing["choice"] == SLOW_DOWN:
        await let_the_voice_catch_up(narration_backlog())
    return arrived > 0
