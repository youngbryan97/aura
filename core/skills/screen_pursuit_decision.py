"""One decision, from one reading of the screen.

This was a closure inside `pursue_on_screen`, which is how it came to be
eighteen hundred lines: everything it needed was already in scope, so nothing
ever had to be named. What it needs is named now — handed over at the call —
and the cost of adding one more is visible.

Nothing else changed. The body is the body, and the scope is built at the call
rather than when the loop starts, because three of those values are rebound
after the point where the closure used to be defined.
"""
from __future__ import annotations

from .screen_pursuit_decision_reading import (  # noqa: F401  (re-exported: they were defined here)
    _decide_the_next_move_how_long_whole,
    _decide_the_next_move_laid_out,
    _decide_the_next_move_part_4,
    _decide_the_next_move_seen,
    _decide_the_next_move_what_she_looking,
)
import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the return annotation only; the runtime import
    from core.skills.fluid_executor import Step  # is inside the function

import time
from types import SimpleNamespace
from typing import Any

from core.runtime.errors import record_degradation

from .screen_pursuit_acting import carry_out_the_move
from .screen_pursuit_bearings import (
    _AS_IT_USUALLY_IS,  # noqa: F401
    A_SCREENFUL_AT_LEAST,  # noqa: F401
    ENOUGH_TO_BE_A_THING,  # noqa: F401
    LANGUAGE_EVERY,
    MOST_OF_A_SCREEN,  # noqa: F401
    PRESS_ON,
    SAY_LESS,
    SCREENFULS_TO_LOOK,  # noqa: F401
    SEE_IT_THROUGH,
    SETTLE_AFTER_SCROLL_S,  # noqa: F401
    SLOW_DOWN,
    START_OVER,
    _a_screenful,  # noqa: F401
    _a_step_back,  # noqa: F401
    _by_how_much_room,  # noqa: F401
    _expects,
    _how_much_the_tally_moved,
    _in_the_same_grid,
    _is_a_thing_laid_out,
    _left_her_better_off,
    _moves_she_will_not_make,
    _moves_that_leave_her_nothing,  # noqa: F401
    _say_what_kind_of_problem,
    _say_what_she_worked_out,
    _the_rest_of_the_run,
    _the_same_thing_without,
    _time_left,  # noqa: F401
    _was_of_that_window,
    _what_she_is_not_reading,
    _what_there_is_to_aim_at,
    _within_a_move,  # noqa: F401
    _within_the_run,
    _worth_holding,
    a_run_she_can_carry,
    am_i_there,
    let_the_voice_catch_up,
    narration_backlog,
    pacing_options,
    screen_options,
    ways_out,
)
from .screen_pursuit_looking import (
    _ANSWERING_TOOK,  # noqa: F401
    ASKING_TO_CONFIRM,  # noqa: F401
    LONGER_THAN_USUAL,  # noqa: F401
    OBSERVE_TIMEOUT_S,  # noqa: F401
    _bring_the_thing_back_to_the_front,
    _expected_of,  # noqa: F401
    _her_reasoning,
    _how_full,  # noqa: F401
    _how_it_has_been_going,
    _move_her_own_surface_aside,  # noqa: F401
    _narrate,  # noqa: F401
    _no_more_than_a_fresh_one_is_worth,
    _placed_in,
    _put_her_own_window_away,
    _reasoning_for_a_plan,
    _the_best_reading_available,
    _the_kind_of_world_this_is,
    _the_thing_she_is_acting_in,
    _what_she_could_not_learn_from,
    _where,  # noqa: F401
    _where_clicked,  # noqa: F401
    _where_it_asks,  # noqa: F401
    _why_nothing_answers,
    clear_what_is_in_front,
)
from .screen_pursuit_surface import (
    LABEL_REACH,  # noqa: F401
    PRESSABLE_KEYS,
    _a_pass_in_moves,
    _bound_to_a_window,  # noqa: F401
    _ensure_page,
    _matches,  # noqa: F401
    _screen_size,  # noqa: F401
    _value_is_on_screen,  # noqa: F401
    click_normalized,
    content_text,
    labelled_by,  # noqa: F401
)

#: What an extracted block returns when it fell through to the code after it.



async def _decide_the_next_move_part_6(
    anchor: Any,
    confirmed_here: Any,
    expect_page: Any,
    open_page: Any,
    seen: Any,
    target_app: Any,
    seen_by_window: bool=False,
) -> None:
    confirmed_here["value"] = am_i_there(
        open_page or expect_page, seen, anchor["page"], anchor["app"]
    )
    if confirmed_here["value"]:
        # Arriving at a window is not the same as reading it.
        #
        # This check is about identity — the address, the title, the
        # name of the window — and it is right to be: words on a
        # screen can be borrowed from anyone's window. But the pixels
        # she then reads are whatever is drawn on top, and her own
        # interface is drawn on top of everything by design. LIVE
        # 2026-08-31: Chrome held the address she was sent to, she
        # confirmed she was there, and every reading for eighteen
        # moves was of her own panels — LIVE NEURAL FEED, TELEMETRY,
        # MEMORY, SETTINGS. She pressed keys into herself and her
        # predictions about what would change were correct.
        #
        # So before the first key: her own window goes away and the
        # thing is asked to the front. Asking alone is not enough —
        # hers is drawn above everything and comes straight back —
        # and putting hers away alone leaves the wrong window
        # frontmost. Both, in that order, are what make the pixels
        # agree with the identity.
        #
        # Not where the reading was taken by window number: those pixels
        # are the window's own, whatever is drawn over it, and her keys go
        # to its process. Putting her window away then only hides the
        # conversation from the person watching her play.
        if not seen_by_window:
            await _put_her_own_window_away()
        await _bring_the_thing_back_to_the_front(anchor["app"] or target_app)

def _decide_the_next_move_world_where_she(
    expected: Any,
    far: Any,
    in_flight: Any,
    laid_out: Any,
    observation: dict[str, Any],
    pending: Any,
    previous: Any,
    responds: Any,
    carries: Any=None,
) -> Any:
    from core.perception.how_it_moves import prediction_held

    from .screen_pursuit import logger
    # Was the world where she said it would be? That is what decides
    # how far she goes next time, and it is the only thing that does.
    in_flight.it_landed(previous.chosen.name if previous.chosen else "")
    if expected["took"] >= 1 and expected["after"] is not None and laid_out is not None:
        # Came out, meaning nothing she predicted is missing. Not
        # meaning identical: in a world that deals a tile after every
        # move of hers, the board she predicted is never the board
        # that is there, and asking for identical is asking for a run
        # that can never come out. What matters is whether her acts
        # did what she thought — an arrival she never claimed to know
        # about is the world's business, and she has somewhere else
        # to put those.
        same = prediction_held(expected["after"], laid_out)
        (far.it_was_where_she_said if same else far.it_was_not)(
            expected["took"]
        )
        if not same:
            logger.info("the run did not come out: %s", far.describe())
        if carries is not None and pending["arranged"] is not None and expected.get("confidence") is not None:
            carries.it_predicted(
                distance=expected["took"],
                confidence=expected["confidence"],
                was_right=same,
                would_no_change_have_been_right=prediction_held(pending["arranged"], laid_out),
            )
        # Learning below still needs the delivered count. Consume only the forecast.
        expected["after"] = None
    # What leaning on these acts has come to. The tally she found on
    # the screen herself is the measure where there is one, because
    # nobody has told her what progress is and something on the screen
    # has been keeping score the whole time.
    rose = _how_much_the_tally_moved(
        responds["moving"], pending["watched"], observation
    )
    return rose

def _decide_the_next_move_what_true_time(
    attempt: Any,
    can_do: Any,
    confirmed_here: Any,
    in_the_way: Any,
    opens: Any,
    previous: Any,
    responds: Any,
) -> None:
    from core.cognition.two_ways_out import how_long_it_holds

    from .screen_pursuit import logger
    # And what was true at the time, so an act that does nothing
    # can become an act that needs something.
    opens.she_tried(
        previous.chosen.name,
        holding=[
            one
            for one in (
                "the thing in front" if confirmed_here["value"] else "",
                "nothing over it" if not in_the_way["last"] else "",
                "a grid she trusts" if responds["lattice"].held else "",
            )
            if one
        ],
        it_worked=bool(attempt.verdict.observed_change),
    )
    # Whether she is alive here or merely still going. Two acts
    # that work are two ways out; one is a thing standing until
    # something takes it.
    stands = how_long_it_holds(
        can_do,
        ways_out=lambda one: [
            name for name in one.told if one.does_something(name)
        ]
        if hasattr(one, "told") and hasattr(one, "does_something")
        else [],
        room=lambda one: len(getattr(one, "told", ()) or ()),
    )
    if not stands.alive and stands.ways_out == 1:
        logger.info("one way out here: %s", stands.describe())

def _decide_the_next_move_her_own_move(
    dropped: Any,
    looking_at_the_thing: Any,
    pending: Any,
    previous: Any,
    responds: Any,
) -> None:
    # Her own move, and what it did. Three things she already had and
    # threw away after one glance, which is why she could never try a
    # move without making it.
    # Learned from the part that answers to her, once she knows which
    # part that is.
    #
    # Before the band settles a reading is the whole page — the tabs,
    # the address, the score, a Give Feedback button — and no rule
    # about what her own act moves can match one. Learning from it
    # anyway fills the counters with failures that then take longer to
    # recover from than starting clean. LIVE 2026-08-26: nineteen
    # moves in, every hypothesis discredited, and she was choosing
    # blind on a board she could read perfectly well.
    if (
        pending["arranged"] is not None
        and previous.chosen is not None
        and not looking_at_the_thing
    ):
        dropped["not the thing itself"] += 1
    # And nothing is learned from acts the world was not taking.
    #
    # A move that changed nothing is evidence about that move only
    # while other moves are changing things — then it means "not
    # here", which is worth knowing. When NOTHING she does changes
    # anything, it means the thing has ended, and none of it is
    # evidence about how the thing works.
    #
    # Measured live 2026-09-02, three games back to back on the real
    # app. The first went well: she began already knowing the rule at
    # 71%, looked ahead on 178 of 220 moves, three moves changed
    # nothing, and she reached 512. Then it finished, and she went on
    # pressing keys at a finished board: seventy-eight of eighty-three
    # moves changed nothing, and what she wrote down at the end was
    # "this does not move", right 98% of 62 — over a rule that had
    # been confirmed 236 times. The third game began holding that,
    # looked ahead on nothing at all, and was over in fifteen moves.
    #
    # So one ended game undid everything she knew and cost her the
    # two after it. What she keeps is only as good as her refusing to
    # learn from a world that has stopped answering.
    if (
        pending["arranged"] is not None
        and previous.chosen is not None
        and looking_at_the_thing
        and responds["state"].nothing_answers()
    ):
        dropped["the thing had stopped answering"] += 1

def _decide_the_next_move_part_10(
    dropped: Any,
    expected: Any,
    knows: Any,
    laid_out: Any,
    pending: Any,
    plan: Any,
    previous: Any,
    responds: Any,
    success_when: Any,
) -> Any:
    if expected["took"] > 1:
        # Several acts, one reading. Which of them did what cannot
        # be told from a board seen only at the end, and crediting
        # the first is not weak evidence — it is a claim about an
        # act that did not produce this.
        #
        # LIVE 2026-09-02, the sitting that first went more than
        # one act between looks: a rule carried in at 82% fell to
        # one right out of sixty four, because every pair she
        # learned from named the wrong act.
        dropped["more than one act, one reading"] += 1
    elif getattr(pending["arranged"], "unknown", ()) or getattr(laid_out, "unknown", ()):
        # A place that held something she could not read is not an empty
        # place, and a pair with one in it is not evidence about the world.
        dropped["a place she could not read"] += 1
    elif _in_the_same_grid(
        responds["lattice"], pending["arranged"], laid_out
    ):
        knows.watched(pending["arranged"], previous.chosen.name, laid_out)
    else:
        # Counted, because it used to be silent.
        #
        # A pair thrown away here is a move she made and learned
        # nothing from, and nothing said so. LIVE 2026-08-31:
        # fifty-four moves, one of them watched, and the only
        # trace was a line saying how this moves is not worked out
        # yet — which reads as a hard world rather than as
        # evidence going missing on the way to the learner.
        dropped["a different frame"] += 1
    # And whether it left her better off, against the kind of
    # position it was made from. This is experience turning into
    # skill: the same triple she learns the world's rules from
    # also says which move is worth making again from a shape
    # like that one.
    went_well = _left_her_better_off(
        pending["arranged"],
        laid_out,
        success_when,
        plan["held"].approach if plan["held"] is not None else "",
    )
    return went_well


async def decide_the_next_move(
    observation: dict[str, Any],
    run: SimpleNamespace,
) -> Step | None:
    # These were imports inside the function this was lifted from, and
    # they stay inside it. A module-level import binds once; a test that
    # patches one of these on its own module would then never be seen,
    # which is exactly what happened to `learn_about`.
    import re
    from collections.abc import Sequence
    from dataclasses import replace

    from core.agency import what_she_is_doing as doing
    from core.agency.deliberate_action import confirm, deliberate
    from core.agency.how_good_is_this import terms, worth_comparing
    from core.agency.looking_ahead import (
        at_the_worlds_mercy,
        look_ahead,
        whether_to_take_the_wide_option,
        worth_finding_out,
    )
    from core.agency.standing_strategy import settle_on_an_approach, still_holds
    from core.agency.task_knowledge import learn_about, stuck, work_out_what_it_means
    from core.agency.what_worked_before import WhatWorkedBefore
    from core.agency.worth_thinking_about import worth_a_pass
    from core.cognition.a_shape_that_makes_it_safe import what_makes_it_safe
    from core.cognition.a_window_not_a_maximum import AWindow, which_act_lands_in_it
    from core.cognition.enough_rather_than_most import the_one_most_likely_to_do
    from core.perception.how_it_moves import HowItMoves
    from core.perception.what_the_world_does import WhatTheWorldDoes
    from core.perception.where_it_responds import within
    from core.perception.why_nothing_answers import ELSEWHERE, ENDED, work_out_why
    from core.runtime.what_she_learned import recall
    from core.skills.fluid_executor import Step
    # The closure, named. It assigns none of these, so binding them back
    # to locals of the same name is the same program.
    anchor = run.anchor
    asked = run.asked
    at_rest = run.at_rest
    beats = run.beats
    began = run.began
    began_at = run.began_at
    blocker_attempts = run.blocker_attempts
    busy = run.busy
    can_do = run.can_do
    cannot_explain = run.cannot_explain
    clear_blocker = run.clear_blocker
    coming = run.coming
    confirmed_here = run.confirmed_here
    costs = run.costs
    drawn = run.drawn
    dropped = run.dropped
    ends_at = run.ends_at
    expect_page = run.expect_page
    expected = run.expected
    far = run.far
    foreseen = run.foreseen
    furthest = run.furthest
    goal = run.goal
    got_to = run.got_to
    graph = run.graph
    history = run.history
    in_flight = run.in_flight
    in_the_way = run.in_the_way
    intending = run.intending
    knowledge = run.knowledge
    knows = run.knows
    last_call = run.last_call
    like_it = run.like_it
    lines = run.lines
    lines_held = run.lines_held
    lived = run.lived
    marks = run.marks
    matters = run.matters
    move_keys = run.move_keys
    going = getattr(run, "going", None)
    carries = getattr(run, "carries", None)
    moves = run.moves
    narrate = run.narrate
    needs_person = run.needs_person
    no_move = run.no_move
    not_there = run.not_there
    observe = run.observe
    offered_a_restart = run.offered_a_restart
    open_page = run.open_page
    opens = run.opens
    pacing = run.pacing
    pending = run.pending
    plan = run.plan
    policy = run.policy
    reaches = run.reaches
    region_bottom = run.region_bottom
    region_top = run.region_top
    repeats = run.repeats
    research = run.research
    responds = run.responds
    restarts = run.restarts
    said_it_ended = run.said_it_ended
    seen_through = run.seen_through
    she_keeps = run.she_keeps
    skilled = run.skilled
    spine = run.spine
    stakes = run.stakes
    stretch = run.stretch
    success_when = run.success_when
    target_app = run.target_app
    think = run.think
    trying = run.trying
    undecided = run.undecided
    went = run.went
    world = run.world

    # Reached at call time: the module this came from imports this one,
    # so the other direction cannot be a module-level import, and a
    # call-time one still sees a test's patch of the original.
    from .screen_pursuit import (
        A_LINE_HERE,
        MAX_RELEARNS,
        RECENT_ATTEMPTS,
        _frontmost,
        _tell,
        _whats_on_top,
        logger,
    )

    available, in_front = await _decide_the_next_move_how_long_whole(anchor, costs, responds, target_app)
    if in_front and in_front != in_the_way["last"]:
        in_the_way["last"] = in_front
        if await clear_what_is_in_front(in_front):
            in_the_way["last"] = ""
            if narrate:
                _tell(f"{in_front} was in front of this. Closed it.")
            no_move["because"] = "a blocker was cleared, so this cycle is spent"
            return None

    _left__ = await _decide_the_next_move_blocker(blocker_attempts, clear_blocker, needs_person, no_move, observation)
    if _left__ is not _FALL_THROUGH:
        return _left__

    band, looking_at_the_thing = await _decide_the_next_move_what_she_looking(anchor, drawn, narrate, observation, responds, target_app)
    lattice, seen = await _decide_the_next_move_seen(band, coming, in_the_way, observation, responds, target_app)
    answering, lattice = _decide_the_next_move_part_4(
        knows, lattice, move_keys, responds, skilled=skilled, world=world
    )
    # The same reading, with a place for each thing in it. What she reads
    # is the string; what her claims are checked against is this.
    # The thing she is acting on, not the page it is drawn on.
    #
    # A reading of a screen is a reading of everything on it. Handed all
    # of it, the shape is called open because two hundred places is not
    # small, no rule about movement can match because most of a page never
    # moves, and two readings a second apart disagree about how many rows
    # there are. LIVE 2026-08-29: readings of 12x17 then 7x7 of a board
    # that is four by four, "how this moves is not worked out yet" after
    # eighty-four moves, and therefore a full language generation for
    # every single one of them — about twenty-eight seconds a move.
    whole = await _the_best_reading_available(
        observation,
        band,
        like=pending["whole"],
        in_a_browser=bool(anchor["page"]),
        answering=answering,
        lattice=lattice,
    )
    laid_out = _decide_the_next_move_laid_out(confirmed_here, got_to, lattice, moves, pending, whole)
    if not confirmed_here["value"]:
        await _decide_the_next_move_part_6(
            anchor, confirmed_here, expect_page, open_page, seen, target_app,
            seen_by_window=bool(observation.get("window_number")),
        )
        if not confirmed_here["value"]:
            not_there["reason"] = (
                f"{(open_page or expect_page)!r} is not what is in front of me — "
                f"{anchor['app'] or 'this window'} is"
            )
            logger.info("she is not where she was asked to be: %s", not_there["reason"])
            no_move["because"] = "she is not where she was asked to be"
            return None

    # Grade the last prediction before making another one.
    #
    # This is the difference between a loop that acts and one that steers.
    # The move it just made claimed something would be different; this
    # reading is the only chance to find out. A prediction that held is
    # weak evidence the move was understood, and one that broke is strong
    # evidence it was not — measured live, a run pressed the same key
    # forty times because nothing ever checked that the board moved.
    previous = pending["deliberation"]
    if previous is not None:
        if expected["took"] > 1 and expected["after"] is not None:
            from core.perception.how_it_moves import prediction_held

            previous.expected = replace(
                previous.expected or previous.chosen.expectation,
                becomes=expected["after"],
                becomes_holds=lambda after, forecast=expected["after"]: prediction_held(forecast, after),
            )
        attempt = confirm(
            previous,
            pending["before"],
            seen,
            spine=spine,
            graph=graph,
            toward=success_when,
            seen_before=pending["arranged"],
            seen_after=laid_out,
        )
        history.append(attempt)
        if pending["arranged"] is not None and attempt.progressed is not None:
            went.append((pending["arranged"], bool(attempt.progressed)))
            # And against WHAT KIND of situation it went that way, which
            # is the fact that carries to a place she has not been.
            if previous.chosen is not None and expected["took"] == 1:
                was = pending["arranged"]
                kind = was.as_shape() if hasattr(was, "as_shape") else ""
                if kind:
                    beats.it_went(
                        previous.chosen.name,
                        against=kind,
                        well=bool(attempt.progressed),
                    )
                    repeats.she_saw(kind, previous.chosen.name, laid_out)
        rose = _decide_the_next_move_world_where_she(expected, far, in_flight, laid_out, observation, pending, previous, responds, carries)
        if rose is not None:
            reaches.went(rose)
            if rose > 0:
                stretch["rises"] += 1
            # And what the situation this move left was like, against how
            # much the world's own count rose for it. That is a grade no
            # weight of hers can shift, which is what makes it able to
            # teach her the weights.
            if laid_out is not None and looking_at_the_thing:
                was_worked_out = matters.worked_out()
                matters.what_came_of_it(
                    terms(
                        laid_out,
                        toward=success_when or _what_there_is_to_aim_at(laid_out),
                        approach=plan["held"].approach if plan["held"] is not None else "",
                    ),
                    rose,
                )
                if matters.worked_out() and not was_worked_out:
                    logger.info("%s", matters.says())
                    if narrate:
                        _tell(f"I know what matters here now — {matters.says()}.")
        if previous.chosen is not None and expected["took"] == 1:
            # A key that never changes anything is not one of her actions
            # in this world, whoever wrote it down.
            can_do.tried(previous.chosen.name, attempt.verdict.observed_change)
            _it_did_nothing_from_here(
                pending, laid_out, previous.chosen.name, attempt.verdict.observed_change
            )
            # And whether the way it was sent reaches the thing at all.
            from .screen_pursuit_surface import it_answered

            it_answered(
                target_app or anchor["app"],
                previous.chosen.name,
                attempt.verdict.observed_change,
            )
            # And what stood around it when it did no harm — the two
            # pieces either side of a gap, found by taking things away
            # rather than by being described.
            if (
                attempt.verdict.observed_change
                and pending["arranged"] is not None
                and previous.chosen is not None
                and len(getattr(pending["arranged"], "cells", ())) <= 24
            ):
                shape = what_makes_it_safe(
                    pending["arranged"],
                    previous.chosen.name,
                    safe=lambda one, act: bool(
                        knows.rules.expect(one, act) is not None
                    ),
                    parts_of=lambda one: list(getattr(one, "cells", ())),
                    without=_the_same_thing_without,
                    where_of=lambda one: (one.row, one.column),
                    kind_of=lambda one: one.says,
                    about=lambda act: (0, 0),
                )
                if shape.around and shape.established:
                    logger.debug("what made %r safe: %s", previous.chosen.name, shape.describe())
            _decide_the_next_move_what_true_time(attempt, can_do, confirmed_here, in_the_way, opens, previous, responds)
            if can_do.dead() and not foreseen.get("acts"):
                foreseen["acts"] = True
                logger.info("what works here: %s", can_do.says())
        _decide_the_next_move_her_own_move(dropped, looking_at_the_thing, pending, previous, responds)
        if (
            pending["arranged"] is not None
            and previous.chosen is not None
            and looking_at_the_thing
            and not responds["state"].nothing_answers()
        ):
            # What a rule said would happen, before it is folded in. The
            # difference between that and what she actually saw is the
            # world's doing, and it is free at exactly this moment.
            if expected["took"] == 1:
                foretold = knows.rules.expect(pending["arranged"], previous.chosen.name)
                world.watched(foretold, knows.rules.the_thing(laid_out))
            went_well = _decide_the_next_move_part_10(dropped, expected, knows, laid_out, pending, plan, previous, responds, success_when)
            if plan["held"] is not None:
                lines.learned(A_LINE_HERE, plan["held"].approach, went_well)
                lines_held[plan["held"].approach] = plan["held"].as_memory()
            _decide_the_next_move_what_she_what(began_at, cannot_explain, laid_out, narrate, pending, plan, success_when, trying)
            further, made = _decide_the_next_move_part_12(furthest, laid_out, pending, plan, previous, responds, skilled, success_when, single_action=expected["took"] == 1)
            if further:
                furthest["here"] = max(furthest["here"], made)
                if narrate:
                    _tell(further)
            if going is not None and made and not moves:
                # What was on the board when she arrived is where she starts,
                # not something she climbed to.
                going.starting_from(made)
                # Where this run began, for playing things out from later.
                pending.setdefault("first_arranged", laid_out)
            elif going is not None and made:
                # A rung, and what it cost, and what she was holding while she
                # worked on it. Said with the reaching, because "the biggest
                # so far" on its own tells nobody whether it is going well.
                reached = going.noticed(
                    made,
                    len(moves),
                    holding=plan["held"].approach if plan["held"] is not None else "",
                )
                if reached:
                    # And the rest of her hears about it: what she is doing
                    # is not only what she set out to do and how she is going
                    # about it, but whether any of it is working.
                    doing.getting_somewhere(
                        going.where_it_stands(len(moves)), reached=made
                    )
                    # And the promise this run belongs to, when a request made
                    # one: what she carries into conversation about it said
                    # 0% for as long as she worked on it.
                    from core.agency.commitment_engine import report_progress

                    report_progress(
                        going.share_done(len(moves)), going.where_it_stands(len(moves))
                    )
                    # How the line she was holding did against what she said
                    # it would do. A prediction nobody reports on is a wish.
                    if narrate:
                        against = going.how_it_turned_out(len(moves))
                        if against:
                            _tell(against.capitalize() + ".")
                    # The line that was being held when she got up a rung is
                    # a line that worked, and that is what a line is for.
                    # Graded per move, an approach is judged on whether the
                    # last keystroke came out — which is the move's business,
                    # not the approach's.
                    if plan["held"] is not None:
                        lines.learned(A_LINE_HERE, plan["held"].approach, True)
                    if narrate:
                        _tell(f"Where this stands: {going.where_it_stands(len(moves))}.")
            _say_what_she_worked_out(knows, foreseen)
            _say_what_kind_of_problem(
                knows,
                screen_options(move_keys),
                laid_out,
                # What she is actually playing for, not what the request said.
                # The four other readers of this objective already derive it
                # when nothing was named; this one passed the request through,
                # so on an unnamed goal `recognise` was handed "" and reported
                # no countable goal — the shape went unnamed on exactly the
                # runs where naming it is the whole demonstration.
                success_when or _what_there_is_to_aim_at(laid_out),
                foreseen,
            )
            if laid_out is not None and pending.pop("judge_on_the_next_board", False):
                # The first board read after a restart is the new game's
                # first board, which is where judging has the most to decide.
                pending["first_arranged"] = laid_out
                judge = pending.get("when_a_game_ends")
                if callable(judge):
                    judge()
            if len(moves) % 6 == 0 and knows.rules is not None:
                logger.info(
                    "after %d move(s): %s%s | reading %dx%d%s | %s",
                    len(moves),
                    knows.rules.says(),
                    _what_she_is_not_reading(knows.rules),
                    laid_out.rows,
                    laid_out.columns,
                    _what_she_could_not_learn_from(dropped),
                    # What a move costs her, said where anyone watching the
                    # run can see it: a demo is a latency measurement.
                    "a look took %.2fs of which %.2fs was reading it, and she "
                    "thought for %.2fs"
                    % (
                        float(observation.get("seconds_to_still", 0.0) or 0.0)
                        + float(observation.get("seconds_reading", 0.0) or 0.0),
                        float(observation.get("seconds_reading", 0.0) or 0.0),
                        float(pending.get("thought_for", 0.0) or 0.0),
                    ),
                )
        _decide_the_next_move_learned_same_measurement(anchor, attempt, observation, pending, previous, responds, target_app)
        if moves:
            moves[-1]["held"] = attempt.verdict.held
            moves[-1]["outcome"] = attempt.verdict.why()
        pending["deliberation"] = None
        expected["after"], expected["took"] = None, 0

    if policy is not None:
        try:
            intent = await policy(observation)
        except (RuntimeError, TypeError, ValueError, KeyError) as exc:
            record_degradation(
                "screen_pursuit",
                exc,
                severity="info",
                action="ended a screen pursuit cycle without a move",
            )
            no_move["because"] = "the policy raised rather than answered"
            return None
        if not intent:
            no_move["because"] = "the policy offered no move"
            return None
        key = str(intent.get("key") or "").strip().lower()
        if key not in PRESSABLE_KEYS:
            no_move["because"] = "the policy named a key nothing can press"
            return None
        because = str(intent.get("because") or "").strip()
        # What reaches the learner is read off `pending`, and only the
        # deliberating branch below ever wrote it — a run driven by a
        # policy pressed keys and learned nothing at all, forever,
        # because the pair it moved through was never recorded as one.
        # A policy is a way of choosing, not a different kind of move, so
        # it is recorded the same way: the reading it chose from, the
        # name it chose, and why.
        from core.agency.deliberate_action import ActionOption, Deliberation

        pending["deliberation"] = Deliberation(
            goal=goal,
            situation=seen,
            chosen=ActionOption(name=key, detail=because),
            rationale=because,
        )
        pending["before"] = seen
        pending["arranged"] = laid_out
        pending["whole"] = whole
        pending["watched"] = observation
    else:
        # Find out how this is done — at the start, and again when what
        # she is doing has stopped working.
        #
        # A loop that only reads the screen in front of it can play badly
        # forever: it has the board, the moves and its own last few
        # outcomes, and none of that contains the thing a person would go
        # and look up. Being stuck is the signal, because a run of broken
        # predictions means the current approach is not working whatever
        # the reason.
        # In real time, words are for when acting and looking have stopped
        # carrying her: her last few predictions broke and she cannot see past
        # the next move. Everywhere else the world does not wait while she
        # reads up, states a line or talks a move over — asked to play, she
        # plays, and finds out what her acts do by doing them. LIVE 2026-09-17:
        # a lookup, an interpretation and a plan before the first key, a plan
        # asked for again every few moves, and every one of them a pause in a
        # game somebody was watching.
        from core.agency.looking_ahead import how_far_she_can_see

        # Lost is two things together. She cannot see past the next move, and
        # acting has stopped teaching her anything: her last predictions all
        # broke, or she has tried every act and still has no model of what
        # they do. Before either, acting and looking is the reasoning, and it
        # is faster than words.
        sees = (
            knows.rules is not None
            and knows.rules.rule() is not None
            and how_far_she_can_see() >= 2
        )
        # Every act still thought to do something here, taken at least once.
        # Read from the moves she made rather than from where changes were
        # seen, because an act she made that changed nothing is still an act
        # she has tried.
        made_moves = {str(move.get("key") or "") for move in moves}
        tried_everything = all(
            str(name) in made_moves for name in (can_do.available() or move_keys)
        )
        no_model = knows.rules is None or knows.rules.rule() is None
        lost = not sees and (stuck(history) or (tried_everything and no_model))
        # A lookup that ran out of time is not tried again at once. Its
        # timeout left nothing held, and nothing held was the reason to look,
        # so it ran again on every move she was lost — nine seconds each
        # (live, 2026-09-18). Twice as long to wait each time it fails.
        may_look = len(moves) >= int(knowledge.get("quiet_until", 0))
        if lost and may_look and (knowledge["held"] is None or knowledge["relearned"] < MAX_RELEARNS):
            if knowledge["held"] is not None:
                knowledge["relearned"] += 1
            relearning = knowledge["held"] is not None
            knowledge["meant"] = []
            # For as long as the world she is in will wait, and no longer.
            #
            # Reading up is worth a pause in a world that stands still and is
            # abandoning the game in one that does not. LIVE 2026-09-17,
            # playing a board that answered in a fifth of a second: one lookup
            # went to the web and took three minutes and fifty-two seconds,
            # another forty-three, while a person watched a board that did not
            # move. The time a lookup may take is the time her own moves take
            # — ten of them, so it is a pause and not a stop.
            looks = list(getattr(run, "reading_took", None) or [])
            a_look = (sum(looks) / len(looks)) if looks else 0.3
            may_take = min(max(2.0, 10.0 * a_look), max(0.0, ends_at - time.monotonic()))
            try:
                knowledge["held"] = await asyncio.wait_for(
                    learn_about(
                        goal,
                        search=research,
                        remember=not relearning,
                        because_stuck=relearning,
                        situation=content_text(
                            observation, region_top=region_top, region_bottom=region_bottom
                        ),
                        history=history[-RECENT_ATTEMPTS:],
                    ),
                    timeout=may_take,
                )
            except TimeoutError:
                knowledge["timeouts"] = int(knowledge.get("timeouts", 0)) + 1
                knowledge["quiet_until"] = len(moves) + min(64, 4 * 2 ** int(knowledge["timeouts"]))
                logger.info(
                    "reading up took longer than %.1fs, which is longer than this "
                    "world waits; carrying on with what she knows, and not asking "
                    "again for %d move(s)",
                    may_take,
                    int(knowledge["quiet_until"]) - len(moves),
                )
                knowledge["held"] = None
        learned = knowledge["held"].as_evidence() if knowledge["held"] is not None else []
        # Work out what it means HERE before deciding with it.
        #
        # Retrieving advice is not applying it. "Keep your largest tile in
        # a corner" is a fact about the game; what it means depends on
        # where the tiles actually are, and that comparison is the step
        # between reading something and playing differently.
        if lost and knowledge["held"] is not None and knowledge["held"].known and not knowledge["meant"]:
            knowledge["meant"] = await work_out_what_it_means(
                knowledge["held"],
                seen,
                screen_options(move_keys),
                think=_within_the_run(
                    think or _her_reasoning(stakes), ends_at, _a_move_here(run)
                ),
                history=history[-RECENT_ATTEMPTS:],
            )
        learned = learned + [meaning.as_evidence() for meaning in knowledge["meant"]]

        # The line she is taking, and the thing that would end it.
        #
        # Choosing a move from what is on screen is reacting: it assumes
        # the world sits still and corrects once the world says
        # otherwise. Anywhere the world keeps moving while she works, an
        # approach is the missing middle — a line held across moves, with
        # the condition that would make it wrong named when she adopts it
        # rather than discovered when it fails. The condition is checked
        # here, before she acts, so a pivot is something she was watching
        # for and not something that happened to her.
        holding, ended = still_holds(plan["held"], seen, len(moves))
        # An approach is reconsidered because it has stopped working, not
        # because a number of moves has gone by. Her own record of what a
        # rung costs here is what says so.
        if holding and going is not None:
            stalled = going.why_reassess(len(moves))
            if stalled:
                holding = False
                ended = f"it has not moved me on: {stalled}"
                # And a line dropped for not moving her is a line that did
                # not work here, which is the other half of learning one.
                if plan["held"] is not None:
                    lines.learned(A_LINE_HERE, plan["held"].approach, False)
                going.it_was_reassessed()
                logger.info("the approach is being looked at again: %s", stalled)
        # The line is thought over beside the play, not in place of it.
        #
        # Asked in line, the question held the board still until it was
        # answered, so it was given the time ten moves take and no more.
        # LIVE 2026-09-23: the 27B needs about a hundred seconds to read the
        # question and write a line, the board moved every two and a half,
        # and every question was cancelled at eight with nothing said. A
        # person playing fast thinks about how to play while their hands keep
        # going, and takes up the answer when it comes. Asked that way it
        # costs no move, so it is not kept for when she is lost.
        thinking = plan.get("thinking")
        fresh = None
        if thinking is not None and thinking.done():
            plan["thinking"] = None
            fresh = _what_the_thought_came_to(thinking)
            # A voice that came back with nothing is asked again later each
            # time, the way a lookup that timed out is. Asked again five moves
            # on, it was put the same question every ten seconds while it
            # could not answer any of them (live, 2026-09-23).
            plan["unanswered"] = 0 if fresh is not None else plan.get("unanswered", 0) + 1
            ended = plan.pop("asked_because", ended)
            # Counted from the answer, so the next question waits for her to
            # have played under this one.
            plan["asked_at"] = len(moves)
        a_line_is_due = (
            plan.get("thinking") is None
            and fresh is None
            and _decide_the_next_move_part_14(ended, holding, looking_at_the_thing, moves, plan, costs)
        )
        if not holding and a_line_is_due:
            plan["asked_at"] = len(moves)
            plan["asked_because"] = ended
            plan["thinking"] = _thought_over_beside_the_play(
                settle_on_an_approach(
                    goal,
                    seen,
                    screen_options(move_keys),
                    # Deciding the line she will hold across a hundred moves
                    # is not the same question as deciding one of them, and
                    # asking it with the thinking that suits a move got the
                    # model's own warm-up handed back as a plan.
                    #
                    # Bounded by the run and not by a move: nothing waits.
                    think=_within_the_run(think or _reasoning_for_a_plan(), ends_at),
                    knowledge=learned,
                    history=history[-RECENT_ATTEMPTS:],
                    previous=plan["held"],
                    moves_made=len(moves),
                )
            )
        if fresh is not None:
            changing = plan["held"] is not None
            # The same line again is not a new line. Asking and getting the
            # answer she already had says the question is not what is
            # changing, so it is not said again and she waits longer before
            # putting it again. LIVE 2026-09-17, a plan about a corner the
            # board did not yet hold: judged broken on every move, asked
            # again on every move, and announced on every move.
            same = changing and " ".join(fresh.approach.split()).casefold() == " ".join(
                plan["held"].approach.split()
            ).casefold()
            plan["same_again"] = plan.get("same_again", 0) + 1 if same else 0
            plan["held"] = fresh
            plan["changes"] += 1 if changing and not same else 0
            # Held where the rest of her can see it, not in this loop.
            doing.going_about_it(
                fresh.approach,
                because=fresh.because,
                watching_for=fresh.holds_while.describes,
                alternatives=fresh.otherwise,
                spine=spine,
                lived=lived,
            )
            if narrate and not same:
                said = fresh.narrate()
                # And what she expects it to do, which is what makes it a
                # line rather than a remark: the rung it is for, and what
                # a rung has cost here.
                if going is not None:
                    wants = going.expecting(fresh.approach, len(moves))
                    if wants:
                        said = f"{said} — {wants}"
                # And whether anything she read bears it out. Her voice
                # is one witness; what she read is another, and a line
                # only her own voice vouches for is held knowing that.
                read = knowledge["held"].findings if knowledge["held"] is not None else []
                if read:
                    from core.agency.what_agrees import borne_out

                    borne = borne_out(fresh.approach, read)
                    said = f"{said} ({borne.says_so()})"
                    logger.info("the line she took, against what she read: %s", borne.says_so())
                _tell(f"{said} ({ended})" if changing and ended else said)
            elif going is not None:
                going.expecting(fresh.approach, len(moves))
        if plan["held"] is not None:
            learned = learned + plan["held"].as_evidence()

        available, ended = _decide_the_next_move_nothing_task_working(
            can_do, move_keys, observation, offered_a_restart, responds, knows=knows, laid_out=laid_out
        )
        if not ended:
            available, ended = _leaving_out_what_just_did_nothing(available, pending, laid_out)
        if ended:
            mine_now = target_app or anchor["app"]
            why = work_out_why(
                mine=mine_now,
                # A picture taken by window number is of her window whatever
                # is in front, and keys addressed to its process reach it
                # whatever is in front. Another application in front is then
                # not a reason nothing answers: the person may simply be
                # watching the conversation while she plays.
                in_front=(
                    str(observation.get("owner") or mine_now)
                    if observation.get("window_number")
                    else await _frontmost()
                ),
                on_top=await _whats_on_top(mine_now, over=responds["state"].band()),
                still_there=_is_a_thing_laid_out(laid_out),
            )
            if why.can_fix:
                # Not an ending. Something she can do something about, and
                # the doing is the answer rather than the reporting.
                if narrate and not said_it_ended["value"]:
                    _tell(why.says())
                mended = (
                    await _ensure_page(anchor["page"])
                    if why.because == ELSEWHERE
                    else await clear_what_is_in_front(why.what)
                )
                if mended:
                    # Reachable again, so this cycle is not an ending. The
                    # rest of it proceeds normally: she has her window
                    # back and there is a move to choose.
                    responds["state"].began_again()
                    ended = False
                elif responds.get("ended_by") == ONLY_SILENCE:
                    # Something over it that will not move is still not an
                    # ending. Silence was the only evidence, and silence is
                    # what a dialog holding the keyboard looks like; the
                    # thing is still there under it. Kept as an ending, it
                    # offered starting again without needing a reason, and
                    # LIVE 2026-09-24 she threw away a board with a 64 on it
                    # to a notification she could not close. It is somebody
                    # else's to answer, so she says so once and keeps trying
                    # the thing she was asked to do.
                    responds["state"].began_again()
                    ended = False
                    if narrate and not said_it_ended.get("blocked"):
                        said_it_ended["blocked"] = True
                        _tell(
                            f"{why.what} will not close, and it is not mine to "
                            "answer. What I am working on is still under it, so "
                            "I am carrying on."
                        )
            elif why.because == ENDED and narrate and not said_it_ended["value"]:
                said_it_ended["value"] = True
                _tell(why.says())
        # Seeing it through is a choice about something still going. Chosen
        # once while a game was going badly, it closed this door for the rest
        # of the run, so when the game really ended no way to begin again was
        # offered: LIVE 2026-09-23, eighteen minutes of pressing right into
        # "Game over!". An ending is offered its way out whatever was chosen
        # before it.
        if ended or (stuck(history) and not seen_through["value"]):
            out = ways_out(observation, ended=ended)
            if ended and out:
                # Pressing a move key into something that has finished is
                # not one of the things she could do. Offering it beside
                # the real choices made every cycle look like a decision
                # between four keys and a restart, so she went on playing
                # a game that was over — measured live, thirty-nine moves
                # after Game Over, each one costing a language pass
                # because the situation was unusual.
                available = out
            else:
                available = available + out
            if ended and not said_it_ended["value"]:
                said_it_ended["value"] = True
                if narrate:
                    _tell(
                        await _why_nothing_answers(
                            target_app or anchor["app"],
                            over=responds["state"].band(),
                        )
                    )
        # Her own pacing is hers to decide, once there is really a gap.
        behind = narration_backlog() if narrate else {}
        # A pace chosen because the commentary was behind ends when it
        # is not. Left standing, one choice governs the whole run and she
        # never gets asked again.
        if pacing["choice"] and not behind.get("waiting"):
            pacing["choice"] = ""
            pacing["brief"] = False
        offered_pacing = bool(behind.get("waiting")) and not pacing["choice"]
        if offered_pacing:
            available = available + pacing_options(behind)

        # Effort follows what rides on this one. A routine move is a
        # routine move; a run that has stopped getting anywhere, or one
        # weighing whether to start over, is worth more than one pass.
        # What is unusual is the situation, not the number of buttons.
        #
        # This counted the options: a way out is appended whenever the
        # screen has one, and a game page has a New Game button on it
        # permanently — so every ordinary move was treated as a moment
        # worth weighing, and paid a language pass for it. Measured live
        # 2026-08-26: seventeen passes for fourteen moves, twenty seconds
        # a move, and a run that spent its budget deliberating over a
        # button she was never going to press.
        held_line = plan["held"].approach if plan["held"] is not None else ""
        unusual = stuck(history) or ended or offered_pacing
        weight = stakes if unusual else min(stakes, 0.3)
        # What she has decided to keep true takes moves off the table
        # before anything looks ahead, which is where holding something
        # pays: the tree it searches is smaller at every level.
        #
        # Only while looking is short-sighted. Once her last look saw past the
        # next move, what a move leads to is in front of her, and a property
        # guessed from a handful of positions removes moves the look would
        # have kept. LIVE 2026-09-17, through her whole loop on a board she
        # could search three deep: "holding that it holds two 32s, so not
        # down, up", two moves to choose from all game, lost at a 128.
        from core.agency.looking_ahead import how_far_she_can_see

        wont, ruled_out = (
            _moves_she_will_not_make(
                she_keeps,
                went,
                laid_out,
                [option.name for option in available],
                knows,
                len(moves),
                world,
            )
            if how_far_she_can_see() < 2
            else (frozenset(), "")
        )
        if wont:
            available = [one for one in available if one.name not in wont]
            if ruled_out != she_keeps.get("said"):
                she_keeps["said"] = ruled_out
                logger.info("%s", ruled_out)

        available = _decide_the_next_move_act_has_done(available, knows, laid_out, reaches, responds, stretch)

        # Has she been anywhere LIKE this before.
        #
        # What she learns is filed under the thing she learned it in, so
        # a second world that moves in exactly the same way used to start
        # as ignorant as the first, and the fortieth was no better off
        # than the second. Two worlds are of a kind when they are the
        # same size, take the same acts and are countable in the same
        # ways — and worlds of a kind move alike, which is the whole of
        # what gets carried.
        #
        # Asked once, on the first reading worth naming a kind from, and
        # only where this world has taught her nothing yet.
        if (
            not like_it["looked"]
            and laid_out is not None
            and laid_out.occupied()
            and knows.rules is not None
            and not knows.rules.seen
        ):
            like_it["looked"] = True
            like_it["kind"] = _the_kind_of_world_this_is(
                laid_out,
                [option.name for option in available],
                success_when or _what_there_is_to_aim_at(laid_out),
            )
            elsewhere = recall(like_it["kind"]) if like_it["kind"] else {}
            # And only from a world she was reading through a grid of
            # this shape. A rule that survived somewhere else is about the
            # thing it was watched in; read through a grid of another
            # shape it is not weak evidence about this one, it is about
            # something that is not here. Discounting it does not help,
            # because early on a handful of counts is the whole difference
            # between looking ahead and not.
            if elsewhere and responds["lattice"].held:
                read_through = (elsewhere.get("moves") or {}).get("read_through")
                here = [responds["lattice"].rows, responds["lattice"].columns]
                if list(read_through or ()) != here:
                    elsewhere = {}
            if elsewhere:
                carried = _no_more_than_a_fresh_one_is_worth(elsewhere.get("moves"))
                knows.rules.__dict__.update(
                    HowItMoves.from_memory(
                        elsewhere.get("moves") or {}, carried
                    ).__dict__
                )
                skilled.__dict__.update(
                    WhatWorkedBefore.from_memory(
                        elsewhere.get("skill") or {}, carried
                    ).__dict__
                )
                world.__dict__.update(
                    WhatTheWorldDoes.from_memory(
                        elsewhere.get("world") or {}, carried
                    ).__dict__
                )
                _tell(
                    f"I have been somewhere like this before — {like_it['kind']} — "
                    "so I will start from what that moved like."
                )
                logger.info(
                    "borrowed from %r at %.2f trust: %s",
                    like_it["kind"], carried, knows.rules.says(),
                )

        # Where each move would lead, when she has worked out how this
        # moves and there is anything to prefer one future over another by.
        ahead: dict[str, tuple[float, str]] = {}
        # What she is playing for, which the request does not always say.
        aiming_at = success_when or _what_there_is_to_aim_at(laid_out)
        pending["aiming_at"] = aiming_at
        if worth_comparing(aiming_at, held_line):
            # As far ahead as there is time to look, which is decided from
            # what a level of looking has been measured costing.
            # About as long to think as it takes to look. Looking is the pace
            # the world already sets on every move; thinking far past it makes
            # a move stand still on screen for what a further level mostly
            # confirms. Measured 2026-09-17 on eight simulated games: at 0.3s
            # a move all eight reached 2048, and a two-second allowance won the
            # same eight with pauses of nearly three seconds.
            # As long as reading one costs her, not as long as the world
            # takes to answer.
            #
            # The time between her act and her next look is mostly the world
            # moving: a board slides and settles. Counting that as the cost of
            # looking made her think for two seconds a move on a game that
            # answers in a fifth of one, which is a demo nobody would call
            # real time (live, 2026-09-17).
            a_read = float(observation.get("seconds_reading", 0.0) or 0.0)
            if a_read <= 0.0:
                looks = list(getattr(run, "reading_took", None) or [])
                a_read = (sum(looks) / len(looks)) if looks else 0.3
            thinking_for = _how_long_to_think(
                laid_out,
                least=max(0.3, a_read),
                most=min(2.0, (ends_at - time.monotonic()) * 0.02),
            )
            thought_from = time.monotonic()
            # No deeper than her model has been worth here.
            #
            # A level past where her predictions stop beating "nothing
            # changed" is a level of fiction, and it looks surer the further
            # it goes. Measured per world from her own graded predictions;
            # nothing until enough of them are graded, and then a real bound.
            as_far_as_it_carries = carries.carries_to() if carries is not None else 0
            ahead = look_ahead(
                knows.rules,
                laid_out,
                [option.name for option in available],
                toward=aiming_at,
                approach=held_line,
                budget_s=thinking_for,
                world=world,
                # What matters HERE, once she has watched enough to say.
                weights=matters.weights(),
                no_deeper_than=as_far_as_it_carries,
            )
            pending["thought_for"] = time.monotonic() - thought_from
        # And what a move would TELL her, which is a different question
        # from where it leads.
        #
        # Two rules that both fit everything she has seen disagree about
        # some acts and agree about others. An act they agree on can go
        # well and still leave her exactly as unsure as before; an act
        # they split over settles which of them is right whatever happens.
        # Being right about the rule improves every move after this one,
        # so early on it is worth more than the position it costs — which
        # is why a person pushes a thing one way once, early, and then
        # never needs to again.
        #
        # It goes to nought by itself as the evidence rules them out, so
        # there is nothing to turn off.
        telling = worth_finding_out(
            knows.rules,
            laid_out,
            [option.name for option in available],
            ahead,
            # And, before any of that can mean anything, the acts she has
            # not taken here.
            never_tried=[
                option.name
                for option in available
                if option.name not in responds["state"].tried
            ],
        )
        if telling:
            if ahead:
                ahead = {
                    name: (value + telling.get(name, 0.0), reason)
                    for name, (value, reason) in ahead.items()
                }
            else:
                ahead = {
                    name: (value, "this is the one that would settle how this moves")
                    for name, value in telling.items()
                }
            logger.info(
                "acting to find out: %s",
                ", ".join(
                    f"{name} {value:.3f}"
                    for name, value in sorted(
                        telling.items(), key=lambda pair: -pair[1]
                    )[:4]
                ),
            )
        # And how much of what happens next is the world's rather than
        # hers, which is the other thing looking ahead averages away.
        #
        # A position where every reply leaves her fine is not the same as
        # one where the average reply leaves her fine and one of them
        # ruins her. That difference is what a grip buys: a hand on a
        # wrist does not improve the position and is not free, and what
        # it gets is that whatever happens next she is still where she
        # was. Keeping the largest thing in a corner buys exactly that.
        #
        # Which way to lean on it is not a temperament. Ahead with time
        # to spare, an uncertain position is worth avoiding — she only
        # has to keep doing what works. Behind with the clock going, it
        # is worth seeking, because the average outcome of what she is
        # doing is already a loss and the spread is the only thing that
        # contains a win. Both come off the run: how much budget is left,
        # and what she has been getting per act.
        exposed = at_the_worlds_mercy(
            knows.rules,
            laid_out,
            [option.name for option in available],
            toward=success_when or _what_there_is_to_aim_at(laid_out),
            approach=held_line,
            world=world,
        )
        if exposed and ahead:
            worths = [value for value, _why in ahead.values()]
            spread = max(worths) - min(worths)
            lean = whether_to_take_the_wide_option(
                max(0.0, ends_at - time.monotonic()) / max(1e-9, ends_at - began),
                _how_it_has_been_going(began_at, laid_out),
                against_a_clock=not success_when,
            )
            if spread > 0.0 and lean:
                ahead = {
                    name: (value + lean * spread * exposed.get(name, 0.0), why)
                    for name, (value, why) in ahead.items()
                }
                logger.info(
                    "the world could swing this; leaning %+.2f: %s",
                    lean,
                    ", ".join(
                        f"{name} {share:.2f}"
                        for name, share in sorted(
                            exposed.items(), key=lambda pair: -pair[1]
                        )[:4]
                    ),
                )
        # What she is aiming at, where it names a number or a band.
        #
        # "More is better" is true of some goals and quietly false of many:
        # a load high enough to be worth running and low enough to survive,
        # a bid over one number and under another. Overshooting looks like
        # success right up to the moment it is a disaster, because the
        # measure that says more says more all the way past the edge.
        if ahead and aiming_at:
            numbers = [
                float(one)
                for one in re.findall(r"-?\d+(?:\.\d+)?", str(aiming_at))
            ][:2]
            if len(numbers) == 2 and numbers[0] < numbers[1]:
                band = AWindow(at_least=numbers[0], at_most=numbers[1])
                landing = which_act_lands_in_it(
                    list(ahead),
                    now=0.0,
                    what_it_moves=lambda one: ahead[one][0],
                    window=band,
                )
                if landing and not landing[0][2]:
                    ahead = {landing[0][0]: ahead[landing[0][0]]}
            elif len(numbers) == 1:
                # A bar rather than a band: which most often clears it,
                # which is not the same as which averages best.
                took = the_one_most_likely_to_do(
                    {one: [worth] for one, (worth, _why) in ahead.items()},
                    needs=numbers[0],
                )
                if took is not None and took.clears_it > 0:
                    ahead = {took.name: ahead[took.name]}

        kind = _decide_the_next_move_where_move_she(ahead, aiming_at, available, goal, laid_out, marks, wont)
        # What she has learned about this KIND of situation, where the
        # world is the sort that repeats. Where it is dealt fresh, a fact
        # about one place is noise she would be storing at her own
        # expense, and the ordering falls back to what it was.
        if kind and beats.against and repeats.worth_remembering_places():
            liked = [
                one
                for one, _how, tried in beats.in_order(
                    [option.name for option in available], against=kind
                )
                if tried
            ]
            if liked:
                order = {name: at for at, name in enumerate(liked)}
                available = sorted(
                    available, key=lambda one: order.get(one.name, len(order))
                )
        recognised = (
            skilled.suggests(kind, tuple(option.name for option in available))
            if kind
            else ""
        )
        asking, because_of = worth_a_pass(
            ahead,
            stakes=weight,
            # Moves since she last said anything. Narrating every move is
            # saying something every move, so silence is no reason for a pass.
            since_words=0 if narrate else len(moves) - asked["at"],
            horizon=LANGUAGE_EVERY,
            unusual=unusual or not moves or restarts["count"] > asked["after_restarts"],
            recognised=recognised,
            # How far she can trust her own arithmetic here, which is how
            # often the rule she is using has been right about this world.
            how_sure=(
                carries.how_sure_she_should_be(knows.rules.confidence())
                if carries is not None and knows.rules is not None
                else knows.rules.confidence() if knows.rules is not None else 0.0
            ),
            # What a pass costs, in moves not made, from this run's own
            # clock. Live on a resident model it was about ten.
            costs_moves=_a_pass_in_moves(costs),
        )
        if asking and not (lost or offered_pacing):
            # A pass is for when she is lost, or when there is a choice about
            # her own pace to make. Otherwise the move is made from what she
            # can see, at the speed the world is going.
            asking, because_of = False, f"playing in real time ({because_of})"
        if len(available) == 1:
            asking, because_of = False, "there is only one thing to do"
        # A voice that has not been answering is given time before it is
        # asked again. Asked every move while it could not answer, it cost
        # thirteen seconds a move for as long as she was lost (live,
        # 2026-09-18), and the move she made at the end of each wait was the
        # one she would have made without it.
        if asking and len(moves) < int(asked.get("quiet_until", 0)):
            asking, because_of = False, "my voice has not been answering, so I am deciding this myself"
        if recognised and not asking:
            skilled.took(kind)
        if asking != last_call["asked"] or last_call["why"] != because_of:
            last_call.update({"asked": asking, "why": because_of})
            logger.info("%s: %s", "thinking about this one" if asking else "no need to think", because_of)
        if asking:
            asked["at"] = len(moves)
            asked["after_restarts"] = restarts["count"]
        thinking_from = time.monotonic()
        chosen = await deliberate(
            goal,
            seen,
            available,
            foresight=ahead or None,
            seeing=laid_out,
            think=(
                _within_the_run(think or _her_reasoning(weight), ends_at, _a_move_here(run))
                if asking
                else None
            ),
            knowledge=learned,
            history=history[-RECENT_ATTEMPTS:],
            stakes=stakes,
            control_point="screen_pursuit.next_move",
            # Her plan reaches the moves she does not put into words,
            # which is most of them.
            approach=plan["held"].approach if plan["held"] is not None else "",
            lived=lived,
            spine=spine,
            graph=graph,
            # Reported once, when the body acts, with the reasoning on it.
            announce=False,
        )
        if asking:
            costs["pass_s"] += time.monotonic() - thinking_from
            costs["passes"] += 1.0
            # Twice as long each time it does not answer, and from nothing
            # again the first time it does.
            if chosen.spoke:
                asked["misses"] = 0
            else:
                asked["misses"] = int(asked.get("misses", 0)) + 1
                asked["quiet_until"] = len(moves) + min(32, 2 ** int(asked["misses"]))
        else:
            costs["was_quiet"] = 1.0
        pending["ahead"] = dict(ahead or {})
        pending["held_line"] = held_line
        if not chosen.reached:
            # Stop rather than press something for no reason. A loop that
            # keeps acting once its judgement is out of reach is the exact
            # failure this decision path was built to end.
            undecided["reason"] = chosen.reason
            no_move["because"] = "she could not settle on one"
            return None
        key = chosen.chosen.name
        because = chosen.rationale

        if key in {SLOW_DOWN, SAY_LESS, PRESS_ON}:
            # A decision about herself. It changes how the next moves are
            # made rather than being one, so the cycle ends here and the
            # next one acts on it.
            pacing["choice"] = key
            pacing["because"] = because
            pacing["brief"] = key == SAY_LESS
            if key == SLOW_DOWN:
                await let_the_voice_catch_up(behind)
                pacing["waits"] += 1
                # Chosen once and then re-decided as the gap changes,
                # rather than committing the rest of the run to one pace.
                pacing["choice"] = ""
            no_move["because"] = "she chose a pace rather than a move"
            return None

        if key == SEE_IT_THROUGH:
            # Chosen once. It says "stop offering me the way out", not
            # "do something", so the loop carries on with the moves it has.
            seen_through["value"] = True
            seen_through["because"] = because
            no_move["because"] = "she chose to see it through"
            return None

        if key == START_OVER:
            # What she cannot get back is only at stake while she can still
            # act on it. A thing that has ended has already taken it.
            _left__ = (
                _FALL_THROUGH
                if ended
                else _decide_the_next_move_while_there_something(
                    available, chosen, laid_out, no_move, responds, she_keeps
                )
            )
            if _left__ is not _FALL_THROUGH:
                return _left__
            params = dict(chosen.chosen.params)
            label = str(params.get("label") or "")
            rx, ry = float(params.get("x", 0.0)), float(params.get("y", 0.0))
            frame = list(observation.get("bounds") or [])
            # Always a reason, even when her wording was unusable. The
            # filters drop an echo of the evidence, which is right, and a
            # decision with no recorded reason is not much better than an
            # unexplained one.
            restarts["because"] = because or "nothing here was moving the board"
            intending["value"] = START_OVER
            history.clear()
            was_showing = seen

            async def begin_again() -> bool:
                clicked = await click_normalized(
                    rx, ry, expect_app=target_app or anchor["app"], bounds=frame
                )
                if not clicked:
                    return False
                # Deciding to start again is not starting again.
                #
                # The verdict that nothing answers was cleared when she
                # CHOSE to restart, so a click that landed on nothing left
                # her believing the world was fresh — and she went back to
                # pressing keys into a finished game. LIVE 2026-08-26:
                # "Nothing I do is changing anything here — this attempt is
                # over", and then "Going right", at a board reading Game
                # Over the whole time.
                #
                # A click reports success for having happened. Only the
                # screen can say whether it did anything.
                after = await observe()
                now_showing = within(after, responds["state"].band(), responds["state"])
                if now_showing.strip() == was_showing.strip():
                    logger.info("the restart did not take — the screen is unchanged")
                    return False
                restarts["count"] += 1
                responds["state"].began_again()
                logger.info("began again: %s", label or "restart")
                # One game, played through by what she judges by. Rehearsing
                # them at a game's end rather than only at the run's is the
                # difference between carrying a measure for one game and
                # carrying it for hours. Rehearsed from the NEW game's first
                # board once it is read: the run's first board can be the
                # last one of an old game, where every way of judging goes
                # nowhere and nothing is learned (live, 2026-09-23).
                pending.pop("first_arranged", None)
                pending["judge_on_the_next_board"] = True
                # And what she has been getting per act is counted from this
                # game's start. Counted from the board before the restart, a
                # finished game whose tiles summed to far more than a new one,
                # every board of the new game read as a loss, and the lean on
                # what the world could swing said "behind, take the wide
                # option" for a whole game, harder as the clock went (live,
                # 2026-09-23: +0.03 rising to +0.33 over 175 moves).
                began_at["worth"], began_at["seen"] = None, 0
                # And what counts as furniture is what the new start shows.
                # Measured against the run's first reading, the last board of
                # an old game, the fresh game's own New Game read as a way to
                # start again that had just appeared: she ended every game as
                # it began and pressed New Game on an empty board for
                # minutes (live, 2026-09-24).
                offered_a_restart["was_there"] = None
                offered_a_restart["said"] = False
                return True

            return Step(name=f"begin again with {label!r}", action=begin_again)

        # What she actually expects to see, when she knows how this moves.
        #
        # A move carries a claim, and the claim is what her being right
        # gets measured by — the length of her plans, which part of the
        # screen she believes answers to her, and what her moves are worth
        # are all read off that verdict. The claim on offer was "the view
        # will be different", which almost any keystroke satisfies: LIVE
        # 2026-09-01 every move on a real board came back
        # predicted='the view to be different after left', held=True, and
        # holding meant nothing.
        #
        # She has a better claim available and was not making it. The rule
        # she worked out by watching says what the arrangement becomes,
        # exactly, and that claim can be wrong — which is the only kind
        # worth checking, because being wrong about it is her model of
        # this world being wrong.
        foretold_by_the_rule = None
        try:
            foretold_by_the_rule = knows.rules.expect(laid_out, key)
        # not a failure: a reading the rule cannot be applied to has no expectation to give.
        except (AttributeError, TypeError, ValueError):
            foretold_by_the_rule = None
        if foretold_by_the_rule is not None:
            from core.perception.how_it_moves import prediction_held

            was = chosen.expected or chosen.chosen.expectation
            chosen.expected = replace(
                was,
                becomes=foretold_by_the_rule,
                # Checked the way the rules themselves are scored, so a
                # tile the world deals does not read as her being wrong.
                becomes_holds=(
                    lambda after, foretold=foretold_by_the_rule: prediction_held(
                        foretold, knows.rules.the_thing(after)
                    )
                ),
                describes=(
                    was.describes
                    or f"the thing to become what {key} makes of it"
                ),
            )
        pending["deliberation"] = chosen
        pending["before"] = seen
        pending["arranged"] = laid_out
        pending["whole"] = whole
        # Kept whole for the comparison that finds where she has effects,
        # which cannot use a band it has not worked out yet.
        pending["watched"] = observation

    # The first move is recorded when it lands, like every other one.
    #
    # Written here, before the body was asked to do anything, it counted
    # whether or not the keystroke reached a window. LIVE 2026-08-26:
    # thirty-five moves in the record, a board that had not changed once,
    # and no correction said out loud — because only the FOLLOW-ONS of a
    # sequence were written from what landed, and a plan of one has no
    # follow-ons.
    about_to = {"key": key, "because": because, "at": time.time()}
    # What she says as she makes it: what her rule says the move does, and
    # why it beat the next best. Said for every move when she was asked to
    # narrate, because a person watching wants each move and what it was for.
    move_said = (
        _what_she_says_as_she_moves(
            key,
            laid_out,
            knows,
            pending.get("ahead") or {},
            weights=matters.weights(),
            toward=str(pending.get("aiming_at") or success_when or ""),
            approach=str(pending.get("held_line") or ""),
            biggest_so_far=max(float(furthest.get("here") or 0.0), float(furthest.get("again") or 0.0)),
            last_said=pending.setdefault("reason_last_said", {}),
        )
        if narrate
        else ""
    )
    # Nothing is said here on purpose.
    #
    # Every decision is published to the deliberation stream as it is
    # made, and a narrator — if one is running — speaks about it on its
    # own schedule. Saying the line inline made the next move wait on
    # language, which is backwards: she should be able to play at full
    # speed and describe it, play silently, or narrate something else
    # entirely, and the loop should read the same in all three cases.

    made = pending["deliberation"]
    named = [option.name for option in getattr(made, "then", ()) or ()] if made else []
    # And where she named no sequence, one taken on the model instead.
    expected["after"], expected["took"] = None, 0
    foresee = _expects(knows)
    choices = [option.name for option in available]

    def pick(board: Any, names: Sequence[str]) -> str:
        ahead_now = look_ahead(
            knows.rules, board, list(names),
            toward=aiming_at, approach=held_line,
            budget_s=0.05, world=world, weights=matters.weights(), settles_how_far=False,
        )
        return max(ahead_now, key=lambda one: ahead_now[one][0]) if ahead_now else ""

    # A sequence she cannot carry from one step to the next is a list of
    # alternatives, not a plan.
    follow_on = a_run_she_can_carry(
        named,
        foresee(pending["arranged"], key)
        if (foresee is not None and pending["arranged"] is not None)
        else None,
        foresee,
        pick,
        choices,
        int(getattr(responds["state"], "acts", 0) or 0),
        int(getattr(responds["state"], "effective", 0) or 0),
    )
    # A world that adds things of its own between acts cannot be predicted past
    # the next of them: what the model foresees after one act is the board
    # without the thing the world is about to put on it, so a second act is
    # chosen for a board that will not exist. Looking after every act is the
    # only honest pace there, and looking is a third of a second.
    if follow_on and knows.rules is not None and (
        knows.rules.world_adds_things() or world.acts_with_arrivals >= 2
    ):
        follow_on = []
    if (
        not follow_on
        and foresee is not None
        and pending["arranged"] is not None
        and (knows.rules is None or not knows.rules.world_adds_things())
        and world.acts_with_arrivals < 2
    ):
        trusted = float(knows.rules.confidence())
        going = far.how_many(trusted=carries.how_sure_she_should_be(trusted) if carries is not None else trusted)
        if going > 1:
            follow_on, _ = _the_rest_of_the_run(
                key,
                pending["arranged"],
                foresee,
                choices,
                pick,
                going,
            )
            if follow_on:
                logger.info(
                    "going %d without looking (%s)",
                    len(follow_on) + 1,
                    far.describe(),
                )
    # How many acts the next reading will be the result of, and what she
    # expects it to be. Said once, for both ways a run gets made.
    #
    # This used to be counted only for the run she built herself from the
    # model. A deliberation that named its own sequence left it at one
    # while four keys went out — so the guard that refuses to learn from
    # several acts and one reading never fired, and every pair the learner
    # was handed named an act that did not produce it.
    #
    # LIVE 2026-09-04 on the real board: four keys a cycle, nine
    # comparisons kept, and the best hypothesis right once. The board was
    # being read perfectly the whole time; the learner was being told the
    # wrong question.
    expected["took"] = len(follow_on) + 1
    expected["prefixes"] = []
    expected["confidence"] = float(knows.rules.confidence()) if knows.rules is not None else None
    if foresee is not None and pending["arranged"] is not None:
        # Folded over the whole run, because a claim about one act is not
        # a claim about the board she will actually be looking at.
        where: Any = pending["arranged"]
        for step in (key, *follow_on):
            where = foresee(where, step)
            if where is None:
                break
            expected["prefixes"].append(where)
        expected["after"] = where
    logger.info(
        "about to press %r then %s (brief=%s, made=%s)",
        key,
        follow_on,
        pacing["brief"],
        made is not None,
    )

    async def act() -> bool:
        """Lifted to screen_pursuit_acting.py; the scope is handed over per call."""
        return await carry_out_the_move(
            SimpleNamespace(
                about_to=about_to,
                move_said=move_said,
                anchor=anchor,
                at_rest=at_rest,
                busy=busy,
                expected=expected,
                follow_on=follow_on,
                goal=goal,
                in_flight=in_flight,
                key=key,
                laid_out=laid_out,
                lattice=lattice,
                made=made,
                moves=moves,
                narrate=narrate,
                pacing=pacing,
                pending=pending,
                responds=responds,
                target_app=target_app,
                world=world,
            ),
        )

    return Step(name=f"press {key}", action=act)


# One sentinel, defined beside the branches that return it. Two would be
# two distinct object()s, and `is _FALL_THROUGH` would never match across
# the split — a silent change of meaning, which is what a move must not do.
from core.skills.screen_pursuit_decision_branches import (  # noqa: E402
    _FALL_THROUGH,
    ONLY_SILENCE,
    _a_move_here,
    _decide_the_next_move_act_has_done,
    _decide_the_next_move_blocker,
    _decide_the_next_move_learned_same_measurement,
    _decide_the_next_move_nothing_task_working,
    _decide_the_next_move_part_12,
    _decide_the_next_move_part_14,
    _decide_the_next_move_what_she_what,
    _decide_the_next_move_where_move_she,
    _decide_the_next_move_while_there_something,
    _how_long_to_think,
    _it_did_nothing_from_here,
    _leaving_out_what_just_did_nothing,
    _thought_over_beside_the_play,
    _what_she_says_as_she_moves,
    _what_the_thought_came_to,
)
