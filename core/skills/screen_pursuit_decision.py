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
    _ask_again_after,
    _both_of_the_thing,
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
    _she_got_further,
    _somewhere_else,
    _the_biggest_thing_on_it,
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
    a_way_back_that_was_not_there,
    am_i_there,
    let_the_voice_catch_up,
    narration_backlog,
    pacing_options,
    restart_controls,
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
    from core.cognition.two_ways_out import how_long_it_holds
    from core.cognition.what_nobody_could_show import WhatIsHidden
    from core.cognition.what_she_cannot_afford_to_lose import what_she_cannot_afford_to_lose
    from core.cognition.when_the_move_is_forbidden import a_way_round
    from core.cognition.when_to_say_it_outright import whether_to_say_it
    from core.perception.how_it_moves import HowItMoves
    from core.perception.the_lattice_she_holds import TheLatticeSheHolds
    from core.perception.what_moves_within_itself import MovesWithinItself
    from core.perception.what_the_world_does import WhatTheWorldDoes
    from core.perception.where_it_responds import (
        noticed,
        places_and_text,
        the_places_that_answer,
        within,
    )
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
        MAX_BLOCKER_ATTEMPTS,
        MAX_RELEARNS,
        RECENT_ATTEMPTS,
        _frontmost,
        _tell,
        _whats_on_top,
        logger,
    )

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
    if in_front and in_front != in_the_way["last"]:
        in_the_way["last"] = in_front
        if await clear_what_is_in_front(in_front):
            in_the_way["last"] = ""
            if narrate:
                _tell(f"{in_front} was in front of this. Closed it.")
            no_move["because"] = "a blocker was cleared, so this cycle is spent"
            return None

    blocker = await clear_blocker(observation)
    if blocker is not None:
        # Verified, not assumed. A blocker still present after the previous
        # attempt means that attempt did not work, whatever its receipt
        # said.
        if blocker_attempts["count"] >= MAX_BLOCKER_ATTEMPTS:
            blocker_attempts["last"] = blocker.name
            no_move["because"] = "something is in front of it that will not move"
            return None
        blocker_attempts["count"] += 1
        blocker_attempts["dismissed"] += 1
        blocker_attempts["last"] = blocker.name
        return blocker
    if needs_person["reason"]:
        no_move["because"] = "declining what is in front of it"
        return None
    blocker_attempts["count"] = 0
    if not observation.get("ok"):
        # What the reading actually said went wrong.
        #
        # Every failed read was reported as something being in front of
        # the thing and waited out. A read that timed out on a busy
        # machine, a capture that errored, a window that had gone — all
        # of them came back as an occlusion, which is a diagnosis of a
        # cause nobody had established, and the answer to it is to wait,
        # so she waited. Live 2026-09-07: three of those in a row ended
        # the run as "no move available" after seventeen moves, with
        # nothing on screen in front of anything.
        went_wrong = str(observation.get("error") or "").strip()
        no_move["because"] = (
            f"the last reading did not come back: {went_wrong}"
            if went_wrong
            else "the last reading did not come back, and did not say why"
        )
        logger.info("no move this cycle: %s", no_move["because"])
        return None

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
    looking_at_the_thing = (already or band is not None) and _was_of_that_window(
        observation, target_app or anchor["app"]
    )
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
        await _put_her_own_window_away()
        if target_app:
            await _bring_the_thing_back_to_the_front(target_app)
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
            if responds["lattice"].built_from(
                itself,
                moving.acts,
                # A grid worked out from what moves cannot be believed
                # until she has moved every way she can.
                tried=responds["state"].tried,
                available=move_keys,
            ):
                now = (responds["lattice"].rows, responds["lattice"].columns)
                if was != now and knows.rules is not None:
                    # Everything counted while the grid was the wrong
                    # shape was counted about a thing that does not exist.
                    knows.rules.learned_through_a_different_reading()
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
    if not confirmed_here["value"]:
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
            await _put_her_own_window_away()
            await _bring_the_thing_back_to_the_front(anchor["app"] or target_app)
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
            if previous.chosen is not None:
                was = pending["arranged"]
                kind = was.as_shape() if hasattr(was, "as_shape") else ""
                if kind:
                    beats.it_went(
                        previous.chosen.name,
                        against=kind,
                        well=bool(attempt.progressed),
                    )
                    repeats.she_saw(kind, previous.chosen.name, laid_out)
        # Was the world where she said it would be? That is what decides
        # how far she goes next time, and it is the only thing that does.
        in_flight.it_landed(previous.chosen.name if previous.chosen else "")
        if expected["took"] >= 1 and expected["after"] is not None:
            # Came out, meaning nothing she predicted is missing. Not
            # meaning identical: in a world that deals a tile after every
            # move of hers, the board she predicted is never the board
            # that is there, and asking for identical is asking for a run
            # that can never come out. What matters is whether her acts
            # did what she thought — an arrival she never claimed to know
            # about is the world's business, and she has somewhere else
            # to put those.
            said = {
                (one.row, one.column): one.says
                for one in getattr(expected["after"], "cells", ())
            }
            really = {
                (one.row, one.column): one.says
                for one in getattr(laid_out, "cells", ())
            }
            same = bool(said) and all(
                really.get(where) == what for where, what in said.items()
            )
            (far.it_was_where_she_said if same else far.it_was_not)(
                expected["took"]
            )
            if not same:
                logger.info("the run did not come out: %s", far.describe())
            expected["after"], expected["took"] = None, 0
        # What leaning on these acts has come to. The tally she found on
        # the screen herself is the measure where there is one, because
        # nobody has told her what progress is and something on the screen
        # has been keeping score the whole time.
        rose = _how_much_the_tally_moved(
            responds["moving"], pending["watched"], observation
        )
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
        if previous.chosen is not None:
            # A key that never changes anything is not one of her actions
            # in this world, whoever wrote it down.
            can_do.tried(previous.chosen.name, attempt.verdict.observed_change)
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
            if can_do.dead() and not foreseen.get("acts"):
                foreseen["acts"] = True
                logger.info("what works here: %s", can_do.says())
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
        if (
            pending["arranged"] is not None
            and previous.chosen is not None
            and looking_at_the_thing
            and not responds["state"].nothing_answers()
        ):
            # What a rule said would happen, before it is folded in. The
            # difference between that and what she actually saw is the
            # world's doing, and it is free at exactly this moment.
            foretold = knows.rules.expect(pending["arranged"], previous.chosen.name)
            world.watched(foretold, knows.rules.the_thing(laid_out))
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
            if plan["held"] is not None:
                lines.learned(A_LINE_HERE, plan["held"].approach, went_well)
                lines_held[plan["held"].approach] = plan["held"].as_memory()
            # What she was in, what she made of it, and what it came to.
            #
            # Two situations she scores alike, one of which went on to do
            # much better, is a difference her measure cannot account for —
            # and the only honest place a property nobody wrote can come
            # from. Gathered here because this is where both halves exist.
            from core.agency.how_good_is_this import (
                how_good as _how_good,
            )
            from core.agency.how_good_is_this import (
                how_the_trial_is_going as _how_the_trial_is_going,
            )

            _worth_here = sum(laid_out.numbers() or (0.0,))
            if began_at["worth"] is None:
                began_at["worth"] = _worth_here
            began_at["seen"] += 1
            _for = success_when or _what_there_is_to_aim_at(laid_out)
            if pending["arranged"] is not None and _for:
                cannot_explain.been_here(
                    pending["arranged"],
                    _how_good(
                        pending["arranged"],
                        toward=_for,
                        approach=plan["held"].approach if plan["held"] is not None else "",
                    ),
                    _worth_here,
                )
            if trying["name"]:
                verdict = _how_the_trial_is_going(trying["name"], _worth_here)
                if verdict:
                    if narrate:
                        _tell(
                            f"{trying['name']} {'earned its place' if verdict == 'kept' else 'did not earn its place'}."
                        )
                    trying["name"] = ""
            skilled.learned(
                pending["arranged"].as_shape(),
                previous.chosen.name,
                _left_her_better_off(
                    pending["arranged"],
                    laid_out,
                    success_when,
                    plan["held"].approach if plan["held"] is not None else "",
                ),
            )
            # And when she has just built the biggest thing she has ever
            # built here, which is the thing somebody watching came for.
            # Only once she knows what the thing itself is. Before the
            # grid settles the reading is the whole window, and the
            # largest number in it is somebody's best score: LIVE
            # 2026-09-04, "I have a 5292 on the board" on a board holding
            # a 4 and two 2s.
            # And not from a reading taken after it stopped answering.
            #
            # What is on the screen then is an ending — an overlay, a
            # score, a way to begin again — and its words land in the
            # board's places like anything else. LIVE 2026-09-04: "A
            # 11619 — the biggest I have made here", said over a finished
            # game whose best tile was 128.
            made = (
                _the_biggest_thing_on_it(
                    laid_out,
                    _placed_in(
                        responds["lattice"],
                        responds["moving"].the_things_that_report(),
                    ),
                )
                if responds["lattice"].held
                and responds["moving"].settled()
                and not responds["state"].nothing_answers()
                else 0.0
            )
            # A record she has not been able to earn again is not one.
            #
            # It is written from a reading, and a reading can be wrong: a
            # single bad one put 11619 in this record for a board whose
            # best tile was 128, and because the record only ever goes up
            # she could never say anything about her own progress here
            # again. Everything else she carries comes back discounted for
            # exactly this reason.
            #
            # So a carried record has to turn up again before it counts,
            # which is the same rule the places she remembers are held to.
            furthest["again"] = max(furthest["again"], made)
            beaten = (
                furthest["here"] if furthest["again"] >= furthest["here"] else furthest["again"]
            )
            further = _she_got_further(made, beaten)
            if further:
                furthest["here"] = max(furthest["here"], made)
                if narrate:
                    _tell(further)
            _say_what_she_worked_out(knows, foreseen)
            _say_what_kind_of_problem(
                knows, screen_options(move_keys), laid_out, success_when, foreseen
            )
            if len(moves) % 6 == 0 and knows.rules is not None:
                logger.info(
                    "after %d move(s): %s%s | reading %dx%d%s",
                    len(moves),
                    knows.rules.says(),
                    _what_she_is_not_reading(knows.rules),
                    laid_out.rows,
                    laid_out.columns,
                    _what_she_could_not_learn_from(dropped),
                )
        # Learned from the same measurement. A move that changed nothing
        # is the control: whatever still changed across it was changing
        # on its own, and a page whose advertising animates as often as
        # the task does cannot be separated any other way.
        if pending["watched"] and _both_of_the_thing(
            target_app or anchor["app"], pending["watched"], observation
        ):
            # The same places `noticed` uses, so the two sets can be
            # intersected at all.
            responds["moving"].saw(
                places_and_text(pending["watched"]),
                places_and_text(observation),
            )
            noticed(
                responds["state"],
                pending["watched"],
                observation,
                # Whether the act had an effect, not whether her claim
                # about it was right.
                #
                # These were the same answer while the only claim a move
                # carried was that the view would differ, and they came
                # apart the moment a claim could say something. A move
                # that moved the board and did not do the specific thing
                # she predicted was being counted as a move that did
                # nothing — the control for working out which part of the
                # screen answers to her — so the band stopped settling and
                # nothing downstream of it could form.
                worked=attempt.verdict.observed_change,
                # Which act it was, so a run of one thing doing nothing
                # is not read as the world having ended.
                acting=previous.chosen.name if previous.chosen is not None else "",
            )
        if moves:
            moves[-1]["held"] = attempt.verdict.held
            moves[-1]["outcome"] = attempt.verdict.why()
        pending["deliberation"] = None

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
        if knowledge["held"] is None or (
            stuck(history) and knowledge["relearned"] < MAX_RELEARNS
        ):
            if knowledge["held"] is not None:
                knowledge["relearned"] += 1
            relearning = knowledge["held"] is not None
            knowledge["meant"] = []
            knowledge["held"] = await learn_about(
                goal,
                search=research,
                remember=not relearning,
                because_stuck=relearning,
                situation=content_text(
                    observation, region_top=region_top, region_bottom=region_bottom
                ),
                history=history[-RECENT_ATTEMPTS:],
            )
        learned = knowledge["held"].as_evidence() if knowledge["held"] is not None else []
        # Work out what it means HERE before deciding with it.
        #
        # Retrieving advice is not applying it. "Keep your largest tile in
        # a corner" is a fact about the game; what it means depends on
        # where the tiles actually are, and that comparison is the step
        # between reading something and playing differently.
        if knowledge["held"] is not None and knowledge["held"].known and not knowledge["meant"]:
            knowledge["meant"] = await work_out_what_it_means(
                knowledge["held"],
                seen,
                screen_options(move_keys),
                think=_within_the_run(think or _her_reasoning(stakes), ends_at),
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
        if plan["held"] is not None and holding is False:
            logger.info("the line she was taking stopped holding: %s", ended)
        # A pivot is immediate; a first attempt is not retried every move.
        #
        # The condition breaking is news and is worth the pass that
        # answers it. Having no stated approach yet is not news, and a run
        # that asks for one every cycle pays a full language pass per move
        # for an answer that was not there last time either.
        # Asked when there is something to base an approach on.
        #
        # This asked on the first cycle, when the only thing she has seen
        # is a whole screen: on a page holding a game that is the board,
        # the score, the browser's own tabs and bookmarks, an "Ask Gemini"
        # button and a copyright line. Live 2026-08-26, she was asked how
        # she would play and answered by reading the page back, three
        # runs in a row, because that is what the question was about.
        #
        # The part of the screen that answers to her is known a few moves
        # in, from what changed when she acted. That is the first moment
        # the question has a subject. The count is still a backstop, so a
        # screen that never resolves into anything is not a screen she
        # goes on playing with no line at all.
        # A pivot costs a move: she cannot judge a line she never tried.
        #
        # The condition on a fresh approach is checked on the very next
        # cycle, before she has acted under it, and an anchor bound to a
        # tile that merges away breaks at once. Live 2026-08-26: ten
        # approaches decided for nine moves made, each one a full pass at
        # her reasoning, and the run spent its whole budget deciding how
        # to play rather than playing.
        tried_it = len(moves) > plan["asked_at"]
        time_to_ask = (
            holding is False
            and plan["held"] is not None
            and tried_it
            or len(moves) - plan["asked_at"] >= _ask_again_after(plan["asked_at"])
            or (plan["asked_at"] < 0 and looking_at_the_thing)
        )
        if not holding and time_to_ask:
            plan["asked_at"] = len(moves)
            fresh = await settle_on_an_approach(
                goal,
                seen,
                screen_options(move_keys),
                # Deciding the line she will hold across a hundred moves
                # is not the same question as deciding one of them, and
                # asking it with the thinking that suits a move got the
                # model's own warm-up handed back as a plan.
                think=_within_the_run(think or _reasoning_for_a_plan(), ends_at),
                knowledge=learned,
                history=history[-RECENT_ATTEMPTS:],
                previous=plan["held"],
                moves_made=len(moves),
            )
            if fresh is not None:
                changing = plan["held"] is not None
                plan["held"] = fresh
                plan["changes"] += 1 if changing else 0
                # Held where the rest of her can see it, not in this loop.
                doing.going_about_it(
                    fresh.approach,
                    because=fresh.because,
                    watching_for=fresh.holds_while.describes,
                    alternatives=fresh.otherwise,
                    spine=spine,
                    lived=lived,
                )
                if narrate:
                    said = fresh.narrate()
                    _tell(f"{said} ({ended})" if changing and ended else said)
        if plan["held"] is not None:
            learned = learned + plan["held"].as_evidence()

        # When nothing in the task is working, the task itself becomes a
        # choice. Both ways out are hers, and both are recorded as
        # decisions with reasons rather than happening to her.
        available = screen_options(can_do.available() or move_keys)
        # The ways out are offered when what she is doing has stopped
        # working, and when the thing itself has stopped responding.
        #
        # Two different facts. Predictions breaking says her moves are
        # wrong; nothing answering at all says the attempt is over — a
        # finished game, an expired session, a form already submitted.
        # Measured live: she played to Game Over and went on pressing
        # arrow keys into a dead board, because a run of broken
        # predictions had not accumulated in the way the first test
        # wanted.
        # Nothing is answering. WHICH of the three is it?
        #
        # "Nothing I do changes anything" is a good ending test and a poor
        # diagnosis: it is equally true of a finished game, a dialog over
        # the board, and somebody else's window in front. Those want
        # opposite responses, and collapsing them is what had her pressing
        # keys into a finished board and narrating moves as though a game
        # were happening. LIVE 2026-08-29: the page said "Game Over, 940
        # points scored in 100 moves" and she went on saying "Going right".
        ended = responds["state"].nothing_answers()
        # A way to start again, offered where there was none before, is the
        # thing saying it has finished.
        #
        # The other test asks whether what she was acting on is gone, and
        # on a board that keeps its tiles under a "Play Again" overlay it
        # never fires — so she went on pressing keys into a game that was
        # over, which is the failure it exists to prevent. A control that
        # appears only at the end is better evidence than the absence of
        # one, and it is general: a finished form, an expired session and
        # a lost game all put one up.
        ways_back = restart_controls(observation)
        appeared = a_way_back_that_was_not_there(
            ways_back, offered_a_restart["was_there"]
        )
        if offered_a_restart["was_there"] is None:
            # The first reading of this task settles what counts as
            # furniture here. Nothing before this line has been compared
            # against, so nothing before it can have appeared.
            offered_a_restart["was_there"] = ways_back
        if not ended and appeared:
            if not offered_a_restart["said"]:
                offered_a_restart["said"] = True
                logger.info(
                    "a way to start again has appeared, so this has ended"
                )
            ended = True
        if ended:
            mine_now = target_app or anchor["app"]
            why = work_out_why(
                mine=mine_now,
                in_front=await _frontmost(),
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
            elif why.because == ENDED and narrate and not said_it_ended["value"]:
                said_it_ended["value"] = True
                _tell(why.says())
        if (stuck(history) or ended) and not seen_through["value"]:
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
        wont, ruled_out = _moves_she_will_not_make(
            she_keeps,
            went,
            laid_out,
            [option.name for option in available],
            knows,
            len(moves),
            world,
        )
        if wont:
            available = [one for one in available if one.name not in wont]
            if ruled_out != she_keeps.get("said"):
                she_keeps["said"] = ruled_out
                logger.info("%s", ruled_out)

        # And an act that has done nothing since the last time anything
        # happened is not the act to take again.
        #
        # "It has worked here before" is a lifetime record; the board in
        # front of her is now. LIVE 2026-09-04: fifteen presses of up
        # into a board with nothing above anything, each one chosen
        # because up had worked earlier in the same game.
        #
        # The record clears the moment anything answers, so this is the
        # freshest evidence there is about the next move — and it never
        # empties the choice, because every act failing is the thing
        # having ended, which is judged elsewhere.
        doing_nothing = responds["state"].unanswered_by
        if doing_nothing:
            still_worth = [
                one for one in available if one.name not in doing_nothing
            ]
            if still_worth:
                available = still_worth

        # And of what is left, the one she reaches for.
        #
        # Watching somebody clear 2048: two of the four keys, almost
        # exclusively, and a third only when the board left them nothing
        # else. Measured on the game, leaning on two of the four reaches
        # more than twice what taking any legal move reaches, and beats
        # every property of the board she could have held instead. It is
        # not a fact about the board, so looking at the board never finds
        # it; it is found by leaning on things and seeing what came of it.
        foreseeable = [
            one.name
            for one in available
            if _somewhere_else(laid_out, one.name, _expects(knows)) is not None
        ] if laid_out is not None and _expects(knows) is not None else []
        if foreseeable:
            if not reaches.leaning_on:
                took_up = reaches.start_a_stretch(foreseeable)
                if took_up:
                    logger.info("leaning on %s for a while", ", ".join(took_up))
            elif stretch["rises"] >= len(reaches.ways_of_leaning(foreseeable)):
                reaches.end_the_stretch()
                stretch["rises"] = 0
                reaches.start_a_stretch(foreseeable)
            # The ones she is leaning on, not one of them.
            #
            # A habit that picks the member as well as the set leaves
            # nothing for looking ahead to do: the choice arrived at the
            # deliberation with one option in it, every move came back
            # "the only thing available", and the search she has never
            # ran. Leaning on two acts means those two are the moves she
            # considers, and which the position calls for is the question
            # the model answers.
            wants = reaches.the_ones_to_consider(foreseeable)
            # And it has to leave her something to consider.
            #
            # A leaning of one act, applied as a filter, is not a
            # preference — it is the whole policy, and it takes her
            # looking ahead off the board altogether: the deliberation
            # arrives with one option and reports "the only thing
            # available". Her search is worth far more than any leaning
            # (offline, median 512 against 128 for taking any legal move),
            # so a habit must never be able to replace it.
            if len(wants) < 2:
                wants = ()
            if wants and len(wants) < len(foreseeable):
                # Everything she cannot foresee stays on the table: the way
                # out and the ways of asking are never narrowed by a habit.
                available = [
                    one
                    for one in available
                    if one.name in wants or one.name not in foreseeable
                ]

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
        if worth_comparing(aiming_at, held_line):
            # As far ahead as there is time to look, which is decided from
            # what a level of looking has been measured costing.
            ahead = look_ahead(
                knows.rules,
                laid_out,
                [option.name for option in available],
                toward=aiming_at,
                approach=held_line,
                budget_s=max(0.05, min(2.0, (ends_at - time.monotonic()) * 0.02)),
                world=world,
                # What matters HERE, once she has watched enough to say.
                weights=matters.weights(),
            )
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

        # And where the move she wants is not one she may make, something
        # elsewhere that obliges the world to let her.
        if wont and ahead:
            blocked = [one for one in ahead if one in wont]
            if blocked:
                round_it = a_way_round(
                    blocked[0],
                    allowed=lambda one: one not in wont,
                    elsewhere=[option.name for option in available],
                    they_must_answer=lambda one: float(ahead.get(one, (0.0, ""))[0]),
                    after_they_answer=lambda one: None,
                    worth_of_the_fight=0.0,
                )
                if round_it.found and round_it.spend_a_turn_on in ahead:
                    logger.info("cannot take %s — %s", blocked[0], round_it.describe())

        # And a routine move in a fast loop does not always need words.
        #
        # What a thought is worth here, rather than how long since the
        # last one. A counter cannot tell a forced move from the one that
        # decides the shape of the next thirty, so it spends the same on
        # both and is wrong about both.
        # What has worked before from a position of this kind, if anything.
        #
        # Recognition is what frees her from deciding. Where it disagrees
        # with the arithmetic, that disagreement is the surest sign this
        # position is not the routine one it looked like, and it buys a
        # thought rather than saving one.
        kind = laid_out.as_shape() if laid_out is not None else ""
        # A mark where she has been, so a place is recognised rather than
        # recalled — and so the way back is on the ground.
        if kind:
            marks.she_marked(kind, saying=aiming_at or goal)
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
            since_words=len(moves) - asked["at"],
            horizon=LANGUAGE_EVERY,
            unusual=unusual or not moves or restarts["count"] > asked["after_restarts"],
            recognised=recognised,
            # How far she can trust her own arithmetic here, which is how
            # often the rule she is using has been right about this world.
            how_sure=(
                knows.rules.confidence() if knows.rules is not None else 0.0
            ),
            # What a pass costs, in moves not made, from this run's own
            # clock. Live on a resident model it was about ten.
            costs_moves=_a_pass_in_moves(costs),
        )
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
            think=_within_the_run(think or _her_reasoning(weight), ends_at) if asking else None,
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
        else:
            costs["was_quiet"] = 1.0
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
            # Not while there is something here she cannot get back.
            #
            # Starting over is the one act of hers that destroys what she
            # has made. Everywhere else a bad move costs a move; here it
            # costs the whole thing, and she reaches for it exactly when
            # she is stuck — which is also when a position is at its most
            # developed and worth the most.
            #
            # What is precious is not declared. Take a part of the thing
            # away and ask whether what she is holding survives without
            # it: a board's largest tile is what "the largest thing is at
            # the far end" rests on, so losing it is losing the plan, and
            # a board of small ones costs nothing to leave.
            keeping = she_keeps.get("it")
            if keeping is not None and laid_out is not None and available:
                precious = what_she_cannot_afford_to_lose(
                    laid_out,
                    holding=keeping.holds,
                    parts_of=lambda one: list(getattr(one, "cells", ())),
                    without=_the_same_thing_without,
                )
                others = [
                    one.name for one in available if one.name != START_OVER
                ]
                if precious and others:
                    # And how sure she would have to be, given what it
                    # costs to be wrong. Starting over destroys what is
                    # here; another move costs a move. Those are different
                    # sizes, so the certainty needed is not a level — it
                    # is the comparison.
                    say_it = whether_to_say_it(
                        how_sure=float(chosen.confidence),
                        being_wrong_costs=float(len(precious)),
                        another_look_costs=1.0,
                        waiting_might_lose_it=(
                            1.0 if responds["state"].nothing_answers() else 0.0
                        ),
                        what_it_is_worth=float(len(precious)),
                    )
                    if not say_it.now:
                        logger.info(
                            "not starting over: %d thing(s) she cannot get back "
                            "(%s)",
                            len(precious),
                            say_it.describe(),
                        )
                        no_move["because"] = (
                            "there is something here worth keeping"
                        )
                        return None
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
            budget_s=0.05, world=world, weights=matters.weights(),
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
    if not follow_on and foresee is not None and pending["arranged"] is not None:
        going = far.how_many(trusted=float(knows.rules.confidence()))
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
    if foresee is not None and pending["arranged"] is not None:
        # Folded over the whole run, because a claim about one act is not
        # a claim about the board she will actually be looking at.
        where: Any = pending["arranged"]
        for step in (key, *follow_on):
            where = foresee(where, step)
            if where is None:
                break
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
