"""The branches of one decision, moved out of the module that holds it.

``decide_the_next_move`` is 1,326 lines and the method-size sweep lifted
its self-contained runs into helpers beside it. That pushed
``screen_pursuit_decision`` to 2,436 lines, past the 2,000-line ceiling
``tests/test_a_god_object_only_shrinks.py`` says a module NOT already in
the baseline never gets grandfathered past.

These eleven came out whole. They take nothing from the module they left
but the fall-through sentinel, which came with them, and only
``decide_the_next_move`` calls them — so this is a move, not a rewrite.
"""

from __future__ import annotations

from typing import (
    Any,
)

from core.agency.how_good_is_this import (
    terms,
)
from core.perception.where_it_responds import (
    noticed,
    places_and_text,
)
from core.skills.screen_pursuit_bearings import (
    START_OVER,
    _ask_again_after,
    _both_of_the_thing,
    _expects,
    _left_her_better_off,
    _she_got_further,
    _somewhere_else,
    _the_biggest_thing_on_it,
    _the_same_thing_without,
    _what_there_is_to_aim_at,
    a_way_back_that_was_not_there,
    restart_controls,
    screen_options,
)
from core.skills.screen_pursuit_looking import (
    _placed_in,
)
from core.skills.screen_pursuit_surface import (
    _a_pass_in_moves,
)

#: Returned by a branch that declines to decide, so the next one is asked.
_FALL_THROUGH = object()


def _decide_the_next_move_what_she_what(
    began_at: Any,
    cannot_explain: Any,
    laid_out: Any,
    narrate: Any,
    pending: Any,
    plan: Any,
    success_when: Any,
    trying: Any,
) -> None:
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

    from .screen_pursuit import _tell

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

def _decide_the_next_move_part_12(
    furthest: Any,
    laid_out: Any,
    pending: Any,
    plan: Any,
    previous: Any,
    responds: Any,
    skilled: Any,
    success_when: Any,
    single_action: bool=True,
) -> tuple[Any, Any]:
    if single_action:
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
    return further, made

def _decide_the_next_move_learned_same_measurement(
    anchor: Any,
    attempt: Any,
    observation: dict[str, Any],
    pending: Any,
    previous: Any,
    responds: Any,
    target_app: Any,
) -> None:
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

def _decide_the_next_move_part_14(
    ended: Any,
    holding: bool,
    looking_at_the_thing: Any,
    moves: Any,
    plan: Any,
    costs: Any=None,
) -> bool:
    from .screen_pursuit import logger
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
    since = len(moves) - plan["asked_at"]
    # What a pass costs in moves not made, measured on this run. Asking every
    # few moves is right where a pass is nearly free and wrong where one costs
    # the time of ten moves: then it is most of what she does.
    dear = max(1.0, _a_pass_in_moves(costs or {}))
    # And each time the same line came back, twice as long before asking again.
    patience = 2 ** min(6, int(plan.get("same_again", 0)))
    time_to_ask = (
        holding is False
        and plan["held"] is not None
        and tried_it
        and since >= patience
        or since >= _ask_again_after(plan["asked_at"]) * dear * patience
        or (plan["asked_at"] < 0 and looking_at_the_thing)
    )
    return time_to_ask

def _a_move_here(run: Any) -> float:
    """How long a move in this world has been taking, from her own looks.

    What a thought may cost is a fact about the world she is thinking in: a
    board that answers in a fifth of a second does not wait half a minute for
    words about it.
    """
    looks = list(getattr(run, "reading_took", None) or [])
    return (sum(looks) / len(looks)) if looks else 0.0


def _decide_the_next_move_nothing_task_working(
    can_do: Any,
    move_keys: Any,
    observation: dict[str, Any],
    offered_a_restart: Any,
    responds: Any,
    knows: Any=None,
    laid_out: Any=None,
) -> tuple[Any, Any]:
    from .screen_pursuit import logger
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
    # And what her own model says. Where the rule she trusts says that none of
    # her acts would change anything, the thing has finished, and pressing
    # each of them to find that out is asking a question she has the answer
    # to. The other tests need every act to have failed in a row, which a
    # habit of using only some of them can keep from ever happening.
    if not ended and knows is not None and laid_out is not None and getattr(laid_out, "cells", None):
        rules = getattr(knows, "rules", None)
        # A rule that has never seen anything move says what she watched, not
        # how this world works. LIVE 2026-09-17: her first four keys went into
        # a board that had already finished, so what she composed was "this
        # does not move" — and that rule then said a freshly dealt board was
        # finished too. She began again, read a new board, was told it was
        # dead, began again, forty times in a minute. Where nothing has ever
        # moved she finds out by acting, which is what the other tests here
        # are for.
        has_seen_movement = int(getattr(rules, "moved", 0) or 0) > 0
        if rules is not None and has_seen_movement and rules.rule() is not None:
            names = [str(key) for key in (can_do.available() or move_keys)]
            futures = [rules.expect(laid_out, name) for name in names]
            if names and all(
                future is not None and future.as_text() == laid_out.as_text() for future in futures
            ):
                if not offered_a_restart.get("model_said"):
                    offered_a_restart["model_said"] = True
                    logger.info("none of her acts would change anything here, so this has ended")
                ended = True
    return available, ended

def _what_she_says_as_she_moves(
    key: str,
    laid_out: Any,
    knows: Any,
    ahead: dict[str, tuple[float, str]],
    *,
    weights: Any,
    toward: str,
    approach: str,
    biggest_so_far: float,
) -> str:
    """One line for the move she is about to make. Empty when there is no model to say it from."""
    from core.agency.how_good_is_this import AS_GOOD_A_GUESS_AS_ANY
    from core.agency.saying_what_a_move_does import what_a_move_does, why_this_one

    rules = getattr(knows, "rules", None)
    if rules is None or laid_out is None or rules.rule() is None:
        return ""
    try:
        after = rules.expect(laid_out, key)
    except (AttributeError, TypeError, ValueError):
        after = None
    if after is None:
        return ""
    because = ""
    others = [
        (value, name) for name, (value, _why) in (ahead or {}).items() if name != key
    ]
    if others:
        _value, runner_up = max(others)
        other_after = rules.expect(laid_out, runner_up)
        if other_after is not None:
            names = list(ahead)
            because = why_this_one(
                terms(after, toward=toward, approach=approach, knows=rules, acts=names),
                terms(other_after, toward=toward, approach=approach, knows=rules, acts=names),
                weights or AS_GOOD_A_GUESS_AS_ANY,
                runner_up_name=runner_up,
            )
    return what_a_move_does(laid_out, key, after, biggest_so_far=biggest_so_far, because=because)


def _decide_the_next_move_act_has_done(
    available: Any,
    knows: Any,
    laid_out: Any,
    reaches: Any,
    responds: Any,
    stretch: Any,
) -> Any:
    from .screen_pursuit import logger
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
        # And only while she cannot see past the next move. A leaning is what
        # stands in for foresight; once her look reaches further, narrowing
        # the moves it considers removes the one that would have saved the
        # position. LIVE 2026-09-17, through her whole loop on a board she
        # could search three deep: every decision between down and left,
        # never up or right, lost at a 128.
        from core.agency.looking_ahead import how_far_she_can_see

        if how_far_she_can_see() >= 2:
            wants = ()
        if wants and len(wants) < len(foreseeable):
            # Everything she cannot foresee stays on the table: the way
            # out and the ways of asking are never narrowed by a habit.
            available = [
                one
                for one in available
                if one.name in wants or one.name not in foreseeable
            ]
    return available

def _decide_the_next_move_where_move_she(
    ahead: Any,
    aiming_at: Any,
    available: Any,
    goal: Any,
    laid_out: Any,
    marks: Any,
    wont: Any,
) -> Any:
    from core.cognition.when_the_move_is_forbidden import a_way_round

    from .screen_pursuit import logger
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
    return kind

async def _decide_the_next_move_blocker(
    blocker_attempts: Any,
    clear_blocker: Any,
    needs_person: Any,
    no_move: Any,
    observation: dict[str, Any],
) -> Any:
    from .screen_pursuit import MAX_BLOCKER_ATTEMPTS, logger
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
    return _FALL_THROUGH

def _decide_the_next_move_while_there_something(
    available: Any,
    chosen: Any,
    laid_out: Any,
    no_move: Any,
    responds: Any,
    she_keeps: Any,
) -> Any:
    from core.cognition.what_she_cannot_afford_to_lose import what_she_cannot_afford_to_lose
    from core.cognition.when_to_say_it_outright import whether_to_say_it

    from .screen_pursuit import logger
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
    return _FALL_THROUGH
