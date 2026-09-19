"""The steps one on-screen pursuit is assembled from.

Lifted whole out of `screen_pursuit`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

from typing import Any


def _pursue_on_screen_where_her_actions(knew, knows, skilled, world):
    from core.perception.the_lattice_she_holds import TheLatticeSheHolds
    from core.perception.what_moves_within_itself import MovesWithinItself
    from core.perception.where_it_responds import Responsive

    from .screen_pursuit import (
        TRUST_CARRIED_OVER,
        TheOnesSheReachesFor,
    )
    #: Where her actions have been having their effects.
    responds: dict[str, Any] = {
        "state": Responsive.from_memory(knew.get("responds") or {}, TRUST_CARRIED_OVER),
        # Answering to her is not the same as being the thing. A score, a move
        # counter and a clock all answer to her, and in one frame they look
        # exactly like a board. What a board does that none of them do is show
        # her values that were already in it.
        "moving": MovesWithinItself.from_memory(
            knew.get("moves_within") or {}, TRUST_CARRIED_OVER
        ),
        # The grid itself, held between glances rather than worked out from
        # each one. Where a thing is tends to keep, so it is remembered.
        "lattice": TheLatticeSheHolds.from_memory(knew.get("lattice") or {}),
    }
    # And what she remembers about how the thing moves is dropped where it was
    # read through a grid of a different shape from the one she is holding.
    #
    # Dropping it DURING a run, when the grid corrects itself, was not enough.
    # The counts are written down at the end and read back at the start, so a
    # sitting that began with the wrong grid poisons every sitting after it,
    # and the correction never fires because by then the grid is already
    # right. LIVE 2026-09-02: three sittings running, the grid correct in the
    # last two, the rule stuck at two hundred and five right out of three
    # hundred and one, and not one move looked ahead on in any of them.
    #
    # Counts that cannot say what grid they were read through are dropped too.
    # Evidence that cannot say what it was gathered under cannot be known to
    # still apply, and this particular evidence decides whether she looks
    # ahead at all — a wrong one costs her every move of every sitting, and
    # relearning a right one costs about thirty moves of one.
    if knows.rules is not None and responds["lattice"].held:
        grid = (responds["lattice"].rows, responds["lattice"].columns)
        if knows.rules.read_through != grid:
            knows.rules.learned_through_a_different_reading()
            # And everything else that was read through the same grid. What
            # worked before is looked up by the SHAPE of the situation, and a
            # shape written down under a four by seven grid names nothing
            # under a four by four one. What the world does on its own is
            # learned by comparing two arrangements, so it is the same story:
            # one world file on this machine has it believing that "History"
            # and "Help" turn up on their own, which is the browser's menu bar
            # read as though it were the game.
            skilled.forget_what_was_read_differently()
            world.forget_what_was_read_differently()
    #: The few acts she reaches for, out of all the ones she could. A habit is
    #: worth carrying between sittings, so it is remembered too.
    reaches = TheOnesSheReachesFor.from_memory(
        knew.get("reaches") or {}, TRUST_CARRIED_OVER
    )
    return reaches, responds

async def _pursue_on_screen_part_2(already, anchor, began, first, fresh, intending, restarts, settle, target_app):
    from .screen_pursuit import (
        START_OVER,
        _answer_own_confirmation,
        click_normalized,
    )

    if settle.reached and settle.chosen is not None and settle.chosen.name == START_OVER:
        label, rx, ry = fresh
        intending["value"] = START_OVER
        frame = list(first.get("bounds") or [])
        if await click_normalized(rx, ry, expect_app=target_app or anchor["app"], bounds=frame):
            # Look again before judging anything, and before
            # claiming anything.
            #
            # A reset takes a moment to land, and the reading
            # taken before it did is of the game she just
            # abandoned. A click that lands on nothing still
            # reports success — the click happened — so waiting
            # is not enough on its own. Measured live: "Began
            # again 1 time(s)" while the score sat unchanged at
            # 996 the whole time.
            _after, began = await _answer_own_confirmation(
                first, target_app, label
            )
            if began:
                restarts["count"] += 1
                restarts["because"] = (
                    settle.rationale or "the goal was already met by an old game"
                )
            else:
                already["value"] = True
        else:
            already["value"] = True
    else:
        already["value"] = True
    return began

def _pursue_on_screen_she_has_taken(goal, target_app):
    from core.agency import what_she_is_doing as doing
    from core.skills.fluid_executor import FluidExecutor

    from .screen_pursuit import (
        record_degradation,
    )
    # She has taken something on, and the rest of her should know it.
    doing.taking_on(goal, where=target_app or "")
    executor = FluidExecutor(verifier=None, gateway=None)
    # And it holds the foreground while it runs.
    #
    # A task somebody asked for is foreground for as long as it takes, not for
    # the length of the sentence that started it. Her background thinking
    # already stands aside for a foreground turn — it checks — and a turn ends
    # the moment the reply is composed, so everything she does AFTER that,
    # which is the whole of the task, ran as background beside her own
    # research loops.
    #
    # Live 2026-09-07, playing a game on this machine: thirty-two decisions
    # about what to do next, nine of them refused outright because the
    # inference lanes were exhausted — by her own reimplementation lab and
    # curriculum loop, running against the model she needed to choose a move.
    holding_the_foreground = None
    # And the display stays awake while she works on it.
    #
    # She types with keys addressed to a process and looks with a picture of
    # a window, so from the outside the machine looks untouched and locks
    # itself. LIVE 2026-09-17: the screen locked on move 226 of a game she
    # had been asked to play and was winning.
    try:
        from core.capabilities.keeping_the_screen_awake import (  # noqa: PLC0415
            keeping_it_awake,
        )

        awake = keeping_it_awake(f"she is working on {target_app or 'the screen'}")
        awake.__enter__()
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        awake = None
        record_degradation(
            "screen_pursuit", exc, severity="info",
            action="worked on a screen that may sleep under her",
        )
    try:
        from core.runtime.foreground_guard import begin_foreground_turn  # noqa: PLC0415

        holding_the_foreground = begin_foreground_turn(
            owner="screen_pursuit", source="desktop_task"
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "screen_pursuit", exc, severity="info",
            action="pursued a task without holding the foreground",
        )
    return executor, holding_the_foreground, awake

def _pursue_on_screen_result(already, blocker_attempts, history, moves, pacing, receipt, restarts, seen_through, success_when):
    from .screen_pursuit import (
        MAX_BLOCKER_ATTEMPTS,
    )

    result = receipt.to_dict()
    if blocker_attempts["count"] >= MAX_BLOCKER_ATTEMPTS and not receipt.completed:
        # Say what stopped it. "out_of_cycles" describes the budget running
        # out; this describes the reason, which is the part a caller can act
        # on — by dismissing it themselves, or by deciding the dialog is the
        # task.
        result["outcome"] = "blocked_by_overlay"
        result["blocked_by"] = blocker_attempts["last"]
    result["moves"] = moves
    if already["value"]:
        # Said plainly rather than claimed. The person asked her to do
        # something and the condition was true before she started, so what
        # they get is that fact and not a receipt for work nobody did.
        result["already_true_at_the_start"] = success_when
        result["outcome"] = "already_true"
        result["completed"] = False
    result["restarts"] = restarts["count"]
    if pacing["choice"] or pacing["waits"]:
        result["pacing"] = {
            "chose": pacing["choice"],
            "because": pacing["because"],
            "waited": pacing["waits"],
        }
    if restarts["because"]:
        result["restarted_because"] = restarts["because"]
    if seen_through["value"]:
        result["played_out_because"] = seen_through["because"]
    result["attempts"] = [
        {"option": a.option, "expected": a.expected, "held": a.verdict.held, "why": a.verdict.why()}
        for a in history
    ]
    return result

def _pursue_on_screen_part_5(began_at, cannot_explain, endings, narrate, pending, plan, result, trying):
    from .screen_pursuit import (
        WORTH_TRYING_AT,
        _tell,
        logger,
        which_way_to_win,
    )

    if plan["changes"]:
        result["changed_approach"] = plan["changes"]
    if plan["held"] is not None:
        result["approach"] = plan["held"].approach
    # A property nobody wrote, if what she could not explain calls for one.
    #
    # At the end, because the outcome of a situation is what came after it and
    # that is only known once the run is over. Put on TRIAL rather than
    # adopted: she cannot replay a life to A/B a change to her own judgement,
    # so the trial happens in the next one and the property keeps its place
    # only if things actually go better with it.
    if not trying["name"]:
        from core.agency.how_good_is_this import on_trial, what_it_was_like_before

        ended_at = pending["arranged"]
        finished_on = sum(ended_at.numbers() or (0.0,)) if ended_at is not None else 0.0
        # The rate this run moved at, which is what the trial will produce.
        # A baseline measured as a total and a trial measured as a rate
        # compares nothing.
        how_it_went = (
            (finished_on - float(began_at["worth"])) / max(1, began_at["seen"])
            if began_at["worth"] is not None
            else 0.0
        )
        for measure, held_back, _pairs in cannot_explain.worth_trying(most=1):
            name = on_trial(measure, WORTH_TRYING_AT)
            if name:
                what_it_was_like_before(name, how_it_went)
                trying["name"] = name
                if narrate:
                    _tell(
                        f"There is something my measure could not account for. "
                        f"I am going to try judging by {name!r} and see whether "
                        f"it does better."
                    )
                logger.info(
                    "put %r on trial — it accounted for %.0f%% of what she could not",
                    name, held_back * 100.0,
                )
            break

    # Which way of finishing she has a route to.
    #
    # Somebody beating a hard checkers engine chose a way of winning that
    # gives pieces away, and said outright it was not the best strategy — it
    # was the one they could reach against that opponent. Most things that
    # play a game never make that decision: they score a position and take the
    # best number, which quietly fixes the ending as whichever the score was
    # built around.
    if len(endings) > 1:
        ways = {
            one: (lambda finish, want=one: finish == want)
            for one in {ended for _shapes, ended in endings if ended}
        }
        if len(ways) > 1:
            for way in which_way_to_win(ways, endings)[:1]:
                logger.info("the ending she has a route to: %s", way.describe())

def _pursue_on_screen_part_6(cannot_see, moves, no_move, not_there, receipt, result, success_when, undecided):
    from .screen_pursuit import (
        logger,
    )

    if cannot_see["reason"] and not receipt.completed and not moves:
        # Named apart from every other ending. She was never able to look, so
        # nothing about the screen's contents is being reported and no amount
        # of acting differently would have helped.
        result["outcome"] = "cannot_see"
        result["cannot_see"] = cannot_see["reason"]
    if not_there["reason"] and not receipt.completed:
        # Named apart from every other way a run can end. "Nothing offered a
        # move" would say the screen had nothing on it; this says she was
        # never in the thing she was asked to act in, which is a different
        # failure with a different fix.
        result["outcome"] = "could_not_get_there"
        result["could_not_get_there"] = not_there["reason"]
    if no_move["because"] and not receipt.completed:
        # Whatever else the run managed. A stall after four moves has a cause
        # as much as a stall after none, and gating this on "no moves at all"
        # meant the one run that mattered stayed silent.
        result["why_no_move"] = no_move["because"]
        logger.info(
            "the last cycle made no move: %s (after %d move(s))",
            no_move["because"],
            len(moves),
        )
    if undecided["reason"] and not receipt.completed:
        # Name the judgement, not the budget. "no_move_available" would say
        # the screen offered nothing; this says she could not decide, and why.
        result["outcome"] = "cannot_decide"
        result["cannot_decide"] = undecided["reason"]
    result["success_when"] = success_when

def _pursue_on_screen_part_7(anchor, expect_page, lost_page, moves, needs_person, receipt, result, target_app):
    from .screen_pursuit import (
        logger,
    )

    result["target_app"] = target_app
    result["expect_page"] = expect_page
    result["anchored_to"] = anchor["page"]
    if needs_person["reason"] and not receipt.completed:
        # Name the question rather than the symptom. "no_move_available"
        # describes a loop with nothing to do; this says a person is being
        # waited on, and quotes what they are being asked.
        result["outcome"] = "needs_person"
        result["needs_person"] = needs_person["reason"]
    if lost_page["value"] and not receipt.completed:
        # Name it. "no_move_available" would describe the symptom of reading a
        # page that is not the task's, and hide that the browser had moved.
        result["outcome"] = "navigated_away"
    # How it ended, said once, whatever ended it.
    #
    # A run that stopped left nothing behind saying so. Live 2026-09-07,
    # driving a real game: nineteen moves, a rule worked out, and then the
    # log went quiet — no outcome, no reason, no count. Whether she had
    # finished, run out of budget, lost the window or hit a wall was not
    # recoverable from anything she wrote down, and every question about the
    # run had to start by guessing which of those it was.
    logger.info(
        "the run is over: %s after %d move(s)%s%s",
        result.get("outcome") or ("done" if result.get("completed") else "stopped"),
        len(moves),
        f", {result['restarts']} restart(s)" if result.get("restarts") else "",
        f" — {result['error']}" if result.get("error") else "",
    )

