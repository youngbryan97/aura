"""Steps of `decide_the_next_move`: mending a failed move, reading an aim, progress, the rule's predictions, what made a move safe, and rules carried from a world like this one.

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


async def _mend_what_went_wrong(
    *,
    anchor: Any,
    ended: Any,
    narrate: Any,
    responds: Any,
    said_it_ended: Any,
    why: Any,
) -> Any:
    """Where the reason a move failed can be fixed, say so and mend the plan.

    Moved out of ``decide_the_next_move`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 8 name(s) from the turn and hands back
    1.
    """
    from .screen_pursuit_decision import (
        ONLY_SILENCE,
        _ensure_page,
        clear_what_is_in_front,
    )

    from .screen_pursuit import _tell
    from core.perception.why_nothing_answers import ELSEWHERE
    from core.perception.why_nothing_answers import ENDED
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
    return ended


def _read_where_she_is_aiming(
    *,
    ahead: Any,
    aiming_at: Any,
) -> Any:
    """Read the numbers in what she is aiming at, for the move ahead.

    Moved out of ``decide_the_next_move`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 6 name(s) from the turn and hands back
    1.
    """
    from core.cognition.a_window_not_a_maximum import AWindow
    from core.cognition.a_window_not_a_maximum import which_act_lands_in_it
    from core.cognition.enough_rather_than_most import the_one_most_likely_to_do
    import re
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
    return ahead


def _report_the_progress_made(
    *,
    going: Any,
    lines: Any,
    made: Any,
    moves: Any,
    narrate: Any,
    plan: Any,
    reached: Any,
) -> None:
    """Tell the progress watch how far the moves have got.

    Moved out of ``decide_the_next_move`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 10 name(s) from the turn and hands back
    0.
    """
    from .screen_pursuit import A_LINE_HERE
    from .screen_pursuit import _tell
    from core.agency import what_she_is_doing as doing
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


def _log_every_sixth_move(
    *,
    dropped: Any,
    knows: Any,
    laid_out: Any,
    moves: Any,
    observation: Any,
    pending: Any,
) -> None:
    """Every six moves, log what she knows about how this world moves.

    Moved out of ``decide_the_next_move`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 7 name(s) from the turn and hands back
    0.
    """
    from .screen_pursuit_decision import (
        _how_far_she_saw,
        _what_she_could_not_learn_from,
        _what_she_is_not_reading,
    )

    from .screen_pursuit import logger
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
            # And how far that thinking reached, which is what the
            # time buys: the same second is a different depth in a
            # quiet process and in a busy one.
            "a look took %.2fs of which %.2fs was reading it, and she "
            "thought for %.2fs, seeing %d move(s) ahead"
            % (
                float(observation.get("seconds_to_still", 0.0) or 0.0)
                + float(observation.get("seconds_reading", 0.0) or 0.0),
                float(observation.get("seconds_reading", 0.0) or 0.0),
                float(pending.get("thought_for", 0.0) or 0.0),
                _how_far_she_saw(),
            ),
        )


def _hold_the_rule_to_its_prediction(
    *,
    chosen: Any,
    foretold_by_the_rule: Any,
    key: Any,
    knows: Any,
) -> None:
    """Hold what the rule foretold against what happened.

    Moved out of ``decide_the_next_move`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 5 name(s) from the turn and hands back
    0.
    """
    from dataclasses import replace
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


def _learn_what_made_a_move_safe(
    *,
    attempt: Any,
    knows: Any,
    pending: Any,
    previous: Any,
) -> None:
    """After a move that changed something, learn what made it safe.

    Moved out of ``decide_the_next_move`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 6 name(s) from the turn and hands back
    0.
    """
    from .screen_pursuit_decision import (
        _the_same_thing_without,
    )

    from .screen_pursuit import logger
    from core.cognition.a_shape_that_makes_it_safe import what_makes_it_safe
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


def _carry_rules_from_a_world_like_it(
    *,
    elsewhere: Any,
    knows: Any,
    like_it: Any,
    skilled: Any,
    world: Any,
) -> None:
    """Start from what a world like this one taught, no more than a fresh start is worth.

    Moved out of ``decide_the_next_move`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 10 name(s) from the turn and hands back
    0.
    """
    from .screen_pursuit_decision import (
        _no_more_than_a_fresh_one_is_worth,
    )

    from .screen_pursuit import _tell
    from .screen_pursuit import logger
    from core.agency.what_worked_before import WhatWorkedBefore
    from core.perception.how_it_moves import HowItMoves
    from core.perception.what_the_world_does import WhatTheWorldDoes
    if elsewhere:
        like_it["carried"] = True
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


def _borrow_from_the_world_it_is_most_like(
    *,
    knows: Any,
    laid_out: Any,
    like_it: Any,
    world: Any,
) -> None:
    """Start from the solved world shaped most like this one, where none of its kind was.

    The kind carried above is an exact name — the same size, the same number
    of acts — so a five-by-five board that moves like a four-by-four one she
    has solved got nothing from it. This asks by shape instead, on every look
    that shows more than the last while her own rule is still unsettled, and
    lends at most once. See `core.agency.the_world_it_is_most_like`.

    And when her own rule settles, the look in front of her then is kept as
    this world's shape, for the next world that might be like it.
    """
    if laid_out is None or not laid_out.occupied() or knows.rules is None:
        return
    from core.agency.the_world_it_is_most_like import (
        graph_to_memory,
        lend_to,
        lent,
        most_like,
        shape_of,
        solved_worlds,
    )

    look = shape_of(laid_out)
    if knows.rules.rule() is not None:
        if look.relations and not like_it.get("shape"):
            like_it["shape"] = graph_to_memory(look)
        return
    if like_it.get("lent") or like_it.get("carried"):
        return
    if len(look.relations) <= int(like_it.get("asked_at") or 0):
        return
    like_it["asked_at"] = len(look.relations)
    from core.runtime.what_she_learned import kept_worlds, recall

    from .screen_pursuit import _tell, logger

    if "solved" not in like_it:
        like_it["solved"] = {
            name: found
            for name, found in solved_worlds(kept_worlds(), recall).items()
            if name != like_it.get("this_world")
        }
    solved = like_it["solved"]
    likeness = most_like(
        look,
        {name: graph for name, (graph, _knew) in solved.items()},
        teaches={name: str(knew.get("_teaches") or "") for name, (_g, knew) in solved.items()},
    )
    if likeness is None:
        return
    given = lent(solved[likeness.world][1], likeness)
    if not lend_to(knows.rules, given, world):
        return
    like_it["lent"] = likeness.world
    _tell(
        f"This is shaped like {likeness.world}, which I worked out before, so I "
        "will start from how that moved and let my moves here decide."
    )
    logger.info(
        "lent by %s, at %.2f of a fresh start: %s",
        likeness.says(), float(given.get("trust") or 0.0), knows.rules.says(),
    )


def _narrate_a_fresh_plan(
    *,
    changing: Any,
    ended: Any,
    fresh: Any,
    going: Any,
    knowledge: Any,
    moves: Any,
    narrate: Any,
    same: Any,
) -> None:
    """Say a new plan aloud and tell the watch what it expects.

    Moved out of ``decide_the_next_move`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 10 name(s) from the turn and hands back
    0.
    """
    from .screen_pursuit import _tell
    from .screen_pursuit import logger
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


def _lean_where_the_world_could_swing(
    ahead: Any,
    *,
    knows: Any,
    laid_out: Any,
    choices: list[str],
    held_line: str,
    world: Any,
    success_when: str,
    ends_at: float,
    began: float,
    began_at: Any,
) -> Any:
    """Where the world's own move could swing the result, lean toward or away from it.

    How much of what happens next is the world's rather than hers, which is the
    other thing looking ahead averages away. Ahead with time to spare, an
    uncertain position is worth avoiding; behind with the clock going, it is
    worth seeking. Lifted whole out of ``decide_the_next_move``.
    """
    from core.agency.looking_ahead import at_the_worlds_mercy, whether_to_take_the_wide_option

    from .screen_pursuit import logger
    from .screen_pursuit_decision import _how_it_has_been_going, _what_there_is_to_aim_at

    exposed = at_the_worlds_mercy(
        knows.rules,
        laid_out,
        choices,
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
    return ahead


def _made_so_far(moves: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every key she has pressed in this run, oldest first."""
    return [str(move.get("key") or "") for move in moves if move.get("key")]


def _grade_how_soon_she_came_back(run: Any, act: str, held: bool, moves: Sequence[Mapping[str, Any]]) -> None:
    """Whether what she expected of this act held, against how soon she had made it before.

    The act being graded is the last of its name she pressed; what came before
    that press is how soon she came back to it.
    """
    wears = getattr(run, "wears", None)
    if wears is None:
        return
    made = _made_so_far(moves)
    name = str(act or "")
    last = max((at for at, one in enumerate(made) if one == name), default=len(made))
    wears.it_went(name, held, made[:last])


def _mark_down_what_she_keeps_doing(
    run: Any,
    ahead: Any,
    available: list[Any],
    moves: Sequence[Mapping[str, Any]],
    narrate: bool,
) -> tuple[Any, list[Any]]:
    """In a world that has learned her habits, the act she keeps making is worth less.

    See core/cognition/what_wears_out.py. Where she can see where each act
    leads, an act she would be coming back to soon is marked down by what that
    has cost it, in the units of what she sees ahead; where she cannot, the
    acts she has rested are offered first. In a world that does not learn her
    every mark is nought and nothing moves.
    """
    from .screen_pursuit import _tell, logger

    wears = getattr(run, "wears", None)
    if wears is None or not available:
        return ahead, available
    worn = wears.worn([option.name for option in available], _made_so_far(moves))
    if not any(worn.values()):
        return ahead, available
    if wears.learns_her() >= 0.5 and not wears.noticed:
        wears.noticed = True
        logger.info("%s", wears.says())
        if narrate:
            said = wears.says()
            _tell(f"{said[:1].upper()}{said[1:]}, so I will not lean on one move.")
    if ahead:
        worths = [value for value, _why in ahead.values()]
        # In the units of what she sees ahead; where every act looks the same,
        # any mark at all is what decides between them.
        spread = (max(worths) - min(worths)) or 1.0
        ahead = {
            name: (value - spread * worn.get(name, 0.0), why)
            for name, (value, why) in ahead.items()
        }
        return ahead, available
    return ahead, sorted(available, key=lambda option: worn.get(option.name, 0.0))

