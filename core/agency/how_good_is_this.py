"""How good a situation is, so futures she has not visited can be compared.

Imagining what a move would do is only half of being able to try one: the
other half is being able to say which of the imagined results is better. Two
things make a situation good, and neither of them is about any particular kind
of screen.

The first is nearness to what she was asked for. Where the goal names
something, a state either has it or is some way off having it, and that is
computable without knowing what the thing is.

The second is her own approach. She states a line — "keep the largest in the
bottom-left corner", "clear the left column first" — and that line is already
something that can be checked against a state. Scoring a future by whether it
satisfies the line she is holding is what makes the plan cause the moves
rather than accompany them.

Two smaller terms are older than either, and both are about what a situation
affords rather than about what is wanted from it. Room to act: a situation
with more space left in it affords more of whatever comes next — true of a
board, a form, a queue and a disk. And order: a thing whose contents run in
order along a line is easier to act in than one where they are jumbled,
because what goes with what is already next to it. A sorted list, a stacked
shelf, a tidy desk.

Neither is a strategy. Both are properties of a laid-out thing that make the
next act easier whatever the next act turns out to be.

Nothing here encodes a strategy. Where she has stated no line and the goal
names nothing measurable, this says so, and the caller falls back to acting
and looking.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from typing import Any, Sequence

logger = logging.getLogger("Aura.HowGoodIsThis")

__all__ = ["ROOM_MATTERS", "bound_to", "how_good", "worth_comparing"]

#: What having somewhere left to act is worth, beside being closer to the goal
#: and beside holding the line she said she would hold. Small on purpose: room
#: is what lets a plan continue, not a reason to do anything in particular.
ROOM_MATTERS = 0.15

#: What running in order is worth. Level with room, because the two are the
#: same kind of thing: a situation is easier to work in for having either, and
#: neither says anything about what she is trying to do.
ORDER_MATTERS = 0.15

#: What it is worth for neighbouring things to be near each other in value.
#:
#: Measured, not chosen. Six games at each weight, run to a dead board, the
#: same seeds, with her line held and the world model on:
#:
#:     smoothness    median best tile    total
#:            0.0                1024     2133
#:           0.15                1536     2670
#:            0.4                2048     4020
#:            1.0                1536     2463
#:
#: The median best tile DOUBLES, and totals do too. Past this it turns back:
#: closeness bought at the price of progress is a board that is easy to work
#: in and going nowhere.
#:
#: Asked for a 4096 instead, eight games each and room to play them out:
#:
#:     smoothness    median best tile    best seen
#:            0.3                2048         4096
#:            0.4                1536         4096
#:           0.55                2048         2048
#:
#: A 4096 tile is reachable, and it is not the ordinary case. What to say
#: about her play is that she reaches 2048 as a rule and 4096 on a good run.
SMOOTHNESS_MATTERS = 0.4

#: What holding her own stated line is worth. Level with nearness to the goal,
#: because a line she is holding is her judgement about how the goal is reached
#: and discounting it would make the plan decorative.
LINE_MATTERS = 1.0

#: A target written as a number, which is the case where nearness is
#: computable without knowing anything about the thing.
_A_TARGET = re.compile(r"^\d[\d,]*$")

#: A line that refers to whatever the biggest thing is, rather than naming it.
#:
#: "Keep the largest in the bottom-left corner" is the ordinary way to state
#: this kind of plan, and it names no value at all — so read literally the
#: claim was empty and holding the line contributed nothing to any score.
#: Measured 2026-08-26: her stated approach made no difference to a single
#: choice.
_A_SUPERLATIVE = re.compile(
    r"\b(?:(?P<most>largest|biggest|highest|greatest|max|maximum)"
    r"|(?P<least>smallest|lowest|least|fewest|min|minimum))\b",
    re.IGNORECASE,
)


def bound_to(approach: str, state: Any) -> str:
    """The line she is holding, with any superlative bound to what it names here.

    NOT USED BY THE SEARCH, and the reason is worth keeping.

    The argument for it is good. "The largest" names a specific thing once she
    has looked at a state — the one sitting in the corner right now, not
    whichever turns out biggest in some future a search has only imagined.
    Left unbound, a multi-step search re-reads the word against every state it
    tries, so a merge that makes something bigger elsewhere reads as losing
    the line rather than as leaving the thing she meant untouched.

    Measured 2026-08-29, five games each run to a dead board, the same seeds:

        her line + the world model      without binding      with binding
        median best tile                       1024                  512
        best seen                              2048                 1024
        total                                  2142                 1658
        moves                                  1069                  827

    Half her play. In a world that combines, the superlative is MEANT to
    float: a line about keeping the largest in a corner is about whatever is
    largest as the game goes on, and binding it to the 128 that was there when
    the line was formed means that the moment she merges past it, the line
    names something that no longer exists and contributes nothing.

    Kept, because the argument survives the measurement for worlds where the
    extreme thing does not change — a seating plan, a price list, a leaderboard
    between updates. Wire it there, measure it there, and do not assume.
    """
    said = str(approach or "")
    found = _A_SUPERLATIVE.search(said)
    if not found:
        return said
    wanted = _least(state) if found.group("least") else _biggest(state)
    if wanted <= 0:
        return said
    return f"{said[:found.start()]}{wanted:g}{said[found.end():]}"


def worth_comparing(toward: str, approach: str) -> bool:
    """Whether there is anything here to score a situation by.

    Said plainly rather than guessed at. A caller with neither a measurable
    goal nor a line she is holding has no business ranking futures, and should
    act and look instead.
    """
    return bool(_target(toward)) or bool(str(approach or "").strip())


def terms(
    state: Any,
    *,
    toward: str = "",
    approach: str = "",
) -> dict[str, float]:
    """What there is to like about a situation, each thing on its own.

    ``how_good`` adds these up with weights. Kept apart, they can be weighed
    differently in a world where different things matter — which is a fact
    about the world and not something anybody can know in advance.
    """
    said = {
        "nearness": _nearness(state, toward),
        "line": _holds_her_line(state, approach),
        "room": _room(state),
        "order": _order(state),
        "smoothness": _smoothness(state),
    }
    # And anything she worked out for herself.
    #
    # Every property above was written down by somebody. That is the whole of
    # what a mind can judge a situation by until it can add one, and adding one
    # is not a matter of noticing — it is finding what your own measure cannot
    # account for, composing something that does, and proving it plays better.
    # See core/agency/what_i_cannot_explain.py for where they come from.
    for name, measure in INVENTED.items():
        try:
            said[name] = float(measure.read(state))
        except (AttributeError, TypeError, ValueError):
            continue
    return said


#: What each thing is worth when nothing has been learned about this world.
#: Properties she worked out for herself, by name. Empty until she finds one,
#: and everything that reads a situation sees them beside the authored ones —
#: there is no second-class shelf for a measure because a person did not write
#: it. Promotion goes through ``promote``, which will not take one on the
#: strength of explaining the past.
INVENTED: dict[str, Any] = {}


def promote(measure: Any, worth: float) -> str:
    """Take a property she invented into the measure she judges situations by.

    The weight is a measurement, not an opinion: nothing calls this until a
    property has been shown to improve the play, and by how much.
    """
    name = str(getattr(measure, "name", "") or "").strip()
    if not name or not callable(getattr(measure, "read", None)):
        return ""
    INVENTED[name] = measure
    AS_GOOD_A_GUESS_AS_ANY[name] = float(worth)
    logger.info("she added %r to what she judges a situation by, worth %.2f", name, worth)
    return name


#: How many outcomes a property gets to prove itself over before she keeps or
#: drops it. Enough that one lucky run does not decide, few enough that a bad
#: property is not steering her for long.
A_FAIR_TRIAL = 60


def on_trial(measure: Any, worth: float) -> str:
    """Take a property on provisionally, to find out what it does.

    She cannot replay her life to A/B a change to her own judgement, so the
    trial has to happen in it. A property goes in, she acts on it, and what
    happens next is the evidence — compared against how things were going
    before it arrived.

    This is the only honest way an invented measure reaches her at all.
    Offline it can be tested by replaying a world; nothing replays a life.
    """
    name = promote(measure, worth)
    if name:
        ON_TRIAL[name] = {"since": 0, "before": None, "worth": worth}
    return name


#: Properties she is currently trying out, and what was true before each.
ON_TRIAL: dict[str, dict[str, Any]] = {}


def how_the_trial_is_going(name: str, standing_at: float) -> str:
    """One observation under a trial. Returns "kept", "dropped" or "" mid-trial.

    ``standing_at`` is where she stands right now by whatever this world
    counts. What the trial actually measures is the RATE that moves at, which
    is the only thing comparable across two different stretches of a life.

    An average of where she stood is not: the middle of a run and the end of a
    run are different places under an identical policy, so comparing a mean of
    one against the last value of the other measures where the window sat
    rather than whether the property helped. Both sides are gain per
    observation now, which does not care where in a trajectory it was taken.
    """
    trial = ON_TRIAL.get(str(name))
    if trial is None:
        return ""
    standing = float(standing_at)
    if trial.get("from") is None:
        trial["from"] = standing
    trial["to"] = standing
    trial["seen"] = int(trial.get("seen", 0)) + 1
    if trial["seen"] < A_FAIR_TRIAL:
        return ""
    was = trial.get("before")
    now = (float(trial["to"]) - float(trial["from"])) / max(1, trial["seen"])
    ON_TRIAL.pop(str(name), None)
    if was is None or now >= float(was):
        logger.info(
            "%r earned its place: %.4f a move against %s", name, now, was
        )
        return "kept"
    forget(str(name))
    logger.info(
        "%r did not earn its place: %.4f a move against %.4f — dropped",
        name, now, float(was),
    )
    return "dropped"


def what_it_was_like_before(name: str, a_move: float) -> None:
    """The rate things were moving at before a property went on trial.

    The same statistic the trial will produce — gain per observation — because
    a baseline measured one way and a trial measured another compares nothing.
    """
    trial = ON_TRIAL.get(str(name))
    if trial is not None and trial.get("before") is None:
        trial["before"] = float(a_move)


def a_trial_is_running() -> str:
    """The property currently on trial, if one is. Empty when none is.

    Read from here rather than carried in a caller, because a trial outlives
    the run that started it — sixty observations is more than one run — and a
    caller that keeps its own handle loses it the moment that run ends. Which
    is what happened: the property was invented, promoted and used, and no
    verdict was ever reached on it.
    """
    return next(iter(ON_TRIAL), "")


def forget(name: str) -> bool:
    """Drop a property she invented. What earned its place can lose it."""
    gone = INVENTED.pop(str(name), None) is not None
    AS_GOOD_A_GUESS_AS_ANY.pop(str(name), None)
    return gone


AS_GOOD_A_GUESS_AS_ANY: dict[str, float] = {
    "nearness": 1.0,
    "line": LINE_MATTERS,
    "room": ROOM_MATTERS,
    "order": ORDER_MATTERS,
    "smoothness": SMOOTHNESS_MATTERS,
}


def how_good(
    state: Any,
    *,
    toward: str = "",
    approach: str = "",
    weights: Mapping[str, float] | None = None,
) -> float:
    """How good this situation is, between nearness, her line, room and order.

    ``state`` is anything that reads like an arrangement: it is asked for its
    numbers, its free places, and whether a claim holds in it. Nothing else is
    assumed about it.

    ``weights`` is what each of those is worth here, when she has worked that
    out. Without it the standing weights apply, which are a guess — a good
    enough one to start from and no more than that.
    """
    weighed = AS_GOOD_A_GUESS_AS_ANY if weights is None else weights
    here = terms(state, toward=toward, approach=approach)
    return sum(here[name] * float(weighed.get(name, 0.0)) for name in here)


def rank(
    futures: dict[str, Any],
    *,
    toward: str = "",
    approach: str = "",
) -> list[tuple[str, float]]:
    """Every way it could go, best first."""
    scored = [
        (name, how_good(state, toward=toward, approach=approach))
        for name, state in futures.items()
    ]
    return sorted(scored, key=lambda row: row[1], reverse=True)


def why(state: Any, *, toward: str = "", approach: str = "") -> str:
    """What makes this situation the one she picked, in a line she can say."""
    parts: list[str] = []
    target = _target(toward)
    if target:
        biggest = _biggest(state)
        if biggest >= target:
            parts.append(f"it has the {target:g}")
        elif biggest:
            parts.append(f"the largest is {biggest:g}")
    if approach and _holds_her_line(state, approach):
        parts.append("it keeps the line I am taking")
    room = _free(state)
    if room is not None:
        parts.append(f"{room} place(s) left")
    return ", ".join(parts)


# ── the three things that make a situation good ──────────────────────────


def _nearness(state: Any, toward: str) -> float:
    """How near this is to what she was asked for, where that is computable.

    On a scale where reaching it is one. Doubling is the step that matters in
    anything built by combining, and a plain ratio makes every early move look
    like nothing, so nearness is counted in doublings.
    """
    target = _target(toward)
    if not target:
        return 0.0
    biggest = _biggest(state)
    if biggest <= 0:
        return 0.0
    if biggest >= target:
        return 1.0
    from math import log2

    return max(0.0, min(1.0, log2(biggest) / log2(target)))


def _holds_her_line(state: Any, approach: str) -> float:
    """Whether the line she said she was taking is still true of this."""
    said = str(approach or "").strip()
    if not said:
        return 0.0
    try:
        from core.agency.standing_strategy import claim_in  # noqa: PLC0415
        from core.perception.what_is_there import holds_in  # noqa: PLC0415

        claim = claim_in(said)
        superlative = _A_SUPERLATIVE.search(said)
        if not claim.says_something() and claim.at_place and superlative:
            # A line about whichever thing is the extreme one, bound to
            # whatever that turns out to be. Both ends, because "keep the
            # smallest out of the middle" is as ordinary a plan as its
            # opposite, and reading only one of them makes a whole class of
            # stated line unusable.
            wanted = _least(state) if superlative.group("least") else _biggest(state)
            if wanted > 0:
                named = f"{wanted:g}"
                claim = claim.__class__(
                    changed=claim.changed,
                    contains=(named,),
                    absent=claim.absent,
                    describes=claim.describes,
                    at_place=claim.at_place,
                    keeping=(named,),
                )
        if not claim.says_something():
            return 0.0
        # The content of the claim only. Whether the situation CHANGED is a
        # question about a move, and this is a question about a situation —
        # asked of a state against itself, a claim that something will differ
        # is false of every state including the good ones.
        ok, _why = holds_in(
            state,
            contains=claim.contains,
            absent=claim.absent,
            at_place=claim.at_place,
            keeping=claim.keeping,
        )
        return 1.0 if ok else 0.0
    except (ImportError, AttributeError, TypeError, ValueError):
        return 0.0


def _order(state: Any) -> float:
    """How much of this runs in order along its lines, either way.

    Read off the thing rather than assumed about it: every row and every
    column is a line, and a line is orderly to the extent that consecutive
    things along it do not reverse direction. Ascending and descending count
    alike — which end is the big end is not this function's business, and on a
    board it is whichever end she said she was keeping things at.
    """
    lines = _lines_of(state)
    if not lines:
        return 0.0
    scored = [_runs_one_way(line) for line in lines if len(line) > 1]
    return sum(scored) / len(scored) if scored else 0.0


def _smoothness(state: Any) -> float:
    """How near neighbouring things are to each other in value, on a doubling scale.

    A thing can be perfectly ordered and impossible to work with: 2, 32, 4,
    64 runs one way along no line, and 2, 4, 512, 1024 runs one way along
    every line while offering nothing that can combine. What makes a situation
    workable is that the things beside each other are CLOSE — one step apart
    rather than eight — because that is what lets them come together at all.

    Counted in doublings, like nearness is, because in anything built by
    combining a step is a doubling and a plain difference makes every gap
    among small things look like nothing.
    """
    lines = _lines_of(state)
    if not lines:
        return 0.0
    apart: list[float] = []
    for line in lines:
        for one, other in zip(line, line[1:]):
            if one <= 0 or other <= 0:
                continue
            apart.append(abs(math.log2(one) - math.log2(other)))
    if not apart:
        return 0.0
    # One doubling apart is as close as two different things can be, so that
    # is the scale a gap is measured against.
    return 1.0 / (1.0 + sum(apart) / len(apart))


def _runs_one_way(values: Sequence[float]) -> float:
    """The share of steps along a line that go the same way as the rest."""
    steps = [b - a for a, b in zip(values, values[1:]) if b != a]
    if not steps:
        return 1.0
    up = sum(1 for step in steps if step > 0)
    return max(up, len(steps) - up) / len(steps)


def _lines_of(state: Any) -> list[list[float]]:
    """Every row and column of a laid-out thing, as the numbers along it."""
    rows = int(getattr(state, "rows", 0) or 0)
    columns = int(getattr(state, "columns", 0) or 0)
    row_at = getattr(state, "row_at", None)
    column_at = getattr(state, "column_at", None)
    if not rows or not columns or not callable(row_at) or not callable(column_at):
        return []
    lines: list[list[float]] = []
    for index in range(rows):
        lines.append(_numbers_along(row_at(index)))
    for index in range(columns):
        lines.append(_numbers_along(column_at(index)))
    return [line for line in lines if line]


def _numbers_along(cells: Sequence[Any]) -> list[float]:
    found: list[float] = []
    for cell in cells:
        value = cell.number() if cell is not None and hasattr(cell, "number") else None
        if value is not None:
            found.append(value)
    return found


def _room(state: Any) -> float:
    """The share of this that is still free to act in."""
    free = _free(state)
    places = getattr(state, "places", None)
    total = places() if callable(places) else 0
    if free is None or not total:
        return 0.0
    return max(0.0, min(1.0, free / total))


# ── asking a state about itself ──────────────────────────────────────────


def _biggest(state: Any) -> float:
    numbers = getattr(state, "numbers", None)
    values: Sequence[float] = numbers() if callable(numbers) else ()
    return max(values) if values else 0.0


def _least(state: Any) -> float:
    numbers = getattr(state, "numbers", None)
    values = numbers() if callable(numbers) else ()
    return min(values) if values else 0.0


def _free(state: Any) -> int | None:
    empty = getattr(state, "empty", None)
    return empty() if callable(empty) else None


def _target(toward: str) -> float:
    said = str(toward or "").strip().replace(",", "")
    if not _A_TARGET.match(said):
        return 0.0
    try:
        return float(said)
    except ValueError:
        return 0.0
