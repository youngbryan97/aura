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

A third term is smaller and older than either: room to act. A situation with
more space left in it affords more of whatever comes next. That is true of a
board, a form, a queue, and a disk.

Nothing here encodes a strategy. Where she has stated no line and the goal
names nothing measurable, this says so, and the caller falls back to acting
and looking.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

__all__ = ["ROOM_MATTERS", "how_good", "worth_comparing"]

#: What having somewhere left to act is worth, beside being closer to the goal
#: and beside holding the line she said she would hold. Small on purpose: room
#: is what lets a plan continue, not a reason to do anything in particular.
ROOM_MATTERS = 0.15

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


def worth_comparing(toward: str, approach: str) -> bool:
    """Whether there is anything here to score a situation by.

    Said plainly rather than guessed at. A caller with neither a measurable
    goal nor a line she is holding has no business ranking futures, and should
    act and look instead.
    """
    return bool(_target(toward)) or bool(str(approach or "").strip())


def how_good(
    state: Any,
    *,
    toward: str = "",
    approach: str = "",
) -> float:
    """How good this situation is, between nearness, her line, and room.

    ``state`` is anything that reads like an arrangement: it is asked for its
    numbers, its free places, and whether a claim holds in it. Nothing else is
    assumed about it.
    """
    return (
        _nearness(state, toward)
        + LINE_MATTERS * _holds_her_line(state, approach)
        + ROOM_MATTERS * _room(state)
    )


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
