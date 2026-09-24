"""One line for one move: what she did, what it does, and why that one.

A commentary of "Going left", "Going up" is a log of keystrokes. A person
watching wants, for each move, what it was for, and she has all of it before
her hand moves: the rule she learned says what the move makes of the situation,
and the measure she chose by says why it beat the moves she did not make.

So the line is read off those, not written for any thing in particular:

* what came together — two of a kind that left a line, and one worth both of
  them that turned up on it, and where it sits;
* why this move and not the next best, from the term of her own measure that
  separated them most — more room, things kept in order, neighbours kept near
  in value, more ways left to move, nearer to what she was asked for, her line
  kept.

Nothing here knows what the things are. A place is named the way a person names
places, and a number is a number.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

__all__ = ["what_a_move_does", "why_this_one"]

#: What each term of her measure is, said as a reason a person would give.
#:
#: These are her terms, not a vocabulary of strategies: each is the plain
#: meaning of a quantity she computes. An invented measure says its own name.
_WHAT_A_TERM_MEANS = {
    "room": ("it leaves more room", "it leaves more room than {other} would"),
    "order": ("it keeps things in order", "it keeps things more in order than {other} would"),
    "smoothness": (
        "it keeps neighbours close in value",
        "it keeps neighbours closer in value than {other} would",
    ),
    "freedom": ("it leaves more ways to move", "it leaves more ways to move than {other} would"),
    "nearness": ("it gets closer to what I'm after", "it gets closer to what I'm after than {other} would"),
    "line": ("it keeps to the plan I'm holding", "it keeps to my plan, where {other} wouldn't"),
    "newness": ("it goes somewhere new", "it goes somewhere new, where {other} goes back"),
}


def _named_place(arrangement: Any, row: int, column: int) -> str:
    rows = int(getattr(arrangement, "rows", 0) or 0)
    columns = int(getattr(arrangement, "columns", 0) or 0)
    if rows <= 0 or columns <= 0 or row < 0 or column < 0:
        return ""
    vertical = "top" if row == 0 else "bottom" if row == rows - 1 else ""
    horizontal = "left" if column == 0 else "right" if column == columns - 1 else ""
    if vertical and horizontal:
        return f"in the {vertical}-{horizontal} corner"
    if vertical:
        return f"on the {vertical} row"
    if horizontal:
        return f"on the {horizontal} edge"
    return "in the middle"


def _number(said: str) -> float | None:
    try:
        return float(str(said).replace(",", ""))
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return None


def _a(said: str) -> str:
    """The article a number takes when it is said aloud: an 8, an 11, a 16."""
    digits = str(said).replace(",", "").split(".")[0]
    spoken_with_a_vowel = digits.startswith("8") or (
        digits.startswith(("11", "18")) and len(digits) % 3 == 2
    )
    return f"an {said}" if spoken_with_a_vowel else f"a {said}"


def _came_together(before: Any, after: Any) -> list[tuple[str, str, str]]:
    """Pairs that became one: what two of, what they made, and where it is.

    Worked up from the smallest value, because one push can join two 2s into a
    4 and two 4s into an 8 at once, and then the count of 4s does not change at
    all. What a pair of a value made is part of what there is of the next
    value up.
    """
    was = Counter(cell.says for cell in before.cells)
    now = Counter(cell.says for cell in after.cells)
    values = sorted(
        {text for text in list(was) + list(now) if _number(text) is not None},
        key=lambda text: _number(text) or 0.0,
    )
    made_from_below: Counter[str] = Counter()
    pairs_of: dict[str, int] = {}
    for said in values:
        value = _number(said) or 0.0
        present = was.get(said, 0) + made_from_below.get(said, 0)
        pairs = max(0, (present - now.get(said, 0)) // 2)
        if pairs:
            pairs_of[said] = pairs
            made_from_below[f"{value * 2:g}"] += pairs
    old_places = {(cell.row, cell.column, cell.says) for cell in before.cells}
    joined: list[tuple[str, str, str]] = []
    for said in sorted(pairs_of, key=lambda text: _number(text) or 0.0, reverse=True):
        made = f"{(_number(said) or 0.0) * 2:g}"
        landed = [
            cell
            for cell in after.cells
            if cell.says == made and (cell.row, cell.column, made) not in old_places
        ]
        for index in range(pairs_of[said]):
            where = (
                _named_place(after, landed[index].row, landed[index].column)
                if index < len(landed)
                else ""
            )
            joined.append((said, made, where))
    return joined


def why_this_one(
    chosen: Mapping[str, float],
    runner_up: Mapping[str, float],
    weights: Mapping[str, float],
    *,
    runner_up_name: str = "",
) -> str:
    """The term of her measure that most separated the move from the next best."""
    best_name, best_gap = "", 0.0
    for name, value in chosen.items():
        gap = (float(value) - float(runner_up.get(name, 0.0))) * float(weights.get(name, 0.0))
        if gap > best_gap:
            best_name, best_gap = name, gap
    if not best_name:
        return ""
    plain, compared = _WHAT_A_TERM_MEANS.get(best_name) or _by_her_own_measure(best_name)
    return compared.format(other=runner_up_name) if runner_up_name else plain


def _by_her_own_measure(name: str) -> tuple[str, str]:
    """A reason given by a property she invented, said as hers.

    Its name is a recipe — what it looks at, how it adds them up — and read
    into the old frame it made no sentence: "it scores better on how small it
    is between neighbours, on average than right" (live, 2026-09-23). Said as
    what it is, a measure she worked out herself, it is also the part of the
    reason a person watching would most want to know about.
    """
    braces = str(name).replace("{", "{{").replace("}", "}}")
    return (
        f"by a measure I worked out myself ({braces}), it comes out ahead",
        f"by a measure I worked out myself ({braces}), it comes out ahead of {{other}}",
    )


def what_a_move_does(
    before: Any,
    action: str,
    after: Any,
    *,
    biggest_so_far: float = 0.0,
    because: str = "",
) -> str:
    """One sentence for one move.

    ``after`` is what her rule says the move makes of ``before``, before the
    world adds anything of its own: the move is hers and the addition is not.
    ``because`` is why this move rather than the next best, when there was a
    difference worth saying.
    """
    direction = str(action or "").strip()
    said_move = direction[:1].upper() + direction[1:] if direction else "That"
    if before is None or after is None or not hasattr(before, "cells"):
        return f"{said_move} — {because}." if because else f"{said_move}."
    parts: list[str] = []
    joined = _came_together(before, after)
    if joined:
        shown = []
        for said, made, where in joined[:2]:
            shown.append(f"two {said}s make {_a(made)}" + (f" {where}" if where else ""))
        said_joins = " and ".join(shown)
        if len(joined) > 2:
            more = len(joined) - 2
            said_joins += f", plus {more} more pair{'s' if more > 1 else ''}"
        parts.append(said_joins)
    if because:
        parts.append(because)
    free = int(getattr(after, "rows", 0) or 0) * int(getattr(after, "columns", 0) or 0) - len(after.cells)
    if 0 <= free <= 2 and not joined:
        parts.append(f"only {free} place{'' if free == 1 else 's'} left")
    if not parts:
        return f"{said_move}."
    sentence = parts[0] if len(parts) == 1 else "; ".join(parts[:-1]) + ", and " + parts[-1]
    return f"{said_move} — {sentence}."
