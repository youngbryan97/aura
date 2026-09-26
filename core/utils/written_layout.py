"""A grid written in a sentence, read out of it.

People write a layout the way they would draw it on one line: rows with "/"
between them, the things in each row with spaces between, and "_" for a place
left empty — "until it reads 1 2 3 / 4 5 6 / 7 8 _". The words around it are
not part of it: "until it reads" is not the top row.

So a row is only what looks like a place on a board: a number, a single
character, or a mark for an empty place. The first row is the run of those
just before the first "/", the last row the run just after the last one, and
every row has to be as long as every other.
"""

from __future__ import annotations

import re

__all__ = ["EMPTY_MARKS", "written_layout"]

#: How a place left empty is written.
EMPTY_MARKS = frozenset({"_", ".", "-", "·"})

_A_PLACE = re.compile(r"^(?:\d[\d,]*|[^\W\d_]|[_.·-])$")


def _trailing(tokens: list[str]) -> list[str]:
    run: list[str] = []
    for token in reversed(tokens):
        if not _A_PLACE.match(token):
            break
        run.append(token)
    return run[::-1]


def _leading(tokens: list[str]) -> list[str]:
    run: list[str] = []
    for token in tokens:
        if not _A_PLACE.match(token):
            break
        run.append(token)
    return run


def written_layout(text: str) -> str:
    """The layout written in ``text``, rows joined by " / ", or empty when there is none."""
    said = str(text or "").replace("\n", " / ")
    parts = [part.split() for part in said.split("/")]
    if len(parts) < 2:
        return ""
    best: list[list[str]] = []
    # Every stretch of consecutive parts that reads as a grid; the longest wins.
    for start in range(len(parts) - 1):
        first = _trailing([token.strip(".,;:!?") or token for token in parts[start]])
        if len(first) < 2:
            continue
        rows = [first]
        for index in range(start + 1, len(parts)):
            tokens = [token.rstrip(",;:!?") for token in parts[index]]
            whole = all(_A_PLACE.match(token) for token in tokens) and len(tokens) == len(first)
            last = index == len(parts) - 1 or not whole
            row = tokens if whole else _leading(tokens)
            if len(row) < len(first):
                break
            rows.append(row[: len(first)])
            if last:
                break
        if len(rows) >= 2 and len(rows) > len(best):
            best = rows
    return " / ".join(" ".join(row) for row in best)
