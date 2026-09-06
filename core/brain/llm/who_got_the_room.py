"""What each part of a prompt was given, and which part paid for the squeeze.

:mod:`core.brain.llm.context_budget` fits a prompt to a budget by scoring how
much each section bears on the request and dropping what scores lowest. That is
the right rule and it says nothing about what happened. A turn that came back
thin is indistinguishable from a turn where recalled memory scored badly and
was cut to nothing, because the only record afterwards is a shorter prompt.

So: named allocations, and a ledger.

* Six parts, each with a **floor** — the smallest share that is still worth
  having. Below its floor a part is not a small version of itself; identity
  trimmed to fifty characters is not a reduced identity, it is a stranger.
* Each fitted prompt records what every part actually got.
* :func:`who_was_squeezed` names the parts that went under their floor, and
  :func:`how_the_room_was_shared` reports it over a run.

The floors are fractions of the budget rather than absolute counts, because a
budget changes with the model and a floor written in characters would be
generous on one and impossible on another. They sum to less than one on
purpose: what is left is the room the fit gets to allocate on merit.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.WhoGotTheRoom")

__all__ = [
    "APart",
    "THE_PARTS",
    "part_of",
    "what_each_part_got",
    "who_was_squeezed",
    "note_a_fit",
    "how_the_room_was_shared",
    "forget_everything",
]


@dataclass(frozen=True, slots=True)
class APart:
    """One named part of a prompt, and the least of it worth sending."""

    name: str
    #: Smallest useful share of the budget, as a fraction.
    floor: float
    #: What the section headers for this part look like.
    reads_like: tuple[str, ...]
    why: str


#: The six, in the order they appear in an assembled prompt. Floors sum to
#: 0.42; the rest is what the fit allocates on how much each section bears on
#: the request.
THE_PARTS: tuple[APart, ...] = (
    APart(
        name="identity",
        floor=0.10,
        reads_like=("", "identity", "you are", "about you", "self"),
        why="trimmed past a point it is not a shorter identity, it is a stranger",
    ),
    APart(
        name="memory",
        floor=0.08,
        reads_like=("memory", "recall", "remembered", "earlier", "history of"),
        why="a turn with no recalled memory answers as though it has never met them",
    ),
    APart(
        name="interiority",
        floor=0.04,
        reads_like=("felt", "affect", "mood", "interior", "state of mind"),
        why="the smallest part, and the one whose absence changes the voice",
    ),
    APart(
        name="tools",
        floor=0.06,
        reads_like=("tool", "skill", "capabilit", "available action"),
        why="a tool she cannot see is a tool she reports not having",
    ),
    APart(
        name="history",
        floor=0.08,
        reads_like=("conversation", "transcript", "turn", "said", "message"),
        why="without the last few turns every answer restarts the conversation",
    ),
    APart(
        name="reply",
        floor=0.06,
        reads_like=("instruction", "task", "question", "asked", "request"),
        why="what is actually being asked; nothing else matters if this is cut",
    ),
)

_BY_NAME = {part.name: part for part in THE_PARTS}


def part_of(header: str) -> str:
    """Which of the six a section header belongs to, or 'other'.

    An empty header is the identity block — the text before the first header
    in an assembled prompt, which the budget module also treats as the thing
    the rest hangs off.
    """
    said = str(header or "").strip().lower()
    if not said:
        return "identity"
    for part in THE_PARTS:
        for mark in part.reads_like:
            if mark and mark in said:
                return part.name
    return "other"


def what_each_part_got(prompt: str) -> dict[str, int]:
    """Characters per named part in an assembled prompt."""
    from core.brain.llm.context_budget import sections_of

    got: dict[str, int] = {part.name: 0 for part in THE_PARTS}
    got["other"] = 0
    for section in sections_of(prompt):
        got[part_of(section.header)] += len(section.text)
    return got


def who_was_squeezed(
    after: str, *, budget: int, before: str | None = None
) -> dict[str, dict[str, Any]]:
    """Parts that came out under their floor, and by how much.

    A part that was never in the prompt is not squeezed. A part that was there
    and came out at zero is the worst case there is, and reading only the
    fitted prompt cannot tell those apart — which is why ``before`` matters.
    The first version of this took the fitted prompt alone, and on the first
    real fit it reported nothing squeezed while the question being asked had
    been cut entirely.
    """
    kept = what_each_part_got(after)
    had = what_each_part_got(before) if before is not None else kept
    under: dict[str, dict[str, Any]] = {}
    for part in THE_PARTS:
        floor = int(part.floor * max(0, budget))
        have = kept.get(part.name, 0)
        was = had.get(part.name, 0)
        if was <= 0:
            continue
        # Nothing was taken, so nothing was squeezed. A short prompt that fits
        # is under every floor and is not a finding: the floor is the least
        # worth KEEPING, not the least worth having.
        if have >= was:
            continue
        # And a part cannot be short of more than it brought.
        floor = min(floor, was)
        if have >= floor:
            continue
        under[part.name] = {
            "had": was,
            "got": have,
            "floor": floor,
            "short_by": floor - have,
            "emptied": have == 0,
            "why_it_matters": part.why,
        }
    return under


_LEDGER: list[dict[str, Any]] = []
_LOCK = threading.Lock()
_KEEP = 200


def note_a_fit(before: str, after: str, *, budget: int, request: str = "") -> dict[str, Any]:
    """Record one fit: what each part had, what it kept, who went under.

    Returns the record rather than only storing it, so a caller can act on a
    squeeze in the turn it happened rather than reading about it later.
    """
    had = what_each_part_got(before)
    kept = what_each_part_got(after)
    squeezed = who_was_squeezed(after, budget=budget, before=before)
    record = {
        "budget": budget,
        "request": request[:120],
        "before": len(before),
        "after": len(after),
        "had": had,
        "kept": kept,
        "lost": {
            name: had[name] - kept.get(name, 0)
            for name in had
            if had[name] > kept.get(name, 0)
        },
        "squeezed": squeezed,
    }
    with _LOCK:
        _LEDGER.append(record)
        del _LEDGER[:-_KEEP]
    emptied = sorted(n for n, one in squeezed.items() if one["emptied"])
    if emptied:
        logger.warning(
            "prompt fit removed %s entirely", ", ".join(emptied)
        )
    elif squeezed:
        logger.info(
            "prompt fit put %s under the floor", ", ".join(sorted(squeezed))
        )
    return record


def how_the_room_was_shared() -> dict[str, Any]:
    """For the health report: who keeps paying for the fits."""
    with _LOCK:
        rows = list(_LEDGER)
    if not rows:
        return {"fits": 0, "parts": [p.name for p in THE_PARTS]}
    squeezed_counts: dict[str, int] = {}
    emptied_counts: dict[str, int] = {}
    lost_totals: dict[str, int] = {}
    for row in rows:
        for name, one in row["squeezed"].items():
            squeezed_counts[name] = squeezed_counts.get(name, 0) + 1
            if one.get("emptied"):
                emptied_counts[name] = emptied_counts.get(name, 0) + 1
        for name, amount in row["lost"].items():
            lost_totals[name] = lost_totals.get(name, 0) + amount
    return {
        "fits": len(rows),
        "parts": [p.name for p in THE_PARTS],
        "floors": {p.name: p.floor for p in THE_PARTS},
        "fits_that_squeezed_someone": sum(1 for r in rows if r["squeezed"]),
        "squeezed_how_often": dict(sorted(squeezed_counts.items())),
        "removed_entirely_how_often": dict(sorted(emptied_counts.items())),
        "characters_lost": dict(sorted(lost_totals.items())),
        "pays_most_often": max(squeezed_counts, key=squeezed_counts.get)
        if squeezed_counts
        else "",
    }


def forget_everything() -> None:
    with _LOCK:
        _LEDGER.clear()
