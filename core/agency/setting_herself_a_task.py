"""The next thing to practise in a world, chosen from what is in front of her and how she has done.

SIMA 2's task setter proposes tasks "likely to be achievable from the current
state" and steers toward what the agent is weak at (arXiv 2512.04797, §4.5).
Voyager's automatic curriculum does the same from the agent's own record, and
without it Voyager found 93% fewer things. Both ask a large model to write the
task. Here the task is read off the screen and her record, because both are
facts: what is in view is what she can reach from here, and how each attempt
went is kept in `what_she_tried`.

What is worth practising most is what she might or might not manage. A task
never tried says nothing yet and comes first. Among tried ones, the one whose
record is closest to even odds is where one more attempt tells her most; one
she always manages or never manages tells her least. That is the measure of
how much an attempt can still change what she knows, written as the spread of
a coin with those odds: largest at even, nothing at either end.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["things_in_view", "what_to_practise"]

def things_in_view(layout: Sequence[dict[str, Any]], *, title: str = "") -> list[str]:
    """Names on screen that read as things, whole, other than the window's own title."""
    from core.agency.going_to_what_she_sees import reads_as_a_thing  # noqa: PLC0415

    found: list[str] = []
    for region in layout or ():
        said = " ".join(str(region.get("text", "")).split())
        if not said or said.lower() == str(title or "").strip().lower() or not reads_as_a_thing(region):
            continue
        name = said.lower()
        if name not in found:
            found.append(name)
    return found


def _worth_another_try(tried: int, worked: int) -> float:
    """How much one more attempt can still tell her: the spread of a coin with her odds on it."""
    if tried <= 0:
        return 1.0
    odds = worked / tried
    return odds * (1.0 - odds)


def what_to_practise(
    layout: Sequence[dict[str, Any]],
    record: Mapping[str, tuple[int, int]],
    *,
    title: str = "",
) -> str:
    """The task most worth trying next from here, or empty when nothing in view is a thing.

    ``record`` is `what_she_tried.how_it_goes` for this world: per task, tries
    and successes. Ties go to the thing tried least.
    """
    candidates = [f"go to the {thing}" for thing in things_in_view(layout, title=title)]
    if not candidates:
        return ""

    def rank(task: str) -> tuple[float, int]:
        tried, worked = record.get(task, (0, 0))
        return (_worth_another_try(tried, worked), -tried)

    best = max(candidates, key=rank)
    # Something always managed or never managed has nothing left to teach
    # from here. Live, with only the chest she had just opened in view, she
    # tried it again rather than turn to find the door she had not tried.
    return best if rank(best)[0] > 0.0 else ""
