"""What a bar's rise or fall is to her: a gain, a loss, or nothing, and how much she has left.

core/perception/how_full_a_bar_is.py finds the strips that fill and empty and
how far each ran. This says what each change means for her: the words beside a
bar say what it measures ("health", "paint", "time"), and a fall of what she
has left is a loss as a counter's fall is; a bar with no words is taken for
something running out, worse lower, until play says otherwise. The lowest of
what she has left is how careful she is (core/agency/playing_as_it_happens.py
gives what costs her a wider berth below CAREFUL_BELOW).

Nothing here knows a game.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

__all__ = ["CAREFUL_BELOW", "bars_read", "what_a_bar_measures", "what_is_left"]

#: How full her bars may be before she is careful: below this share of their fullest, what costs is given wider berth.
CAREFUL_BELOW = 0.5


def what_a_bar_measures(bar: Any, regions: list[dict[str, Any]], wide: int, tall: int) -> tuple[str, str]:
    """The words written nearest a bar, before it on its line or just over it, and what a change in them means; a bar
    with no words is taken for something running out, worse lower."""
    from core.agency.what_meeting_things_does import _meaning

    left, top, right, bottom = bar.where(wide, tall)
    near = []
    for region in regions:
        said = " ".join(str(region.get("text") or "").split())
        x, y = float(region.get("center_x", -1.0)), float(region.get("center_y", -1.0))
        if not any(ch.isalpha() for ch in said):
            continue
        before = abs(y - (top + bottom) / 2) <= 0.06 and -0.02 <= left - x <= 0.25
        over = 0.0 <= top - y <= 0.1 and left - 0.05 <= x <= right + 0.05
        if before or over:
            near.append((abs(y - (top + bottom) / 2) + abs(x - left), said))
    label = min(near)[1] if near else ""
    return label, (_meaning(label) if label else "") or "down is bad"


def bars_read(bars: Any, picture: Any, at: float, regions: list[dict[str, Any]],
              place: Callable[[float, float], str]) -> Iterator[dict[str, Any]]:
    """Every bar's rise or fall in one picture, as a verdict: a "gain" or a "loss" as its words say, named for its words
    or, without any, for ``place`` (where on the screen a point, as shares of it, is)."""
    for change in bars.read(picture, at):
        bar = change["bar"]
        tall, wide = bars.shape
        if not bar.meaning:
            bar.label, bar.meaning = what_a_bar_measures(bar, regions, wide, tall)
        if bar.meaning == "neither":
            continue
        fell = change["to"] < change["from"]
        lost = fell if bar.meaning == "down is bad" else not fell
        named = f"the {bar.label} bar" if bar.label else f"the bar at the {place((bar.start + bar.end) / 2 / max(1, wide), bar.row / max(1, tall))}"
        yield {"what": "loss" if lost else "gain", "at": at, "since": change["since"], "counter": named, "by": -1 if lost else 1}


def what_is_left(bars: Any) -> float:
    """The lowest of what she has left, as a share of its fullest: 1 where no bar says."""
    left = [bar.full for bar in bars.bars() if bar.meaning == "down is bad"]
    return min(left) if left else 1.0
