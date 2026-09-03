"""When to commit to something there is no taking back.

The end of a game of Cluedo. Two buttons: End Turn, and Final Accusation —
under a line of text saying that if the accusation is wrong, the game is over.
Nobody presses it because they are confident. They press it when the cost of
waiting has caught up with the cost of being wrong, and those are different
sizes: being wrong ends everything, and waiting costs one turn and the chance
somebody else gets there first.

She does irreversible things constantly. Sending, deleting, publishing,
deploying, and answering — an answer is irreversible in the way that matters,
because the person acts on it. Every one of those had the same shape of
decision behind it and none of them had this: she either did the thing or
asked permission, and how sure she needed to be did not depend on what being
wrong would cost.

It is not a confidence level. Needing to be ninety per cent sure is the same
number whether the mistake costs a retry or a career, and that cannot be
right. What settles it is a comparison: what committing now costs if it is
wrong, against what waiting costs — and waiting is never free, or she would
wait for ever and that is its own way of failing.

So there are three numbers and all three come from the situation rather than
from here. How sure she is. What being wrong would cost. What another look
would cost, which includes the chance that waiting loses the thing anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["WhetherToCommit", "whether_to_say_it"]


@dataclass(frozen=True)
class WhetherToCommit:
    """Whether to do the thing there is no taking back, and why."""

    #: What it costs to commit now, given she might be wrong.
    committing: float
    #: What it costs to look again first.
    waiting: float
    how_sure: float
    why: str

    @property
    def now(self) -> bool:
        return self.committing <= self.waiting

    def describe(self) -> str:
        said = "now" if self.now else "not yet"
        return (
            f"{said}: {self.how_sure:.0%} sure, "
            f"committing costs {self.committing:.2f} against {self.waiting:.2f} to wait "
            f"({self.why})"
        )


def whether_to_say_it(
    *,
    how_sure: float,
    being_wrong_costs: float,
    another_look_costs: float,
    waiting_might_lose_it: float = 0.0,
    what_it_is_worth: float = 1.0,
) -> WhetherToCommit:
    """Weigh committing now against looking again.

    ``how_sure`` is what she makes the chance of being right. ``being_wrong_costs``
    and ``another_look_costs`` are in whatever units the situation measures
    itself in — they only ever meet each other here, so nothing needs
    converting.

    ``waiting_might_lose_it`` is the chance that looking again loses the thing
    anyway: somebody else answers first, the window closes, the page changes.
    Without it, waiting always looks cheaper than a bad commitment and she
    never commits at all — which is not caution, it is a different way of
    getting it wrong, and it is the one that looks responsible while it
    happens.
    """
    sure = max(0.0, min(1.0, float(how_sure)))
    committing = (1.0 - sure) * float(being_wrong_costs)
    waiting = float(another_look_costs) + max(0.0, min(1.0, float(waiting_might_lose_it))) * float(
        what_it_is_worth
    )
    if committing <= waiting:
        why = (
            "being wrong costs little enough"
            if sure < 0.9
            else "sure enough that waiting is the more expensive mistake"
        )
    else:
        why = (
            "not sure enough for what being wrong would cost"
            if being_wrong_costs > another_look_costs
            else "another look is cheap"
        )
    return WhetherToCommit(
        committing=committing, waiting=waiting, how_sure=sure, why=why
    )
