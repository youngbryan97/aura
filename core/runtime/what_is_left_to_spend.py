"""A budget a child spends from its parent, and cannot spend past.

Voyager's closure asked for a hierarchical BudgetContext: a root turn or task
budget, child budgets for model calls, tools and retries, and every nested
retry consuming the parent so it cannot escape it.

The property that matters is the last one. A retry limit per call is not a
budget — three calls with three retries each is nine attempts, and the caller
who allowed three has no way to say so. What a caller can say is how much the
whole thing may cost, and everything under it spends from that.

So a budget is a tree. A child is opened against its parent, takes at most
what the parent has left, and every spend travels up. A child that asks for
more than remains gets what remains, which is the same rule
``AnExecutionContext`` already uses for deadlines: they narrow and never
widen.

Refusing rather than raising, deliberately. A retry that cannot happen is a
decision the caller has to make — sometimes the right answer is to answer with
less — and an exception there would make every budget check a try/except.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.WhatIsLeftToSpend")

__all__ = ["ABudget", "a_budget_of"]


@dataclass
class ABudget:
    """How much of something is left, and who else is spending it."""

    what: str
    allowed: float
    spent: float = 0.0
    parent: "ABudget | None" = None
    children: list["ABudget"] = field(default_factory=list)
    #: Every refusal, so "it stopped early" has an answer.
    refusals: list[str] = field(default_factory=list)

    @property
    def left(self) -> float:
        return max(0.0, self.allowed - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.left <= 0.0

    def under(self, what: str, *, at_most: float) -> "ABudget":
        """A child budget. Takes at most what is left here.

        A child asking for more than remains gets what remains. That is the
        same rule the execution context uses for deadlines, and for the same
        reason: a caller cannot be allowed to grant itself more than it was
        given.
        """
        child = ABudget(
            what=str(what),
            allowed=min(float(at_most), self.left),
            parent=self,
        )
        self.children.append(child)
        return child

    def spend(self, how_much: float = 1.0, *, on: str = "") -> bool:
        """Spend, if there is room. Returns whether it happened.

        Every spend travels up, so a leaf spending exhausts the root. Refusing
        rather than raising: a retry that cannot happen is a decision, and an
        exception here would make every check a try/except.
        """
        amount = float(how_much)
        if amount <= 0:
            return True
        if amount > self.left:
            why = (
                f"{on or self.what} wanted {amount:g} of {self.what} and "
                f"{self.left:g} was left"
            )
            self.refusals.append(why)
            logger.debug("%s", why)
            return False
        # Up first. If an ancestor refuses, nothing here has been spent.
        if self.parent is not None and not self.parent.spend(amount, on=on or self.what):
            self.refusals.append(
                f"{on or self.what} was refused by {self.parent.what}"
            )
            return False
        self.spent += amount
        return True

    def report(self) -> dict[str, Any]:
        return {
            "what": self.what,
            "allowed": self.allowed,
            "spent": self.spent,
            "left": self.left,
            "refusals": list(self.refusals),
            "children": [one.report() for one in self.children],
        }

    def everything_refused(self) -> list[str]:
        """Every refusal in the tree, so "it stopped early" has an answer."""
        found = list(self.refusals)
        for one in self.children:
            found.extend(one.everything_refused())
        return found


def a_budget_of(what: str, at_most: float) -> ABudget:
    """The root of a budget tree — a turn, or a task."""
    return ABudget(what=str(what), allowed=float(at_most))
