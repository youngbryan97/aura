"""Which status may become which, and whether a change actually applied.

The blind comparison revised its opinion of AutoGPT upward: the current
platform has typed execution contexts, persisted graph and node executions,
explicit legal status transitions, cancellation events, typed failure
classifications — and two details it singled out.

The first is that before running a node concurrently it makes a node-specific
copy of the execution context, so several nodes cannot mutate a shared one.
Aura has that shape already in ``AnExecutionContext.under``, which gives a
child token and a deadline that can only narrow; ``a_context_for_each`` is the
name for using it that way.

The second is subtler and is the reason this module exists: its tests
distinguish a transition that actually applied from a database predicate that
matched zero rows. Those are different facts and a boolean cannot hold both.
A caller told False cannot tell "that move is not allowed" from "somebody else
already moved it", and those need opposite responses — the first is a defect
to fix, the second is a race to accept.

So a change returns one of three things: it applied, it was refused because
the move is not legal, or nothing matched because the status was not what the
caller believed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable

logger = logging.getLogger("Aura.WhatAStatusMayBecome")

__all__ = [
    "AStatusChange",
    "ATransitionTable",
    "HowItWent",
    "a_context_for_each",
    "the_workflow_statuses",
]


class HowItWent(StrEnum):
    """The three outcomes. A boolean can hold two of them and loses the third."""

    APPLIED = "applied"
    #: The move is not legal from where it was. A defect in the caller.
    REFUSED = "refused"
    #: The status was not what the caller believed. A race, not a defect.
    NOTHING_MATCHED = "nothing matched"


@dataclass(frozen=True)
class AStatusChange:
    """What happened when a change was asked for."""

    went: HowItWent
    was: Any
    now: Any
    why: str = ""

    @property
    def applied(self) -> bool:
        return self.went is HowItWent.APPLIED

    def __bool__(self) -> bool:
        # Deliberate: `if change:` reads as "did it apply", which is the one
        # question a caller almost always means. The other two are told apart
        # by `went`, not by a second boolean.
        return self.applied


class ATransitionTable:
    """The legal moves for one kind of thing, and what is final."""

    def __init__(
        self,
        name: str,
        *,
        allowed: dict[Any, Iterable[Any]],
        terminal: Iterable[Any] = (),
    ) -> None:
        self.name = str(name)
        self._allowed = {
            status: frozenset(nexts) for status, nexts in allowed.items()
        }
        self._terminal = frozenset(terminal)
        for status in self._terminal:
            if self._allowed.get(status):
                raise ValueError(
                    f"{self.name}: {status} is terminal and also has moves out of it"
                )

    @property
    def terminal(self) -> frozenset:
        return self._terminal

    def may(self, was: Any, now: Any) -> bool:
        """Whether this move is legal at all."""
        if was in self._terminal:
            return False
        return now in self._allowed.get(was, frozenset())

    def change(self, believed: Any, actual: Any, wanted: Any) -> AStatusChange:
        """Move from ``believed`` to ``wanted``, given the status really is ``actual``.

        Three answers, because there are three things that can be true.
        """
        if believed != actual:
            return AStatusChange(
                went=HowItWent.NOTHING_MATCHED,
                was=actual,
                now=actual,
                why=(
                    f"expected {believed} and found {actual}; "
                    "somebody else moved it"
                ),
            )
        if actual in self._terminal:
            return AStatusChange(
                went=HowItWent.REFUSED,
                was=actual,
                now=actual,
                why=f"{actual} is final and cannot become {wanted}",
            )
        if not self.may(actual, wanted):
            return AStatusChange(
                went=HowItWent.REFUSED,
                was=actual,
                now=actual,
                why=f"{actual} may not become {wanted}",
            )
        return AStatusChange(went=HowItWent.APPLIED, was=actual, now=wanted)

    def report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "statuses": sorted(str(one) for one in self._allowed),
            "terminal": sorted(str(one) for one in self._terminal),
            "moves": {
                str(status): sorted(str(one) for one in nexts)
                for status, nexts in sorted(
                    self._allowed.items(), key=lambda pair: str(pair[0])
                )
            },
        }


def the_workflow_statuses() -> ATransitionTable:
    """The table for ``durable_workflow.WorkflowStatus``.

    Six statuses that had no declared transitions: nothing said a completed
    workflow could not go back to running, so nothing could refuse it.
    """
    from core.runtime.durable_workflow import WorkflowStatus as S

    return ATransitionTable(
        "workflow",
        allowed={
            S.PENDING: (S.RUNNING, S.CANCELED),
            S.RUNNING: (S.PAUSED_FOR_APPROVAL, S.COMPLETED, S.FAILED, S.CANCELED),
            # An approval can be granted, refused, or overtaken by a shutdown.
            S.PAUSED_FOR_APPROVAL: (S.RUNNING, S.CANCELED, S.FAILED),
            # A failed workflow can be run again from its checkpoint. That is
            # what a durable workflow is for, and calling FAILED terminal here
            # made `resume` refuse every workflow it exists to rescue —
            # caught by the resume test, which had been asserting it for
            # longer than this table has existed.
            S.FAILED: (S.RUNNING, S.CANCELED),
        },
        # Completed and canceled are endings somebody reached on purpose.
        # Failed is a place work stops, not a place it has to stay.
        terminal=(S.COMPLETED, S.CANCELED),
    )


def a_context_for_each(context: Any, nodes: Iterable[str]) -> dict[str, Any]:
    """One execution context per node, so none of them share a token.

    The detail the review liked in AutoGPT. Each child carries its own stop
    signal and inherits the parent's deadline, which can narrow and never
    widen — so cancelling one node does not cancel its siblings, and no node
    can give itself longer than the turn it belongs to.
    """
    return {str(one): context.under(f"node:{one}") for one in nodes}
