"""Changing a plan that has stopped working, without inventing new machinery.

:class:`GoalPursuitEngine` re-plans when a run stalls, and took the new plan
from an injected callable — the same hole :mod:`core.agency.deliberate_action`
closed on the other loop. This fills it for plans.

A plan is a list of steps carrying real callables, so nothing here asks a
model to write one. What a stalled run actually needs is narrower and can be
chosen from a closed set: try the failing step again, go around it, drop it
and carry on with the rest, or start over. Each repair names a step that
really exists in the plan that really failed, so a repair is always something
the run can do.

The choice runs through :func:`deliberate`, which means a repair arrives with
a prediction — the run gets further than it did — and the next receipt grades
it. A repair that does not move the run further is recorded as not having
worked, and the next stall reads that.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from core.agency.deliberate_action import (
    ActionOption,
    Attempt,
    Deliberation,
    Expectation,
    deliberate,
)

logger = logging.getLogger(__name__)

#: Repairs available to any stalled plan, whatever the plan is made of.
RETRY = "retry"
GO_AROUND = "go around"
DROP = "drop"
START_OVER = "start over"


@dataclass(frozen=True)
class Repair:
    """A change to a plan, and the plan it produces."""

    kind: str
    step: str
    plan: list[Any]
    deliberation: Deliberation | None = None

    def narrate(self) -> str:
        if self.deliberation is not None:
            return self.deliberation.narrate()
        return f"{self.kind} {self.step}".strip()


def failed_step(receipt: Any) -> str:
    """The step a run got stuck on, by name."""
    for result in getattr(receipt, "steps", ()) or ():
        if getattr(result, "ok", True) is False or getattr(result, "verified", True) is False:
            return str(getattr(result, "name", "") or "")
    return ""


def progress_of(receipt: Any) -> int:
    return int(getattr(receipt, "verified_progress", 0) or 0)


def _repairs(plan: Sequence[Any], stuck: str) -> list[ActionOption]:
    """The repairs that make sense for this plan, stuck at this step."""
    names = [str(getattr(step, "name", "") or "") for step in plan]
    if stuck not in names:
        stuck = names[0] if names else ""
    further = Expectation(changed=True, describes="the run to get further than it did")
    options = [
        ActionOption(
            name=RETRY,
            params={"step": stuck},
            detail=f"try {stuck!r} again from the top of the plan",
            expectation=further,
        ),
        ActionOption(
            name=START_OVER,
            params={"step": stuck},
            detail="run the whole plan again from the beginning",
            expectation=further,
        ),
    ]
    where = names.index(stuck) if stuck in names else -1
    if where >= 0 and where < len(names) - 1:
        options.append(
            ActionOption(
                name=DROP,
                params={"step": stuck},
                detail=f"leave {stuck!r} out and carry on with what follows it",
                expectation=further,
            )
        )
    if where >= 0 and getattr(plan[where], "optional", False) is False:
        options.append(
            ActionOption(
                name=GO_AROUND,
                params={"step": stuck},
                detail=f"mark {stuck!r} as one the run may finish without",
                expectation=further,
            )
        )
    return options


def apply_repair(kind: str, plan: Sequence[Any], stuck: str) -> list[Any]:
    """The plan a repair produces. Never mutates the plan it was given."""
    steps = list(plan)
    names = [str(getattr(step, "name", "") or "") for step in steps]
    if kind in (RETRY, START_OVER) or stuck not in names:
        return steps
    where = names.index(stuck)
    if kind == DROP:
        return steps[:where] + steps[where + 1 :]
    if kind == GO_AROUND:
        from dataclasses import replace  # noqa: PLC0415

        try:
            steps[where] = replace(steps[where], optional=True)
        except (TypeError, ValueError):
            return steps
        return steps
    return steps


async def replan(
    goal: str,
    plan: Sequence[Any],
    receipt: Any,
    *,
    think: Any,
    history: Sequence[Attempt] = (),
    stakes: float = 0.5,
    lived: bool = True,
    spine: Any = None,
    graph: Any = None,
) -> Repair | None:
    """Choose a repair for a plan that stalled, or None when there is none.

    Returning None is a real answer. A run that cannot say what to change is
    better stopped than restarted on the same plan that just failed.
    """
    steps = list(plan)
    if not steps:
        return None
    stuck = failed_step(receipt)
    options = _repairs(steps, stuck)
    if not options:
        return None

    situation = (
        f"The plan stalled at {stuck!r} after {progress_of(receipt)} verified step(s) "
        f"of {len(steps)}."
    )
    chosen = await deliberate(
        goal,
        situation,
        options,
        think=think,
        history=history,
        stakes=stakes,
        control_point="agency.replan",
        lived=lived,
        spine=spine,
        graph=graph,
    )
    if not chosen.reached:
        logger.info("🛑 [Replan] no repair for '%s': %s", goal, chosen.reason)
        return None
    kind = chosen.chosen.name
    return Repair(
        kind=kind,
        step=str(chosen.chosen.params.get("step") or stuck),
        plan=apply_repair(kind, steps, stuck),
        deliberation=chosen,
    )


def replanner(
    goal: str,
    plan: Sequence[Any],
    *,
    think: Any,
    stakes: float = 0.5,
    lived: bool = True,
    spine: Any = None,
    graph: Any = None,
):
    """A ``replan`` callable shaped for :meth:`GoalPursuitEngine.pursue`.

    The engine hands a receipt and wants a plan back. This keeps the plan it
    started from and the repairs already tried, so a second stall does not
    choose the repair that just failed.
    """
    state: dict[str, Any] = {"plan": list(plan), "history": [], "last": None, "progress": -1}

    async def _replan(receipt: Any) -> list[Any] | None:
        # Grade the repair already tried before choosing another one.
        #
        # This receipt is the only evidence about the LAST repair, so it is
        # graded first and attributed to that repair. Recording it against the
        # repair chosen next would credit a change for an outcome that
        # happened before it was made.
        last: Repair | None = state["last"]
        if last is not None:
            state["history"].append(
                Attempt(
                    option=last.kind,
                    expected=(
                        last.deliberation.chosen.expectation.describes
                        if last.deliberation is not None and last.deliberation.chosen is not None
                        else "the run to get further"
                    ),
                    verdict=_graded(receipt, state),
                )
            )
        else:
            state["progress"] = progress_of(receipt)

        repair = await replan(
            goal,
            state["plan"],
            receipt,
            think=think,
            history=state["history"],
            stakes=stakes,
            lived=lived,
            spine=spine,
            graph=graph,
        )
        if repair is None:
            return None
        state["last"] = repair
        state["plan"] = repair.plan
        return repair.plan

    return _replan


def _graded(receipt: Any, state: dict[str, Any]):
    """Whether the last repair got the run further than the one before it."""
    from core.agency.deliberate_action import Verdict  # noqa: PLC0415

    made = progress_of(receipt)
    before = int(state.get("progress", -1))
    state["progress"] = made
    moved = made > before
    return Verdict(held=moved, observed_change=moved, stalled=not moved)
