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
#: Doing first what the stuck step has only ever worked after (core/cognition/what_having_it_lets_her_do.py): a wall
#: that is an errand, learned from Minecraft's first hour, where bare hands get wood and wood makes the tool.
FIRST_WHAT_IT_WAITS_FOR = "do first what it waits for"
#: Where what her plans' steps waited for is kept between sittings.
_KEPT_AS = "the steps of her plans: what opens what"
_OPENS: list[Any] = []


def what_opens_what() -> Any:
    """Which steps of her plans have worked only once which others were done, as she has seen it across runs."""
    from core.cognition.what_having_it_lets_her_do import WhatOpensWhat

    if not _OPENS:
        try:
            from core.runtime.what_she_learned import recall

            _OPENS.append(WhatOpensWhat.from_memory(recall(_KEPT_AS).get("opens") or {}))
        except (ImportError, OSError, ValueError, TypeError, AttributeError) as exc:
            logger.info("what her plans' steps wait for was not read back (%s)", exc)
            _OPENS.append(WhatOpensWhat())
    return _OPENS[0]


def done_before(receipt: Any, stuck: str) -> list[str]:
    """The steps a run did and saw done before the one it got stuck on."""
    done: list[str] = []
    for result in getattr(receipt, "steps", ()) or ():
        name = str(getattr(result, "name", "") or "")
        if name == stuck:
            break
        if getattr(result, "ok", False) and getattr(result, "verified", False):
            done.append(name)
    return done


def learn_from_the_run(receipt: Any) -> None:
    """Each step of a run, what had been done before it, and whether it worked: kept, for the next plan that stalls.
    Blocking (it writes what she learned); called off the event loop."""
    opens = what_opens_what()
    done: list[str] = []
    for result in getattr(receipt, "steps", ()) or ():
        name = str(getattr(result, "name", "") or "")
        if not name:
            continue
        worked = bool(getattr(result, "ok", False) and getattr(result, "verified", False))
        opens.she_tried(name, holding=done, it_worked=worked)
        if worked:
            done.append(name)
    try:
        from core.runtime.what_she_learned import remember

        remember(_KEPT_AS, {"opens": opens.as_memory()})
    except (ImportError, OSError, ValueError, TypeError) as exc:
        logger.info("what her plans' steps wait for was not kept (%s)", exc)


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
        if getattr(result, "ok", True) is False:
            return str(getattr(result, "name", "") or "")
    return ""


def progress_of(receipt: Any) -> int:
    return int(getattr(receipt, "verified_progress", 0) or 0)


def _repairs(plan: Sequence[Any], stuck: str, done: Sequence[str] = ()) -> list[ActionOption]:
    """The repairs that make sense for this plan, stuck at this step, with ``done`` the steps seen done before it."""
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
    if where >= 0 and where < len(names) - 1 and getattr(plan[where], "optional", False):
        options.append(
            ActionOption(
                name=DROP,
                params={"step": stuck},
                detail=f"leave {stuck!r} out and carry on with what follows it",
                expectation=further,
            )
        )
    # What the stuck step has only ever worked after, where that is a step of this plan still to come.
    if where >= 0:
        for first in what_opens_what().why_it_will_not_work(stuck, holding=done):
            if first in names[where + 1:]:
                options.append(
                    ActionOption(
                        name=FIRST_WHAT_IT_WAITS_FOR,
                        params={"step": stuck, "first": first},
                        detail=f"do {first!r} first: {stuck!r} has worked only once {first!r} was done",
                        expectation=further,
                    )
                )
                break
    return options


def apply_repair(kind: str, plan: Sequence[Any], stuck: str, first: str = "") -> list[Any]:
    """The plan a repair produces. Never mutates the plan it was given."""
    steps = list(plan)
    names = [str(getattr(step, "name", "") or "") for step in steps]
    if kind in (RETRY, START_OVER) or stuck not in names:
        return steps
    if kind == FIRST_WHAT_IT_WAITS_FOR and first in names and names.index(first) > names.index(stuck):
        moved = steps.pop(names.index(first))
        steps.insert(names.index(stuck), moved)
        return steps
    where = names.index(stuck)
    if kind == DROP and getattr(steps[where], "optional", False) and where < len(steps) - 1:
        return steps[:where] + steps[where + 1 :]
    # Retained legacy repair names cannot weaken the task's original obligation.
    # An alternate implementation must still satisfy the required step.
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
    options = _repairs(steps, stuck, done_before(receipt, stuck))
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
        plan=apply_repair(kind, steps, stuck, str(chosen.chosen.params.get("first") or "")),
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
