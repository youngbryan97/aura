"""core/cognition/tool_plan.py — a program over tools, not a conversation about them.

Tool coordination in Aura is a sequence of model turns. To read six pages and
compare them, the model calls a tool, receives the whole result into its
context, decides, calls the next. Six round trips, six results in the window,
and the model spends most of its attention on data it is passing through rather
than reasoning about.

A tool plan is a small typed program: map, filter, join, assert, retry, branch,
loop over typed tool results. The model authors it; the executor runs it; only
what the plan RETURNS reaches the context. That is the whole saving, and it is
measurable — :class:`Execution` records round trips and bytes returned against
what the sequential path would have cost.

Governance is not in the plan
-----------------------------
Every tool call still goes through whatever authority normally decides. A plan
cannot call a tool the caller could not call, cannot escalate, and cannot
retry past its declared bound. The IR is a way to spend fewer model turns, not
a way around the gateway, and :meth:`Executor.run` takes the permitted tool set
as an argument rather than reading one.

Assert is the interesting operator
----------------------------------
``Assert`` fails the plan rather than the step. A plan that filters six results
to zero and carries on returns an empty answer that looks like a finding; one
that asserts non-empty returns a failure that looks like a failure. Making that
explicit in the plan is what stops "no results" being reported as "no such
thing".
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = [
    "Op",
    "Step",
    "Plan",
    "Execution",
    "Executor",
    "PlanRefused",
    "PlanFailed",
]


class Op(StrEnum):
    CALL = "call"
    MAP = "map"
    FILTER = "filter"
    JOIN = "join"
    ASSERT = "assert"
    RETRY = "retry"
    BRANCH = "branch"
    LOOP = "loop"
    RETURN = "return"


class PlanRefused(PermissionError):
    """A plan reached for a tool the caller was not permitted."""


class PlanFailed(RuntimeError):
    """An assertion in the plan did not hold."""


@dataclass(frozen=True, slots=True)
class Step:
    """One instruction. ``into`` names the binding its result lands in."""

    op: Op
    into: str = ""
    tool: str = ""
    args: Mapping[str, Any] = field(default_factory=dict)
    over: str = ""
    #: For MAP: when ``tool`` is set the mapped function IS a tool call, one per
    #: item, counted and bounded. ``args`` names the parameter the item fills.
    fn: Callable[..., Any] | None = None
    message: str = ""
    attempts: int = 1
    body: tuple[Step, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op.value, "into": self.into, "tool": self.tool,
            "over": self.over, "attempts": self.attempts, "body": len(self.body),
        }


@dataclass(frozen=True, slots=True)
class Plan:
    """A typed program over tools, authored by the model and run by the executor."""

    name: str
    steps: tuple[Step, ...]
    #: Tools this plan declares it will use. Checked against what the caller
    #: permits before anything runs, so a refusal costs nothing.
    uses: frozenset[str] = frozenset()
    max_calls: int = 50

    def declared_tools(self) -> frozenset[str]:
        def walk(steps: Sequence[Step]) -> set[str]:
            found: set[str] = set()
            for step in steps:
                if step.tool:
                    found.add(step.tool)
                found |= walk(step.body)
            return found

        return frozenset(walk(self.steps)) | self.uses


@dataclass(frozen=True, slots=True)
class Execution:
    """What running the plan cost, against what the sequential path would have."""

    plan: str
    returned: Any
    tool_calls: int
    bytes_returned: int
    bytes_through_tools: int
    seconds: float
    round_trips: int = 1
    failed: str = ""

    @property
    def sequential_round_trips(self) -> int:
        """One model turn per tool call is what the plan replaces."""
        return self.tool_calls

    @property
    def context_saved(self) -> int:
        return max(0, self.bytes_through_tools - self.bytes_returned)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan,
            "tool_calls": self.tool_calls,
            "round_trips": self.round_trips,
            "sequential_round_trips": self.sequential_round_trips,
            "bytes_returned": self.bytes_returned,
            "bytes_through_tools": self.bytes_through_tools,
            "context_saved": self.context_saved,
            "seconds": self.seconds,
            "failed": self.failed,
        }


def _size(value: Any) -> int:
    try:
        return len(repr(value))
    except Exception:  # noqa: BLE001
        return 0


class Executor:
    """Runs a plan under the caller's own permissions, and never widens them."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.cognition.tool_plan.Executor", reentrant=True)
        self._executions: list[Execution] = []

    def run(
        self,
        plan: Plan,
        *,
        tools: Mapping[str, Callable[..., Any]],
        permitted: frozenset[str],
        bindings: Mapping[str, Any] | None = None,
    ) -> Execution:
        """Execute, refusing before anything runs if the plan overreaches."""
        declared = plan.declared_tools()
        overreach = declared - permitted
        if overreach:
            raise PlanRefused(
                f"plan {plan.name!r} reaches for {sorted(overreach)}, which the caller is "
                "not permitted; a plan is a way to spend fewer model turns, not a way "
                "around the gateway"
            )

        env: dict[str, Any] = dict(bindings or {})
        started = time.monotonic()
        calls = 0
        through = 0
        returned: Any = None
        failure = ""

        def call(tool: str, args: Mapping[str, Any]) -> Any:
            nonlocal calls, through
            if calls >= plan.max_calls:
                raise PlanFailed(f"plan {plan.name!r} exceeded {plan.max_calls} tool calls")
            calls += 1
            result = tools[tool](**args)
            through += _size(result)
            return result

        def execute(steps: Sequence[Step]) -> Any:
            nonlocal returned
            for step in steps:
                if step.op is Op.CALL:
                    env[step.into] = call(step.tool, {
                        k: env.get(v[1:], v) if isinstance(v, str) and v.startswith("$") else v
                        for k, v in step.args.items()
                    })
                elif step.op is Op.MAP:
                    items = env.get(step.over, [])
                    if step.tool:
                        # A tool call per item, counted and bounded like any other.
                        # Doing this through the plan rather than through a closure
                        # is what makes the round-trip saving measurable.
                        argument = next(iter(step.args), "value")
                        env[step.into] = [call(step.tool, {argument: x}) for x in items]
                    else:
                        env[step.into] = [step.fn(x) for x in items]
                elif step.op is Op.FILTER:
                    env[step.into] = [x for x in env.get(step.over, []) if step.fn(x)]
                elif step.op is Op.JOIN:
                    env[step.into] = step.fn(*(env.get(name) for name in step.over.split(",")))
                elif step.op is Op.ASSERT:
                    if not step.fn(env.get(step.over)):
                        raise PlanFailed(step.message or f"assertion on {step.over!r} failed")
                elif step.op is Op.RETRY:
                    last: Exception | None = None
                    for _ in range(max(1, step.attempts)):
                        try:
                            execute(step.body)
                            last = None
                            break
                        except Exception as exc:  # noqa: BLE001 - retry is the point
                            last = exc
                    if last is not None:
                        raise last
                elif step.op is Op.BRANCH:
                    if step.fn(env.get(step.over)):
                        execute(step.body)
                elif step.op is Op.LOOP:
                    for item in env.get(step.over, []):
                        env[step.into or "item"] = item
                        execute(step.body)
                elif step.op is Op.RETURN:
                    returned = env.get(step.over)
                    return returned
            return returned

        try:
            execute(plan.steps)
        except PlanFailed as exc:
            failure = str(exc)
        except PlanRefused:
            raise
        except Exception as exc:  # noqa: BLE001
            failure = f"{type(exc).__name__}: {exc}"

        execution = Execution(
            plan=plan.name, returned=returned, tool_calls=calls,
            bytes_returned=_size(returned), bytes_through_tools=through,
            seconds=time.monotonic() - started, failed=failure,
        )
        with self._lock:
            self._executions.append(execution)
        return execution

    def report(self) -> dict[str, Any]:
        with self._lock:
            executions = list(self._executions)
        if not executions:
            return {"executions": 0}
        return {
            "executions": len(executions),
            "tool_calls": sum(e.tool_calls for e in executions),
            "round_trips": sum(e.round_trips for e in executions),
            "sequential_round_trips": sum(e.sequential_round_trips for e in executions),
            "round_trips_saved": sum(
                e.sequential_round_trips - e.round_trips for e in executions
            ),
            "context_saved": sum(e.context_saved for e in executions),
            "failures": [e.failed for e in executions if e.failed],
        }
