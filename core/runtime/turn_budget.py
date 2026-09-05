"""Ceilings on a unit of work, and knowing before you spend rather than after.

`context_manager.token_budget` is a *context assembly* budget — how much text
to pack into a prompt. Nothing anywhere bounded the work itself. A task could
take four hundred steps, hold the resident 32B for an hour, and the only thing
that would eventually stop it was something else breaking.

That matters more on this host than on a rented one. The usual framing of
runaway agents is dollars, and Aura is local-first so most turns cost none. The
scarce resources here are the 20GB of wired memory the live model already holds
and the wall clock during which nothing else can have it. A loop that spends no
money can still take the machine away for an hour.

This is the complement to the stuck detector: that one catches work going
nowhere, this one catches work going somewhere *without end*. Neither subsumes
the other — a productive-looking search can expand forever, and a rut can be
cheap.

Two things it does that a plain counter does not:

* **It answers prospectively.** ``can_afford()`` lets a caller decline to
  *start* a step it cannot pay for. A retrospective-only budget always
  overshoots by exactly one step, and on a 32B one step is the expensive unit.
* **It warns before it stops.** Crossing the soft threshold is a signal to wind
  down — finish the thought, write the summary — rather than a guillotine
  mid-action. Being cut off mid-write is how a half-finished edit gets left on
  disk.

There are no default ceilings. An unset limit means unlimited, and
``Budget.unlimited()`` says so out loud, because a number nobody chose is worse
than no number: it looks like a decision and is not one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable

__all__ = [
    "BudgetAxis",
    "Budget",
    "Breach",
    "BudgetExceeded",
    "BudgetLedger",
]


class BudgetAxis(StrEnum):
    STEPS = "steps"
    TOKENS = "tokens"
    SECONDS = "seconds"
    COST_USD = "cost_usd"


class BudgetExceeded(RuntimeError):
    """Raised by ``spend`` when a ceiling would be crossed."""

    def __init__(self, breach: "Breach") -> None:
        super().__init__(breach.describe())
        self.breach = breach


@dataclass(frozen=True)
class Budget:
    """Ceilings for one unit of work. ``None`` on an axis means no ceiling.

    ``soft_fraction`` is where "wind down" begins — the point at which the work
    should start closing itself out rather than opening new threads.
    """

    max_steps: int | None = None
    max_tokens: int | None = None
    max_seconds: float | None = None
    max_cost_usd: float | None = None
    soft_fraction: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 < self.soft_fraction <= 1.0:
            raise ValueError("soft_fraction must be in (0, 1]")
        for name in ("max_steps", "max_tokens", "max_seconds", "max_cost_usd"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(
                    f"{name} must be positive, or None for no ceiling. Zero would "
                    "forbid the work outright, which is a different decision and "
                    "should be made somewhere it is visible."
                )

    @classmethod
    def unlimited(cls) -> "Budget":
        """No ceilings, stated deliberately rather than by omission."""
        return cls()

    @property
    def is_unlimited(self) -> bool:
        return all(
            getattr(self, name) is None
            for name in ("max_steps", "max_tokens", "max_seconds", "max_cost_usd")
        )

    def limit_for(self, axis: BudgetAxis) -> float | None:
        return {
            BudgetAxis.STEPS: self.max_steps,
            BudgetAxis.TOKENS: self.max_tokens,
            BudgetAxis.SECONDS: self.max_seconds,
            BudgetAxis.COST_USD: self.max_cost_usd,
        }[axis]


@dataclass(frozen=True)
class Breach:
    """Which ceiling, what it was, and what the work had reached."""

    axis: BudgetAxis
    limit: float
    used: float
    requested: float = 0.0

    def describe(self) -> str:
        projected = self.used + self.requested
        tail = (
            f" (a further {self.requested:g} would reach {projected:g})"
            if self.requested else ""
        )
        return (
            f"{self.axis} budget exhausted: {self.used:g} of {self.limit:g} used{tail}"
        )


@dataclass
class BudgetLedger:
    """Tracks spend against a budget, prospectively and retrospectively."""

    budget: Budget
    clock: Callable[[], float] = time.monotonic
    steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    _started: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._started = self.clock()

    # -- reading -----------------------------------------------------------

    @property
    def elapsed(self) -> float:
        return self.clock() - self._started

    def used(self, axis: BudgetAxis) -> float:
        return {
            BudgetAxis.STEPS: float(self.steps),
            BudgetAxis.TOKENS: float(self.tokens),
            BudgetAxis.SECONDS: self.elapsed,
            BudgetAxis.COST_USD: self.cost_usd,
        }[axis]

    def remaining(self, axis: BudgetAxis) -> float | None:
        limit = self.budget.limit_for(axis)
        if limit is None:
            return None
        return max(0.0, limit - self.used(axis))

    def utilization(self) -> dict[str, float]:
        """Fraction consumed per bounded axis. Unbounded axes are omitted —
        a ratio against infinity is not a number anyone can act on."""
        out: dict[str, float] = {}
        for axis in BudgetAxis:
            limit = self.budget.limit_for(axis)
            if limit is None:
                continue
            out[str(axis)] = self.used(axis) / limit
        return out

    # -- verdicts ----------------------------------------------------------

    def breach(self, *, steps: int = 0, tokens: int = 0, cost_usd: float = 0.0) -> Breach | None:
        """The first ceiling that is crossed, or would be by the given spend.

        With no arguments this is retrospective ("have I overrun?"). With them
        it is prospective ("can I afford this next step?").
        """
        requested = {
            BudgetAxis.STEPS: float(steps),
            BudgetAxis.TOKENS: float(tokens),
            BudgetAxis.SECONDS: 0.0,
            BudgetAxis.COST_USD: cost_usd,
        }
        for axis in BudgetAxis:
            limit = self.budget.limit_for(axis)
            if limit is None:
                continue
            used = self.used(axis)
            if used + requested[axis] > limit:
                return Breach(
                    axis=axis, limit=float(limit), used=used,
                    requested=requested[axis],
                )
        return None

    def can_afford(self, *, steps: int = 1, tokens: int = 0, cost_usd: float = 0.0) -> bool:
        """Whether the next step fits. Ask before paying, not after."""
        return self.breach(steps=steps, tokens=tokens, cost_usd=cost_usd) is None

    def exhausted_axis(self) -> BudgetAxis | None:
        """The first axis with no room left, if any.

        Distinct from ``breach()``, which asks whether a ceiling has been
        *crossed*. At exactly the limit nothing has been exceeded and yet
        nothing further can be afforded — conflating the two either lets a run
        take one step past its budget or reports an overrun that never
        happened.
        """
        for axis in BudgetAxis:
            if self.budget.limit_for(axis) is None:
                continue
            if self.remaining(axis) == 0:
                return axis
        return None

    @property
    def exhausted(self) -> bool:
        """No room left on some bounded axis. The condition to stop looping."""
        return self.exhausted_axis() is not None

    @property
    def winding_down(self) -> bool:
        """Past the soft threshold on any bounded axis.

        The signal to finish the thought and write the summary, while there is
        still budget to do it in. A guillotine mid-action leaves half-finished
        work on disk.
        """
        return any(
            fraction >= self.budget.soft_fraction
            for fraction in self.utilization().values()
        )

    # -- writing -----------------------------------------------------------

    def spend(self, *, steps: int = 1, tokens: int = 0, cost_usd: float = 0.0) -> None:
        """Record work done. Raises rather than silently overrunning.

        Charged *after* the fact deliberately: the true token cost of a step is
        not known until it returns. Callers that must not overrun at all should
        gate on ``can_afford`` first — that is what it is for.
        """
        breach = self.breach(steps=steps, tokens=tokens, cost_usd=cost_usd)
        if breach is not None:
            raise BudgetExceeded(breach)
        self.steps += steps
        self.tokens += tokens
        self.cost_usd += cost_usd

    def record(self, *, steps: int = 1, tokens: int = 0, cost_usd: float = 0.0) -> Breach | None:
        """Record work and report an overrun instead of raising.

        For callers that cannot unwind a step that already happened: the tokens
        were spent whether or not the ledger approves, and refusing to record
        them would make the ledger a worse record of reality than the log.
        """
        self.steps += steps
        self.tokens += tokens
        self.cost_usd += cost_usd
        return self.breach()

    def snapshot(self) -> dict[str, object]:
        """Everything a caller needs to explain why work stopped."""
        overrun = self.breach()
        spent = self.exhausted_axis()
        if overrun is not None:
            reason = overrun.describe()
        elif spent is not None:
            limit = self.budget.limit_for(spent)
            reason = f"{spent} budget exhausted: {self.used(spent):g} of {limit:g} used"
        else:
            reason = None
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 6),
            "elapsed_s": round(self.elapsed, 3),
            "utilization": {k: round(v, 4) for k, v in self.utilization().items()},
            "winding_down": self.winding_down,
            "exhausted": spent is not None,
            "breach": reason,
        }
