"""What an answer has to satisfy before it is served, and what a retry costs.

CrewAI's closure asked for an output guardrail protocol with a retry budget.
Aura checks its answers in several places and none of them says, in one
sentence, what an answer must satisfy — so a reader cannot tell whether a
particular check runs, and a failed check has no agreed way to ask for another
attempt.

Two things make this more than a validation chain.

A guardrail says why it refused, in words that could be handed back to the
thing that produced the answer. "Refused" is not actionable; "the reply names
a file that does not exist" is. A guardrail with no reason is refused at
registration.

And a retry spends from a budget the caller owns rather than from a counter
the guardrail keeps. Three guardrails with three retries each is nine
attempts, and the caller who allowed three never said so — the same defect the
budget tree exists for, which is why it is the same budget.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger("Aura.WhatAnAnswerMustPass")

__all__ = [
    "AGuardrail",
    "AVerdict",
    "TheGuardrails",
    "a_guardrail",
]


@dataclass(frozen=True)
class AVerdict:
    """What one guardrail thought of an answer."""

    passed: bool
    why: str = ""
    #: What the producer could do differently. Empty when there is nothing
    #: useful to say, which is itself worth seeing.
    instead: str = ""

    def __bool__(self) -> bool:
        return self.passed


@runtime_checkable
class AGuardrail(Protocol):
    """Something an answer has to satisfy."""

    name: str

    def check(self, answer: Any) -> AVerdict:
        ...


@dataclass
class _AGuardrail:
    name: str
    look: Callable[[Any], AVerdict]

    def check(self, answer: Any) -> AVerdict:
        return self.look(answer)


def a_guardrail(name: str, look: Callable[[Any], AVerdict]) -> AGuardrail:
    """Make a guardrail from a function that returns a verdict."""
    if not str(name).strip():
        raise ValueError("a guardrail needs a name; a refusal from an unnamed one "
                         "cannot be acted on")
    return _AGuardrail(name=str(name), look=look)


@dataclass
class TheGuardrails:
    """Every check an answer must pass, in the order they are applied."""

    rails: list[AGuardrail] = field(default_factory=list)
    #: Every refusal, with which rail and why.
    refusals: list[dict[str, Any]] = field(default_factory=list)
    attempts: int = 0

    def add(self, rail: AGuardrail) -> "TheGuardrails":
        self.rails.append(rail)
        return self

    def check(self, answer: Any) -> AVerdict:
        """The first refusal, or a pass. Stops at the first no.

        Stopping is deliberate: a producer given five reasons at once fixes
        the first and gets the second, and five round trips cost what one
        would have if it had been told the first alone.
        """
        for rail in self.rails:
            try:
                said = rail.check(answer)
            except Exception as exc:  # noqa: BLE001 — a broken rail is not a refusal
                logger.warning("guardrail %s raised: %s", rail.name, exc)
                continue
            if not said:
                self.refusals.append(
                    {"rail": rail.name, "why": said.why, "instead": said.instead}
                )
                return said
        return AVerdict(passed=True)

    def until_it_passes(
        self,
        produce: Callable[[str], Any],
        *,
        budget: Any,
        on: str = "a guarded answer",
    ) -> tuple[Any, AVerdict]:
        """Ask for an answer until one passes, spending from ``budget``.

        The budget is the caller's, not a counter kept here. Three guardrails
        with three retries each is nine attempts, and the caller who allowed
        three never said so.

        Returns the last answer and the last verdict, so a caller that runs
        out of budget still has something to serve and knows why it is not
        good enough.
        """
        answer: Any = None
        said = AVerdict(passed=False, why="nothing was produced")
        instead = ""
        while budget.spend(1, on=on):
            self.attempts += 1
            answer = produce(instead)
            said = self.check(answer)
            if said:
                return answer, said
            instead = said.instead or said.why
        return answer, said

    def report(self) -> dict[str, Any]:
        return {
            "rails": [one.name for one in self.rails],
            "attempts": self.attempts,
            "refusals": list(self.refusals),
        }
