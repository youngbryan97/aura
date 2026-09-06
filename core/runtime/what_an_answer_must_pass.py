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

**A rail says what its own failure means.** The first version caught every
exception from a check and carried on, on the reasoning that a broken rail is
not a refusal. That is right for a rail whose job is quality and wrong for one
whose job is safety: a check that cannot run has established nothing, and
treating "it crashed" as "it passed" is how the one rail that mattered stops
mattering. So each rail declares what happens when it cannot answer — carry
on, refuse, abstain and say so, or escalate — and the default for a rail that
does not say is to refuse, because the safe default has to be the one you get
by not thinking about it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger("Aura.WhatAnAnswerMustPass")

__all__ = [
    "AGuardrail",
    "AVerdict",
    "TheGuardrails",
    "WhenItCannotAnswer",
    "a_guardrail",
]


class WhenItCannotAnswer(StrEnum):
    """What a rail's own failure means for the answer it was checking."""

    #: The check was about quality. Losing it costs a little polish.
    CARRY_ON = "carry on"
    #: The check was the reason the answer is allowed out. A check that
    #: cannot run has established nothing.
    REFUSE = "refuse"
    #: Neither. The answer goes out and the report says this rail did not
    #: run, so nobody later reads silence as a pass.
    ABSTAIN = "abstain"
    #: A rail failing is itself the finding. Refuse, and record a degradation
    #: rather than a log line.
    ESCALATE = "escalate"


@dataclass(frozen=True)
class AVerdict:
    """What one guardrail thought of an answer."""

    passed: bool
    why: str = ""
    #: What the producer could do differently. Empty when there is nothing
    #: useful to say, which is itself worth seeing.
    instead: str = ""
    #: True where this verdict came from a rail that could not run, rather
    #: than from a rail that looked and decided.
    from_a_broken_rail: bool = False

    def __bool__(self) -> bool:
        return self.passed


@runtime_checkable
class AGuardrail(Protocol):
    """Something an answer has to satisfy."""

    name: str
    #: What this rail's own failure means. A rail that does not say gets
    #: REFUSE, because the safe default has to be the one you get by not
    #: thinking about it.
    when_it_cannot_answer: WhenItCannotAnswer

    def check(self, answer: Any) -> AVerdict:
        ...


@dataclass
class _AGuardrail:
    name: str
    look: Callable[[Any], AVerdict]
    when_it_cannot_answer: WhenItCannotAnswer = WhenItCannotAnswer.REFUSE

    def check(self, answer: Any) -> AVerdict:
        return self.look(answer)


def a_guardrail(
    name: str,
    look: Callable[[Any], AVerdict],
    *,
    when_it_cannot_answer: WhenItCannotAnswer = WhenItCannotAnswer.REFUSE,
) -> AGuardrail:
    """Make a guardrail from a function that returns a verdict.

    ``when_it_cannot_answer`` defaults to REFUSE. A rail whose job is quality
    should say CARRY_ON explicitly; the point of the default is that nobody
    gets fail-open by forgetting.
    """
    if not str(name).strip():
        raise ValueError("a guardrail needs a name; a refusal from an unnamed one "
                         "cannot be acted on")
    return _AGuardrail(
        name=str(name), look=look, when_it_cannot_answer=when_it_cannot_answer
    )


@dataclass
class TheGuardrails:
    """Every check an answer must pass, in the order they are applied."""

    rails: list[AGuardrail] = field(default_factory=list)
    #: Every refusal, with which rail and why.
    refusals: list[dict[str, Any]] = field(default_factory=list)
    #: Rails that could not run, and what their declaration made of it.
    could_not_run: list[dict[str, Any]] = field(default_factory=list)
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
            except Exception as exc:  # noqa: BLE001 — the rail says what this means
                broken = self._a_rail_that_could_not_run(rail, exc)
                if broken is not None:
                    return broken
                continue
            if not said:
                self.refusals.append(
                    {"rail": rail.name, "why": said.why, "instead": said.instead}
                )
                return said
        abstained = [
            one["rail"]
            for one in self.could_not_run
            if one["declared"] in (str(WhenItCannotAnswer.ABSTAIN),
                                   str(WhenItCannotAnswer.CARRY_ON))
        ]
        if abstained:
            # A pass that some rails never looked at. Said out loud, because
            # a caller reading only `passed` would otherwise read silence as
            # agreement from every rail there is.
            return AVerdict(
                passed=True,
                why="passed the rails that ran; "
                + ", ".join(sorted(set(abstained)))
                + " could not run",
            )
        return AVerdict(passed=True)

    def _a_rail_that_could_not_run(
        self, rail: AGuardrail, exc: BaseException
    ) -> AVerdict | None:
        """What this rail's own failure means. None to carry on to the next.

        A check that could not run has established nothing. Whether that stops
        the answer is the rail's declaration, not this loop's opinion.
        """
        says = getattr(rail, "when_it_cannot_answer", WhenItCannotAnswer.REFUSE)
        note = {
            "rail": rail.name,
            "raised": f"{type(exc).__name__}: {exc}",
            "declared": str(says),
        }
        self.could_not_run.append(note)
        if says is WhenItCannotAnswer.CARRY_ON:
            logger.warning(
                "guardrail %s raised and declares carry-on: %s", rail.name, exc
            )
            return None
        if says is WhenItCannotAnswer.ABSTAIN:
            logger.warning(
                "guardrail %s raised and abstains; the report says it did not run: %s",
                rail.name,
                exc,
            )
            return None
        if says is WhenItCannotAnswer.ESCALATE:
            try:
                from core.runtime.errors import record_degradation

                record_degradation(
                    "guardrails",
                    exc,
                    action=f"refused an answer because {rail.name} could not run",
                )
            except Exception:  # noqa: BLE001 - reporting must not be the failure
                logger.error("guardrail %s could not run: %s", rail.name, exc)
        why = (
            f"{rail.name} could not run ({type(exc).__name__}), and a check that "
            "could not run has established nothing"
        )
        self.refusals.append({"rail": rail.name, "why": why, "instead": ""})
        return AVerdict(passed=False, why=why, from_a_broken_rail=True)

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
            "when_they_cannot_answer": {
                one.name: str(
                    getattr(one, "when_it_cannot_answer", WhenItCannotAnswer.REFUSE)
                )
                for one in self.rails
            },
            "fail_open_rails": sorted(
                one.name
                for one in self.rails
                if getattr(one, "when_it_cannot_answer", WhenItCannotAnswer.REFUSE)
                is WhenItCannotAnswer.CARRY_ON
            ),
            "attempts": self.attempts,
            "refusals": list(self.refusals),
            "could_not_run": list(self.could_not_run),
        }
