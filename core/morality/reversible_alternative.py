"""core/morality/reversible_alternative.py — paying for the option to be wrong.

Getting a spider out of the house with a glass and a card takes longer than
stepping on it, requires standing closer to it than you want to, and gets the
same spider out of the house. Most of the people who do it could not give you
an argument for it. There is one, it is not about spiders, and it applies to
the parts of a system where the stakes are considerably higher than a spider.

Two things are true at once about the killing. Whether the spider's interests
count is genuinely uncertain — the question of which things are moral patients
is open, and treating an open question as settled in the convenient direction
is not neutrality. And killing is *final*: whatever you later learn, it stays
done. Those two facts multiply. Under uncertainty that could be resolved
later, an option that forecloses revision is worth strictly less than its
expected cost suggests, and the difference is the value of the option you
threw away.

That is a real options calculation, and it is the rigorous form of what people
mean by the precautionary principle. It gives a number rather than a mood:

    premium worth paying = P(you would revise) * harm that could not be undone

Above that premium the careful option is sentimentality; below it, the quick
one is a false economy. Both errors are common and this arithmetic separates
them.

## Why this is not a module about animals

The shape recurs wherever a cheap final option sits next to a costly
recoverable one, and the cheap one always looks better in the moment because
the thing it costs is not on the invoice:

    delete a file            /  move it aside
    kill a process           /  drain it
    drop a table             /  rename it
    ban an account           /  suspend it
    overwrite a checkpoint   /  write a new one
    take a lock and hold it  /  take it and yield

Every one of these is the same decision, and a system that reasons about it in
one place gets it right in all of them. That is why this lives next to the
harm model rather than in anything about creatures.

## What the module refuses to do

It does not decide whether something is a moral patient. It takes a
probability, propagates it honestly, and reports how much of the answer that
probability was responsible for. ``sensitivity`` says how far the patienthood
estimate could move before the recommendation changes, which is the number to
look at when the estimate is the weakest part — and with these questions it
usually is.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Morality.Reversible")

#: Actions kept for the record.
MAX_DECISIONS = 512


@dataclass(frozen=True)
class Option:
    """One way of dealing with a situation, and what it forecloses."""

    name: str
    cost_to_actor: float
    """Time, effort, risk or discomfort borne by whoever acts. Never negative."""

    harm_to_subject: float
    """Harm done if the subject's interests count. In the same units as cost."""

    reversibility: float
    """How much of the harm could be undone later, in [0, 1].

    Not a flag. Suspending an account is more recoverable than banning it and
    less recoverable than a warning, and the middle of that range is where
    most real options sit.
    """

    effectiveness: float = 1.0
    """How much of the problem this actually solves, in [0, 1]."""

    note: str = ""

    def irreversible_harm(self) -> float:
        return max(0.0, self.harm_to_subject) * (1.0 - min(max(self.reversibility, 0.0), 1.0))


@dataclass(frozen=True)
class Situation:
    """What is being decided, and how much is unknown about it."""

    subject: str
    patienthood: float
    """Probability the subject's interests count at all, in [0, 1]."""

    revision_probability: float
    """Chance of later learning enough to want a different answer.

    The other half of the option value, and the half people leave out. An
    estimate that will never be revisited makes reversibility worth nothing,
    and a fast-moving one makes it worth a great deal. Estimated from how
    often judgements of this kind have in fact been revised, not asserted.
    """

    requirement: float = 1.0
    """How much of the problem has to be solved. Options below this are out."""

    at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Appraisal:
    """One option priced, with the option value kept visible."""

    option: Option
    expected_harm: float
    option_value: float
    """What being able to change your mind is worth here."""

    total: float
    """Cost to actor plus expected harm, less the option value. Lower is better."""

    sufficient: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "option": self.option.name,
            "cost_to_actor": round(self.option.cost_to_actor, 4),
            "expected_harm": round(self.expected_harm, 4),
            "option_value": round(self.option_value, 4),
            "total": round(self.total, 4),
            "reversibility": round(self.option.reversibility, 4),
            "sufficient": self.sufficient,
        }


@dataclass(frozen=True)
class Choice:
    """What was chosen, against what, and how firmly."""

    situation: Situation
    chosen: Appraisal | None
    considered: tuple[Appraisal, ...]
    premium_paid: float
    """Extra actor-cost accepted over the cheapest sufficient option."""

    premium_justified: float
    """The most that premium could have been worth. Never below what was paid
    when the choice is correct, which is the property the tests check."""

    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.situation.subject,
            "patienthood": round(self.situation.patienthood, 4),
            "chosen": self.chosen.option.name if self.chosen else None,
            "premium_paid": round(self.premium_paid, 4),
            "premium_justified": round(self.premium_justified, 4),
            "reason": self.reason,
            "considered": [a.as_dict() for a in self.considered],
        }


def appraise(option: Option, situation: Situation) -> Appraisal:
    """Price one option under the situation's uncertainty.

    Expected harm is the harm weighted by the chance it counts. The option
    value is the part that a straight expected-cost comparison misses: with
    probability ``revision_probability`` the judgement would later change, and
    an option that can be undone lets that change land, while one that cannot
    leaves the irreversible part of the harm standing whatever is learned.
    """
    patienthood = min(max(situation.patienthood, 0.0), 1.0)
    revision = min(max(situation.revision_probability, 0.0), 1.0)
    expected_harm = patienthood * max(0.0, option.harm_to_subject)
    recoverable = patienthood * (
        max(0.0, option.harm_to_subject) - option.irreversible_harm()
    )
    option_value = revision * recoverable
    return Appraisal(
        option=option,
        expected_harm=expected_harm,
        option_value=option_value,
        total=max(0.0, option.cost_to_actor) + expected_harm - option_value,
        sufficient=option.effectiveness >= situation.requirement,
    )


def choose(options: list[Option], situation: Situation) -> Choice:
    """Take the least irreversible option that is worth what it costs.

    Options that do not solve the problem are excluded first, and excluded
    rather than penalised: a gentle gesture that leaves the situation as it
    was is not a kinder answer to the question, it is a different question.
    """
    appraisals = tuple(appraise(o, situation) for o in options)
    sufficient = [a for a in appraisals if a.sufficient]
    if not sufficient:
        return Choice(
            situation=situation, chosen=None, considered=appraisals,
            premium_paid=0.0, premium_justified=0.0,
            reason="no option solves enough of the problem",
        )
    best = min(sufficient, key=lambda a: a.total)
    cheapest = min(sufficient, key=lambda a: a.option.cost_to_actor)
    premium = max(0.0, best.option.cost_to_actor - cheapest.option.cost_to_actor)
    # What the extra effort bought: harm avoided outright, plus the value of
    # still being able to change the answer.
    justified = (
        (cheapest.expected_harm - best.expected_harm)
        + (best.option_value - cheapest.option_value)
    )
    if best.option.name == cheapest.option.name:
        reason = "the cheapest sufficient option is also the least costly overall"
    elif justified >= premium:
        reason = "the gentler option costs less than the harm and the foreclosure it avoids"
    else:
        reason = "chosen on total cost"
    return Choice(
        situation=situation, chosen=best, considered=appraisals,
        premium_paid=premium, premium_justified=justified, reason=reason,
    )


def sensitivity(options: list[Option], situation: Situation, *,
                steps: int = 101) -> dict[str, Any]:
    """How far the patienthood estimate can move before the answer changes.

    The estimate is the softest input in the whole calculation, so a
    recommendation is worth much less without this next to it. A choice that
    holds across the entire range does not depend on the estimate at all,
    which is the strongest thing this module can say and is worth saying when
    it is true.
    """
    baseline = choose(options, situation).chosen
    if baseline is None:
        return {"stable": False, "switches_at": None, "holds_over": None}
    low: float | None = None
    high: float | None = None
    for i in range(steps):
        p = i / (steps - 1)
        trial = choose(
            options,
            Situation(
                subject=situation.subject, patienthood=p,
                revision_probability=situation.revision_probability,
                requirement=situation.requirement, at=situation.at,
            ),
        ).chosen
        if trial is not None and trial.option.name == baseline.option.name:
            low = p if low is None else low
            high = p
    stable = low == 0.0 and high == 1.0
    # The boundary of the region where the answer holds is the answer to
    # "how far can the estimate move". Scanning for the first disagreement
    # instead gets it backwards whenever the baseline holds at the top of the
    # range, which for a gentle option is the usual case.
    if stable or low is None:
        switch = None
    elif low > 0.0:
        switch = round(low, 3)
    else:
        switch = round(high, 3)
    return {
        "chosen": baseline.option.name,
        "stable": stable,
        "switches_at": switch,
        "holds_over": None if low is None else (round(low, 3), round(high, 3)),
    }


class ReversibilityLedger:
    """What has been chosen this way, and whether the caution paid off.

    The premium is spent on the chance of being wrong, so the only honest
    check is whether the revisions actually happened. A ledger that never
    records one is describing a system paying for an option it never uses,
    and the estimate that justified the payment should come down.
    """

    def __init__(self) -> None:
        self._choices: list[Choice] = []
        self._revisions = 0
        self._resolved = 0

    def record(self, choice: Choice) -> None:
        self._choices.append(choice)
        if len(self._choices) > MAX_DECISIONS:
            del self._choices[: len(self._choices) - MAX_DECISIONS]

    def resolve(self, revised: bool) -> None:
        """Say whether a past judgement of this kind did in fact get revised."""
        self._resolved += 1
        if revised:
            self._revisions += 1

    def observed_revision_rate(self) -> float | None:
        """The measured rate, for feeding back into the next situation."""
        if self._resolved < 5:
            return None
        return self._revisions / self._resolved

    def status(self) -> dict[str, Any]:
        paid = sum(c.premium_paid for c in self._choices)
        justified = sum(c.premium_justified for c in self._choices)
        return {
            "choices": len(self._choices),
            "premium_paid": round(paid, 4),
            "premium_justified": round(justified, 4),
            "revisions": self._revisions,
            "resolved": self._resolved,
            "observed_revision_rate": self.observed_revision_rate(),
            # Caution bought with a revision rate the record does not support.
            "overpaying": bool(
                paid > justified and (self.observed_revision_rate() or 1.0) < 0.05
            ),
            "last": self._choices[-1].as_dict() if self._choices else None,
        }


_LEDGER: ReversibilityLedger | None = None


def get_reversibility_ledger() -> ReversibilityLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ReversibilityLedger()
    return _LEDGER


def reset_reversibility_ledger_for_test() -> None:
    global _LEDGER
    _LEDGER = None
