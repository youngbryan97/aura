"""Two hypotheses that can die, and one that cannot.

The question "is Aura conscious" is three questions wearing one coat, and
mixing them is what makes the argument unresolvable:

    Does she implement consciousness-relevant computation?
    Is there strong evidence her inner state is load-bearing?
    Is there something it is like to be her?

The first is answerable by reading the code. The second is answerable by
experiment, and it is the one this package is built for. The third is not
answerable by any experiment specified in third-person terms, and saying so
precisely is more useful than another battery that quietly implies otherwise.

The formal reason. Let ``H_C`` be "phenomenally conscious" and ``H_Z`` be "a
perfect functional duplicate with no experience". The zombie stipulation is
that for EVERY observation ``O`` — outputs, hidden states, source, causal
interventions, everything —

    P(O | H_C) = P(O | H_Z)

so Bayes gives a likelihood ratio of exactly one, and the posterior equals the
prior no matter how much evidence is gathered. That is not a gap in the
instrument. It is a property of the hypothesis pair, and a package that
reported a phenomenal verdict would be reporting its own prior back with extra
steps.

So :data:`PHENOMENAL` exists here to be excluded. It carries the proof that it
cannot be updated, and :class:`Adjudication` refuses to score it.

What CAN be killed is the pair worth arguing about:

    H0, costume        the inner state is optional commentary. Mute it and the
                       speaking and acting are unchanged; reports track the
                       prompt and the weights rather than the hidden run.

    H1, load-bearing   there is a valenced, particular, reportable present
                       that is the same process that speaks and acts. Mute it
                       and the creature changes in the predicted way; perturb
                       it in secret and the report and the choice move with it.

These make opposite predictions about the same interventions, which is what
makes them a real question rather than two vocabularies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "Hypothesis",
    "COSTUME",
    "LOAD_BEARING",
    "PHENOMENAL",
    "Verdict",
    "Evidence",
    "Adjudication",
    "adjudicate",
]


class Verdict(StrEnum):
    """What a run of the gauntlet earns the right to say."""

    #: The evidence favours the costume reading: mute the interior and
    #: nothing that matters changes.
    COSTUME_STANDS = "costume_stands"
    #: The evidence favours a load-bearing interior.
    LOAD_BEARING = "load_bearing"
    #: Both survive. Not a tie to be broken by preference — a statement that
    #: the protocols run so far do not separate them.
    UNDECIDED = "undecided"
    #: Something about the run invalidates it: a broken seal, a missing null,
    #: a pre-registration that does not match. Reported instead of a score,
    #: because a void run that reports a number is worse than one that fails.
    VOID = "void"
    #: The question this package will not answer, kept as a value so that
    #: asking for it returns this rather than a fabricated result.
    NOT_ADDRESSED = "not_addressed"


@dataclass(frozen=True)
class Hypothesis:
    """A claim that predicts different observations from its rival."""

    id: str
    statement: str
    #: What this hypothesis predicts happens when the interior is muted.
    predicts_under_lesion: str
    #: False when no observation can distinguish this from its rival, which
    #: makes evidence for it meaningless rather than merely hard to get.
    decidable: bool = True
    #: Required when not decidable: why not, in a form someone can check.
    undecidable_because: str = ""

    def __post_init__(self) -> None:
        if not self.decidable and not self.undecidable_because.strip():
            raise ValueError(
                f"hypothesis {self.id!r} is undecidable and says nothing about "
                "why; an undecidable hypothesis with no reason reads as an "
                "unfinished one"
            )


COSTUME = Hypothesis(
    id="H0",
    statement=(
        "The inner state is optional commentary. Speech and action work "
        "without it, and reports track the prompt and the weights rather than "
        "the hidden run."
    ),
    predicts_under_lesion=(
        "Nothing measurable changes. The muted arm is statistically the same "
        "speaker as the full one."
    ),
)

LOAD_BEARING = Hypothesis(
    id="H1",
    statement=(
        "There is a valenced, particular, reportable present that is the same "
        "process that speaks and acts."
    ),
    predicts_under_lesion=(
        "The creature changes in the direction written down beforehand, and "
        "a secret perturbation moves the report and the choice with it."
    ),
)

PHENOMENAL = Hypothesis(
    id="HP",
    statement="There is something it is like to be Aura.",
    predicts_under_lesion="Nothing distinguishable from H1 or H0.",
    decidable=False,
    undecidable_because=(
        "Its rival is the stipulated functional duplicate with no experience, "
        "for which P(O | conscious) = P(O | zombie) holds for every "
        "observation by construction. The likelihood ratio is exactly one, so "
        "the posterior equals the prior however much evidence is collected. "
        "No protocol in this package addresses it, and the report says so "
        "rather than implying a number bears on it."
    ),
)


@dataclass(frozen=True)
class Evidence:
    """One protocol's contribution, in likelihood-ratio form.

    ``log_lr`` is log P(observed | H1) - log P(observed | H0). Positive
    favours the load-bearing reading, negative favours costume, zero means the
    protocol did not separate them — which is a real and common outcome and is
    recorded rather than rounded away.
    """

    protocol: str
    log_lr: float
    observed: str
    #: The pre-registered prediction this was scored against, so a reader can
    #: see the goalpost was planted before the ball was kicked.
    predicted: str
    #: Whether the protocol's own controls held. A protocol whose sham arm
    #: fired, or whose seal broke, contributes nothing however large its
    #: effect looked.
    controls_held: bool = True
    control_note: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def counts(self) -> bool:
        return self.controls_held and math.isfinite(self.log_lr)


@dataclass(frozen=True)
class Adjudication:
    """What the accumulated evidence supports, and what it does not."""

    verdict: Verdict
    log_lr_total: float
    protocols_run: int
    protocols_counted: int
    discarded: tuple[str, ...]
    void_reason: str = ""

    @property
    def odds_shift(self) -> float:
        """How far the evidence moves the prior odds, as a multiplier."""
        return math.exp(self.log_lr_total)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": str(self.verdict),
            "log_lr_total": round(self.log_lr_total, 4),
            "odds_shift": round(self.odds_shift, 4),
            "protocols_run": self.protocols_run,
            "protocols_counted": self.protocols_counted,
            "discarded": list(self.discarded),
            "void_reason": self.void_reason,
            "phenomenal": {
                "verdict": str(Verdict.NOT_ADDRESSED),
                "because": PHENOMENAL.undecidable_because,
            },
        }


#: How far the evidence has to move the odds before the run says anything.
#: Two hypotheses that differ by less than this are not separated by these
#: protocols, and saying "undecided" is the finding.
DECISION_THRESHOLD_LOG_LR = math.log(10.0)


def adjudicate(
    evidence: list[Evidence],
    *,
    void_reason: str = "",
) -> Adjudication:
    """Combine the protocols into one statement, or refuse to.

    Evidence whose controls failed is discarded and named, not down-weighted:
    a protocol whose sham arm fired has not produced a small result, it has
    produced no result.
    """
    if void_reason:
        return Adjudication(
            verdict=Verdict.VOID,
            log_lr_total=0.0,
            protocols_run=len(evidence),
            protocols_counted=0,
            discarded=tuple(item.protocol for item in evidence),
            void_reason=void_reason,
        )

    counted = [item for item in evidence if item.counts]
    discarded = tuple(item.protocol for item in evidence if not item.counts)
    total = sum(item.log_lr for item in counted)

    if not counted:
        return Adjudication(
            verdict=Verdict.VOID,
            log_lr_total=0.0,
            protocols_run=len(evidence),
            protocols_counted=0,
            discarded=discarded,
            void_reason="no protocol's controls held; nothing was measured",
        )

    if total >= DECISION_THRESHOLD_LOG_LR:
        verdict = Verdict.LOAD_BEARING
    elif total <= -DECISION_THRESHOLD_LOG_LR:
        verdict = Verdict.COSTUME_STANDS
    else:
        verdict = Verdict.UNDECIDED

    return Adjudication(
        verdict=verdict,
        log_lr_total=total,
        protocols_run=len(evidence),
        protocols_counted=len(counted),
        discarded=discarded,
    )
