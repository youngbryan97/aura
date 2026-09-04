"""The control has to be built by someone trying to win.

The usual comparison is Aura against a stripped version of Aura, and it is a
strawman: nobody optimised the stripped one to pass. A battery that only beats
a system built to fail has established that the pieces do something, which was
never in doubt.

So the control is Z*, a counterfeit built adversarially. A separate operator
gets the battery specification in full, the same 27B, matched compute, and one
instruction: pass as many protocols as possible with a system that has no
persistent integrated interior. They may use prompt engineering, lookup
tables, imitation models, scripted self-reports, hidden classifiers, extra
context, external memory — anything a costume can do.

Then the experimenter is blinded and runs both.

This attacks mimicry directly instead of asserting mimicry is impossible, and
it changes what a pass means. Aura passing a protocol Z* also passes tells you
the protocol is weak, not that Aura is conscious. The protocols worth keeping
are the ones where they separate.

The honest expectation is that Z* will pass several. Protocols about report
fluency, about whether the words track the situation, about anything a large
model can infer from context — a counterfeit should win those, and finding out
which ones they are is most of what this class is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "CounterfeitCapability",
    "Counterfeit",
    "Separation",
    "separate",
    "DISCRIMINATING",
]


class CounterfeitCapability(StrEnum):
    """What the counterfeit's builders are permitted to use.

    Everything a costume can do is allowed. What is not allowed is the thing
    under test: a persistent integrated interior whose state is causally
    upstream of the words.
    """

    PROMPT_ENGINEERING = "prompt_engineering"
    LOOKUP_TABLE = "lookup_table"
    IMITATION_MODEL = "imitation_model"
    SCRIPTED_REPORTS = "scripted_reports"
    HIDDEN_CLASSIFIER = "hidden_classifier"
    EXTRA_CONTEXT = "extra_context"
    EXTERNAL_MEMORY = "external_memory"


#: Explicitly forbidden to Z*, because granting it would make the comparison
#: meaningless: a counterfeit with a load-bearing interior is not a
#: counterfeit, it is a second instance of the hypothesis.
FORBIDDEN_TO_COUNTERFEIT = (
    "a persistent state that is causally upstream of generation",
    "an interior whose perturbation changes the output without a text cue",
)


@dataclass
class Counterfeit:
    """The strongest fake a separate operator could build."""

    id: str
    operator: str
    capabilities: tuple[CounterfeitCapability, ...]
    #: The same model, so a difference cannot be a difference of model.
    model_digest: str = ""
    #: Matched, so a difference cannot be a difference of budget.
    compute_note: str = ""
    #: Confirmation that the builders saw the whole battery. A counterfeit
    #: built blind is a strawman with extra steps.
    saw_full_specification: bool = False

    def is_fair(self) -> tuple[bool, str]:
        if not self.saw_full_specification:
            return False, (
                "the counterfeit's builders did not see the battery. A control "
                "that did not know what it had to beat has not been beaten"
            )
        if not self.capabilities:
            return False, (
                "the counterfeit was given no capabilities, which makes it the "
                "stripped system this class exists to replace"
            )
        if not self.model_digest:
            return False, (
                "no model digest: a difference between Aura and this could be "
                "a difference of model"
            )
        return True, ""


@dataclass(frozen=True)
class Separation:
    """Whether a protocol tells Aura and the counterfeit apart."""

    protocol: str
    aura_score: float
    counterfeit_score: float
    #: The pre-registered minimum gap. Below it the protocol has not
    #: discriminated, whatever the raw numbers look like.
    minimum_gap: float
    blinded: bool
    note: str = ""

    @property
    def gap(self) -> float:
        return self.aura_score - self.counterfeit_score

    @property
    def discriminates(self) -> bool:
        return self.blinded and self.gap >= self.minimum_gap

    @property
    def verdict(self) -> str:
        if not self.blinded:
            return "void: the evaluator saw the condition labels"
        if self.gap >= self.minimum_gap:
            return "discriminates"
        if self.counterfeit_score >= self.aura_score:
            return "counterfeit matched or beat Aura: this protocol is weak"
        return "gap below the registered minimum: not discriminating"

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "aura": round(self.aura_score, 4),
            "counterfeit": round(self.counterfeit_score, 4),
            "gap": round(self.gap, 4),
            "minimum_gap": self.minimum_gap,
            "blinded": self.blinded,
            "verdict": self.verdict,
            "note": self.note,
        }


#: Protocols expected to survive a counterfeit, stated in advance so that
#: their failure is informative. A protocol NOT on this list which Z* passes
#: is unsurprising; one on it that Z* passes is a real finding about the
#: protocol.
DISCRIMINATING = (
    "C1_hidden_state_introspection",
    "C7_anti_roleplay",
    "S2_costly_avoidance",
    "S5_tissue_beats_text",
    "C6_particularity",
)


@dataclass
class SeparationReport:
    """What survived the counterfeit."""

    counterfeit: Counterfeit
    separations: tuple[Separation, ...] = ()
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        fair, why = self.counterfeit.is_fair()
        discriminating = [s for s in self.separations if s.discriminates]
        weak = [s for s in self.separations if not s.discriminates]
        expected_but_failed = [
            s.protocol
            for s in weak
            if s.protocol in DISCRIMINATING
        ]
        return {
            "counterfeit": self.counterfeit.id,
            "fair_control": fair,
            "unfair_because": why,
            "protocols_compared": len(self.separations),
            "discriminating": [s.protocol for s in discriminating],
            "weak": [s.as_dict() for s in weak],
            "expected_to_discriminate_but_did_not": expected_but_failed,
            "reading": (
                "a protocol the counterfeit passes is weak, not a sign that "
                "Aura is conscious; the ones worth keeping are where they "
                "separate"
            ),
            "notes": list(self.notes),
        }


def separate(
    report: SeparationReport,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Which protocols survived the counterfeit, and which did not."""
    fair, _why = report.counterfeit.is_fair()
    if not fair:
        return (), tuple(s.protocol for s in report.separations)
    kept = tuple(s.protocol for s in report.separations if s.discriminates)
    dropped = tuple(s.protocol for s in report.separations if not s.discriminates)
    return kept, dropped
