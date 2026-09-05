"""core/cognition/native_concepts.py — concepts that are not words.

The end state worth wanting is internal representations optimised for
computation rather than for communication, with language as one readout
channel among others: learned concepts that cannot be translated word for word
and still take part in planning and reasoning.

Both halves of that are needed and the second is the one that gets skipped. A
latent that resists translation is easy to produce — most of them do, and most
of them are noise. What makes one a concept rather than a leftover is that
using it beats using the best word available for it on something she actually
has to do. Untranslatable and useful is a native concept. Untranslatable and
useless is a residual, and calling it ineffable is how a system acquires a
private vocabulary that means nothing.

So a concept is judged on two measurements, both against the same lexicon:

**Translatability.** How much of what the concept distinguishes its nearest
word also distinguishes. High means the word will do and there is nothing here
that language was losing.

**Participation.** How much better a downstream discrimination goes with the
concept than with that word. This is the one that separates a concept from a
residual, and it is measured on a task rather than asserted.

The verdict is over the pair, and one of its outcomes is that the interesting
thing turned out to be nothing. That outcome has to be reachable or the
instrument is a way of agreeing with whoever ran it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Cognition.Native")

#: Coverage above which the nearest word is taken to say what the concept
#: says. A word that captures four fifths of what a concept distinguishes is
#: the word for it.
TRANSLATABLE_ABOVE = 0.8

#: How much better the concept has to do than its nearest word before it is
#: taken to be contributing rather than tying.
PARTICIPATION_MARGIN = 0.05

#: Instances needed before either measurement means anything.
MIN_INSTANCES = 8


class Kind(StrEnum):
    """What a candidate concept turned out to be."""

    #: Resists translation and earns its keep. A concept language was losing.
    NATIVE = "native"
    #: A word says it. Nothing was being lost.
    VERBAL = "verbal"
    #: Resists translation and does nothing. A residual, not a concept.
    RESIDUAL = "residual"
    #: Too little evidence to tell these apart.
    UNMEASURED = "unmeasured"

    @property
    def worth_keeping(self) -> bool:
        return self in {Kind.NATIVE, Kind.VERBAL}


@dataclass(frozen=True)
class Instance:
    """One thing the concept was formed from, and what it turned out to be."""

    features: tuple[float, ...]
    #: The word a person would use, if any.
    label: str = ""
    #: The downstream answer, for measuring participation.
    outcome: bool | None = None


@dataclass(frozen=True)
class Assessment:
    """What a candidate concept is, and the two numbers that decide it."""

    name: str
    kind: Kind
    translatability: float
    participation: float
    nearest_word: str
    instances: int
    because: str

    @property
    def is_native(self) -> bool:
        return self.kind is Kind.NATIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": str(self.kind),
            "translatability": round(self.translatability, 4),
            "participation": round(self.participation, 4),
            "nearest_word": self.nearest_word,
            "instances": self.instances,
            "is_native": self.is_native,
            "because": self.because,
        }


def _agreement(left: Sequence[bool], right: Sequence[bool]) -> float:
    """Fraction of positions where two partitions agree."""
    if not left:
        return 0.0
    return sum(1 for a, b in zip(left, right, strict=True) if a == b) / len(left)


def _concept_partition(
    instances: Sequence[Instance], direction: Sequence[float]
) -> list[bool]:
    """Which side of the concept each instance falls on.

    The split is at the median projection rather than at zero, so a concept
    that is real but off-centre is not scored as though it separated nothing.
    """
    projections = [
        sum(f * d for f, d in zip(i.features, direction, strict=False)) for i in instances
    ]
    ordered = sorted(projections)
    middle = len(ordered) // 2
    threshold = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return [p > threshold for p in projections]


def _word_partition(instances: Sequence[Instance], word: str) -> list[bool]:
    return [i.label == word for i in instances]


def assess(
    name: str,
    direction: Sequence[float],
    instances: Sequence[Instance],
    *,
    lexicon: Sequence[str] | None = None,
) -> Assessment:
    """Whether this is a concept language was losing, a word, or a residual."""
    usable = [i for i in instances if i.features]
    if len(usable) < MIN_INSTANCES:
        return Assessment(
            name=name,
            kind=Kind.UNMEASURED,
            translatability=0.0,
            participation=0.0,
            nearest_word="",
            instances=len(usable),
            because=(
                f"{len(usable)} instances; {MIN_INSTANCES} are needed before "
                "either measurement means anything"
            ),
        )

    words = list(lexicon) if lexicon is not None else sorted(
        {i.label for i in usable if i.label}
    )
    concept = _concept_partition(usable, direction)

    best_word, best_coverage = "", 0.0
    for word in words:
        coverage = _agreement(concept, _word_partition(usable, word))
        # A word that anti-correlates says the same thing inverted, which is
        # still the word for it.
        coverage = max(coverage, 1.0 - coverage)
        if coverage > best_coverage:
            best_word, best_coverage = word, coverage

    participation = _participation(usable, concept, best_word)

    if best_coverage >= TRANSLATABLE_ABOVE:
        return Assessment(
            name=name,
            kind=Kind.VERBAL,
            translatability=best_coverage,
            participation=participation,
            nearest_word=best_word,
            instances=len(usable),
            because=(
                f"{best_word!r} covers {best_coverage:.0%} of what this "
                "distinguishes; the word will do"
            ),
        )
    if participation <= PARTICIPATION_MARGIN:
        return Assessment(
            name=name,
            kind=Kind.RESIDUAL,
            translatability=best_coverage,
            participation=participation,
            nearest_word=best_word,
            instances=len(usable),
            because=(
                f"no word covers it — the best is {best_word or 'none'} at "
                f"{best_coverage:.0%} — and using it beats the word by "
                f"{participation:+.0%}, which is nothing. Untranslatable and "
                "useless is a residual, not a concept"
            ),
        )
    return Assessment(
        name=name,
        kind=Kind.NATIVE,
        translatability=best_coverage,
        participation=participation,
        nearest_word=best_word,
        instances=len(usable),
        because=(
            f"no word covers it — the best is {best_word or 'none'} at "
            f"{best_coverage:.0%} — and using it beats that word by "
            f"{participation:+.0%} on the task"
        ),
    )


def _participation(
    instances: Sequence[Instance], concept: Sequence[bool], word: str
) -> float:
    """How much better the concept predicts the outcome than the word does.

    On the task, not asserted. A concept that resists translation and does not
    help is a residual, and this is the measurement that says so.
    """
    graded = [
        (c, i) for c, i in zip(concept, instances, strict=True) if i.outcome is not None
    ]
    if len(graded) < MIN_INSTANCES:
        return 0.0
    outcomes = [i.outcome for _c, i in graded]
    by_concept = [c for c, _i in graded]
    by_word = [i.label == word for _c, i in graded]
    concept_accuracy = max(
        _agreement(by_concept, outcomes), 1.0 - _agreement(by_concept, outcomes)
    )
    word_accuracy = max(
        _agreement(by_word, outcomes), 1.0 - _agreement(by_word, outcomes)
    )
    return concept_accuracy - word_accuracy


def assess_all(
    candidates: Mapping[str, Sequence[float]],
    instances: Sequence[Instance],
    *,
    lexicon: Sequence[str] | None = None,
) -> tuple[Assessment, ...]:
    """Assess several candidate directions against the same instances."""
    out = [
        assess(name, direction, instances, lexicon=lexicon)
        for name, direction in candidates.items()
    ]
    out.sort(key=lambda a: (-a.participation, a.translatability, a.name))
    return tuple(out)


def medium_report(assessments: Sequence[Assessment]) -> dict[str, Any]:
    """How much of what she thinks in is not sayable.

    The number the item is about. Zero natives means her internal
    representation is language with extra steps, which is a real finding and
    the one to expect at first.
    """
    kept = [a for a in assessments if a.kind.worth_keeping]
    natives = [a for a in assessments if a.is_native]
    return {
        "candidates": len(assessments),
        "native": len(natives),
        "verbal": sum(1 for a in assessments if a.kind is Kind.VERBAL),
        "residual": sum(1 for a in assessments if a.kind is Kind.RESIDUAL),
        "unmeasured": sum(1 for a in assessments if a.kind is Kind.UNMEASURED),
        "native_fraction": (len(natives) / len(kept)) if kept else 0.0,
        "names": [a.name for a in natives],
    }


__all__ = [
    "MIN_INSTANCES",
    "PARTICIPATION_MARGIN",
    "TRANSLATABLE_ABOVE",
    "Assessment",
    "Instance",
    "Kind",
    "assess",
    "assess_all",
    "medium_report",
]
