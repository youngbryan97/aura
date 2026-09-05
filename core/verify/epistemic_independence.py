"""core/verify/epistemic_independence.py — the criterion comes first.

An adaptive mechanism that decides what counts as success after seeing how it
did has measured nothing. The failure is not dishonesty and it does not look
like a bug: the threshold is chosen from the data because the data is what is
available, the baseline is computed over the same run it is meant to judge,
the target is set to whatever was achieved and then reported as met. Every
step is locally reasonable and the result is a mechanism that cannot fail.

Five arms of one campaign in this repository were constants written to come
out right, and the honest versions gave smaller, better numbers. That is the
class this module exists to make impossible.

A :class:`Criterion` is declared before the run, sealed, and only then may
judge. Sealing takes a digest of the predicate's own source and of every
constant in it, so a criterion edited after the fact is a different criterion
and says so. Judging records the verdict against the seal, and a second
declaration for a name that has already judged raises rather than replacing
it.

What this does not do is decide whether a threshold is a good one. A criterion
sealed before the run can still be too easy. It just cannot be made too easy
afterwards, which is the part that was happening.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Verify.Independence")


class IndependenceError(RuntimeError):
    """A criterion was set, changed or read out of order."""


class SealBrokenError(IndependenceError):
    """The criterion is not the one that was sealed."""


@dataclass(frozen=True)
class Judgement:
    """One verdict, and the seal it was made under."""

    name: str
    passed: bool
    observed: float
    threshold: float
    seal: str
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "seal": self.seal,
            "at": self.at,
        }


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


class Criterion:
    """What would count as success, fixed before there is anything to judge."""

    def __init__(
        self,
        name: str,
        *,
        threshold: float,
        direction: str = "above",
        rationale: str,
        predicate: Callable[[float, float], bool] | None = None,
    ) -> None:
        if direction not in {"above", "below"}:
            raise IndependenceError("direction must be 'above' or 'below'")
        if not rationale or len(rationale) < 12:
            # A threshold with no stated reason is a number somebody liked.
            # Requiring the sentence is what makes a later reader able to
            # tell a considered bar from a convenient one.
            raise IndependenceError(
                f"criterion {name!r} needs a rationale saying why this threshold"
            )
        self.name = str(name)
        self.threshold = float(threshold)
        self.direction = direction
        self.rationale = str(rationale)
        self._predicate = predicate
        self._judgements: list[Judgement] = []
        self._lock = threading.RLock()
        self._seal = self._compute_seal()

    def _compute_seal(self) -> str:
        body = ""
        if self._predicate is not None:
            try:
                body = inspect.getsource(self._predicate)
            except (OSError, TypeError):
                body = repr(self._predicate)
        return _digest(
            {
                "name": self.name,
                "threshold": self.threshold,
                "direction": self.direction,
                "rationale": self.rationale,
                "predicate": body,
            }
        )

    @property
    def seal(self) -> str:
        return self._seal

    @property
    def has_judged(self) -> bool:
        return bool(self._judgements)

    def judge(self, observed: float) -> Judgement:
        """Apply the sealed criterion. Raises if it has changed since sealing."""
        current = self._compute_seal()
        if current != self._seal:
            raise SealBrokenError(
                f"criterion {self.name!r} changed after it was sealed "
                f"({self._seal} -> {current}). A criterion edited after the "
                "run is a different criterion; declare a new one and say so."
            )
        value = float(observed)
        if self._predicate is not None:
            passed = bool(self._predicate(value, self.threshold))
        elif self.direction == "above":
            passed = value > self.threshold
        else:
            passed = value < self.threshold
        record = Judgement(
            name=self.name,
            passed=passed,
            observed=value,
            threshold=self.threshold,
            seal=self._seal,
        )
        with self._lock:
            self._judgements.append(record)
        return record

    def judgements(self) -> tuple[Judgement, ...]:
        with self._lock:
            return tuple(self._judgements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "direction": self.direction,
            "rationale": self.rationale,
            "seal": self._seal,
            "judgements": len(self._judgements),
        }


class CriterionRegistry:
    """Every declared criterion, and a refusal to redefine one that has judged."""

    def __init__(self) -> None:
        self._criteria: dict[str, Criterion] = {}
        self._lock = threading.RLock()

    def declare(self, criterion: Criterion) -> Criterion:
        with self._lock:
            existing = self._criteria.get(criterion.name)
            if existing is not None:
                if existing.seal == criterion.seal:
                    return existing
                if existing.has_judged:
                    raise IndependenceError(
                        f"criterion {criterion.name!r} has already judged "
                        f"{len(existing.judgements())} time(s) and is now being "
                        "redefined. Deciding what counts as success after "
                        "seeing the result measures nothing."
                    )
                logger.info(
                    "Criterion %s redeclared before any judgement", criterion.name
                )
            self._criteria[criterion.name] = criterion
            return criterion

    def get(self, name: str) -> Criterion | None:
        with self._lock:
            return self._criteria.get(name)

    def all(self) -> tuple[Criterion, ...]:
        with self._lock:
            return tuple(self._criteria.values())

    def snapshot(self) -> dict[str, Any]:
        criteria = self.all()
        return {
            "declared": len(criteria),
            "judged": sum(1 for c in criteria if c.has_judged),
            "criteria": [c.to_dict() for c in criteria],
        }

    def clear(self) -> None:
        with self._lock:
            self._criteria.clear()


_REGISTRY = CriterionRegistry()


def registry() -> CriterionRegistry:
    return _REGISTRY


def declare(
    name: str,
    *,
    threshold: float,
    rationale: str,
    direction: str = "above",
    predicate: Callable[[float, float], bool] | None = None,
) -> Criterion:
    """Declare and seal a success criterion before the run that will meet it."""
    return _REGISTRY.declare(
        Criterion(
            name,
            threshold=threshold,
            direction=direction,
            rationale=rationale,
            predicate=predicate,
        )
    )


__all__ = [
    "INDEPENDENT_CHANNELS_REQUIRED",
    "Channel",
    "Criterion",
    "CriterionRegistry",
    "IndependenceError",
    "Evidence",
    "Judgement",
    "Support",
    "SealBrokenError",
    "declare",
    "registry",
    "support_for",
]


# ── independent evidence channels ────────────────────────────────────────
#
# Sealing the criterion fixes WHEN success is defined. It says nothing about
# WHO says it was met, and for an important change those are different
# questions. A mechanism that proposes a change, runs the check, and reports
# the result has supplied all three, and the seal is satisfied throughout.
#
# The principle is organism-wide: for an important change, the evidence has to
# come from somewhere the mechanism does not control. External reality, a test
# it has not seen, a model that is not it, or a person. Which of those, and
# how many, is what `Channels` records — and a change evidenced only by its
# own author is refused the same way a criterion edited after the run is.


class Channel(StrEnum):
    """Where a piece of evidence came from."""

    #: The mechanism proposing the change. Never sufficient alone.
    SELF = "self"
    #: A test the mechanism did not have access to when it made the change.
    HELD_OUT = "held_out"
    #: A different model or implementation, scoring the same thing.
    ALTERNATE_MODEL = "alternate_model"
    #: The world: a measurement of what actually happened.
    EXTERNAL_REALITY = "external_reality"
    #: A person.
    HUMAN = "human"

    @property
    def independent(self) -> bool:
        return self is not Channel.SELF


#: Independent channels an important change needs. Two rather than one,
#: because a single independent channel that is wrong is indistinguishable
#: from a single independent channel that is right.
INDEPENDENT_CHANNELS_REQUIRED = 2


@dataclass(frozen=True)
class Evidence:
    """One piece of evidence, and where it came from."""

    channel: Channel
    verdict: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": str(self.channel),
            "verdict": self.verdict,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Support:
    """Whether a change is evidenced by anything but its own author."""

    sufficient: bool
    independent: int
    agreeing: int
    disagreeing: int
    because: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "independent": self.independent,
            "agreeing": self.agreeing,
            "disagreeing": self.disagreeing,
            "because": self.because,
        }


def support_for(
    evidence: Sequence[Evidence], *, important: bool = True
) -> Support:
    """Whether the evidence comes from somewhere the mechanism does not control.

    An unimportant change may be evidenced by its author; the whole point of
    the distinction is that not everything needs this. An important one needs
    agreement from at least two channels that are not it, and a disagreement
    from any independent channel is enough to withhold support — because the
    interesting case is exactly the one where the mechanism's own check passes
    and something else does not.
    """
    independent = [e for e in evidence if e.channel.independent]
    distinct = {e.channel for e in independent}
    agreeing = [e for e in independent if e.verdict]
    disagreeing = [e for e in independent if not e.verdict]

    if not important:
        own = [e for e in evidence if not e.channel.independent]
        passed = all(e.verdict for e in evidence) and bool(evidence)
        return Support(
            sufficient=passed,
            independent=len(distinct),
            agreeing=len(agreeing),
            disagreeing=len(disagreeing),
            because=(
                f"not an important change; {len(own)} self-report(s) and "
                f"{len(independent)} independent"
            ),
        )
    if disagreeing:
        return Support(
            sufficient=False,
            independent=len(distinct),
            agreeing=len(agreeing),
            disagreeing=len(disagreeing),
            because=(
                f"{len(disagreeing)} independent channel(s) disagree "
                f"({', '.join(sorted(str(e.channel) for e in disagreeing))}); "
                "a mechanism's own check passing while something else does not "
                "is the case this exists for"
            ),
        )
    if len(distinct) < INDEPENDENT_CHANNELS_REQUIRED:
        return Support(
            sufficient=False,
            independent=len(distinct),
            agreeing=len(agreeing),
            disagreeing=0,
            because=(
                f"{len(distinct)} independent channel(s); "
                f"{INDEPENDENT_CHANNELS_REQUIRED} are needed, because one that "
                "is wrong looks exactly like one that is right"
            ),
        )
    return Support(
        sufficient=True,
        independent=len(distinct),
        agreeing=len(agreeing),
        disagreeing=0,
        because=(
            f"{len(distinct)} channels the mechanism does not control agree: "
            f"{', '.join(sorted(str(c) for c in distinct))}"
        ),
    )
