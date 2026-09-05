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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
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
    "Criterion",
    "CriterionRegistry",
    "IndependenceError",
    "Judgement",
    "SealBrokenError",
    "declare",
    "registry",
]
