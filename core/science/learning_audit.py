"""core/science/learning_audit.py — a learning claim with a link missing is not one.

"She learned X" is a claim with five links, and a claim missing any one of them
is a story:

    observation -> update -> stored artifact -> later retrieval -> outcome delta

The chain is what separates learning from three things it looks like. Without
the *observation*, an artifact appeared and nobody knows from what. Without the
*stored artifact*, a within-session improvement is the context window. Without
the *later retrieval*, the artifact exists and has never been used. Without the
*outcome delta*, it was used and nothing got better.

This is the verifier card 214 asks for. It does not judge how good the learning
was; it refuses to call something learning when a link is absent, and names
which one. That refusal is the useful part, because every missing link has a
plausible-sounding story attached to it and the stories are what got past
review before.

The delta needs a comparison
----------------------------
``outcome_delta`` is a number and a number alone proves nothing, so a claim
must also name what it improved OVER: the same task before the artifact
existed, or a matched run without it. A delta with no comparator is recorded as
an incomplete link, not as a small effect.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "Link",
    "LearningClaim",
    "LearningAudit",
    "get_learning_audit",
    "reset_learning_audit_for_test",
]


class Link(StrEnum):
    OBSERVATION = "observation"
    UPDATE = "update"
    ARTIFACT = "artifact"
    RETRIEVAL = "retrieval"
    DELTA = "delta"


_WHAT_IT_MEANS_WHEN_MISSING = {
    Link.OBSERVATION: "an artifact appeared and nothing says what it came from",
    Link.UPDATE: "nothing records the moment the system changed",
    Link.ARTIFACT: "the improvement is in the context window, not in the system",
    Link.RETRIEVAL: "the artifact exists and has never been used",
    Link.DELTA: "the artifact was used and nothing got better",
}


@dataclass
class LearningClaim:
    """One claim that something was learned, and the chain behind it."""

    name: str
    #: Observation ids the learning came from.
    observations: tuple[str, ...] = ()
    #: The update event: a cognitive event seq, a transaction id, a commit.
    update_ref: str = ""
    #: Where the learned thing is stored, and under what id.
    artifact_ref: str = ""
    #: Occasions the artifact was retrieved and used after being stored.
    retrievals: tuple[str, ...] = ()
    #: Improvement, and what it improved over.
    outcome_delta: float | None = None
    comparator: str = ""
    at: float = field(default_factory=time.time)

    @property
    def missing(self) -> tuple[Link, ...]:
        gaps: list[Link] = []
        if not self.observations:
            gaps.append(Link.OBSERVATION)
        if not self.update_ref:
            gaps.append(Link.UPDATE)
        if not self.artifact_ref:
            gaps.append(Link.ARTIFACT)
        if not self.retrievals:
            gaps.append(Link.RETRIEVAL)
        if self.outcome_delta is None or not self.comparator:
            gaps.append(Link.DELTA)
        return tuple(gaps)

    @property
    def complete(self) -> bool:
        return not self.missing

    def explain(self) -> str:
        if self.complete:
            return (
                f"{self.name}: learned from {len(self.observations)} observation(s), "
                f"stored as {self.artifact_ref}, used {len(self.retrievals)} time(s), "
                f"delta {self.outcome_delta:+.4g} against {self.comparator}"
            )
        return f"{self.name}: not a learning claim — " + "; ".join(
            f"{link.value} missing ({_WHAT_IT_MEANS_WHEN_MISSING[link]})" for link in self.missing
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "observations": list(self.observations),
            "update_ref": self.update_ref,
            "artifact_ref": self.artifact_ref,
            "retrievals": list(self.retrievals),
            "outcome_delta": self.outcome_delta,
            "comparator": self.comparator,
            "complete": self.complete,
            "missing": [link.value for link in self.missing],
            "explanation": self.explain(),
        }


class LearningAudit:
    """Every claim that something was learned, and whether its chain closes."""

    def __init__(self, *, max_claims: int = 8192) -> None:
        self._lock = threading.RLock()
        self._claims: dict[str, LearningClaim] = {}
        self._max = int(max_claims)

    def open(self, name: str, *, observations: Sequence[str] = ()) -> LearningClaim:
        with self._lock:
            if len(self._claims) >= self._max:
                self._claims.pop(next(iter(self._claims)))
            claim = LearningClaim(name=name, observations=tuple(observations))
            self._claims[name] = claim
            return claim

    def record_update(self, name: str, update_ref: str) -> None:
        with self._lock:
            if claim := self._claims.get(name):
                claim.update_ref = update_ref

    def record_artifact(self, name: str, artifact_ref: str) -> None:
        with self._lock:
            if claim := self._claims.get(name):
                claim.artifact_ref = artifact_ref

    def record_retrieval(self, name: str, occasion: str) -> None:
        with self._lock:
            if claim := self._claims.get(name):
                claim.retrievals = (*claim.retrievals, occasion)

    def record_delta(self, name: str, delta: float, *, comparator: str) -> None:
        with self._lock:
            if claim := self._claims.get(name):
                claim.outcome_delta = float(delta)
                claim.comparator = comparator

    def verify(self, name: str) -> dict[str, Any]:
        """Whether this claim is learning, and which link fails if not."""
        with self._lock:
            claim = self._claims.get(name)
        if claim is None:
            return {"name": name, "known": False, "accepted": False}
        return {"name": name, "known": True, "accepted": claim.complete, **claim.to_dict()}

    def report(self) -> dict[str, Any]:
        with self._lock:
            claims = list(self._claims.values())
        broken: dict[str, int] = {}
        for claim in claims:
            for link in claim.missing:
                broken[link.value] = broken.get(link.value, 0) + 1
        return {
            "claims": len(claims),
            "accepted": sum(1 for c in claims if c.complete),
            "rejected": sum(1 for c in claims if not c.complete),
            "missing_links": dict(sorted(broken.items())),
            "rejections": [c.explain() for c in claims if not c.complete][:20],
        }


_lock = threading.Lock()
_audit: LearningAudit | None = None


def get_learning_audit() -> LearningAudit:
    global _audit
    with _lock:
        if _audit is None:
            _audit = LearningAudit()
        return _audit


def reset_learning_audit_for_test() -> LearningAudit:
    global _audit
    with _lock:
        _audit = LearningAudit()
        return _audit
