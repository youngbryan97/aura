"""core/science/claim_ladder.py — how much a claim has actually earned.

``core/organism/model_validation.py`` already refuses a claim with no test and
already says what KIND of evidence a test provides — measured live, measured on
a synthetic system, unmeasured, retracted. That is the right axis and it is not
the only one. "The workspace broadcasts" and "the workspace broadcast is why
the answer was better" are both MEASURED_LIVE and they are not the same claim,
and only the second is a reason to keep the workspace.

The ladder is the second axis: not what kind of measurement, but how far the
measurement gets you.

    EXISTS      the code is there and runs
    WIRED       it is on a live path; something calls it in production
    CAUSAL      lesion it and behaviour changes
    USEFUL      the change is an improvement on a task, at matched compute
    GENERALIZES it improves held-out tasks it was not tuned on
    REPLICATED  someone outside this repository reproduced it
    THEORY      it discriminates between competing explanations

Each rung requires the one below. That is the whole enforcement: a claim
registered at USEFUL with no CAUSAL artifact is refused, because "it helps" from
a system that never showed the mechanism does anything is a correlation with a
mechanism attached by hand. Aura has retracted two claims for exactly that
shape - the CAA steering A/B whose null passed decisively, and a phi estimator
that ranked a memoryless system above a coupled one.

The artifacts are paths, and they must exist. A rung supported by a file that
was deleted is a rung that fell off.

What this is not
----------------
It is not a confidence score and the rungs are not weights. A claim at CAUSAL
is not "60 percent proven"; it is a claim that has shown a mechanism and has
not shown a benefit, which is a specific and reportable state. Collapsing that
to a number is what the ladder exists to stop.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = [
    "Rung",
    "Rung",
    "LadderClaim",
    "ClaimLadder",
    "get_ladder",
    "reset_ladder_for_test",
    "ladder_reset",
    "PrerequisiteMissing",
]

ROOT = Path(__file__).resolve().parent.parent.parent


class Rung(IntEnum):
    """How far a claim has got. Ordered, and each requires the one below."""

    EXISTS = 1
    WIRED = 2
    CAUSAL = 3
    USEFUL = 4
    GENERALIZES = 5
    EXTERNALLY_REPLICATED = 6
    THEORY_SUPPORT = 7

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def question(self) -> str:
        return {
            Rung.EXISTS: "is the code there and does it run",
            Rung.WIRED: "does anything in production call it",
            Rung.CAUSAL: "does lesioning it change behaviour",
            Rung.USEFUL: "is the change an improvement, at matched compute",
            Rung.GENERALIZES: "does it improve tasks it was not tuned on",
            Rung.EXTERNALLY_REPLICATED: "has anyone outside this repository reproduced it",
            Rung.THEORY_SUPPORT: "does it discriminate between competing explanations",
        }[self]


class PrerequisiteMissing(ValueError):
    """A claim stood on a rung whose supports were not there."""


@dataclass(frozen=True, slots=True)
class Support:
    """The artifact that establishes one rung."""

    rung: Rung
    artifact: str
    note: str = ""

    @property
    def exists(self) -> bool:
        return (ROOT / self.artifact.split("::")[0]).exists()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung.label,
            "artifact": self.artifact,
            "exists": self.exists,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class LadderClaim:
    """A statement about Aura, and how far it has actually got."""

    statement: str
    owner: str
    supports: tuple[Support, ...] = ()
    #: The claim's own boundary: what it does NOT say. Required, because the
    #: overclaim is almost never in the sentence - it is in what a reader adds.
    boundary: str = ""
    #: Names the claim registered in core/organism/model_validation.py, so the
    #: two registries agree about what is being asserted.
    validation_claim: str = ""

    @property
    def rung(self) -> Rung | None:
        """The highest rung whose supports, and every rung below, are present."""
        held = {s.rung for s in self.supports if s.exists}
        reached: Rung | None = None
        for rung in Rung:
            if rung not in held:
                break
            reached = rung
        return reached

    @property
    def broken_supports(self) -> tuple[Support, ...]:
        return tuple(s for s in self.supports if not s.exists)

    def to_dict(self) -> dict[str, Any]:
        rung = self.rung
        return {
            "statement": self.statement,
            "owner": self.owner,
            "rung": rung.label if rung else "unsupported",
            "rung_value": int(rung) if rung else 0,
            "question_answered": rung.question if rung else "",
            "boundary": self.boundary,
            "supports": [s.to_dict() for s in self.supports],
            "broken_supports": [s.artifact for s in self.broken_supports],
            "validation_claim": self.validation_claim,
        }


class ClaimLadder:
    """Every claim, and the rung its artifacts actually reach."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.science.claim_ladder.ClaimLadder", reentrant=True)
        self._claims: dict[str, LadderClaim] = {}

    def register(
        self,
        statement: str,
        *,
        owner: str,
        supports: Sequence[tuple[Rung | str, str] | Support],
        boundary: str,
        validation_claim: str = "",
    ) -> LadderClaim:
        """Register a claim. Refuses a rung whose prerequisites are missing.

        ``boundary`` is required. A claim with no stated boundary is a claim
        whose limits the next reader will invent.
        """
        if not boundary.strip():
            raise ValueError(
                f"claim {statement!r} states no boundary; without one, what it does NOT "
                "say is left to whoever quotes it"
            )
        normalised: list[Support] = []
        for entry in supports:
            if isinstance(entry, Support):
                normalised.append(entry)
                continue
            rung, artifact = entry
            normalised.append(
                Support(rung if isinstance(rung, Rung) else Rung[str(rung).upper()], artifact)
            )
        normalised.sort(key=lambda s: int(s.rung))

        held = {s.rung for s in normalised if s.exists}
        missing_artifacts = [s.artifact for s in normalised if not s.exists]
        if missing_artifacts:
            raise PrerequisiteMissing(
                f"claim {statement!r} rests on artifacts that do not exist: "
                + ", ".join(missing_artifacts)
            )
        for support in normalised:
            for lower in Rung:
                if lower >= support.rung:
                    break
                if lower not in held:
                    raise PrerequisiteMissing(
                        f"claim {statement!r} claims {support.rung.label} with no "
                        f"{lower.label} evidence. {support.rung.question.capitalize()} "
                        f"cannot be answered before: {lower.question}"
                    )

        claim = LadderClaim(
            statement=statement,
            owner=owner,
            supports=tuple(normalised),
            boundary=boundary,
            validation_claim=validation_claim,
        )
        with self._lock:
            self._claims[statement] = claim
        return claim

    def get(self, statement: str) -> LadderClaim | None:
        with self._lock:
            return self._claims.get(statement)

    def claims(self) -> list[LadderClaim]:
        with self._lock:
            return sorted(self._claims.values(), key=lambda c: c.statement)

    def audit(self) -> dict[str, Any]:
        """Which claims have lost their supports since they were registered.

        A rung is not a badge. An artifact deleted in a refactor takes its rung
        with it, and this is where that shows up rather than in a document
        somebody re-reads once a quarter.
        """
        with self._lock:
            claims = list(self._claims.values())
        degraded = [c for c in claims if c.broken_supports]
        by_rung: dict[str, int] = {}
        for claim in claims:
            rung = claim.rung
            key = rung.label if rung else "unsupported"
            by_rung[key] = by_rung.get(key, 0) + 1
        return {
            "claims": len(claims),
            "by_rung": dict(sorted(by_rung.items())),
            "degraded": [
                {"statement": c.statement, "missing": [s.artifact for s in c.broken_supports]}
                for c in degraded
            ],
            "ok": not degraded,
            "at_or_above_causal": sum(
                1 for c in claims if (c.rung or 0) >= Rung.CAUSAL
            ),
            "at_or_above_useful": sum(
                1 for c in claims if (c.rung or 0) >= Rung.USEFUL
            ),
        }

    def to_json(self) -> str:
        return json.dumps(
            {"claims": [c.to_dict() for c in self.claims()], "audit": self.audit()}, indent=2
        )


_lock = checked_lock("core.science.claim_ladder.singleton")
_ladder: ClaimLadder | None = None


def get_ladder() -> ClaimLadder:
    global _ladder
    with _lock:
        if _ladder is None:
            _ladder = ClaimLadder()
            _install_defaults(_ladder)
        return _ladder


def reset_ladder_for_test(*, defaults: bool = False) -> ClaimLadder:
    """Replace the process-wide ladder. Prefer :func:`ladder_reset`.

    The ladder is a singleton installed once per process, so a bare reset is
    permanent for the rest of the session: a test that empties it leaves every
    later reader seeing an empty ladder. That has already happened once here,
    in health_fragments, and the fix is the same - the context manager below
    puts back what it took.
    """
    global _ladder
    with _lock:
        _ladder = ClaimLadder()
        if defaults:
            _install_defaults(_ladder)
        return _ladder


@contextlib.contextmanager
def ladder_reset(*, defaults: bool = False) -> Iterator[ClaimLadder]:
    """An empty ladder for the body, and the real one back afterwards."""
    global _ladder
    with _lock:
        saved = _ladder
        _ladder = ClaimLadder()
        if defaults:
            _install_defaults(_ladder)
        fresh = _ladder
    try:
        yield fresh
    finally:
        with _lock:
            _ladder = saved


def _install_defaults(ladder: ClaimLadder) -> None:
    """The claims this session's work can actually support, at their real rung.

    Deliberately short. Registering everything Aura might claim at the rung it
    would like to reach is the failure the ladder exists to prevent, so this
    holds only what has an artifact behind every rung named.
    """
    ladder.register(
        "Duplicate evidence cannot inflate belief in the AtomSpace",
        owner="core/evidence/packet.py",
        supports=[
            (Rung.EXISTS, "core/evidence/packet.py"),
            (Rung.WIRED, "core/knowledge/atomspace.py"),
            (Rung.CAUSAL, "tests/test_evidence_independence.py"),
        ],
        boundary=(
            "Only on paths that pass a source identity. Unsourced assertions keep the "
            "old behaviour and are counted in evidence_report(); nothing here says what "
            "fraction of the live system passes a source."
        ),
    )
    ladder.register(
        "A learner cannot train on an action whose effect was never established",
        owner="core/cognition/action_receipt.py",
        supports=[
            (Rung.EXISTS, "core/cognition/action_receipt.py"),
            (Rung.WIRED, "core/cognition/architecture_invariants.py"),
            (Rung.CAUSAL, "tests/test_cognitive_identity_contracts.py"),
        ],
        boundary=(
            "The gate raises when it is called. Nothing here establishes that every "
            "environment learner calls it; the invariant reports learners that never "
            "qualify anything, which is a different and weaker guarantee."
        ),
    )
    ladder.register(
        "A compiled rule cannot rest on a fact nobody observed",
        owner="core/cognition/cognitive_event.py",
        supports=[
            (Rung.EXISTS, "core/cognition/cognitive_event.py"),
            (Rung.WIRED, "core/cognition/substate.py"),
            (Rung.CAUSAL, "tests/test_cognitive_event_dag.py"),
        ],
        boundary=(
            "True of minimal_support(). Learners that compile from their own episode "
            "features rather than from the event graph are unaffected."
        ),
    )
    ladder.register(
        "Procedures from different learners compete under one value",
        owner="core/cognition/procedure.py",
        supports=[
            (Rung.EXISTS, "core/cognition/procedure.py"),
            (Rung.WIRED, "core/cognition/procedure_adapters.py"),
            (Rung.CAUSAL, "tests/test_procedure_currency.py"),
        ],
        boundary=(
            "The currency reproduces each backend's own arithmetic and ranks across "
            "them. No claim is made that ranking across backends improves any task; "
            "that is the USEFUL rung and there is no artifact for it."
        ),
    )
