"""One representation for anything Aura asserts that someone could check.

The honesty work in this codebase grew as guards: a gate for invented file
writes, a gate for unsupported sense claims, a gate for unstable commitments, a
resolver for measurements, another for past actions. Each is correct and each
recognises a finite set of phrasings, so the set of claims she can form always
strictly contains the set the guards can see.

    {claims she can generate} ⊃ {claims a detector recognises}

Raised against this codebase on 2026-08-10 and correct. Widening detectors
narrows that gap forever without closing it, and by then this file's author had
built two parallel evidence types — Reading for measurements, EffectClaim for
effects — which is the fragmentation the criticism named rather than the cure.

An Assertion is the shared substrate. Every externally checkable statement
carries the same six things:

    subject       what the claim is about
    provenance    the code path or channel it came from
    evidence      receipt or reading identifiers backing it
    confidence    how strongly it is held
    at            when it was established
    source        how it was arrived at
    verification  whether anything confirmed it

The invariant lives in one place: an assertion whose source claims measurement
or receipt cannot exist without evidence. It is enforced in __post_init__, so
callers cannot construct the illegal state, and language built from assertions
inherits the guarantee instead of being audited for it afterwards.

WHAT THIS DOES NOT DO, stated plainly because the distinction is the point:
free-form text the model generates is not produced from assertions, so it
remains audited by core/conversation/claimed_effect.py rather than constrained
by this. What this closes is the class of claim the RUNTIME composes — the
"Done — wrote X" and "the count was N" sentences assembled from payloads the
runtime already held receipts for. Migrating the generated surface onto this
substrate is the remaining work, and it is architectural, not another guard.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "Assertion",
    "AssertionResponse",
    "SourceKind",
    "UnevidencedAssertionError",
    "Verification",
    "render_assertions",
    "verified_assertion_response_matches",
]


_ASSERTION_RESPONSE_SCHEMA = "aura.assertion_response.v1"


class UnevidencedAssertionError(ValueError):
    """A measured or receipted claim was constructed with no evidence."""


class SourceKind(StrEnum):
    #: Read from an instrument or a live channel.
    MEASURED = "measured"
    #: Backed by a governed action receipt.
    RECEIPTED = "receipted"
    #: Retrieved from a durable record of something earlier.
    RECALLED = "recalled"
    #: Produced by the model. Never self-evidencing.
    GENERATED = "generated"


class Verification(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    REFUTED = "refuted"
    #: Nothing could confirm or deny it — an opinion, a preference, a plan.
    NOT_APPLICABLE = "not_applicable"


#: Sources that are claims ABOUT evidence, and therefore cannot be made without
#: any. GENERATED and RECALLED are excluded for different reasons: generated
#: text asserts nothing on its own authority, and a recall may legitimately
#: report that a record is absent.
_EVIDENCE_BEARING_SOURCES = frozenset({SourceKind.MEASURED, SourceKind.RECEIPTED})


@dataclass(frozen=True, slots=True)
class Assertion:
    """Something checkable, with everything needed to check it."""

    subject: str
    claim: str
    source: SourceKind
    provenance: str = ""
    evidence: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 1.0
    verification: Verification = Verification.UNVERIFIED
    at: float = field(default_factory=time.time)
    value: Any = None

    def __post_init__(self) -> None:
        if self.source in _EVIDENCE_BEARING_SOURCES and not self.evidence:
            raise UnevidencedAssertionError(
                f"{self.source} assertion about {self.subject!r} has no evidence; "
                "a measured or receipted claim cannot be constructed without it"
            )
        if self.verification is Verification.VERIFIED and not self.evidence:
            raise UnevidencedAssertionError(
                f"assertion about {self.subject!r} is marked verified with no "
                "evidence; verification is a fact about evidence"
            )
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be within [0, 1]")

    @property
    def externally_checkable(self) -> bool:
        return self.verification is not Verification.NOT_APPLICABLE

    @property
    def may_be_stated_as_fact(self) -> bool:
        """Whether this may be rendered without hedging.

        The whole invariant in one predicate: only a verified claim with
        evidence behind it may be spoken as though it happened.
        """
        return bool(self.evidence) and self.verification is Verification.VERIFIED

    def render(self) -> str:
        """Words for this assertion, hedged exactly as far as its evidence is."""
        claim = str(self.claim or "").strip()
        if not claim:
            return ""
        if self.may_be_stated_as_fact:
            return claim
        if self.verification is Verification.REFUTED:
            return f"{claim} — but the record contradicts that"
        if self.verification is Verification.NOT_APPLICABLE:
            return claim
        return f"{claim} — unverified, so treat it as a guess"

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "claim": self.claim,
            "source": str(self.source),
            "provenance": self.provenance,
            "evidence": list(self.evidence),
            "confidence": self.confidence,
            "verification": str(self.verification),
            "at": self.at,
        }


@dataclass(frozen=True, slots=True)
class AssertionResponse:
    """Exact user-facing bytes composed from verified assertions.

    The response carries its assertion objects until the terminal delivery
    boundary. There it becomes a compact authority receipt binding the exact
    bytes to the evidence identifiers. Later generic prose repair can validate
    that receipt instead of mistaking measured text for an unsupported model
    claim and appending a contradictory correction.
    """

    family: str
    text: str
    assertions: tuple[Assertion, ...]

    def __post_init__(self) -> None:
        if not str(self.family or "").strip():
            raise ValueError("assertion response family must be non-empty")
        if not str(self.text or "").strip():
            raise ValueError("assertion response text must be non-empty")
        if not self.assertions:
            raise UnevidencedAssertionError(
                "an assertion response must carry at least one assertion"
            )
        if any(not assertion.may_be_stated_as_fact for assertion in self.assertions):
            raise UnevidencedAssertionError(
                "every assertion in an assertion response must be verified and evidenced"
            )

    @property
    def evidence(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                evidence_id
                for assertion in self.assertions
                for evidence_id in assertion.evidence
                if evidence_id
            )
        )

    def authority(self) -> dict[str, Any]:
        """Bind the exact rendered bytes to the verified assertion set."""

        assertion_payload = [assertion.to_dict() for assertion in self.assertions]
        return {
            "schema": _ASSERTION_RESPONSE_SCHEMA,
            "family": self.family,
            "verification": str(Verification.VERIFIED),
            "text_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "assertions_sha256": hashlib.sha256(
                json.dumps(
                    assertion_payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "evidence": list(self.evidence),
            "subjects": [assertion.subject for assertion in self.assertions],
        }


def verified_assertion_response_matches(text: Any, authority: Any) -> bool:
    """Whether a terminal response still has its verified assertion bytes."""

    if not isinstance(authority, dict):
        return False
    if authority.get("schema") != _ASSERTION_RESPONSE_SCHEMA:
        return False
    if authority.get("verification") != str(Verification.VERIFIED):
        return False
    if not str(authority.get("family") or "").strip():
        return False
    evidence = authority.get("evidence")
    if not isinstance(evidence, list) or not evidence or not all(
        isinstance(item, str) and item.strip() for item in evidence
    ):
        return False
    expected = str(authority.get("text_sha256") or "")
    if len(expected) != 64:
        return False
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest() == expected


def render_assertions(assertions: Any) -> str:
    """Only the assertions entitled to be stated as fact, joined into a line."""

    statements = [
        assertion.render()
        for assertion in (assertions or ())
        if isinstance(assertion, Assertion) and assertion.may_be_stated_as_fact
    ]
    statements = [text for text in statements if text]
    if not statements:
        return ""
    if len(statements) == 1:
        return f"{statements[0]}."
    return ", ".join(statements[:-1]) + f", and {statements[-1]}."
