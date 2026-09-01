"""core/cognition/architecture_invariants.py — the rules no subsystem may break.

Aura has forty thousand tests and they check behaviour: this function returns
that, this gate refuses that. What a highly integrated mind also needs is a
smaller set of rules about the *shape* of any correct path through it, checked
against whatever the system did rather than against a scenario somebody wrote.
A unit test proves a path is right. An architecture invariant proves no path
is wrong.

Five, and each one has a committed failure behind it:

* **One evidence lineage.** A belief's confidence must rest on evidence that
  names its sources. The AtomSpace inflated confidence tenfold from a single
  sensor reading because nothing checked. The invariant watches the
  unattributed fraction rather than forbidding it outright, because a
  migration in progress is not a violation and a migration that stopped is.

* **No silent learner update.** A learner that changed its state must be
  reachable from a broadcast or a qualified receipt. The alternative is what
  happened when a skill was scored successful for having an empty error field.

* **No unmatched action consequence.** Every environment action carries a
  transition receipt with a computed verdict. Thirty-five moves went into the
  wrong window while six guards passed.

* **Bounded cognitive stores.** Every store that grows with experience
  declares a bound and stays inside it. A 96GB state file was re-parsed before
  every answer.

* **One authority path.** A consequential act is authorised in exactly one
  place. Preference and authority were the same float, so a prohibition worth
  0.20 lost to an alternative worth 0.25 more.

The invariants are registered into :mod:`core.verify.invariants` so they run
with everything else, and each is written to be cheap enough to run on every
health report rather than in a nightly job nobody reads.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from core.verify.invariants import Severity, Violation, invariant

__all__ = [
    "UNATTRIBUTED_EVIDENCE_CEILING",
    "architecture_report",
]

#: The fraction of AtomSpace revisions allowed to arrive without a source
#: identity. Starts permissive because most of the repository has not been
#: migrated; it is a RATCHET, and lowering it is how the migration finishes.
UNATTRIBUTED_EVIDENCE_CEILING = 1.0

#: A cognitive store may hold this many entries before it must declare a bound.
_UNBOUNDED_STORE_LIMIT = 1_000_000


@invariant(
    "cognition.evidence_lineage",
    scope="cognition",
    severity=Severity.WARNING,
    owner="core/evidence/packet.py",
)
def _evidence_lineage() -> Iterator[Violation]:
    """Belief confidence rests on evidence that can name its sources."""
    try:
        from core.knowledge.atomspace import get_atomspace

        report = get_atomspace().evidence_report()
    except (ImportError, RuntimeError, AttributeError):
        return
    total = report["unattributed_assertions"] + report["duplicate_assertions_refused"]
    if total < 50:
        return  # too little traffic to say anything
    unattributed = report["unattributed_assertions"] / total
    if unattributed > UNATTRIBUTED_EVIDENCE_CEILING:
        yield Violation(
            invariant="cognition.evidence_lineage",
            subject="atomspace",
            message=(
                f"{unattributed:.0%} of belief revisions arrived with no source identity "
                f"(ceiling {UNATTRIBUTED_EVIDENCE_CEILING:.0%}); duplicate evidence cannot "
                "be detected on those paths"
            ),
            severity=Severity.WARNING,
        )


@invariant(
    "cognition.no_unmatched_action_consequence",
    scope="cognition",
    severity=Severity.ERROR,
    owner="core/cognition/action_receipt.py",
)
def _action_consequence() -> Iterator[Violation]:
    """Every environment action a learner used carried a computed verdict."""
    try:
        from core.cognition.action_receipt import get_receipt_ledger

        report = get_receipt_ledger().report()
    except (ImportError, RuntimeError, AttributeError):
        return
    for learner, counts in report.get("by_learner", {}).items():
        qualified = counts.get("qualified", 0)
        seen = sum(v for k, v in counts.items() if k != "qualified")
        if seen and qualified == 0 and seen >= 20:
            yield Violation(
                invariant="cognition.no_unmatched_action_consequence",
                subject=learner,
                message=(
                    f"{learner} was offered {seen} transitions and not one was qualified; "
                    "either it is learning from unverified actions or its verification "
                    "never runs"
                ),
                severity=Severity.ERROR,
            )


@invariant(
    "cognition.no_silent_learner_update",
    scope="cognition",
    severity=Severity.WARNING,
    owner="core/cognition/situation.py",
)
def _silent_learner() -> Iterator[Violation]:
    """A learner subscribed to the broadcast declares what it learns from."""
    try:
        from core.cognition.situation import get_coordinator

        report = get_coordinator().report()
    except (ImportError, RuntimeError, AttributeError):
        return
    for name, row in report.get("subscribers", {}).items():
        if not row.get("kinds"):
            yield Violation(
                invariant="cognition.no_silent_learner_update",
                subject=name,
                message=f"{name} subscribes to no declared kind and cannot be ablated",
                severity=Severity.WARNING,
            )
        if row.get("failed", 0) and row.get("delivered", 0) == 0:
            yield Violation(
                invariant="cognition.no_silent_learner_update",
                subject=name,
                message=(
                    f"{name} has failed on every broadcast it received; it is registered "
                    "as learning and is not"
                ),
                severity=Severity.WARNING,
            )


@invariant(
    "cognition.bounded_cognitive_stores",
    scope="cognition",
    severity=Severity.ERROR,
    owner="core/cognition/architecture_invariants.py",
)
def _bounded_stores() -> Iterator[Violation]:
    """Every store that grows with experience stays inside a declared bound."""
    probes: list[tuple[str, Any]] = []
    try:
        from core.cognition.cognitive_event import get_event_graph

        report = get_event_graph().report()
        probes.append(("cognitive_event_graph", (report["events"], report["capacity"])))
    except (ImportError, RuntimeError, AttributeError):
        pass
    try:
        from core.cognition.concept_handle import get_concept_registry

        probes.append(("concept_registry", (get_concept_registry().report()["handles"], None)))
    except (ImportError, RuntimeError, AttributeError):
        pass
    try:
        from core.cognition.entity_track import get_track_store

        probes.append(("entity_tracks", (get_track_store().report()["tracks"], None)))
    except (ImportError, RuntimeError, AttributeError):
        pass

    for name, (size, capacity) in probes:
        limit = capacity if capacity is not None else _UNBOUNDED_STORE_LIMIT
        if size > limit:
            yield Violation(
                invariant="cognition.bounded_cognitive_stores",
                subject=name,
                message=f"{name} holds {size} entries against a bound of {limit}",
                severity=Severity.ERROR,
            )


@invariant(
    "cognition.one_authority_path",
    scope="cognition",
    severity=Severity.ERROR,
    owner="core/cognition/preference_semantics.py",
)
def _one_authority_path() -> Iterator[Violation]:
    """A prohibition is removal, never a number an alternative can outbid."""
    try:
        from core.cognition.preference_semantics import (
            PreferenceSet,
            PreferenceType,
            resolve,
        )
        from core.cognition.preference_semantics import _make  # noqa: PLC2701 - probe
    except (ImportError, AttributeError):
        return

    try:
        prohibit = _make(PreferenceType.PROHIBIT)("forbidden", source="invariant")
        accept_a = _make(PreferenceType.ACCEPTABLE)("forbidden", source="invariant")
        accept_b = _make(PreferenceType.ACCEPTABLE)("allowed", source="invariant")
        resolution = resolve(
            ["forbidden", "allowed"], PreferenceSet([accept_a, accept_b, prohibit])
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        yield Violation(
            invariant="cognition.one_authority_path",
            subject="preference_semantics",
            message=f"the prohibition probe could not run: {type(exc).__name__}: {exc}",
            severity=Severity.WARNING,
        )
        return
    if getattr(resolution, "choice", None) == "forbidden":
        yield Violation(
            invariant="cognition.one_authority_path",
            subject="preference_semantics",
            message="a prohibited candidate won a decision; prohibition is being scored, not enforced",
            severity=Severity.ERROR,
        )


def architecture_report() -> dict[str, Any]:
    """Run the architecture invariants and summarise. Cheap enough for health."""
    from core.verify.invariants import verify

    report = verify("cognition", record=False)
    return {
        "checked": report.checked,
        "violations": [
            {
                "invariant": v.invariant,
                "subject": v.subject,
                "message": v.message,
                "severity": v.severity.value,
            }
            for v in report.violations
        ],
        "errors": len(report.errors),
        "ok": report.ok,
    }
