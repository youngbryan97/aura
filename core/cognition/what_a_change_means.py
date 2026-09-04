"""The three things that must be true of every way Aura changes herself.

She has many mechanisms for changing what she is and had no one account of
what a change *is*. The growth ladder adjudicates a proposal; the
developmental actions install one; the promotion ledger records one. Each was
written well and each held its own rules, and the rules disagreed in ways no
test could see, because every test was inside one mechanism.

An external review put it as: the rate at which mutable mechanisms are added
exceeds the rate at which mutation semantics are unified. The three sentences
below are that unification. They are small on purpose — a rule that spans
subsystems has to be checkable in every one of them.

    Reject(m)            ⇒  the state afterwards is the state before
    Promote(m, shadow)   ⇒  m is reversible and the record says what it replaced
    Denied(m)            ⇒  m does not run, at every interface that asks

The second is not the invariant the review proposed. It asked for
``Authority(m) = 0 until validation``, which would mean a shadow change is
isolated. It is not: a change here is installed the moment it is made, and
building isolation for a registry mutation would mean a second copy of every
registry. What is enforceable — and what was promised and untrue — is that a
shadow change can be undone and the record knows what to undo it to. Saying
the weaker true thing beats asserting the stronger false one.

Each check reads a live subsystem rather than a source file. A rule proven by
reading source is a rule about source.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from core.verify.invariants import Severity, Violation, invariant

logger = logging.getLogger(__name__)

__all__ = ["WHAT_A_CHANGE_MEANS", "what_a_change_means"]


@invariant(
    "development.a_refusal_changes_nothing",
    scope="cognition",
    severity=Severity.ERROR,
    owner="core/cognition/what_she_can_take_back.py",
    observational=False,
)
def _a_refusal_changes_nothing() -> Iterator[Violation]:
    """Reject(m) ⇒ the state afterwards is the state before.

    Held by running one: a change is made inside a trial, the trial is not
    kept, and every registry must read as it did. This is the mechanism that
    letting go of a part used to get wrong by hand.
    """
    try:
        from core.cognition.an_invented_kind import WHERE_FROM
        from core.cognition.what_she_can_take_back import (
            as_it_stands,
            only_if_it_pays,
        )
    except ImportError:
        return

    was = as_it_stands()
    probe = "an_invariant_checking_itself"
    with only_if_it_pays("the invariant checking itself"):
        WHERE_FROM[probe] = lambda a, b: a
    if probe in WHERE_FROM:
        WHERE_FROM.pop(probe, None)
        yield Violation(
            invariant="development.a_refusal_changes_nothing",
            subject="core/cognition/what_she_can_take_back.py",
            message=(
                "a change that was not kept stayed in the registry, so a "
                "rejected development leaves the state altered"
            ),
            severity=Severity.ERROR,
        )
        return
    stubborn = was.what_changed()
    if stubborn:
        yield Violation(
            invariant="development.a_refusal_changes_nothing",
            subject=", ".join(sorted(stubborn)),
            message=(
                "these registries did not read as they did before the trial: "
                f"{stubborn}"
            ),
            severity=Severity.ERROR,
        )


@invariant(
    "development.a_promotion_can_go_back",
    scope="cognition",
    severity=Severity.ERROR,
    owner="core/cognition/how_a_change_is_promoted.py",
    observational=False,
)
def _a_promotion_can_go_back() -> Iterator[Violation]:
    """Promote(m, shadow) ⇒ m is reversible and the record says what it replaced.

    The promotion module claimed this for a long time while no caller passed
    ``replaced`` at all, so the stack was empty and nothing could be undone.
    The check is that a promotion carrying a restorable snapshot actually
    restores.
    """
    try:
        from core.cognition.an_invented_kind import WHERE_FROM
        from core.cognition.how_a_change_is_promoted import (
            a_ledger_of_its_own,
            promote,
            put_it_back,
            what_it_replaced,
        )
        from core.cognition.what_she_can_take_back import as_it_stands
    except ImportError:
        return

    # A fresh address each run. A fixed one means a check that leaves an entry
    # behind — a failing one does exactly that — answers for the next run, and
    # two checks in flight at once answer for each other.
    from uuid import uuid4

    once = uuid4().hex[:8]
    at = f"the words/an_invariant_checking_a_rollback_{once}"
    probe = f"a_word_the_invariant_wrote_{once}"
    stood = as_it_stands()
    # In a ledger of its own, because a check that reads the record must not
    # write into it. Two lines per health report would bury what she actually
    # did under the checks that looked.
    with a_ledger_of_its_own():
        WHERE_FROM[probe] = lambda a, b: a
        promote(
            at,
            became="shadow",
            started_by="she",
            evidence="an invariant checking that a promotion can go back",
            replaced=stood,
        )
        nothing_to_undo = what_it_replaced(at) is None
        if not nothing_to_undo:
            put_it_back(at)
    if nothing_to_undo:
        WHERE_FROM.pop(probe, None)
        yield Violation(
            invariant="development.a_promotion_can_go_back",
            subject=at,
            message="a promotion recorded nothing to go back to",
            severity=Severity.ERROR,
        )
        return
    if probe in WHERE_FROM:
        WHERE_FROM.pop(probe, None)
        yield Violation(
            invariant="development.a_promotion_can_go_back",
            subject=at,
            message=(
                "put_it_back left the change in place, so a promotion that "
                "regressed cannot be undone"
            ),
            severity=Severity.ERROR,
        )


#: The three sentences, by the name each is registered under. Read through the
#: registry rather than called directly, because the third protects the growth
#: ladder and lives beside it — core.cognition may not import
#: core.self_modification, and an invariant belongs next to what it protects.
WHAT_A_CHANGE_MEANS: dict[str, str] = {
    "a refusal changes nothing": "development.a_refusal_changes_nothing",
    "a promotion can go back": "development.a_promotion_can_go_back",
    "a refused proposal does not run": "development.a_refused_proposal_does_not_run",
}


def what_a_change_means(*, execute_checks: bool = True) -> dict[str, object]:
    """The three sentences, and whether each holds right now.

    A sentence whose invariant is not registered reads "not registered"
    rather than True. An unregistered check is one nobody runs, and reporting
    that as holding is the failure this whole module exists to stop.
    """
    from core.verify.invariants import get_registry, latest_invariant_result

    known = {one.name: one for one in get_registry().specs()}
    held: dict[str, object] = {}
    for said, name in WHAT_A_CHANGE_MEANS.items():
        spec = known.get(name)
        if spec is None:
            held[said] = ["not registered, so nobody runs it"]
            continue
        if not execute_checks:
            latest = latest_invariant_result(name)
            if latest is None:
                held[said] = ["not yet verified in this process"]
                continue
            broken = [
                str(one.get("message", "invariant failed"))
                for one in latest.get("violations", [])
            ]
            held[said] = broken or True
            continue
        try:
            broken = [one.message for one in spec.check()]
        except Exception as exc:  # noqa: BLE001 - a check that raises is a breach
            broken = [f"{type(exc).__name__}: {exc}"]
        held[said] = broken or True
    return {"schema": "aura.development.mutation_semantics.v1", "holds": held}
