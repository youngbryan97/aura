"""Structural invariants over the reason classification itself.

Theseus's ``Objective`` refuses to build a problem whose declarations
contradict: a variable may not be both an optimization variable and an
auxiliary one, two variables may not share a name unless they are the same
object, and a cost weight may not depend on an optimization variable ("the
jacobians computed by our optimizers will be incorrect"). It rejects at
assembly rather than producing a subtly wrong answer at solve time.

The reason sets in ``surface_disposition`` are the same kind of declaration and
had no such check. They are read by ~115 call sites through three different
doors — ``assessment.ok``, ``disposition_for``, and raw ``.reasons`` — and when
those doors disagreed, a reply was served by one gate and destroyed by another.
That has happened twice: ``disposition_for`` did not know ADVISORY_ONLY existed
and returned REPAIR for every advisory (memory lost), and
``cognitive_ingress._conversation_pre_admission`` read ``.reasons`` directly and
refused a memory the same assessment had just marked ok.

These are the contradictions a set membership can express. A check that raises
counts as a violation.
"""

from __future__ import annotations

import re
from typing import Iterator

from core.verify.invariants import Severity, Violation, invariant

#: Reason names are compared as strings across module and process boundaries,
#: so a stray space or capital is a silent non-match, never an error.
_REASON_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _sets() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    from core.conversation.surface_disposition import (
        ADVISORY_ONLY_REASONS,
        CONTINUITY_SAFE_REASONS,
        UNSPEAKABLE_REASONS,
    )

    return ADVISORY_ONLY_REASONS, CONTINUITY_SAFE_REASONS, UNSPEAKABLE_REASONS


@invariant(
    "surface.advisory_never_destroys",
    scope="conversation",
    owner="core/conversation/surface_disposition.py",
    description="no reason may both never-condemn and always-destroy a reply",
)
def _advisory_is_not_unspeakable() -> Iterator[Violation]:
    advisory, _continuity, unspeakable = _sets()
    for reason in sorted(advisory & unspeakable):
        yield Violation(
            subject=reason,
            message=(
                "declared ADVISORY_ONLY (invisible to `ok`, served) and "
                "UNSPEAKABLE (destroys the reply). Two gates reading the same "
                "assessment reach opposite verdicts on the same turn."
            ),
            remedy="decide which one it is; a heuristic belongs in neither",
        )


@invariant(
    "surface.advisory_keeps_the_exchange",
    scope="conversation",
    owner="core/conversation/surface_disposition.py",
    description="a reason too weak to condemn a reply cannot cost the turn its memory",
)
def _advisory_is_continuity_safe() -> Iterator[Violation]:
    advisory, continuity, _unspeakable = _sets()
    for reason in sorted(advisory - continuity):
        yield Violation(
            subject=reason,
            message=(
                "ADVISORY_ONLY but not CONTINUITY_SAFE: it cannot hold the "
                "reply against itself, yet it can drop the record that the "
                "person spoke at all."
            ),
            remedy="add it to CONTINUITY_SAFE_REASONS",
        )


@invariant(
    "surface.reason_names_are_comparable",
    scope="conversation",
    owner="core/conversation/surface_disposition.py",
    description="reason names are matched by string equality everywhere",
)
def _reason_names_are_well_formed() -> Iterator[Violation]:
    for group in _sets():
        for reason in sorted(group):
            if not _REASON_NAME.match(reason):
                yield Violation(
                    subject=reason,
                    message="not a lower_snake_case token; string comparison will miss it",
                    remedy="rename to lower_snake_case",
                )


@invariant(
    "surface.disposition_agrees_with_the_sets",
    scope="conversation",
    owner="core/conversation/surface_disposition.py",
    description="disposition_for must implement the classification it is given",
    severity=Severity.ERROR,
)
def _disposition_matches_membership() -> Iterator[Violation]:
    from core.conversation.surface_disposition import (
        SurfaceDisposition,
        disposition_for,
    )

    advisory, _continuity, unspeakable = _sets()
    for reason in sorted(advisory - unspeakable):
        verdict = disposition_for((reason,))
        if verdict is not SurfaceDisposition.SERVE:
            yield Violation(
                subject=reason,
                message=f"declared advisory, but disposition_for returns {verdict.value}",
                remedy="the exact drift that lost memories on 2026-07-30",
            )
    for reason in sorted(unspeakable):
        verdict = disposition_for((reason,))
        if verdict is not SurfaceDisposition.DISCARD:
            yield Violation(
                subject=reason,
                message=f"declared unspeakable, but disposition_for returns {verdict.value}",
                remedy="UNSPEAKABLE_REASONS is the allowlist disposition_for reads",
            )
