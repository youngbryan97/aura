"""Compare attributed meaning evidence without inferring a speaker's motive.

Incongruity can be an error, a change, a joke, fiction, a misleading claim, or
an incomplete observation. A source-bound contradiction is evidence to ask
about those possibilities, not a classifier for any one of them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from core.language.contextual_usage import UsageEvent, UsageRelation


def _relation_fit(left: UsageRelation, right: UsageRelation) -> str | None:
    if (left.source_id == right.source_id
            or (left.subject, left.predicate, left.scope)
            != (right.subject, right.predicate, right.scope)
            or left.kind == "withholding" or right.kind == "withholding"):
        return None
    if left.value == right.value:
        return "aligned" if left.polarity == right.polarity else "incongruent"
    if left.exclusive and right.exclusive and left.polarity and right.polarity:
        return "incongruent"
    return None


def compare_pragmatic_context(
    focal: UsageEvent, context: Sequence[UsageEvent], *, as_of: float,
) -> dict[str, Any]:
    """Measure compatible relations and explicit omissions in one context.

    Only a witnessed contradiction in the same declared scope is counted.
    Different sources may disagree, but neither source is thereby marked as
    truthful or deceptive. Future evidence is excluded by the caller's cutoff.
    """
    if not math.isfinite(as_of) or as_of < focal.observed_at:
        raise ValueError("pragmatic context cannot precede the focal event")
    rows = []
    withheld = []
    focal_relations = tuple(relation for relation in focal.relations
                            if relation.observed_at <= as_of)

    def retain(left: UsageRelation, right: UsageRelation) -> None:
        fit = _relation_fit(left, right)
        if fit is not None:
            rows.append({"status": fit, "focal_source_id": left.source_id,
                         "context_source_id": right.source_id,
                         "subject": left.subject, "predicate": left.predicate,
                         "scope": left.scope, "context_kind": right.kind})

    for index, relation in enumerate(focal_relations):
        if relation.kind == "withholding":
            withheld.append({"source_id": relation.source_id,
                             "subject": relation.subject,
                             "predicate": relation.predicate,
                             "scope": relation.scope})
        for other in focal_relations[index + 1:]:
            retain(relation, other)
    for event in context:
        if (event.source_id == focal.source_id or event.context_id != focal.context_id
                or event.observed_at > as_of):
            continue
        for relation in event.relations:
            if relation.observed_at > as_of:
                continue
            if relation.kind == "withholding":
                withheld.append({"source_id": relation.source_id,
                                 "subject": relation.subject,
                                 "predicate": relation.predicate,
                                 "scope": relation.scope})
            for claim in focal_relations:
                retain(claim, relation)
    aligned = sum(row["status"] == "aligned" for row in rows)
    incongruent = len(rows) - aligned
    return {"status": "measured" if rows else "no_comparable_relations",
            "comparable": len(rows), "aligned": aligned,
            "incongruent": incongruent,
            "congruence": aligned / len(rows) if rows else None,
            "comparisons": tuple(rows),
            "explicit_withholding": tuple(withheld),
            "intent": "unmeasured", "serving_authority": False}


__all__ = ["compare_pragmatic_context"]
