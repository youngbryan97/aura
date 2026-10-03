"""Which operation a mention is an argument of, read from where it stands.

Every operation is offered every mention between its neighbouring operations,
plus the request's best mentions overall. In a two- or three-step request that
window rarely holds another operation's argument. In a five-step one it nearly
always does: "add primary result and firing adjustment. Call that the refined
result. Unify the two paths and multiply refined result by auxiliary result"
offers "primary result" to the multiplication. The role head scores a mention
against an operation from their two pooled span vectors only, so the
multiplication took "primary result", and the program came out reordered. On
the 48 five-step composition requests of 2 October every operation was found
and named, and all 16 misses were bindings of this kind.

Where a mention stands relative to the operations around it is evidence the
role head never sees: which side of an operation it is on, how many tokens
away, how many other operations lie between, and whether this is the nearest
operation on either side. A logistic readout over those features, fitted on
the training rows' annotated argument mentions against every other operation
of the same request, gives P(operation o owns mention m), normalised over the
request's operations. Its log is one more term in each argument option's
score.

Training rows only; validation and test rows are refused by the fitter.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from core.learning.semantic_program_ir import TokenSpan

OWNERSHIP_SCHEMA: Final = "aura.semantic_argument_ownership.v1"

#: Names of the features, in the order the weights read them.
FEATURES: Final = (
    "after",
    "log_gap",
    "after_times_log_gap",
    "no_operation_between",
    "one_operation_between",
    "several_operations_between",
    "nearest_before",
    "nearest_after",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _overlaps(left: TokenSpan, right: TokenSpan) -> bool:
    return left.start < right.end and right.start < left.end


def ownership_features(
    mention: TokenSpan, operation: TokenSpan, operations: Sequence[TokenSpan]
) -> list[float]:
    """Where ``mention`` stands relative to ``operation`` among the request's operations."""
    after = mention.start >= operation.end
    gap = mention.start - operation.end if after else operation.start - mention.end
    low, high = (operation.end, mention.start) if after else (mention.end, operation.start)
    between = sum(
        1 for other in operations if other != operation and other.start >= low and other.end <= high
    )
    before_it = [other for other in operations if other.end <= mention.start]
    after_it = [other for other in operations if other.start >= mention.end]
    nearest_before = bool(before_it) and max(before_it, key=lambda other: other.end) == operation
    nearest_after = bool(after_it) and min(after_it, key=lambda other: other.start) == operation
    side = 1.0 if after else -1.0
    log_gap = math.log1p(max(gap, 0))
    return [
        side,
        log_gap,
        side * log_gap,
        float(between == 0),
        float(between == 1),
        float(between >= 2),
        float(nearest_before),
        float(nearest_after),
    ]


@dataclass(frozen=True)
class ArgumentOwnership:
    """P(operation owns mention), from position alone."""

    weight: tuple[float, ...]
    bias: float
    fit_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.weight) != len(FEATURES) or not all(
            math.isfinite(value) for value in (*self.weight, self.bias)
        ):
            raise ValueError("argument ownership parameters are invalid")

    def log_probabilities(
        self, mention: TokenSpan, operations: Sequence[TokenSpan]
    ) -> tuple[float, ...]:
        """Log P(each operation owns ``mention``); an operation the mention overlaps owns nothing."""
        logits = [
            -math.inf
            if _overlaps(mention, operation)
            else float(
                np.dot(self.weight, ownership_features(mention, operation, operations)) + self.bias
            )
            for operation in operations
        ]
        top = max(logits)
        if not math.isfinite(top):
            return tuple(-math.inf for _ in operations)
        total = math.log(sum(math.exp(value - top) for value in logits if math.isfinite(value)))
        return tuple(value - top - total if math.isfinite(value) else -math.inf for value in logits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OWNERSHIP_SCHEMA,
            "features": list(FEATURES),
            "weight": list(self.weight),
            "bias": self.bias,
            "fit_receipt": dict(self.fit_receipt),
        }

    @property
    def identity_sha256(self) -> str:
        return _sha(self.to_dict())


def argument_ownership_from_dict(value: Mapping[str, Any]) -> ArgumentOwnership:
    if value.get("schema") != OWNERSHIP_SCHEMA or list(value.get("features", ())) != list(FEATURES):
        raise ValueError("argument ownership payload is not this schema")
    return ArgumentOwnership(
        tuple(float(item) for item in value["weight"]),
        float(value["bias"]),
        dict(value["fit_receipt"]),
    )


def fit_argument_ownership(examples: Sequence[Any]) -> ArgumentOwnership:
    """Fit on each training argument mention against every operation of its request."""
    from sklearn.linear_model import LogisticRegression

    examples = tuple(examples)
    if not examples or any(item.split != "train" for item in examples):
        raise ValueError("argument ownership is fitted on training rows only")
    rows, owned = [], []
    for item in examples:
        operations = [instruction.operation_span for instruction in item.ir.instructions]
        for instruction in item.ir.instructions:
            for mention in instruction.argument_spans:
                for operation in operations:
                    if _overlaps(mention, operation):
                        continue
                    rows.append(ownership_features(mention, operation, operations))
                    owned.append(int(operation == instruction.operation_span))
    if len(set(owned)) < 2:
        raise ValueError("argument ownership needs requests with more than one operation")
    fitted = LogisticRegression(max_iter=2000, C=1.0).fit(np.asarray(rows), np.asarray(owned))
    receipt = {
        "training_sources": sorted(item.ir.source_text_sha256 for item in examples),
        "training_rows": len(examples),
        "pairs": len(owned),
        "owned_pairs": int(sum(owned)),
        "splits_used": ["train"],
    }
    return ArgumentOwnership(
        tuple(float(value) for value in fitted.coef_[0]),
        float(fitted.intercept_[0]),
        {**receipt, "receipt_sha256": _sha(receipt)},
    )
