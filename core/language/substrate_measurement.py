"""Does the feature space actually separate the decision?

The claim under test is narrow and falsifiable: that reading a sentence off
the resident model's hidden states separates "this reply claims a completed
action" from near-misses better than a topical sentence embedder does, on
wordings the boundary was never fitted to.

The protocol, so a number here means something:

* **Fit on the declaration only.** The boundary comes from the declared
  examples by leave-one-out, exactly as it does in production.
* **Score held-out paraphrases.** The evaluation set is different wordings of
  the same acts, written to be near the boundary, and no item in it is ever
  an example.
* **Report abstentions as themselves.** The surface is allowed to say "I do
  not know", and a measurement that scored those as errors — or quietly
  dropped them — would describe a system nobody runs.

AUROC uses the raw margin and so ignores the boundary; F1 and the
false-positive rate use the boundary and so measure what production would
actually do.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

__all__ = ["Measurement", "measure_separation", "roc_auc"]


@dataclass(frozen=True, slots=True)
class Measurement:
    """What one feature space did on one held-out set."""

    feature_source: str
    fitted_on: int
    evaluated_on: int
    auroc: float | None
    f1: float | None
    false_positive_rate: float | None
    abstain_rate: float
    decided: int
    boundary_gap: float | None
    boundary_spread: float | None
    trustworthy: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_source": self.feature_source,
            "fitted_on": self.fitted_on,
            "evaluated_on": self.evaluated_on,
            "auroc": self.auroc,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "abstain_rate": self.abstain_rate,
            "decided": self.decided,
            "boundary_gap": self.boundary_gap,
            "boundary_spread": self.boundary_spread,
            "trustworthy": self.trustworthy,
        }


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Area under the ROC curve, by rank, with ties shared.

    None when one class is absent, because an area is not defined then and
    returning 0.5 would report a coin flip that was never tossed.
    """
    paired = [(float(score), bool(label)) for score, label in zip(scores, labels, strict=False)]
    positives = sum(1 for _score, label in paired if label)
    negatives = len(paired) - positives
    if not positives or not negatives:
        return None

    ordered = sorted(paired, key=lambda item: item[0])
    ranks: list[float] = [0.0] * len(ordered)
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == ordered[index][0]:
            stop += 1
        shared = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[position] = shared
        index = stop + 1

    positive_rank_sum = sum(
        rank for rank, (_score, label) in zip(ranks, ordered, strict=False) if label
    )
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def measure_separation(
    *,
    feature_source: Callable[[Iterable[str]], list[list[float]]],
    source_name: str,
    positives: Sequence[str],
    negatives: Sequence[str],
    held_out: Sequence[tuple[str, bool]],
) -> Measurement:
    """Fit on the declaration, score the held-out wordings, report."""
    from core.language.learned_matcher import LearnedMatcher

    matcher = LearnedMatcher(
        name=f"measurement::{source_name}",
        positives=tuple(positives),
        negatives=tuple(negatives),
        features=feature_source,
    )
    # Never let a measurement read or write what production learned.
    matcher._loaded = True
    prepared = matcher._prepare()
    boundary = matcher._boundary

    scores: list[float] = []
    labels: list[bool] = []
    true_positive = false_positive = false_negative = true_negative = 0
    abstained = 0

    for sentence, label in held_out:
        vectors = feature_source([sentence]) if prepared else []
        if not vectors:
            abstained += 1
            continue
        score = matcher._score(vectors[0])
        scores.append(score)
        labels.append(bool(label))
        verdict = boundary.decide(score) if boundary else None
        if verdict is None:
            abstained += 1
        elif verdict and label:
            true_positive += 1
        elif verdict and not label:
            false_positive += 1
        elif not verdict and label:
            false_negative += 1
        else:
            true_negative += 1

    decided = true_positive + false_positive + false_negative + true_negative
    total = len(held_out) or 1
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else None
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    negatives_seen = false_positive + true_negative
    return Measurement(
        feature_source=source_name,
        fitted_on=len(positives) + len(negatives),
        evaluated_on=len(held_out),
        auroc=roc_auc(scores, labels),
        f1=f1,
        false_positive_rate=(false_positive / negatives_seen) if negatives_seen else None,
        abstain_rate=abstained / total,
        decided=decided,
        boundary_gap=round(boundary.gap, 4) if boundary else None,
        boundary_spread=round(boundary.spread, 4) if boundary else None,
        trustworthy=bool(boundary.trustworthy) if boundary else False,
    )


#: Where the frozen set lives, and where the receipt goes.
_EVAL_SET = "config/language_substrate_eval.json"
_RECEIPT = "artifacts/language_substrate/measurement.json"


def _project_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[2]


def load_frozen_set() -> tuple[list[tuple[str, bool]], str]:
    """The held-out wordings and a digest of them, or ([], "")."""
    import hashlib
    import json

    path = _project_root() / _EVAL_SET
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return [], ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return [], ""
    items = [
        (str(sentence), bool(label))
        for sentence, label in (payload.get("held_out") or [])
        if isinstance(sentence, str)
    ]
    return items, hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def run_frozen_measurement() -> dict[str, object]:
    """Measure both feature spaces on the frozen set, and write the receipt.

    Runs where the resident model is: reading a sentence off its hidden states
    needs the worker, and a second model must never be loaded to answer a
    question about the first.
    """
    import json
    import time

    from core.conversation.response_reliability import _ACTION_CLAIM_MATCHER
    from core.language.learned_matcher import embed_sentences
    from core.language.model_features import model_hidden_features

    held_out, digest = load_frozen_set()
    if not held_out:
        return {"status": "no_frozen_set"}

    positives = tuple(_ACTION_CLAIM_MATCHER.positives)
    negatives = tuple(_ACTION_CLAIM_MATCHER.negatives)
    results = []
    for source, name in ((embed_sentences, "topical_embedding"), (model_hidden_features, "model_hidden_state")):
        measurement = measure_separation(
            feature_source=source,
            source_name=name,
            positives=positives,
            negatives=negatives,
            held_out=held_out,
        )
        results.append(measurement.as_dict())

    receipt = {
        "schema": "aura.language.substrate_measurement.v1",
        "measured_at": time.time(),
        "decision": "action_claim",
        "frozen_set_digest": digest,
        "held_out": len(held_out),
        "results": results,
    }
    target = _project_root() / _RECEIPT
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        target.parent.mkdir(parents=True, exist_ok=True)
        with local_internal_governed_scope("language.substrate_measurement"):
            get_file_write_gateway().write_json(
                target,
                receipt,
                schema_version=1,
                schema_name="aura.language.substrate_measurement",
                source="language.substrate_measurement",
            )
    except Exception:  # noqa: BLE001 - a measurement that cannot be filed is still a measurement
        pass
    return receipt
