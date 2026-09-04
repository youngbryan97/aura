"""Score the battery, or say why it cannot be scored.

The report this produces is designed to be disappointing. Most runs will come
back UNDECIDED with a list of protocols that were discarded, and that is the
correct output of an honest instrument pointed at a hard question.

Three refusals are built in and none of them can be argued around at scoring
time:

* a protocol whose controls failed contributes nothing, rather than a
  smaller number. A sham arm that fired has not produced a weak result.
* a run whose pre-registration digest does not match is VOID. Not adjusted,
  not re-scored — the predictions were changed after the fact and the run is
  no longer evidence.
* the phenomenal question is never scored. It appears in the report as
  NOT_ADDRESSED with the reason attached, so that reading the report cannot
  leave anyone with the impression a number bore on it.

What the report CAN say is which of two hypotheses the evidence favours, by
how much, and which protocols did the work. That is a real result and it is
the one worth having.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.phenomenology.battery import BATTERY, by_id
from core.phenomenology.causal_ladder import log_likelihood_ratio
from core.phenomenology.counterfeit import SeparationReport
from core.phenomenology.hypothesis import (
    COSTUME,
    LOAD_BEARING,
    PHENOMENAL,
    Adjudication,
    Evidence,
    Verdict,
    adjudicate,
)
from core.phenomenology.preregistration import Preregistration
from core.phenomenology.protocol import Outcome, Protocol

__all__ = ["Run", "score", "report"]


@dataclass
class Run:
    """One pass of the gauntlet against one frozen system."""

    registration: Preregistration
    #: Published before the run. A mismatch voids it.
    published_digest: str
    outcomes: list[Outcome] = field(default_factory=list)
    #: What was run against, so a result cannot migrate to another system.
    model_digest: str = ""
    source_commit: str = ""
    #: The adversarial control, if one was built. Its absence is recorded
    #: rather than assumed away.
    counterfeit: SeparationReport | None = None
    operator: str = ""
    replication_of: str = ""

    def void_reason(self) -> str:
        if self.published_digest and self.registration.digest() != self.published_digest:
            return (
                "the predictions do not match the digest published before the "
                "run; they were changed after the fact"
            )
        if self.model_digest and self.registration.model_digest:
            if self.model_digest != self.registration.model_digest:
                return (
                    "registered against a different model than the one that "
                    "ran; the result cannot be carried across checkpoints"
                )
        if not self.outcomes:
            return "nothing was run"
        return ""


#: The three things a registered direction can mean. Kept explicit because
#: "falls below the null" and "returns to the null" are different claims and
#: reading them as one cost a correct S4 result: a lesion that takes an effect
#: to exactly zero has removed it, and scoring that as a failure to go
#: NEGATIVE is a category error about what a null is.
_RISES = ("rise", "rises", "increase", "above", "exceeds", "higher")
_VANISHES = ("vanish", "disappear", "returns to the null", "back to the null",
             "falls to the null", "removed", "collapses to the null")


def _direction_holds(prediction_direction: str, outcome: Outcome) -> bool:
    """Whether the measured value moved the way it was registered to move."""
    if not outcome.has_null:
        return False
    wanted = prediction_direction.lower()
    low, high = min(outcome.nulls), max(outcome.nulls)
    pad = max((high - low) * 0.5, 1e-6)

    if any(word in wanted for word in _VANISHES):
        # The effect went away: back inside the band that says "no effect".
        return low - pad <= outcome.value <= high + pad
    if any(word in wanted for word in _RISES):
        return outcome.value > high + pad
    return outcome.value < low - pad


def score_one(protocol: Protocol, outcome: Outcome, run: Run) -> Evidence:
    """One protocol's contribution, with its controls checked first."""
    prediction = run.registration.for_protocol(protocol.id)
    if prediction is None:
        return Evidence(
            protocol=protocol.id,
            log_lr=0.0,
            observed=f"{outcome.measure}={outcome.value:.4f}",
            predicted="",
            controls_held=False,
            control_note=(
                "no pre-registered prediction for this protocol; it was run "
                "without a stated losing condition"
            ),
        )

    usable, why = protocol.usable(outcome)
    if not usable:
        return Evidence(
            protocol=protocol.id,
            log_lr=0.0,
            observed=f"{outcome.measure}={outcome.value:.4f}",
            predicted=prediction.direction,
            controls_held=False,
            control_note=why,
        )

    held = _direction_holds(prediction.direction, outcome)

    # How big the result is, measured in the terms the prediction was made in.
    #
    # For a prediction that something RISES or FALLS, the effect size is the
    # distance from the null. For one that VANISHES it is the opposite: the
    # value SHOULD sit at the null, and the size of the finding is how far it
    # dropped to get there. Measuring distance-from-null in that case scored a
    # perfect lesion as a null result, because a perfect lesion is exactly
    # zero away from the null.
    vanishes = any(word in prediction.direction.lower() for word in _VANISHES)
    if vanishes and outcome.claim is not None and outcome.claim.baseline is not None:
        magnitude = abs(outcome.claim.baseline.value - outcome.value)
    else:
        magnitude = abs(outcome.value - (sum(outcome.nulls) / len(outcome.nulls)))
    if held and magnitude < prediction.minimum_effect:
        return Evidence(
            protocol=protocol.id,
            log_lr=0.0,
            observed=f"{outcome.measure}={outcome.value:.4f}",
            predicted=prediction.direction,
            controls_held=True,
            control_note=(
                f"moved the registered way but by {magnitude:.4f}, under the "
                f"registered minimum of {prediction.minimum_effect}"
            ),
        )

    # A causal claim carries its own weight from the ladder; a bare directional
    # result is worth much less, and the difference is the point.
    if outcome.claim is not None:
        weight = log_likelihood_ratio(outcome.claim)
    else:
        weight = 0.693  # log 2: a single directional result is one bit at most

    return Evidence(
        protocol=protocol.id,
        log_lr=weight if held else -weight,
        observed=f"{outcome.measure}={outcome.value:.4f}",
        predicted=prediction.direction,
        controls_held=True,
        detail={
            "magnitude": round(magnitude, 4),
            "minimum_effect": prediction.minimum_effect,
            "report_free": protocol.report_free,
            "claim": outcome.claim.as_dict() if outcome.claim else None,
        },
    )


def score(run: Run) -> tuple[Adjudication, list[Evidence]]:
    """Everything the run earns, and nothing it does not."""
    void = run.void_reason()
    evidence: list[Evidence] = []
    for outcome in run.outcomes:
        protocol = by_id(outcome.protocol)
        if protocol is None:
            evidence.append(
                Evidence(
                    protocol=outcome.protocol,
                    log_lr=0.0,
                    observed="",
                    predicted="",
                    controls_held=False,
                    control_note="not a declared protocol in this battery",
                )
            )
            continue
        evidence.append(score_one(protocol, outcome, run))

    # A protocol the counterfeit also passes tells us about the protocol, not
    # about Aura, so it is dropped from the adjudication and named.
    if run.counterfeit is not None:
        weak = {
            separation.protocol
            for separation in run.counterfeit.separations
            if not separation.discriminates
        }
        evidence = [
            item
            if item.protocol not in weak
            else Evidence(
                protocol=item.protocol,
                log_lr=0.0,
                observed=item.observed,
                predicted=item.predicted,
                controls_held=False,
                control_note=(
                    "the adversarial counterfeit passed this too, so it "
                    "discriminates nothing"
                ),
            )
            for item in evidence
        ]

    return adjudicate(evidence, void_reason=void), evidence


def report(run: Run, path: Path | str | None = None) -> dict[str, Any]:
    """The document. Says what was shown, and says what was not."""
    verdict, evidence = score(run)
    document = {
        "schema": "aura.phenomenology.gauntlet.v1",
        "operator": run.operator,
        "model_digest": run.model_digest,
        "source_commit": run.source_commit,
        "replication_of": run.replication_of,
        "preregistration_digest": run.registration.digest(),
        "published_digest": run.published_digest,
        "hypotheses": {
            COSTUME.id: COSTUME.statement,
            LOAD_BEARING.id: LOAD_BEARING.statement,
        },
        "adjudication": verdict.as_dict(),
        "evidence": [
            {
                "protocol": item.protocol,
                "log_lr": round(item.log_lr, 4),
                "counted": item.counts,
                "observed": item.observed,
                "predicted": item.predicted,
                "control_note": item.control_note,
                "detail": item.detail,
            }
            for item in evidence
        ],
        "counterfeit": run.counterfeit.as_dict() if run.counterfeit else {
            "built": False,
            "consequence": (
                "no adversarial control was run, so every protocol here is "
                "compared against nothing that tried to beat it"
            ),
        },
        "not_addressed": {
            "question": PHENOMENAL.statement,
            "verdict": str(Verdict.NOT_ADDRESSED),
            "because": PHENOMENAL.undecidable_because,
        },
        "protocols_declared": len(BATTERY),
        "protocols_attempted": len(run.outcomes),
    }
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return document
