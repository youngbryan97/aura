"""core/conation/calibration.py — the parameters, and who is allowed to set them.

Every free parameter in this package is declared in one place and defaults to
the assumption-free setting rather than a tuned one. That is the whole policy,
and it exists because a motivational model is unusually easy to make say
whatever the author expected: pick weights until the snail scores well, and
the model has learned the author's intuition rather than anything about Aura.

Two states are possible and they are reported differently.

**Declared.** Nothing has been measured on this system, so the defaults stand
and the readout says ``learned=False`` with the reason. A caller reading a
declared calibration knows it is reading an assumption.

**Learned.** A head in ``core/ontogeny`` has earned authority over these
values against a counterfactual slice, and the readout names it. Until such a
head exists this branch returns nothing and the declared values stand.

The distinction matters more than the values. A system that cannot tell a
measured weight from a chosen one will eventually report a chosen one as
though it had been measured, which is the failure that makes every number
above it unfalsifiable.

## What would grade a weight vector

The objective is already in the package: ``OutcomeReport.epsilon_liking``.
A set of arbitration weights is good if the choices made under it lead to
outcomes that turned out to be liked, and the hedonic prediction error is
exactly that measurement. It is recorded here per weight source so that a
head has something to be trained against when one is built, and so that the
claim "these weights are better" can be checked rather than asserted.

No head is registered today. Recording the evidence before there is a learner
is deliberate — the alternative is a learner with no history to learn from on
the day it arrives.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.conation.salience import SalienceCalibration
from core.conation.state import VECTOR_FIELDS
from core.runtime.errors import record_degradation

#: Ontogeny control point this package would use. Named here so the absence is
#: greppable: a search for it finds this constant and this docstring rather
#: than nothing at all.
CONTROL_POINT = "conation_arbitration_weights"


@dataclass
class WeightEvidence:
    """Outcomes observed under one weight source, for a future learner."""

    source: str
    errors: deque[float] = field(default_factory=lambda: deque(maxlen=256))
    choices: int = 0
    last_update: float = field(default_factory=time.time)

    #: Outcomes needed before the mean error means anything. Ten is the
    #: smallest window over which a hedonic average stops tracking single
    #: results, matching the delta-rule time constant in the salience model.
    MIN_OUTCOMES = 10

    def observe(self, epsilon_liking: float | None) -> None:
        self.choices += 1
        if epsilon_liking is not None:
            self.errors.append(float(epsilon_liking))
        self.last_update = time.time()

    def mean_error(self) -> float | None:
        """Average hedonic surprise under this weight source.

        Negative means choices made under these weights kept turning out worse
        than predicted, which is the measurable sense in which a weight vector
        can be wrong.
        """
        if len(self.errors) < self.MIN_OUTCOMES:
            return None
        return sum(self.errors) / len(self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "choices": self.choices,
            "outcomes": len(self.errors),
            "mean_error": self.mean_error(),
        }


def declared_salience() -> SalienceCalibration:
    """The assumption-free salience parameters, marked as unmeasured."""
    return SalienceCalibration(learned=False, source="declared_default")


def declared_weights() -> dict[str, float]:
    """Uniform arbitration weights: the refusal to invent an ordering.

    Equal weighting is the maximum-entropy choice given no data. Any other
    vector asserts that one motivational channel matters more than another on
    this system, which nothing here has observed.
    """
    return {name: 1.0 for name in VECTOR_FIELDS}


def learned_weights() -> tuple[dict[str, float], str] | None:
    """Weights from a promoted ontogenetic head, or ``None``.

    Returns ``None`` whenever no head holds authority over this control point,
    which is the case today. The lookup is written rather than stubbed so that
    the day a head is promoted, the only change needed is the promotion.
    """
    try:
        from core.ontogeny.service import get_ontogeny

        organ = get_ontogeny()
        resolve = getattr(organ, "resolve", None)
        if not callable(resolve):
            return None
        decision = resolve(CONTROL_POINT)
        if decision is None:
            return None
        weights = getattr(decision, "action", None) or getattr(decision, "value", None)
        if not isinstance(weights, dict):
            return None
        if set(weights) != set(VECTOR_FIELDS):
            return None
        stage = getattr(decision, "stage", "unknown")
        return (
            {name: float(weights[name]) for name in VECTOR_FIELDS},
            f"ontogeny/{CONTROL_POINT}@{stage}",
        )
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, LookupError):
        # A missing control point is the expected case, not a fault. Recording
        # a degradation on every appraisal for a head nobody has built would
        # bury the records that matter.
        return None


class CalibrationRegistry:
    """Holds which parameters are in force, and the evidence under each."""

    def __init__(self) -> None:
        self._evidence: dict[str, WeightEvidence] = {}
        self._source = "declared_default"

    @property
    def source(self) -> str:
        return self._source

    @property
    def learned(self) -> bool:
        return self._source != "declared_default"

    def resolve_weights(self) -> tuple[dict[str, float], str]:
        """Weights in force now, and where they came from."""
        promoted = learned_weights()
        if promoted is not None:
            weights, source = promoted
            self._source = source
            return weights, source
        self._source = "declared_default"
        return declared_weights(), "declared_default"

    def observe_outcome(self, epsilon_liking: float | None) -> None:
        """Record one outcome under whatever weights were in force."""
        evidence = self._evidence.get(self._source)
        if evidence is None:
            evidence = WeightEvidence(source=self._source)
            self._evidence[self._source] = evidence
        evidence.observe(epsilon_liking)

    def status(self) -> dict[str, Any]:
        return {
            "control_point": CONTROL_POINT,
            "source": self._source,
            "learned": self.learned,
            "note": (
                None if self.learned
                else "no ontogenetic head holds authority; declared defaults in force"
            ),
            "evidence": [e.to_dict() for e in self._evidence.values()],
        }
