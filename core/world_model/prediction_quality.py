"""core/world_model/prediction_quality.py — is the world model any good, and at what.

``learned_world_model.py`` reports ``confidence = 1 - surprise``. That is a
restatement of the error, not a belief about the error, and it cannot be wrong:
whatever the model predicts, its confidence agrees with how well it predicted,
after the fact. A confidence that cannot be wrong cannot inform a decision.

Four things this adds, all of them the model asking questions about itself that
the current one cannot:

* **Calibration.** Bin predicted confidence against measured transition error
  and plot the reliability. A model that says 0.9 and is right 0.6 of the time
  is overconfident by 0.3, and that number is what an epistemic action should
  be triggered on.
* **An ensemble.** Disagreement between independently initialised predictors is
  epistemic uncertainty - the model not knowing - which is a different thing
  from a wide posterior, which is the world being noisy. Conflating them is why
  "the model is uncertain" has never been an actionable signal.
* **Latent versus reconstruction.** The current objective reconstructs the
  observation vector, so capacity goes into predicting pixels-equivalents that
  do not matter. :class:`ObjectiveComparison` runs both and scores them on
  downstream control rather than on their own loss, which is the only
  comparison that settles it.
* **Timescale.** :class:`MultiTimescalePrediction` asks the model to predict
  one step, ten steps and a hundred, and reports where it stops being useful.
  A planner that knows the horizon can pick a level; one that does not will
  roll out flat until the prediction is noise.

Counterfactual structure
------------------------
:class:`InterventionLedger` is the difference between prediction and
understanding. A model that predicts the next state well can still be
correlational: it has learned that A and B co-occur without learning that
changing A changes B. Doing the intervention and comparing the prediction to
what happened is the test, and a model that fails it is a good predictor and a
bad world model.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PredictionOutcome",
    "CalibrationCurve",
    "EnsembleDisagreement",
    "ensemble_disagreement",
    "ObjectiveComparison",
    "MultiTimescalePrediction",
    "InterventionLedger",
]

BINS = 10


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    """One prediction, its stated confidence, and how wrong it turned out."""

    confidence: float
    error: float
    #: Error below which the prediction counts as correct, in the caller's unit.
    tolerance: float = 0.1

    @property
    def correct(self) -> bool:
        return self.error <= self.tolerance


@dataclass
class CalibrationCurve:
    """Predicted confidence against measured correctness."""

    counts: list[int] = field(default_factory=lambda: [0] * BINS)
    hits: list[int] = field(default_factory=lambda: [0] * BINS)
    squared: float = 0.0
    n: int = 0
    confidence_sum: float = 0.0
    correct_sum: float = 0.0

    def observe(self, outcome: PredictionOutcome) -> None:
        index = min(BINS - 1, max(0, int(outcome.confidence * BINS)))
        self.counts[index] += 1
        self.hits[index] += 1 if outcome.correct else 0
        self.squared += (outcome.confidence - (1.0 if outcome.correct else 0.0)) ** 2
        self.confidence_sum += outcome.confidence
        self.correct_sum += 1.0 if outcome.correct else 0.0
        self.n += 1

    @property
    def brier(self) -> float | None:
        return self.squared / self.n if self.n else None

    @property
    def overconfidence(self) -> float | None:
        """Mean stated confidence minus measured accuracy. Positive is bluffing."""
        if not self.n:
            return None
        return (self.confidence_sum - self.correct_sum) / self.n

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "brier": self.brier,
            "overconfidence": self.overconfidence,
            "calibrated": self.n >= 30 and abs(self.overconfidence or 1.0) < 0.1,
            "diagram": [
                {
                    "bin": i,
                    "stated": (i + 0.5) / BINS,
                    "measured": (self.hits[i] / self.counts[i]) if self.counts[i] else None,
                    "n": self.counts[i],
                }
                for i in range(BINS)
            ],
        }


@dataclass(frozen=True, slots=True)
class EnsembleDisagreement:
    """What the members disagree about, kept apart from what the world does.

    ``epistemic`` is spread between independently initialised predictors: the
    model not knowing, which more data would fix. ``aleatoric`` is the mean
    within-member variance: the world being noisy, which more data would not.
    Only the first is a reason to go and look.
    """

    epistemic: float
    aleatoric: float

    @property
    def worth_investigating(self) -> bool:
        return self.epistemic > self.aleatoric

    def to_dict(self) -> dict[str, Any]:
        return {
            "epistemic": self.epistemic,
            "aleatoric": self.aleatoric,
            "worth_investigating": self.worth_investigating,
            "reading": (
                "the model does not know; an observation would help"
                if self.worth_investigating
                else "the world is noisy here; more looking will not help"
            ),
        }


def ensemble_disagreement(
    member_means: Sequence[float], member_variances: Sequence[float]
) -> EnsembleDisagreement:
    """Split predictive uncertainty into what data would fix and what it would not."""
    if not member_means:
        return EnsembleDisagreement(0.0, 0.0)
    mean = sum(member_means) / len(member_means)
    epistemic = sum((m - mean) ** 2 for m in member_means) / len(member_means)
    aleatoric = sum(member_variances) / len(member_variances) if member_variances else 0.0
    return EnsembleDisagreement(epistemic=epistemic, aleatoric=aleatoric)


@dataclass(frozen=True, slots=True)
class ObjectiveComparison:
    """Reconstruction against latent prediction, scored on control not on loss."""

    reconstruction_loss: float
    latent_loss: float
    reconstruction_control_success: float
    latent_control_success: float
    compute_matched: bool

    @property
    def verdict(self) -> str:
        if not self.compute_matched:
            return "void: the two objectives were not given the same compute"
        if self.latent_control_success > self.reconstruction_control_success:
            return "latent prediction plans better"
        if self.latent_control_success < self.reconstruction_control_success:
            return "reconstruction plans better"
        return "no difference in planning"

    @property
    def loss_disagrees_with_control(self) -> bool:
        """Whether the objective with the better loss is the worse planner.

        This is the case the comparison exists for. A reconstruction loss can
        win while the latent objective plans better, because the reconstruction
        is spending capacity on detail that control does not use.
        """
        loss_winner_is_latent = self.latent_loss < self.reconstruction_loss
        control_winner_is_latent = self.latent_control_success > self.reconstruction_control_success
        return loss_winner_is_latent != control_winner_is_latent

    def to_dict(self) -> dict[str, Any]:
        return {
            "reconstruction_loss": self.reconstruction_loss,
            "latent_loss": self.latent_loss,
            "reconstruction_control_success": self.reconstruction_control_success,
            "latent_control_success": self.latent_control_success,
            "compute_matched": self.compute_matched,
            "verdict": self.verdict,
            "loss_disagrees_with_control": self.loss_disagrees_with_control,
        }


class MultiTimescalePrediction:
    """Where the model stops being useful, by horizon."""

    def __init__(self, *, horizons: Sequence[int] = (1, 10, 100)) -> None:
        self._lock = threading.RLock()
        self._errors: dict[int, list[float]] = {h: [] for h in horizons}
        self._baseline: dict[int, list[float]] = {h: [] for h in horizons}

    def observe(self, horizon: int, *, error: float, baseline_error: float) -> None:
        """Record an error at a horizon, beside what predicting nothing costs."""
        with self._lock:
            self._errors.setdefault(horizon, []).append(float(error))
            self._baseline.setdefault(horizon, []).append(float(baseline_error))

    def useful_horizon(self) -> dict[str, Any]:
        """The longest horizon at which the model still beats the baseline."""
        with self._lock:
            rows = []
            for horizon in sorted(self._errors):
                errors, baseline = self._errors[horizon], self._baseline[horizon]
                if not errors or not baseline:
                    continue
                mean_error = sum(errors) / len(errors)
                mean_baseline = sum(baseline) / len(baseline)
                rows.append(
                    {
                        "horizon": horizon,
                        "error": mean_error,
                        "baseline": mean_baseline,
                        "beats_baseline": mean_error < mean_baseline,
                        "n": len(errors),
                    }
                )
        useful = [r["horizon"] for r in rows if r["beats_baseline"]]
        return {
            "by_horizon": rows,
            "useful_to": max(useful) if useful else None,
            "flat_rollout_is_enough_beyond": max(useful) if useful else 0,
        }


@dataclass
class InterventionLedger:
    """Predictions under intervention, which is what separates a model from a fit."""

    predictions: list[tuple[str, float, float]] = field(default_factory=list)
    observational_predictions: list[tuple[str, float, float]] = field(default_factory=list)

    def record_intervention(self, variable: str, predicted: float, observed: float) -> None:
        self.predictions.append((variable, predicted, observed))

    def record_observation(self, variable: str, predicted: float, observed: float) -> None:
        self.observational_predictions.append((variable, predicted, observed))

    @staticmethod
    def _rmse(rows: Sequence[tuple[str, float, float]]) -> float | None:
        if not rows:
            return None
        return math.sqrt(sum((p - o) ** 2 for _, p, o in rows) / len(rows))

    def verdict(self) -> dict[str, Any]:
        """Whether the model survives being intervened on.

        A model that predicts observations well and interventions badly has
        learned what goes with what. That is a good predictor and a bad world
        model, and the two are indistinguishable until somebody intervenes.
        """
        interventional = self._rmse(self.predictions)
        observational = self._rmse(self.observational_predictions)
        if interventional is None or observational is None:
            return {"measurable": False, "reason": "both arms are needed"}
        return {
            "measurable": True,
            "interventional_rmse": interventional,
            "observational_rmse": observational,
            "ratio": interventional / observational if observational else None,
            "causal": interventional <= observational * 1.5 + 1e-9,
            "reading": (
                "predicts interventions about as well as observations; the structure "
                "is doing work"
                if interventional <= observational * 1.5 + 1e-9
                else "predicts observations and not interventions; it has learned what "
                "goes with what"
            ),
            "worst_variable": max(
                ((v, abs(p - o)) for v, p, o in self.predictions),
                key=lambda pair: pair[1],
                default=None,
            ),
        }
