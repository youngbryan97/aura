"""L3d — fitting from a life: time-ordered, recency-weighted, and honestly held out.

Three decisions in this file do most of the work of not fooling ourselves.

**The holdout is the future, never a random sample.** Shuffling episodes and
holding out a random tenth is the standard way to produce an impressive number
that does not survive contact with next week. Aura's world is non-stationary by
construction — her own code changes daily, and a parallel agent rewrites
subsystems underneath her — so the only honest question is "fitted on what I
had lived by Tuesday, how did it do on Wednesday?" The split is by time, and
the holdout is always the most recent slice.

**Recency is weighted, not truncated.** Old episodes still carry signal, just
less of it. An exponential half-life keeps the long tail contributing without
letting a subsystem that was retired in May outvote this week.

**The reservoir state is replayed, not stored.** A head is fitted on
[features, presence, hidden state], and the hidden state at episode *t* is not
in the database — storing ninety-six floats per row would multiply the corpus
for no reason. Because the reservoir's weights are fixed and seeded, the state
sequence is exactly reconstructible by replaying the episodes in order. The
first steps of a replay are discarded as washout, since a reservoir started
from zero needs a little while before its state means anything.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.ontogeny.authority import AuthorityLedger
from core.ontogeny.calibration import (
    CANDIDATE_VALIDATION,
    CalibrationMonitor,
    CalibrationObservation,
)
from core.ontogeny.experience import Episode, ExperienceSpine, OutcomeKind
from core.ontogeny.features import FeatureSchema, RunningMoments, design_row, row_names
from core.ontogeny.heads import MIN_FIT_SAMPLES, PredictionHead
from core.ontogeny.state import OntogeneticState

logger = logging.getLogger("Aura.Ontogeny.Trainer")

#: Disjoint future cohorts. Temperature fitting must not see the episodes used
#: to estimate candidate lift or calibration; doing both on one holdout reports
#: in-sample confidence as if it were independent evidence.
TEMPERATURE_FRACTION = 0.10
EVALUATION_FRACTION = 0.15
HOLDOUT_FRACTION = TEMPERATURE_FRACTION + EVALUATION_FRACTION

#: Recency half-life. An episode from a fortnight ago counts half as much as
#: one from today.
RECENCY_HALF_LIFE_S = 14 * 86400.0

#: Reservoir steps discarded at the start of a replay.
WASHOUT_STEPS = 50


class TrainingPreempted(RuntimeError):
    """A foreground turn interrupted optional ontogeny training."""


@dataclass
class TrainingResult:
    """Everything a promotion decision is allowed to rest on."""

    control_point: str
    fitted: bool
    reason: str = ""
    samples: int = 0
    temperature_samples: int = 0
    holdout_samples: int = 0
    holdout_accuracy: float | None = None
    holdout_base_rate: float | None = None
    #: Accuracy of always predicting the most common outcome. A head that
    #: cannot beat this has learned nothing, however good its raw score looks.
    lift: float | None = None
    temperature: float = 1.0
    fit_evidence: dict[str, Any] = field(default_factory=dict)
    authority: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_point": self.control_point,
            "fitted": self.fitted,
            "reason": self.reason,
            "samples": self.samples,
            "temperature_samples": self.temperature_samples,
            "holdout_samples": self.holdout_samples,
            "holdout_accuracy": round(self.holdout_accuracy, 4) if self.holdout_accuracy is not None else None,
            "holdout_base_rate": round(self.holdout_base_rate, 4) if self.holdout_base_rate is not None else None,
            "lift": round(self.lift, 4) if self.lift is not None else None,
            "temperature": round(self.temperature, 3),
            "fit_evidence": dict(self.fit_evidence),
            "authority": dict(self.authority),
        }


def replay_design(
    episodes: Sequence[Episode],
    schema: FeatureSchema,
    *,
    units: int,
    seed: int,
    washout: int = WASHOUT_STEPS,
    should_stop: Callable[[], bool] | None = None,
    cooperate: Callable[[], None] | None = None,
) -> tuple[np.ndarray, list[Episode], RunningMoments]:
    """Rebuild the exact [features, presence, hidden] rows the heads see.

    A fresh reservoir with the same seed and the same episode order produces
    the same states the live organ produced. That reproducibility is why the
    state is not stored: it is a pure function of the corpus.

    The action is not in the row. Each action has its own head, fitted only on
    the episodes where that action was taken.
    """
    moments = RunningMoments(len(schema.names))
    replay_state = OntogeneticState(input_width=schema.width, units=units, seed=seed)
    rows: list[np.ndarray] = []
    kept: list[Episode] = []
    for i, episode in enumerate(episodes):
        if i % 64 == 0:
            if should_stop is not None and should_stop():
                raise TrainingPreempted("foreground_preempted")
            if cooperate is not None:
                cooperate()
        vector = schema.vector(episode.features)
        base = design_row(vector, moments, update=True)
        reading = replay_state.step(base, learn_distribution=True)
        if i < washout:
            continue
        rows.append(np.concatenate([base, reading.hidden]))
        kept.append(episode)
    if not rows:
        return np.zeros((0, design_width(schema, units))), [], moments
    return np.vstack(rows), kept, moments


def design_width(schema: FeatureSchema, units: int) -> int:
    return schema.width + units


def design_names(schema: FeatureSchema, units: int) -> tuple[str, ...]:
    return (*row_names(schema), *(f"state[{i}]" for i in range(units)))


class Trainer:
    """Fits heads from the corpus and asks the authority ledger for a verdict."""

    def __init__(
        self,
        spine: ExperienceSpine,
        authority: AuthorityLedger,
        calibration: CalibrationMonitor,
        *,
        units: int,
        seed: int,
    ) -> None:
        self._spine = spine
        self._authority = authority
        self._calibration = calibration
        self._units = int(units)
        self._seed = int(seed)
        self.last_result: dict[str, TrainingResult] = {}

    def train(
        self,
        control_point: str,
        schema: FeatureSchema,
        heads: Mapping[str, PredictionHead],
        actions: Sequence[str],
        *,
        limit: int = 50_000,
        should_stop: Callable[[], bool] | None = None,
        cooperate: Callable[[], None] | None = None,
    ) -> TrainingResult:
        """One training pass over every action's head.

        Nothing running is modified until a fit succeeds. A failed pass leaves
        the live organ exactly as it was, which is the only safe behaviour for
        something that may be holding a decision.
        """
        if should_stop is not None and should_stop():
            return TrainingResult(
                control_point=control_point,
                fitted=False,
                reason="foreground_preempted",
            )
        episodes = self._spine.episodes(
            control_point,
            evidence_only=True,
            limit=limit,
            feature_schema=schema.schema_id,
        )
        # The spine returns newest-first; a replay needs lived order.
        episodes = list(reversed(episodes))
        if len(episodes) < MIN_FIT_SAMPLES + WASHOUT_STEPS:
            result = TrainingResult(
                control_point=control_point, fitted=False,
                reason=f"{len(episodes)} graded episodes; need {MIN_FIT_SAMPLES + WASHOUT_STEPS}",
                samples=len(episodes),
            )
            self.last_result[control_point] = result
            self._authority.evaluate(control_point, episodes, head_ready=False)
            return result

        try:
            rows, kept, _ = replay_design(
                episodes,
                schema,
                units=self._units,
                seed=self._seed,
                should_stop=should_stop,
                cooperate=cooperate,
            )
        except TrainingPreempted:
            return TrainingResult(
                control_point=control_point,
                fitted=False,
                reason="foreground_preempted",
                samples=len(episodes),
            )
        if rows.shape[0] < MIN_FIT_SAMPLES:
            result = TrainingResult(
                control_point=control_point, fitted=False,
                reason="not enough rows survived washout", samples=int(rows.shape[0]),
            )
            self.last_result[control_point] = result
            return result

        labels = [_label_of(ep) for ep in kept]
        train_end = max(1, int(len(kept) * (1.0 - HOLDOUT_FRACTION)))
        temperature_end = max(train_end, int(len(kept) * (1.0 - EVALUATION_FRACTION)))
        temperature_end = min(len(kept), temperature_end)
        now = time.time()
        per_action: dict[str, dict[str, Any]] = {}
        fitted_any = False
        total_train = total_temperature = total_holdout = 0
        weighted_accuracy = weighted_base = 0.0
        weighted_temperature = 0.0
        candidate_observations: list[CalibrationObservation] = []
        candidate_updates: dict[str, PredictionHead] = {}
        candidate_generation = max((head.version for head in heads.values()), default=0) + 1

        for action in actions:
            if cooperate is not None:
                cooperate()
            if should_stop is not None and should_stop():
                return TrainingResult(
                    control_point=control_point,
                    fitted=False,
                    reason="foreground_preempted",
                    samples=len(kept),
                )
            head = heads.get(action)
            if head is None:
                continue
            taken = [i for i, ep in enumerate(kept) if ep.decision == action]
            train_idx = [i for i in taken if i < train_end]
            temperature_idx = [i for i in taken if train_end <= i < temperature_end]
            holdout_idx = [i for i in taken if i >= temperature_end]
            if len(train_idx) < MIN_FIT_SAMPLES:
                per_action[action] = {
                    "fitted": False,
                    "reason": f"{len(train_idx)} episodes chose this action; need {MIN_FIT_SAMPLES}",
                    "train_samples": len(train_idx),
                }
                continue

            train_rows = rows[train_idx]
            train_labels = [labels[i] for i in train_idx]
            weights = [
                math.pow(0.5, (now - kept[i].decided_at) / RECENCY_HALF_LIFE_S)
                * max(1, kept[i].repeat_count)
                for i in train_idx
            ]
            candidate = PredictionHead(
                control_point=head.control_point,
                options=head.options,
                input_width=rows.shape[1],
                input_names=design_names(schema, self._units),
                learning_rate=head.learning_rate,
                l2=head.l2,
            )
            # A fresh candidate starts at version zero. Carry the deployed
            # generation forward so every successful refit creates a distinct
            # head cohort instead of repeatedly calling itself version one.
            candidate.version = candidate_generation - 1
            evidence = candidate.fit(
                train_rows,
                train_labels,
                weights=weights,
                should_stop=should_stop,
                cooperate=cooperate,
            )
            if not evidence.get("fitted"):
                if evidence.get("reason") == "foreground_preempted":
                    return TrainingResult(
                        control_point=control_point,
                        fitted=False,
                        reason="foreground_preempted",
                        samples=len(kept),
                    )
                per_action[action] = {"fitted": False, "reason": evidence.get("reason", "fit refused")}
                continue

            temperature_rows = (
                rows[temperature_idx]
                if temperature_idx else np.zeros((0, rows.shape[1]))
            )
            temperature_labels = [labels[i] for i in temperature_idx]
            holdout_rows = rows[holdout_idx] if holdout_idx else np.zeros((0, rows.shape[1]))
            holdout_labels = [labels[i] for i in holdout_idx]
            temperature = candidate.calibrate(temperature_rows, temperature_labels)
            accuracy, base_rate = _score(candidate, holdout_rows, holdout_labels)

            candidate_updates[action] = candidate
            fitted_any = True
            total_train += len(train_idx)
            total_temperature += len(temperature_idx)
            total_holdout += len(holdout_idx)
            weighted_temperature += temperature * len(temperature_idx)
            if accuracy is not None and base_rate is not None:
                weighted_accuracy += accuracy * len(holdout_idx)
                weighted_base += base_rate * len(holdout_idx)

            for i in holdout_idx:
                prediction = candidate.predict(rows[i])
                episode = kept[i]
                candidate_observations.append(CalibrationObservation(
                    episode_id=f"candidate:{candidate.version}:{episode.episode_id}",
                    control_point=control_point,
                    confidence=prediction.confidence,
                    correct=prediction.choice == labels[i],
                    decided_at=episode.decided_at,
                    observed_at=now,
                    runtime_revision=str(
                        (episode.context or {}).get("runtime_revision") or "training-corpus"
                    ),
                    head_version=candidate.version,
                    action=action,
                    provenance=CANDIDATE_VALIDATION,
                ))
            per_action[action] = {
                "fitted": True,
                "train_samples": len(train_idx),
                "temperature_samples": len(temperature_idx),
                "holdout_samples": len(holdout_idx),
                "holdout_accuracy": round(accuracy, 4) if accuracy is not None else None,
                "holdout_base_rate": round(base_rate, 4) if base_rate is not None else None,
                "temperature": round(temperature, 3),
                "train_accuracy": evidence.get("train_accuracy"),
                "version": candidate.version,
            }

        if not fitted_any:
            result = TrainingResult(
                control_point=control_point, fitted=False,
                reason="no action had enough evidence for its own head",
                samples=len(kept), fit_evidence={"per_action": per_action},
            )
            self.last_result[control_point] = result
            self._authority.evaluate(control_point, kept, head_ready=False)
            return result

        if should_stop is not None and should_stop():
            return TrainingResult(
                control_point=control_point,
                fitted=False,
                reason="foreground_preempted",
                samples=len(kept),
            )
        for action, candidate in candidate_updates.items():
            head = heads[action]
            head.load_state(candidate.state_dict())
            head.input_names = candidate.input_names

        # Candidate validation is a frozen evaluation plane. Replace it
        # atomically so a repeated fit cannot append duplicate observations or
        # retain a previous model's verdict when the new cohort has no support.
        self._calibration.replace_observations(
            control_point,
            candidate_observations,
            provenance=CANDIDATE_VALIDATION,
        )
        accuracy = weighted_accuracy / total_holdout if total_holdout else None
        base_rate = weighted_base / total_holdout if total_holdout else None
        ready = sum(1 for a in actions if heads.get(a) is not None and heads[a].ready) >= 2
        verdict = self._authority.evaluate(
            control_point, kept, head_ready=ready, holdout_accuracy=accuracy
        )
        result = TrainingResult(
            control_point=control_point,
            fitted=True,
            samples=total_train,
            temperature_samples=total_temperature,
            holdout_samples=total_holdout,
            holdout_accuracy=accuracy,
            holdout_base_rate=base_rate,
            lift=(accuracy - base_rate) if accuracy is not None and base_rate is not None else None,
            temperature=(weighted_temperature / total_temperature if total_temperature else 1.0),
            fit_evidence={
                "per_action": per_action,
                "corpus": len(kept),
                "temporal_cohorts": {
                    "training": _cohort_receipt(kept[:train_end]),
                    "temperature": _cohort_receipt(kept[train_end:temperature_end]),
                    "evaluation": _cohort_receipt(kept[temperature_end:]),
                },
            },
            authority=verdict,
        )
        self.last_result[control_point] = result
        logger.info(
            "ontogeny: fitted %s across %d actions on %d episodes — holdout %.3f vs base %.3f (%s)",
            control_point, sum(1 for v in per_action.values() if v.get("fitted")),
            total_train, accuracy or 0.0, base_rate or 0.0, verdict.get("action", "hold"),
        )
        return result


def _label_of(episode: Episode) -> str:
    """What the head is asked to predict.

    Not the decision that was taken — a head fitted on that learns to imitate
    the incumbent, which is worth nothing. It predicts whether the episode
    *went well*, which is the thing that can be better than the incumbent.
    """
    outcome = episode.outcome
    if outcome is None:
        return "unobserved"
    return "success" if outcome.kind is OutcomeKind.SUCCESS else "failure"


def _cohort_receipt(episodes: Sequence[Episode]) -> dict[str, Any]:
    """Auditable temporal extent without copying episode payloads into reports."""
    if not episodes:
        return {"samples": 0, "first_decided_at": None, "last_decided_at": None}
    return {
        "samples": len(episodes),
        "first_decided_at": episodes[0].decided_at,
        "last_decided_at": episodes[-1].decided_at,
        "first_episode_id": episodes[0].episode_id,
        "last_episode_id": episodes[-1].episode_id,
    }


def _score(
    head: PredictionHead, rows: np.ndarray, labels: Sequence[str]
) -> tuple[float | None, float | None]:
    if rows.shape[0] == 0 or not labels:
        return None, None
    correct = sum(1 for row, label in zip(rows, labels, strict=True) if head.predict(row).choice == label)
    accuracy = correct / len(labels)
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    base_rate = max(counts.values()) / len(labels)
    return accuracy, base_rate


__all__ = [
    "EVALUATION_FRACTION",
    "HOLDOUT_FRACTION",
    "RECENCY_HALF_LIFE_S",
    "TEMPERATURE_FRACTION",
    "WASHOUT_STEPS",
    "Trainer",
    "TrainingResult",
    "design_names",
    "design_width",
    "replay_design",
]


# The per-action design deliberately has no action column: see ``ControlPoint``
# in service.py for why one model with a one-hot action cannot represent the
# problem at all.
