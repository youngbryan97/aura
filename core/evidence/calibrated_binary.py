"""A small, auditable correctness scorer trained only on verified outcomes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

CALIBRATED_BINARY_SCORER_SCHEMA: Final = "aura.evidence.calibrated_binary_scorer.v1"
CALIBRATED_BINARY_FIT_RECEIPT_SCHEMA: Final = (
    "aura.evidence.calibrated_binary_fit_receipt.v1"
)
MIN_FIT_OBSERVATIONS: Final = 24
MIN_CALIBRATION_OBSERVATIONS: Final = 24
MAX_ADMITTED_BRIER: Final = 0.22


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _parameter_sha(
    feature_names: Sequence[str],
    means: Sequence[float],
    scales: Sequence[float],
    weights: Sequence[float],
) -> str:
    return _sha(
        {
            "feature_names": list(feature_names),
            "means": list(means),
            "scales": list(scales),
            "weights": list(weights),
        }
    )


def _finite(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class VerifiedBinaryObservation:
    """One feature vector whose outcome was established independently."""

    values: tuple[tuple[str, float], ...]
    verified_correct: bool
    source_ref: str

    def __post_init__(self) -> None:
        names = tuple(name for name, _value in self.values)
        if (
            not self.values
            or names != tuple(sorted(names))
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for _name, value in self.values
            )
            or type(self.verified_correct) is not bool
            or not isinstance(self.source_ref, str)
            or not self.source_ref
        ):
            raise ValueError("verified binary observation is invalid")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, float],
        *,
        verified_correct: bool,
        source_ref: str,
    ) -> VerifiedBinaryObservation:
        if not isinstance(values, Mapping):
            raise ValueError("verified binary features must be a mapping")
        if any(not isinstance(name, str) or not name for name in values):
            raise ValueError("verified binary feature names must be non-empty strings")
        normalized = tuple(
            sorted(
                (
                    name,
                    _finite(value, field=f"feature {name!r}"),
                )
                for name, value in values.items()
            )
        )
        return cls(
            values=normalized,
            verified_correct=verified_correct,
            source_ref=source_ref,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _value in self.values)

    def vector(self) -> tuple[float, ...]:
        return tuple(float(value) for _name, value in self.values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": [list(row) for row in self.values],
            "verified_correct": self.verified_correct,
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True, slots=True)
class CalibratedBinaryScorer:
    """Frozen logistic scorer admitted on independent calibration outcomes."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    fit_receipt: dict[str, Any]
    schema: str = CALIBRATED_BINARY_SCORER_SCHEMA

    def __post_init__(self) -> None:
        body = {
            key: value for key, value in self.fit_receipt.items() if key != "receipt_sha256"
        }
        width = len(self.feature_names)
        if (
            self.schema != CALIBRATED_BINARY_SCORER_SCHEMA
            or not self.feature_names
            or self.feature_names != tuple(sorted(self.feature_names))
            or len(set(self.feature_names)) != width
            or len(self.means) != width
            or len(self.scales) != width
            or len(self.weights) != width + 1
            or any(not math.isfinite(value) for value in (*self.means, *self.scales, *self.weights))
            or any(scale <= 0.0 for scale in self.scales)
            or self.fit_receipt.get("schema") != CALIBRATED_BINARY_FIT_RECEIPT_SCHEMA
            or self.fit_receipt.get("feature_names") != list(self.feature_names)
            or self.fit_receipt.get("parameters_sha256")
            != _parameter_sha(
                self.feature_names,
                self.means,
                self.scales,
                self.weights,
            )
            or self.fit_receipt.get("labels_available_to_fit") is not True
            or self.fit_receipt.get("labels_available_to_runtime") is not False
            or self.fit_receipt.get("text_available_to_scorer") is not False
            or self.fit_receipt.get("domain_identity_available_to_scorer") is not False
            or self.fit_receipt.get("admitted") is not True
            or self.fit_receipt.get("receipt_sha256") != _sha(body)
        ):
            raise ValueError("calibrated binary scorer envelope is invalid")
        object.__setattr__(self, "fit_receipt", json.loads(json.dumps(self.fit_receipt)))

    @property
    def receipt_sha256(self) -> str:
        return str(self.fit_receipt["receipt_sha256"])

    def predict(self, values: Mapping[str, float]) -> float:
        if not isinstance(values, Mapping) or set(values) != set(self.feature_names):
            raise ValueError("calibrated scorer feature schema differs")
        vector = tuple(
            _finite(values[name], field=f"feature {name!r}") for name in self.feature_names
        )
        z = self.weights[0] + sum(
            weight * ((value - mean) / scale)
            for weight, value, mean, scale in zip(
                self.weights[1:], vector, self.means, self.scales, strict=True
            )
        )
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "weights": list(self.weights),
            "fit_receipt": self.fit_receipt,
        }


def _calibration_report(
    scorer: CalibratedBinaryScorer | None,
    rows: Sequence[VerifiedBinaryObservation],
    *,
    fallback_predict: Any = None,
    baseline_probability: float,
    bins: int = 5,
) -> dict[str, Any]:
    predictions = [
        (
            scorer.predict(dict(row.values))
            if scorer is not None
            else float(fallback_predict(dict(row.values)))
        )
        for row in rows
    ]
    truths = [1.0 if row.verified_correct else 0.0 for row in rows]
    brier = sum((p - t) ** 2 for p, t in zip(predictions, truths, strict=True)) / len(rows)
    baseline_brier = sum((baseline_probability - t) ** 2 for t in truths) / len(rows)
    reliability = []
    weighted_gap = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            (prediction, truth)
            for prediction, truth in zip(predictions, truths, strict=True)
            if low <= prediction < high or (index == bins - 1 and prediction == 1.0)
        ]
        if not members:
            continue
        mean_prediction = sum(item[0] for item in members) / len(members)
        observed_rate = sum(item[1] for item in members) / len(members)
        weighted_gap += len(members) * abs(mean_prediction - observed_rate)
        reliability.append(
            {
                "bin": [round(low, 2), round(high, 2)],
                "n": len(members),
                "mean_prediction": round(mean_prediction, 6),
                "observed_rate": round(observed_rate, 6),
            }
        )
    return {
        "observations": len(rows),
        "brier": round(brier, 8),
        "baseline_probability": round(baseline_probability, 8),
        "baseline_brier": round(baseline_brier, 8),
        "brier_improvement": round(baseline_brier - brier, 8),
        "beats_fit_constant_predictor": bool(brier < baseline_brier),
        "expected_calibration_error": round(weighted_gap / len(rows), 8),
        "reliability": reliability,
    }


def fit_calibrated_binary_scorer(
    fit_rows: Sequence[VerifiedBinaryObservation],
    calibration_rows: Sequence[VerifiedBinaryObservation],
    *,
    epochs: int = 400,
    learning_rate: float = 0.1,
    l2: float = 1e-3,
    max_brier: float = MAX_ADMITTED_BRIER,
) -> tuple[CalibratedBinaryScorer | None, dict[str, Any]]:
    """Fit on one source split and admit only on an independent source split."""
    if len(fit_rows) < MIN_FIT_OBSERVATIONS:
        return None, {
            "admitted": False,
            "reason": "insufficient_fit_observations",
            "observations": len(fit_rows),
            "required": MIN_FIT_OBSERVATIONS,
        }
    if len(calibration_rows) < MIN_CALIBRATION_OBSERVATIONS:
        return None, {
            "admitted": False,
            "reason": "insufficient_calibration_observations",
            "observations": len(calibration_rows),
            "required": MIN_CALIBRATION_OBSERVATIONS,
        }
    names = fit_rows[0].names
    if (
        any(row.names != names for row in (*fit_rows, *calibration_rows))
        or len({row.source_ref for row in fit_rows}) != len(fit_rows)
        or len({row.source_ref for row in calibration_rows}) != len(calibration_rows)
        or {row.source_ref for row in fit_rows} & {row.source_ref for row in calibration_rows}
    ):
        raise ValueError("calibrated scorer splits or feature schemas differ")
    if len({row.verified_correct for row in fit_rows}) < 2:
        return None, {"admitted": False, "reason": "single_fit_outcome_class"}
    width = len(names)
    columns = tuple(tuple(row.vector()[index] for row in fit_rows) for index in range(width))
    means = tuple(sum(column) / len(column) for column in columns)
    scales = tuple(
        max(
            math.sqrt(sum((value - mean) ** 2 for value in column) / len(column)),
            1e-9,
        )
        for column, mean in zip(columns, means, strict=True)
    )
    normalized = tuple(
        tuple(
            (value - mean) / scale
            for value, mean, scale in zip(row.vector(), means, scales, strict=True)
        )
        for row in fit_rows
    )
    weights = [0.0] * (width + 1)
    for _ in range(max(1, int(epochs))):
        gradients = [0.0] * len(weights)
        for row, vector in zip(fit_rows, normalized, strict=True):
            z = weights[0] + sum(
                weight * value
                for weight, value in zip(weights[1:], vector, strict=True)
            )
            z = max(-30.0, min(30.0, z))
            error = (1.0 / (1.0 + math.exp(-z))) - float(row.verified_correct)
            gradients[0] += error
            for index, value in enumerate(vector, start=1):
                gradients[index] += error * value
        count = len(fit_rows)
        weights = [
            weight
            - learning_rate
            * (gradient / count + (0.0 if index == 0 else l2 * weight))
            for index, (weight, gradient) in enumerate(
                zip(weights, gradients, strict=True)
            )
        ]
    fit_base_rate = sum(row.verified_correct for row in fit_rows) / len(fit_rows)

    def raw_predict(values: Mapping[str, float]) -> float:
        vector = tuple(float(values[name]) for name in names)
        z = weights[0] + sum(
            weight * ((value - mean) / scale)
            for weight, value, mean, scale in zip(
                weights[1:], vector, means, scales, strict=True
            )
        )
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    calibration = _calibration_report(
        None,
        calibration_rows,
        fallback_predict=raw_predict,
        baseline_probability=fit_base_rate,
    )
    admitted = bool(
        calibration["beats_fit_constant_predictor"]
        and calibration["brier"] <= _finite(max_brier, field="max_brier")
    )
    report = {
        "admitted": admitted,
        "reason": "independent_calibration_passed" if admitted else "independent_calibration_failed",
        "feature_names": list(names),
        "fit_observations": len(fit_rows),
        "calibration": calibration,
    }
    if not admitted:
        return None, report
    fit_hash = _sha([row.to_dict() for row in fit_rows])
    calibration_hash = _sha([row.to_dict() for row in calibration_rows])
    receipt_body = {
        "schema": CALIBRATED_BINARY_FIT_RECEIPT_SCHEMA,
        "algorithm": "standardized_logistic_v1",
        "feature_names": list(names),
        "fit_observations": len(fit_rows),
        "calibration_observations": len(calibration_rows),
        "fit_observations_sha256": fit_hash,
        "calibration_observations_sha256": calibration_hash,
        "fit_base_rate": fit_base_rate,
        "calibration": calibration,
        "parameters_sha256": _parameter_sha(names, means, scales, weights),
        "labels_available_to_fit": True,
        "labels_available_to_runtime": False,
        "text_available_to_scorer": False,
        "domain_identity_available_to_scorer": False,
        "admitted": True,
    }
    scorer = CalibratedBinaryScorer(
        feature_names=names,
        means=means,
        scales=scales,
        weights=tuple(weights),
        fit_receipt={**receipt_body, "receipt_sha256": _sha(receipt_body)},
    )
    return scorer, report


def calibrated_binary_scorer_from_dict(value: Any) -> CalibratedBinaryScorer:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "feature_names",
        "means",
        "scales",
        "weights",
        "fit_receipt",
    }:
        raise ValueError("calibrated binary scorer payload is invalid")
    return CalibratedBinaryScorer(
        schema=str(value["schema"]),
        feature_names=tuple(str(name) for name in value["feature_names"]),
        means=tuple(float(item) for item in value["means"]),
        scales=tuple(float(item) for item in value["scales"]),
        weights=tuple(float(item) for item in value["weights"]),
        fit_receipt=dict(value["fit_receipt"]),
    )


__all__ = [
    "CALIBRATED_BINARY_FIT_RECEIPT_SCHEMA",
    "CALIBRATED_BINARY_SCORER_SCHEMA",
    "MAX_ADMITTED_BRIER",
    "MIN_CALIBRATION_OBSERVATIONS",
    "MIN_FIT_OBSERVATIONS",
    "CalibratedBinaryScorer",
    "VerifiedBinaryObservation",
    "calibrated_binary_scorer_from_dict",
    "fit_calibrated_binary_scorer",
]
