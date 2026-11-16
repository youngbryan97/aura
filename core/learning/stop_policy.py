"""Calibrated recurrent stop policy trained from verified trajectories.

The policy is intentionally a small portable logistic head over public,
bounded process signals. It does not inspect chain-of-thought text. A head is
admissible only after task-disjoint held-out calibration, and live execution
requires the exact artifact SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

STOP_POLICY_SCHEMA = "aura.rlc.stop_policy.v1"
STOP_POLICY_FEATURE_SCHEMA = "aura.rlc.stop_policy_features.v1"
STOP_WORKLOAD_CERTIFICATE_SCHEMA = "aura.rlc.stop_workload_certificate.v1"

STOP_FEATURE_NAMES = (
    "step_fraction",
    "residual",
    "residual_contraction_ratio",
    "quality_probability",
    "quality_uncertainty",
    "evidence_improvement",
    "verifier_score",
    "verifier_delta",
    "policy_uncertainty",
    "expected_gain_lcb",
    "expected_cost_ucb",
    "expected_net_value",
    "budget_remaining_fraction",
    "proposal_accepted",
    "quality_measured",
    "evoc_measured",
    "verifier_available",
)

MAX_STOP_ARTIFACT_BYTES = 262_144
MAX_STOP_EXAMPLES_PER_SPLIT = 100_000
MIN_STOP_TASKS_PER_SPLIT = 8
MIN_STOP_TASKS_PER_CLASS = 4


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


STOP_FEATURE_SCHEMA_SHA256 = canonical_sha256(
    {"schema": STOP_POLICY_FEATURE_SCHEMA, "features": STOP_FEATURE_NAMES}
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _bounded_text(value: Any, *, name: str, limit: int = 160) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > limit
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError(f"{name} must be bounded printable text")
    return normalized


def _feature_vector(
    value: Mapping[str, Any] | Sequence[float],
) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        if set(value) != set(STOP_FEATURE_NAMES):
            raise ValueError("stop-policy feature fields differ from schema")
        values = tuple(float(value[name]) for name in STOP_FEATURE_NAMES)
    else:
        values = tuple(float(item) for item in value)
        if len(values) != len(STOP_FEATURE_NAMES):
            raise ValueError("stop-policy feature vector has wrong width")
    if any(not math.isfinite(item) or abs(item) > 4.0 for item in values):
        raise ValueError("stop-policy features must be finite and bounded")
    return values


def _read_stable_json(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 1 <= before.st_size <= MAX_STOP_ARTIFACT_BYTES
        ):
            raise ValueError("stop-policy artifact size/type is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                raise ValueError("stop-policy artifact was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("stop-policy artifact grew during read")
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ValueError("stop-policy artifact changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp = math.exp(-min(value, 80.0))
        return 1.0 / (1.0 + exp)
    exp = math.exp(max(value, -80.0))
    return exp / (1.0 + exp)


@dataclass(frozen=True, slots=True)
class VerifiedStopExample:
    """One task step labelled by an independent suffix-outcome verifier."""

    example_id: str
    task_id: str
    features: tuple[float, ...]
    should_stop: bool
    verifier_receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "example_id",
            _bounded_text(self.example_id, name="stop example id"),
        )
        object.__setattr__(
            self,
            "task_id",
            _bounded_text(self.task_id, name="stop task id"),
        )
        object.__setattr__(self, "features", _feature_vector(self.features))
        if type(self.should_stop) is not bool:
            raise ValueError("stop-policy label must be boolean")
        if not _is_sha256(self.verifier_receipt_sha256):
            raise ValueError("stop-policy verifier receipt must be a SHA-256")

    @classmethod
    def from_values(
        cls,
        *,
        example_id: str,
        task_id: str,
        features: Mapping[str, Any] | Sequence[float],
        should_stop: bool,
        verifier_receipt_sha256: str,
    ) -> VerifiedStopExample:
        return cls(
            example_id=example_id,
            task_id=task_id,
            features=features,  # type: ignore[arg-type]
            should_stop=should_stop,
            verifier_receipt_sha256=verifier_receipt_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "task_id": self.task_id,
            "features": {
                name: round(value, 10)
                for name, value in zip(
                    STOP_FEATURE_NAMES,
                    self.features,
                    strict=True,
                )
            },
            "should_stop": self.should_stop,
            "verifier_receipt_sha256": self.verifier_receipt_sha256,
        }


def _calibration_metrics(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    threshold: float,
) -> dict[str, Any]:
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("stop-policy calibration rows do not align")
    positives = sum(labels)
    negatives = len(labels) - positives
    accepted = [probability >= threshold for probability in probabilities]
    tp = sum(
        prediction and label
        for prediction, label in zip(accepted, labels, strict=True)
    )
    tn = sum(
        not prediction and not label
        for prediction, label in zip(accepted, labels, strict=True)
    )
    fp = sum(
        prediction and not label
        for prediction, label in zip(accepted, labels, strict=True)
    )
    fn = sum(
        not prediction and label
        for prediction, label in zip(accepted, labels, strict=True)
    )
    pairs = 0
    wins = 0.0
    for p_index, positive in enumerate(labels):
        if not positive:
            continue
        for n_index, negative in enumerate(labels):
            if negative:
                continue
            pairs += 1
            if probabilities[p_index] > probabilities[n_index]:
                wins += 1.0
            elif probabilities[p_index] == probabilities[n_index]:
                wins += 0.5
    brier = sum(
        (probability - float(label)) ** 2
        for probability, label in zip(probabilities, labels, strict=True)
    ) / len(labels)
    ece = 0.0
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        members = [
            row
            for row, probability in enumerate(probabilities)
            if lower <= probability < upper
            or index == 9 and probability == 1.0
        ]
        if members:
            confidence = sum(probabilities[row] for row in members) / len(members)
            accuracy = sum(labels[row] for row in members) / len(members)
            ece += len(members) / len(labels) * abs(confidence - accuracy)
    tpr = tp / positives if positives else 0.0
    tnr = tn / negatives if negatives else 0.0
    return {
        "n": len(labels),
        "positives": positives,
        "negatives": negatives,
        "auc": wins / pairs if pairs else 0.5,
        "brier": brier,
        "ece_10_bin": ece,
        "false_stop_rate": fp / negatives if negatives else 1.0,
        "missed_stop_rate": fn / positives if positives else 1.0,
        "balanced_accuracy": 0.5 * (tpr + tnr),
    }


class StopPolicyHead:
    """Portable calibrated probability that further recurrence is not useful."""

    def __init__(
        self,
        *,
        means: Sequence[float],
        scales: Sequence[float],
        weights: Sequence[float],
        bias: float,
        threshold: float,
        calibration: Mapping[str, Any],
        training_data_sha256: str,
        calibration_data_sha256: str,
        training_task_sha256s: Sequence[str],
        calibration_task_sha256s: Sequence[str],
    ) -> None:
        self.means = tuple(float(value) for value in means)
        self.scales = tuple(float(value) for value in scales)
        self.weights = tuple(float(value) for value in weights)
        self.bias = float(bias)
        self.threshold = float(threshold)
        self.calibration = dict(calibration)
        self.training_data_sha256 = str(training_data_sha256)
        self.calibration_data_sha256 = str(calibration_data_sha256)
        self.training_task_sha256s = tuple(sorted(str(value) for value in training_task_sha256s))
        self.calibration_task_sha256s = tuple(
            sorted(str(value) for value in calibration_task_sha256s)
        )
        self._validate()

    def _validate(self) -> None:
        width = len(STOP_FEATURE_NAMES)
        if not all(
            len(values) == width
            for values in (self.means, self.scales, self.weights)
        ):
            raise ValueError("stop-policy parameter width is invalid")
        values = (*self.means, *self.scales, *self.weights, self.bias, self.threshold)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("stop-policy parameters must be finite")
        if any(scale <= 1e-8 for scale in self.scales):
            raise ValueError("stop-policy scales must be positive")
        if not 0.5 <= self.threshold < 1.0:
            raise ValueError("stop-policy threshold must be inside [0.5, 1)")
        required = {
            "schema",
            "admitted",
            "n",
            "positives",
            "negatives",
            "auc",
            "brier",
            "ece_10_bin",
            "false_stop_rate",
            "missed_stop_rate",
            "balanced_accuracy",
            "threshold",
        }
        if set(self.calibration) != required:
            raise ValueError("stop-policy calibration fields are invalid")
        if (
            self.calibration["schema"] != STOP_POLICY_SCHEMA
            or self.calibration["admitted"] is not True
            or type(self.calibration["n"]) is not int
            or type(self.calibration["positives"]) is not int
            or type(self.calibration["negatives"]) is not int
            or self.calibration["n"] < 32
            or min(
                self.calibration["positives"],
                self.calibration["negatives"],
            )
            < 8
            or self.calibration["positives"] + self.calibration["negatives"]
            != self.calibration["n"]
            or not all(
                _finite(self.calibration[name])
                for name in (
                    "auc",
                    "brier",
                    "ece_10_bin",
                    "false_stop_rate",
                    "missed_stop_rate",
                    "balanced_accuracy",
                    "threshold",
                )
            )
            or not all(
                0.0 <= float(self.calibration[name]) <= 1.0
                for name in (
                    "auc",
                    "brier",
                    "ece_10_bin",
                    "false_stop_rate",
                    "missed_stop_rate",
                    "balanced_accuracy",
                )
            )
            or float(self.calibration["auc"]) < 0.75
            or float(self.calibration["balanced_accuracy"]) < 0.70
            or float(self.calibration["brier"]) > 0.25
            or float(self.calibration["ece_10_bin"]) > 0.20
            or float(self.calibration["false_stop_rate"]) > 0.10
            or not math.isclose(
                float(self.calibration["threshold"]),
                self.threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not _is_sha256(self.training_data_sha256)
            or not _is_sha256(self.calibration_data_sha256)
            or not self.training_task_sha256s
            or not self.calibration_task_sha256s
            or len(self.training_task_sha256s) < MIN_STOP_TASKS_PER_SPLIT
            or len(self.calibration_task_sha256s) < MIN_STOP_TASKS_PER_SPLIT
            or len(set(self.training_task_sha256s)) != len(self.training_task_sha256s)
            or len(set(self.calibration_task_sha256s))
            != len(self.calibration_task_sha256s)
            or any(
                not _is_sha256(value)
                for value in (
                    *self.training_task_sha256s,
                    *self.calibration_task_sha256s,
                )
            )
            or set(self.training_task_sha256s) & set(self.calibration_task_sha256s)
        ):
            raise ValueError("stop-policy calibration was not admitted")

    def probability(
        self,
        features: Mapping[str, Any] | Sequence[float],
    ) -> float:
        values = _feature_vector(features)
        logit = self.bias + sum(
            weight * ((value - mean) / scale)
            for value, mean, scale, weight in zip(
                values,
                self.means,
                self.scales,
                self.weights,
                strict=True,
            )
        )
        return _sigmoid(logit)

    def to_dict(self) -> dict[str, Any]:
        self._validate()
        return {
            "schema": STOP_POLICY_SCHEMA,
            "feature_schema": STOP_POLICY_FEATURE_SCHEMA,
            "feature_schema_sha256": STOP_FEATURE_SCHEMA_SHA256,
            "feature_names": list(STOP_FEATURE_NAMES),
            "means": list(self.means),
            "scales": list(self.scales),
            "weights": list(self.weights),
            "bias": self.bias,
            "threshold": self.threshold,
            "training_data_sha256": self.training_data_sha256,
            "calibration_data_sha256": self.calibration_data_sha256,
            "training_task_sha256s": list(self.training_task_sha256s),
            "calibration_task_sha256s": list(self.calibration_task_sha256s),
            "training_task_set_sha256": canonical_sha256(
                list(self.training_task_sha256s)
            ),
            "calibration_task_set_sha256": canonical_sha256(
                list(self.calibration_task_sha256s)
            ),
            "calibration": dict(self.calibration),
        }

    def manifest(self) -> dict[str, Any]:
        artifact = self.to_dict()
        return {
            key: artifact[key]
            for key in (
                "schema",
                "feature_schema",
                "feature_schema_sha256",
                "feature_names",
                "threshold",
                "training_data_sha256",
                "calibration_data_sha256",
                "training_task_set_sha256",
                "calibration_task_set_sha256",
                "calibration",
            )
        }

    def save(self, path: str | Path) -> str:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        target = Path(path).expanduser()
        if target.suffix != ".json":
            raise ValueError("stop-policy artifact path must end in .json")
        raw = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        if len(raw) > MAX_STOP_ARTIFACT_BYTES:
            raise ValueError("stop-policy artifact exceeds its size bound")
        with local_internal_governed_scope("stop_policy.save"):
            get_file_write_gateway().write_bytes(target, raw, source="stop_policy.save")
        return hashlib.sha256(_read_stable_json(target)).hexdigest()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
    ) -> StopPolicyHead:
        if not _is_sha256(expected_sha256):
            raise ValueError("stop-policy artifact requires a pinned SHA-256")
        raw = _read_stable_json(Path(path).expanduser())
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("stop-policy artifact digest mismatch")

        def reject_constant(value: str) -> Never:
            raise ValueError(f"stop-policy JSON constant is invalid: {value}")

        try:
            payload = json.loads(raw, parse_constant=reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("stop-policy artifact JSON is invalid") from exc
        expected_fields = {
            "schema",
            "feature_schema",
            "feature_schema_sha256",
            "feature_names",
            "means",
            "scales",
            "weights",
            "bias",
            "threshold",
            "training_data_sha256",
            "calibration_data_sha256",
            "training_task_sha256s",
            "calibration_task_sha256s",
            "training_task_set_sha256",
            "calibration_task_set_sha256",
            "calibration",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected_fields
            or payload["schema"] != STOP_POLICY_SCHEMA
            or payload["feature_schema"] != STOP_POLICY_FEATURE_SCHEMA
            or payload["feature_schema_sha256"] != STOP_FEATURE_SCHEMA_SHA256
            or payload["feature_names"] != list(STOP_FEATURE_NAMES)
            or payload["training_task_set_sha256"]
            != canonical_sha256(payload["training_task_sha256s"])
            or payload["calibration_task_set_sha256"]
            != canonical_sha256(payload["calibration_task_sha256s"])
        ):
            raise ValueError("stop-policy artifact schema is invalid")
        try:
            return cls(
                means=payload["means"],
                scales=payload["scales"],
                weights=payload["weights"],
                bias=payload["bias"],
                threshold=payload["threshold"],
                calibration=payload["calibration"],
                training_data_sha256=payload["training_data_sha256"],
                calibration_data_sha256=payload["calibration_data_sha256"],
                training_task_sha256s=payload["training_task_sha256s"],
                calibration_task_sha256s=payload["calibration_task_sha256s"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("stop-policy artifact values are invalid") from exc


def fit_stop_policy_head(
    training: Sequence[VerifiedStopExample],
    calibration: Sequence[VerifiedStopExample],
    *,
    epochs: int = 1400,
    learning_rate: float = 0.06,
    l2: float = 1e-3,
) -> StopPolicyHead:
    """Fit a deterministic task-disjoint stop classifier."""

    if type(epochs) is not int or not 1 <= epochs <= 100_000:
        raise ValueError("stop-policy epochs are invalid")
    if not _finite(learning_rate) or not 0.0 < float(learning_rate) <= 1.0:
        raise ValueError("stop-policy learning rate is invalid")
    if not _finite(l2) or not 0.0 <= float(l2) <= 1.0:
        raise ValueError("stop-policy L2 is invalid")
    train_rows = list(training)
    calibration_rows = list(calibration)
    if (
        not 32 <= len(train_rows) <= MAX_STOP_EXAMPLES_PER_SPLIT
        or not 32 <= len(calibration_rows) <= MAX_STOP_EXAMPLES_PER_SPLIT
    ):
        raise ValueError("stop-policy fitting requires 32 rows per split")
    train_example_ids = [row.example_id for row in train_rows]
    calibration_example_ids = [row.example_id for row in calibration_rows]
    train_task_ids = {row.task_id for row in train_rows}
    calibration_task_ids = {row.task_id for row in calibration_rows}
    if (
        len(set(train_example_ids)) != len(train_example_ids)
        or len(set(calibration_example_ids)) != len(calibration_example_ids)
        or set(train_example_ids) & set(calibration_example_ids)
        or train_task_ids & calibration_task_ids
    ):
        raise ValueError("stop-policy train/calibration identities overlap")
    if min(len(train_task_ids), len(calibration_task_ids)) < MIN_STOP_TASKS_PER_SPLIT:
        raise ValueError("stop-policy splits need eight unique tasks")
    for rows in (train_rows, calibration_rows):
        if min(
            sum(row.should_stop for row in rows),
            sum(not row.should_stop for row in rows),
        ) < 8:
            raise ValueError("stop-policy splits need eight examples per class")
        if min(
            len({row.task_id for row in rows if row.should_stop}),
            len({row.task_id for row in rows if not row.should_stop}),
        ) < MIN_STOP_TASKS_PER_CLASS:
            raise ValueError(
                "stop-policy splits need four unique tasks per class"
            )

    width = len(STOP_FEATURE_NAMES)
    means = tuple(
        sum(row.features[index] for row in train_rows) / len(train_rows)
        for index in range(width)
    )
    scales: list[float] = []
    for index, mean in enumerate(means):
        variance = sum(
            (row.features[index] - mean) ** 2 for row in train_rows
        ) / len(train_rows)
        scales.append(max(math.sqrt(variance), 1e-6))
    normalized = [
        tuple(
            (value - mean) / scale
            for value, mean, scale in zip(
                row.features,
                means,
                scales,
                strict=True,
            )
        )
        for row in train_rows
    ]
    positives = sum(row.should_stop for row in train_rows)
    negatives = len(train_rows) - positives
    positive_weight = len(train_rows) / (2.0 * positives)
    negative_weight = len(train_rows) / (2.0 * negatives)
    weights = [0.0] * width
    bias = 0.0
    for _ in range(epochs):
        gradients = [0.0] * width
        bias_gradient = 0.0
        total_weight = 0.0
        for row, values in zip(train_rows, normalized, strict=True):
            sample_weight = positive_weight if row.should_stop else negative_weight
            probability = _sigmoid(
                bias
                + sum(
                    weight * value
                    for weight, value in zip(weights, values, strict=True)
                )
            )
            error = (probability - float(row.should_stop)) * sample_weight
            bias_gradient += error
            total_weight += sample_weight
            for index, value in enumerate(values):
                gradients[index] += error * value
        for index in range(width):
            gradient = gradients[index] / total_weight + float(l2) * weights[index]
            weights[index] -= float(learning_rate) * gradient
        bias -= float(learning_rate) * bias_gradient / total_weight

    def probability(features: tuple[float, ...]) -> float:
        logit = bias + sum(
            weight * ((value - mean) / scale)
            for value, mean, scale, weight in zip(
                features,
                means,
                scales,
                weights,
                strict=True,
            )
        )
        return _sigmoid(logit)

    probabilities = [probability(row.features) for row in calibration_rows]
    labels = [row.should_stop for row in calibration_rows]
    candidates = sorted(
        {0.5, *(min(0.99, max(0.5, value)) for value in probabilities)}
    )
    rows = [
        (threshold, _calibration_metrics(probabilities, labels, threshold))
        for threshold in candidates
    ]
    constrained = [row for row in rows if row[1]["false_stop_rate"] <= 0.10]
    threshold, metrics = max(
        constrained or rows,
        key=lambda row: (
            row[1]["balanced_accuracy"],
            -row[1]["false_stop_rate"],
            row[0],
        ),
    )
    admitted = bool(
        metrics["auc"] >= 0.75
        and metrics["balanced_accuracy"] >= 0.70
        and metrics["brier"] <= 0.25
        and metrics["ece_10_bin"] <= 0.20
        and metrics["false_stop_rate"] <= 0.10
    )
    calibration_record = {
        "schema": STOP_POLICY_SCHEMA,
        "admitted": admitted,
        **metrics,
        "threshold": threshold,
    }
    if not admitted:
        raise ValueError(
            f"stop-policy calibration failed admission: {calibration_record}"
        )
    return StopPolicyHead(
        means=means,
        scales=scales,
        weights=weights,
        bias=bias,
        threshold=threshold,
        calibration=calibration_record,
        training_data_sha256=canonical_sha256(
            [row.to_dict() for row in train_rows]
        ),
        calibration_data_sha256=canonical_sha256(
            [row.to_dict() for row in calibration_rows]
        ),
        training_task_sha256s=tuple(
            sorted(
                hashlib.sha256(task_id.encode("utf-8")).hexdigest()
                for task_id in train_task_ids
            )
        ),
        calibration_task_sha256s=tuple(
            sorted(
                hashlib.sha256(task_id.encode("utf-8")).hexdigest()
                for task_id in calibration_task_ids
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class VerifiedStopTrajectory:
    """Held-out task trajectory with an independently verified required depth."""

    task_id: str
    difficulty: str
    steps: tuple[Mapping[str, Any], ...]
    correct_by_step: tuple[bool, ...]
    required_step: int
    verifier_receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_id",
            _bounded_text(self.task_id, name="trajectory task id"),
        )
        if self.difficulty not in {"easy", "hard"}:
            raise ValueError("trajectory difficulty must be easy or hard")
        normalized_steps = tuple(
            {
                name: value
                for name, value in zip(
                    STOP_FEATURE_NAMES,
                    _feature_vector(step),
                    strict=True,
                )
            }
            for step in self.steps
        )
        object.__setattr__(self, "steps", normalized_steps)
        if (
            len(normalized_steps) < 2
            or len(self.correct_by_step) != len(normalized_steps)
            or any(type(value) is not bool for value in self.correct_by_step)
            or type(self.required_step) is not int
            or not 1 <= self.required_step <= len(normalized_steps)
            or self.correct_by_step[self.required_step - 1] is not True
            or not _is_sha256(self.verifier_receipt_sha256)
        ):
            raise ValueError("verified stop trajectory is invalid")


def certify_stop_workload(
    head: StopPolicyHead,
    trajectories: Sequence[VerifiedStopTrajectory],
) -> dict[str, Any]:
    """Certify efficiency on easy tasks and hard-task accuracy non-regression."""

    rows = list(trajectories)
    if len(rows) < 8 or len({row.task_id for row in rows}) != len(rows):
        raise ValueError("stop workload needs eight unique held-out tasks")
    if min(
        sum(row.difficulty == "easy" for row in rows),
        sum(row.difficulty == "hard" for row in rows),
    ) < 4:
        raise ValueError("stop workload needs four easy and four hard tasks")
    heldout_task_sha256s = {
        hashlib.sha256(row.task_id.encode("utf-8")).hexdigest() for row in rows
    }
    if heldout_task_sha256s & (
        set(head.training_task_sha256s) | set(head.calibration_task_sha256s)
    ):
        raise ValueError("stop workload overlaps training or calibration tasks")
    results: list[dict[str, Any]] = []
    for trajectory in rows:
        selected_step = len(trajectory.steps)
        for index, features in enumerate(trajectory.steps, start=1):
            if (
                features["quality_measured"] >= 0.5
                and features["evoc_measured"] >= 0.5
                and head.probability(features) >= head.threshold
            ):
                selected_step = index
                break
        results.append(
            {
                "task_id": trajectory.task_id,
                "difficulty": trajectory.difficulty,
                "required_step": trajectory.required_step,
                "baseline_step": len(trajectory.steps),
                "selected_step": selected_step,
                "baseline_correct": trajectory.correct_by_step[-1],
                "selected_correct": trajectory.correct_by_step[selected_step - 1],
                "verifier_receipt_sha256": trajectory.verifier_receipt_sha256,
            }
        )
    easy = [row for row in results if row["difficulty"] == "easy"]
    hard = [row for row in results if row["difficulty"] == "hard"]
    baseline_accuracy = sum(row["baseline_correct"] for row in results) / len(results)
    selected_accuracy = sum(row["selected_correct"] for row in results) / len(results)
    hard_baseline_accuracy = sum(row["baseline_correct"] for row in hard) / len(hard)
    hard_selected_accuracy = sum(row["selected_correct"] for row in hard) / len(hard)
    easy_step_reduction = sum(
        row["baseline_step"] - row["selected_step"] for row in easy
    ) / len(easy)
    hard_premature_stops = sum(
        row["selected_step"] < row["required_step"] for row in hard
    )
    admitted = bool(
        easy_step_reduction >= 1.0
        and selected_accuracy + 1e-12 >= baseline_accuracy
        and hard_selected_accuracy + 1e-12 >= hard_baseline_accuracy
        and hard_premature_stops == 0
    )
    payload = {
        "schema": STOP_WORKLOAD_CERTIFICATE_SCHEMA,
        "head_manifest": head.manifest(),
        "task_count": len(results),
        "easy_tasks": len(easy),
        "hard_tasks": len(hard),
        "easy_mean_step_reduction": round(easy_step_reduction, 8),
        "baseline_accuracy": round(baseline_accuracy, 8),
        "selected_accuracy": round(selected_accuracy, 8),
        "hard_baseline_accuracy": round(hard_baseline_accuracy, 8),
        "hard_selected_accuracy": round(hard_selected_accuracy, 8),
        "hard_premature_stops": hard_premature_stops,
        "heldout_task_set_sha256": canonical_sha256(
            sorted(heldout_task_sha256s)
        ),
        "admitted": admitted,
        "results": results,
    }
    if not admitted:
        raise ValueError(f"stop workload certificate rejected: {payload}")
    return {**payload, "certificate_sha256": canonical_sha256(payload)}


__all__ = [
    "MAX_STOP_ARTIFACT_BYTES",
    "MAX_STOP_EXAMPLES_PER_SPLIT",
    "MIN_STOP_TASKS_PER_CLASS",
    "MIN_STOP_TASKS_PER_SPLIT",
    "STOP_FEATURE_NAMES",
    "STOP_FEATURE_SCHEMA_SHA256",
    "STOP_POLICY_FEATURE_SCHEMA",
    "STOP_POLICY_SCHEMA",
    "STOP_WORKLOAD_CERTIFICATE_SCHEMA",
    "StopPolicyHead",
    "VerifiedStopExample",
    "VerifiedStopTrajectory",
    "canonical_sha256",
    "certify_stop_workload",
    "fit_stop_policy_head",
]
