"""Calibrated learned admission for recurrent latent-state proposals.

The head is deliberately small: a sigmoid over stable, evidence-conditioned
transition features.  It is trained on independently labelled transition
outcomes rather than self-reported confidence, calibrated on a disjoint split,
and serialized with the complete calibration/provenance record.  Runtime code
may load only an admitted artifact whose bytes match a pinned SHA-256.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UPDATE_ACCEPTANCE_HEAD_SCHEMA = "aura.rlc.update_acceptance_head.v1"
UPDATE_ACCEPTANCE_FEATURE_SCHEMA = "aura.rlc.update_acceptance_features.v1"

FEATURE_NAMES = (
    "proposal_residual",
    "anchor_alignment_delta",
    "evidence_alignment_delta",
    "anchor_distance_improvement",
    "evidence_distance_improvement",
    "proposal_previous_cosine",
    "delta_anchor_cosine",
    "delta_evidence_cosine",
    "proposal_anchor_log_rms_error",
    "previous_anchor_log_rms_error",
    "residual_contraction_ratio",
    "delta_cosine_previous",
    "evidence_available",
)


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


FEATURE_SCHEMA_SHA256 = _canonical_sha256(
    {"schema": UPDATE_ACCEPTANCE_FEATURE_SCHEMA, "features": FEATURE_NAMES}
)

MAX_HEAD_ARTIFACT_BYTES = 1_048_576
MAX_HEAD_UNCOMPRESSED_BYTES = 2_097_152
_NPZ_MEMBER_NAMES = {
    "means.npy",
    "scales.npy",
    "weights.npy",
    "bias.npy",
    "manifest.npy",
}


def _read_stable_artifact(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 1 <= before.st_size <= MAX_HEAD_ARTIFACT_BYTES
        ):
            raise ValueError("update-acceptance artifact size/type is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                raise ValueError("update-acceptance artifact was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("update-acceptance artifact grew during read")
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ValueError("update-acceptance artifact changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_npz_container(artifact_bytes: bytes) -> None:
    """Reject ambiguous or expansion-heavy containers before NumPy allocates."""

    try:
        with zipfile.ZipFile(io.BytesIO(artifact_bytes)) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if (
                len(names) != len(_NPZ_MEMBER_NAMES)
                or set(names) != _NPZ_MEMBER_NAMES
                or any(member.is_dir() or member.flag_bits & 0x1 for member in members)
            ):
                raise ValueError("update-acceptance NPZ members are invalid")
            expanded_bytes = sum(member.file_size for member in members)
            if (
                expanded_bytes > MAX_HEAD_UNCOMPRESSED_BYTES
                or any(
                    member.file_size > MAX_HEAD_UNCOMPRESSED_BYTES
                    for member in members
                )
            ):
                raise ValueError("update-acceptance NPZ expands beyond its bound")
    except zipfile.BadZipFile as exc:
        raise ValueError("update-acceptance artifact is not a valid NPZ") from exc


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _feature_vector(value: Mapping[str, Any] | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        if set(value) != set(FEATURE_NAMES):
            raise ValueError("update-acceptance feature fields differ from schema")
        values = tuple(float(value[name]) for name in FEATURE_NAMES)
    else:
        values = tuple(float(item) for item in value)
        if len(values) != len(FEATURE_NAMES):
            raise ValueError("update-acceptance feature vector has wrong width")
    if any(not math.isfinite(item) or abs(item) > 32.0 for item in values):
        raise ValueError("update-acceptance features must be finite and bounded")
    return values


@dataclass(frozen=True)
class VerifiedTransitionExample:
    """One transition labelled by an external/process-verifiable outcome."""

    example_id: str
    features: tuple[float, ...]
    improved: bool
    verifier_receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id.strip():
            raise ValueError("transition example id must be non-empty")
        if type(self.improved) is not bool:
            raise ValueError("transition improvement label must be boolean")
        if not _is_sha256(self.verifier_receipt_sha256):
            raise ValueError("transition verifier receipt must be a SHA-256")
        object.__setattr__(self, "example_id", self.example_id.strip())
        object.__setattr__(self, "features", _feature_vector(self.features))

    @classmethod
    def from_values(
        cls,
        *,
        example_id: str,
        features: Mapping[str, Any] | Sequence[float],
        improved: bool,
        verifier_receipt_sha256: str,
    ) -> VerifiedTransitionExample:
        return cls(
            example_id=example_id,
            features=features,  # type: ignore[arg-type]
            improved=improved,
            verifier_receipt_sha256=verifier_receipt_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "features": {
                name: round(value, 10)
                for name, value in zip(FEATURE_NAMES, self.features, strict=True)
            },
            "improved": self.improved,
            "verifier_receipt_sha256": self.verifier_receipt_sha256,
        }


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp = math.exp(-min(value, 80.0))
        return 1.0 / (1.0 + exp)
    exp = math.exp(max(value, -80.0))
    return exp / (1.0 + exp)


def _metrics(probabilities: Sequence[float], labels: Sequence[bool]) -> dict[str, Any]:
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("calibration probabilities and labels must align")
    positives = sum(labels)
    negatives = len(labels) - positives
    brier = sum(
        (float(probability) - float(label)) ** 2
        for probability, label in zip(probabilities, labels, strict=True)
    ) / len(labels)
    ece = 0.0
    for lower in (index / 10.0 for index in range(10)):
        upper = lower + 0.1
        members = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= probability < upper or upper >= 1.0 and probability == 1.0
        ]
        if not members:
            continue
        confidence = sum(probabilities[index] for index in members) / len(members)
        accuracy = sum(labels[index] for index in members) / len(members)
        ece += len(members) / len(labels) * abs(confidence - accuracy)
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
    auc = wins / pairs if pairs else 0.5
    return {
        "n": len(labels),
        "positives": positives,
        "negatives": negatives,
        "brier": brier,
        "ece_10_bin": ece,
        "auc": auc,
    }


def _threshold_metrics(
    probabilities: Sequence[float], labels: Sequence[bool], threshold: float
) -> dict[str, float]:
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
    positives = tp + fn
    negatives = tn + fp
    tpr = tp / positives if positives else 0.0
    tnr = tn / negatives if negatives else 0.0
    return {
        "false_accept_rate": fp / negatives if negatives else 1.0,
        "false_reject_rate": fn / positives if positives else 1.0,
        "balanced_accuracy": 0.5 * (tpr + tnr),
    }


class UpdateAcceptanceHead:
    """Portable logistic head over evidence/process transition features."""

    def __init__(
        self,
        *,
        means: Sequence[float] | None = None,
        scales: Sequence[float] | None = None,
        weights: Sequence[float] | None = None,
        bias: float = 0.0,
        threshold: float = 0.5,
        calibration: Mapping[str, Any] | None = None,
        training_data_sha256: str = "",
        calibration_data_sha256: str = "",
    ) -> None:
        width = len(FEATURE_NAMES)
        self.means = tuple(float(value) for value in (means or (0.0,) * width))
        self.scales = tuple(float(value) for value in (scales or (1.0,) * width))
        self.weights = tuple(float(value) for value in (weights or (0.0,) * width))
        self.bias = float(bias)
        self.threshold = float(threshold)
        self.calibration = dict(calibration or {})
        self.training_data_sha256 = str(training_data_sha256)
        self.calibration_data_sha256 = str(calibration_data_sha256)
        self._validate(allow_uncalibrated=True)

    def _validate(self, *, allow_uncalibrated: bool) -> None:
        width = len(FEATURE_NAMES)
        if not all(
            len(values) == width
            for values in (self.means, self.scales, self.weights)
        ):
            raise ValueError("update-acceptance head parameter width is invalid")
        numeric = (*self.means, *self.scales, *self.weights, self.bias, self.threshold)
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("update-acceptance head parameters must be finite")
        if any(scale <= 1e-8 for scale in self.scales):
            raise ValueError("update-acceptance feature scales must be positive")
        if not 0.5 <= self.threshold < 1.0:
            raise ValueError("update-acceptance threshold must be inside [0.5, 1)")
        if not self.calibration:
            if allow_uncalibrated:
                return
            raise ValueError("update-acceptance head has no held-out calibration")
        required = {
            "schema",
            "admitted",
            "n",
            "positives",
            "negatives",
            "brier",
            "ece_10_bin",
            "auc",
            "false_accept_rate",
            "false_reject_rate",
            "balanced_accuracy",
            "threshold",
        }
        if set(self.calibration) != required:
            raise ValueError("update-acceptance calibration fields are invalid")
        for name in (
            "brier",
            "ece_10_bin",
            "auc",
            "false_accept_rate",
            "false_reject_rate",
            "balanced_accuracy",
            "threshold",
        ):
            if not _finite(self.calibration[name]):
                raise ValueError("update-acceptance calibration is non-finite")
        if (
            self.calibration["schema"] != UPDATE_ACCEPTANCE_HEAD_SCHEMA
            or self.calibration["admitted"] is not True
            or type(self.calibration["n"]) is not int
            or type(self.calibration["positives"]) is not int
            or type(self.calibration["negatives"]) is not int
            or self.calibration["n"] < 32
            or min(self.calibration["positives"], self.calibration["negatives"]) < 8
            or self.calibration["positives"] + self.calibration["negatives"]
            != self.calibration["n"]
            or float(self.calibration["auc"]) < 0.75
            or float(self.calibration["balanced_accuracy"]) < 0.70
            or float(self.calibration["brier"]) > 0.25
            or float(self.calibration["ece_10_bin"]) > 0.20
            or float(self.calibration["false_accept_rate"]) > 0.25
            or not math.isclose(
                float(self.calibration["threshold"]),
                self.threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not _is_sha256(self.training_data_sha256)
            or not _is_sha256(self.calibration_data_sha256)
        ):
            raise ValueError("update-acceptance calibration was not admitted")

    @property
    def calibrated(self) -> bool:
        try:
            self._validate(allow_uncalibrated=False)
        # not a failure: an update that fails its own validation is not calibrated, which
        # is what this property reports.
        except ValueError:
            return False
        return True

    def probability(self, features: Mapping[str, Any] | Sequence[float]) -> float:
        values = _feature_vector(features)
        normalized = tuple(
            (value - mean) / scale
            for value, mean, scale in zip(
                values, self.means, self.scales, strict=True
            )
        )
        logit = self.bias + sum(
            weight * value
            for weight, value in zip(self.weights, normalized, strict=True)
        )
        probability = _sigmoid(logit)
        if not math.isfinite(probability):
            raise ValueError("update-acceptance probability is non-finite")
        return probability

    def to_manifest(self) -> dict[str, Any]:
        self._validate(allow_uncalibrated=False)
        return {
            "schema": UPDATE_ACCEPTANCE_HEAD_SCHEMA,
            "feature_schema": UPDATE_ACCEPTANCE_FEATURE_SCHEMA,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "feature_names": list(FEATURE_NAMES),
            "threshold": self.threshold,
            "training_data_sha256": self.training_data_sha256,
            "calibration_data_sha256": self.calibration_data_sha256,
            "calibration": dict(self.calibration),
        }

    def save(self, path: str | Path) -> str:
        """Persist one admitted artifact and return its exact file SHA-256."""

        import numpy as np

        self._validate(allow_uncalibrated=False)
        target = Path(path).expanduser()
        if target.suffix != ".npz":
            raise ValueError("update-acceptance artifact path must end in .npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest = json.dumps(
            self.to_manifest(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez(
                    handle,
                    means=np.asarray(self.means, dtype=np.float64),
                    scales=np.asarray(self.scales, dtype=np.float64),
                    weights=np.asarray(self.weights, dtype=np.float64),
                    bias=np.asarray([self.bias], dtype=np.float64),
                    manifest=np.asarray([manifest]),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
            directory_fd = os.open(
                target.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        return hashlib.sha256(_read_stable_artifact(target)).hexdigest()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
    ) -> UpdateAcceptanceHead:
        import numpy as np

        source = Path(path).expanduser()
        if not _is_sha256(expected_sha256):
            raise ValueError("update-acceptance head requires a pinned SHA-256")
        artifact_bytes = _read_stable_artifact(source)
        observed_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValueError("update-acceptance head artifact digest mismatch")
        _validate_npz_container(artifact_bytes)
        with np.load(io.BytesIO(artifact_bytes), allow_pickle=False) as payload:
            if set(payload.files) != {"means", "scales", "weights", "bias", "manifest"}:
                raise ValueError("update-acceptance artifact fields are invalid")
            try:
                manifest = json.loads(str(payload["manifest"][0]))
            except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
                raise ValueError("update-acceptance manifest is invalid") from exc
            required_manifest = {
                "schema",
                "feature_schema",
                "feature_schema_sha256",
                "feature_names",
                "threshold",
                "training_data_sha256",
                "calibration_data_sha256",
                "calibration",
            }
            if (
                not isinstance(manifest, dict)
                or set(manifest) != required_manifest
                or manifest["schema"] != UPDATE_ACCEPTANCE_HEAD_SCHEMA
                or manifest["feature_schema"] != UPDATE_ACCEPTANCE_FEATURE_SCHEMA
                or manifest["feature_schema_sha256"] != FEATURE_SCHEMA_SHA256
                or manifest["feature_names"] != list(FEATURE_NAMES)
                or payload["means"].shape != (len(FEATURE_NAMES),)
                or payload["scales"].shape != (len(FEATURE_NAMES),)
                or payload["weights"].shape != (len(FEATURE_NAMES),)
                or payload["bias"].shape != (1,)
            ):
                raise ValueError("update-acceptance artifact manifest differs")
            head = cls(
                means=payload["means"].tolist(),
                scales=payload["scales"].tolist(),
                weights=payload["weights"].tolist(),
                bias=float(payload["bias"][0]),
                threshold=float(manifest["threshold"]),
                calibration=manifest["calibration"],
                training_data_sha256=str(manifest["training_data_sha256"]),
                calibration_data_sha256=str(manifest["calibration_data_sha256"]),
            )
        head._validate(allow_uncalibrated=False)
        return head


def fit_update_acceptance_head(
    training: Sequence[VerifiedTransitionExample],
    calibration: Sequence[VerifiedTransitionExample],
    *,
    epochs: int = 1200,
    learning_rate: float = 0.08,
    l2: float = 1e-3,
) -> UpdateAcceptanceHead:
    """Fit and calibrate a deterministic full-batch logistic admission head."""

    if type(epochs) is not int or not 1 <= epochs <= 100_000:
        raise ValueError("update-acceptance epochs must be inside [1, 100000]")
    if not _finite(learning_rate) or not 0.0 < learning_rate <= 1.0:
        raise ValueError("update-acceptance learning rate must be inside (0, 1]")
    if not _finite(l2) or not 0.0 <= l2 <= 1.0:
        raise ValueError("update-acceptance L2 must be inside [0, 1]")
    train_rows = list(training)
    calibration_rows = list(calibration)
    if len(train_rows) < 32 or len(calibration_rows) < 32:
        raise ValueError("update-acceptance fitting requires 32 rows per split")
    train_ids = [row.example_id for row in train_rows]
    calibration_ids = [row.example_id for row in calibration_rows]
    if (
        len(set(train_ids)) != len(train_ids)
        or len(set(calibration_ids)) != len(calibration_ids)
        or set(train_ids) & set(calibration_ids)
    ):
        raise ValueError("update-acceptance train/calibration identities overlap")
    if min(sum(row.improved for row in rows) for rows in (train_rows, calibration_rows)) < 8:
        raise ValueError("update-acceptance splits need at least eight positives")
    if min(sum(not row.improved for row in rows) for rows in (train_rows, calibration_rows)) < 8:
        raise ValueError("update-acceptance splits need at least eight negatives")

    width = len(FEATURE_NAMES)
    means = tuple(
        sum(row.features[index] for row in train_rows) / len(train_rows)
        for index in range(width)
    )
    scales = []
    for index, mean in enumerate(means):
        variance = sum(
            (row.features[index] - mean) ** 2 for row in train_rows
        ) / len(train_rows)
        scales.append(max(math.sqrt(variance), 1e-6))
    normalized = [
        tuple(
            (value - mean) / scale
            for value, mean, scale in zip(
                row.features, means, scales, strict=True
            )
        )
        for row in train_rows
    ]
    positive_count = sum(row.improved for row in train_rows)
    negative_count = len(train_rows) - positive_count
    positive_weight = len(train_rows) / (2.0 * positive_count)
    negative_weight = len(train_rows) / (2.0 * negative_count)
    weights = [0.0] * width
    bias = 0.0
    for _ in range(epochs):
        grad_weights = [0.0] * width
        grad_bias = 0.0
        total_weight = 0.0
        for row, values in zip(train_rows, normalized, strict=True):
            sample_weight = positive_weight if row.improved else negative_weight
            probability = _sigmoid(
                bias
                + sum(
                    weight * value
                    for weight, value in zip(weights, values, strict=True)
                )
            )
            error = (probability - float(row.improved)) * sample_weight
            grad_bias += error
            total_weight += sample_weight
            for index, value in enumerate(values):
                grad_weights[index] += error * value
        grad_bias /= total_weight
        for index in range(width):
            gradient = grad_weights[index] / total_weight + l2 * weights[index]
            weights[index] -= float(learning_rate) * gradient
        bias -= float(learning_rate) * grad_bias

    provisional = UpdateAcceptanceHead(
        means=means,
        scales=scales,
        weights=weights,
        bias=bias,
    )
    probabilities = [provisional.probability(row.features) for row in calibration_rows]
    labels = [row.improved for row in calibration_rows]
    candidates = sorted(
        {0.5, *[min(0.99, max(0.5, probability)) for probability in probabilities]}
    )
    threshold_rows = [
        (threshold, _threshold_metrics(probabilities, labels, threshold))
        for threshold in candidates
    ]
    constrained = [
        row for row in threshold_rows if row[1]["false_accept_rate"] <= 0.25
    ]
    pool = constrained or threshold_rows
    threshold, threshold_result = max(
        pool,
        key=lambda row: (
            row[1]["balanced_accuracy"],
            -row[1]["false_accept_rate"],
            row[0],
        ),
    )
    calibration_metrics = _metrics(probabilities, labels)
    admitted = bool(
        calibration_metrics["n"] >= 32
        and min(
            calibration_metrics["positives"],
            calibration_metrics["negatives"],
        )
        >= 8
        and calibration_metrics["auc"] >= 0.75
        and calibration_metrics["brier"] <= 0.25
        and calibration_metrics["ece_10_bin"] <= 0.20
        and threshold_result["balanced_accuracy"] >= 0.70
        and threshold_result["false_accept_rate"] <= 0.25
    )
    calibration_record = {
        "schema": UPDATE_ACCEPTANCE_HEAD_SCHEMA,
        "admitted": admitted,
        **calibration_metrics,
        **threshold_result,
        "threshold": threshold,
    }
    head = UpdateAcceptanceHead(
        means=means,
        scales=scales,
        weights=weights,
        bias=bias,
        threshold=threshold,
        calibration=calibration_record,
        training_data_sha256=_canonical_sha256(
            [row.to_dict() for row in train_rows]
        ),
        calibration_data_sha256=_canonical_sha256(
            [row.to_dict() for row in calibration_rows]
        ),
    )
    if not admitted:
        raise ValueError(
            "update-acceptance calibration failed admission: "
            f"{calibration_record}"
        )
    head._validate(allow_uncalibrated=False)
    return head


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_SHA256",
    "MAX_HEAD_ARTIFACT_BYTES",
    "MAX_HEAD_UNCOMPRESSED_BYTES",
    "UPDATE_ACCEPTANCE_FEATURE_SCHEMA",
    "UPDATE_ACCEPTANCE_HEAD_SCHEMA",
    "UpdateAcceptanceHead",
    "VerifiedTransitionExample",
    "fit_update_acceptance_head",
]
