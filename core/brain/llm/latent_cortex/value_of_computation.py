"""Typed value-of-computation policy for recurrent cognitive actions.

The execution controller chooses an episode configuration.  This policy owns
the finer question asked *inside* that episode: given the public state signals
available now, which cognitive operator is expected to buy the most verified
progress per unit of compute?

No private reasoning text enters the policy or its receipts.  Decisions use
bounded scalar signals, independently checked transition outcomes, and an
explicit executor inventory.  An action without a real executor is in the
vocabulary but is never selectable.  Sparse bootstrap exploration is named as
such; it is never presented as learned evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from core.brain.llm.latent_cortex.action_calibration import (
    ACTION_CALIBRATION_EVIDENCE_SCHEMA,
    ACTION_CALIBRATION_FINAL_VERIFIER_SCHEMA,
    ACTION_CALIBRATION_WORKER_ADMISSION_SCHEMA,
    ACTION_RESOURCE_DIMENSIONS,
    GLOBAL_BOUND_FAMILY_COUNT,
    MIN_CERTIFIED_TASKS_PER_ACTION,
)
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.runtime.file_read_gateway import read_stable_bytes

VALUE_OF_COMPUTATION_SCHEMA = "aura.rlc.value_of_computation.v1"
ACTION_EVIDENCE_SCHEMA = "aura.rlc.value_of_computation.evidence.v1"
ACTION_TRANSITION_SCHEMA = "aura.rlc.value_of_computation.transition.v1"
_ACTION_CALIBRATION_TRUST_ROOT_ENV = "AURA_RLC_ACTION_CALIBRATION_TRUST_ROOT"

MIN_ACTION_TRIALS = 8
MAX_ACTION_TRIALS = 100_000
_Z95 = 1.959963984540054
_EPSILON_COST = 1e-6

ACTION_VOCABULARY: tuple[OperationKind, ...] = tuple(OperationKind)
_ORDINARY_DECISION_MODES = frozenset(
    {
        "verified_stop",
        "verified_execute",
        "budget_stop",
        "budget_abstain",
        "budget_last_action",
        "irreducible_abstain",
        "bounded_explore",
        "measured",
        "bootstrap",
        "constraint_recovery",
    }
)
_CAMPAIGN_DECISION_MODE = "campaign_forced"


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"value-of-computation payload is not canonical: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    # not a failure: a value that is not a number is not one this can read.
    except ValueError:
        return False
    return True


def _finite(value: Any, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return parsed


def _bounded_text(value: Any, *, name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > limit
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError(f"{name} must be bounded printable text")
    return normalized


@dataclass(frozen=True, slots=True)
class ActionEvidence:
    """Legacy online moments retained only for bounded bootstrap exploration."""

    n: int = 0
    gain_sum: float = 0.0
    gain_sq_sum: float = 0.0
    cost_sum: float = 0.0
    cost_sq_sum: float = 0.0

    def __post_init__(self) -> None:
        if type(self.n) is not int or not 0 <= self.n <= MAX_ACTION_TRIALS:
            raise ValueError("action evidence count is out of bounds")
        for name, value, maximum in (
            ("gain_sum", self.gain_sum, 4.0 * max(1, self.n)),
            ("gain_sq_sum", self.gain_sq_sum, 16.0 * max(1, self.n)),
            ("cost_sum", self.cost_sum, 1.0 * max(1, self.n)),
            ("cost_sq_sum", self.cost_sq_sum, 1.0 * max(1, self.n)),
        ):
            parsed = _finite(value, name=name, minimum=-maximum, maximum=maximum)
            if (name.endswith("sq_sum") or name.startswith("cost_")) and parsed < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, parsed)
        if self.n == 0 and any(
            abs(value) > 1e-12
            for value in (
                self.gain_sum,
                self.gain_sq_sum,
                self.cost_sum,
                self.cost_sq_sum,
            )
        ):
            raise ValueError("empty action evidence cannot carry measurements")
        if self.n > 0:
            minimum_gain_square = self.gain_sum * self.gain_sum / self.n
            minimum_cost_square = self.cost_sum * self.cost_sum / self.n
            moment_tolerance = 1e-9 * max(
                1.0,
                self.gain_sq_sum,
                minimum_gain_square,
                self.cost_sum,
                self.cost_sq_sum,
                minimum_cost_square,
            )
            if self.gain_sq_sum + moment_tolerance < minimum_gain_square:
                raise ValueError("action gain moments are mathematically inconsistent")
            if self.cost_sq_sum + moment_tolerance < minimum_cost_square:
                raise ValueError("action cost moments are mathematically inconsistent")
            if self.cost_sq_sum > self.cost_sum + moment_tolerance:
                raise ValueError("action cost moments exceed bounded observations")

    @classmethod
    def from_dict(cls, value: Any) -> ActionEvidence:
        fields = {"n", "gain_sum", "gain_sq_sum", "cost_sum", "cost_sq_sum"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("action evidence fields differ")
        return cls(**{name: value[name] for name in fields})

    def to_dict(self) -> dict[str, int | float]:
        return {
            "n": self.n,
            "gain_sum": self.gain_sum,
            "gain_sq_sum": self.gain_sq_sum,
            "cost_sum": self.cost_sum,
            "cost_sq_sum": self.cost_sq_sum,
        }

    def append(self, *, gain: float, cost: float) -> ActionEvidence:
        if self.n >= MAX_ACTION_TRIALS:
            raise ValueError("action evidence trial cap is exhausted")
        measured_gain = _finite(gain, name="gain", minimum=-4.0, maximum=4.0)
        measured_cost = _finite(cost, name="cost", minimum=0.0, maximum=1.0)
        return ActionEvidence(
            n=self.n + 1,
            gain_sum=self.gain_sum + measured_gain,
            gain_sq_sum=self.gain_sq_sum + measured_gain * measured_gain,
            cost_sum=self.cost_sum + measured_cost,
            cost_sq_sum=self.cost_sq_sum + measured_cost * measured_cost,
        )

    @staticmethod
    def _bound(*, total: float, square_total: float, n: int, lower: bool) -> float:
        mean = total / n
        if n <= 1:
            return mean
        variance = max(0.0, (square_total - n * mean * mean) / (n - 1))
        radius = _Z95 * math.sqrt(variance / n)
        return mean - radius if lower else mean + radius

    def estimate(self) -> dict[str, int | float | bool]:
        if self.n == 0:
            return {
                "n": 0,
                "measured": False,
                "gain_mean": 0.0,
                "gain_lcb": -4.0,
                "cost_mean": 1.0,
                "cost_ucb": 1.0,
            }
        gain_mean = self.gain_sum / self.n
        cost_mean = self.cost_sum / self.n
        return {
            "n": self.n,
            "measured": False,
            "gain_mean": round(gain_mean, 8),
            "gain_lcb": round(
                max(
                    -4.0,
                    self._bound(
                        total=self.gain_sum,
                        square_total=self.gain_sq_sum,
                        n=self.n,
                        lower=True,
                    ),
                ),
                8,
            ),
            "cost_mean": round(cost_mean, 8),
            "cost_ucb": round(
                min(
                    1.0,
                    max(
                        _EPSILON_COST,
                        self._bound(
                            total=self.cost_sum,
                            square_total=self.cost_sq_sum,
                            n=self.n,
                            lower=False,
                        ),
                    ),
                ),
                8,
            ),
        }


@dataclass(frozen=True, slots=True)
class CertifiedActionEvidence:
    """Externally certified paired evidence for one cognitive action."""

    n: int
    unique_task_count: int
    measured: bool
    gain_mean: float
    gain_lcb: float
    gain_ucb: float
    cost_mean: float
    cost_ucb: float
    gain_bounds: dict[str, Any]
    cost_bounds: dict[str, Any]
    calibration_candidate_sha256: str
    policy_sha256: str

    @classmethod
    def from_dict(cls, value: Any) -> CertifiedActionEvidence:
        fields = {
            "n",
            "unique_task_count",
            "measured",
            "gain_mean",
            "gain_lcb",
            "gain_ucb",
            "cost_mean",
            "cost_ucb",
            "gain_bounds",
            "cost_bounds",
            "calibration_candidate_sha256",
            "policy_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("certified action evidence fields differ")
        n = value.get("n")
        unique = value.get("unique_task_count")
        measured = value.get("measured")
        if (
            type(n) is not int
            or not MIN_ACTION_TRIALS <= n <= MAX_ACTION_TRIALS
            or unique != n
            or type(measured) is not bool
            or not _is_sha256(value.get("calibration_candidate_sha256"))
            or not _is_sha256(value.get("policy_sha256"))
        ):
            raise ValueError("certified action evidence identity is invalid")
        gain_mean = _finite(
            value.get("gain_mean"),
            name="certified gain mean",
            minimum=-1.0,
            maximum=1.0,
        )
        gain_lcb = _finite(
            value.get("gain_lcb"),
            name="certified gain lcb",
            minimum=-1.0,
            maximum=1.0,
        )
        gain_ucb = _finite(
            value.get("gain_ucb"),
            name="certified gain ucb",
            minimum=-1.0,
            maximum=1.0,
        )
        cost_mean = _finite(
            value.get("cost_mean"),
            name="certified cost mean",
            minimum=0.0,
            maximum=1.0,
        )
        cost_ucb = _finite(
            value.get("cost_ucb"),
            name="certified cost ucb",
            minimum=_EPSILON_COST,
            maximum=1.0,
        )
        gain_bounds = value.get("gain_bounds")
        cost_bounds = value.get("cost_bounds")
        gain_bound_fields = {
            "method",
            "family_count",
            "family_alpha",
            "component_alpha",
            "simultaneous_coverage_lower",
            "lower",
            "upper",
            "certified",
        }
        cost_bound_fields = {
            "method",
            "family_count",
            "family_alpha",
            "bounded_interval",
            "normalization",
            "dimensions",
        }

        def rational_number(raw: Any) -> float:
            if (
                not isinstance(raw, Mapping)
                or set(raw) != {"numerator", "denominator"}
                or type(raw.get("numerator")) is not int
                or type(raw.get("denominator")) is not int
                or raw["denominator"] <= 0
            ):
                raise ValueError("certified action evidence rational bound is invalid")
            return raw["numerator"] / raw["denominator"]

        if (
            gain_lcb > gain_mean
            or gain_mean > gain_ucb
            or cost_mean > cost_ucb
            or not isinstance(gain_bounds, Mapping)
            or not isinstance(cost_bounds, Mapping)
            or set(gain_bounds) != gain_bound_fields
            or set(cost_bounds) != cost_bound_fields
            or gain_bounds.get("certified") is not True
            or gain_bounds.get("family_count") != GLOBAL_BOUND_FAMILY_COUNT
            or cost_bounds.get("family_count") != GLOBAL_BOUND_FAMILY_COUNT
            or gain_bounds.get("method") != "simultaneous rational Clopper-Pearson contrast bounds"
            or cost_bounds.get("method") != "simultaneous Hoeffding upper bound"
            or cost_bounds.get("bounded_interval") != [0.0, 1.0]
            or cost_bounds.get("normalization")
            != "max fraction of preregistered action-resource caps"
            or cost_bounds.get("dimensions") != list(ACTION_RESOURCE_DIMENSIONS)
        ):
            raise ValueError("certified action evidence bounds are invalid")
        bound_lower = rational_number(gain_bounds["lower"])
        bound_upper = rational_number(gain_bounds["upper"])
        rational_number(gain_bounds["family_alpha"])
        rational_number(gain_bounds["component_alpha"])
        rational_number(gain_bounds["simultaneous_coverage_lower"])
        rational_number(cost_bounds["family_alpha"])
        if abs(gain_lcb - bound_lower) > 2e-12 or abs(gain_ucb - bound_upper) > 2e-12:
            raise ValueError("certified action evidence rounded bounds differ")
        expected_measured = n >= MIN_CERTIFIED_TASKS_PER_ACTION
        if measured is not expected_measured:
            raise ValueError("certified action evidence measured status is invalid")
        return cls(
            n=n,
            unique_task_count=unique,
            measured=measured,
            gain_mean=gain_mean,
            gain_lcb=gain_lcb,
            gain_ucb=gain_ucb,
            cost_mean=cost_mean,
            cost_ucb=cost_ucb,
            gain_bounds=dict(gain_bounds),
            cost_bounds=dict(cost_bounds),
            calibration_candidate_sha256=value["calibration_candidate_sha256"],
            policy_sha256=value["policy_sha256"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "unique_task_count": self.unique_task_count,
            "measured": self.measured,
            "gain_mean": self.gain_mean,
            "gain_lcb": self.gain_lcb,
            "gain_ucb": self.gain_ucb,
            "cost_mean": self.cost_mean,
            "cost_ucb": self.cost_ucb,
            "gain_bounds": dict(self.gain_bounds),
            "cost_bounds": dict(self.cost_bounds),
            "calibration_candidate_sha256": self.calibration_candidate_sha256,
            "policy_sha256": self.policy_sha256,
        }

    def estimate(self) -> dict[str, int | float | bool]:
        return {
            "n": self.n,
            "measured": self.measured,
            "gain_mean": self.gain_mean,
            "gain_lcb": self.gain_lcb,
            "cost_mean": self.cost_mean,
            "cost_ucb": self.cost_ucb,
        }


def build_evidence_snapshot(
    *,
    bucket: str,
    cells: Mapping[OperationKind | str, ActionEvidence | Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the exact bounded evidence object sent to the worker."""

    normalized_bucket = _bounded_text(bucket, name="bucket", limit=160)
    normalized: dict[str, dict[str, int | float]] = {}
    for raw_action, raw_cell in cells.items():
        try:
            action = (
                raw_action if isinstance(raw_action, OperationKind) else OperationKind(raw_action)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown cognitive action: {raw_action!r}") from exc
        cell = (
            raw_cell if isinstance(raw_cell, ActionEvidence) else ActionEvidence.from_dict(raw_cell)
        )
        normalized[action.value] = cell.to_dict()
    payload = {
        "schema": ACTION_EVIDENCE_SCHEMA,
        "bucket": normalized_bucket,
        "cells": {name: normalized[name] for name in sorted(normalized)},
    }
    return {**payload, "snapshot_sha256": _canonical_sha256(payload)}


def _validate_certified_admission(
    value: Any,
    *,
    bucket: str,
    candidate_sha256: str,
    policy_sha256: str,
    candidate_cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    fields = {
        "schema",
        "campaign_name",
        "policy_validated_at_unix",
        "policy_document",
        "final_verifier_payload",
        "final_verifier_attestation",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != ACTION_CALIBRATION_WORKER_ADMISSION_SCHEMA
    ):
        raise ValueError("certified action evidence admission fields differ")
    campaign_name = _bounded_text(
        value.get("campaign_name"),
        name="campaign_name",
        limit=200,
    )
    policy_validated_at_unix = value.get("policy_validated_at_unix")
    final_attestation = value.get("final_verifier_attestation")
    signed_payload = (
        final_attestation.get("signed_payload") if isinstance(final_attestation, Mapping) else None
    )
    if (
        type(policy_validated_at_unix) is not int
        or policy_validated_at_unix <= 0
        or not isinstance(signed_payload, Mapping)
        or signed_payload.get("signed_at_unix") != policy_validated_at_unix
    ):
        raise ValueError("certified action evidence admission time is invalid")
    root_path = os.environ.get(_ACTION_CALIBRATION_TRUST_ROOT_ENV)
    if not isinstance(root_path, str) or not root_path.strip():
        raise ValueError("certified action evidence trust root is not configured")
    try:
        trusted_root = read_stable_bytes(
            Path(root_path).expanduser(),
            max_bytes=64 * 1024,
        )
        from core.brain.llm.latent_cortex.campaign_trust import (
            EVIDENCE_VERIFIER,
            validate_campaign_trust_policy,
            verify_role_attestation,
        )

        policy = validate_campaign_trust_policy(
            value.get("policy_document"),
            trusted_root_public_key_pem=trusted_root,
            expected_campaign_name=campaign_name,
            expected_policy_sha256=policy_sha256,
            now_unix=policy_validated_at_unix,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("certified action evidence trust admission failed") from exc
    final_payload = value.get("final_verifier_payload")
    final_fields = {
        "schema",
        "accepted",
        "candidate_sha256",
        "calibration_bucket",
        "plan_sha256",
        "policy_sha256",
        "campaign_manifest_sha256",
        "journal_head_sha256",
        "journal_event_count",
        "observations_sha256",
        "cells_sha256",
        "pair_count",
        "execution_count",
        "frontier_claim_eligible",
    }
    if (
        not isinstance(final_payload, Mapping)
        or set(final_payload) != final_fields
        or final_payload.get("schema") != ACTION_CALIBRATION_FINAL_VERIFIER_SCHEMA
        or final_payload.get("accepted") is not True
        or final_payload.get("candidate_sha256") != candidate_sha256
        or final_payload.get("calibration_bucket") != bucket
        or final_payload.get("policy_sha256") != policy.policy_sha256
        or final_payload.get("cells_sha256") != _canonical_sha256(candidate_cells)
        or final_payload.get("frontier_claim_eligible") is not False
        or type(final_payload.get("pair_count")) is not int
        or final_payload["pair_count"] < len(ACTION_VOCABULARY) * MIN_ACTION_TRIALS
        or final_payload.get("execution_count") != final_payload["pair_count"] * 2
        or type(final_payload.get("journal_event_count")) is not int
        or final_payload["journal_event_count"] < final_payload["execution_count"] * 2
        or any(
            not _is_sha256(final_payload.get(name))
            for name in (
                "candidate_sha256",
                "plan_sha256",
                "policy_sha256",
                "campaign_manifest_sha256",
                "journal_head_sha256",
                "observations_sha256",
                "cells_sha256",
            )
        )
    ):
        raise ValueError("certified action evidence final verdict is invalid")
    try:
        verify_role_attestation(
            policy,
            value.get("final_verifier_attestation"),
            role=EVIDENCE_VERIFIER,
            expected_payload=final_payload,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("certified action evidence verifier attestation failed") from exc
    return {
        "schema": ACTION_CALIBRATION_WORKER_ADMISSION_SCHEMA,
        "campaign_name": campaign_name,
        "policy_validated_at_unix": policy_validated_at_unix,
        "policy_document": dict(policy.document),
        "final_verifier_payload": dict(final_payload),
        "final_verifier_attestation": dict(value["final_verifier_attestation"]),
    }


def validate_evidence_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("action evidence snapshot fields differ")
    schema = value.get("schema")
    if schema == ACTION_CALIBRATION_EVIDENCE_SCHEMA:
        fields = {
            "schema",
            "bucket",
            "candidate_sha256",
            "policy_sha256",
            "admission",
            "cells",
            "snapshot_sha256",
        }
        if set(value) != fields:
            raise ValueError("certified action evidence snapshot fields differ")
        bucket = _bounded_text(value.get("bucket"), name="bucket", limit=160)
        candidate_sha256 = value.get("candidate_sha256")
        policy_sha256 = value.get("policy_sha256")
        raw_cells = value.get("cells")
        if (
            not _is_sha256(candidate_sha256)
            or not _is_sha256(policy_sha256)
            or not isinstance(raw_cells, Mapping)
            or len(raw_cells) > len(ACTION_VOCABULARY)
        ):
            raise ValueError("certified action evidence snapshot is invalid")
        certified_cells: dict[str, dict[str, Any]] = {}
        candidate_cells: dict[str, dict[str, Any]] = {}
        for raw_action, raw_cell in raw_cells.items():
            try:
                action = OperationKind(raw_action)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unknown certified action evidence cell: {raw_action!r}") from exc
            cell = CertifiedActionEvidence.from_dict(raw_cell)
            if (
                cell.calibration_candidate_sha256 != candidate_sha256
                or cell.policy_sha256 != policy_sha256
            ):
                raise ValueError("certified action evidence lineage does not match snapshot")
            certified_cells[action.value] = cell.to_dict()
            candidate_cells[action.value] = {
                name: item
                for name, item in cell.to_dict().items()
                if name
                not in {
                    "calibration_candidate_sha256",
                    "policy_sha256",
                }
            }
        admission = _validate_certified_admission(
            value.get("admission"),
            bucket=bucket,
            candidate_sha256=candidate_sha256,
            policy_sha256=policy_sha256,
            candidate_cells=candidate_cells,
        )
        payload = {
            "schema": ACTION_CALIBRATION_EVIDENCE_SCHEMA,
            "bucket": bucket,
            "candidate_sha256": candidate_sha256,
            "policy_sha256": policy_sha256,
            "admission": admission,
            "cells": {name: certified_cells[name] for name in sorted(certified_cells)},
        }
        expected = _canonical_sha256(payload)
        if value.get("snapshot_sha256") != expected:
            raise ValueError("certified action evidence snapshot digest does not match")
        return {**payload, "snapshot_sha256": expected}
    fields = {"schema", "bucket", "cells", "snapshot_sha256"}
    if set(value) != fields:
        raise ValueError("action evidence snapshot fields differ")
    if schema != ACTION_EVIDENCE_SCHEMA:
        raise ValueError("action evidence snapshot schema is invalid")
    bucket = _bounded_text(value.get("bucket"), name="bucket", limit=160)
    raw_cells = value.get("cells")
    if not isinstance(raw_cells, Mapping) or len(raw_cells) > len(ACTION_VOCABULARY):
        raise ValueError("action evidence snapshot cells are invalid")
    cells: dict[str, dict[str, int | float]] = {}
    for raw_action, raw_cell in raw_cells.items():
        try:
            action = OperationKind(raw_action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown action evidence cell: {raw_action!r}") from exc
        cells[action.value] = ActionEvidence.from_dict(raw_cell).to_dict()
    payload = {
        "schema": ACTION_EVIDENCE_SCHEMA,
        "bucket": bucket,
        "cells": {name: cells[name] for name in sorted(cells)},
    }
    expected = _canonical_sha256(payload)
    if value.get("snapshot_sha256") != expected:
        raise ValueError("action evidence snapshot digest does not match")
    return {**payload, "snapshot_sha256": expected}


@dataclass(frozen=True, slots=True)
class CognitiveStateSignal:
    """Public, bounded features available at one recurrent decision point."""

    step_index: int
    max_steps: int
    neural_steps: int
    min_neural_steps: int
    active_branches: int
    total_branches: int
    residual: float
    residual_delta: float
    verifier_score: float | None
    verifier_delta: float | None
    disagreement: float
    uncertainty: float
    budget_remaining_fraction: float
    has_memory: bool
    has_evidence: bool
    has_verifier: bool
    has_savepoint: bool
    can_execute: bool
    answer_verified: bool
    irreducible_uncertainty: bool
    pending_constraint_action: OperationKind | None = None
    omitted_action_count: int = 0
    previously_selected: tuple[OperationKind, ...] = ()

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("step_index", self.step_index, 0),
            ("max_steps", self.max_steps, 1),
            ("neural_steps", self.neural_steps, 0),
            ("min_neural_steps", self.min_neural_steps, 1),
            ("active_branches", self.active_branches, 0),
            ("total_branches", self.total_branches, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} is invalid")
        if self.step_index > self.max_steps:
            raise ValueError("step_index exceeds max_steps")
        if self.neural_steps > self.max_steps or self.min_neural_steps > self.max_steps:
            raise ValueError("neural step bounds exceed max_steps")
        if self.active_branches > self.total_branches:
            raise ValueError("active branch count exceeds total")
        for name, value in (
            ("residual", self.residual),
            ("residual_delta", self.residual_delta),
            ("disagreement", self.disagreement),
            ("uncertainty", self.uncertainty),
            ("budget_remaining_fraction", self.budget_remaining_fraction),
        ):
            object.__setattr__(
                self,
                name,
                _finite(
                    value,
                    name=name,
                    minimum=-1.0 if name == "residual_delta" else 0.0,
                    maximum=1.0,
                ),
            )
        for name in (
            "has_memory",
            "has_evidence",
            "has_verifier",
            "has_savepoint",
            "can_execute",
            "answer_verified",
            "irreducible_uncertainty",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("verifier_score", "verifier_delta"):
            value = getattr(self, name)
            if value is not None:
                minimum = -1.0 if name == "verifier_delta" else 0.0
                object.__setattr__(
                    self,
                    name,
                    _finite(value, name=name, minimum=minimum, maximum=1.0),
                )
        if any(not isinstance(action, OperationKind) for action in self.previously_selected):
            raise ValueError("previously_selected contains an invalid action")
        if self.pending_constraint_action is not None and self.pending_constraint_action not in {
            OperationKind.FALSIFY,
            OperationKind.CHECK_ASSUMPTION,
        }:
            raise ValueError("pending_constraint_action is invalid")
        if (
            type(self.omitted_action_count) is not int
            or not 0 <= self.omitted_action_count <= self.step_index
        ):
            raise ValueError("omitted_action_count is invalid")
        if len(self.previously_selected) + self.omitted_action_count != self.step_index:
            raise ValueError("previously_selected must cover every prior action step")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "max_steps": self.max_steps,
            "neural_steps": self.neural_steps,
            "min_neural_steps": self.min_neural_steps,
            "active_branches": self.active_branches,
            "total_branches": self.total_branches,
            "residual": self.residual,
            "residual_delta": self.residual_delta,
            "verifier_score": self.verifier_score,
            "verifier_delta": self.verifier_delta,
            "disagreement": self.disagreement,
            "uncertainty": self.uncertainty,
            "budget_remaining_fraction": self.budget_remaining_fraction,
            "has_memory": self.has_memory,
            "has_evidence": self.has_evidence,
            "has_verifier": self.has_verifier,
            "has_savepoint": self.has_savepoint,
            "can_execute": self.can_execute,
            "answer_verified": self.answer_verified,
            "irreducible_uncertainty": self.irreducible_uncertainty,
            "pending_constraint_action": (
                self.pending_constraint_action.value
                if self.pending_constraint_action is not None
                else None
            ),
            "omitted_action_count": self.omitted_action_count,
            "previously_selected": [action.value for action in self.previously_selected],
        }

    @classmethod
    def from_dict(cls, value: Any) -> CognitiveStateSignal:
        fields = {
            "step_index",
            "max_steps",
            "neural_steps",
            "min_neural_steps",
            "active_branches",
            "total_branches",
            "residual",
            "residual_delta",
            "verifier_score",
            "verifier_delta",
            "disagreement",
            "uncertainty",
            "budget_remaining_fraction",
            "has_memory",
            "has_evidence",
            "has_verifier",
            "has_savepoint",
            "can_execute",
            "answer_verified",
            "irreducible_uncertainty",
            "pending_constraint_action",
            "omitted_action_count",
            "previously_selected",
        }
        accepted_fields = {
            frozenset(fields),
            frozenset(fields - {"omitted_action_count"}),
            frozenset(fields - {"pending_constraint_action"}),
            frozenset(fields - {"omitted_action_count", "pending_constraint_action"}),
        }
        actual_fields = frozenset(value) if isinstance(value, Mapping) else frozenset()
        if not isinstance(value, Mapping) or actual_fields not in accepted_fields:
            raise ValueError("cognitive state signal fields differ")
        normalized_value = dict(value)
        normalized_value.setdefault("omitted_action_count", 0)
        normalized_value.setdefault("pending_constraint_action", None)
        previous = normalized_value.get("previously_selected")
        if not isinstance(previous, list) or len(previous) > int(
            normalized_value.get("max_steps") or 0
        ):
            raise ValueError("cognitive state previous actions are invalid")
        try:
            previous_actions = tuple(OperationKind(item) for item in previous)
        except (TypeError, ValueError) as exc:
            raise ValueError("cognitive state previous actions are invalid") from exc
        pending_raw = normalized_value.get("pending_constraint_action")
        try:
            pending_action = OperationKind(pending_raw) if pending_raw is not None else None
        except (TypeError, ValueError) as exc:
            raise ValueError("cognitive state pending constraint is invalid") from exc
        return cls(
            **{
                name: normalized_value[name]
                for name in fields - {"previously_selected", "pending_constraint_action"}
            },
            pending_constraint_action=pending_action,
            previously_selected=previous_actions,
        )


_BASE_COST: dict[OperationKind, float] = {
    OperationKind.DECOMPOSE: 0.10,
    OperationKind.BLIND_RESOLVE: 0.14,
    OperationKind.BRANCH: 0.22,
    OperationKind.SEARCH_MEMORY: 0.05,
    OperationKind.RETRIEVE_EVIDENCE: 0.08,
    OperationKind.EXECUTE: 0.18,
    OperationKind.SIMULATE: 0.16,
    OperationKind.FALSIFY: 0.13,
    OperationKind.CHECK_ASSUMPTION: 0.12,
    OperationKind.REGENERATE_FROM_PREFIX: 0.17,
    OperationKind.FORMALIZE: 0.15,
    OperationKind.COMPARE: 0.10,
    OperationKind.BACKTRACK: 0.03,
    OperationKind.COMPRESS_STATE: 0.04,
    OperationKind.ANSWER: 0.01,
    OperationKind.ABSTAIN: 0.01,
}


def action_cost_estimate(
    evidence_snapshot: Mapping[str, Any],
    action: OperationKind | str,
) -> dict[str, Any]:
    """Return the conservative measured or declared bootstrap action cost."""

    snapshot = validate_evidence_snapshot(evidence_snapshot)
    try:
        operation = action if isinstance(action, OperationKind) else OperationKind(action)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown cognitive action: {action!r}") from exc
    raw = snapshot["cells"].get(operation.value)
    if raw is None:
        cell: ActionEvidence | CertifiedActionEvidence = ActionEvidence()
    elif snapshot["schema"] == ACTION_CALIBRATION_EVIDENCE_SCHEMA:
        cell = CertifiedActionEvidence.from_dict(raw)
    else:
        cell = ActionEvidence.from_dict(raw)
    estimate = cell.estimate()
    measured = bool(estimate["measured"])
    return {
        "action": operation.value,
        "n": int(estimate["n"]),
        "measured": measured,
        "basis": "measured_cost_ucb" if measured else "declared_bootstrap_cost",
        "gain_basis": "measured_gain_lcb" if measured else "unmeasured",
        "gain_lower_bound": (round(float(estimate["gain_lcb"]), 8) if measured else None),
        "cost_upper_bound": round(
            float(estimate["cost_ucb"]) if measured else _BASE_COST[operation],
            8,
        ),
        "evidence_snapshot_sha256": snapshot["snapshot_sha256"],
    }


_TERMINAL_ACTIONS: Final = frozenset(
    {OperationKind.ANSWER, OperationKind.ABSTAIN}
)


def _bootstrap_gain(action: OperationKind, state: CognitiveStateSignal) -> float:
    """Conservative structural prior used only while measurements are sparse."""

    progress = state.step_index / max(1, state.max_steps)
    regression = max(0.0, -float(state.verifier_delta or 0.0))
    stagnation = max(0.0, state.residual - max(0.0, state.residual_delta))
    values = {
        OperationKind.DECOMPOSE: 0.32 if state.step_index == 0 else 0.08,
        OperationKind.BLIND_RESOLVE: 0.20 + 0.12 * state.uncertainty,
        OperationKind.BRANCH: 0.18 + 0.35 * state.disagreement,
        OperationKind.SEARCH_MEMORY: 0.28 if state.has_memory else -1.0,
        OperationKind.RETRIEVE_EVIDENCE: 0.30 if state.has_evidence else -1.0,
        OperationKind.EXECUTE: 0.34 if state.can_execute else -1.0,
        OperationKind.SIMULATE: 0.18 + 0.20 * state.uncertainty,
        OperationKind.FALSIFY: 0.16 + 0.34 * state.disagreement,
        OperationKind.CHECK_ASSUMPTION: 0.18 + 0.30 * state.uncertainty,
        OperationKind.REGENERATE_FROM_PREFIX: 0.08 + 0.55 * regression,
        OperationKind.FORMALIZE: 0.16 + 0.16 * state.uncertainty,
        OperationKind.COMPARE: 0.12 + 0.40 * state.disagreement,
        OperationKind.BACKTRACK: 0.06 + 0.65 * max(regression, stagnation),
        OperationKind.COMPRESS_STATE: 0.04 + 0.22 * progress,
        OperationKind.ANSWER: 0.75 if state.answer_verified else 0.04 * progress,
        OperationKind.ABSTAIN: (
            0.70 if state.irreducible_uncertainty else 0.25 * state.uncertainty * progress
        ),
    }
    return values[action]


def feasible_actions(
    state: CognitiveStateSignal,
    *,
    executors: tuple[OperationKind, ...],
) -> tuple[OperationKind, ...]:
    available = set(executors)
    if state.pending_constraint_action is not None:
        if state.pending_constraint_action not in available or not state.has_verifier:
            return ()
        return (state.pending_constraint_action,)
    neural_actions = {
        OperationKind.DECOMPOSE,
        OperationKind.BLIND_RESOLVE,
        OperationKind.BRANCH,
        OperationKind.SEARCH_MEMORY,
        OperationKind.RETRIEVE_EVIDENCE,
        OperationKind.SIMULATE,
        OperationKind.FALSIFY,
        OperationKind.CHECK_ASSUMPTION,
        OperationKind.REGENERATE_FROM_PREFIX,
        OperationKind.FORMALIZE,
    }
    neural_floor_pending = state.neural_steps < state.min_neural_steps
    feasible: list[OperationKind] = []
    for action in ACTION_VOCABULARY:
        if action not in available:
            continue
        if neural_floor_pending and action not in neural_actions:
            continue
        allowed = True
        if action is OperationKind.DECOMPOSE:
            allowed = state.step_index <= 1 and action not in state.previously_selected
        elif action is OperationKind.BRANCH:
            allowed = state.active_branches >= 2
        elif action is OperationKind.SEARCH_MEMORY:
            allowed = state.has_memory and action not in state.previously_selected
        elif action is OperationKind.RETRIEVE_EVIDENCE:
            allowed = state.has_evidence and action not in state.previously_selected
        elif action is OperationKind.EXECUTE:
            allowed = state.can_execute
        elif action in {OperationKind.FALSIFY, OperationKind.CHECK_ASSUMPTION}:
            allowed = state.has_verifier
        elif action in {
            OperationKind.REGENERATE_FROM_PREFIX,
            OperationKind.BACKTRACK,
        }:
            allowed = state.has_savepoint and (
                float(state.verifier_delta or 0.0) < -1e-9 or state.residual_delta < -1e-9
            )
        elif action is OperationKind.COMPARE:
            allowed = state.active_branches >= 2
        elif action is OperationKind.COMPRESS_STATE:
            allowed = state.step_index >= 2
        elif action is OperationKind.ANSWER:
            allowed = state.step_index >= 2
        elif action is OperationKind.ABSTAIN:
            allowed = state.step_index >= 2
        elif action in {
            OperationKind.SIMULATE,
            OperationKind.FORMALIZE,
        }:
            allowed = state.active_branches >= 1
        elif action is OperationKind.BLIND_RESOLVE:
            allowed = state.active_branches == 1
        if allowed:
            feasible.append(action)
    return tuple(feasible)


class ValueOfComputationPolicy:
    """Select one executable action from measured gain-per-cost evidence."""

    def __init__(self, evidence_snapshot: Mapping[str, Any]) -> None:
        snapshot = validate_evidence_snapshot(evidence_snapshot)
        self.snapshot = snapshot
        self.bucket = snapshot["bucket"]
        cell_type = (
            CertifiedActionEvidence
            if snapshot["schema"] == ACTION_CALIBRATION_EVIDENCE_SCHEMA
            else ActionEvidence
        )
        self.cells: dict[
            OperationKind,
            ActionEvidence | CertifiedActionEvidence,
        ] = {
            OperationKind(name): cell_type.from_dict(cell)
            for name, cell in snapshot["cells"].items()
        }

    def choose(
        self,
        state: CognitiveStateSignal,
        *,
        executors: tuple[OperationKind, ...],
    ) -> dict[str, Any]:
        feasible = feasible_actions(state, executors=executors)
        if not feasible:
            raise ValueError("no executable cognitive action is feasible")

        # Terminal rules are explicit, preregistered, and dominate exploration.
        if state.pending_constraint_action is not None:
            chosen = state.pending_constraint_action
            mode = "constraint_recovery"
        elif OperationKind.EXECUTE in feasible and state.can_execute and state.answer_verified:
            chosen = OperationKind.EXECUTE
            mode = "verified_execute"
        elif OperationKind.ANSWER in feasible and state.answer_verified:
            chosen = OperationKind.ANSWER
            mode = "verified_stop"
        elif state.budget_remaining_fraction <= 0.03:
            if OperationKind.ANSWER in feasible and state.uncertainty <= 0.35:
                chosen = OperationKind.ANSWER
                mode = "budget_stop"
            elif OperationKind.ABSTAIN in feasible:
                chosen = OperationKind.ABSTAIN
                mode = "budget_abstain"
            else:
                chosen = feasible[0]
                mode = "budget_last_action"
        elif state.irreducible_uncertainty and OperationKind.ABSTAIN in feasible:
            chosen = OperationKind.ABSTAIN
            mode = "irreducible_abstain"
        else:
            scored: list[tuple[float, str, OperationKind, dict[str, Any]]] = []
            for action in feasible:
                cell = self.cells.get(action, ActionEvidence())
                estimate = cell.estimate()
                measured = bool(estimate["measured"])
                gain = float(estimate["gain_lcb"]) if measured else _bootstrap_gain(action, state)
                cost = float(estimate["cost_ucb"]) if measured else _BASE_COST[action]
                value = gain / max(_EPSILON_COST, cost)
                scored.append(
                    (
                        value,
                        action.value,
                        action,
                        {
                            **estimate,
                            "basis": "measured_lcb_per_cost_ucb" if measured else "bootstrap_prior",
                            "gain_used": round(gain, 8),
                            "cost_used": round(cost, 8),
                            "value": round(value, 8),
                        },
                    )
                )

            # Stopping is only preferred when continuing is worthless.
            #
            # ANSWER and ABSTAIN cost ~0.01 to execute but END the episode, so
            # ranking them by value-per-cost inflated them ~100x against
            # anything that does real work: at step 2 of 8 with uncertainty
            # 0.5, abstain scored 3.1 against check_assumption's 2.75, and
            # every full-stack episode halted at the floor depth of 2.
            #
            # Charging them the forfeited budget instead is WRONG, because
            # gain/cost is not monotonic in cost once gain can be negative:
            # with measured evidence of -0.05 for every action, a larger
            # denominator makes the score LESS negative, so a bigger forfeit
            # made stopping win harder in exactly the regime where continuing
            # is measurably worthless.
            #
            # The economics without that trap: an episode should keep going
            # while any continuing action still has positive expected value,
            # and may stop once none does. This needs no new constant, keeps
            # the bootstrap case spending depth, and leaves measured evidence
            # free to halt early -- which is the learned stop head's whole
            # contract.
            best_continuing = max(
                (row[0] for row in scored if row[2] not in _TERMINAL_ACTIONS),
                default=0.0,
            )
            if best_continuing > 0.0:
                scored = [
                    (
                        row[0] if row[2] not in _TERMINAL_ACTIONS else float("-inf"),
                        *row[1:],
                    )
                    for row in scored
                ]

            # Sparse deterministic exploration only among non-terminal actions.
            under_sampled = sorted(
                (
                    action
                    for action in feasible
                    if action not in {OperationKind.ANSWER, OperationKind.ABSTAIN}
                    and not bool(self.cells.get(action, ActionEvidence()).estimate()["measured"])
                ),
                key=lambda action: (
                    self.cells.get(action, ActionEvidence()).n,
                    action.value,
                ),
            )
            if under_sampled and (state.step_index + 1) % 4 == 0:
                chosen = under_sampled[0]
                mode = "bounded_explore"
            else:
                _, _, chosen, _ = max(scored, key=lambda row: (row[0], row[1]))
                chosen_cell = self.cells.get(chosen, ActionEvidence())
                mode = "measured" if chosen_cell.estimate()["measured"] else "bootstrap"

        selected_cell = self.cells.get(chosen, ActionEvidence())
        selected_estimate = selected_cell.estimate()
        gain_used = (
            float(selected_estimate["gain_lcb"])
            if selected_estimate["measured"]
            else _bootstrap_gain(chosen, state)
        )
        cost_used = (
            float(selected_estimate["cost_ucb"])
            if selected_estimate["measured"]
            else _BASE_COST[chosen]
        )
        decision = {
            "schema": VALUE_OF_COMPUTATION_SCHEMA,
            "bucket": self.bucket,
            "snapshot_sha256": self.snapshot["snapshot_sha256"],
            "step_index": state.step_index,
            "action": chosen.value,
            "mode": mode,
            "feasible_actions": [action.value for action in feasible],
            "evidence": {
                **selected_estimate,
                "basis": (
                    "measured_lcb_per_cost_ucb"
                    if selected_estimate["measured"]
                    else "bootstrap_prior"
                ),
                "gain_used": round(gain_used, 8),
                "cost_used": round(cost_used, 8),
                "value": round(gain_used / max(_EPSILON_COST, cost_used), 8),
            },
        }
        decision["decision_sha256"] = _canonical_sha256(decision)
        return decision

    def choose_forced(
        self,
        state: CognitiveStateSignal,
        *,
        executors: tuple[OperationKind, ...],
        action: OperationKind,
    ) -> dict[str, Any]:
        """Construct the exact decision for an authenticated causal intervention.

        Scientific interventions intentionally bypass the observational
        feasibility policy, but never the executor inventory.  This method is
        not reachable from ordinary policy selection; independent trace replay
        only accepts its mode alongside a verified campaign intervention.
        """

        if not isinstance(action, OperationKind) or action not in executors:
            raise ValueError("forced cognitive action has no resident executor")
        if len(set(executors)) != len(executors):
            raise ValueError("forced cognitive action executor inventory is invalid")
        selected_cell = self.cells.get(action, ActionEvidence())
        selected_estimate = selected_cell.estimate()
        gain_used = (
            float(selected_estimate["gain_lcb"])
            if selected_estimate["measured"]
            else _bootstrap_gain(action, state)
        )
        cost_used = (
            float(selected_estimate["cost_ucb"])
            if selected_estimate["measured"]
            else _BASE_COST[action]
        )
        decision = {
            "schema": VALUE_OF_COMPUTATION_SCHEMA,
            "bucket": self.bucket,
            "snapshot_sha256": self.snapshot["snapshot_sha256"],
            "step_index": state.step_index,
            "action": action.value,
            "mode": "campaign_forced",
            "feasible_actions": [action.value],
            "evidence": {
                **selected_estimate,
                "basis": (
                    "measured_lcb_per_cost_ucb"
                    if selected_estimate["measured"]
                    else "bootstrap_prior"
                ),
                "gain_used": round(gain_used, 8),
                "cost_used": round(cost_used, 8),
                "value": round(gain_used / max(_EPSILON_COST, cost_used), 8),
            },
        }
        decision["decision_sha256"] = _canonical_sha256(decision)
        return decision


def transition_reward(
    *,
    verified_delta: float,
    information_gain: float,
    diversity_gain: float,
    unsupported_confidence: float,
    cost: float,
) -> dict[str, Any]:
    """Measure Spark's progress reward from public before/after signals."""

    verified = _finite(
        verified_delta,
        name="verified_delta",
        minimum=-1.0,
        maximum=1.0,
    )
    information = _finite(
        information_gain,
        name="information_gain",
        minimum=-1.0,
        maximum=1.0,
    )
    diversity = _finite(
        diversity_gain,
        name="diversity_gain",
        minimum=-1.0,
        maximum=1.0,
    )
    gaming = _finite(
        unsupported_confidence,
        name="unsupported_confidence",
        minimum=0.0,
        maximum=1.0,
    )
    measured_cost = _finite(cost, name="cost", minimum=0.0, maximum=1.0)
    gain = verified + 0.35 * information + 0.20 * diversity - 0.55 * gaming
    reward = gain - 0.20 * measured_cost
    return {
        "verified_delta": round(verified, 8),
        "information_gain": round(information, 8),
        "diversity_gain": round(diversity, 8),
        "unsupported_confidence": round(gaming, 8),
        "cost": round(measured_cost, 8),
        "gain": round(gain, 8),
        "reward": round(reward, 8),
    }


def validate_action_decision(value: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "bucket",
        "snapshot_sha256",
        "step_index",
        "action",
        "mode",
        "feasible_actions",
        "evidence",
        "decision_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("action decision fields differ")
    normalized = dict(value)
    if normalized.get("schema") != VALUE_OF_COMPUTATION_SCHEMA:
        raise ValueError("action decision schema is invalid")
    normalized["bucket"] = _bounded_text(normalized.get("bucket"), name="bucket", limit=160)
    for name in ("snapshot_sha256", "decision_sha256"):
        digest = normalized.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"action decision {name} is invalid")
    if type(normalized.get("step_index")) is not int or normalized["step_index"] < 0:
        raise ValueError("action decision step_index is invalid")
    try:
        selected = OperationKind(normalized.get("action"))
    except (TypeError, ValueError) as exc:
        raise ValueError("action decision selected action is invalid") from exc
    feasible_raw = normalized.get("feasible_actions")
    if (
        not isinstance(feasible_raw, list)
        or not feasible_raw
        or len(feasible_raw) > len(ACTION_VOCABULARY)
    ):
        raise ValueError("action decision feasible actions are invalid")
    try:
        feasible = tuple(OperationKind(item) for item in feasible_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("action decision feasible actions are invalid") from exc
    if len(set(feasible)) != len(feasible) or selected not in feasible:
        raise ValueError("action decision selected an infeasible or duplicate action")
    normalized["action"] = selected.value
    normalized["feasible_actions"] = [action.value for action in feasible]
    normalized["mode"] = _bounded_text(normalized.get("mode"), name="mode", limit=32)
    if normalized["mode"] not in _ORDINARY_DECISION_MODES | {_CAMPAIGN_DECISION_MODE}:
        raise ValueError("action decision mode is unsupported")
    evidence_fields = {
        "n",
        "measured",
        "gain_mean",
        "gain_lcb",
        "cost_mean",
        "cost_ucb",
        "basis",
        "gain_used",
        "cost_used",
        "value",
    }
    evidence = normalized.get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != evidence_fields:
        raise ValueError("action decision evidence fields differ")
    evidence = dict(evidence)
    if type(evidence.get("n")) is not int or not 0 <= evidence["n"] <= MAX_ACTION_TRIALS:
        raise ValueError("action decision evidence count is invalid")
    if type(evidence.get("measured")) is not bool:
        raise ValueError("action decision measured flag is invalid")
    if evidence["measured"] and evidence["n"] < MIN_CERTIFIED_TASKS_PER_ACTION:
        raise ValueError("action decision measured flag lacks certified trial count")
    expected_basis = "measured_lcb_per_cost_ucb" if evidence["measured"] else "bootstrap_prior"
    if evidence.get("basis") != expected_basis:
        raise ValueError("action decision evidence basis is contradictory")
    bounds = {
        "gain_mean": (-4.0, 4.0),
        "gain_lcb": (-4.0, 4.0),
        "cost_mean": (0.0, 1.0),
        "cost_ucb": (_EPSILON_COST, 1.0),
        "gain_used": (-4.0, 4.0),
        "cost_used": (_EPSILON_COST, 1.0),
        "value": (-4.0 / _EPSILON_COST, 4.0 / _EPSILON_COST),
    }
    for name, (minimum, maximum) in bounds.items():
        evidence[name] = _finite(
            evidence.get(name),
            name=f"evidence.{name}",
            minimum=minimum,
            maximum=maximum,
        )
    expected_value = evidence["gain_used"] / evidence["cost_used"]
    if abs(evidence["value"] - expected_value) > 1e-6:
        raise ValueError("action decision value does not equal gain per cost")
    normalized["evidence"] = evidence
    payload = {name: normalized[name] for name in fields - {"decision_sha256"}}
    if normalized["decision_sha256"] != _canonical_sha256(payload):
        raise ValueError("action decision digest does not match")
    return normalized


def validate_action_transition(
    value: Any,
    *,
    require_checked: bool = True,
    allow_campaign_forced: bool = False,
) -> dict[str, Any]:
    fields = {
        "schema",
        "bucket",
        "snapshot_sha256",
        "decision_sha256",
        "step_index",
        "action",
        "mode",
        "outcome",
        "checked",
        "metrics",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("action transition fields differ")
    if value.get("schema") != ACTION_TRANSITION_SCHEMA:
        raise ValueError("action transition schema is invalid")
    normalized = dict(value)
    normalized["bucket"] = _bounded_text(value.get("bucket"), name="bucket", limit=160)
    for name in ("snapshot_sha256", "decision_sha256"):
        digest = value.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"action transition {name} is invalid")
    if type(value.get("step_index")) is not int or value["step_index"] < 0:
        raise ValueError("action transition step_index is invalid")
    try:
        action = OperationKind(value.get("action"))
    except (TypeError, ValueError) as exc:
        raise ValueError("action transition action is invalid") from exc
    normalized["action"] = action.value
    normalized["mode"] = _bounded_text(value.get("mode"), name="mode", limit=32)
    if normalized["mode"] not in _ORDINARY_DECISION_MODES:
        if not (allow_campaign_forced and normalized["mode"] == _CAMPAIGN_DECISION_MODE):
            raise ValueError("action transition mode is unsupported")
    normalized["outcome"] = _bounded_text(value.get("outcome"), name="outcome", limit=64)
    if type(value.get("checked")) is not bool:
        raise ValueError("action transition checked flag is invalid")
    if require_checked and value.get("checked") is not True:
        raise ValueError("action transition is not independently checked")
    metrics = value.get("metrics")
    expected_metrics = {
        "verified_delta",
        "information_gain",
        "diversity_gain",
        "unsupported_confidence",
        "cost",
        "gain",
        "reward",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != expected_metrics:
        raise ValueError("action transition metrics fields differ")
    recomputed = transition_reward(
        verified_delta=metrics["verified_delta"],
        information_gain=metrics["information_gain"],
        diversity_gain=metrics["diversity_gain"],
        unsupported_confidence=metrics["unsupported_confidence"],
        cost=metrics["cost"],
    )
    if any(abs(float(metrics[name]) - float(recomputed[name])) > 1e-7 for name in recomputed):
        raise ValueError("action transition reward does not match measured components")
    normalized["metrics"] = recomputed
    return normalized


def validate_action_trace_row(
    value: Any,
    *,
    evidence_snapshot: Mapping[str, Any],
    executors: tuple[OperationKind, ...],
    action_intervention: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute one worker action from public state and transition signals."""

    base_fields = {
        "decision",
        "transition",
        "state_signal",
        "state_before",
        "state_after",
        "affected_branches",
        "verification",
    }
    constraint_fields = {
        "transient_constraint",
        "transient_constraint_attempt",
    }
    actual_fields = frozenset(value) if isinstance(value, Mapping) else frozenset()
    if not isinstance(value, Mapping) or actual_fields not in {
        frozenset(base_fields),
        frozenset(base_fields | constraint_fields),
    }:
        raise ValueError("cognitive action trace fields differ")
    transient_constraint = value.get("transient_constraint", {})
    transient_attempt = value.get("transient_constraint_attempt", {})
    if not isinstance(transient_constraint, Mapping) or not isinstance(
        transient_attempt,
        Mapping,
    ):
        raise ValueError("cognitive action transient evidence is invalid")
    for evidence, schema, digest_field in (
        (
            transient_constraint,
            "aura.rlc.transient_constraint_application.v1",
            "application_sha256",
        ),
        (
            transient_attempt,
            "aura.rlc.transient_constraint_attempt.v1",
            "attempt_sha256",
        ),
    ):
        if evidence:
            payload = dict(evidence)
            digest = payload.pop(digest_field, None)
            if (
                payload.get("schema") != schema
                or not isinstance(digest, str)
                or digest != _canonical_sha256(payload)
            ):
                raise ValueError("cognitive action transient evidence digest differs")
    if not isinstance(executors, tuple) or not executors:
        raise ValueError("cognitive action executor inventory is invalid")
    if len(set(executors)) != len(executors) or any(
        not isinstance(action, OperationKind) for action in executors
    ):
        raise ValueError("cognitive action executor inventory is invalid")

    signal = CognitiveStateSignal.from_dict(value.get("state_signal"))
    decision = validate_action_decision(value.get("decision"))
    policy = ValueOfComputationPolicy(evidence_snapshot)
    if action_intervention is None:
        expected_decision = policy.choose(signal, executors=executors)
    else:
        from core.brain.llm.latent_cortex.action_intervention import (
            TREATMENT_ARM,
            validate_action_intervention,
        )

        normalized_intervention = validate_action_intervention(
            action_intervention,
            require_current_policy=False,
        )
        authority = normalized_intervention["authority_payload"]
        if (
            authority["arm"] != TREATMENT_ARM
            or authority["intervention_ordinal"] != signal.step_index
        ):
            raise ValueError("forced cognitive action intervention lineage differs")
        expected_decision = policy.choose_forced(
            signal,
            executors=executors,
            action=OperationKind(authority["action"]),
        )
    if decision != expected_decision:
        raise ValueError("cognitive action decision does not match policy and state")
    if signal.pending_constraint_action is not None:
        if (
            decision["mode"] != "constraint_recovery"
            or not transient_constraint
            or transient_constraint.get("source_action") != signal.pending_constraint_action.value
            or transient_constraint.get("applied_action") != signal.pending_constraint_action.value
            or transient_constraint.get("applied_action_step") != signal.step_index
            or type(transient_constraint.get("created_action_step")) is not int
            or transient_constraint["created_action_step"] >= signal.step_index
        ):
            raise ValueError("constraint recovery decision lacks admitted authority")
    elif decision["mode"] == "constraint_recovery":
        raise ValueError("constraint recovery decision lacks a pending constraint")
    transition = validate_action_transition(
        value.get("transition"),
        require_checked=False,
        allow_campaign_forced=action_intervention is not None,
    )
    if (
        transition["decision_sha256"] != decision["decision_sha256"]
        or transition["step_index"] != decision["step_index"]
        or transition["action"] != decision["action"]
        or transition["mode"] != decision["mode"]
        or transition["bucket"] != decision["bucket"]
        or transition["bucket"] != policy.snapshot["bucket"]
        or transition["snapshot_sha256"] != decision["snapshot_sha256"]
        or transition["snapshot_sha256"] != policy.snapshot["snapshot_sha256"]
    ):
        raise ValueError("cognitive action transition differs from decision")

    def normalized_state(raw: Any, *, after: bool) -> dict[str, float | None]:
        expected = {"residual", "disagreement", "verifier_score"}
        if after:
            expected.add("observed_verifier_score")
        else:
            expected.add("budget_remaining_fraction")
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ValueError("cognitive action public state fields differ")
        normalized = {
            "residual": _finite(
                raw.get("residual"), name="trace residual", minimum=0.0, maximum=1.0
            ),
            "disagreement": _finite(
                raw.get("disagreement"),
                name="trace disagreement",
                minimum=0.0,
                maximum=1.0,
            ),
        }
        verifier_score = raw.get("verifier_score")
        normalized["verifier_score"] = (
            None
            if verifier_score is None
            else _finite(
                verifier_score,
                name="trace verifier score",
                minimum=0.0,
                maximum=1.0,
            )
        )
        if not after:
            normalized["budget_remaining_fraction"] = _finite(
                raw.get("budget_remaining_fraction"),
                name="trace remaining budget",
                minimum=0.0,
                maximum=1.0,
            )
        else:
            observed_score = raw.get("observed_verifier_score")
            normalized["observed_verifier_score"] = (
                None
                if observed_score is None
                else _finite(
                    observed_score,
                    name="trace observed verifier score",
                    minimum=0.0,
                    maximum=1.0,
                )
            )
        return normalized

    before = normalized_state(value.get("state_before"), after=False)
    after = normalized_state(value.get("state_after"), after=True)
    for name, expected in (
        ("residual", signal.residual),
        ("disagreement", signal.disagreement),
        ("budget_remaining_fraction", signal.budget_remaining_fraction),
    ):
        if abs(float(before[name]) - expected) > 1e-7:
            raise ValueError("cognitive action state signal differs from trace")
    if before["verifier_score"] != signal.verifier_score:
        raise ValueError("cognitive action verifier state differs from trace")

    before_score = before["verifier_score"]
    after_score = after["verifier_score"]
    observed_score = after["observed_verifier_score"]
    raw_verification = value.get("verification")
    verification_decision = (
        raw_verification.get("decision") if isinstance(raw_verification, Mapping) else None
    )
    checked = before_score is not None and observed_score is not None
    if transition["checked"] is not checked:
        raise ValueError("cognitive action checked status differs from public state")
    verified_delta = float(observed_score) - float(before_score) if checked else 0.0
    if verification_decision == "preserve_verified":
        if before_score is None or observed_score is None:
            raise ValueError("verified-best preservation lacks comparable scores")
        expected_accepted_score = before_score
    elif verification_decision == "reject_verified_failure":
        expected_accepted_score = before_score
    elif observed_score is None:
        expected_accepted_score = before_score
    elif before_score is not None and observed_score < before_score - 1e-9:
        if "regression_reverted" not in transition["outcome"]:
            raise ValueError("cognitive action verifier regression was not reverted")
        expected_accepted_score = before_score
    else:
        expected_accepted_score = observed_score
    if after_score != expected_accepted_score:
        raise ValueError("cognitive action accepted verifier state is invalid")
    before_uncertainty = max(
        float(before["residual"]),
        float(before["disagreement"]),
        1.0 - float(before_score) if before_score is not None else 1.0,
    )
    after_uncertainty = max(
        float(after["residual"]),
        float(after["disagreement"]),
        1.0 - float(observed_score) if observed_score is not None else before_uncertainty,
    )
    expected_metrics = transition_reward(
        verified_delta=max(-1.0, min(1.0, verified_delta)),
        information_gain=max(
            -1.0,
            min(1.0, before_uncertainty - after_uncertainty),
        ),
        diversity_gain=max(
            -1.0,
            min(
                1.0,
                float(after["disagreement"]) - float(before["disagreement"]),
            ),
        ),
        unsupported_confidence=(max(0.0, min(1.0, -verified_delta)) if checked else 0.0),
        cost=transition["metrics"]["cost"],
    )
    if transition["metrics"] != expected_metrics:
        raise ValueError("cognitive action metrics differ from public transition")
    affected_branches = value.get("affected_branches")
    if type(affected_branches) is not int or not 0 <= affected_branches <= signal.total_branches:
        raise ValueError("cognitive action affected branch count is invalid")
    verification = value.get("verification")
    verification_base_fields = {
        "target_branch",
        "observation",
        "decision",
        "restored",
    }
    verification_lineage_fields = {
        "attempt_parent_state_sha256",
        "constraint_input_state_sha256",
        "candidate_state_sha256",
        "restore_target_state_sha256",
        "kv_boundary_before_sha256",
        "kv_boundary_after_sha256",
        "branch_step_before",
        "branch_step_after",
    }
    if not isinstance(verification, Mapping) or frozenset(verification) not in {
        frozenset(verification_base_fields),
        frozenset(verification_base_fields | verification_lineage_fields),
    }:
        raise ValueError("cognitive action verification fields differ")
    verification = dict(verification)
    has_lineage = set(verification) == (verification_base_fields | verification_lineage_fields)
    if verification["target_branch"] is None:
        if (
            verification["observation"] != {}
            or verification["decision"] != "not_run"
            or verification["restored"] is not False
            or observed_score is not None
            or (
                has_lineage
                and (
                    verification["attempt_parent_state_sha256"]
                    or verification["constraint_input_state_sha256"]
                    or verification["candidate_state_sha256"]
                    or verification["restore_target_state_sha256"]
                    or verification["kv_boundary_before_sha256"]
                    or verification["kv_boundary_after_sha256"]
                    or verification["branch_step_before"] is not None
                    or verification["branch_step_after"] is not None
                )
            )
        ):
            raise ValueError("absent cognitive verification is contradictory")
    else:
        from core.brain.llm.latent_cortex.verified_best import (
            validate_observation,
        )

        if (
            type(verification["target_branch"]) is not int
            or not 0 <= verification["target_branch"] < signal.total_branches
            or verification["decision"]
            not in {
                "ranking_only",
                "promote",
                "preserve_verified",
                "reject_verified_failure",
            }
            or type(verification["restored"]) is not bool
            or (
                has_lineage
                and (
                    not _is_sha256(verification["attempt_parent_state_sha256"])
                    or not _is_sha256(verification["constraint_input_state_sha256"])
                    or not _is_sha256(verification["candidate_state_sha256"])
                    or not _is_sha256(verification["kv_boundary_before_sha256"])
                    or not _is_sha256(verification["kv_boundary_after_sha256"])
                    or type(verification["branch_step_before"]) is not int
                    or verification["branch_step_before"] < 0
                    or verification["branch_step_after"] != verification["branch_step_before"] + 1
                )
            )
        ):
            raise ValueError("cognitive verification decision is invalid")
        verification["observation"] = validate_observation(verification["observation"])
        if (
            observed_score is None
            or abs(float(verification["observation"]["score"]) - float(observed_score)) > 1e-9
            or (
                verification["decision"] == "ranking_only"
                and verification["observation"]["authoritative"]
            )
            or (
                verification["decision"] != "ranking_only"
                and not verification["observation"]["authoritative"]
            )
            or (
                verification["decision"] not in {"preserve_verified", "reject_verified_failure"}
                and verification["restored"]
            )
            or (
                verification["decision"] == "reject_verified_failure"
                and (
                    verification["restored"] is not True
                    or verification["observation"]["basis"]
                    not in {"deterministic_exact", "calibrated_interval"}
                    or float(verification["observation"]["upper_bound"]) > 1e-9
                    or (
                        has_lineage
                        and (
                            not _is_sha256(verification["restore_target_state_sha256"])
                            or verification["restore_target_state_sha256"]
                            != verification["attempt_parent_state_sha256"]
                        )
                    )
                )
            )
            or (
                has_lineage
                and verification["decision"] != "reject_verified_failure"
                and verification["restore_target_state_sha256"]
            )
        ):
            raise ValueError("cognitive verification evidence is contradictory")
    normalized = {
        "decision": decision,
        "transition": transition,
        "state_signal": signal.to_dict(),
        "state_before": before,
        "state_after": after,
        "affected_branches": affected_branches,
        "verification": verification,
    }
    if actual_fields == frozenset(base_fields | constraint_fields):
        normalized["transient_constraint"] = dict(transient_constraint)
        normalized["transient_constraint_attempt"] = dict(transient_attempt)
    return normalized


def validate_action_trace(
    value: Any,
    *,
    evidence_snapshot: Mapping[str, Any],
    executors: tuple[OperationKind, ...],
    action_intervention: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one complete, ordered cognitive-action lineage."""

    normalized_intervention: dict[str, Any] | None = None
    intervention_action: OperationKind | None = None
    intervention_arm = ""
    is_control_intervention = False
    if action_intervention is not None:
        from core.brain.llm.latent_cortex.action_intervention import (
            CONTROL_ARM,
            validate_action_intervention,
        )

        normalized_intervention = validate_action_intervention(
            action_intervention,
            require_current_policy=False,
        )
        authority = normalized_intervention["authority_payload"]
        intervention_action = OperationKind(authority["action"])
        intervention_arm = authority["arm"]
        is_control_intervention = intervention_arm == CONTROL_ARM
        if intervention_action not in executors:
            raise ValueError("intervened cognitive action has no resident executor")
        if is_control_intervention:
            executors = tuple(action for action in executors if action != intervention_action)
            if not executors and value:
                raise ValueError("matched control has no post-intervention executor")
    if not isinstance(value, list) or (not value and normalized_intervention is None):
        raise ValueError("cognitive action trace must be a non-empty list")
    if len(value) > 256:
        raise ValueError("cognitive action trace exceeds its bounded length")
    rows: list[dict[str, Any]] = []
    selected: list[OperationKind] = []
    omitted_action_count = 1 if is_control_intervention else 0
    previous_after: Mapping[str, Any] | None = None
    previous_budget = 1.0
    terminal_actions = {
        OperationKind.ANSWER,
        OperationKind.ABSTAIN,
        OperationKind.EXECUTE,
    }
    for index, raw_row in enumerate(value):
        forced_intervention = (
            normalized_intervention if index == 0 and intervention_arm == "forced_action" else None
        )
        row = validate_action_trace_row(
            raw_row,
            evidence_snapshot=evidence_snapshot,
            executors=executors,
            action_intervention=forced_intervention,
        )
        signal = CognitiveStateSignal.from_dict(row["state_signal"])
        decision_action = OperationKind(row["decision"]["action"])
        if signal.step_index != index + omitted_action_count:
            raise ValueError("cognitive action trace step lineage is discontinuous")
        if signal.omitted_action_count != omitted_action_count:
            raise ValueError("cognitive action trace omitted-opportunity lineage differs")
        if signal.previously_selected != tuple(selected):
            raise ValueError("cognitive action trace history is discontinuous")
        if index and selected[-1] in terminal_actions:
            raise ValueError("cognitive action trace continued after a terminal action")
        current_before = row["state_before"]
        if previous_after is not None:
            for name in ("residual", "disagreement"):
                if abs(float(current_before[name]) - float(previous_after[name])) > 1e-7:
                    raise ValueError("cognitive action public state lineage is discontinuous")
        current_budget = float(current_before["budget_remaining_fraction"])
        if current_budget > previous_budget + 1e-7:
            raise ValueError("cognitive action budget lineage increased")
        previous_budget = current_budget
        previous_after = row["state_after"]
        selected.append(decision_action)
        rows.append(row)
        if index == 0 and forced_intervention is not None:
            executors = tuple(action for action in executors if action != intervention_action)
    if normalized_intervention is not None:
        expected_occurrences = 1 if intervention_arm == "forced_action" else 0
        actual_occurrences = sum(action == intervention_action for action in selected)
        if actual_occurrences != expected_occurrences:
            raise ValueError("cognitive action intervention occurrence count differs")
    return {
        "rows": rows,
        "selected_actions": [action.value for action in selected],
        "trace_sha256": _canonical_sha256(rows),
    }


__all__ = [
    "ACTION_EVIDENCE_SCHEMA",
    "ACTION_TRANSITION_SCHEMA",
    "ACTION_VOCABULARY",
    "MIN_ACTION_TRIALS",
    "VALUE_OF_COMPUTATION_SCHEMA",
    "ActionEvidence",
    "CertifiedActionEvidence",
    "CognitiveStateSignal",
    "ValueOfComputationPolicy",
    "action_cost_estimate",
    "build_evidence_snapshot",
    "feasible_actions",
    "transition_reward",
    "validate_action_decision",
    "validate_action_trace",
    "validate_action_trace_row",
    "validate_action_transition",
    "validate_evidence_snapshot",
]
