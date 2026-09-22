"""Journaled exactly-once mutation for verified recurrent transition groups."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never, cast

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.recurrent_grpo import (
    RECURRENT_GRPO_SCHEMA,
    RecurrentGRPOConfig,
    VerifiedTrajectoryGroupConfig,
    build_verified_trajectory_group_source_binding,
    exact_adjoint_verified_transition_group_value_and_grad,
    recurrent_policy_sha256,
    validate_verified_trajectory_group_receipt,
    validate_verified_trajectory_group_source_binding,
    verified_optimization_token_counts,
)
from core.learning.verified_transition_campaign import (
    VerifiedTransitionCampaignLedger,
)
from core.learning.verified_transition_episode import (
    TransitionArtifactStore,
    canonical_json_bytes,
)
from core.learning.verified_transition_group_admission import (
    validate_verified_transition_group_admission,
)
from core.learning.verified_transition_reward import VerifiedTransitionEvidence
from core.runtime.file_read_gateway import read_stable_bytes
from core.runtime.file_write_gateway import FileWriteGateway

VERIFIED_TRANSITION_UPDATE_SCHEMA = "aura.verified_transition.update_receipt.v1"
VERIFIED_TRANSITION_RESERVATION_SCHEMA = "aura.verified_transition.update_reservation.v1"
VERIFIED_TRANSITION_RESERVATION_SCHEMA_V2 = (
    "aura.verified_transition.update_reservation.v2"
)
VERIFIED_TRANSITION_COMMIT_SCHEMA = "aura.verified_transition.update_commit.v1"
VERIFIED_TRANSITION_OBJECTIVE_SCHEMA = "aura.verified_transition.objective_record.v1"
VERIFIED_TRANSITION_OBJECTIVE_SCHEMA_V2 = (
    "aura.verified_transition.objective_record.v2"
)
VERIFIED_TRANSITION_RECONCILIATION_SCHEMA = "aura.verified_transition.update_reconciliation.v1"
_JOURNAL_ENTRY_RE = re.compile(
    r"(?P<admission>[0-9a-f]{64})\."
    r"(?P<suffix>reserved|objective|committed|reconciled)\.json\Z"
)


class VerifiedTransitionUpdateError(RuntimeError):
    """Raised when an admitted update cannot complete exactly once."""


def _fail(code: str) -> Never:
    raise VerifiedTransitionUpdateError(code)


def _require_sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _require_time(value: Any, *, role: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(f"{role}_invalid")
    return value


def _seal(document: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(document)
    sealed["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(sealed)).hexdigest()
    return sealed


def _validate_seal(document: Mapping[str, Any], *, role: str) -> None:
    observed = _require_sha256(document.get("receipt_sha256"), role=f"{role}_receipt")
    unsigned = dict(document)
    unsigned.pop("receipt_sha256", None)
    if observed != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest():
        _fail(f"{role}_digest_mismatch")


def _validate_reservation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("verified_transition_reservation_schema_invalid")
    document = dict(value)
    schema = document.get("schema")
    required = {
        "schema",
        "admission_sha256",
        "policy_before_sha256",
        "reserved_at_unix_ns",
        "receipt_sha256",
    }
    if schema == VERIFIED_TRANSITION_RESERVATION_SCHEMA_V2:
        required |= {
            "campaign_sequence",
            "trainer_step",
            "execution_spec_sha256",
            "group_manifest_sha256",
            "pre_measurement_required",
        }
    elif schema != VERIFIED_TRANSITION_RESERVATION_SCHEMA:
        _fail("verified_transition_reservation_schema_invalid")
    if set(document) != required:
        _fail("verified_transition_reservation_schema_invalid")
    _validate_seal(document, role="verified_transition_reservation")
    _require_sha256(
        document.get("admission_sha256"),
        role="verified_transition_reservation_admission",
    )
    _require_sha256(
        document.get("policy_before_sha256"),
        role="verified_transition_reservation_policy",
    )
    _require_time(
        document.get("reserved_at_unix_ns"),
        role="verified_transition_reservation_time",
    )
    if schema == VERIFIED_TRANSITION_RESERVATION_SCHEMA_V2:
        sequence = document.get("campaign_sequence")
        trainer_step = document.get("trainer_step")
        if (
            type(sequence) is not int
            or sequence < 0
            or trainer_step != sequence + 1
            or document.get("pre_measurement_required") is not True
        ):
            _fail("verified_transition_reservation_measurement_scope_invalid")
        _require_sha256(
            document.get("execution_spec_sha256"),
            role="verified_transition_reservation_execution_spec",
        )
        _require_sha256(
            document.get("group_manifest_sha256"),
            role="verified_transition_reservation_group_manifest",
        )
    return document


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(document),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _seal_objective(document: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(document)
    sealed["receipt_sha256"] = hashlib.sha256(_json_bytes(sealed)).hexdigest()
    return sealed


def _validate_objective_seal(document: Mapping[str, Any]) -> None:
    observed = _require_sha256(
        document.get("receipt_sha256"), role="verified_transition_objective_receipt"
    )
    unsigned = dict(document)
    unsigned.pop("receipt_sha256", None)
    if observed != hashlib.sha256(_json_bytes(unsigned)).hexdigest():
        _fail("verified_transition_objective_digest_mismatch")


def _validate_objective_receipt(
    value: Any,
    *,
    expected_admission_sha256: str | None = None,
    expected_trajectory_source_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base_required = {
        "schema",
        "mode",
        "advantage_report",
        "reference_kl",
        "old_policy_approx_kl",
        "clip_fraction",
        "policy_loss",
        "objective_at_sampling",
        "gradient_surrogate_value",
        "completion_count",
        "token_count",
        "branch_indices",
        "has_gradient",
    }
    if not isinstance(value, Mapping):
        _fail("verified_transition_objective_receipt_schema_invalid")
    receipt = dict(value)
    mode = receipt.get("mode")
    required = (
        base_required
        if mode == "exact_adjoint_single_update"
        else base_required
        | {
            "trajectory_objective_value",
            "composite_objective_at_sampling",
            "composite_gradient_surrogate_value",
            "trajectory_receipt",
        }
        if mode == "exact_adjoint_trajectory_composite_single_update"
        else set()
    )
    if set(receipt) != required:
        _fail("verified_transition_objective_receipt_schema_invalid")
    if (
        receipt.get("schema") != RECURRENT_GRPO_SCHEMA
        or mode
        not in {
            "exact_adjoint_single_update",
            "exact_adjoint_trajectory_composite_single_update",
        }
        or not isinstance(receipt.get("advantage_report"), Mapping)
        or receipt.get("has_gradient") is not True
        or type(receipt.get("completion_count")) is not int
        or receipt["completion_count"] < 2
        or type(receipt.get("token_count")) is not int
        or receipt["token_count"] < receipt["completion_count"]
        or not isinstance(receipt.get("branch_indices"), list)
        or len(receipt["branch_indices"]) != receipt["completion_count"]
        or any(type(index) is not int or index < 0 for index in receipt["branch_indices"])
    ):
        _fail("verified_transition_objective_receipt_invalid")
    for field in (
        "reference_kl",
        "old_policy_approx_kl",
        "clip_fraction",
        "policy_loss",
        "objective_at_sampling",
        "gradient_surrogate_value",
    ):
        observed = receipt.get(field)
        if type(observed) not in {int, float} or not math.isfinite(float(observed)):
            _fail("verified_transition_objective_receipt_invalid")
    if mode == "exact_adjoint_trajectory_composite_single_update":
        for field in (
            "trajectory_objective_value",
            "composite_objective_at_sampling",
            "composite_gradient_surrogate_value",
        ):
            observed = receipt.get(field)
            if type(observed) not in {int, float} or not math.isfinite(float(observed)):
                _fail("verified_transition_objective_receipt_invalid")
        trajectory_value = float(receipt["trajectory_objective_value"])
        if not math.isclose(
            float(receipt["composite_objective_at_sampling"]),
            float(receipt["objective_at_sampling"]) + trajectory_value,
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or not math.isclose(
            float(receipt["composite_gradient_surrogate_value"]),
            float(receipt["gradient_surrogate_value"]) + trajectory_value,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            _fail("verified_transition_objective_composite_arithmetic_invalid")
        try:
            trajectory_receipt = validate_verified_trajectory_group_receipt(
                receipt["trajectory_receipt"],
                advantage_report=receipt["advantage_report"],
                expected_source_binding=expected_trajectory_source_binding,
            )
        except (TypeError, ValueError) as exc:
            raise VerifiedTransitionUpdateError(
                "verified_transition_trajectory_receipt_invalid"
            ) from exc
        if (
            not math.isclose(
                float(trajectory_receipt["trajectory_objective_value"]),
                trajectory_value,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or trajectory_receipt["sample_branch_indices"] != receipt["branch_indices"]
            or (
                expected_admission_sha256 is not None
                and trajectory_receipt["group_admission_sha256"] != expected_admission_sha256
            )
        ):
            _fail("verified_transition_trajectory_binding_invalid")
    elif expected_trajectory_source_binding is not None:
        _fail("verified_transition_unexpected_trajectory_source_binding")
    return receipt


def _validate_objective_record(
    document: Mapping[str, Any],
    *,
    expected_admission_sha256: str,
    expected_policy_sha256: str | None = None,
    expected_trajectory_source_binding: Mapping[str, Any] | None = None,
    expected_pre_measurement_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay a durable objective record before any commit is published."""

    if not isinstance(document, Mapping):
        _fail("verified_transition_objective_record_invalid")
    record = dict(document)
    objective_value = record.get("objective_receipt")
    mode = objective_value.get("mode") if isinstance(objective_value, Mapping) else None
    required = {
        "schema",
        "admission_sha256",
        "objective_receipt",
        "objective_receipt_sha256",
        "receipt_sha256",
    }
    schema = record.get("schema")
    if mode == "exact_adjoint_trajectory_composite_single_update":
        required.add("trajectory_source_binding")
    if schema == VERIFIED_TRANSITION_OBJECTIVE_SCHEMA_V2:
        required.add("pre_measurement_sha256")
    if set(record) != required or schema not in {
        VERIFIED_TRANSITION_OBJECTIVE_SCHEMA,
        VERIFIED_TRANSITION_OBJECTIVE_SCHEMA_V2,
    }:
        _fail("verified_transition_objective_schema_invalid")
    _validate_objective_seal(record)
    source_binding: dict[str, Any] | None = None
    expected_source: dict[str, Any] | None = None
    if expected_trajectory_source_binding is not None:
        try:
            expected_source = validate_verified_trajectory_group_source_binding(
                expected_trajectory_source_binding
            )
        except (TypeError, ValueError) as exc:
            raise VerifiedTransitionUpdateError(
                "verified_transition_expected_trajectory_source_invalid"
            ) from exc
    if mode == "exact_adjoint_trajectory_composite_single_update":
        try:
            source_binding = validate_verified_trajectory_group_source_binding(
                record["trajectory_source_binding"]
            )
        except (TypeError, ValueError) as exc:
            raise VerifiedTransitionUpdateError(
                "verified_transition_trajectory_source_binding_invalid"
            ) from exc
        if source_binding["group_admission_sha256"] != expected_admission_sha256 or (
            expected_policy_sha256 is not None
            and source_binding["policy_sha256"] != expected_policy_sha256
        ):
            _fail("verified_transition_trajectory_source_admission_mismatch")
        if expected_source is not None and source_binding != expected_source:
            _fail("verified_transition_trajectory_source_reconstruction_mismatch")
        intervention = (
            source_binding.get("config", {}).get("intervention_config")
            if isinstance(source_binding.get("config"), Mapping)
            else None
        )
        if intervention is not None and schema != VERIFIED_TRANSITION_OBJECTIVE_SCHEMA_V2:
            _fail("verified_transition_pre_measurement_binding_required")
    elif expected_source is not None:
        _fail("verified_transition_expected_trajectory_objective_missing")
    objective = _validate_objective_receipt(
        objective_value,
        expected_admission_sha256=expected_admission_sha256,
        expected_trajectory_source_binding=expected_source or source_binding,
    )
    objective_sha256 = hashlib.sha256(_json_bytes(objective)).hexdigest()
    if (
        record.get("admission_sha256") != expected_admission_sha256
        or record.get("objective_receipt_sha256") != objective_sha256
    ):
        _fail("verified_transition_objective_record_binding_mismatch")
    if schema == VERIFIED_TRANSITION_OBJECTIVE_SCHEMA_V2:
        observed_pre_measurement = _require_sha256(
            record.get("pre_measurement_sha256"),
            role="objective_pre_measurement",
        )
        if (
            expected_pre_measurement_sha256 is not None
            and observed_pre_measurement
            != _require_sha256(
                expected_pre_measurement_sha256,
                role="expected_pre_measurement",
            )
        ):
            _fail("verified_transition_pre_measurement_binding_mismatch")
    elif expected_pre_measurement_sha256 is not None:
        _fail("verified_transition_pre_measurement_binding_missing")
    return record, objective


@dataclass(frozen=True, slots=True)
class VerifiedTransitionUpdateJournal:
    """Create-once reservation and commit records keyed by admission digest."""

    root: Path
    gateway: FileWriteGateway

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        gateway: FileWriteGateway | None = None,
    ) -> VerifiedTransitionUpdateJournal:
        resolved_gateway = gateway or FileWriteGateway()
        path = Path(
            resolved_gateway.ensure_directory(
                Path(root),
                source="verified_transition_update.journal",
            )
        )
        return cls(root=path, gateway=resolved_gateway)

    def _path(self, admission_sha256: str, suffix: str) -> Path:
        digest = _require_sha256(admission_sha256, role="update_admission")
        return self.root / f"{digest}.{suffix}.json"

    def reserve(
        self,
        *,
        admission_sha256: str,
        policy_before_sha256: str,
        reserved_at_unix_ns: int,
        campaign_sequence: int | None = None,
        execution_spec_sha256: str | None = None,
        group_manifest_sha256: str | None = None,
        pre_measurement_required: bool = False,
    ) -> dict[str, Any]:
        scoped = any(
            value is not None
            for value in (
                campaign_sequence,
                execution_spec_sha256,
                group_manifest_sha256,
            )
        ) or pre_measurement_required
        material: dict[str, Any] = {
            "schema": (
                VERIFIED_TRANSITION_RESERVATION_SCHEMA_V2
                if scoped
                else VERIFIED_TRANSITION_RESERVATION_SCHEMA
            ),
            "admission_sha256": _require_sha256(
                admission_sha256,
                role="reservation_admission",
            ),
            "policy_before_sha256": _require_sha256(
                policy_before_sha256,
                role="reservation_policy_before",
            ),
            "reserved_at_unix_ns": _require_time(
                reserved_at_unix_ns,
                role="reservation_time",
            ),
        }
        if scoped:
            if (
                type(campaign_sequence) is not int
                or campaign_sequence < 0
                or not pre_measurement_required
            ):
                _fail("verified_transition_reservation_measurement_scope_invalid")
            material.update(
                {
                    "campaign_sequence": campaign_sequence,
                    "trainer_step": campaign_sequence + 1,
                    "execution_spec_sha256": _require_sha256(
                        execution_spec_sha256,
                        role="reservation_execution_spec",
                    ),
                    "group_manifest_sha256": _require_sha256(
                        group_manifest_sha256,
                        role="reservation_group_manifest",
                    ),
                    "pre_measurement_required": True,
                }
            )
        reservation = _validate_reservation(_seal(material))
        created = self.gateway.write_bytes_if_absent(
            self._path(admission_sha256, "reserved"),
            _json_bytes(reservation),
            source="verified_transition_update.reserve",
            durable=True,
        )
        if not created:
            _fail("verified_transition_admission_already_reserved")
        return reservation

    def commit(
        self,
        *,
        admission_sha256: str,
        reservation_sha256: str,
        policy_before_sha256: str,
        policy_after_sha256: str,
        objective_record_sha256: str,
        objective_receipt_sha256: str,
        committed_at_unix_ns: int,
    ) -> dict[str, Any]:
        if not self.exists(admission_sha256, "reserved"):
            _fail("verified_transition_reservation_missing")
        reservation = _validate_reservation(
            self.read(admission_sha256, "reserved")
        )
        if not self.exists(admission_sha256, "objective"):
            _fail("verified_transition_objective_missing")
        objective = self.read(admission_sha256, "objective")
        objective, _replayed_objective = _validate_objective_record(
            objective,
            expected_admission_sha256=admission_sha256,
            expected_policy_sha256=policy_before_sha256,
        )
        if (
            reservation.get("receipt_sha256") != reservation_sha256
            or reservation.get("policy_before_sha256")
            != policy_before_sha256
            or objective.get("admission_sha256") != admission_sha256
            or objective.get("receipt_sha256") != objective_record_sha256
            or objective.get("objective_receipt_sha256") != objective_receipt_sha256
        ):
            _fail("verified_transition_objective_commit_binding_mismatch")
        if (
            objective.get("schema") == VERIFIED_TRANSITION_OBJECTIVE_SCHEMA_V2
            and reservation.get("schema")
            != VERIFIED_TRANSITION_RESERVATION_SCHEMA_V2
        ):
            _fail("verified_transition_pre_measurement_reservation_required")
        commit = _seal(
            {
                "schema": VERIFIED_TRANSITION_COMMIT_SCHEMA,
                "admission_sha256": _require_sha256(admission_sha256, role="commit_admission"),
                "reservation_sha256": _require_sha256(
                    reservation_sha256, role="commit_reservation"
                ),
                "policy_before_sha256": _require_sha256(
                    policy_before_sha256, role="commit_policy_before"
                ),
                "policy_after_sha256": _require_sha256(
                    policy_after_sha256, role="commit_policy_after"
                ),
                "objective_record_sha256": _require_sha256(
                    objective_record_sha256, role="commit_objective_record"
                ),
                "objective_receipt_sha256": _require_sha256(
                    objective_receipt_sha256, role="commit_objective"
                ),
                "committed_at_unix_ns": _require_time(committed_at_unix_ns, role="commit_time"),
            }
        )
        created = self.gateway.write_bytes_if_absent(
            self._path(admission_sha256, "committed"),
            _json_bytes(commit),
            source="verified_transition_update.commit",
            durable=True,
        )
        if not created:
            _fail("verified_transition_admission_already_committed")
        return commit

    def record_objective(
        self,
        *,
        admission_sha256: str,
        objective_receipt: Mapping[str, Any],
        trajectory_source_binding: Mapping[str, Any] | None = None,
        pre_measurement_sha256: str | None = None,
    ) -> dict[str, Any]:
        admission = _require_sha256(
            admission_sha256,
            role="objective_admission",
        )
        if not self.exists(admission, "reserved"):
            _fail("verified_transition_reservation_missing")
        reservation = _validate_reservation(self.read(admission, "reserved"))
        source_binding: dict[str, Any] | None = None
        if trajectory_source_binding is not None:
            try:
                source_binding = validate_verified_trajectory_group_source_binding(
                    trajectory_source_binding
                )
            except (TypeError, ValueError) as exc:
                raise VerifiedTransitionUpdateError(
                    "verified_transition_trajectory_source_binding_invalid"
                ) from exc
        objective = _validate_objective_receipt(
            objective_receipt,
            expected_admission_sha256=admission,
            expected_trajectory_source_binding=source_binding,
        )
        is_trajectory = objective["mode"] == "exact_adjoint_trajectory_composite_single_update"
        if (is_trajectory and source_binding is None) or (
            not is_trajectory and source_binding is not None
        ):
            _fail("verified_transition_trajectory_source_binding_required")
        intervention = (
            source_binding.get("config", {}).get("intervention_config")
            if isinstance(source_binding, Mapping)
            and isinstance(source_binding.get("config"), Mapping)
            else None
        )
        if (intervention is not None) is not (
            pre_measurement_sha256 is not None
        ):
            _fail("verified_transition_pre_measurement_binding_required")
        if (
            (pre_measurement_sha256 is not None)
            is not (
                reservation.get("schema")
                == VERIFIED_TRANSITION_RESERVATION_SCHEMA_V2
            )
        ):
            _fail("verified_transition_pre_measurement_reservation_required")
        material = {
            "schema": (
                VERIFIED_TRANSITION_OBJECTIVE_SCHEMA_V2
                if pre_measurement_sha256 is not None
                else VERIFIED_TRANSITION_OBJECTIVE_SCHEMA
            ),
            "admission_sha256": admission,
            "objective_receipt": objective,
            "objective_receipt_sha256": hashlib.sha256(_json_bytes(objective)).hexdigest(),
        }
        if source_binding is not None:
            material["trajectory_source_binding"] = source_binding
        if pre_measurement_sha256 is not None:
            material["pre_measurement_sha256"] = _require_sha256(
                pre_measurement_sha256,
                role="objective_pre_measurement",
            )
        record = _seal_objective(material)
        if not self.gateway.write_bytes_if_absent(
            self._path(admission_sha256, "objective"),
            _json_bytes(record),
            source="verified_transition_update.objective",
            durable=True,
        ):
            _fail("verified_transition_objective_already_recorded")
        return record

    def read(self, admission_sha256: str, suffix: str) -> dict[str, Any]:
        if suffix not in {"reserved", "objective", "committed", "reconciled"}:
            _fail("verified_transition_journal_suffix_invalid")
        payload = read_stable_bytes(
            self._path(admission_sha256, suffix),
            max_bytes=4 * 1024 * 1024 if suffix == "objective" else 65_536,
        )
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VerifiedTransitionUpdateError("verified_transition_journal_json_invalid") from exc
        if not isinstance(document, dict) or _json_bytes(document) != payload:
            _fail("verified_transition_journal_noncanonical")
        return document

    def exists(self, admission_sha256: str, suffix: str) -> bool:
        if suffix not in {"reserved", "objective", "committed", "reconciled"}:
            _fail("verified_transition_journal_suffix_invalid")
        path = self._path(admission_sha256, suffix)
        if path.is_symlink():
            _fail("verified_transition_journal_symlink_rejected")
        return path.is_file()

    def inventory(self) -> tuple[dict[str, Any], ...]:
        """Reopen every create-once admission state without trusting filenames."""

        observed: dict[str, set[str]] = {}
        for path in self.root.iterdir():
            if path.is_symlink() or not path.is_file():
                _fail("verified_transition_journal_entry_invalid")
            match = _JOURNAL_ENTRY_RE.fullmatch(path.name)
            if match is None:
                _fail("verified_transition_journal_entry_name_invalid")
            admission = match.group("admission")
            suffix = match.group("suffix")
            observed.setdefault(admission, set()).add(suffix)
        inventory: list[dict[str, Any]] = []
        for admission in sorted(observed):
            suffixes = observed[admission]
            if "reserved" not in suffixes:
                _fail("verified_transition_journal_reservation_missing")
            if "committed" in suffixes and "reconciled" in suffixes:
                _fail("verified_transition_journal_terminal_conflict")
            reservation = _validate_reservation(
                self.read(admission, "reserved")
            )
            if reservation["admission_sha256"] != admission:
                _fail("verified_transition_journal_admission_mismatch")
            objective = (
                self.read(admission, "objective")
                if "objective" in suffixes
                else None
            )
            if objective is not None:
                objective, _replayed = _validate_objective_record(
                    objective,
                    expected_admission_sha256=admission,
                    expected_policy_sha256=reservation[
                        "policy_before_sha256"
                    ],
                )
                if (
                    (objective.get("schema")
                    == VERIFIED_TRANSITION_OBJECTIVE_SCHEMA_V2)
                    is not (
                        reservation.get("schema")
                        == VERIFIED_TRANSITION_RESERVATION_SCHEMA_V2
                    )
                ):
                    _fail(
                        "verified_transition_pre_measurement_reservation_required"
                    )
            commit = (
                self.read(admission, "committed")
                if "committed" in suffixes
                else None
            )
            if commit is not None:
                if objective is None:
                    _fail("verified_transition_journal_committed_objective_missing")
                if commit.get("schema") != VERIFIED_TRANSITION_COMMIT_SCHEMA:
                    _fail("verified_transition_commit_schema_invalid")
                _validate_seal(commit, role="verified_transition_commit")
                if (
                    commit.get("admission_sha256") != admission
                    or commit.get("reservation_sha256")
                    != reservation["receipt_sha256"]
                    or commit.get("policy_before_sha256")
                    != reservation["policy_before_sha256"]
                ):
                    _fail("verified_transition_journal_commit_binding_mismatch")
            reconciliation = (
                self.read(admission, "reconciled")
                if "reconciled" in suffixes
                else None
            )
            if reconciliation is not None:
                _validate_seal(
                    reconciliation,
                    role="verified_transition_reconciliation",
                )
                if (
                    reconciliation.get("schema")
                    != VERIFIED_TRANSITION_RECONCILIATION_SCHEMA
                    or reconciliation.get("admission_sha256") != admission
                    or reconciliation.get("reservation_sha256")
                    != reservation["receipt_sha256"]
                    or reconciliation.get("policy_before_sha256")
                    != reservation["policy_before_sha256"]
                    or reconciliation.get("admission_reusable") is not False
                    or reconciliation.get("requires_fresh_admission") is not True
                ):
                    _fail(
                        "verified_transition_journal_reconciliation_binding_mismatch"
                    )
            inventory.append(
                {
                    "admission_sha256": admission,
                    "reservation": reservation,
                    "objective": objective,
                    "commit": commit,
                    "reconciliation": reconciliation,
                }
            )
        return tuple(inventory)

    def reconcile(
        self,
        *,
        admission_sha256: str,
        reservation_sha256: str,
        policy_before_sha256: str,
        observed_policy_sha256: str,
        classification: str,
        reconciled_at_unix_ns: int,
    ) -> dict[str, Any]:
        if classification not in {
            "reserved_no_policy_change",
            "policy_changed_without_commit",
        }:
            _fail("verified_transition_reconciliation_classification_invalid")
        if self.exists(admission_sha256, "committed"):
            _fail("verified_transition_reconciliation_after_commit")
        expected_changed = classification == "policy_changed_without_commit"
        if (observed_policy_sha256 != policy_before_sha256) is not expected_changed:
            _fail("verified_transition_reconciliation_policy_mismatch")
        receipt = _seal(
            {
                "schema": VERIFIED_TRANSITION_RECONCILIATION_SCHEMA,
                "admission_sha256": _require_sha256(
                    admission_sha256, role="reconciliation_admission"
                ),
                "reservation_sha256": _require_sha256(
                    reservation_sha256, role="reconciliation_reservation"
                ),
                "policy_before_sha256": _require_sha256(
                    policy_before_sha256, role="reconciliation_policy_before"
                ),
                "observed_policy_sha256": _require_sha256(
                    observed_policy_sha256, role="reconciliation_policy_observed"
                ),
                "classification": classification,
                "admission_reusable": False,
                "requires_fresh_admission": True,
                "requires_checkpoint_recovery": expected_changed,
                "reconciled_at_unix_ns": _require_time(
                    reconciled_at_unix_ns, role="reconciliation_time"
                ),
            }
        )
        if not self.gateway.write_bytes_if_absent(
            self._path(admission_sha256, "reconciled"),
            _json_bytes(receipt),
            source="verified_transition_update.reconcile",
            durable=True,
        ):
            _fail("verified_transition_admission_already_reconciled")
        return receipt


def apply_verified_transition_group_update(
    model: Any,
    optimizer: Any,
    prompt_tokens: Sequence[int],
    samples: Sequence[Any],
    group_admission_receipt: Mapping[str, Any],
    reward_receipt: Mapping[str, Any],
    transition_evidence: Sequence[VerifiedTransitionEvidence],
    *,
    transition_store: TransitionArtifactStore,
    group_manifest: Mapping[str, Any],
    group_manifest_attestation: Mapping[str, Any],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    spec: RLCExecutionSpec,
    journal: VerifiedTransitionUpdateJournal,
    campaign_ledger: VerifiedTransitionCampaignLedger,
    campaign_sequence: int,
    bridge_tokens: Sequence[int] = (),
    config: RecurrentGRPOConfig | None = None,
    trajectory_group_config: VerifiedTrajectoryGroupConfig | None = None,
    now_unix_ns: Callable[[], int] = time.time_ns,
    return_terminal_receipt: bool = False,
    transaction_coordinator: Any | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Validate, reserve, rehash, update exactly once, and receipt the result."""

    if trajectory_group_config is not None:
        if not isinstance(
            trajectory_group_config,
            VerifiedTrajectoryGroupConfig,
        ):
            _fail("verified_transition_trajectory_config_invalid")
        trajectory = trajectory_group_config.trajectory_config
        intervention = trajectory_group_config.intervention_config
        try:
            if trajectory is not None:
                trajectory.validate_depth(spec.recurrent_steps)
            if intervention is not None:
                intervention.validate_depth(spec.recurrent_steps)
        except (TypeError, ValueError) as exc:
            raise VerifiedTransitionUpdateError(
                "verified_transition_trajectory_config_invalid"
            ) from exc
    if transaction_coordinator is None:
        _fail("verified_transition_transaction_coordinator_missing")

    campaign_ledger.validate_started_group(
        sequence=campaign_sequence,
        group_manifest=group_manifest,
    )

    admission = validate_verified_transition_group_admission(
        transition_store,
        group_admission_receipt,
        reward_receipt,
        transition_evidence,
        samples,
        prompt_tokens,
        group_manifest=group_manifest,
        group_manifest_attestation=group_manifest_attestation,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
    )
    if admission.get("optimizer_admitted") is not True:
        _fail("verified_transition_group_not_admitted")
    if admission.get("recurrent_execution_spec_sha256") != spec.sha256:
        _fail("verified_transition_execution_spec_mismatch")
    policy_before = recurrent_policy_sha256(model, spec)
    if admission.get("policy_sha256") != policy_before:
        _fail("verified_transition_policy_before_mismatch")
    reserved_at = _require_time(now_unix_ns(), role="update_reserved_at")
    intervention_required = (
        trajectory_group_config is not None
        and trajectory_group_config.intervention_config is not None
    )
    reservation = journal.reserve(
        admission_sha256=cast_sha256(admission.get("receipt_sha256")),
        policy_before_sha256=policy_before,
        reserved_at_unix_ns=reserved_at,
        campaign_sequence=(
            campaign_sequence if intervention_required else None
        ),
        execution_spec_sha256=(
            spec.sha256 if intervention_required else None
        ),
        group_manifest_sha256=(
            cast_sha256(group_manifest.get("manifest_sha256"))
            if intervention_required
            else None
        ),
        pre_measurement_required=intervention_required,
    )

    resolved_config = config or RecurrentGRPOConfig()
    trajectory_source_binding: dict[str, Any] | None = None
    if trajectory_group_config is not None:
        optimization_token_counts = verified_optimization_token_counts(
            samples,
            transition_evidence,
        )
        trajectory_source_binding = build_verified_trajectory_group_source_binding(
            admission,
            reward_receipt,
            samples,
            prompt_tokens,
            spec=spec,
            trajectory_group_config=trajectory_group_config,
            advantage_clip=resolved_config.advantage_clip,
            optimization_token_counts=optimization_token_counts,
        )
    pre_measurement: dict[str, Any] | None = None
    if intervention_required:
        record_pre_measurement = getattr(
            transaction_coordinator,
            "record_pre_measurement",
            None,
        )
        if not callable(record_pre_measurement):
            _fail("verified_transition_pre_measurement_coordinator_missing")
        pre_measurement = record_pre_measurement(
            group_admission_sha256=cast_sha256(
                admission["receipt_sha256"]
            ),
            reservation_sha256=cast_sha256(
                reservation["receipt_sha256"]
            ),
            policy_before_sha256=policy_before,
            trajectory_source_binding=trajectory_source_binding,
            recurrent_grpo_config=resolved_config,
            bridge_tokens=tuple(bridge_tokens),
            recorded_at_unix_ns=reserved_at,
        )

    objective = exact_adjoint_verified_transition_group_value_and_grad(
        model,
        prompt_tokens,
        samples,
        admission,
        reward_receipt,
        transition_evidence,
        transition_store=transition_store,
        group_manifest=group_manifest,
        group_manifest_attestation=group_manifest_attestation,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        spec=spec,
        bridge_tokens=bridge_tokens,
        config=resolved_config,
        trajectory_group_config=trajectory_group_config,
    )
    if objective.gradients is None:
        _fail("verified_transition_gradient_missing")
    policy_immediately_before_update = recurrent_policy_sha256(model, spec)
    if policy_immediately_before_update != policy_before:
        _fail("verified_transition_policy_changed_before_update")

    objective_receipt = objective.receipt()
    if trajectory_group_config is None:
        if objective_receipt.get("mode") != "exact_adjoint_single_update":
            _fail("verified_transition_unrequested_trajectory_objective")
    elif (
        objective_receipt.get("mode") != "exact_adjoint_trajectory_composite_single_update"
        or not isinstance(
            objective_receipt.get("trajectory_receipt"),
            Mapping,
        )
        or objective_receipt["trajectory_receipt"].get("config")
        != trajectory_group_config.to_dict()
    ):
        _fail("verified_transition_trajectory_objective_missing")
    else:
        assert trajectory_source_binding is not None
    objective_record = journal.record_objective(
        admission_sha256=cast_sha256(admission["receipt_sha256"]),
        objective_receipt=objective_receipt,
        trajectory_source_binding=trajectory_source_binding,
        pre_measurement_sha256=(
            cast_sha256(pre_measurement["receipt_sha256"])
            if pre_measurement is not None
            else None
        ),
    )
    objective_sha256 = cast_sha256(objective_record["objective_receipt_sha256"])

    optimizer.update(model, objective.gradients)
    try:
        import mlx.core as mx

        mx.eval(model.trainable_parameters(), optimizer.state)
    # not a failure: no MLX here, so there is nothing to evaluate eagerly.
    except (ImportError, AttributeError):
        pass
    policy_after = recurrent_policy_sha256(model, spec)
    if policy_after == policy_before:
        _fail("verified_transition_optimizer_did_not_change_policy")
    transaction_coordinator.stage_post_update(
        policy_before_sha256=policy_before,
        policy_after_sha256=policy_after,
        group_admission_sha256=cast_sha256(admission["receipt_sha256"]),
    )
    committed_at = _require_time(now_unix_ns(), role="update_committed_at")
    if committed_at < reserved_at:
        _fail("verified_transition_update_time_reversed")
    commit = journal.commit(
        admission_sha256=cast_sha256(admission["receipt_sha256"]),
        reservation_sha256=cast_sha256(reservation["receipt_sha256"]),
        policy_before_sha256=policy_before,
        policy_after_sha256=policy_after,
        objective_record_sha256=cast_sha256(objective_record["receipt_sha256"]),
        objective_receipt_sha256=objective_sha256,
        committed_at_unix_ns=committed_at,
    )
    receipt = _seal(
        {
            "schema": VERIFIED_TRANSITION_UPDATE_SCHEMA,
            "group_admission_sha256": admission["receipt_sha256"],
            "reservation_sha256": reservation["receipt_sha256"],
            "commit_sha256": commit["receipt_sha256"],
            "objective_record_sha256": objective_record["receipt_sha256"],
            "objective_receipt_sha256": objective_sha256,
            "policy_before_sha256": policy_before,
            "policy_after_sha256": policy_after,
            "optimizer_update_count": 1,
            "reserved_at_unix_ns": reserved_at,
            "committed_at_unix_ns": committed_at,
        }
    )
    transaction_coordinator.record_update_commit(receipt)
    terminal = campaign_ledger.finish_group(
        sequence=campaign_sequence,
        status="updated",
        reward_receipt_sha256=cast_sha256(reward_receipt["receipt_sha256"]),
        group_admission_sha256=cast_sha256(admission["receipt_sha256"]),
        update_receipt_sha256=cast_sha256(receipt["receipt_sha256"]),
        terminal_reason="optimizer_update_committed",
        finished_at_unix_ns=committed_at,
        policy_after_sha256=policy_after,
    )
    transaction_coordinator.record_campaign_terminal(terminal)
    if return_terminal_receipt:
        return receipt, terminal
    return receipt


def validate_verified_transition_update_receipt(
    journal: VerifiedTransitionUpdateJournal,
    receipt: Mapping[str, Any],
    *,
    expected_trajectory_source_binding: Mapping[str, Any] | None = None,
    expected_pre_measurement_sha256: str | None = None,
    expected_campaign_sequence: int | None = None,
    expected_execution_spec_sha256: str | None = None,
    expected_group_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay durable reservation and commit bytes against the final receipt."""

    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema",
        "group_admission_sha256",
        "reservation_sha256",
        "commit_sha256",
        "objective_record_sha256",
        "objective_receipt_sha256",
        "policy_before_sha256",
        "policy_after_sha256",
        "optimizer_update_count",
        "reserved_at_unix_ns",
        "committed_at_unix_ns",
        "receipt_sha256",
    }:
        _fail("verified_transition_update_receipt_schema_invalid")
    if receipt.get("schema") != VERIFIED_TRANSITION_UPDATE_SCHEMA:
        _fail("verified_transition_update_receipt_version_invalid")
    _validate_seal(receipt, role="verified_transition_update")
    admission_sha256 = cast_sha256(receipt.get("group_admission_sha256"))
    reservation = journal.read(admission_sha256, "reserved")
    objective = journal.read(admission_sha256, "objective")
    commit = journal.read(admission_sha256, "committed")
    reservation = _validate_reservation(reservation)
    expected_scope = (
        expected_campaign_sequence,
        expected_execution_spec_sha256,
        expected_group_manifest_sha256,
    )
    if expected_pre_measurement_sha256 is not None or any(
        value is not None for value in expected_scope
    ):
        if (
            reservation.get("schema")
            != VERIFIED_TRANSITION_RESERVATION_SCHEMA_V2
            or reservation.get("pre_measurement_required") is not True
            or (
                expected_campaign_sequence is not None
                and reservation.get("campaign_sequence")
                != expected_campaign_sequence
            )
            or (
                expected_execution_spec_sha256 is not None
                and reservation.get("execution_spec_sha256")
                != expected_execution_spec_sha256
            )
            or (
                expected_group_manifest_sha256 is not None
                and reservation.get("group_manifest_sha256")
                != expected_group_manifest_sha256
            )
        ):
            _fail("verified_transition_reservation_scope_mismatch")
    if commit.get("schema") != VERIFIED_TRANSITION_COMMIT_SCHEMA:
        _fail("verified_transition_commit_schema_invalid")
    _validate_seal(commit, role="verified_transition_commit")
    objective, replayed_objective = _validate_objective_record(
        objective,
        expected_admission_sha256=admission_sha256,
        expected_policy_sha256=cast(str, receipt.get("policy_before_sha256")),
        expected_trajectory_source_binding=expected_trajectory_source_binding,
        expected_pre_measurement_sha256=(
            expected_pre_measurement_sha256
        ),
    )
    replayed_objective_sha256 = hashlib.sha256(_json_bytes(replayed_objective)).hexdigest()
    if (
        reservation.get("receipt_sha256") != receipt.get("reservation_sha256")
        or commit.get("receipt_sha256") != receipt.get("commit_sha256")
        or reservation.get("admission_sha256") != admission_sha256
        or commit.get("admission_sha256") != admission_sha256
        or commit.get("reservation_sha256") != reservation.get("receipt_sha256")
        or objective.get("admission_sha256") != admission_sha256
        or objective.get("receipt_sha256") != receipt.get("objective_record_sha256")
        or commit.get("objective_record_sha256") != objective.get("receipt_sha256")
        or objective.get("objective_receipt_sha256") != replayed_objective_sha256
        or objective.get("objective_receipt_sha256") != receipt.get("objective_receipt_sha256")
        or reservation.get("policy_before_sha256") != receipt.get("policy_before_sha256")
        or commit.get("policy_before_sha256") != receipt.get("policy_before_sha256")
        or commit.get("policy_after_sha256") != receipt.get("policy_after_sha256")
        or commit.get("objective_receipt_sha256") != receipt.get("objective_receipt_sha256")
        or reservation.get("reserved_at_unix_ns") != receipt.get("reserved_at_unix_ns")
        or commit.get("committed_at_unix_ns") != receipt.get("committed_at_unix_ns")
        or receipt.get("optimizer_update_count") != 1
    ):
        _fail("verified_transition_update_reconstruction_mismatch")
    if receipt.get("policy_before_sha256") == receipt.get("policy_after_sha256"):
        _fail("verified_transition_update_policy_unchanged")
    if cast(int, receipt["committed_at_unix_ns"]) < cast(int, receipt["reserved_at_unix_ns"]):
        _fail("verified_transition_update_time_reversed")
    return dict(receipt)


def recover_committed_verified_transition_update(
    journal: VerifiedTransitionUpdateJournal,
    admission_sha256: str,
) -> dict[str, Any]:
    """Reconstruct an update receipt after commit publication interrupted return."""

    admission = cast_sha256(admission_sha256)
    reservation = journal.read(admission, "reserved")
    objective = journal.read(admission, "objective")
    commit = journal.read(admission, "committed")
    reservation = _validate_reservation(reservation)
    if commit.get("schema") != VERIFIED_TRANSITION_COMMIT_SCHEMA:
        _fail("verified_transition_commit_schema_invalid")
    objective, _replayed_objective = _validate_objective_record(
        objective,
        expected_admission_sha256=admission,
        expected_policy_sha256=cast(str, commit.get("policy_before_sha256")),
    )
    _validate_seal(commit, role="verified_transition_commit")
    receipt = _seal(
        {
            "schema": VERIFIED_TRANSITION_UPDATE_SCHEMA,
            "group_admission_sha256": admission,
            "reservation_sha256": reservation["receipt_sha256"],
            "commit_sha256": commit["receipt_sha256"],
            "objective_record_sha256": objective["receipt_sha256"],
            "objective_receipt_sha256": commit["objective_receipt_sha256"],
            "policy_before_sha256": commit["policy_before_sha256"],
            "policy_after_sha256": commit["policy_after_sha256"],
            "optimizer_update_count": 1,
            "reserved_at_unix_ns": reservation["reserved_at_unix_ns"],
            "committed_at_unix_ns": commit["committed_at_unix_ns"],
        }
    )
    return validate_verified_transition_update_receipt(journal, receipt)


def commit_staged_verified_transition_update(
    journal: VerifiedTransitionUpdateJournal,
    *,
    admission_sha256: str,
    policy_before_sha256: str,
    policy_after_sha256: str,
    committed_at_unix_ns: int | None = None,
) -> dict[str, Any]:
    """Roll a durably staged post-update policy through the journal commit.

    This is only valid after an independent transaction store has preserved
    the exact post-update model and optimizer tensors. The reservation and
    objective must already exist, and an existing commit is reconstructed
    rather than written a second time.
    """

    admission = cast_sha256(admission_sha256)
    before = cast_sha256(policy_before_sha256)
    after = cast_sha256(policy_after_sha256)
    if before == after:
        _fail("verified_transition_staged_policy_unchanged")
    reservation = journal.read(admission, "reserved")
    objective = journal.read(admission, "objective")
    reservation = _validate_reservation(reservation)
    objective, _replayed_objective = _validate_objective_record(
        objective,
        expected_admission_sha256=admission,
        expected_policy_sha256=before,
    )
    if (
        reservation.get("admission_sha256") != admission
        or reservation.get("policy_before_sha256") != before
        or objective.get("admission_sha256") != admission
    ):
        _fail("verified_transition_staged_journal_binding_mismatch")
    if journal.exists(admission, "committed"):
        recovered = recover_committed_verified_transition_update(journal, admission)
        if recovered["policy_before_sha256"] != before or recovered["policy_after_sha256"] != after:
            _fail("verified_transition_staged_commit_policy_mismatch")
        return recovered
    committed_at = (
        time.time_ns()
        if committed_at_unix_ns is None
        else _require_time(committed_at_unix_ns, role="staged_commit_time")
    )
    if committed_at < cast(int, reservation["reserved_at_unix_ns"]):
        _fail("verified_transition_staged_commit_time_reversed")
    journal.commit(
        admission_sha256=admission,
        reservation_sha256=cast_sha256(reservation["receipt_sha256"]),
        policy_before_sha256=before,
        policy_after_sha256=after,
        objective_record_sha256=cast_sha256(objective["receipt_sha256"]),
        objective_receipt_sha256=cast_sha256(objective["objective_receipt_sha256"]),
        committed_at_unix_ns=committed_at,
    )
    return recover_committed_verified_transition_update(journal, admission)


def recover_committed_campaign_group(
    journal: VerifiedTransitionUpdateJournal,
    campaign_ledger: VerifiedTransitionCampaignLedger,
    *,
    campaign_sequence: int,
    admission_sha256: str,
    reward_receipt_sha256: str,
) -> dict[str, Any]:
    """Finish campaign custody from a durable update commit after process death."""

    receipt = recover_committed_verified_transition_update(journal, admission_sha256)
    campaign_ledger.finish_group(
        sequence=campaign_sequence,
        status="updated",
        reward_receipt_sha256=cast_sha256(reward_receipt_sha256),
        group_admission_sha256=cast_sha256(admission_sha256),
        update_receipt_sha256=cast_sha256(receipt["receipt_sha256"]),
        terminal_reason="optimizer_update_recovered_from_commit",
        finished_at_unix_ns=cast(int, receipt["committed_at_unix_ns"]),
    )
    return receipt


def reconcile_interrupted_verified_transition_update(
    model: Any,
    spec: RLCExecutionSpec,
    journal: VerifiedTransitionUpdateJournal,
    admission_sha256: str,
    *,
    now_unix_ns: Callable[[], int] = time.time_ns,
) -> dict[str, Any]:
    """Classify a durable reservation that has no durable commit.

    An interrupted admission is always burned. If its policy changed, exact
    checkpoint recovery is required before training may continue; this helper
    deliberately does not guess whether the optimizer update completed.
    """

    admission = cast_sha256(admission_sha256)
    reservation = journal.read(admission, "reserved")
    reservation = _validate_reservation(reservation)
    if reservation.get("admission_sha256") != admission:
        _fail("verified_transition_reservation_admission_mismatch")
    if journal.exists(admission, "committed"):
        _fail("verified_transition_reconciliation_after_commit")
    observed = recurrent_policy_sha256(model, spec)
    before = cast_sha256(reservation.get("policy_before_sha256"))
    classification = (
        "reserved_no_policy_change" if observed == before else "policy_changed_without_commit"
    )
    return journal.reconcile(
        admission_sha256=admission,
        reservation_sha256=cast_sha256(reservation.get("receipt_sha256")),
        policy_before_sha256=before,
        observed_policy_sha256=observed,
        classification=classification,
        reconciled_at_unix_ns=_require_time(now_unix_ns(), role="reconciliation_observed_at"),
    )


def validate_verified_transition_reconciliation_receipt(
    journal: VerifiedTransitionUpdateJournal,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema",
        "admission_sha256",
        "reservation_sha256",
        "policy_before_sha256",
        "observed_policy_sha256",
        "classification",
        "admission_reusable",
        "requires_fresh_admission",
        "requires_checkpoint_recovery",
        "reconciled_at_unix_ns",
        "receipt_sha256",
    }:
        _fail("verified_transition_reconciliation_receipt_schema_invalid")
    if receipt.get("schema") != VERIFIED_TRANSITION_RECONCILIATION_SCHEMA:
        _fail("verified_transition_reconciliation_receipt_version_invalid")
    _validate_seal(receipt, role="verified_transition_reconciliation")
    admission = cast_sha256(receipt.get("admission_sha256"))
    reservation = journal.read(admission, "reserved")
    durable = journal.read(admission, "reconciled")
    changed = receipt.get("policy_before_sha256") != receipt.get("observed_policy_sha256")
    expected_classification = (
        "policy_changed_without_commit" if changed else "reserved_no_policy_change"
    )
    if (
        durable != dict(receipt)
        or reservation.get("receipt_sha256") != receipt.get("reservation_sha256")
        or reservation.get("admission_sha256") != admission
        or reservation.get("policy_before_sha256") != receipt.get("policy_before_sha256")
        or receipt.get("classification") != expected_classification
        or receipt.get("admission_reusable") is not False
        or receipt.get("requires_fresh_admission") is not True
        or receipt.get("requires_checkpoint_recovery") is not changed
        or journal.exists(admission, "committed")
    ):
        _fail("verified_transition_reconciliation_reconstruction_mismatch")
    return dict(receipt)


def cast_sha256(value: Any) -> str:
    return _require_sha256(value, role="verified_transition_digest")


__all__ = [
    "VERIFIED_TRANSITION_COMMIT_SCHEMA",
    "VERIFIED_TRANSITION_OBJECTIVE_SCHEMA",
    "VERIFIED_TRANSITION_OBJECTIVE_SCHEMA_V2",
    "VERIFIED_TRANSITION_RECONCILIATION_SCHEMA",
    "VERIFIED_TRANSITION_RESERVATION_SCHEMA",
    "VERIFIED_TRANSITION_RESERVATION_SCHEMA_V2",
    "VERIFIED_TRANSITION_UPDATE_SCHEMA",
    "VerifiedTransitionUpdateError",
    "VerifiedTransitionUpdateJournal",
    "apply_verified_transition_group_update",
    "commit_staged_verified_transition_update",
    "recover_committed_campaign_group",
    "recover_committed_verified_transition_update",
    "reconcile_interrupted_verified_transition_update",
    "validate_verified_transition_reconciliation_receipt",
    "validate_verified_transition_update_receipt",
]
