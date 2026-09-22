"""Claim-grade calibration for Spark's cognitive action controller.

The online value-of-computation policy must not learn from its own verifier
signals as though they were independent ground truth.  This module defines a
separate, blinded paired campaign:

* every one of the sixteen epistemic actions receives at least eight globally
  unique tasks;
* treatment and control begin from the same committed state and information
  envelope;
* a campaign authority forces exactly one treatment action while the control
  performs a matched no-action transition;
* an external issuer, runner, contamination auditor, and evidence verifier
  attest disjoint parts of the evidence;
* the crash-safe CampaignJournal supplies the only accepted completion
  manifest; and
* simultaneous effect bounds and bounded cost receipts are recomputed into a
  compact certificate that can be independently verified.

The certificate measures functional task-value per bounded compute for a
particular checkpoint, runtime, task distribution, and state bucket.  It does
not establish frontier intelligence or generalize outside those commitments.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final, Never, cast

from core.brain.llm.latent_cortex.campaign_journal import (
    ACTION_INTERVENTION_CLAIMED,
    ARM_RESULT,
    COMMITTED,
    EVENT_SCHEMA,
    PLAN_EVENT,
    STARTED,
    VERIFIED,
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    CONTAMINATION_AUDITOR,
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
    VerifiedCampaignTrustPolicy,
    externally_custodied_roles,
    verify_role_attestation,
)
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.llm.latent_cortex.exact_paired_statistics import (
    DEFAULT_FAMILY_ALPHA,
    Rational,
    certified_rational_effect_bounds,
)
from core.brain.llm.latent_cortex.frontier_tasks import (
    FrontierTask,
    PublicTaskRecord,
    build_public_task_manifest,
    build_task_commitment,
    build_task_manifest,
)
from core.brain.llm.latent_cortex.resource_accounting import (
    RESOURCE_COUNTERS,
    certify_comparison_accounting,
    validate_comparison_accounting_certificate,
    validate_information_receipt,
    validate_resource_receipt,
)

ACTION_CALIBRATION_PROTOCOL_SCHEMA: Final = "aura.rlc.action_calibration.protocol.v1"
ACTION_CALIBRATION_DESIGN_SCHEMA: Final = "aura.rlc.action_calibration.design.v1"
ACTION_CALIBRATION_RESULT_SCHEMA: Final = "aura.rlc.action_calibration.arm_result.v1"
ACTION_CALIBRATION_VERIFICATION_SCHEMA: Final = "aura.rlc.action_calibration.verification.v1"
ACTION_CALIBRATION_CERTIFICATE_SCHEMA: Final = "aura.rlc.action_calibration.certificate.v1"
ACTION_CALIBRATION_CANDIDATE_SCHEMA: Final = "aura.rlc.action_calibration.candidate.v1"
ACTION_CALIBRATION_EVIDENCE_SCHEMA: Final = "aura.rlc.value_of_computation.certified_evidence.v2"
ACTION_CALIBRATION_ISSUER_PAYLOAD_SCHEMA: Final = "aura.rlc.action_calibration.issuer_payload.v1"
ACTION_CALIBRATION_RUNNER_PAYLOAD_SCHEMA: Final = "aura.rlc.action_calibration.runner_payload.v1"
ACTION_CALIBRATION_AUDIT_PAYLOAD_SCHEMA: Final = (
    "aura.rlc.action_calibration.contamination_payload.v1"
)
ACTION_CALIBRATION_VERIFIER_PAYLOAD_SCHEMA: Final = (
    "aura.rlc.action_calibration.verifier_payload.v1"
)
ACTION_CALIBRATION_OUTPUT_SEAL_SCHEMA: Final = "aura.rlc.action_calibration.output_seal.v1"
ACTION_CALIBRATION_FINAL_VERIFIER_SCHEMA: Final = (
    "aura.rlc.action_calibration.final_verifier_payload.v1"
)
ACTION_CALIBRATION_WORKER_ADMISSION_SCHEMA: Final = (
    "aura.rlc.action_calibration.worker_admission.v1"
)
ACTION_CALIBRATION_STATE_CAPTURE_SCHEMA: Final = "aura.rlc.action_calibration.state_capture.v1"
ACTION_CALIBRATION_INTERVENTION_EVIDENCE_SCHEMA: Final = (
    "aura.rlc.action_calibration.intervention_evidence.v1"
)
ACTION_CALIBRATION_SAMPLING_FRAME_SCHEMA: Final = "aura.rlc.action_calibration.sampling_frame.v1"

TREATMENT_ARM: Final = "forced_action"
CONTROL_ARM: Final = "matched_no_action"
CALIBRATION_ARMS: Final = (TREATMENT_ARM, CONTROL_ARM)
MIN_UNIQUE_TASKS_PER_ACTION: Final = 8
MIN_CERTIFIED_TASKS_PER_ACTION: Final = 20
EXPECTED_ACTION_COUNT: Final = 16
MIN_PAIR_COUNT: Final = EXPECTED_ACTION_COUNT * MIN_UNIQUE_TASKS_PER_ACTION
MIN_EXECUTION_COUNT: Final = MIN_PAIR_COUNT * len(CALIBRATION_ARMS)
GLOBAL_BOUND_FAMILY_COUNT: Final = 34
ACTION_RESOURCE_DIMENSIONS: Final = ("estimated_flops", *RESOURCE_COUNTERS)
_COST_BOUND_METHOD: Final = "simultaneous Hoeffding upper bound"
_COST_NORMALIZATION: Final = "max fraction of preregistered action-resource caps"
_GAIN_BOUND_METHOD: Final = "simultaneous rational Clopper-Pearson contrast bounds"


class ActionCalibrationError(ValueError):
    """Stable fail-closed action-calibration contract error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    error = ActionCalibrationError(code)
    raise error


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_equal(left: Any, right: Any) -> bool:
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    # not a failure: two values that will not canonicalise are not equal, and False
    # is the refusing direction.
    except (TypeError, ValueError, RecursionError, OverflowError):
        return False


def _action_resource_caps(value: Any) -> dict[str, int]:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(ACTION_RESOURCE_DIMENSIONS)
        or any(
            type(value.get(name)) is not int or value[name] <= 0
            for name in ACTION_RESOURCE_DIMENSIONS
        )
    ):
        _fail("action_calibration_resource_caps_invalid")
    return {name: cast(int, value[name]) for name in ACTION_RESOURCE_DIMENSIONS}


def _normalized_action_cost(
    resources: Mapping[str, Any],
    *,
    caps: Mapping[str, int],
) -> float:
    if (
        not isinstance(resources, Mapping)
        or set(resources) != set(ACTION_RESOURCE_DIMENSIONS)
        or any(
            type(resources.get(name)) is not int or not 0 <= resources[name] <= caps[name]
            for name in ACTION_RESOURCE_DIMENSIONS
        )
    ):
        _fail("action_calibration_action_resource_vector_invalid")
    return max(resources[name] / caps[name] for name in ACTION_RESOURCE_DIMENSIONS)


def _operation(value: OperationKind | str) -> OperationKind:
    try:
        return value if isinstance(value, OperationKind) else OperationKind(value)
    except (TypeError, ValueError) as exc:
        raise ActionCalibrationError("action_calibration_action_invalid") from exc


def _task_manifest(
    tasks: Sequence[FrontierTask | PublicTaskRecord],
) -> tuple[tuple[PublicTaskRecord, ...], dict[str, Any]]:
    if not tasks:
        _fail("action_calibration_tasks_empty")
    if all(isinstance(task, FrontierTask) for task in tasks):
        full_tasks = cast(Sequence[FrontierTask], tasks)
        public = tuple(task.public for task in full_tasks)
        manifest = build_task_manifest(full_tasks)
    elif all(isinstance(task, PublicTaskRecord) for task in tasks):
        public = tuple(cast(Sequence[PublicTaskRecord], tasks))
        manifest = build_public_task_manifest(public)
    else:
        _fail("action_calibration_task_types_mixed")
    return public, manifest.to_dict()


def _pair_id(action: OperationKind, task: PublicTaskRecord) -> str:
    return "pair-" + _sha256(
        {
            "action": action.value,
            "schema": ACTION_CALIBRATION_PROTOCOL_SCHEMA,
            "task_id": task.task_id,
            "task_payload_sha256": task.task_payload_sha256,
        }
    )


def _task_sampling_identity(task: PublicTaskRecord) -> str:
    """Return a task identity unaffected by answer reblinding."""

    return _sha256(
        {
            "schema": "aura.rlc.action_calibration.task_sampling_identity.v1",
            "registry_version": task.registry_version,
            "domain": task.domain,
            "generator_id": task.generator_id,
            "generator_version": task.generator_version,
            "difficulty": task.difficulty,
            "prompt": task.prompt,
            "response_contract": task.response_contract,
            "scorer_id": task.scorer_id,
            "scorer_version": task.scorer_version,
            "contamination_fingerprints": [
                item.to_dict() for item in task.contamination_fingerprints
            ],
            "excluded_training_families": list(task.excluded_training_families),
        }
    )


def _task_sampling_stratum(task: PublicTaskRecord) -> str:
    return _sha256(
        {
            "schema": "aura.rlc.action_calibration.task_sampling_stratum.v1",
            "registry_version": task.registry_version,
            "domain": task.domain,
            "generator_id": task.generator_id,
            "generator_version": task.generator_version,
            "difficulty": task.difficulty,
            "scorer_id": task.scorer_id,
            "scorer_version": task.scorer_version,
        }
    )


def build_action_calibration_design(
    campaign_name: str,
    tasks_by_action: Mapping[
        OperationKind | str,
        Sequence[FrontierTask | PublicTaskRecord],
    ],
    *,
    model_identity: Mapping[str, Any],
    execution_config: Mapping[str, Any],
    calibration_bucket: str,
    campaign_trust: Mapping[str, Any] | None,
    claim_eligible: bool,
) -> dict[str, Any]:
    """Freeze the answer-blind campaign design before state capture.

    The final campaign plan contains capture receipts, so it cannot be the
    authority used to request those captures. This design is the acyclic
    preregistration root: it fixes every task, assignment, arm order, runtime,
    and trust input that exists before capture, while containing no private
    answer material or captured resident state.
    """

    if (
        not isinstance(campaign_name, str)
        or not campaign_name
        or campaign_name != campaign_name.strip()
        or type(claim_eligible) is not bool
        or not isinstance(calibration_bucket, str)
        or not calibration_bucket
        or calibration_bucket != calibration_bucket.strip()
        or len(calibration_bucket) > 160
        or not isinstance(model_identity, Mapping)
        or not model_identity
        or not isinstance(execution_config, Mapping)
        or not execution_config
        or (
            campaign_trust is not None
            and not isinstance(campaign_trust, Mapping)
        )
    ):
        _fail("action_calibration_design_invalid")
    actions: dict[OperationKind, tuple[FrontierTask | PublicTaskRecord, ...]] = {}
    for raw_action, raw_tasks in tasks_by_action.items():
        action = _operation(raw_action)
        if (
            action in actions
            or isinstance(raw_tasks, (str, bytes))
            or not isinstance(raw_tasks, Sequence)
        ):
            _fail("action_calibration_design_invalid")
        actions[action] = tuple(raw_tasks)
    if set(actions) != set(OperationKind):
        _fail("action_calibration_action_coverage_invalid")

    all_tasks: list[FrontierTask | PublicTaskRecord] = []
    assignments: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    seen_sampling_identities: set[str] = set()
    for action in sorted(actions, key=lambda item: item.value):
        raw_tasks = actions[action]
        if len(raw_tasks) < MIN_UNIQUE_TASKS_PER_ACTION:
            _fail("action_calibration_action_underpowered")
        public, _manifest = _task_manifest(raw_tasks)
        ordered = tuple(sorted(public, key=lambda task: task.task_id))
        if len({task.domain for task in ordered}) < 2:
            _fail("action_calibration_action_domain_coverage_invalid")
        treatment_first_offset = (
            int(
                hashlib.sha256(
                    f"{campaign_name}:{action.value}:counterbalance".encode()
                ).hexdigest(),
                16,
            )
            % 2
        )
        for task_ordinal, task in enumerate(ordered):
            sampling_identity = _task_sampling_identity(task)
            if (
                task.task_id in seen_task_ids
                or sampling_identity in seen_sampling_identities
            ):
                _fail("action_calibration_underlying_task_reused")
            seen_task_ids.add(task.task_id)
            seen_sampling_identities.add(sampling_identity)
            treatment_first = (task_ordinal + treatment_first_offset) % 2 == 0
            arm_order = (
                CALIBRATION_ARMS
                if treatment_first
                else tuple(reversed(CALIBRATION_ARMS))
            )
            assignments.append(
                {
                    "action": action.value,
                    "arm_order": list(arm_order),
                    "pair_id": _pair_id(action, task),
                    "task_id": task.task_id,
                    "task_payload_sha256": task.task_payload_sha256,
                    "task_sampling_identity_sha256": sampling_identity,
                    "task_sampling_stratum_sha256": _task_sampling_stratum(task),
                }
            )
        all_tasks.extend(raw_tasks)

    _all_public, manifest = _task_manifest(tuple(all_tasks))
    public_tasks = tuple(
        sorted(
            (
                task.public if isinstance(task, FrontierTask) else task
                for task in all_tasks
            ),
            key=lambda task: task.task_id,
        )
    )
    commitment = build_task_commitment(build_public_task_manifest(public_tasks)).to_dict()
    body = {
        "schema": ACTION_CALIBRATION_DESIGN_SCHEMA,
        "protocol_schema": ACTION_CALIBRATION_PROTOCOL_SCHEMA,
        "campaign_name": campaign_name,
        "claim_eligible": claim_eligible,
        "calibration_bucket": calibration_bucket,
        "model_identity": dict(model_identity),
        "execution_config": dict(execution_config),
        "campaign_trust": None if campaign_trust is None else dict(campaign_trust),
        "task_manifest": manifest,
        "task_commitment": commitment,
        "assignments": assignments,
        "assignment_sha256": _sha256(assignments),
    }
    return {**body, "campaign_design_sha256": _sha256(body)}


def action_calibration_starting_state_payload(
    *,
    campaign_name: str,
    action: OperationKind,
    task: PublicTaskRecord,
    model_identity: Mapping[str, Any],
    execution_config: Mapping[str, Any],
    calibration_bucket: str,
    capture_id: str,
    captured_at_unix: int,
    bucket_classifier_sha256: str,
    bucket_evidence_sha256: str,
    state_component_sha256: Mapping[str, str],
    campaign_design_sha256: str,
) -> dict[str, Any]:
    component_names = {
        "latent_slots_sha256",
        "branch_state_sha256",
        "kv_cache_sha256",
        "evidence_state_sha256",
        "memory_state_sha256",
        "public_action_state_sha256",
        "durable_state_sha256",
        "rng_state_sha256",
    }
    if (
        not isinstance(campaign_name, str)
        or not campaign_name
        or not isinstance(capture_id, str)
        or not capture_id
        or type(captured_at_unix) is not int
        or captured_at_unix <= 0
        or not isinstance(calibration_bucket, str)
        or not calibration_bucket
        or not _is_sha256(bucket_classifier_sha256)
        or not _is_sha256(bucket_evidence_sha256)
        or not isinstance(state_component_sha256, Mapping)
        or set(state_component_sha256) != component_names
        or any(not _is_sha256(state_component_sha256[name]) for name in component_names)
        or not _is_sha256(campaign_design_sha256)
    ):
        _fail("action_calibration_state_capture_invalid")
    body = {
        "schema": ACTION_CALIBRATION_STATE_CAPTURE_SCHEMA,
        "capture_mode": "externally_captured_runtime_state_v1",
        "capture_id": capture_id,
        "captured_at_unix": captured_at_unix,
        "campaign_name": campaign_name,
        "campaign_design_sha256": campaign_design_sha256,
        "action": action.value,
        "task_id": task.task_id,
        "task_sampling_identity_sha256": _task_sampling_identity(task),
        "calibration_bucket": calibration_bucket,
        "bucket_classifier_sha256": bucket_classifier_sha256,
        "bucket_evidence_sha256": bucket_evidence_sha256,
        **dict(state_component_sha256),
        "model_identity_sha256": _sha256(dict(model_identity)),
        "continuation_policy_sha256": execution_config["continuation_policy_sha256"],
        "budget_policy_sha256": execution_config["budget_policy_sha256"],
    }
    return {**body, "state_sha256": _sha256(body)}


def _validate_starting_state_receipt(
    value: Any,
    *,
    campaign_name: str,
    action: OperationKind,
    task: PublicTaskRecord,
    model_identity: Mapping[str, Any],
    execution_config: Mapping[str, Any],
    calibration_bucket: str,
    campaign_design_sha256: str,
    policy: VerifiedCampaignTrustPolicy | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("action_calibration_state_capture_invalid")
    capture_attestation = value.get("capture_attestation")
    raw_payload = {name: item for name, item in value.items() if name != "capture_attestation"}
    expected_fields = {
        "schema",
        "capture_mode",
        "capture_id",
        "captured_at_unix",
        "campaign_name",
        "campaign_design_sha256",
        "action",
        "task_id",
        "task_sampling_identity_sha256",
        "calibration_bucket",
        "bucket_classifier_sha256",
        "bucket_evidence_sha256",
        "latent_slots_sha256",
        "branch_state_sha256",
        "kv_cache_sha256",
        "evidence_state_sha256",
        "memory_state_sha256",
        "public_action_state_sha256",
        "durable_state_sha256",
        "rng_state_sha256",
        "model_identity_sha256",
        "continuation_policy_sha256",
        "budget_policy_sha256",
        "state_sha256",
    }
    if (
        set(raw_payload) != expected_fields
        or "capture_attestation" not in value
        or not isinstance(capture_attestation, Mapping)
        or raw_payload.get("schema") != ACTION_CALIBRATION_STATE_CAPTURE_SCHEMA
        or raw_payload.get("capture_mode") != "externally_captured_runtime_state_v1"
        or raw_payload.get("campaign_name") != campaign_name
        or raw_payload.get("campaign_design_sha256")
        != campaign_design_sha256
        or raw_payload.get("action") != action.value
        or raw_payload.get("task_id") != task.task_id
        or raw_payload.get("task_sampling_identity_sha256") != _task_sampling_identity(task)
        or raw_payload.get("calibration_bucket") != calibration_bucket
    ):
        _fail("action_calibration_state_capture_invalid")
    expected = action_calibration_starting_state_payload(
        campaign_name=campaign_name,
        action=action,
        task=task,
        model_identity=model_identity,
        execution_config=execution_config,
        calibration_bucket=calibration_bucket,
        capture_id=cast(str, raw_payload.get("capture_id")),
        captured_at_unix=cast(int, raw_payload.get("captured_at_unix")),
        bucket_classifier_sha256=cast(str, raw_payload.get("bucket_classifier_sha256")),
        bucket_evidence_sha256=cast(str, raw_payload.get("bucket_evidence_sha256")),
        state_component_sha256={
            name: cast(str, raw_payload.get(name))
            for name in (
                "latent_slots_sha256",
                "branch_state_sha256",
                "kv_cache_sha256",
                "evidence_state_sha256",
                "memory_state_sha256",
                "public_action_state_sha256",
                "durable_state_sha256",
                "rng_state_sha256",
            )
        },
        campaign_design_sha256=campaign_design_sha256,
    )
    if not _strict_equal(raw_payload, expected):
        _fail("action_calibration_state_capture_invalid")
    if policy is not None:
        verify_role_attestation(
            policy,
            capture_attestation,
            role=CAMPAIGN_RUNNER,
            expected_payload=expected,
        )
    return {**expected, "capture_attestation": dict(cast(Mapping[str, Any], capture_attestation))}


def _validate_plan_sampling_frame(plan: CampaignPlan) -> dict[str, Any]:
    metadata = _metadata(plan)
    raw_manifest = metadata.get("task_manifest")
    assignments = metadata.get("assignments")
    sampling_frame = metadata.get("sampling_frame")
    execution_config = metadata.get("execution_config")
    if (
        not isinstance(raw_manifest, Mapping)
        or not isinstance(raw_manifest.get("tasks"), list)
        or not isinstance(assignments, list)
        or not isinstance(sampling_frame, Mapping)
        or not isinstance(execution_config, Mapping)
    ):
        _fail("action_calibration_sampling_frame_invalid")
    try:
        tasks = {
            task.task_id: task
            for task in (
                PublicTaskRecord.from_dict(cast(Mapping[str, Any], raw))
                for raw in raw_manifest["tasks"]
            )
        }
    except (TypeError, ValueError) as exc:
        raise ActionCalibrationError("action_calibration_sampling_frame_invalid") from exc
    if len(tasks) != len(raw_manifest["tasks"]):
        _fail("action_calibration_underlying_task_reused")
    required_assignment_fields = {
        "action",
        "arm_order",
        "pair_id",
        "starting_state_sha256",
        "task_id",
        "task_payload_sha256",
        "task_sampling_identity_sha256",
        "task_sampling_stratum_sha256",
    }
    identities: set[str] = set()
    strata_by_action: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    task_count_by_action: dict[str, int] = defaultdict(int)
    for row in assignments:
        if not isinstance(row, Mapping) or set(row) != required_assignment_fields:
            _fail("action_calibration_sampling_frame_invalid")
        action = _operation(row.get("action"))
        task = tasks.get(cast(str, row.get("task_id")))
        identity = _task_sampling_identity(task) if task is not None else None
        stratum = _task_sampling_stratum(task) if task is not None else None
        if (
            task is None
            or row.get("task_payload_sha256") != task.task_payload_sha256
            or row.get("task_sampling_identity_sha256") != identity
            or row.get("task_sampling_stratum_sha256") != stratum
            or identity in identities
        ):
            _fail("action_calibration_underlying_task_reused")
        identities.add(cast(str, identity))
        strata_by_action[action.value][cast(str, stratum)] += 1
        task_count_by_action[action.value] += 1
    if set(task_count_by_action) != {action.value for action in OperationKind}:
        _fail("action_calibration_sampling_frame_unbalanced")
    reference_action = min(task_count_by_action)
    reference_count = task_count_by_action[reference_action]
    reference_strata = dict(sorted(strata_by_action[reference_action].items()))
    if any(
        task_count_by_action[action.value] != reference_count
        or dict(sorted(strata_by_action[action.value].items())) != reference_strata
        for action in OperationKind
    ):
        _fail("action_calibration_sampling_frame_unbalanced")
    expected = {
        "schema": ACTION_CALIBRATION_SAMPLING_FRAME_SCHEMA,
        "assignment_policy": execution_config.get("task_assignment_policy"),
        "assignment_seed_sha256": execution_config.get("task_assignment_seed_sha256"),
        "tasks_per_action": reference_count,
        "stratum_counts": reference_strata,
        "task_sampling_identities": sorted(identities),
    }
    expected = {**expected, "sampling_frame_sha256": _sha256(expected)}
    if not _strict_equal(sampling_frame, expected):
        _fail("action_calibration_sampling_frame_invalid")
    return expected


def build_action_calibration_plan(
    campaign_name: str,
    tasks_by_action: Mapping[
        OperationKind | str,
        Sequence[FrontierTask | PublicTaskRecord],
    ],
    *,
    model_identity: Mapping[str, Any],
    execution_config: Mapping[str, Any],
    calibration_bucket: str,
    campaign_design: Mapping[str, Any] | None = None,
    starting_state_receipts: Mapping[str, Mapping[str, Any]] | None = None,
    campaign_trust: Mapping[str, Any] | None = None,
    claim_eligible: bool = False,
) -> CampaignPlan:
    """Freeze a complete, counterbalanced 16-action paired campaign."""

    if type(claim_eligible) is not bool:
        _fail("action_calibration_claim_flag_invalid")
    if (
        not isinstance(calibration_bucket, str)
        or not calibration_bucket
        or calibration_bucket != calibration_bucket.strip()
        or len(calibration_bucket) > 160
    ):
        _fail("action_calibration_bucket_invalid")
    actions: dict[OperationKind, tuple[FrontierTask | PublicTaskRecord, ...]] = {}
    for raw_action, raw_tasks in tasks_by_action.items():
        action = _operation(raw_action)
        if action in actions:
            _fail("action_calibration_action_duplicate")
        if isinstance(raw_tasks, (str, bytes)) or not isinstance(raw_tasks, Sequence):
            _fail("action_calibration_tasks_invalid")
        actions[action] = tuple(raw_tasks)
    if set(actions) != set(OperationKind):
        _fail("action_calibration_action_coverage_invalid")

    all_tasks: list[FrontierTask | PublicTaskRecord] = []
    public_by_action: dict[OperationKind, tuple[PublicTaskRecord, ...]] = {}
    seen_task_ids: set[str] = set()
    seen_sampling_identities: set[str] = set()
    stratum_counts_by_action: dict[OperationKind, dict[str, int]] = {}
    for action in sorted(actions, key=lambda item: item.value):
        raw_tasks = actions[action]
        if len(raw_tasks) < MIN_UNIQUE_TASKS_PER_ACTION:
            _fail("action_calibration_action_underpowered")
        public, _manifest = _task_manifest(raw_tasks)
        ordered = tuple(sorted(public, key=lambda task: task.task_id))
        for task in ordered:
            if task.task_id in seen_task_ids:
                _fail("action_calibration_task_reused")
            sampling_identity = _task_sampling_identity(task)
            if sampling_identity in seen_sampling_identities:
                _fail("action_calibration_underlying_task_reused")
            seen_task_ids.add(task.task_id)
            seen_sampling_identities.add(sampling_identity)
        public_by_action[action] = ordered
        stratum_counts: dict[str, int] = defaultdict(int)
        for task in ordered:
            stratum_counts[_task_sampling_stratum(task)] += 1
        stratum_counts_by_action[action] = dict(sorted(stratum_counts.items()))
        all_tasks.extend(raw_tasks)
    reference_strata = next(iter(stratum_counts_by_action.values()))
    reference_count = len(next(iter(public_by_action.values())))
    if any(
        len(public_by_action[action]) != reference_count
        or stratum_counts_by_action[action] != reference_strata
        for action in OperationKind
    ):
        _fail("action_calibration_sampling_frame_unbalanced")

    _all_public, manifest = _task_manifest(tuple(all_tasks))
    commitment = build_task_commitment(
        build_public_task_manifest(
            tuple(
                sorted(
                    (task.public if isinstance(task, FrontierTask) else task for task in all_tasks),
                    key=lambda task: task.task_id,
                )
            )
        )
    ).to_dict()
    if (
        not isinstance(model_identity, Mapping)
        or not model_identity
        or not isinstance(execution_config, Mapping)
        or not execution_config
    ):
        _fail("action_calibration_runtime_identity_invalid")
    resource_caps = _action_resource_caps(execution_config.get("action_resource_caps"))
    cost_budget = execution_config.get("action_cost_budget_estimated_flops")
    if (
        type(cost_budget) is not int
        or cost_budget <= 0
        or cost_budget != resource_caps["estimated_flops"]
    ):
        _fail("action_calibration_cost_budget_invalid")
    if execution_config.get("worker_task_material") != "public_manifest_only":
        _fail("action_calibration_blinding_invalid")
    if execution_config.get("answer_reveal_protocol") != "sealed_outputs_then_issuer_reveal_v1":
        _fail("action_calibration_blinding_invalid")
    if execution_config.get(
        "task_assignment_policy"
    ) != "external_issuer_stratified_random_without_replacement_v1" or not _is_sha256(
        execution_config.get("task_assignment_seed_sha256")
    ):
        _fail("action_calibration_sampling_frame_invalid")
    if claim_eligible:
        if (
            not all(isinstance(task, FrontierTask) for task in all_tasks)
            or execution_config.get("answer_blind_nonce_policy") != "external_issuer_csprng_256"
            or execution_config.get("answer_blind_nonce_disclosure") != "post_seal_answer_reveal"
            or execution_config.get("answer_blind_nonce_count") != len(all_tasks)
            or execution_config.get("answer_blind_nonce_min_entropy_bits") != 256
            or execution_config.get("generation_seed_policy") != "external_issuer_uniform_63bit"
            or execution_config.get("generation_seed_count") != len(all_tasks)
            or type(execution_config.get("generation_seed_min_entropy_bits")) is not int
            or execution_config["generation_seed_min_entropy_bits"] < 60
        ):
            _fail("action_calibration_external_blinding_required")
        blind_nonces: set[str] = set()
        for raw_task in all_tasks:
            task = cast(FrontierTask, raw_task)
            private = task.reveal_for_verifier()
            blind_nonce = private.get("blind_nonce")
            derived_nonce = _sha256(
                {
                    "purpose": "answer_blind",
                    "registry_version": private.get("registry_version"),
                    "domain": private.get("domain"),
                    "seed": private.get("generation_seed"),
                    "difficulty": private.get("difficulty"),
                    "expected": private.get("expected"),
                }
            )
            try:
                nonce_bytes = bytes.fromhex(cast(str, blind_nonce))
            except (TypeError, ValueError):
                _fail("action_calibration_external_blinding_required")
            if (
                not _is_sha256(blind_nonce)
                or blind_nonce == derived_nonce
                or len(nonce_bytes) != 32
                or len(set(nonce_bytes)) < 16
                or blind_nonce in blind_nonces
            ):
                _fail("action_calibration_external_blinding_required")
            blind_nonces.add(cast(str, blind_nonce))
    for commitment_name in (
        "continuation_policy_sha256",
        "budget_policy_sha256",
        "rng_root_sha256",
        "instrumentation_sha256",
        "execute_fixture_policy_sha256",
    ):
        if not _is_sha256(execution_config.get(commitment_name)):
            _fail("action_calibration_execution_commitment_invalid")
    if execution_config.get("execute_calibration_effect_class") not in {
        "deterministic_sandbox",
        "governed_read_only_fixture",
    }:
        _fail("action_calibration_execute_fixture_invalid")
    trust = None if campaign_trust is None else dict(campaign_trust)
    if claim_eligible and (
        not isinstance(trust, dict)
        or trust.get("prelaunch_verified") is not True
        or trust.get("externally_custodied") is not True
        or not _is_sha256(trust.get("policy_sha256"))
    ):
        _fail("action_calibration_external_trust_required")
    expected_design = build_action_calibration_design(
        campaign_name,
        tasks_by_action,
        model_identity=model_identity,
        execution_config=execution_config,
        calibration_bucket=calibration_bucket,
        campaign_trust=trust,
        claim_eligible=claim_eligible,
    )
    if not _strict_equal(campaign_design, expected_design):
        _fail("action_calibration_design_mismatch")
    campaign_design_sha256 = expected_design["campaign_design_sha256"]
    if (
        not isinstance(starting_state_receipts, Mapping)
        or set(starting_state_receipts) != seen_task_ids
    ):
        _fail("action_calibration_state_capture_coverage_invalid")

    assignment_rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    execution_ordinal = 0
    for action in sorted(public_by_action, key=lambda item: item.value):
        tasks = public_by_action[action]
        if len({task.domain for task in tasks}) < 2:
            _fail("action_calibration_action_domain_coverage_invalid")
        treatment_first_offset = (
            int(
                hashlib.sha256(
                    f"{campaign_name}:{action.value}:counterbalance".encode()
                ).hexdigest(),
                16,
            )
            % 2
        )
        for task_ordinal, task in enumerate(tasks):
            pair_id = _pair_id(action, task)
            starting_state = _validate_starting_state_receipt(
                starting_state_receipts.get(task.task_id),
                campaign_name=campaign_name,
                action=action,
                task=task,
                model_identity=model_identity,
                execution_config=execution_config,
                calibration_bucket=calibration_bucket,
                campaign_design_sha256=campaign_design_sha256,
            )
            treatment_first = (task_ordinal + treatment_first_offset) % 2 == 0
            arm_order = CALIBRATION_ARMS if treatment_first else tuple(reversed(CALIBRATION_ARMS))
            assignment_rows.append(
                {
                    "action": action.value,
                    "arm_order": list(arm_order),
                    "pair_id": pair_id,
                    "starting_state_sha256": starting_state["state_sha256"],
                    "task_id": task.task_id,
                    "task_payload_sha256": task.task_payload_sha256,
                    "task_sampling_identity_sha256": _task_sampling_identity(task),
                    "task_sampling_stratum_sha256": _task_sampling_stratum(task),
                }
            )
            for pair_arm_ordinal, arm in enumerate(arm_order):
                cells.append(
                    {
                        "action": action.value,
                        "arm": arm,
                        "execution_ordinal": execution_ordinal,
                        "pair_arm_ordinal": pair_arm_ordinal,
                        "pair_id": pair_id,
                        "starting_state": starting_state,
                        "starting_state_sha256": starting_state["state_sha256"],
                        "task_id": task.task_id,
                        "task_payload_sha256": task.task_payload_sha256,
                        "task_sampling_identity_sha256": _task_sampling_identity(task),
                        "task_sampling_stratum_sha256": _task_sampling_stratum(task),
                    }
                )
                execution_ordinal += 1
    assignment_sha256 = _sha256(assignment_rows)
    sampling_frame = {
        "schema": ACTION_CALIBRATION_SAMPLING_FRAME_SCHEMA,
        "assignment_policy": execution_config["task_assignment_policy"],
        "assignment_seed_sha256": execution_config["task_assignment_seed_sha256"],
        "tasks_per_action": reference_count,
        "stratum_counts": reference_strata,
        "task_sampling_identities": sorted(seen_sampling_identities),
    }
    sampling_frame = {
        **sampling_frame,
        "sampling_frame_sha256": _sha256(sampling_frame),
    }
    metadata = {
        "schema": ACTION_CALIBRATION_PROTOCOL_SCHEMA,
        "action_intervention_required": True,
        "strict_execution_order": claim_eligible,
        "claim_eligible": claim_eligible,
        "claim_scope": (
            "checkpoint-bound value-of-computation calibration"
            if claim_eligible
            else "value-of-computation calibration preflight"
        ),
        "calibration_bucket": calibration_bucket,
        "frontier_claim_eligible": False,
        "action_count": len(public_by_action),
        "pair_count": len(assignment_rows),
        "execution_count": len(cells),
        "minimum_unique_tasks_per_action": MIN_UNIQUE_TASKS_PER_ACTION,
        "assignment_sha256": assignment_sha256,
        "assignments": assignment_rows,
        "sampling_frame": sampling_frame,
        "task_manifest": manifest,
        "task_commitment": commitment,
        "model_identity": dict(model_identity),
        "execution_config": dict(execution_config),
        "campaign_trust": trust,
        "campaign_design": expected_design,
        "campaign_design_sha256": campaign_design_sha256,
    }
    plan = CampaignPlan.build(campaign_name, cells, metadata=metadata)
    _validate_plan_sampling_frame(plan)
    return plan


def action_calibration_issuer_payload(plan: CampaignPlan) -> dict[str, Any]:
    metadata = _metadata(plan)
    return {
        "schema": ACTION_CALIBRATION_ISSUER_PAYLOAD_SCHEMA,
        "campaign_name": plan.campaign_name,
        "plan_sha256": plan.plan_sha256,
        "task_manifest_sha256": metadata["task_manifest"]["manifest_sha256"],
        "task_commitment_sha256": metadata["task_commitment"]["commitment_sha256"],
        "assignment_sha256": metadata["assignment_sha256"],
        "action_count": metadata["action_count"],
        "pair_count": metadata["pair_count"],
    }


def action_calibration_contamination_payload(
    plan: CampaignPlan,
    *,
    corpus_snapshot_sha256: str,
    methods: Sequence[str],
) -> dict[str, Any]:
    metadata = _metadata(plan)
    if not _is_sha256(corpus_snapshot_sha256):
        _fail("action_calibration_contamination_corpus_invalid")
    normalized_methods = sorted(set(methods))
    if not normalized_methods or any(
        not isinstance(method, str) or not method or method != method.strip()
        for method in normalized_methods
    ):
        _fail("action_calibration_contamination_methods_invalid")
    return {
        "schema": ACTION_CALIBRATION_AUDIT_PAYLOAD_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "task_manifest_sha256": metadata["task_manifest"]["manifest_sha256"],
        "assignment_sha256": metadata["assignment_sha256"],
        "status": "passed_zero_overlap",
        "overlap_count": 0,
        "corpus_snapshot_sha256": corpus_snapshot_sha256,
        "methods": normalized_methods,
    }


def action_calibration_runner_payload(
    *,
    plan: CampaignPlan,
    cell_id: str,
    attempt_id: str,
    result_core: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": ACTION_CALIBRATION_RUNNER_PAYLOAD_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "cell_id": cell_id,
        "attempt_id": attempt_id,
        "definition_sha256": _sha256(plan.cell_definition(cell_id)),
        "result_core_sha256": _sha256(dict(result_core)),
    }


def action_calibration_verifier_payload(
    *,
    plan: CampaignPlan,
    cell_id: str,
    result_sha256: str,
    score_receipt: Mapping[str, Any],
    answer_commitment_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": ACTION_CALIBRATION_VERIFIER_PAYLOAD_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "cell_id": cell_id,
        "result_sha256": result_sha256,
        "score_receipt_sha256": _sha256(dict(score_receipt)),
        "answer_commitment_sha256": answer_commitment_sha256,
    }


def action_calibration_output_seal_payload(
    plan: CampaignPlan,
    *,
    result_sha256_by_cell: Mapping[str, str],
    journal_head_sha256: str,
    journal_event_count: int,
) -> dict[str, Any]:
    """Bind every sealed model output before any hidden answer is revealed."""

    if (
        set(result_sha256_by_cell) != set(plan.cell_ids)
        or any(not _is_sha256(result_sha256_by_cell[cell_id]) for cell_id in plan.cell_ids)
        or not _is_sha256(journal_head_sha256)
        or type(journal_event_count) is not int
        or journal_event_count < len(plan.cell_ids) * 2
    ):
        _fail("action_calibration_output_seal_invalid")
    rows = [
        {
            "cell_id": cell_id,
            "result_sha256": result_sha256_by_cell[cell_id],
        }
        for cell_id in plan.cell_ids
    ]
    return {
        "schema": ACTION_CALIBRATION_OUTPUT_SEAL_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "cell_count": len(rows),
        "journal_head_sha256": journal_head_sha256,
        "journal_event_count": journal_event_count,
        "sealed_results_sha256": _sha256(rows),
        "sealed_results": rows,
        "answer_material_revealed": False,
    }


def _metadata(plan: CampaignPlan) -> dict[str, Any]:
    if not isinstance(plan, CampaignPlan):
        _fail("action_calibration_plan_invalid")
    metadata = plan.to_dict().get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != ACTION_CALIBRATION_PROTOCOL_SCHEMA
        or metadata.get("action_intervention_required") is not True
        or type(metadata.get("strict_execution_order")) is not bool
        or not isinstance(metadata.get("campaign_design"), dict)
        or not _is_sha256(metadata.get("campaign_design_sha256"))
        or metadata["campaign_design"].get("campaign_design_sha256")
        != metadata["campaign_design_sha256"]
        or (
            metadata.get("claim_eligible") is True
            and metadata.get("strict_execution_order") is not True
        )
    ):
        _fail("action_calibration_plan_metadata_invalid")
    return metadata


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    plan: CampaignPlan,
) -> dict[str, Any]:
    required = {
        "schema",
        "manifest_version",
        "plan_sha256",
        "journal_head_sha256",
        "journal_event_count",
        "journal_size_bytes",
        "cell_count",
        "cells",
        "manifest_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        _fail("action_calibration_manifest_invalid")
    normalized = dict(manifest)
    material = {name: normalized[name] for name in required - {"manifest_sha256"}}
    if (
        normalized.get("schema") != "aura.latent_cortex.campaign_manifest.v1"
        or normalized.get("manifest_version") != 1
        or normalized.get("plan_sha256") != plan.plan_sha256
        or not _is_sha256(normalized.get("journal_head_sha256"))
        or type(normalized.get("journal_event_count")) is not int
        or normalized["journal_event_count"] < len(plan.cell_ids) * 4
        or type(normalized.get("journal_size_bytes")) is not int
        or normalized["journal_size_bytes"] <= 0
        or normalized.get("cell_count") != len(plan.cell_ids)
        or normalized.get("manifest_sha256") != _sha256(material)
    ):
        _fail("action_calibration_manifest_invalid")
    cells = normalized.get("cells")
    if (
        not isinstance(cells, list)
        or len(cells) != len(plan.cell_ids)
        or [cell.get("cell_id") for cell in cells] != list(plan.cell_ids)
    ):
        _fail("action_calibration_manifest_cell_coverage_invalid")
    for cell in cells:
        if (
            not isinstance(cell, Mapping)
            or set(cell)
            != {
                "arm_result_event_sha256",
                "attempt_id",
                "cell_id",
                "commit_event_sha256",
                "verified_event_sha256",
            }
            or not all(
                _is_sha256(cell[name])
                for name in (
                    "arm_result_event_sha256",
                    "commit_event_sha256",
                    "verified_event_sha256",
                )
            )
            or not isinstance(cell.get("attempt_id"), str)
            or not cell["attempt_id"]
        ):
            _fail("action_calibration_manifest_cell_invalid")
    return normalized


def _validate_journal_transcript(
    value: Any,
    *,
    plan: CampaignPlan,
    manifest: Mapping[str, Any],
    output_seal: Mapping[str, Any],
    supplied_records: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    event_fields = {
        "schema",
        "sequence",
        "plan_sha256",
        "previous_event_sha256",
        "event",
        "cell_id",
        "attempt_id",
        "payload",
        "event_sha256",
    }
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != manifest["journal_event_count"]
    ):
        _fail("action_calibration_journal_transcript_invalid")
    transcript: list[dict[str, Any]] = []
    state: dict[str, dict[str, Any]] = {}
    previous_event_sha256 = "0" * 64
    seal_head_sha256: str | None = None
    next_execution_ordinal = 0
    seal_event_count = output_seal.get("journal_event_count")
    if type(seal_event_count) is not int:
        _fail("action_calibration_journal_transcript_invalid")
    for sequence, raw_event in enumerate(value):
        if not isinstance(raw_event, Mapping) or set(raw_event) != event_fields:
            _fail("action_calibration_journal_event_invalid")
        event = dict(raw_event)
        base = {name: event[name] for name in event_fields - {"event_sha256"}}
        if (
            event.get("schema") != EVENT_SCHEMA
            or event.get("sequence") != sequence
            or event.get("plan_sha256") != plan.plan_sha256
            or event.get("previous_event_sha256") != previous_event_sha256
            or event.get("event_sha256") != _sha256(base)
        ):
            _fail("action_calibration_journal_chain_invalid")
        event_name = event.get("event")
        cell_id = event.get("cell_id")
        attempt_id = event.get("attempt_id")
        payload = event.get("payload")
        if sequence == 0:
            if (
                event_name != PLAN_EVENT
                or cell_id is not None
                or attempt_id is not None
                or not _strict_equal(payload, {"plan": plan.to_dict()})
            ):
                _fail("action_calibration_journal_genesis_invalid")
        else:
            if (
                not isinstance(cell_id, str)
                or cell_id not in plan.cell_ids
                or not isinstance(attempt_id, str)
                or not isinstance(payload, Mapping)
            ):
                _fail("action_calibration_journal_event_invalid")
            expected_attempt_id = "attempt-" + _sha256(
                {
                    "attempt_number": 1,
                    "cell_id": cell_id,
                    "plan_sha256": plan.plan_sha256,
                    "schema": "aura.latent_cortex.campaign_attempt.v1",
                }
            )
            if attempt_id != expected_attempt_id:
                _fail("action_calibration_journal_attempt_invalid")
            cell_state = state.get(cell_id)
            if event_name == STARTED:
                definition = plan.cell_definition(cell_id)
                if (
                    cell_state is not None
                    or not _strict_equal(payload, {"attempt_number": 1})
                    or definition.get("execution_ordinal") != next_execution_ordinal
                ):
                    _fail("action_calibration_journal_transition_invalid")
                state[cell_id] = {
                    "state": STARTED,
                    "attempt_id": attempt_id,
                }
                next_execution_ordinal += 1
            elif event_name == ACTION_INTERVENTION_CLAIMED:
                claim_fields = {
                    "intervention_sha256",
                    "request_payload_sha256",
                    "signed_journal_head_sha256",
                    "signed_journal_event_count",
                }
                if (
                    cell_state is None
                    or cell_state["state"] != STARTED
                    or set(payload) != claim_fields
                    or not _is_sha256(payload.get("intervention_sha256"))
                    or not _is_sha256(payload.get("request_payload_sha256"))
                    or payload.get("signed_journal_head_sha256") != event["previous_event_sha256"]
                    or payload.get("signed_journal_event_count") != sequence
                ):
                    _fail("action_calibration_journal_transition_invalid")
                cell_state.update(
                    {
                        "state": ACTION_INTERVENTION_CLAIMED,
                        "action_intervention_claim": dict(payload),
                        "action_intervention_claim_event_sha256": event["event_sha256"],
                    }
                )
            elif event_name == ARM_RESULT:
                if (
                    cell_state is None
                    or cell_state["state"] != ACTION_INTERVENTION_CLAIMED
                    or set(payload) != {"result"}
                    or not isinstance(payload.get("result"), Mapping)
                ):
                    _fail("action_calibration_journal_transition_invalid")
                cell_state.update(
                    {
                        "state": ARM_RESULT,
                        "result": dict(payload["result"]),
                        "arm_result_event_sha256": event["event_sha256"],
                    }
                )
            elif event_name == VERIFIED:
                if (
                    cell_state is None
                    or cell_state["state"] != ARM_RESULT
                    or set(payload) != {"verification"}
                    or not isinstance(payload.get("verification"), Mapping)
                ):
                    _fail("action_calibration_journal_transition_invalid")
                cell_state.update(
                    {
                        "state": VERIFIED,
                        "verification": dict(payload["verification"]),
                        "verified_event_sha256": event["event_sha256"],
                    }
                )
            elif event_name == COMMITTED:
                if (
                    cell_state is None
                    or cell_state["state"] != VERIFIED
                    or set(payload) != {"commit"}
                    or not isinstance(payload.get("commit"), Mapping)
                ):
                    _fail("action_calibration_journal_transition_invalid")
                cell_state.update(
                    {
                        "state": COMMITTED,
                        "commit": dict(payload["commit"]),
                        "commit_event_sha256": event["event_sha256"],
                    }
                )
            else:
                _fail("action_calibration_journal_event_invalid")
        previous_event_sha256 = cast(str, event["event_sha256"])
        transcript.append(event)
        if sequence + 1 == seal_event_count:
            if set(state) != set(plan.cell_ids) or any(
                cell["state"] != ARM_RESULT for cell in state.values()
            ):
                _fail("action_calibration_journal_seal_prefix_invalid")
            seal_head_sha256 = previous_event_sha256
    if (
        set(state) != set(plan.cell_ids)
        or any(cell["state"] != COMMITTED for cell in state.values())
        or previous_event_sha256 != manifest["journal_head_sha256"]
        or seal_head_sha256 != output_seal.get("journal_head_sha256")
        or manifest["journal_size_bytes"]
        != sum(len(canonical_json_bytes(event)) + 1 for event in transcript)
    ):
        _fail("action_calibration_journal_final_state_invalid")
    manifest_cells = {row["cell_id"]: row for row in manifest["cells"]}
    records: dict[str, dict[str, Any]] = {}
    intervention_claims: dict[str, dict[str, Any]] = {}
    for cell_id in plan.cell_ids:
        cell = state[cell_id]
        if (
            set(cell["commit"]) != {"result_sha256", "verification_sha256"}
            or cell["commit"].get("result_sha256") != _sha256(cell["result"])
            or cell["commit"].get("verification_sha256") != _sha256(cell["verification"])
        ):
            _fail("action_calibration_journal_commit_binding_invalid")
        expected_manifest_cell = {
            "arm_result_event_sha256": cell["arm_result_event_sha256"],
            "attempt_id": cell["attempt_id"],
            "cell_id": cell_id,
            "commit_event_sha256": cell["commit_event_sha256"],
            "verified_event_sha256": cell["verified_event_sha256"],
        }
        if not _strict_equal(
            manifest_cells.get(cell_id),
            expected_manifest_cell,
        ):
            _fail("action_calibration_journal_manifest_binding_invalid")
        records[cell_id] = {
            "cell_id": cell_id,
            "attempt_id": cell["attempt_id"],
            "definition": plan.cell_definition(cell_id),
            "result": cell["result"],
            "verification": cell["verification"],
            "commit": cell["commit"],
        }
        intervention_claims[cell_id] = {
            "claim": cell["action_intervention_claim"],
            "claim_event_sha256": cell["action_intervention_claim_event_sha256"],
        }
    if supplied_records is not None:
        supplied: dict[str, Mapping[str, Any]] = {}
        for record in supplied_records:
            if (
                not isinstance(record, Mapping)
                or not isinstance(record.get("cell_id"), str)
                or record["cell_id"] in supplied
            ):
                _fail("action_calibration_journal_records_invalid")
            supplied[cast(str, record["cell_id"])] = record
        if not _strict_equal(supplied, records):
            _fail("action_calibration_journal_records_mismatch")
    return transcript, records, intervention_claims


def _rational(value: Rational) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _cost_upper_bound(costs: Sequence[float]) -> float:
    if not costs:
        _fail("action_calibration_costs_empty")
    mean = sum(costs) / len(costs)
    alpha = DEFAULT_FAMILY_ALPHA.numerator / DEFAULT_FAMILY_ALPHA.denominator
    radius = math.sqrt(math.log((2.0 * GLOBAL_BOUND_FAMILY_COUNT) / alpha) / (2.0 * len(costs)))
    # Round away from the data so serialization cannot make the bound tighter.
    return min(1.0, math.ceil((mean + radius) * 10**12) / 10**12)


def _evidence_cells(
    observations: Sequence[Mapping[str, Any]],
    *,
    action_resource_caps: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    by_action: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        by_action[cast(str, observation["action"])].append(observation)
    if set(by_action) != {action.value for action in OperationKind}:
        _fail("action_calibration_observation_coverage_invalid")
    cells: dict[str, dict[str, Any]] = {}
    for action in sorted(by_action):
        rows = sorted(by_action[action], key=lambda row: cast(str, row["task_id"]))
        if len(rows) < MIN_UNIQUE_TASKS_PER_ACTION:
            _fail("action_calibration_action_underpowered")
        wins = sum(int(row["treatment_success"]) > int(row["control_success"]) for row in rows)
        losses = sum(int(row["treatment_success"]) < int(row["control_success"]) for row in rows)
        ties = len(rows) - wins - losses
        bounds = certified_rational_effect_bounds(
            wins,
            losses,
            ties,
            family_count=GLOBAL_BOUND_FAMILY_COUNT,
            family_alpha=DEFAULT_FAMILY_ALPHA,
        )
        costs = [
            _normalized_action_cost(
                cast(Mapping[str, Any], row["treatment_action_resources"]),
                caps=action_resource_caps,
            )
            for row in rows
        ]
        measured = len(rows) >= MIN_CERTIFIED_TASKS_PER_ACTION and bounds.certified
        cells[action] = {
            "n": len(rows),
            "unique_task_count": len({row["task_id"] for row in rows}),
            "measured": measured,
            "gain_mean": round((wins - losses) / len(rows), 12),
            "gain_lcb": round(
                bounds.lower.numerator / bounds.lower.denominator,
                12,
            ),
            "gain_ucb": round(
                bounds.upper.numerator / bounds.upper.denominator,
                12,
            ),
            "cost_mean": round(sum(costs) / len(costs), 12),
            "cost_ucb": (
                _cost_upper_bound(costs) if len(rows) >= MIN_CERTIFIED_TASKS_PER_ACTION else 1.0
            ),
            "gain_bounds": {
                "method": _GAIN_BOUND_METHOD,
                "family_count": GLOBAL_BOUND_FAMILY_COUNT,
                "family_alpha": _rational(bounds.family_alpha),
                "component_alpha": _rational(bounds.component_alpha),
                "simultaneous_coverage_lower": _rational(bounds.simultaneous_coverage_lower),
                "lower": _rational(bounds.lower),
                "upper": _rational(bounds.upper),
                "certified": bounds.certified,
            },
            "cost_bounds": {
                "method": _COST_BOUND_METHOD,
                "family_count": GLOBAL_BOUND_FAMILY_COUNT,
                "family_alpha": _rational(DEFAULT_FAMILY_ALPHA),
                "bounded_interval": [0.0, 1.0],
                "normalization": _COST_NORMALIZATION,
                "dimensions": list(ACTION_RESOURCE_DIMENSIONS),
            },
        }
    return cells


def _validate_action_intervention_evidence(
    value: Any,
    *,
    plan: CampaignPlan,
    policy: VerifiedCampaignTrustPolicy,
    cell_id: str,
    attempt_id: str,
    action_intervention_claim: Mapping[str, Any],
    campaign_journal: Sequence[Mapping[str, Any]],
    issuer_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_fields = {
        "schema",
        "authority_payload",
        "runner_attestation",
        "intervention_sha256",
        "worker_receipt",
        "cognitive_action_trace",
        "evidence_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != evidence_fields:
        _fail("action_calibration_intervention_evidence_invalid")
    body = {name: value[name] for name in evidence_fields - {"evidence_sha256"}}
    if value.get("schema") != ACTION_CALIBRATION_INTERVENTION_EVIDENCE_SCHEMA or value.get(
        "evidence_sha256"
    ) != _sha256(body):
        _fail("action_calibration_intervention_evidence_invalid")
    definition = plan.cell_definition(cell_id)
    metadata = _metadata(plan)
    authority = value.get("authority_payload")
    claim = action_intervention_claim.get("claim")
    claim_event_sha256 = action_intervention_claim.get("claim_event_sha256")
    trace = value.get("cognitive_action_trace")
    if (
        not isinstance(authority, Mapping)
        or not isinstance(claim, Mapping)
        or not _is_sha256(claim_event_sha256)
        or not isinstance(trace, Sequence)
        or isinstance(trace, (str, bytes))
    ):
        _fail("action_calibration_intervention_evidence_invalid")
    try:
        from core.brain.llm.latent_cortex.action_intervention import (
            ACTION_INTERVENTION_SCHEMA,
            action_intervention_authority_payload,
            validate_action_intervention_receipt_authority,
        )

        starting_state = cast(Mapping[str, Any], definition["starting_state"])
        component_names = (
            "latent_slots_sha256",
            "branch_state_sha256",
            "kv_cache_sha256",
            "evidence_state_sha256",
            "memory_state_sha256",
            "public_action_state_sha256",
            "durable_state_sha256",
            "rng_state_sha256",
        )
        components = {name: starting_state[name] for name in component_names}
        task_rows = cast(Mapping[str, Any], metadata["task_manifest"])["tasks"]
        task = next(
            row
            for row in task_rows
            if isinstance(row, Mapping) and row.get("task_id") == definition["task_id"]
        )
        normalized_authority = action_intervention_authority_payload(
            campaign_name=plan.campaign_name,
            campaign_plan_sha256=plan.plan_sha256,
            campaign_protocol_sha256=policy.document["protocol_sha256"],
            policy_sha256=policy.policy_sha256,
            policy_revision=policy.document["policy_revision"],
            cell_id=cell_id,
            definition_sha256=_sha256(definition),
            pair_id=definition["pair_id"],
            task_id=definition["task_id"],
            task_payload_sha256=definition["task_payload_sha256"],
            starting_state_sha256=definition["starting_state_sha256"],
            starting_state_components=components,
            expected_pre_state_sha256=_sha256(
                {name: components[name] for name in sorted(components)}
            ),
            expected_pre_kv_sha256=components["kv_cache_sha256"],
            action=definition["action"],
            arm=definition["arm"],
            execution_ordinal=definition["execution_ordinal"],
            attempt_number=1,
            attempt_id=attempt_id,
            campaign_journal_path_sha256=authority.get("campaign_journal_path_sha256"),
            journal_head_sha256=claim.get("signed_journal_head_sha256"),
            journal_event_count=claim.get("signed_journal_event_count"),
            request_payload_sha256=authority.get("request_payload_sha256"),
            engine_request_sha256=authority.get("engine_request_sha256"),
            task_prompt_sha256=hashlib.sha256(
                cast(str, task["prompt"]).encode("utf-8")
            ).hexdigest(),
        )
        if not _strict_equal(authority, normalized_authority):
            _fail("action_calibration_intervention_authority_mismatch")
        prefix_count = normalized_authority["journal_event_count"]
        prefix = campaign_journal[:prefix_count]
        if (
            len(prefix) != prefix_count
            or not prefix
            or prefix[-1].get("event_sha256") != normalized_authority["journal_head_sha256"]
            or claim.get("request_payload_sha256") != normalized_authority["request_payload_sha256"]
        ):
            _fail("action_calibration_intervention_journal_binding_invalid")
        intervention_body = {
            "schema": ACTION_INTERVENTION_SCHEMA,
            "authority_payload": normalized_authority,
            "campaign_plan": plan.to_dict(),
            "campaign_journal_prefix": [dict(row) for row in prefix],
            "policy_document": dict(policy.document),
            "task_issuer_attestation": dict(issuer_attestation),
            "runner_attestation": dict(value["runner_attestation"]),
        }
        if (
            value["intervention_sha256"] != _sha256(intervention_body)
            or claim.get("intervention_sha256") != value["intervention_sha256"]
        ):
            _fail("action_calibration_intervention_envelope_mismatch")
        verify_role_attestation(
            policy,
            value["runner_attestation"],
            role=CAMPAIGN_RUNNER,
            expected_payload=normalized_authority,
        )
        receipt = validate_action_intervention_receipt_authority(
            value["worker_receipt"],
            authority_payload=normalized_authority,
            intervention_sha256=cast(str, value["intervention_sha256"]),
            cognitive_action_trace=cast(Sequence[Mapping[str, Any]], trace),
        )
    except ActionCalibrationError:
        raise
    except (KeyError, StopIteration, TypeError, ValueError) as exc:
        raise ActionCalibrationError("action_calibration_intervention_evidence_invalid") from exc
    consumption = receipt["consumption_event"]
    if (
        consumption.get("campaign_claim_event_sha256") != claim_event_sha256
        or receipt.get("cell_id") != cell_id
        or receipt.get("attempt_id") != attempt_id
        or receipt.get("campaign_plan_sha256") != plan.plan_sha256
        or receipt.get("execution_ordinal") != definition["execution_ordinal"]
        or receipt.get("arm") != definition["arm"]
        or receipt.get("action") != definition["action"]
    ):
        _fail("action_calibration_intervention_receipt_binding_invalid")
    return {
        **dict(value),
        "authority_payload": normalized_authority,
        "worker_receipt": receipt,
        "cognitive_action_trace": [dict(row) for row in trace],
    }


def _validated_record(
    record: Mapping[str, Any],
    *,
    plan: CampaignPlan,
    policy: VerifiedCampaignTrustPolicy,
    issuer_tasks: Mapping[str, FrontierTask],
    output_sealed_at_unix: int,
    action_intervention_claim: Mapping[str, Any],
    campaign_journal: Sequence[Mapping[str, Any]],
    issuer_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(record, Mapping) or set(record) != {
        "cell_id",
        "attempt_id",
        "definition",
        "result",
        "verification",
        "commit",
    }:
        _fail("action_calibration_record_invalid")
    cell_id = record.get("cell_id")
    if (
        not isinstance(cell_id, str)
        or cell_id not in plan.cell_ids
        or not isinstance(record.get("attempt_id"), str)
        or not record["attempt_id"]
    ):
        _fail("action_calibration_record_cell_invalid")
    definition = plan.cell_definition(cell_id)
    if not _strict_equal(record.get("definition"), definition):
        _fail("action_calibration_record_definition_mismatch")
    result = record.get("result")
    verification = record.get("verification")
    commit = record.get("commit")
    if (
        not isinstance(result, Mapping)
        or not isinstance(verification, Mapping)
        or not isinstance(commit, Mapping)
    ):
        _fail("action_calibration_record_shape_invalid")
    result_fields = {
        "schema",
        "arm",
        "action",
        "pair_id",
        "campaign_plan_sha256",
        "attempt_id",
        "starting_state_sha256",
        "starting_state",
        "action_intervention_evidence",
        "action_execution",
        "action_trace",
        "text",
        "output_sha256",
        "runtime_identity",
        "resource_accounting",
        "action_resource_accounting",
        "available_information_accounting",
        "consumed_information_accounting",
        "host_telemetry",
        "mutation_erasure",
        "runner_attestation",
    }
    if set(result) != result_fields:
        _fail("action_calibration_result_fields_invalid")
    core = {name: result[name] for name in result_fields - {"runner_attestation"}}
    metadata = _metadata(plan)
    action = definition["action"]
    arm = definition["arm"]
    task = issuer_tasks.get(definition["task_id"])
    if task is None:
        _fail("action_calibration_issuer_task_missing")
    _validate_starting_state_receipt(
        definition.get("starting_state"),
        campaign_name=plan.campaign_name,
        action=_operation(action),
        task=task.public,
        model_identity=cast(Mapping[str, Any], metadata["model_identity"]),
        execution_config=cast(Mapping[str, Any], metadata["execution_config"]),
        calibration_bucket=cast(str, metadata["calibration_bucket"]),
        campaign_design_sha256=cast(
            str,
            metadata["campaign_design_sha256"],
        ),
        policy=policy,
    )
    execution = result.get("action_execution")
    intervention_evidence = _validate_action_intervention_evidence(
        result.get("action_intervention_evidence"),
        plan=plan,
        policy=policy,
        cell_id=cell_id,
        attempt_id=record["attempt_id"],
        action_intervention_claim=action_intervention_claim,
        campaign_journal=campaign_journal,
        issuer_attestation=issuer_attestation,
    )
    worker_receipt = intervention_evidence["worker_receipt"]
    expected_execution = (
        {
            "selection_mode": "campaign_forced",
            "selected_action": action,
            "campaign_authority_sha256": plan.plan_sha256,
        }
        if arm == TREATMENT_ARM
        else {
            "selection_mode": "matched_no_action_control",
            "selected_action": None,
            "campaign_authority_sha256": plan.plan_sha256,
        }
    )
    text = result.get("text")
    trace = result.get("action_trace")
    expected_trace = {
        "schema": "aura.rlc.action_calibration.action_trace.v1",
        "action": action,
        "intervention_ordinal": 0,
        "selected_action_occurrences": 1 if arm == TREATMENT_ARM else 0,
        "action_excluded_at_intervention": arm == CONTROL_ARM,
        "pre_state_sha256": worker_receipt["pre_state_sha256"],
        "post_state_sha256": worker_receipt["post_state_sha256"],
    }
    telemetry = result.get("host_telemetry")
    erasure = result.get("mutation_erasure")
    if (
        result.get("schema") != ACTION_CALIBRATION_RESULT_SCHEMA
        or result.get("arm") != arm
        or result.get("action") != action
        or result.get("pair_id") != definition["pair_id"]
        or result.get("campaign_plan_sha256") != plan.plan_sha256
        or result.get("attempt_id") != record["attempt_id"]
        or result.get("starting_state_sha256") != definition["starting_state_sha256"]
        or not _strict_equal(result.get("starting_state"), definition["starting_state"])
        or not _strict_equal(execution, expected_execution)
        or not _strict_equal(trace, expected_trace)
        or execution["selection_mode"] != worker_receipt["selection_mode"]
        or execution["selected_action"] != worker_receipt["selected_action"]
        or trace["selected_action_occurrences"] != worker_receipt["selected_action_occurrences"]
        or trace["action_excluded_at_intervention"] != (worker_receipt["selected_action"] is None)
        or not isinstance(text, str)
        or not text
        or result.get("output_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest()
        or not _strict_equal(result.get("runtime_identity"), metadata["model_identity"])
        or not isinstance(telemetry, Mapping)
        or set(telemetry)
        != {
            "schema",
            "instrumentation_sha256",
            "monotonic_start_ns",
            "monotonic_end_ns",
            "cpu_time_ns",
            "peak_resident_bytes",
            "sample_sha256",
            "complete",
        }
        or telemetry.get("schema") != "aura.rlc.action_calibration.host_telemetry.v1"
        or telemetry.get("instrumentation_sha256")
        != metadata["execution_config"]["instrumentation_sha256"]
        or type(telemetry.get("monotonic_start_ns")) is not int
        or type(telemetry.get("monotonic_end_ns")) is not int
        or telemetry["monotonic_end_ns"] < telemetry["monotonic_start_ns"]
        or type(telemetry.get("cpu_time_ns")) is not int
        or telemetry["cpu_time_ns"] < 0
        or type(telemetry.get("peak_resident_bytes")) is not int
        or telemetry["peak_resident_bytes"] <= 0
        or telemetry.get("complete") is not True
        or telemetry.get("sample_sha256")
        != _sha256({name: telemetry[name] for name in telemetry if name != "sample_sha256"})
        or not isinstance(erasure, Mapping)
        or set(erasure)
        != {
            "schema",
            "pre_durable_state_sha256",
            "post_durable_state_sha256",
            "transient_state_erased",
            "receipt_sha256",
        }
        or erasure.get("schema") != "aura.rlc.action_calibration.mutation_erasure.v1"
        or erasure.get("pre_durable_state_sha256")
        != definition["starting_state"]["durable_state_sha256"]
        or erasure.get("post_durable_state_sha256")
        != definition["starting_state"]["durable_state_sha256"]
        or erasure.get("transient_state_erased") is not True
        or erasure.get("receipt_sha256")
        != _sha256({name: erasure[name] for name in erasure if name != "receipt_sha256"})
    ):
        _fail("action_calibration_result_binding_invalid")
    try:
        resource = validate_resource_receipt(result["resource_accounting"])
        action_resource = validate_resource_receipt(result["action_resource_accounting"])
        information = validate_information_receipt(result["available_information_accounting"])
        consumed_information = validate_information_receipt(
            result["consumed_information_accounting"]
        )
    except (TypeError, ValueError) as exc:
        raise ActionCalibrationError("action_calibration_accounting_invalid") from exc
    if (
        resource["accounting_complete"] is not True
        or action_resource["accounting_complete"] is not True
        or information["accounting_complete"] is not True
        or consumed_information["accounting_complete"] is not True
        or type(resource["estimated_flops"]) is not int
        or resource["estimated_flops"] <= 0
        or type(action_resource["estimated_flops"]) is not int
        or action_resource["estimated_flops"] < 0
        or action_resource["model_profile"]["profile_sha256"]
        != resource["model_profile"]["profile_sha256"]
        or action_resource["estimated_flops"] > resource["estimated_flops"]
        or any(
            action_resource["totals"][counter] > resource["totals"][counter]
            for counter in RESOURCE_COUNTERS
        )
        or (arm == CONTROL_ARM and any(action_resource["totals"].values()))
        or action_resource["estimated_flops"]
        > metadata["execution_config"]["action_cost_budget_estimated_flops"]
    ):
        _fail("action_calibration_accounting_incomplete")
    action_resources = {
        "estimated_flops": action_resource["estimated_flops"],
        **{counter: action_resource["totals"][counter] for counter in RESOURCE_COUNTERS},
    }
    _normalized_action_cost(
        action_resources,
        caps=_action_resource_caps(metadata["execution_config"].get("action_resource_caps")),
    )
    runner_payload = action_calibration_runner_payload(
        plan=plan,
        cell_id=cell_id,
        attempt_id=record["attempt_id"],
        result_core=core,
    )
    verify_role_attestation(
        policy,
        result["runner_attestation"],
        role=CAMPAIGN_RUNNER,
        expected_payload=runner_payload,
    )

    verification_fields = {
        "schema",
        "correct",
        "score_receipt",
        "answer_commitment_sha256",
        "result_sha256",
        "verifier_attestation",
    }
    if set(verification) != verification_fields:
        _fail("action_calibration_verification_fields_invalid")
    score = verification.get("score_receipt")
    independent_score = task.score(text).to_dict() if task is not None else None
    result_sha256 = _sha256(dict(result))
    if (
        task is None
        or not isinstance(score, Mapping)
        or not _strict_equal(score, independent_score)
        or type(verification.get("correct")) is not bool
        or verification["correct"] is not independent_score["correct"]
        or verification.get("answer_commitment_sha256") != task.public.answer_commitment_sha256
        or verification.get("result_sha256") != result_sha256
        or commit.get("result_sha256") != result_sha256
        or commit.get("verification_sha256") != _sha256(dict(verification))
    ):
        _fail("action_calibration_verification_binding_invalid")
    verifier_payload = action_calibration_verifier_payload(
        plan=plan,
        cell_id=cell_id,
        result_sha256=result_sha256,
        score_receipt=cast(Mapping[str, Any], score),
        answer_commitment_sha256=task.public.answer_commitment_sha256,
    )
    verify_role_attestation(
        policy,
        verification["verifier_attestation"],
        role=EVIDENCE_VERIFIER,
        expected_payload=verifier_payload,
        not_before_unix=output_sealed_at_unix,
    )
    return {
        "cell_id": cell_id,
        "attempt_id": record["attempt_id"],
        "action": action,
        "arm": arm,
        "pair_id": definition["pair_id"],
        "task_id": definition["task_id"],
        "starting_state_sha256": definition["starting_state_sha256"],
        "correct": verification["correct"],
        "estimated_flops": resource["estimated_flops"],
        "action_estimated_flops": action_resource["estimated_flops"],
        "action_resources": action_resources,
        "resource_accounting": resource,
        "information_accounting": information,
        "consumed_information_accounting": consumed_information,
        "result_sha256": result_sha256,
        "score_receipt_sha256": _sha256(dict(score)),
        "answer_commitment_sha256": task.public.answer_commitment_sha256,
        "runner_attestation": dict(result["runner_attestation"]),
        "verifier_attestation": dict(verification["verifier_attestation"]),
    }


def build_action_calibration_candidate(
    records: Iterable[Mapping[str, Any]],
    *,
    plan: CampaignPlan,
    issuer_tasks: Sequence[FrontierTask],
    campaign_manifest: Mapping[str, Any],
    campaign_journal: Sequence[Mapping[str, Any]],
    policy: VerifiedCampaignTrustPolicy,
    issuer_attestation: Mapping[str, Any],
    contamination_attestation: Mapping[str, Any],
    contamination_payload: Mapping[str, Any],
    output_seal_payload: Mapping[str, Any],
    output_seal_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete preverified body that an independent verifier signs."""

    metadata = _metadata(plan)
    _validate_plan_sampling_frame(plan)
    if (
        metadata.get("claim_eligible") is not True
        or not isinstance(policy, VerifiedCampaignTrustPolicy)
        or policy.document.get("campaign_name") != plan.campaign_name
        or metadata.get("campaign_trust", {}).get("policy_sha256") != policy.policy_sha256
        or not externally_custodied_roles(policy)
    ):
        _fail("action_calibration_claim_trust_invalid")
    manifest = _validate_manifest(campaign_manifest, plan=plan)
    issuer_by_id = {task.task_id: task for task in issuer_tasks}
    planned_tasks = metadata["task_manifest"]["tasks"]
    if (
        len(issuer_by_id) != len(planned_tasks)
        or set(issuer_by_id) != {task["task_id"] for task in planned_tasks}
        or any(
            not _strict_equal(issuer_by_id[task["task_id"]].public.to_dict(), task)
            for task in planned_tasks
        )
    ):
        _fail("action_calibration_issuer_tasks_mismatch")
    issuer_payload = action_calibration_issuer_payload(plan)
    verify_role_attestation(
        policy,
        issuer_attestation,
        role=TASK_ISSUER,
        expected_payload=issuer_payload,
    )
    expected_contamination = action_calibration_contamination_payload(
        plan,
        corpus_snapshot_sha256=cast(str, contamination_payload.get("corpus_snapshot_sha256")),
        methods=cast(Sequence[str], contamination_payload.get("methods")),
    )
    if not _strict_equal(contamination_payload, expected_contamination):
        _fail("action_calibration_contamination_payload_invalid")
    verify_role_attestation(
        policy,
        contamination_attestation,
        role=CONTAMINATION_AUDITOR,
        expected_payload=expected_contamination,
    )
    result_sha256_by_cell: dict[str, str] = {}
    raw_records = tuple(records)
    for raw_record in raw_records:
        if (
            not isinstance(raw_record, Mapping)
            or not isinstance(raw_record.get("cell_id"), str)
            or not isinstance(raw_record.get("result"), Mapping)
            or raw_record["cell_id"] in result_sha256_by_cell
        ):
            _fail("action_calibration_output_seal_records_invalid")
        result_sha256_by_cell[cast(str, raw_record["cell_id"])] = _sha256(
            dict(cast(Mapping[str, Any], raw_record["result"]))
        )
    expected_output_seal = action_calibration_output_seal_payload(
        plan,
        result_sha256_by_cell=result_sha256_by_cell,
        journal_head_sha256=cast(str, output_seal_payload.get("journal_head_sha256")),
        journal_event_count=cast(int, output_seal_payload.get("journal_event_count")),
    )
    if not _strict_equal(output_seal_payload, expected_output_seal):
        _fail("action_calibration_output_seal_invalid")
    if (
        manifest["journal_event_count"]
        != expected_output_seal["journal_event_count"] + len(plan.cell_ids) * 2
        or manifest["journal_head_sha256"] == expected_output_seal["journal_head_sha256"]
    ):
        _fail("action_calibration_post_seal_journal_invalid")
    journal_transcript, journal_records, intervention_claims = _validate_journal_transcript(
        campaign_journal,
        plan=plan,
        manifest=manifest,
        output_seal=expected_output_seal,
        supplied_records=raw_records,
    )
    sealed_attestation = verify_role_attestation(
        policy,
        output_seal_attestation,
        role=CAMPAIGN_RUNNER,
        expected_payload=expected_output_seal,
    )
    output_sealed_at_unix = sealed_attestation["signed_at_unix"]

    normalized_records: dict[str, dict[str, Any]] = {}
    manifest_attempts = {row["cell_id"]: row["attempt_id"] for row in manifest["cells"]}
    for cell_id in plan.cell_ids:
        raw_record = journal_records[cell_id]
        row = _validated_record(
            raw_record,
            plan=plan,
            policy=policy,
            issuer_tasks=issuer_by_id,
            output_sealed_at_unix=output_sealed_at_unix,
            action_intervention_claim=intervention_claims[cell_id],
            campaign_journal=journal_transcript,
            issuer_attestation=issuer_attestation,
        )
        if row["cell_id"] in normalized_records:
            _fail("action_calibration_record_duplicate")
        if manifest_attempts.get(row["cell_id"]) != row["attempt_id"]:
            _fail("action_calibration_manifest_attempt_mismatch")
        normalized_records[row["cell_id"]] = row
    if set(normalized_records) != set(plan.cell_ids):
        _fail("action_calibration_campaign_incomplete")

    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for cell_id in plan.cell_ids:
        row = normalized_records[cell_id]
        if row["arm"] in pairs[row["pair_id"]]:
            _fail("action_calibration_pair_arm_duplicate")
        pairs[row["pair_id"]][row["arm"]] = row
    if len(pairs) != metadata["pair_count"] or any(
        set(arms) != set(CALIBRATION_ARMS) for arms in pairs.values()
    ):
        _fail("action_calibration_pair_coverage_invalid")

    observations: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for pair_id in sorted(pairs):
        treatment = pairs[pair_id][TREATMENT_ARM]
        control = pairs[pair_id][CONTROL_ARM]
        if (
            treatment["action"] != control["action"]
            or treatment["task_id"] != control["task_id"]
            or treatment["starting_state_sha256"] != control["starting_state_sha256"]
            or treatment["task_id"] in seen_tasks
        ):
            _fail("action_calibration_pair_binding_invalid")
        seen_tasks.add(treatment["task_id"])
        accounting = certify_comparison_accounting(
            treatment_resource=treatment["resource_accounting"],
            control_resource=control["resource_accounting"],
            treatment_information=treatment["information_accounting"],
            control_information=control["information_accounting"],
            tolerance_numerator=1,
            tolerance_denominator=1,
            require_compute_parity=False,
        )
        if accounting["admitted"] is not True:
            _fail("action_calibration_pair_accounting_rejected")
        observations.append(
            {
                "action": treatment["action"],
                "pair_id": pair_id,
                "task_id": treatment["task_id"],
                "starting_state_sha256": treatment["starting_state_sha256"],
                "treatment_success": treatment["correct"],
                "control_success": control["correct"],
                "treatment_estimated_flops": treatment["estimated_flops"],
                "control_estimated_flops": control["estimated_flops"],
                "treatment_action_estimated_flops": treatment["action_estimated_flops"],
                "control_action_estimated_flops": control["action_estimated_flops"],
                "treatment_action_resources": treatment["action_resources"],
                "control_action_resources": control["action_resources"],
                "accounting_certificate": accounting,
                "treatment_result_sha256": treatment["result_sha256"],
                "control_result_sha256": control["result_sha256"],
                "treatment_score_receipt_sha256": treatment["score_receipt_sha256"],
                "control_score_receipt_sha256": control["score_receipt_sha256"],
                "answer_commitment_sha256": treatment["answer_commitment_sha256"],
                "treatment_consumed_information_sha256": treatment[
                    "consumed_information_accounting"
                ]["receipt_sha256"],
                "control_consumed_information_sha256": control["consumed_information_accounting"][
                    "receipt_sha256"
                ],
                "runner_attestations": {
                    TREATMENT_ARM: treatment["runner_attestation"],
                    CONTROL_ARM: control["runner_attestation"],
                },
                "verifier_attestations": {
                    TREATMENT_ARM: treatment["verifier_attestation"],
                    CONTROL_ARM: control["verifier_attestation"],
                },
            }
        )
    if len(seen_tasks) < MIN_PAIR_COUNT:
        _fail("action_calibration_global_task_coverage_invalid")
    resource_caps = _action_resource_caps(metadata["execution_config"].get("action_resource_caps"))
    cost_budget = resource_caps["estimated_flops"]
    cells = _evidence_cells(
        observations,
        action_resource_caps=resource_caps,
    )
    body = {
        "schema": ACTION_CALIBRATION_CANDIDATE_SCHEMA,
        "preverified": True,
        "claim_scope": metadata["claim_scope"],
        "calibration_bucket": metadata["calibration_bucket"],
        "frontier_claim_eligible": False,
        "campaign_name": plan.campaign_name,
        "plan_sha256": plan.plan_sha256,
        "campaign_plan": plan.to_dict(),
        "policy_sha256": policy.policy_sha256,
        "campaign_manifest": manifest,
        "campaign_manifest_sha256": manifest["manifest_sha256"],
        "campaign_journal": journal_transcript,
        "journal_head_sha256": manifest["journal_head_sha256"],
        "journal_event_count": manifest["journal_event_count"],
        "task_manifest_sha256": metadata["task_manifest"]["manifest_sha256"],
        "task_commitment_sha256": metadata["task_commitment"]["commitment_sha256"],
        "assignment_sha256": metadata["assignment_sha256"],
        "sampling_frame_sha256": metadata["sampling_frame"]["sampling_frame_sha256"],
        "model_identity_sha256": _sha256(metadata["model_identity"]),
        "execution_config_sha256": _sha256(metadata["execution_config"]),
        "action_count": EXPECTED_ACTION_COUNT,
        "pair_count": len(observations),
        "execution_count": len(normalized_records),
        "minimum_unique_tasks_per_action": MIN_UNIQUE_TASKS_PER_ACTION,
        "cost_budget_estimated_flops": cost_budget,
        "action_resource_caps": resource_caps,
        "global_bound_family_count": GLOBAL_BOUND_FAMILY_COUNT,
        "gain_bound_method": _GAIN_BOUND_METHOD,
        "cost_bound_method": _COST_BOUND_METHOD,
        "issuer_payload": issuer_payload,
        "issuer_attestation": dict(issuer_attestation),
        "contamination_payload": expected_contamination,
        "contamination_attestation": dict(contamination_attestation),
        "output_seal_payload": expected_output_seal,
        "output_seal_attestation": dict(output_seal_attestation),
        "observations": observations,
        "cells": cells,
        "limitations": [
            "checkpoint-bound",
            "state-bucket-bound",
            "task-distribution-bound",
            "not-a-frontier-intelligence-certificate",
            "not-a-consciousness-measurement",
        ],
    }
    return {**body, "candidate_sha256": _sha256(body)}


def _verify_action_calibration_candidate(
    candidate: Mapping[str, Any],
    *,
    policy: VerifiedCampaignTrustPolicy,
) -> dict[str, Any]:
    """Independently recheck the complete preverified campaign body."""

    required = {
        "schema",
        "preverified",
        "claim_scope",
        "calibration_bucket",
        "frontier_claim_eligible",
        "campaign_name",
        "plan_sha256",
        "campaign_plan",
        "policy_sha256",
        "campaign_manifest",
        "campaign_manifest_sha256",
        "campaign_journal",
        "journal_head_sha256",
        "journal_event_count",
        "task_manifest_sha256",
        "task_commitment_sha256",
        "assignment_sha256",
        "sampling_frame_sha256",
        "model_identity_sha256",
        "execution_config_sha256",
        "action_count",
        "pair_count",
        "execution_count",
        "minimum_unique_tasks_per_action",
        "cost_budget_estimated_flops",
        "action_resource_caps",
        "global_bound_family_count",
        "gain_bound_method",
        "cost_bound_method",
        "issuer_payload",
        "issuer_attestation",
        "contamination_payload",
        "contamination_attestation",
        "output_seal_payload",
        "output_seal_attestation",
        "observations",
        "cells",
        "limitations",
        "candidate_sha256",
    }
    if not isinstance(candidate, Mapping) or set(candidate) != required:
        _fail("action_calibration_candidate_fields_invalid")
    normalized = dict(candidate)
    body = {name: normalized[name] for name in required - {"candidate_sha256"}}
    try:
        plan = CampaignPlan.from_dict(cast(Mapping[str, Any], normalized["campaign_plan"]))
    except (TypeError, ValueError) as exc:
        raise ActionCalibrationError("action_calibration_candidate_plan_invalid") from exc
    metadata = _metadata(plan)
    task_manifest_metadata = metadata.get("task_manifest")
    task_commitment_metadata = metadata.get("task_commitment")
    execution_config_metadata = metadata.get("execution_config")
    campaign_trust_metadata = metadata.get("campaign_trust")
    sampling_frame_metadata = metadata.get("sampling_frame")
    if (
        not isinstance(task_manifest_metadata, Mapping)
        or not isinstance(task_commitment_metadata, Mapping)
        or not isinstance(execution_config_metadata, Mapping)
        or not isinstance(campaign_trust_metadata, Mapping)
        or not isinstance(sampling_frame_metadata, Mapping)
    ):
        _fail("action_calibration_candidate_plan_metadata_invalid")
    try:
        public_tasks_by_id = {
            task.task_id: task
            for task in (
                PublicTaskRecord.from_dict(cast(Mapping[str, Any], raw))
                for raw in task_manifest_metadata["tasks"]
            )
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ActionCalibrationError("action_calibration_candidate_plan_metadata_invalid") from exc
    manifest = _validate_manifest(
        cast(Mapping[str, Any], normalized["campaign_manifest"]),
        plan=plan,
    )
    resource_caps = _action_resource_caps(normalized.get("action_resource_caps"))
    planned_resource_caps = _action_resource_caps(
        execution_config_metadata.get("action_resource_caps")
    )
    output_seal = normalized.get("output_seal_payload")
    output_seal_event_count = (
        output_seal.get("journal_event_count") if isinstance(output_seal, Mapping) else None
    )
    if (
        normalized.get("schema") != ACTION_CALIBRATION_CANDIDATE_SCHEMA
        or normalized.get("preverified") is not True
        or normalized.get("frontier_claim_eligible") is not False
        or not isinstance(normalized.get("calibration_bucket"), str)
        or not normalized["calibration_bucket"]
        or normalized.get("policy_sha256") != policy.policy_sha256
        or normalized.get("campaign_name") != policy.document.get("campaign_name")
        or normalized.get("campaign_name") != plan.campaign_name
        or normalized.get("plan_sha256") != plan.plan_sha256
        or normalized.get("campaign_manifest_sha256") != manifest["manifest_sha256"]
        or normalized.get("journal_head_sha256") != manifest["journal_head_sha256"]
        or normalized.get("journal_event_count") != manifest["journal_event_count"]
        or metadata.get("claim_eligible") is not True
        or metadata.get("claim_scope") != normalized.get("claim_scope")
        or metadata.get("calibration_bucket") != normalized.get("calibration_bucket")
        or metadata.get("action_count") != normalized.get("action_count")
        or metadata.get("pair_count") != normalized.get("pair_count")
        or metadata.get("execution_count") != normalized.get("execution_count")
        or metadata.get("minimum_unique_tasks_per_action")
        != normalized.get("minimum_unique_tasks_per_action")
        or metadata.get("assignment_sha256") != normalized.get("assignment_sha256")
        or sampling_frame_metadata.get("sampling_frame_sha256")
        != normalized.get("sampling_frame_sha256")
        or task_manifest_metadata.get("manifest_sha256") != normalized.get("task_manifest_sha256")
        or task_commitment_metadata.get("commitment_sha256")
        != normalized.get("task_commitment_sha256")
        or _sha256(metadata.get("model_identity")) != normalized.get("model_identity_sha256")
        or _sha256(execution_config_metadata) != normalized.get("execution_config_sha256")
        or campaign_trust_metadata.get("policy_sha256") != policy.policy_sha256
        or resource_caps != planned_resource_caps
        or not externally_custodied_roles(policy)
        or normalized.get("candidate_sha256") != _sha256(body)
        or normalized.get("action_count") != EXPECTED_ACTION_COUNT
        or type(normalized.get("pair_count")) is not int
        or normalized["pair_count"] < MIN_PAIR_COUNT
        or normalized.get("execution_count") != normalized["pair_count"] * len(CALIBRATION_ARMS)
        or not isinstance(output_seal, Mapping)
        or type(output_seal_event_count) is not int
        or normalized.get("journal_event_count")
        != output_seal_event_count + normalized["execution_count"] * 2
        or normalized.get("minimum_unique_tasks_per_action") != MIN_UNIQUE_TASKS_PER_ACTION
        or normalized.get("global_bound_family_count") != GLOBAL_BOUND_FAMILY_COUNT
        or type(normalized.get("cost_budget_estimated_flops")) is not int
        or normalized["cost_budget_estimated_flops"] <= 0
        or normalized.get("gain_bound_method") != _GAIN_BOUND_METHOD
        or normalized.get("cost_bound_method") != _COST_BOUND_METHOD
    ):
        _fail("action_calibration_candidate_invalid")
    _validate_plan_sampling_frame(plan)
    if resource_caps["estimated_flops"] != normalized["cost_budget_estimated_flops"]:
        _fail("action_calibration_cost_budget_invalid")
    for name in (
        "plan_sha256",
        "campaign_manifest_sha256",
        "journal_head_sha256",
        "task_manifest_sha256",
        "task_commitment_sha256",
        "assignment_sha256",
        "sampling_frame_sha256",
        "model_identity_sha256",
        "execution_config_sha256",
    ):
        if not _is_sha256(normalized.get(name)):
            _fail("action_calibration_certificate_commitment_invalid")
    issuer_payload = normalized.get("issuer_payload")
    contamination_payload = normalized.get("contamination_payload")
    if (
        not isinstance(issuer_payload, Mapping)
        or issuer_payload.get("plan_sha256") != normalized["plan_sha256"]
        or issuer_payload.get("task_manifest_sha256") != normalized["task_manifest_sha256"]
        or issuer_payload.get("task_commitment_sha256") != normalized["task_commitment_sha256"]
        or issuer_payload.get("assignment_sha256") != normalized["assignment_sha256"]
        or issuer_payload.get("pair_count") != normalized["pair_count"]
        or not isinstance(contamination_payload, Mapping)
        or contamination_payload.get("plan_sha256") != normalized["plan_sha256"]
        or contamination_payload.get("task_manifest_sha256") != normalized["task_manifest_sha256"]
        or contamination_payload.get("assignment_sha256") != normalized["assignment_sha256"]
        or contamination_payload.get("status") != "passed_zero_overlap"
        or contamination_payload.get("overlap_count") != 0
    ):
        _fail("action_calibration_certificate_global_binding_invalid")
    verify_role_attestation(
        policy,
        normalized["issuer_attestation"],
        role=TASK_ISSUER,
        expected_payload=issuer_payload,
    )
    verify_role_attestation(
        policy,
        normalized["contamination_attestation"],
        role=CONTAMINATION_AUDITOR,
        expected_payload=contamination_payload,
    )
    if (
        output_seal.get("schema") != ACTION_CALIBRATION_OUTPUT_SEAL_SCHEMA
        or output_seal.get("plan_sha256") != normalized["plan_sha256"]
        or output_seal.get("cell_count") != normalized["execution_count"]
        or output_seal.get("answer_material_revealed") is not False
        or not isinstance(output_seal.get("sealed_results"), list)
        or output_seal.get("sealed_results_sha256") != _sha256(output_seal["sealed_results"])
    ):
        _fail("action_calibration_certificate_output_seal_invalid")
    journal_transcript, journal_records, intervention_claims = _validate_journal_transcript(
        normalized.get("campaign_journal"),
        plan=plan,
        manifest=manifest,
        output_seal=output_seal,
    )
    if not _strict_equal(
        normalized.get("campaign_journal"),
        journal_transcript,
    ):
        _fail("action_calibration_journal_transcript_invalid")
    sealed_attestation = verify_role_attestation(
        policy,
        normalized["output_seal_attestation"],
        role=CAMPAIGN_RUNNER,
        expected_payload=output_seal,
    )
    output_sealed_at_unix = sealed_attestation["signed_at_unix"]

    observations = normalized.get("observations")
    if not isinstance(observations, list) or len(observations) != normalized["pair_count"]:
        _fail("action_calibration_certificate_observations_invalid")
    seen_pairs: set[str] = set()
    seen_tasks: set[str] = set()
    action_counts: dict[str, int] = defaultdict(int)
    sealed_result_map = {
        row.get("cell_id"): row.get("result_sha256")
        for row in output_seal["sealed_results"]
        if isinstance(row, Mapping)
    }
    if len(sealed_result_map) != normalized["execution_count"] or any(
        not isinstance(cell_id, str) or not _is_sha256(result_sha256)
        for cell_id, result_sha256 in sealed_result_map.items()
    ):
        _fail("action_calibration_certificate_output_seal_invalid")
    observed_sealed_cells: set[str] = set()
    for row in observations:
        if not isinstance(row, Mapping):
            _fail("action_calibration_certificate_observation_invalid")
        observation_fields = {
            "action",
            "pair_id",
            "task_id",
            "starting_state_sha256",
            "treatment_success",
            "control_success",
            "treatment_estimated_flops",
            "control_estimated_flops",
            "treatment_action_estimated_flops",
            "control_action_estimated_flops",
            "treatment_action_resources",
            "control_action_resources",
            "accounting_certificate",
            "treatment_result_sha256",
            "control_result_sha256",
            "treatment_score_receipt_sha256",
            "control_score_receipt_sha256",
            "answer_commitment_sha256",
            "treatment_consumed_information_sha256",
            "control_consumed_information_sha256",
            "runner_attestations",
            "verifier_attestations",
        }
        action = _operation(row.get("action"))
        pair_id = row.get("pair_id")
        task_id = row.get("task_id")
        if (
            set(row) != observation_fields
            or not isinstance(pair_id, str)
            or not pair_id
            or pair_id in seen_pairs
            or not isinstance(task_id, str)
            or not task_id
            or task_id in seen_tasks
            or type(row.get("treatment_success")) is not bool
            or type(row.get("control_success")) is not bool
            or type(row.get("treatment_estimated_flops")) is not int
            or row["treatment_estimated_flops"] <= 0
            or type(row.get("control_estimated_flops")) is not int
            or row["control_estimated_flops"] <= 0
            or type(row.get("treatment_action_estimated_flops")) is not int
            or not 0
            <= row["treatment_action_estimated_flops"]
            <= normalized["cost_budget_estimated_flops"]
            or row.get("control_action_estimated_flops") != 0
            or not isinstance(row.get("treatment_action_resources"), Mapping)
            or not isinstance(row.get("control_action_resources"), Mapping)
            or row["treatment_action_resources"].get("estimated_flops")
            != row["treatment_action_estimated_flops"]
            or row["control_action_resources"].get("estimated_flops") != 0
            or any(row["control_action_resources"].values())
            or not _is_sha256(row.get("treatment_consumed_information_sha256"))
            or not _is_sha256(row.get("control_consumed_information_sha256"))
            or not isinstance(row.get("runner_attestations"), Mapping)
            or set(row["runner_attestations"]) != set(CALIBRATION_ARMS)
            or not isinstance(row.get("verifier_attestations"), Mapping)
            or set(row["verifier_attestations"]) != set(CALIBRATION_ARMS)
        ):
            _fail("action_calibration_certificate_observation_invalid")
        _normalized_action_cost(
            cast(Mapping[str, Any], row["treatment_action_resources"]),
            caps=resource_caps,
        )
        _normalized_action_cost(
            cast(Mapping[str, Any], row["control_action_resources"]),
            caps=resource_caps,
        )
        seen_pairs.add(pair_id)
        seen_tasks.add(task_id)
        action_counts[action.value] += 1
        try:
            accounting = validate_comparison_accounting_certificate(
                row.get("accounting_certificate")
            )
        except (TypeError, ValueError) as exc:
            raise ActionCalibrationError(
                "action_calibration_certificate_accounting_invalid"
            ) from exc
        if accounting["admitted"] is not True or accounting["information_matched"] is not True:
            _fail("action_calibration_certificate_accounting_invalid")
        arm_journal_rows: dict[str, dict[str, Any]] = {}
        for arm in CALIBRATION_ARMS:
            runner_payload = {
                "schema": ACTION_CALIBRATION_RUNNER_PAYLOAD_SCHEMA,
                "plan_sha256": normalized["plan_sha256"],
                "cell_id": None,
                "definition_sha256": None,
                "result_core_sha256": None,
                "attempt_id": None,
            }
            runner_attestation = row["runner_attestations"][arm]
            signed_runner = runner_attestation.get("signed_payload", {}).get("payload")
            if not isinstance(signed_runner, Mapping):
                _fail("action_calibration_certificate_runner_invalid")
            runner_payload = dict(signed_runner)
            if (
                runner_payload.get("schema") != ACTION_CALIBRATION_RUNNER_PAYLOAD_SCHEMA
                or runner_payload.get("plan_sha256") != normalized["plan_sha256"]
                or not isinstance(runner_payload.get("attempt_id"), str)
                or not runner_payload["attempt_id"]
            ):
                _fail("action_calibration_certificate_runner_invalid")
            verify_role_attestation(
                policy,
                runner_attestation,
                role=CAMPAIGN_RUNNER,
                expected_payload=runner_payload,
            )
            verifier_attestation = row["verifier_attestations"][arm]
            signed_verifier = verifier_attestation.get("signed_payload", {}).get("payload")
            if not isinstance(signed_verifier, Mapping):
                _fail("action_calibration_certificate_verifier_invalid")
            if (
                signed_verifier.get("schema") != ACTION_CALIBRATION_VERIFIER_PAYLOAD_SCHEMA
                or signed_verifier.get("plan_sha256") != normalized["plan_sha256"]
            ):
                _fail("action_calibration_certificate_verifier_invalid")
            expected_result = row[
                f"{'treatment' if arm == TREATMENT_ARM else 'control'}_result_sha256"
            ]
            expected_score = row[
                f"{'treatment' if arm == TREATMENT_ARM else 'control'}_score_receipt_sha256"
            ]
            if (
                signed_verifier.get("result_sha256") != expected_result
                or signed_verifier.get("score_receipt_sha256") != expected_score
                or signed_verifier.get("answer_commitment_sha256")
                != row["answer_commitment_sha256"]
            ):
                _fail("action_calibration_certificate_verifier_binding_invalid")
            cell_id = signed_runner.get("cell_id")
            if (
                not isinstance(cell_id, str)
                or cell_id in observed_sealed_cells
                or sealed_result_map.get(cell_id) != expected_result
                or signed_verifier.get("cell_id") != cell_id
                or cell_id not in journal_records
                or _sha256(journal_records[cell_id]["result"]) != expected_result
                or not _strict_equal(
                    journal_records[cell_id]["result"].get("runner_attestation"),
                    runner_attestation,
                )
                or not _strict_equal(
                    journal_records[cell_id]["verification"].get("verifier_attestation"),
                    verifier_attestation,
                )
            ):
                _fail("action_calibration_certificate_output_seal_binding_invalid")
            observed_sealed_cells.add(cell_id)
            verify_role_attestation(
                policy,
                verifier_attestation,
                role=EVIDENCE_VERIFIER,
                expected_payload=signed_verifier,
                not_before_unix=output_sealed_at_unix,
            )
            journal_record = journal_records[cell_id]
            definition = journal_record["definition"]
            result = journal_record["result"]
            verification = journal_record["verification"]
            intervention_evidence = _validate_action_intervention_evidence(
                result.get("action_intervention_evidence"),
                plan=plan,
                policy=policy,
                cell_id=cell_id,
                attempt_id=journal_record["attempt_id"],
                action_intervention_claim=intervention_claims[cell_id],
                campaign_journal=journal_transcript,
                issuer_attestation=cast(
                    Mapping[str, Any],
                    normalized["issuer_attestation"],
                ),
            )
            worker_receipt = intervention_evidence["worker_receipt"]
            public_task = public_tasks_by_id.get(cast(str, definition.get("task_id")))
            if public_task is None:
                _fail("action_calibration_state_capture_invalid")
            starting_state = _validate_starting_state_receipt(
                definition.get("starting_state"),
                campaign_name=plan.campaign_name,
                action=action,
                task=public_task,
                model_identity=cast(Mapping[str, Any], metadata["model_identity"]),
                execution_config=cast(Mapping[str, Any], execution_config_metadata),
                calibration_bucket=cast(str, normalized["calibration_bucket"]),
                campaign_design_sha256=cast(
                    str,
                    metadata["campaign_design_sha256"],
                ),
                policy=policy,
            )
            capture_signed_at = starting_state["capture_attestation"]["signed_payload"][
                "signed_at_unix"
            ]
            runner_signed_at = runner_attestation["signed_payload"]["signed_at_unix"]
            if (
                definition.get("arm") != arm
                or definition.get("action") != action.value
                or definition.get("pair_id") != pair_id
                or definition.get("task_id") != task_id
                or not isinstance(result, Mapping)
                or result.get("action_execution")
                != {
                    "selection_mode": worker_receipt["selection_mode"],
                    "selected_action": worker_receipt["selected_action"],
                    "campaign_authority_sha256": plan.plan_sha256,
                }
                or result.get("action_trace")
                != {
                    "schema": "aura.rlc.action_calibration.action_trace.v1",
                    "action": definition["action"],
                    "intervention_ordinal": 0,
                    "selected_action_occurrences": worker_receipt["selected_action_occurrences"],
                    "action_excluded_at_intervention": arm == CONTROL_ARM,
                    "pre_state_sha256": worker_receipt["pre_state_sha256"],
                    "post_state_sha256": worker_receipt["post_state_sha256"],
                }
                or not isinstance(verification, Mapping)
                or type(verification.get("correct")) is not bool
                or not isinstance(verification.get("score_receipt"), Mapping)
                or verification.get("answer_commitment_sha256")
                != row.get("answer_commitment_sha256")
                or type(capture_signed_at) is not int
                or type(runner_signed_at) is not int
                or capture_signed_at > runner_signed_at
            ):
                _fail("action_calibration_certificate_observation_binding_invalid")
            try:
                resource = validate_resource_receipt(result.get("resource_accounting"))
                action_resource = validate_resource_receipt(
                    result.get("action_resource_accounting")
                )
                information = validate_information_receipt(
                    result.get("available_information_accounting")
                )
                consumed_information = validate_information_receipt(
                    result.get("consumed_information_accounting")
                )
            except (TypeError, ValueError) as exc:
                raise ActionCalibrationError(
                    "action_calibration_certificate_observation_binding_invalid"
                ) from exc
            action_resources = {
                "estimated_flops": action_resource["estimated_flops"],
                **{counter: action_resource["totals"][counter] for counter in RESOURCE_COUNTERS},
            }
            arm_journal_rows[arm] = {
                "correct": verification["correct"],
                "estimated_flops": resource["estimated_flops"],
                "action_estimated_flops": action_resource["estimated_flops"],
                "action_resources": action_resources,
                "resource_accounting": resource,
                "information_accounting": information,
                "consumed_information_accounting": consumed_information,
                "result_sha256": _sha256(dict(result)),
                "score_receipt_sha256": _sha256(dict(verification["score_receipt"])),
                "runner_attestation": dict(cast(Mapping[str, Any], runner_attestation)),
                "verifier_attestation": dict(cast(Mapping[str, Any], verifier_attestation)),
                "starting_state_sha256": definition.get("starting_state_sha256"),
            }
        treatment = arm_journal_rows[TREATMENT_ARM]
        control = arm_journal_rows[CONTROL_ARM]
        reconstructed_accounting = certify_comparison_accounting(
            treatment_resource=treatment["resource_accounting"],
            control_resource=control["resource_accounting"],
            treatment_information=treatment["information_accounting"],
            control_information=control["information_accounting"],
            tolerance_numerator=1,
            tolerance_denominator=1,
            require_compute_parity=False,
        )
        expected_observation = {
            "action": action.value,
            "pair_id": pair_id,
            "task_id": task_id,
            "starting_state_sha256": treatment["starting_state_sha256"],
            "treatment_success": treatment["correct"],
            "control_success": control["correct"],
            "treatment_estimated_flops": treatment["estimated_flops"],
            "control_estimated_flops": control["estimated_flops"],
            "treatment_action_estimated_flops": treatment["action_estimated_flops"],
            "control_action_estimated_flops": control["action_estimated_flops"],
            "treatment_action_resources": treatment["action_resources"],
            "control_action_resources": control["action_resources"],
            "accounting_certificate": reconstructed_accounting,
            "treatment_result_sha256": treatment["result_sha256"],
            "control_result_sha256": control["result_sha256"],
            "treatment_score_receipt_sha256": treatment["score_receipt_sha256"],
            "control_score_receipt_sha256": control["score_receipt_sha256"],
            "answer_commitment_sha256": row["answer_commitment_sha256"],
            "treatment_consumed_information_sha256": treatment["consumed_information_accounting"][
                "receipt_sha256"
            ],
            "control_consumed_information_sha256": control["consumed_information_accounting"][
                "receipt_sha256"
            ],
            "runner_attestations": {
                TREATMENT_ARM: treatment["runner_attestation"],
                CONTROL_ARM: control["runner_attestation"],
            },
            "verifier_attestations": {
                TREATMENT_ARM: treatment["verifier_attestation"],
                CONTROL_ARM: control["verifier_attestation"],
            },
        }
        if (
            treatment["starting_state_sha256"] != control["starting_state_sha256"]
            or reconstructed_accounting["admitted"] is not True
            or not _strict_equal(row, expected_observation)
        ):
            _fail("action_calibration_certificate_observation_binding_invalid")
    if set(action_counts) != {action.value for action in OperationKind} or any(
        count < MIN_UNIQUE_TASKS_PER_ACTION for count in action_counts.values()
    ):
        _fail("action_calibration_certificate_action_coverage_invalid")
    if observed_sealed_cells != set(sealed_result_map):
        _fail("action_calibration_certificate_output_seal_coverage_invalid")
    expected_cells = _evidence_cells(
        observations,
        action_resource_caps=resource_caps,
    )
    if not _strict_equal(normalized.get("cells"), expected_cells):
        _fail("action_calibration_certificate_statistics_invalid")
    return normalized


def action_calibration_final_verifier_payload(
    candidate: Mapping[str, Any],
    *,
    policy: VerifiedCampaignTrustPolicy,
) -> dict[str, Any]:
    """Return the exact completed grade material for detached final signing."""

    verified = _verify_action_calibration_candidate(candidate, policy=policy)
    return {
        "schema": ACTION_CALIBRATION_FINAL_VERIFIER_SCHEMA,
        "accepted": True,
        "candidate_sha256": verified["candidate_sha256"],
        "calibration_bucket": verified["calibration_bucket"],
        "plan_sha256": verified["plan_sha256"],
        "policy_sha256": verified["policy_sha256"],
        "campaign_manifest_sha256": verified["campaign_manifest_sha256"],
        "journal_head_sha256": verified["journal_head_sha256"],
        "journal_event_count": verified["journal_event_count"],
        "observations_sha256": _sha256(verified["observations"]),
        "cells_sha256": _sha256(verified["cells"]),
        "pair_count": verified["pair_count"],
        "execution_count": verified["execution_count"],
        "frontier_claim_eligible": False,
    }


def finalize_action_calibration_certificate(
    candidate: Mapping[str, Any],
    *,
    policy: VerifiedCampaignTrustPolicy,
    final_verifier_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the independent final verdict to an immutable candidate."""

    verified = _verify_action_calibration_candidate(candidate, policy=policy)
    payload = action_calibration_final_verifier_payload(
        verified,
        policy=policy,
    )
    latest_cell_verification = max(
        cast(int, attestation["signed_payload"]["signed_at_unix"])
        for observation in verified["observations"]
        for attestation in observation["verifier_attestations"].values()
    )
    verify_role_attestation(
        policy,
        final_verifier_attestation,
        role=EVIDENCE_VERIFIER,
        expected_payload=payload,
        not_before_unix=latest_cell_verification,
    )
    body = {
        "schema": ACTION_CALIBRATION_CERTIFICATE_SCHEMA,
        "accepted": True,
        "candidate": verified,
        "final_verifier_payload": payload,
        "final_verifier_attestation": dict(final_verifier_attestation),
    }
    return {**body, "certificate_sha256": _sha256(body)}


def verify_action_calibration_certificate(
    certificate: Mapping[str, Any],
    *,
    policy: VerifiedCampaignTrustPolicy,
) -> dict[str, Any]:
    """Independently verify the outer verdict and every nested campaign proof."""

    required = {
        "schema",
        "accepted",
        "candidate",
        "final_verifier_payload",
        "final_verifier_attestation",
        "certificate_sha256",
    }
    if not isinstance(certificate, Mapping) or set(certificate) != required:
        _fail("action_calibration_certificate_fields_invalid")
    normalized = dict(certificate)
    body = {name: normalized[name] for name in required if name != "certificate_sha256"}
    if (
        normalized.get("schema") != ACTION_CALIBRATION_CERTIFICATE_SCHEMA
        or normalized.get("accepted") is not True
        or normalized.get("certificate_sha256") != _sha256(body)
        or not isinstance(normalized.get("candidate"), Mapping)
    ):
        _fail("action_calibration_certificate_invalid")
    candidate = _verify_action_calibration_candidate(
        cast(Mapping[str, Any], normalized["candidate"]),
        policy=policy,
    )
    expected_payload = action_calibration_final_verifier_payload(
        candidate,
        policy=policy,
    )
    if not _strict_equal(normalized.get("final_verifier_payload"), expected_payload):
        _fail("action_calibration_final_verifier_payload_invalid")
    latest_cell_verification = max(
        cast(int, attestation["signed_payload"]["signed_at_unix"])
        for observation in candidate["observations"]
        for attestation in observation["verifier_attestations"].values()
    )
    verify_role_attestation(
        policy,
        normalized["final_verifier_attestation"],
        role=EVIDENCE_VERIFIER,
        expected_payload=expected_payload,
        not_before_unix=latest_cell_verification,
    )
    return normalized


def certified_evidence_snapshot(
    certificate: Mapping[str, Any],
    *,
    policy: VerifiedCampaignTrustPolicy,
    bucket: str,
) -> dict[str, Any]:
    """Project a verified campaign certificate into worker-safe evidence."""

    verified = verify_action_calibration_certificate(
        certificate,
        policy=policy,
    )
    if (
        not isinstance(bucket, str)
        or not bucket
        or bucket != bucket.strip()
        or len(bucket) > 160
        or bucket != verified["candidate"]["calibration_bucket"]
    ):
        _fail("action_calibration_bucket_invalid")
    cells = {
        action: {
            **dict(cell),
            "calibration_candidate_sha256": verified["candidate"]["candidate_sha256"],
            "policy_sha256": verified["candidate"]["policy_sha256"],
        }
        for action, cell in verified["candidate"]["cells"].items()
    }
    admission = {
        "schema": ACTION_CALIBRATION_WORKER_ADMISSION_SCHEMA,
        "campaign_name": verified["candidate"]["campaign_name"],
        "policy_validated_at_unix": verified["final_verifier_attestation"]["signed_payload"][
            "signed_at_unix"
        ],
        "policy_document": dict(policy.document),
        "final_verifier_payload": dict(verified["final_verifier_payload"]),
        "final_verifier_attestation": dict(verified["final_verifier_attestation"]),
    }
    body = {
        "schema": ACTION_CALIBRATION_EVIDENCE_SCHEMA,
        "bucket": bucket,
        "candidate_sha256": verified["candidate"]["candidate_sha256"],
        "policy_sha256": verified["candidate"]["policy_sha256"],
        "admission": admission,
        "cells": cells,
    }
    return {**body, "snapshot_sha256": _sha256(body)}


__all__ = [
    "ACTION_CALIBRATION_AUDIT_PAYLOAD_SCHEMA",
    "ACTION_CALIBRATION_CERTIFICATE_SCHEMA",
    "ACTION_CALIBRATION_CANDIDATE_SCHEMA",
    "ACTION_CALIBRATION_DESIGN_SCHEMA",
    "ACTION_CALIBRATION_EVIDENCE_SCHEMA",
    "ACTION_CALIBRATION_ISSUER_PAYLOAD_SCHEMA",
    "ACTION_CALIBRATION_FINAL_VERIFIER_SCHEMA",
    "ACTION_CALIBRATION_OUTPUT_SEAL_SCHEMA",
    "ACTION_CALIBRATION_PROTOCOL_SCHEMA",
    "ACTION_CALIBRATION_RESULT_SCHEMA",
    "ACTION_CALIBRATION_RUNNER_PAYLOAD_SCHEMA",
    "ACTION_CALIBRATION_SAMPLING_FRAME_SCHEMA",
    "ACTION_CALIBRATION_STATE_CAPTURE_SCHEMA",
    "ACTION_CALIBRATION_VERIFICATION_SCHEMA",
    "ACTION_CALIBRATION_VERIFIER_PAYLOAD_SCHEMA",
    "ACTION_CALIBRATION_WORKER_ADMISSION_SCHEMA",
    "ACTION_RESOURCE_DIMENSIONS",
    "CALIBRATION_ARMS",
    "CONTROL_ARM",
    "EXPECTED_ACTION_COUNT",
    "GLOBAL_BOUND_FAMILY_COUNT",
    "MIN_EXECUTION_COUNT",
    "MIN_CERTIFIED_TASKS_PER_ACTION",
    "MIN_PAIR_COUNT",
    "MIN_UNIQUE_TASKS_PER_ACTION",
    "TREATMENT_ARM",
    "ActionCalibrationError",
    "action_calibration_contamination_payload",
    "action_calibration_final_verifier_payload",
    "action_calibration_issuer_payload",
    "action_calibration_runner_payload",
    "action_calibration_starting_state_payload",
    "action_calibration_output_seal_payload",
    "action_calibration_verifier_payload",
    "build_action_calibration_plan",
    "build_action_calibration_design",
    "build_action_calibration_candidate",
    "certified_evidence_snapshot",
    "finalize_action_calibration_certificate",
    "verify_action_calibration_certificate",
]
