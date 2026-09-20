#!/usr/bin/env python3
"""Freeze a source-bound resident recurrent-SFT bootstrap campaign.

Preparation is intentionally separate from execution. It writes immutable
training-only datasets, model/runtime/source commitments, an ordered invocation
plan, and the controller configuration. It never starts MLX or training.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Final, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
    model_behavior_bundle_identity,
)
from core.learning.recurrence_curriculum import (  # noqa: E402
    RECURRENCE_TRAINING_FAMILIES,
    RecurrenceTrainingTask,
    disjoint_task_split,
)
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    ExactAdjointTrajectoryConfig,
)
from core.learning.recurrence_native_objective_v5 import (  # noqa: E402
    GeneratedRollinSelectionConfig,
)
from core.learning.recurrence_native_objective_v6 import (  # noqa: E402
    BranchSpecializationConfig,
)
from core.learning.resident_recurrent_sft_bootstrap_authority import (  # noqa: E402
    OBJECTIVE_NAME_V5,
    REQUIRED_SOURCE_ROLES,
    TRAINER_CONFIG_SCHEMA_V6,
    ResidentSFTBootstrapConfig,
    build_authority,
    build_dataset_commitment,
    canonical_dataset_payloads,
    sha256_bytes,
    sha256_json,
    validate_authority,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from core.runtime.secure_path_custody import (  # noqa: E402
    DirectoryCustody,
    path_custody_threat_model,
)
from tools.resident_recurrent_sft_bootstrap_identity import (  # noqa: E402
    absent_personality_identity,
    load_resident_bootstrap_tokenizer,
    resident_bootstrap_runtime_identity,
    resident_bootstrap_tokenizer_identity,
)

PREPARATION_SCHEMA: Final = "aura.resident_recurrent_sft_bootstrap_preparation.v1"
CONTROLLER_CONFIG_SCHEMA: Final = "aura.resident_recurrent_sft_controller_config.v1"
TRUST_POLICY_SCHEMA: Final = "aura.resident_recurrent_sft_bootstrap_trust.v1"
PROFILES: Final = frozenset({"canary", "full"})
DEFAULT_MODEL: Final = "training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118"
DEFAULT_SPEC: Final = "config/latent_cortex/resident_32b_recurrent_grpo_execution_spec.json"
MAX_INPUT_BYTES: Final = 512 * 1024 * 1024
PREPARATION_INTENT_SCHEMA: Final = "aura.resident_recurrent_sft_preparation_intent.v1"
_CAMPAIGN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_CAMPAIGN_PREFIX: Final = "resident-32b-recurrent-sft-bootstrap-cp"

SOURCE_PATHS: Final[dict[str, str]] = {
    "authority": "core/learning/resident_recurrent_sft_bootstrap_authority.py",
    "state": "core/learning/resident_recurrent_sft_bootstrap_state.py",
    "bootstrap_execution": "core/learning/resident_recurrent_sft_bootstrap_execution.py",
    "trainer": "tools/train_resident_recurrent_sft_bootstrap.py",
    "preparer": "tools/prepare_resident_recurrent_sft_bootstrap_campaign.py",
    "controller": "tools/run_resident_recurrent_sft_bootstrap_campaign.py",
    "objective": "core/learning/recurrence_native_objective_v2.py",
    "objective_policy": "core/learning/recurrence_native_objective_v5.py",
    "specialization_objective": "core/learning/recurrence_native_objective_v6.py",
    "recurrent_sft_execution": "core/learning/recurrent_sft_execution.py",
    "execution_spec": "core/brain/llm/latent_cortex/execution_spec.py",
    "recurrence_adapter": "core/learning/recurrent_grpo.py",
    "scoped_recurrence_adapter": "core/brain/llm/latent_cortex/recurrence_adapter.py",
    "depth_conditioning": "core/learning/depth_conditioned_lora.py",
    "role_conditioned_adapter": "core/learning/role_conditioned_lora.py",
    "loop_core": "core/brain/llm/latent_cortex/recurrence.py",
    "adapter_identity": ("core/brain/llm/latent_cortex/recurrence_adapter_identity_v2.py"),
    "adapter_package_identity": (
        "core/brain/llm/latent_cortex/resident_recurrent_sft_adapter_identity.py"
    ),
    "adapter_materializer": "tools/materialize_resident_recurrent_sft_adapter.py",
    "paired_campaign_loader": "tools/run_latent_cortex_paired_campaign.py",
    "bootstrap_identity": "tools/resident_recurrent_sft_bootstrap_identity.py",
    "curriculum": "core/learning/recurrence_curriculum.py",
    "tokenizer_validator": "tools/validate_structured_sft_tokenization.py",
    "campaign_journal": "core/brain/llm/latent_cortex/campaign_journal.py",
    "campaign_trust": "core/brain/llm/latent_cortex/campaign_trust.py",
    "campaign_launch_bundle": ("core/brain/llm/latent_cortex/campaign_launch_bundle.py"),
    "detached_campaign_evidence": ("core/brain/llm/latent_cortex/detached_campaign_evidence.py"),
    "detached_runner": "tools/run_detached_step.py",
    "atomic_writer": "core/runtime/atomic_writer.py",
    "secure_path_custody": "core/runtime/secure_path_custody.py",
    "file_read_gateway": "core/runtime/file_read_gateway.py",
    "model_lane_control": "core/runtime/model_lane_control.py",
    "mlx_memory_guard": "core/runtime/mlx_memory_guard.py",
}


class ResidentSFTCampaignPreparationError(RuntimeError):
    """The campaign could not be frozen without weakening its bindings."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentSFTCampaignPreparationError(code)


def _canonical(value: Any) -> bytes:
    payload: bytes = canonical_json_bytes(value)
    return payload


def _validate_campaign_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or _CAMPAIGN_ID.fullmatch(value) is None
        or not value.startswith(_CAMPAIGN_PREFIX)
    ):
        _fail("resident_sft_campaign_identity_invalid")
    return value


def _repo_path(value: str, *, role: str, directory: bool | None = None) -> Path:
    if not value or "\\" in value or "\x00" in value:
        _fail(f"resident_sft_prepare_{role}_path_invalid")
    pure = PurePosixPath(value)
    if pure.is_absolute() or str(pure) != value or ".." in pure.parts:
        _fail(f"resident_sft_prepare_{role}_path_invalid")
    lexical = REPO_ROOT / pure
    current = REPO_ROOT
    for component in pure.parts:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            _fail(f"resident_sft_prepare_{role}_symlink_forbidden")
    resolved = lexical.resolve(strict=directory is not None)
    try:
        resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ResidentSFTCampaignPreparationError(
            f"resident_sft_prepare_{role}_outside_repo"
        ) from exc
    if directory is not None and resolved.is_dir() is not directory:
        _fail(f"resident_sft_prepare_{role}_type_invalid")
    return resolved


def _relative(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(REPO_ROOT.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise ResidentSFTCampaignPreparationError(
            "resident_sft_prepare_output_outside_repo"
        ) from exc


def _binding(
    path: Path,
    *,
    custody: DirectoryCustody | None = None,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail("resident_sft_prepare_binding_file_invalid")
    if custody is None:
        payload = read_stable_bytes(path, max_bytes=MAX_INPUT_BYTES)
    else:
        try:
            relative = path.relative_to(custody.path).as_posix()
        except ValueError as exc:
            raise ResidentSFTCampaignPreparationError(
                "resident_sft_prepare_custody_path_invalid"
            ) from exc
        payload = custody.read_bytes(relative, max_bytes=MAX_INPUT_BYTES)
    return {
        "path": _relative(path),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _write_once(
    path: Path,
    payload: bytes,
    *,
    custody: DirectoryCustody | None = None,
) -> None:
    if custody is not None:
        try:
            relative = path.relative_to(custody.path).as_posix()
        except ValueError as exc:
            raise ResidentSFTCampaignPreparationError(
                "resident_sft_prepare_custody_path_invalid"
            ) from exc
        if custody.write_bytes_once(relative, payload, mode=0o600):
            return
        observed = custody.read_bytes(relative, max_bytes=max(len(payload), 1))
        if observed != payload:
            _fail("resident_sft_prepare_existing_artifact_drift")
        return
    ensure_private_directory(path.parent)
    if atomic_write_bytes_if_absent(path, payload, mode=0o600):
        return
    try:
        observed = read_stable_bytes(path, max_bytes=max(len(payload), 1))
    except OSError as exc:
        raise ResidentSFTCampaignPreparationError(
            "resident_sft_prepare_existing_artifact_unreadable"
        ) from exc
    if observed != payload:
        _fail("resident_sft_prepare_existing_artifact_drift")


def _preparation_intent(
    *,
    root: Path,
    profile: str,
    campaign_id: str,
    model: str,
    execution_spec: str,
    artifact_root: str,
    seed: int,
    committed_at: datetime,
    custody: DirectoryCustody,
    max_minutes: float | None = None,
) -> dict[str, Any]:
    body = {
        "schema": PREPARATION_INTENT_SCHEMA,
        "profile": profile,
        "campaign_id": campaign_id,
        "model": model,
        "execution_spec": execution_spec,
        "artifact_root": artifact_root,
        "seed": seed,
        "committed_at": committed_at.astimezone(UTC).isoformat(),
    }
    if max_minutes is not None:
        body["max_minutes"] = float(max_minutes)
    intent = {**body, "intent_sha256": sha256_json(body)}
    _write_once(root / "preparation-intent.json", _canonical(intent), custody=custody)
    return intent


def _existing_intent_committed_at(
    *,
    profile: str,
    campaign_id: str,
    model: str,
    execution_spec: str,
    artifact_root: str,
    seed: int,
    max_minutes: float | None = None,
) -> datetime | None:
    root = _repo_path(artifact_root, role="artifact_root", directory=None)
    path = root / "preparation-intent.json"
    if not path.is_file():
        return None
    with DirectoryCustody.acquire(root, private=True) as custody:
        payload = custody.read_bytes("preparation-intent.json", max_bytes=64 * 1024)
    try:
        intent = json.loads(payload)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ResidentSFTCampaignPreparationError("resident_sft_prepare_intent_invalid") from exc
    body = dict(intent) if isinstance(intent, dict) else {}
    claimed = body.pop("intent_sha256", None)
    expected = {
        "schema": PREPARATION_INTENT_SCHEMA,
        "profile": profile,
        "campaign_id": campaign_id,
        "model": model,
        "execution_spec": execution_spec,
        "artifact_root": artifact_root,
        "seed": seed,
    }
    if max_minutes is not None:
        expected["max_minutes"] = float(max_minutes)
    if (
        not isinstance(intent, dict)
        or _canonical(intent) != payload
        or claimed != sha256_json(body)
        or any(body.get(key) != value for key, value in expected.items())
        or not isinstance(body.get("committed_at"), str)
    ):
        _fail("resident_sft_prepare_intent_invalid")
    try:
        committed_at = datetime.fromisoformat(body["committed_at"])
    except ValueError as exc:
        raise ResidentSFTCampaignPreparationError("resident_sft_prepare_intent_invalid") from exc
    if committed_at.tzinfo is None:
        _fail("resident_sft_prepare_intent_invalid")
    return committed_at


def _git_source_state() -> dict[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        if result.returncode != 0:
            _fail("resident_sft_prepare_git_state_unavailable")
        return result.stdout.strip()

    if run("diff", "--name-only", "HEAD", "--"):
        _fail("resident_sft_prepare_tracked_source_dirty")
    head = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    upstream = run("rev-parse", "origin/main")
    if head != upstream:
        _fail("resident_sft_prepare_main_not_published")
    return {"branch": branch, "commit": head, "origin_main": upstream}


def _profile_config(
    profile: str,
    *,
    seed: int,
    max_minutes: float | None = None,
) -> tuple[ResidentSFTBootstrapConfig, int, int, tuple[int, ...]]:
    if max_minutes is not None and (
        isinstance(max_minutes, bool)
        or not isinstance(max_minutes, (int, float))
        or not math.isfinite(float(max_minutes))
    ):
        _fail("resident_sft_prepare_max_minutes_invalid")
    if profile == "canary":
        if max_minutes is not None and float(max_minutes) != 120.0:
            _fail("resident_sft_prepare_canary_budget_override_forbidden")
        return (
            ResidentSFTBootstrapConfig(
                seed=seed,
                schema=TRAINER_CONFIG_SCHEMA_V6,
                objective=OBJECTIVE_NAME_V5,
                generated_rollin=GeneratedRollinSelectionConfig(),
                branch_specialization=BranchSpecializationConfig(
                    weight=8.0,
                    target_separation=0.30,
                ),
                trajectory_objective=ExactAdjointTrajectoryConfig(
                    probe_steps=(1, 2),
                    improvement_weight=1.0,
                    improvement_margin=0.05,
                ),
                structural_warmup_steps=4,
                structural_warmup_learning_rate=1e-4,
                role_conditioned_branches=2,
                lora_initialization_seed=(seed ^ 0x51F7A11) & 0xFFFFFFFF,
                max_steps=5,
                max_invocation_steps=1,
                max_minutes=120.0,
                learning_rate=5e-6,
                weight_decay=0.01,
                lora_rank=8,
                lora_scale=20.0,
                lora_targets=("q_proj", "v_proj", "o_proj"),
                lora_layers=8,
                coda_lora_targets=("o_proj", "down_proj"),
                coda_lora_layers=4,
                evaluate_every=1,
                validation_examples=4,
                intermediate_validation_examples=1,
                max_seq_length=512,
                memory_fraction=0.42,
                branch_indices=(0, 1),
            ),
            1,
            1,
            (2,),
        )
    if profile == "full":
        effective_max_minutes = 1_440.0 if max_minutes is None else float(max_minutes)
        if not 1_440.0 <= effective_max_minutes <= 10_080.0:
            _fail("resident_sft_prepare_full_budget_invalid")
        return (
            ResidentSFTBootstrapConfig(
                seed=seed,
                schema=TRAINER_CONFIG_SCHEMA_V6,
                objective=OBJECTIVE_NAME_V5,
                generated_rollin=GeneratedRollinSelectionConfig(),
                branch_specialization=BranchSpecializationConfig(
                    weight=8.0,
                    target_separation=0.30,
                ),
                trajectory_objective=ExactAdjointTrajectoryConfig(
                    probe_steps=(1, 2, 4, 8),
                    improvement_weight=1.0,
                    improvement_margin=0.05,
                ),
                structural_warmup_steps=8,
                structural_warmup_learning_rate=1e-4,
                role_conditioned_branches=2,
                lora_initialization_seed=(seed ^ 0x51F7A11) & 0xFFFFFFFF,
                max_steps=104,
                max_invocation_steps=4,
                max_minutes=effective_max_minutes,
                learning_rate=5e-6,
                weight_decay=0.01,
                lora_rank=8,
                lora_scale=20.0,
                lora_targets=("q_proj", "v_proj", "o_proj"),
                lora_layers=8,
                coda_lora_targets=("o_proj", "down_proj"),
                coda_lora_layers=4,
                evaluate_every=8,
                validation_examples=24,
                intermediate_validation_examples=4,
                max_seq_length=512,
                memory_fraction=0.42,
                branch_indices=(0, 1),
            ),
            4,
            2,
            (2, 4, 8),
        )
    _fail("resident_sft_prepare_profile_invalid")


def _task_row(task: RecurrenceTrainingTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "family": task.family,
        "depth": task.depth,
        "prompt": task.prompt,
        "answer": task.answer,
    }


def _load_spec(path: Path) -> tuple[RLCExecutionSpec, bytes]:
    payload = read_stable_bytes(path, max_bytes=4 * 1024 * 1024)

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("resident_sft_prepare_execution_spec_duplicate_key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> Never:
        _fail("resident_sft_prepare_execution_spec_non_finite")

    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
        spec = RLCExecutionSpec.from_dict(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ResidentSFTCampaignPreparationError(
            "resident_sft_prepare_execution_spec_invalid"
        ) from exc
    if spec.validate():
        _fail("resident_sft_prepare_execution_spec_unsupported")
    return spec, payload


def _probe_training_entrypoint(runtime_identity: Mapping[str, Any]) -> None:
    interpreter = runtime_identity.get("interpreter")
    executable = interpreter.get("executable") if isinstance(interpreter, Mapping) else None
    if not isinstance(executable, str) or not executable:
        _fail("resident_sft_prepare_interpreter_identity_invalid")
    result = subprocess.run(
        [executable, "-c", "import tools.train_resident_recurrent_sft_bootstrap"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120.0,
        check=False,
    )
    if result.returncode != 0:
        _fail("resident_sft_prepare_trainer_import_preflight_failed")


def _source_bindings() -> dict[str, dict[str, Any]]:
    if set(SOURCE_PATHS) != REQUIRED_SOURCE_ROLES:
        _fail("resident_sft_prepare_source_roles_invalid")
    bindings: dict[str, dict[str, Any]] = {}
    for role, path in sorted(SOURCE_PATHS.items()):
        tracked = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "ls-files", "--error-unmatch", "--", path],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=30.0,
            check=False,
        )
        committed = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "show", f"HEAD:{path}"],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=30.0,
            check=False,
        )
        source_path = _repo_path(path, role=f"source_{role}", directory=False)
        payload = read_stable_bytes(source_path, max_bytes=MAX_INPUT_BYTES)
        if tracked.returncode != 0 or committed.returncode != 0 or committed.stdout != payload:
            _fail("resident_sft_prepare_source_not_exact_head")
        bindings[role] = _binding(source_path)
    return bindings


def _build_plan(
    *,
    campaign_id: str,
    profile: str,
    authority_sha256: str,
    config: ResidentSFTBootstrapConfig,
    source_commit: str,
) -> CampaignPlan:
    cells: list[dict[str, Any]] = []
    invocations = math.ceil(config.max_steps / config.max_invocation_steps)
    for ordinal in range(invocations):
        start = ordinal * config.max_invocation_steps
        end = min(config.max_steps, start + config.max_invocation_steps)
        cells.append(
            {
                "schema": "aura.resident_recurrent_sft_invocation_cell.v1",
                "invocation_ordinal": ordinal + 1,
                "expected_start_step": start,
                "required_end_step": end,
                "authority_sha256": authority_sha256,
            }
        )
    return CampaignPlan.build(
        campaign_id,
        cells,
        metadata={
            "schema": "aura.resident_recurrent_sft_campaign_protocol.v1",
            "profile": profile,
            "source_commit": source_commit,
            "strict_execution_order": True,
            "action_intervention_required": False,
            "claim_eligible": False,
        },
    )


def prepare_campaign(
    *,
    profile: str,
    campaign_id: str,
    model: str,
    execution_spec: str,
    artifact_root: str,
    seed: int,
    committed_at: datetime,
    max_minutes: float | None = None,
) -> dict[str, Any]:
    if profile not in PROFILES:
        _fail("resident_sft_prepare_profile_invalid")
    if committed_at.tzinfo is None:
        _fail("resident_sft_prepare_committed_at_naive")
    campaign_id = _validate_campaign_id(campaign_id)
    source_state = _git_source_state()
    model_path = _repo_path(model, role="model", directory=True)
    spec_path = _repo_path(execution_spec, role="execution_spec", directory=False)
    root = _repo_path(artifact_root, role="artifact_root", directory=None)
    if root == REPO_ROOT or not root.is_relative_to(REPO_ROOT):
        _fail("resident_sft_prepare_artifact_root_invalid")
    root_custody = DirectoryCustody.acquire(root, create=True, private=True)
    root = root_custody.path
    _preparation_intent(
        root=root,
        profile=profile,
        campaign_id=campaign_id,
        model=model,
        execution_spec=execution_spec,
        artifact_root=artifact_root,
        seed=seed,
        committed_at=committed_at,
        custody=root_custody,
        max_minutes=max_minutes,
    )

    config, train_per_cell, validation_per_cell, depths = _profile_config(
        profile,
        seed=seed,
        max_minutes=max_minutes,
    )
    train_tasks, validation_tasks = disjoint_task_split(
        families=RECURRENCE_TRAINING_FAMILIES,
        depths=depths,
        train_per_cell=train_per_cell,
        holdout_per_cell=validation_per_cell,
        seed=seed,
    )
    train_rows = [_task_row(task) for task in train_tasks]
    validation_rows = [_task_row(task) for task in validation_tasks]
    train_payload, validation_payload = canonical_dataset_payloads(
        train_rows,
        validation_rows,
    )
    dataset = build_dataset_commitment(train_rows, validation_rows)
    inputs = root_custody.ensure_directory("inputs")
    training_output = root_custody.ensure_directory("training")
    controller_root = root_custody.ensure_directory("controller")
    with DirectoryCustody.acquire(training_output, private=True) as training_custody:
        training_identity = training_custody.identity
    with DirectoryCustody.acquire(controller_root, private=True) as controller_custody:
        controller_identity = controller_custody.identity
    train_path = inputs / "train.json"
    validation_path = inputs / "validation.json"
    _write_once(train_path, train_payload, custody=root_custody)
    _write_once(validation_path, validation_payload, custody=root_custody)

    spec, spec_payload = _load_spec(spec_path)
    del spec_payload
    tokenizer = load_resident_bootstrap_tokenizer(model_path)
    tokenizer_identity = resident_bootstrap_tokenizer_identity(model_path, tokenizer)
    del tokenizer
    source_bindings = _source_bindings()
    runtime_identity = resident_bootstrap_runtime_identity()
    _probe_training_entrypoint(runtime_identity)
    model_identity = full_weight_checkpoint_identity(model_path)
    behavior_identity = model_behavior_bundle_identity(model_path)
    personality_identity = absent_personality_identity()

    trust_body = {
        "schema": TRUST_POLICY_SCHEMA,
        "campaign_id": campaign_id,
        "profile": profile,
        "source": source_state,
        "exclusive_resident_model_owner": True,
        "checkpoint_before_retry_required": True,
        "max_consecutive_no_progress_failures": 2,
        "independent_os_supervisor_required_for_full": True,
        "sleep_inhibitor_required_for_full": True,
        "heartbeat_monitor_supplemental_only": True,
        "trainer_import_preflight_required": True,
        "training_only": True,
        "promotion_allowed": False,
        "gain_claim_allowed": False,
    }
    trust_policy = {**trust_body, "policy_sha256": sha256_json(trust_body)}
    trust_path = inputs / "trust-policy.json"
    trust_payload = _canonical(trust_policy)
    _write_once(trust_path, trust_payload, custody=root_custody)

    authority = build_authority(
        campaign_id=campaign_id,
        campaign_scope=("canary_lifecycle" if profile == "canary" else "full_bootstrap"),
        committed_at=committed_at.astimezone(UTC).isoformat(),
        expires_at=(committed_at.astimezone(UTC) + timedelta(days=7)).isoformat(),
        model_path=_relative(model_path),
        model_identity=model_identity,
        behavior_identity=behavior_identity,
        personality_identity=personality_identity,
        tokenizer_identity=tokenizer_identity,
        execution_spec={
            **_binding(spec_path),
            "semantic_sha256": spec.sha256,
        },
        dataset=dataset,
        dataset_artifacts={
            "train": _binding(train_path, custody=root_custody),
            "validation": _binding(validation_path, custody=root_custody),
        },
        sources=source_bindings,
        runtime_identity=runtime_identity,
        trust_policy={
            **_binding(trust_path, custody=root_custody),
            "semantic_sha256": sha256_json(trust_policy),
        },
        artifact_root=_relative(training_output),
        artifact_root_identity=training_identity,
        config=config,
    )
    validate_authority(
        authority,
        expected_authority_sha256=authority["authority_sha256"],
        observed_model_identity=model_identity,
        observed_behavior_identity=behavior_identity,
        observed_personality_identity=personality_identity,
        observed_tokenizer_identity=tokenizer_identity,
        observed_execution_spec={
            **_binding(spec_path),
            "semantic_sha256": spec.sha256,
        },
        observed_sources=source_bindings,
        now=committed_at.astimezone(UTC),
    )
    authority_path = inputs / "authority.json"
    authority_payload = _canonical(authority)
    _write_once(authority_path, authority_payload, custody=root_custody)

    plan = _build_plan(
        campaign_id=campaign_id,
        profile=profile,
        authority_sha256=authority["authority_sha256"],
        config=config,
        source_commit=source_state["commit"],
    )
    plan_path = inputs / "campaign-plan.json"
    plan_payload = _canonical(plan.to_dict())
    _write_once(plan_path, plan_payload, custody=root_custody)

    controller_body = {
        "schema": CONTROLLER_CONFIG_SCHEMA,
        "campaign_id": campaign_id,
        "profile": profile,
        "source": source_state,
        "authority": {
            **_binding(authority_path, custody=root_custody),
            "semantic_sha256": authority["authority_sha256"],
        },
        "plan": {
            **_binding(plan_path, custody=root_custody),
            "semantic_sha256": plan.plan_sha256,
        },
        "paths": {
            "artifact_root": _relative(root),
            "training_output": _relative(root / "training"),
            "controller_root": _relative(controller_root),
            "journal": _relative(controller_root / "campaign.journal.jsonl"),
            "manifest": _relative(controller_root / "campaign-manifest.json"),
            "detached_attempts": _relative(controller_root / "detached-attempts"),
        },
        "path_custody": {
            "artifact_root": root_custody.identity,
            "training_output": training_identity,
            "controller_root": controller_identity,
        },
        "path_custody_threat_model": path_custody_threat_model(),
        "watchdog": {
            "schema": "aura.resident_recurrent_sft_controller_watchdog.v1",
            "max_attempts_per_cell": 6,
            "max_consecutive_no_progress_failures": 2,
            "poll_interval_s": 5.0,
            "heartbeat_stale_s": 90.0,
            "attempt_timeout_s": 14_400.0 if profile == "canary" else 43_200.0,
            "retry_backoff_s": 15.0,
            "resume_exact_checkpoint_only": True,
        },
        "launch": {
            "label": f"com.aura.resident-sft.{campaign_id}",
            "launchd_required": True,
            "caffeinate_required": True,
        },
        "claim_state": {
            "reasoning_gain_proven": False,
            "frontier_level_proven": False,
            "grpo_admission": False,
            "promotion_allowed": False,
        },
    }
    controller_config = {
        **controller_body,
        "config_sha256": sha256_json(controller_body),
    }
    controller_path = root / "controller-config.json"
    controller_payload = _canonical(controller_config)
    _write_once(controller_path, controller_payload, custody=root_custody)

    preparation_body = {
        "schema": PREPARATION_SCHEMA,
        "campaign_id": campaign_id,
        "profile": profile,
        "source": source_state,
        "authority_sha256": authority["authority_sha256"],
        "plan_sha256": plan.plan_sha256,
        "controller_config_sha256": controller_config["config_sha256"],
        "model_fingerprint": model_identity["fingerprint"],
        "dataset_sha256": dataset["dataset_sha256"],
        "train_count": dataset["train_count"],
        "validation_count": dataset["validation_count"],
        "max_steps": config.max_steps,
        "invocation_count": len(plan.cell_ids),
        "training_started": False,
        "claims_supported": [],
        "paths": {
            "controller_config": _relative(controller_path),
            "authority": _relative(authority_path),
            "plan": _relative(plan_path),
        },
    }
    preparation = {
        **preparation_body,
        "preparation_sha256": sha256_json(preparation_body),
    }
    _write_once(
        root / "preparation-receipt.json",
        _canonical(preparation),
        custody=root_custody,
    )
    root_custody.close()
    return preparation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--execution-spec", default=DEFAULT_SPEC)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--seed", type=int, default=2026080108)
    parser.add_argument("--committed-at")
    parser.add_argument("--max-minutes", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        recovered_committed_at = _existing_intent_committed_at(
            profile=args.profile,
            campaign_id=args.campaign_id,
            model=args.model,
            execution_spec=args.execution_spec,
            artifact_root=args.artifact_root,
            seed=args.seed,
            max_minutes=args.max_minutes,
        )
        requested_committed_at = (
            datetime.fromisoformat(args.committed_at) if args.committed_at else None
        )
        if (
            recovered_committed_at is not None
            and requested_committed_at is not None
            and requested_committed_at.astimezone(UTC) != recovered_committed_at.astimezone(UTC)
        ):
            _fail("resident_sft_prepare_intent_committed_at_conflict")
        committed_at = recovered_committed_at or requested_committed_at or datetime.now(UTC)
        receipt = prepare_campaign(
            profile=args.profile,
            campaign_id=args.campaign_id,
            model=args.model,
            execution_spec=args.execution_spec,
            artifact_root=args.artifact_root,
            seed=args.seed,
            committed_at=committed_at,
            max_minutes=args.max_minutes,
        )
    except Exception as exc:  # noqa: BLE001 - emitted as a structured error receipt
        print(
            json.dumps(
                {
                    "schema": "aura.resident_recurrent_sft_preparation_error.v1",
                    "error_type": type(exc).__name__,
                    "error": str(exc) or "no_message",
                    "claims_supported": [],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
