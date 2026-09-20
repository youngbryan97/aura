#!/usr/bin/env python3
"""Run a bounded on-policy recurrent-GRPO behavioral canary.

The generated-rollin canary proved that lower teacher-forced loss can coexist
with worse free generation. This discriminator instead samples complete answers
from the recurrent policy, grades those exact answers with deterministic task
programs, and differentiates their admitted token traces. A separate disjoint
probe still decides whether the resulting checkpoint improved. It never fuses,
promotes, or activates an adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.commitment_ratchet import RATCHET_SCHEMA  # noqa: E402
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
)
from core.learning.grpo import group_advantages, reward_from_verdict  # noqa: E402
from core.learning.recurrence_curriculum import task_battery  # noqa: E402
from core.learning.recurrence_native_objective_v5 import derive_rollin_seed  # noqa: E402
from core.learning.recurrent_behavioral_probe import (  # noqa: E402
    build_behavioral_probe_report,
    build_paired_full_engine_probe_reports,
    canonical_json_bytes,
    free_generation_sampling_config,
    tokenize_task,
)
from core.learning.recurrent_checkpoint_admission import (  # noqa: E402
    RecurrentCheckpointAdmissionError,
    build_checkpoint_behavioral_admission,
    build_full_engine_behavioral_admission,
    build_recurrence_task_manifest,
    validate_checkpoint_behavioral_admission,
    validate_full_engine_behavioral_admission,
)
from core.learning.recurrent_grpo import (  # noqa: E402
    RecurrentGRPOConfig,
    RecurrentSamplingAdmissionError,
    attach_recurrent_policy_adapters,
    build_recurrent_policy_optimizer,
    exact_adjoint_sampled_group_value_and_grad,
    recurrent_policy_sha256,
    sample_final_recurrent_transition_completion,
)
from core.learning.recurrent_process_curriculum import process_task_battery  # noqa: E402
from core.learning.recurrent_sft_execution import (  # noqa: E402
    adapter_tensor_dict,
    adapter_tensor_fingerprint,
)
from core.learning.verified_token_trace import (  # noqa: E402
    build_resident_tokenizer_trace_adapter,
    observable_completion_from_adapter,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

CANARY_SCHEMA: Final = "aura.recurrent_grpo_behavioral_canary.v5"
LEARNABILITY_SEARCH_POLICY: Final = (
    "bounded_deterministic_same_family_disjoint_until_nonzero_advantage_v1"
)
SOURCE_PATHS: Final = (
    "core/learning/grpo.py",
    "core/learning/recurrence_curriculum.py",
    "core/learning/recurrent_behavioral_probe.py",
    "core/learning/recurrent_checkpoint_admission.py",
    "core/learning/recurrent_grpo.py",
    "core/learning/recurrent_process_curriculum.py",
    "core/learning/recurrence_native_objective_v2.py",
    "core/learning/recurrence_native_objective_v5.py",
    "core/learning/recurrence_native_objective_v6.py",
    "core/learning/recurrent_sft_execution.py",
    "core/learning/verified_token_trace.py",
    "core/brain/llm/latent_cortex/execution_spec.py",
    "core/brain/llm/latent_cortex/engine.py",
    "core/brain/llm/latent_cortex/answer_replacement.py",
    "core/brain/llm/latent_cortex/atomic_decomposition.py",
    "core/brain/llm/latent_cortex/contract_repair.py",
    "core/brain/llm/latent_cortex/commitment_extraction.py",
    "core/brain/llm/latent_cortex/commitment_ratchet.py",
    "core/brain/llm/latent_cortex/commitment_telemetry.py",
    "core/brain/llm/latent_cortex/critic_identity.py",
    "core/brain/llm/latent_cortex/deterministic_verifier_router.py",
    "core/brain/llm/latent_cortex/diagnostic_action_selector.py",
    "core/brain/llm/latent_cortex/disagreement_graph.py",
    "core/brain/llm/latent_cortex/incumbent_artifact.py",
    "core/brain/llm/latent_cortex/local_repair.py",
    "core/brain/llm/latent_cortex/objective_program_verifier.py",
    "core/brain/llm/latent_cortex/recurrence_adapter.py",
    "core/brain/llm/latent_cortex/sequential_exclusion.py",
    "core/brain/llm/latent_cortex/task_verifiers.py",
    "core/brain/llm/latent_cortex/types.py",
    "core/learning/depth_conditioned_lora.py",
    "core/learning/role_conditioned_lora.py",
    "tools/run_recurrent_grpo_behavioral_canary.py",
)


class RecurrentCanarySamplingError(RuntimeError):
    """A bounded group could not produce enough proof-admissible samples."""

    def __init__(
        self,
        *,
        admitted: int,
        requested: int,
        rejected: list[dict[str, Any]],
    ) -> None:
        self.admitted = admitted
        self.requested = requested
        self.rejected = rejected
        self.diagnostics = [
            _sampling_rejection_diagnostics(receipt) for receipt in rejected
        ]
        reasons = sorted(
            {
                reason
                for diagnostic in self.diagnostics
                for reason in diagnostic["failed_gates"]
            }
        )
        super().__init__(
            "recurrent GRPO canary exhausted admissible samples: "
            f"admitted={admitted} requested={requested} "
            f"rejected={len(rejected)} failed_gates={','.join(reasons) or 'unknown'}"
        )


def _sampling_rejection_diagnostics(receipt: dict[str, Any]) -> dict[str, Any]:
    """Reduce a large sample receipt to the exact local admission predicates."""

    sampling = dict(receipt.get("sampling_config") or {})
    activation = dict(receipt.get("cached_recurrence_adapter") or {})
    episode = dict(receipt.get("episode_receipt") or {})
    honest_flags = list(episode.get("honest_flags") or [])
    checks = {
        "max_abs_logprob_drift": float(receipt.get("max_abs_logprob_drift", math.inf))
        <= float(sampling.get("max_abs_logprob_drift", -math.inf)),
        "mean_abs_logprob_drift": float(receipt.get("mean_abs_logprob_drift", math.inf))
        <= float(sampling.get("max_mean_abs_logprob_drift", -math.inf)),
        "clipped_token_fraction": float(receipt.get("clipped_token_fraction", math.inf))
        <= float(sampling.get("max_clipped_token_fraction", -math.inf)),
        "old_policy_approx_kl": float(receipt.get("old_policy_approx_kl", math.inf))
        <= float(sampling.get("max_old_policy_approx_kl", -math.inf)),
        "params_unchanged": receipt.get("cached_params_unchanged") is True,
        "recurrence_adapter_active": bool(
            activation.get("active") is True
            and int(activation.get("calls", 0) or 0) > 0
            and int(activation.get("adapted_positions", 0) or 0) > 0
        ),
        "nonparametric_memory_disabled": (
            receipt.get("cached_nonparametric_memory_status") == "disabled_by_policy"
        ),
        "no_fallback": not any(str(flag).startswith("fallback_") for flag in honest_flags),
        "behavior_admitted": receipt.get("behavior_admitted") is True,
    }
    return {
        "episode_id": receipt.get("episode_id"),
        "seed": receipt.get("seed"),
        "token_count": receipt.get("token_count"),
        "max_abs_logprob_drift": receipt.get("max_abs_logprob_drift"),
        "mean_abs_logprob_drift": receipt.get("mean_abs_logprob_drift"),
        "clipped_token_fraction": receipt.get("clipped_token_fraction"),
        "old_policy_approx_kl": receipt.get("old_policy_approx_kl"),
        "checks": checks,
        "failed_gates": [name for name, passed in checks.items() if not passed],
        "runtime_integrity": receipt.get("cached_runtime_integrity"),
        "recurrence_adapter": activation,
        "nonparametric_memory_status": receipt.get(
            "cached_nonparametric_memory_status"
        ),
        "honest_flags": honest_flags,
    }


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_state() -> tuple[str, dict[str, dict[str, Any]]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("canary requires a clean source worktree")
    head = _git("rev-parse", "HEAD")
    if head != _git("rev-parse", "origin/main"):
        raise RuntimeError("canary source commit is not published on origin/main")
    bindings: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_PATHS:
        payload = (REPO_ROOT / relative).read_bytes()
        committed = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "show", f"{head}:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if payload != committed:
            raise RuntimeError(f"canary source differs from commit: {relative}")
        bindings[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    return head, bindings


def _stable_seed(*parts: Any) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _projection_rollin_seed(
    *,
    campaign_seed: int,
    task_id: str,
    sample_ordinal: int,
    execution_spec_sha256: str,
) -> int:
    """Bind answer-projection roll-ins without extending the sealed phase enum."""

    return derive_rollin_seed(
        campaign_seed=campaign_seed,
        phase="train",
        example_id=f"joint-process-answer-projection:{task_id}",
        sample_ordinal=sample_ordinal,
        execution_spec_sha256=execution_spec_sha256,
    )


def _cyclic_task(tasks: list[Any], *, one_based_step: int) -> Any:
    if not tasks or type(one_based_step) is not int or one_based_step < 1:
        raise ValueError("training task cycle coordinates are invalid")
    return tasks[(one_based_step - 1) % len(tasks)]


def _learnability_task_candidates(
    base_task: Any,
    *,
    campaign_seed: int,
    one_based_step: int,
    max_attempts: int,
    excluded_tasks: list[Any],
) -> list[Any]:
    """Return a sealed task window without consulting model outcomes.

    The nominal curriculum task remains attempt one. Later attempts retain its
    family and depth but use deterministic campaign coordinates that are
    disjoint from every training, projection, and held-out task. The caller may
    inspect rewards only to decide when to stop inside this predeclared window;
    task generation itself never sees model output or held-out grades.
    """

    if type(campaign_seed) is not int or not 0 <= campaign_seed <= 2**63 - 1:
        raise ValueError("campaign_seed must be inside [0, 2^63-1]")
    if type(one_based_step) is not int or one_based_step < 1:
        raise ValueError("one_based_step must be positive")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 16:
        raise ValueError("max_attempts must be inside [1, 16]")
    if not getattr(base_task, "family", None) or not getattr(base_task, "depth", None):
        raise ValueError("base task lacks family/depth coordinates")

    used_ids = {str(task.task_id) for task in excluded_tasks}
    used_prompts = {str(task.prompt) for task in excluded_tasks}
    candidates = [base_task]
    used_ids.add(str(base_task.task_id))
    used_prompts.add(str(base_task.prompt))
    for attempt in range(2, max_attempts + 1):
        for nonce in range(1_024):
            derived_seed = _stable_seed(
                campaign_seed,
                "learnability",
                one_based_step,
                attempt,
                base_task.family,
                base_task.depth,
                nonce,
            )
            candidate = process_task_battery(
                [str(base_task.family)],
                [int(base_task.depth)],
                1,
                seed=derived_seed,
            )[0]
            if (
                str(candidate.task_id) not in used_ids
                and str(candidate.prompt) not in used_prompts
            ):
                candidates.append(candidate)
                used_ids.add(str(candidate.task_id))
                used_prompts.add(str(candidate.prompt))
                break
        else:
            raise RuntimeError("learnability curriculum resampling exhausted")
    return candidates


def _task_sets(seed: int) -> tuple[list[Any], list[Any], list[Any]]:
    """Build disjoint process, answer-projection, and proxy curricula."""

    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be inside [0, 2^63-1]")
    process_tasks = process_task_battery(
        ["boolean", "modular"],
        [2],
        4,
        seed=seed,
    )
    answer_tasks = task_battery(
        ["boolean", "modular"],
        [2],
        4,
        seed=seed + 104_729,
        excluded_prompts=tuple(task.prompt for task in process_tasks),
        excluded_task_ids=tuple(task.task_id for task in process_tasks),
    )
    excluded = [*process_tasks, *answer_tasks]
    proxy_tasks = task_battery(
        ["boolean", "modular"],
        [2],
        2,
        seed=seed + 7_919,
        excluded_prompts=tuple(task.prompt for task in excluded),
        excluded_task_ids=tuple(task.task_id for task in excluded),
    )
    ids = [task.task_id for task in excluded + proxy_tasks]
    prompts = [task.prompt for task in excluded + proxy_tasks]
    if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
        raise RuntimeError("joint recurrent curriculum is not disjoint")
    return process_tasks, answer_tasks, proxy_tasks


def _grade_reward(
    task: Any,
    response_text: str,
    *,
    format_credit: float,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    verdict = dict(task.grade(response_text))
    verdict["correct"] = bool(verdict.get("correct"))
    process_reward = getattr(task, "process_reward", None)
    if callable(process_reward):
        reward_receipt = dict(
            process_reward(
                response_text,
                format_credit=min(float(format_credit), 0.05),
            )
        )
        reward = float(reward_receipt["reward"])
    else:
        reward = reward_from_verdict(verdict, format_credit=format_credit)
        reward_receipt = {
            "schema": "aura.recurrent_terminal_reward.v1",
            "policy": "exact_correctness_plus_bounded_format",
            "correct": verdict["correct"],
            "parsed": verdict.get("parsed") is not None,
            "format_credit": float(format_credit),
            "reward": float(reward),
        }
    return verdict, float(reward), reward_receipt


def _observable_grade_reward(
    task: Any,
    sample: Any,
    token_trace_adapter: Any,
    *,
    format_credit: float,
) -> tuple[dict[str, Any], dict[str, Any], float, dict[str, Any]]:
    """Grade only the authenticated response prefix visible before termination."""

    observable = observable_completion_from_adapter(
        token_trace_adapter,
        sample.tokens,
    )
    verdict, reward, reward_receipt = _grade_reward(
        task,
        str(observable["response_text"]),
        format_credit=format_credit,
    )
    return observable, verdict, reward, reward_receipt


def _commitment_ratchet_coverage(*reports: dict[str, Any]) -> dict[str, Any]:
    """Verify every complete-engine episode durably exposes its ratchet."""

    episode_count = 0
    valid_receipts = 0
    active_episode_count = 0
    turns = 0
    measured_commits = 0
    measured_narrowing = 0.0
    failures: list[dict[str, Any]] = []
    for report in reports:
        for record in report.get("records") or ():
            if not isinstance(record, dict):
                continue
            episode_count += 1
            episode = record.get("episode_receipt")
            ratchet = (
                episode.get("commitment_ratchet")
                if isinstance(episode, dict)
                else None
            )
            failure = ""
            if not isinstance(ratchet, dict):
                failure = "missing_receipt"
            elif ratchet.get("schema") != RATCHET_SCHEMA:
                failure = "schema_mismatch"
            else:
                claimed = ratchet.get("receipt_sha256")
                body = {
                    key: value
                    for key, value in ratchet.items()
                    if key != "receipt_sha256"
                }
                observed = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
                if claimed != observed:
                    failure = "digest_mismatch"
            if failure:
                failures.append(
                    {
                        "task_id": record.get("task_id"),
                        "depth": record.get("depth"),
                        "failure": failure,
                    }
                )
                continue
            valid_receipts += 1
            receipt_turns = int(ratchet.get("turns", 0) or 0)
            receipt_measured = int(ratchet.get("measured_commits", 0) or 0)
            turns += receipt_turns
            measured_commits += receipt_measured
            measured_narrowing += float(ratchet.get("measured_narrowing", 0.0) or 0.0)
            if receipt_turns > 0:
                active_episode_count += 1
    return {
        "schema": "aura.rlc.commitment_ratchet.coverage.v1",
        "episode_count": episode_count,
        "valid_receipts": valid_receipts,
        "active_episode_count": active_episode_count,
        "turns": turns,
        "measured_commits": measured_commits,
        "measured_narrowing_sum": round(measured_narrowing, 6),
        "failures": failures,
    }


def _status(out_dir: Path, phase: str, **detail: Any) -> None:
    body = {
        "schema": "aura.recurrent_grpo_behavioral_canary.status.v1",
        "phase": phase,
        "updated_at_unix_ns": time.time_ns(),
        **detail,
    }
    payload = {**body, "status_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    atomic_write_bytes(out_dir / "status.json", canonical_json_bytes(payload), mode=0o600)


def _sample_group(
    model: Any,
    tokenizer: Any,
    task: Any,
    *,
    spec: RLCExecutionSpec,
    group_size: int,
    campaign_seed: int,
    step: int,
    model_path: str,
    token_trace_adapter: Any,
) -> tuple[list[int], list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_tokens, _answer_tokens = tokenize_task(tokenizer, task.prompt, task.answer)
    sampling = free_generation_sampling_config()
    samples: list[Any] = []
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    max_attempts = group_size * 8
    for attempt in range(max_attempts):
        if len(samples) >= group_size:
            break
        sample_seed = _stable_seed(campaign_seed, "grpo", step, task.task_id, attempt)
        try:
            branch_index = attempt % len(spec.branch_roles)
            sample = sample_final_recurrent_transition_completion(
                model,
                prompt_tokens,
                spec=spec,
                branch_index=branch_index,
                seed=sample_seed,
                sampling=sampling,
                tokenizer=tokenizer,
                model_path=model_path,
                episode_id=(f"grpo-canary-{campaign_seed}-{step}-{attempt}"),
            )
        except RecurrentSamplingAdmissionError as exc:
            rejected.append(exc.sample.receipt())
            continue
        observable, verdict, reward, reward_receipt = _observable_grade_reward(
            task,
            sample,
            token_trace_adapter,
            format_credit=0.1,
        )
        response_text = str(observable["response_text"])
        samples.append(sample)
        rows.append(
            {
                "sample_seed": sample_seed,
                "branch_index": sample.branch_index,
                "response_text": response_text,
                "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
                "verdict": verdict,
                "reward": reward,
                "reward_receipt": reward_receipt,
                "observable_completion": observable,
                "sample_receipt": sample.receipt(),
            }
        )
    if len(samples) != group_size:
        raise RecurrentCanarySamplingError(
            admitted=len(samples),
            requested=group_size,
            rejected=rejected,
        )
    return prompt_tokens, samples, rows, rejected


def run_canary(
    *,
    model_path: Path,
    out_dir: Path,
    steps: int,
    group_size: int,
    max_learnability_attempts: int,
    seed: int,
    learning_rate: float,
    bootstrap_steps: int,
    bootstrap_learning_rate: float,
    specialization_steps: int,
    specialization_learning_rate: float,
    projection_steps: int,
    projection_learning_rate: float,
    memory_fraction: float,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm import load

    from core.learning.recurrence_native_objective_v2 import (
        ExactAdjointTrajectoryConfig,
        cached_supervised_live_path_value_and_grad,
    )
    from core.learning.recurrence_native_objective_v5 import (
        GeneratedRollinSelectionConfig,
    )
    from core.learning.recurrence_native_objective_v6 import (
        COMPOSITE_DEPTH_RECEIPT_SCHEMA,
        BranchSpecializationConfig,
        branch_specialization_live_path_loss,
        branch_specialization_live_path_value_and_grad,
        generated_rollin_specialization_value_and_grad,
        validate_branch_specialization_receipt,
        validate_generated_rollin_specialization_receipt,
    )

    if type(steps) is not int or not 1 <= steps <= 16:
        raise ValueError("steps must be inside [1, 16]")
    if type(group_size) is not int or not 2 <= group_size <= 8:
        raise ValueError("group_size must be inside [2, 8]")
    if (
        type(max_learnability_attempts) is not int
        or not 1 <= max_learnability_attempts <= 16
    ):
        raise ValueError("max_learnability_attempts must be inside [1, 16]")
    if type(bootstrap_steps) is not int or not 1 <= bootstrap_steps <= 32:
        raise ValueError("bootstrap_steps must be inside [1, 32]")
    if type(specialization_steps) is not int or not 1 <= specialization_steps <= 16:
        raise ValueError("specialization_steps must be inside [1, 16]")
    if type(projection_steps) is not int or not 1 <= projection_steps <= 16:
        raise ValueError("projection_steps must be inside [1, 16]")
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be inside [0, 2^63-1]")
    for name, value in (
        ("learning_rate", learning_rate),
        ("bootstrap_learning_rate", bootstrap_learning_rate),
        ("specialization_learning_rate", specialization_learning_rate),
        ("projection_learning_rate", projection_learning_rate),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 1e-7 <= float(value) <= 1e-3
        ):
            raise ValueError(f"{name} must be inside [1e-7, 1e-3]")

    started = time.time()
    source_commit, source_bindings = _source_state()
    out_dir.mkdir(parents=True, exist_ok=False)
    _status(out_dir, "source_bound", source_commit=source_commit)
    base_before = full_weight_checkpoint_identity(model_path)
    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution", "critical_audit"),
        recurrent_steps=2,
        exchange_interval=1,
    )
    grpo_config = RecurrentGRPOConfig(
        kl_coefficient=0.04,
        advantage_clip=4.0,
    )
    generated_config = GeneratedRollinSelectionConfig(
        student_forcing_probability=0.5,
        sampling_temperature=0.8,
        branch_softmin_temperature=0.5,
    )
    specialization_config = BranchSpecializationConfig(
        weight=8.0,
        target_separation=0.30,
    )
    trajectory_config = ExactAdjointTrajectoryConfig(
        probe_steps=(1, 2),
        improvement_weight=1.0,
        improvement_margin=0.05,
    )
    with (
        standalone_model_lane(
            owner_id=f"recurrent-grpo-canary:{out_dir.name}",
            model_path=str(model_path),
            purpose="training",
            preemptible=False,
            metadata={
                "tool": "run_recurrent_grpo_behavioral_canary",
                "source_commit": source_commit,
            },
        ),
        mlx_memory_envelope(
            fraction=memory_fraction,
            restore_limits_on_exit=True,
        ),
    ):
        _status(out_dir, "loading_model")
        model, tokenizer = load(str(model_path))
        token_trace_adapter = build_resident_tokenizer_trace_adapter(
            tokenizer,
            model_path,
        )
        attach_recurrent_policy_adapters(
            model,
            spec,
            lora_rank=4,
            lora_layers=4,
            lora_targets=("o_proj",),
            initialization_seed=(seed ^ 0x51F7A11) & 0xFFFFFFFF,
            lora_scale=1.0,
            depth_conditioned_steps=spec.recurrent_steps,
            role_conditioned_branches=len(spec.branch_roles),
        )
        adapter_before = adapter_tensor_fingerprint(adapter_tensor_dict(model))
        training_tasks, answer_tasks, proxy_tasks = _task_sets(seed)
        answer_rows: list[dict[str, Any]] = []
        for task in answer_tasks:
            prompt_tokens, answer_tokens = tokenize_task(
                tokenizer,
                task.prompt,
                task.answer,
            )
            answer_rows.append(
                {
                    "task_id": task.task_id,
                    "prompt_tokens": prompt_tokens,
                    "answer_tokens": answer_tokens,
                }
            )
        proxy_manifest, proxy_manifest_sha256 = build_recurrence_task_manifest(proxy_tasks)
        _status(out_dir, "baseline_probe")
        before = build_behavioral_probe_report(
            model,
            tokenizer,
            proxy_tasks,
            spec=spec,
            arm="initial_adapter",
            adapter_sha256=adapter_before,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
        )
        _status(out_dir, "initial_complete_engine_probe")
        initial_ordinary, initial_full_engine = build_paired_full_engine_probe_reports(
            model,
            tokenizer,
            proxy_tasks,
            model_path=model_path,
            spec=spec,
            adapter_sha256=adapter_before,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
        )
        bootstrap_optimizer = optim.AdamW(
            learning_rate=float(bootstrap_learning_rate),
            weight_decay=0.0,
        )
        bootstrap_optimizer.init(model.trainable_parameters())
        bootstrap_trail: list[dict[str, Any]] = []
        for bootstrap_step in range(1, bootstrap_steps + 1):
            task = _cyclic_task(training_tasks, one_based_step=bootstrap_step)
            prompt_tokens, answer_tokens = tokenize_task(
                tokenizer,
                task.prompt,
                task.answer,
            )
            _status(
                out_dir,
                "process_bootstrap",
                step=bootstrap_step,
                steps=bootstrap_steps,
                task_id=task.task_id,
            )
            adapter_step_before = adapter_tensor_fingerprint(adapter_tensor_dict(model))
            result = cached_supervised_live_path_value_and_grad(
                model,
                prompt_tokens,
                answer_tokens,
                spec=spec,
            )
            bootstrap_optimizer.update(model, result.gradients)
            mx.eval(model.trainable_parameters(), bootstrap_optimizer.state)
            adapter_step_after = adapter_tensor_fingerprint(adapter_tensor_dict(model))
            if adapter_step_after == adapter_step_before:
                raise RuntimeError("process bootstrap update did not mutate the policy")
            bootstrap_trail.append(
                {
                    "step": bootstrap_step,
                    "task_id": task.task_id,
                    "loss": float(result.value),
                    "branch_values": list(result.branch_values),
                    "answer_token_count": result.answer_token_count,
                    "execution_spec_sha256": result.execution_spec_sha256,
                    "prompt_tokens_sha256": result.prompt_tokens_sha256,
                    "answer_tokens_sha256": result.answer_tokens_sha256,
                    "adapter_before_sha256": adapter_step_before,
                    "adapter_after_sha256": adapter_step_after,
                }
            )
            bootstrap_body = {
                "schema": "aura.recurrent_process_bootstrap.journal.v1",
                "source_commit": source_commit,
                "execution_spec_sha256": spec.sha256,
                "seed": seed,
                "configured_steps": bootstrap_steps,
                "completed_steps": bootstrap_step,
                "trail": bootstrap_trail,
            }
            atomic_write_bytes(
                out_dir / "bootstrap_journal.json",
                canonical_json_bytes(
                    {
                        **bootstrap_body,
                        "journal_sha256": hashlib.sha256(
                            canonical_json_bytes(bootstrap_body)
                        ).hexdigest(),
                    }
                ),
                mode=0o600,
            )
            print(
                "[recurrent-grpo-canary] "
                f"bootstrap={bootstrap_step}/{bootstrap_steps} "
                f"task={task.task_id} loss={result.value:.4f}",
                flush=True,
            )
            del result
            mx.synchronize()
            mx.clear_cache()
        del bootstrap_optimizer
        mx.clear_cache()

        specialization_optimizer = optim.AdamW(
            learning_rate=float(specialization_learning_rate),
            weight_decay=0.0,
        )
        specialization_optimizer.init(model.trainable_parameters())
        specialization_trail: list[dict[str, Any]] = []
        for specialization_step in range(1, specialization_steps + 1):
            row = _cyclic_task(answer_rows, one_based_step=specialization_step)
            _status(
                out_dir,
                "branch_specialization",
                step=specialization_step,
                steps=specialization_steps,
                task_id=row["task_id"],
            )
            adapter_step_before = adapter_tensor_fingerprint(adapter_tensor_dict(model))
            result = branch_specialization_live_path_value_and_grad(
                model,
                row["prompt_tokens"],
                spec=spec,
                config=specialization_config,
            )
            objective_receipt = validate_branch_specialization_receipt(
                result.evaluation.receipt()
            )
            specialization_optimizer.update(model, result.gradients)
            mx.eval(model.trainable_parameters(), specialization_optimizer.state)
            post_update = branch_specialization_live_path_loss(
                model,
                row["prompt_tokens"],
                spec=spec,
                config=specialization_config,
            )
            adapter_step_after = adapter_tensor_fingerprint(adapter_tensor_dict(model))
            if adapter_step_after == adapter_step_before:
                raise RuntimeError("branch specialization update did not mutate the policy")
            specialization_trail.append(
                {
                    "step": specialization_step,
                    "task_id": row["task_id"],
                    "loss_before": float(result.value),
                    "separations_before": list(result.evaluation.separations),
                    "loss_after": float(post_update.value),
                    "separations_after": list(post_update.separations),
                    "objective_receipt": objective_receipt,
                    "adapter_before_sha256": adapter_step_before,
                    "adapter_after_sha256": adapter_step_after,
                }
            )
            specialization_body = {
                "schema": "aura.recurrent_branch_specialization.journal.v1",
                "source_commit": source_commit,
                "execution_spec_sha256": spec.sha256,
                "seed": seed,
                "configured_steps": specialization_steps,
                "completed_steps": specialization_step,
                "trail": specialization_trail,
            }
            atomic_write_bytes(
                out_dir / "specialization_journal.json",
                canonical_json_bytes(
                    {
                        **specialization_body,
                        "journal_sha256": hashlib.sha256(
                            canonical_json_bytes(specialization_body)
                        ).hexdigest(),
                    }
                ),
                mode=0o600,
            )
            print(
                "[recurrent-grpo-canary] "
                f"specialization={specialization_step}/{specialization_steps} "
                f"task={row['task_id']} loss={post_update.value:.4f} "
                f"min_separation={min(post_update.separations):.4f}",
                flush=True,
            )
            del result, post_update
            mx.synchronize()
            mx.clear_cache()
        del specialization_optimizer
        mx.clear_cache()

        projection_optimizer = optim.AdamW(
            learning_rate=float(projection_learning_rate),
            weight_decay=0.0,
        )
        projection_optimizer.init(model.trainable_parameters())
        projection_trail: list[dict[str, Any]] = []
        for projection_step in range(1, projection_steps + 1):
            row = _cyclic_task(answer_rows, one_based_step=projection_step)
            _status(
                out_dir,
                "answer_projection",
                step=projection_step,
                steps=projection_steps,
                task_id=row["task_id"],
            )
            rollin_seed = _projection_rollin_seed(
                campaign_seed=seed,
                task_id=row["task_id"],
                sample_ordinal=projection_step,
                execution_spec_sha256=spec.sha256,
            )
            adapter_step_before = adapter_tensor_fingerprint(adapter_tensor_dict(model))
            policy_before = recurrent_policy_sha256(model, spec)
            result = generated_rollin_specialization_value_and_grad(
                model,
                row["prompt_tokens"],
                row["answer_tokens"],
                spec=spec,
                base_seed=rollin_seed,
                generated_config=generated_config,
                specialization_config=specialization_config,
                trajectory_config=trajectory_config,
                trajectory_policy_sha256=policy_before,
            )
            objective_receipt = validate_generated_rollin_specialization_receipt(
                result.evaluation.receipt()
            )
            projection_optimizer.update(model, result.gradients)
            mx.eval(model.trainable_parameters(), projection_optimizer.state)
            adapter_step_after = adapter_tensor_fingerprint(adapter_tensor_dict(model))
            policy_after = recurrent_policy_sha256(model, spec)
            if adapter_step_after == adapter_step_before or policy_after == policy_before:
                raise RuntimeError("answer projection update did not mutate the policy")
            trajectory = result.evaluation.trajectory
            if trajectory is None:
                raise RuntimeError("answer projection omitted paired-depth evidence")
            projection_trail.append(
                {
                    "step": projection_step,
                    "task_id": row["task_id"],
                    "loss": float(result.value),
                    "lexical_loss": float(result.evaluation.generated.value),
                    "specialization_loss": float(result.evaluation.specialization.value),
                    "trajectory_loss": float(trajectory.value),
                    "branch_separations": list(result.evaluation.specialization.separations),
                    "branch_values": list(result.branch_values),
                    "branch_weights": list(result.branch_weights),
                    "rollin_base_seed": rollin_seed,
                    "objective_receipt": objective_receipt,
                    "adapter_before_sha256": adapter_step_before,
                    "adapter_after_sha256": adapter_step_after,
                    "policy_before_sha256": policy_before,
                    "policy_after_sha256": policy_after,
                }
            )
            projection_body = {
                "schema": "aura.recurrent_answer_projection.journal.v1",
                "source_commit": source_commit,
                "execution_spec_sha256": spec.sha256,
                "seed": seed,
                "configured_steps": projection_steps,
                "completed_steps": projection_step,
                "trail": projection_trail,
            }
            atomic_write_bytes(
                out_dir / "projection_journal.json",
                canonical_json_bytes(
                    {
                        **projection_body,
                        "journal_sha256": hashlib.sha256(
                            canonical_json_bytes(projection_body)
                        ).hexdigest(),
                    }
                ),
                mode=0o600,
            )
            print(
                "[recurrent-grpo-canary] "
                f"projection={projection_step}/{projection_steps} "
                f"task={row['task_id']} loss={result.value:.4f} "
                f"depth_loss={trajectory.value:.4f}",
                flush=True,
            )
            del result
            mx.synchronize()
            mx.clear_cache()
        del projection_optimizer
        mx.clear_cache()

        specialization_panel: list[dict[str, Any]] = []
        for row in answer_rows:
            evaluation = branch_specialization_live_path_loss(
                model,
                row["prompt_tokens"],
                spec=spec,
                config=specialization_config,
            )
            specialization_panel.append(
                {
                    "task_id": row["task_id"],
                    "loss": float(evaluation.value),
                    "separations": list(evaluation.separations),
                    "objective_receipt": validate_branch_specialization_receipt(
                        evaluation.receipt()
                    ),
                }
            )
            del evaluation
            mx.clear_cache()

        optimizer = build_recurrent_policy_optimizer(float(learning_rate))
        optimizer.init(model.trainable_parameters())
        step_receipts: list[dict[str, Any]] = []
        learnability_retry_tasks: list[Any] = []
        optimizer_updates = 0
        all_samples_admitted = True
        for step in range(1, steps + 1):
            base_task = _cyclic_task(training_tasks, one_based_step=step)
            candidate_tasks = _learnability_task_candidates(
                base_task,
                campaign_seed=seed,
                one_based_step=step,
                max_attempts=max_learnability_attempts,
                excluded_tasks=[
                    *training_tasks,
                    *answer_tasks,
                    *proxy_tasks,
                    *learnability_retry_tasks,
                ],
            )
            learnability_retry_tasks.extend(candidate_tasks[1:])
            _status(
                out_dir,
                "sampling",
                step=step,
                steps=steps,
                task_id=base_task.task_id,
                learnability_attempt=1,
                max_learnability_attempts=max_learnability_attempts,
                optimizer_updates=optimizer_updates,
            )
            policy_before = recurrent_policy_sha256(model, spec)
            adapter_step_before = adapter_tensor_fingerprint(adapter_tensor_dict(model))
            attempt_receipts: list[dict[str, Any]] = []
            selected_task = base_task
            selected_advantage: dict[str, Any] | None = None
            selected_objective: dict[str, Any] | None = None
            selected_rows: list[dict[str, Any]] = []
            selected_rejected: list[dict[str, Any]] = []
            selected_attempt = 0
            for attempt, task in enumerate(candidate_tasks, start=1):
                sampling_coordinate = (
                    (step - 1) * max_learnability_attempts + attempt
                )
                _status(
                    out_dir,
                    "sampling",
                    step=step,
                    steps=steps,
                    task_id=task.task_id,
                    nominal_task_id=base_task.task_id,
                    learnability_attempt=attempt,
                    max_learnability_attempts=max_learnability_attempts,
                    optimizer_updates=optimizer_updates,
                )
                attempt_policy_before = recurrent_policy_sha256(model, spec)
                attempt_adapter_before = adapter_tensor_fingerprint(
                    adapter_tensor_dict(model)
                )
                prompt_tokens, samples, rows, rejected = _sample_group(
                    model,
                    tokenizer,
                    task,
                    spec=spec,
                    group_size=group_size,
                    campaign_seed=seed,
                    step=sampling_coordinate,
                    model_path=str(model_path),
                    token_trace_adapter=token_trace_adapter,
                )
                rewards = [float(row["reward"]) for row in rows]
                advantage = group_advantages(
                    rewards,
                    clip=grpo_config.advantage_clip,
                )
                objective: dict[str, Any] | None = None
                if not advantage["degenerate"]:
                    _status(
                        out_dir,
                        "optimizing",
                        step=step,
                        steps=steps,
                        task_id=task.task_id,
                        nominal_task_id=base_task.task_id,
                        learnability_attempt=attempt,
                        optimizer_updates=optimizer_updates,
                    )
                    result = exact_adjoint_sampled_group_value_and_grad(
                        model,
                        prompt_tokens,
                        samples,
                        rewards,
                        spec=spec,
                        config=grpo_config,
                        optimization_token_counts=[
                            int(
                                row["observable_completion"][
                                    "optimization_token_count"
                                ]
                            )
                            for row in rows
                        ],
                    )
                    optimizer.update(model, result.gradients)
                    mx.eval(model.trainable_parameters(), optimizer.state)
                    optimizer_updates += 1
                    objective = {
                        "advantage_report": result.advantage_report,
                        "reference_kl": result.reference_kl,
                        "old_policy_approx_kl": result.old_policy_approx_kl,
                        "clip_fraction": result.clip_fraction,
                        "policy_loss": result.policy_loss,
                        "objective_at_sampling": result.objective_at_sampling,
                        "gradient_surrogate_value": result.gradient_surrogate_value,
                        "completion_count": result.completion_count,
                        "token_count": result.token_count,
                        "branch_indices": list(result.branch_indices),
                    }
                    del result
                attempt_policy_after = recurrent_policy_sha256(model, spec)
                attempt_adapter_after = adapter_tensor_fingerprint(
                    adapter_tensor_dict(model)
                )
                if advantage["degenerate"]:
                    if (
                        attempt_policy_after != attempt_policy_before
                        or attempt_adapter_after != attempt_adapter_before
                    ):
                        raise RuntimeError("degenerate recurrent group mutated the policy")
                elif (
                    attempt_policy_after == attempt_policy_before
                    or attempt_adapter_after == attempt_adapter_before
                ):
                    raise RuntimeError("recurrent optimizer update did not mutate the policy")
                all_samples_admitted = all_samples_admitted and all(
                    bool(sample.behavior_admitted) for sample in samples
                )
                attempt_receipts.append(
                    {
                        "attempt": attempt,
                        "sampling_coordinate": sampling_coordinate,
                        "task_id": task.task_id,
                        "task_family": task.family,
                        "task_depth": task.depth,
                        "task_prompt_sha256": hashlib.sha256(
                            task.prompt.encode("utf-8")
                        ).hexdigest(),
                        "policy_before_sha256": attempt_policy_before,
                        "policy_after_sha256": attempt_policy_after,
                        "adapter_before_sha256": attempt_adapter_before,
                        "adapter_after_sha256": attempt_adapter_after,
                        "advantage_report": advantage,
                        "optimizer_updated": not advantage["degenerate"],
                        "objective": objective,
                        "samples": rows,
                        "rejected_sample_receipts": rejected,
                    }
                )
                selected_task = task
                selected_advantage = advantage
                selected_objective = objective
                selected_rows = rows
                selected_rejected = rejected
                selected_attempt = attempt
                del samples
                mx.synchronize()
                mx.clear_cache()
                if not advantage["degenerate"]:
                    break
            if selected_advantage is None or selected_attempt < 1:
                raise RuntimeError("learnability search produced no sampled group")
            policy_after = recurrent_policy_sha256(model, spec)
            adapter_step_after = adapter_tensor_fingerprint(adapter_tensor_dict(model))
            optimizer_updated = not selected_advantage["degenerate"]
            if optimizer_updated:
                if selected_attempt != len(attempt_receipts):
                    raise RuntimeError("learnability search continued after an update")
            elif len(attempt_receipts) != max_learnability_attempts:
                raise RuntimeError("learnability search stopped before exhaustion")
            step_receipts.append(
                {
                    "step": step,
                    "task_id": base_task.task_id,
                    "selected_task_id": selected_task.task_id,
                    "selected_attempt": selected_attempt,
                    "learnability_search_policy": LEARNABILITY_SEARCH_POLICY,
                    "learnability_search_exhausted": not optimizer_updated,
                    "candidate_task_ids": [
                        candidate.task_id for candidate in candidate_tasks
                    ],
                    "policy_before_sha256": policy_before,
                    "policy_after_sha256": policy_after,
                    "adapter_before_sha256": adapter_step_before,
                    "adapter_after_sha256": adapter_step_after,
                    "advantage_report": selected_advantage,
                    "optimizer_updated": optimizer_updated,
                    "objective": selected_objective,
                    "samples": selected_rows,
                    "rejected_sample_receipts": selected_rejected,
                    "learnability_attempts": attempt_receipts,
                }
            )
            journal_body = {
                "schema": "aura.recurrent_grpo_behavioral_canary.journal.v2",
                "source_commit": source_commit,
                "execution_spec_sha256": spec.sha256,
                "seed": seed,
                "configured_steps": steps,
                "max_learnability_attempts": max_learnability_attempts,
                "learnability_search_policy": LEARNABILITY_SEARCH_POLICY,
                "completed_steps": step,
                "optimizer_updates": optimizer_updates,
                "step_receipts": step_receipts,
            }
            journal = {
                **journal_body,
                "journal_sha256": hashlib.sha256(canonical_json_bytes(journal_body)).hexdigest(),
            }
            atomic_write_bytes(
                out_dir / "step_journal.json",
                canonical_json_bytes(journal),
                mode=0o600,
            )
            print(
                "[recurrent-grpo-canary] "
                f"step={step}/{steps} task={selected_task.task_id} "
                f"attempt={selected_attempt}/{max_learnability_attempts} "
                f"mean_reward={selected_advantage['mean_reward']:.3f} "
                f"reward_std={selected_advantage['reward_std']:.3f} "
                f"updates={optimizer_updates}",
                flush=True,
            )
        adapter = adapter_tensor_dict(model)
        adapter_after = adapter_tensor_fingerprint(adapter)
        _status(
            out_dir,
            "trained_probe",
            optimizer_updates=optimizer_updates,
        )
        after = build_behavioral_probe_report(
            model,
            tokenizer,
            proxy_tasks,
            spec=spec,
            arm="trained_adapter",
            adapter_sha256=adapter_after,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
        )
        _status(out_dir, "trained_complete_engine_probe")
        trained_ordinary, trained_full_engine = build_paired_full_engine_probe_reports(
            model,
            tokenizer,
            proxy_tasks,
            model_path=model_path,
            spec=spec,
            adapter_sha256=adapter_after,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
        )
        admission: dict[str, Any] | None = None
        admission_error = ""
        try:
            admission = build_checkpoint_behavioral_admission(
                initial_report=before,
                trained_report=after,
                task_manifest=proxy_manifest,
                ordinary_decode_report=trained_ordinary,
            )
            validate_checkpoint_behavioral_admission(
                admission,
                initial_report=before,
                trained_report=after,
                task_manifest=proxy_manifest,
                ordinary_decode_report=trained_ordinary,
            )
        except RecurrentCheckpointAdmissionError as exc:
            admission_error = str(exc)
        full_engine_admission: dict[str, Any] | None = None
        full_engine_admission_error = ""
        try:
            full_engine_admission = build_full_engine_behavioral_admission(
                initial_full_engine_report=initial_full_engine,
                trained_full_engine_report=trained_full_engine,
                initial_ordinary_decode_report=initial_ordinary,
                trained_ordinary_decode_report=trained_ordinary,
                task_manifest=proxy_manifest,
            )
            validate_full_engine_behavioral_admission(
                full_engine_admission,
                initial_full_engine_report=initial_full_engine,
                trained_full_engine_report=trained_full_engine,
                initial_ordinary_decode_report=initial_ordinary,
                trained_ordinary_decode_report=trained_ordinary,
                task_manifest=proxy_manifest,
            )
        except RecurrentCheckpointAdmissionError as exc:
            full_engine_admission_error = str(exc)
        mx.save_safetensors(str(out_dir / "adapter.safetensors"), adapter)

    base_after = full_weight_checkpoint_identity(model_path)
    commitment_ratchet_coverage = _commitment_ratchet_coverage(
        initial_full_engine,
        trained_full_engine,
    )
    gates = {
        "base_checkpoint_immutable": base_before == base_after,
        "all_samples_admitted": all_samples_admitted,
        "optimizer_signal_observed": optimizer_updates > 0,
        "process_bootstrap_completed": len(bootstrap_trail) == bootstrap_steps,
        "branch_specialization_completed": (
            len(specialization_trail) == specialization_steps
        ),
        "answer_projection_completed": len(projection_trail) == projection_steps,
        "branch_specialization_target_met": bool(
            specialization_panel
            and all(
                row["separations"]
                and min(row["separations"])
                >= float(specialization_config.target_separation)
                for row in specialization_panel
            )
        ),
        "paired_depth_projection_exercised": bool(
            projection_trail
            and all(
                row["objective_receipt"].get("schema")
                == COMPOSITE_DEPTH_RECEIPT_SCHEMA
                and "trajectory_receipt" in row["objective_receipt"]
                for row in projection_trail
            )
        ),
        "commitment_ratchet_receipted": bool(
            commitment_ratchet_coverage["episode_count"] > 0
            and commitment_ratchet_coverage["valid_receipts"]
            == commitment_ratchet_coverage["episode_count"]
        ),
        "commitment_ratchet_exercised": bool(
            commitment_ratchet_coverage["active_episode_count"] > 0
        ),
        "adapter_mutated": adapter_before != adapter_after,
        "complete_engine_heldout_strict_gain": bool(
            full_engine_admission is not None and full_engine_admission["admitted"]
        ),
    }
    body = {
        "schema": CANARY_SCHEMA,
        "source_commit": source_commit,
        "source_bindings": source_bindings,
        "model_path": str(model_path),
        "base_before": base_before,
        "base_after": base_after,
        "execution_spec": spec.to_dict(),
        "execution_spec_sha256": spec.sha256,
        "grpo_config": {
            "clip_epsilon": grpo_config.clip_epsilon,
            "kl_coefficient": grpo_config.kl_coefficient,
            "advantage_clip": grpo_config.advantage_clip,
            "group_size": group_size,
            "learning_rate": float(learning_rate),
            "sampling": free_generation_sampling_config().to_dict(),
            "reward": "exact_correctness_or_bounded_public_transition_prefix",
            "learnability_search": {
                "policy": LEARNABILITY_SEARCH_POLICY,
                "max_attempts_per_step": max_learnability_attempts,
                "heldout_feedback_allowed": False,
                "task_family_and_depth_preserved": True,
            },
        },
        "adapter_config": {
            "lora_rank": 4,
            "lora_layers": 4,
            "lora_targets": ["o_proj"],
            "depth_conditioned_steps": spec.recurrent_steps,
            "role_conditioned_branches": len(spec.branch_roles),
        },
        "process_bootstrap_config": {
            "steps": bootstrap_steps,
            "learning_rate": float(bootstrap_learning_rate),
            "optimizer": "mlx.optimizers.AdamW",
            "weight_decay": 0.0,
            "objective": "exact_cached_live_path_full_public_trace_ce",
        },
        "joint_answer_projection_config": {
            "specialization_steps": specialization_steps,
            "specialization_learning_rate": float(specialization_learning_rate),
            "projection_steps": projection_steps,
            "projection_learning_rate": float(projection_learning_rate),
            "generated": generated_config.to_dict(),
            "specialization": specialization_config.to_dict(),
            "trajectory": trajectory_config.to_dict(),
            "objective": "generated_prefix_role_and_depth_specialized_answer_ce",
        },
        "seed": seed,
        "steps": steps,
        "optimizer_updates": optimizer_updates,
        "adapter_before_sha256": adapter_before,
        "adapter_after_sha256": adapter_after,
        "process_training_task_ids": [task.task_id for task in training_tasks],
        "learnability_retry_task_ids": [
            task.task_id for task in learnability_retry_tasks
        ],
        "answer_projection_task_ids": [task.task_id for task in answer_tasks],
        "process_bootstrap_trail": bootstrap_trail,
        "branch_specialization_trail": specialization_trail,
        "answer_projection_trail": projection_trail,
        "branch_specialization_panel": specialization_panel,
        "commitment_ratchet_coverage": commitment_ratchet_coverage,
        "proxy_task_manifest": proxy_manifest,
        "proxy_task_manifest_sha256": proxy_manifest_sha256,
        "free_generation_before": before,
        "initial_ordinary_decode": initial_ordinary,
        "initial_full_engine": initial_full_engine,
        "step_receipts": step_receipts,
        "free_generation_after": after,
        "trained_ordinary_decode": trained_ordinary,
        "trained_full_engine": trained_full_engine,
        "checkpoint_behavioral_admission": admission,
        "checkpoint_behavioral_admission_error": admission_error,
        "full_engine_behavioral_admission": full_engine_admission,
        "full_engine_behavioral_admission_error": full_engine_admission_error,
        "gates": gates,
        "passed": all(gates.values()),
        "claim_state": {
            "reasoning_gain_proven": False,
            "frontier_level_proven": False,
            "promotion_allowed": False,
            "fusion_allowed": False,
            "resident_campaign_admitted": False,
        },
        "elapsed_s": time.time() - started,
    }
    receipt = {
        **body,
        "receipt_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }
    atomic_write_bytes(
        out_dir / "receipt.json",
        canonical_json_bytes(receipt),
        mode=0o600,
    )
    _status(
        out_dir,
        "completed",
        passed=receipt["passed"],
        optimizer_updates=optimizer_updates,
        receipt_sha256=receipt["receipt_sha256"],
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-learnability-attempts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026080701)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--bootstrap-steps", type=int, default=32)
    parser.add_argument("--bootstrap-learning-rate", type=float, default=1e-4)
    parser.add_argument("--specialization-steps", type=int, default=8)
    parser.add_argument("--specialization-learning-rate", type=float, default=1e-3)
    parser.add_argument("--projection-steps", type=int, default=8)
    parser.add_argument("--projection-learning-rate", type=float, default=1e-4)
    parser.add_argument("--memory-fraction", type=float, default=0.35)
    args = parser.parse_args()
    out_dir = args.out_dir.expanduser().resolve(strict=False)
    try:
        receipt = run_canary(
            model_path=args.model.expanduser().resolve(strict=True),
            out_dir=out_dir,
            steps=args.steps,
            group_size=args.group_size,
            max_learnability_attempts=args.max_learnability_attempts,
            seed=args.seed,
            learning_rate=args.learning_rate,
            bootstrap_steps=args.bootstrap_steps,
            bootstrap_learning_rate=args.bootstrap_learning_rate,
            specialization_steps=args.specialization_steps,
            specialization_learning_rate=args.specialization_learning_rate,
            projection_steps=args.projection_steps,
            projection_learning_rate=args.projection_learning_rate,
            memory_fraction=args.memory_fraction,
        )
    except Exception as exc:
        if out_dir.is_dir():
            failure = {
                "schema": "aura.recurrent_grpo_behavioral_canary.failure.v1",
                "error_type": type(exc).__name__,
                "error": str(exc)[:4_000],
                "recorded_at_unix_ns": time.time_ns(),
            }
            if isinstance(exc, RecurrentCanarySamplingError):
                failure["sampling_failure"] = {
                    "admitted": exc.admitted,
                    "requested": exc.requested,
                    "rejected": len(exc.rejected),
                    "diagnostics": exc.diagnostics,
                    "rejected_sample_receipts": exc.rejected,
                }
            atomic_write_bytes(
                out_dir / "failure.json",
                canonical_json_bytes(failure),
                mode=0o600,
            )
        raise
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "passed": receipt["passed"],
                "optimizer_updates": receipt["optimizer_updates"],
                "gates": receipt["gates"],
                "elapsed_s": receipt["elapsed_s"],
                "receipt_sha256": receipt["receipt_sha256"],
                "receipt_path": str(out_dir / "receipt.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
