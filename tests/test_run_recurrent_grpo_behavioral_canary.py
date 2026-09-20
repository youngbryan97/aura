from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.recurrence_curriculum import task_battery
from core.learning.recurrent_behavioral_probe import (
    _full_engine_config,
    _normalize_full_engine_probe_depths,
)
from tools import run_recurrent_grpo_behavioral_canary as canary


def _repo(tmp_path: Path) -> tuple[Path, str]:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
    )
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="ascii")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=tmp_path, check=True)
    head = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", head],
        cwd=tmp_path,
        check=True,
    )
    return source, head


def test_source_state_requires_clean_published_exact_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, head = _repo(tmp_path)
    monkeypatch.setattr(canary, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(canary, "SOURCE_PATHS", ("source.py",))

    observed_head, bindings = canary._source_state()

    assert observed_head == head
    assert bindings["source.py"]["size_bytes"] == len(b"VALUE = 1\n")
    source.write_text("VALUE = 2\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="clean source"):
        canary._source_state()


def test_source_state_rejects_unpublished_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source, _head = _repo(tmp_path)
    (tmp_path / "other.py").write_text("VALUE = 2\n", encoding="ascii")
    subprocess.run(["git", "add", "other.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "ahead"], cwd=tmp_path, check=True)
    monkeypatch.setattr(canary, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(canary, "SOURCE_PATHS", ("source.py",))

    with pytest.raises(RuntimeError, match="not published"):
        canary._source_state()


def test_stable_seed_binds_every_sampling_coordinate() -> None:
    first = canary._stable_seed(17, "grpo", 2, "task-a", 0)

    assert first == canary._stable_seed(17, "grpo", 2, "task-a", 0)
    assert first != canary._stable_seed(17, "grpo", 2, "task-a", 1)
    assert first != canary._stable_seed(17, "grpo", 3, "task-a", 0)


def test_projection_rollin_seed_uses_sealed_train_phase_with_domain_identity() -> None:
    first = canary._projection_rollin_seed(
        campaign_seed=17,
        task_id="task-a",
        sample_ordinal=1,
        execution_spec_sha256="a" * 64,
    )

    assert first == canary._projection_rollin_seed(
        campaign_seed=17,
        task_id="task-a",
        sample_ordinal=1,
        execution_spec_sha256="a" * 64,
    )
    assert first != canary._projection_rollin_seed(
        campaign_seed=17,
        task_id="task-a",
        sample_ordinal=2,
        execution_spec_sha256="a" * 64,
    )
    assert first != canary._projection_rollin_seed(
        campaign_seed=17,
        task_id="task-b",
        sample_ordinal=1,
        execution_spec_sha256="a" * 64,
    )


def test_training_cycle_covers_every_task_before_repeating() -> None:
    tasks = task_battery(["boolean", "modular"], [2], 2, seed=31)

    observed = [
        canary._cyclic_task(tasks, one_based_step=step).task_id
        for step in range(1, len(tasks) * 2 + 1)
    ]

    assert observed[: len(tasks)] == [task.task_id for task in tasks]
    assert observed[len(tasks) :] == observed[: len(tasks)]


def test_joint_curricula_are_balanced_and_strictly_disjoint() -> None:
    process, answer, proxy = canary._task_sets(2026080805)

    assert len(process) == 8
    assert len(answer) == 8
    assert len(proxy) == 4
    assert {task.family for task in process} == {"boolean", "modular"}
    assert {task.family for task in answer} == {"boolean", "modular"}
    assert {task.family for task in proxy} == {"boolean", "modular"}
    all_tasks = [*process, *answer, *proxy]
    assert len({task.task_id for task in all_tasks}) == len(all_tasks)
    assert len({task.prompt for task in all_tasks}) == len(all_tasks)


def test_joint_curricula_resample_process_prompt_collisions() -> None:
    process, answer, proxy = canary._task_sets(2026080903)

    all_tasks = [*process, *answer, *proxy]
    assert len(all_tasks) == 20
    assert len({task.task_id for task in all_tasks}) == len(all_tasks)
    assert len({task.prompt for task in all_tasks}) == len(all_tasks)


def test_learnability_search_is_deterministic_disjoint_and_same_stratum() -> None:
    process, answer, proxy = canary._task_sets(2026080903)
    base = process[0]
    excluded = [*process, *answer, *proxy]

    first = canary._learnability_task_candidates(
        base,
        campaign_seed=2026080903,
        one_based_step=1,
        max_attempts=4,
        excluded_tasks=excluded,
    )
    second = canary._learnability_task_candidates(
        base,
        campaign_seed=2026080903,
        one_based_step=1,
        max_attempts=4,
        excluded_tasks=excluded,
    )

    assert [task.task_id for task in first] == [task.task_id for task in second]
    assert first[0] is base
    assert len(first) == 4
    assert {task.family for task in first} == {base.family}
    assert {task.depth for task in first} == {base.depth}
    assert len({task.task_id for task in [*excluded, *first[1:]]}) == (
        len(excluded) + len(first) - 1
    )
    assert len({task.prompt for task in [*excluded, *first[1:]]}) == (
        len(excluded) + len(first) - 1
    )


def test_learnability_search_coordinates_change_the_retry_window() -> None:
    process, answer, proxy = canary._task_sets(2026080903)
    excluded = [*process, *answer, *proxy]

    first = canary._learnability_task_candidates(
        process[0],
        campaign_seed=2026080903,
        one_based_step=1,
        max_attempts=3,
        excluded_tasks=excluded,
    )
    later = canary._learnability_task_candidates(
        process[0],
        campaign_seed=2026080903,
        one_based_step=2,
        max_attempts=3,
        excluded_tasks=excluded,
    )

    assert [task.task_id for task in first[1:]] != [
        task.task_id for task in later[1:]
    ]


def test_source_contract_binds_every_joint_objective() -> None:
    assert len(canary.SOURCE_PATHS) == len(set(canary.SOURCE_PATHS))
    assert "core/learning/recurrence_native_objective_v2.py" in canary.SOURCE_PATHS
    assert "core/learning/recurrence_native_objective_v5.py" in canary.SOURCE_PATHS
    assert "core/learning/recurrence_native_objective_v6.py" in canary.SOURCE_PATHS
    assert "core/brain/llm/latent_cortex/incumbent_artifact.py" in canary.SOURCE_PATHS
    assert "core/brain/llm/latent_cortex/task_verifiers.py" in canary.SOURCE_PATHS
    assert "core/brain/llm/latent_cortex/commitment_extraction.py" in canary.SOURCE_PATHS
    assert "core/brain/llm/latent_cortex/commitment_ratchet.py" in canary.SOURCE_PATHS
    assert "core/brain/llm/latent_cortex/commitment_telemetry.py" in canary.SOURCE_PATHS
    assert "core/brain/llm/latent_cortex/sequential_exclusion.py" in canary.SOURCE_PATHS
    assert "core/brain/llm/latent_cortex/types.py" in canary.SOURCE_PATHS


def test_commitment_ratchet_coverage_requires_every_digest_bound_receipt() -> None:
    from core.brain.llm.latent_cortex.commitment_ratchet import (
        CommitmentRatchet,
        Constraint,
        ConstraintKind,
    )

    ratchet = CommitmentRatchet(["wrong", "right"])
    ratchet.commit(Constraint(kind=ConstraintKind.EXCLUDES, subject="wrong"))
    ratchet.seal()
    report = {
        "records": [
            {
                "task_id": "task-1",
                "depth": 1,
                "episode_receipt": {"commitment_ratchet": ratchet.receipt()},
            }
        ]
    }

    coverage = canary._commitment_ratchet_coverage(report)

    assert coverage["episode_count"] == 1
    assert coverage["valid_receipts"] == 1
    assert coverage["active_episode_count"] == 1
    assert coverage["turns"] == 1
    assert coverage["measured_commits"] == 1
    assert coverage["failures"] == []

    report["records"][0]["episode_receipt"]["commitment_ratchet"][
        "turns"
    ] = 2
    tampered = canary._commitment_ratchet_coverage(report)
    assert tampered["valid_receipts"] == 0
    assert tampered["failures"][0]["failure"] == "digest_mismatch"


def test_complete_engine_probe_is_not_the_naked_latent_ablation() -> None:
    config = _full_engine_config(
        RLCExecutionSpec(
            n_slots=4,
            branch_roles=("constructive_solution", "critical_audit"),
            recurrent_steps=2,
            exchange_interval=1,
        )
    )

    assert config.decode_incumbent_policy == "vanilla_incumbent"
    assert config.answer_replacement_enabled is True
    assert config.local_repair_enabled is True
    assert config.local_repair_max_attempts == 2
    assert config.verifier_probe_contract == "final_answer_v1"
    assert config.verifier_accept_non_regression is True
    assert config.latent_opt.enabled is True
    assert config.fast_weights.enabled is True
    assert config.allow_vanilla_fallback is False


def test_complete_engine_probe_can_remove_the_producer_without_removing_verification() -> None:
    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution", "critical_audit"),
        recurrent_steps=2,
        exchange_interval=1,
    )

    config = _full_engine_config(spec, objective_program_enabled=False)

    assert config.objective_program_enabled is False
    assert config.verified_objective_teacher_enabled is True
    assert config.answer_replacement_enabled is True
    assert config.verifier_probe_contract == "final_answer_v1"


def test_complete_engine_probe_rejects_non_boolean_producer_policy() -> None:
    with pytest.raises(TypeError, match="objective_program_enabled"):
        _full_engine_config(RLCExecutionSpec(), objective_program_enabled=1)


def test_complete_engine_probe_can_independently_remove_verified_neural_teacher() -> None:
    config = _full_engine_config(
        RLCExecutionSpec(),
        objective_program_enabled=False,
        verified_objective_teacher_enabled=False,
    )

    assert config.objective_program_enabled is False
    assert config.verified_objective_teacher_enabled is False

    with pytest.raises(TypeError, match="verified_objective_teacher_enabled"):
        _full_engine_config(
            RLCExecutionSpec(),
            verified_objective_teacher_enabled=1,
        )


def test_complete_engine_probe_depths_must_span_shallow_and_full() -> None:
    spec = RLCExecutionSpec(recurrent_steps=4)

    assert _normalize_full_engine_probe_depths(spec, None) == (1, 4)
    assert _normalize_full_engine_probe_depths(spec, (1, 2, 4)) == (1, 2, 4)
    for invalid in ((4,), (1,), (2, 4), (1, 2), (1, 4, 4), (4, 1)):
        with pytest.raises(ValueError, match="spanning shallow and full"):
            _normalize_full_engine_probe_depths(spec, invalid)


def test_reward_is_correctness_dominant_and_format_credit_bounded() -> None:
    task = task_battery(["boolean"], [2], 1, seed=43)[0]

    correct_verdict, correct_reward, correct_receipt = canary._grade_reward(
        task,
        task.answer,
        format_credit=0.1,
    )
    wrong_verdict, wrong_reward, wrong_receipt = canary._grade_reward(
        task,
        'FINAL_ANSWER: {"value":999}',
        format_credit=0.1,
    )
    invalid_verdict, invalid_reward, invalid_receipt = canary._grade_reward(
        task,
        "not an answer",
        format_credit=0.1,
    )

    assert correct_verdict["correct"] is True
    assert correct_reward == 1.0
    assert correct_receipt["correct"] is True
    assert wrong_verdict["correct"] is False
    assert wrong_reward == 0.1
    assert wrong_receipt["parsed"] is True
    assert invalid_verdict["correct"] is False
    assert invalid_reward == 0.0
    assert invalid_receipt["parsed"] is False


def test_observable_grade_ignores_tokens_after_the_authenticated_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = task_battery(["boolean"], [2], 1, seed=43)[0]
    sample = type("Sample", (), {"tokens": (1, 2, 3, 4)})()
    observable = {
        "response_text": task.answer,
        "optimization_token_count": 2,
        "full_token_count": 4,
        "termination": "contract_complete",
    }
    monkeypatch.setattr(
        canary,
        "observable_completion_from_adapter",
        lambda _adapter, tokens: observable if tuple(tokens) == sample.tokens else None,
    )

    observed, verdict, reward, receipt = canary._observable_grade_reward(
        task,
        sample,
        object(),
        format_credit=0.1,
    )

    assert observed is observable
    assert observed["optimization_token_count"] < observed["full_token_count"]
    assert verdict["correct"] is True
    assert reward == 1.0
    assert receipt["correct"] is True


def _rejected_sample_receipt() -> dict[str, object]:
    return {
        "episode_id": "episode-1",
        "seed": 17,
        "token_count": 4,
        "behavior_admitted": False,
        "max_abs_logprob_drift": 4.5,
        "mean_abs_logprob_drift": 0.2,
        "clipped_token_fraction": 0.5,
        "old_policy_approx_kl": 0.02,
        "sampling_config": {
            "max_abs_logprob_drift": 4.0,
            "max_mean_abs_logprob_drift": 0.5,
            "max_clipped_token_fraction": 0.25,
            "max_old_policy_approx_kl": 0.1,
        },
        "cached_params_unchanged": True,
        "cached_nonparametric_memory_status": "disabled_by_policy",
        "cached_runtime_integrity": {"ok": True},
        "cached_recurrence_adapter": {
            "active": True,
            "calls": 1,
            "adapted_positions": 4,
        },
        "episode_receipt": {"honest_flags": []},
    }


def test_sampling_rejection_diagnostics_name_every_failed_bound() -> None:
    diagnostic = canary._sampling_rejection_diagnostics(_rejected_sample_receipt())

    assert diagnostic["failed_gates"] == [
        "max_abs_logprob_drift",
        "clipped_token_fraction",
        "behavior_admitted",
    ]
    assert diagnostic["checks"]["recurrence_adapter_active"] is True


def test_sampling_failure_retains_receipts_and_compact_causes() -> None:
    receipt = _rejected_sample_receipt()
    error = canary.RecurrentCanarySamplingError(
        admitted=0,
        requested=2,
        rejected=[receipt, json.loads(json.dumps(receipt))],
    )

    assert error.admitted == 0
    assert error.requested == 2
    assert error.rejected == [receipt, receipt]
    assert "clipped_token_fraction" in str(error)
    assert len(error.diagnostics) == 2
