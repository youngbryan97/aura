from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from core.learning import candidate_cortex_kernel as kernel
from core.learning import candidate_cortex_training as training


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(tmp_path: Path) -> tuple[Path, str]:
    model = tmp_path / "Qwen3.8-27B-4bit"
    model.mkdir()
    descriptor: dict[str, Any] = {
        "schema": "aura.model_artifact_descriptor.v1",
        "canonical_path": str(model.resolve()),
        "repository_id": "mlx-community/Qwen3.8-27B-4bit",
        "revision": "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
        "artifact_profile": {"exists": True, "measured": True},
        "weight_identity": {"method": "sha256", "fingerprint": "a" * 64},
        "behavior_identity": {"bundle_sha256": "b" * 64},
    }
    descriptor["descriptor_sha256"] = training.document_sha256(descriptor)
    path = tmp_path / "descriptor.json"
    _write_json(path, descriptor)
    return path, descriptor["descriptor_sha256"]


def _dataset(tmp_path: Path, descriptor: Path, descriptor_sha: str) -> Path:
    records: list[kernel.SourceRecord] = []
    bindings: dict[str, dict[str, Any]] = {}
    for domain in sorted(kernel.CORE_DOMAINS):
        source = tmp_path / "sources" / f"{domain}.json"
        _write_json(source, {"domain": domain})
        bindings[domain] = kernel.file_binding(source)
        for index in range(2):
            records.append(
                kernel.SourceRecord(
                    domain=domain,
                    messages=(
                        ("user", f"{domain} question {index}"),
                        ("assistant", f"{domain} answer {index}"),
                    ),
                    binding_key=domain,
                    source_key=f"fixture/{domain}#records",
                    source_index=index,
                )
            )
    receipt = kernel.build_candidate_cortex_kernel(
        descriptor_path=descriptor,
        expected_descriptor_sha256=descriptor_sha,
        output_root=tmp_path / "compact",
        source_repo_root=Path(__file__).resolve().parents[1],
        valid_fraction=0.25,
        split_seed=19,
        source_bundle=kernel.SourceBundle(tuple(records), bindings, "injected"),
    )
    return Path(receipt["generation_root"]) / "candidate_cortex_kernel_receipt.json"


def _plan(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    descriptor, descriptor_sha = _descriptor(tmp_path)
    receipt = _dataset(tmp_path, descriptor, descriptor_sha)
    python_executable = kwargs.pop("python_executable", Path(sys.executable))
    return training.prepare_training_run(
        descriptor_path=descriptor,
        expected_descriptor_sha256=descriptor_sha,
        dataset_receipt_path=receipt,
        output_root=tmp_path / "runs",
        python_executable=python_executable,
        admission_command=(sys.executable, "admit.py"),
        verify_full_model=False,
        **kwargs,
    )


def _admission(stage: int, *, persona: float = 0.95, retention: float = 1.0) -> dict[str, Any]:
    return {
        "schema": training.ADMISSION_SCHEMA,
        "stage_index": stage,
        "model_free": True,
        "persona_score": persona,
        "retention_score": retention,
        "no_regression_score": 1.0,
        "regressions": 0,
        "checks": 64,
        "evidence_sha256": f"{stage + 1:064x}",
    }


def _observation(stage: int, loss: float, checkpoint: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "stage_index": stage,
        "validation_loss": loss,
        "checkpoint": checkpoint or {},
    }


def test_rejects_descriptor_identity_drift(tmp_path: Path) -> None:
    descriptor, descriptor_sha = _descriptor(tmp_path)
    receipt = _dataset(tmp_path, descriptor, descriptor_sha)
    with pytest.raises(training.CandidateCortexTrainingError, match="not_admitted"):
        training.prepare_training_run(
            descriptor_path=descriptor,
            expected_descriptor_sha256="f" * 64,
            dataset_receipt_path=receipt,
            output_root=tmp_path / "runs",
            python_executable=Path(sys.executable),
            admission_command=(sys.executable, "admit.py"),
            verify_full_model=False,
        )


def test_rejects_dataset_tamper_and_path_escape(tmp_path: Path) -> None:
    descriptor, descriptor_sha = _descriptor(tmp_path)
    receipt = _dataset(tmp_path, descriptor, descriptor_sha)
    (receipt.parent / "data" / "train.jsonl").write_text("tampered\n")
    with pytest.raises(
        training.CandidateCortexTrainingError,
        match="receipt_output_mismatch:train",
    ):
        training.validate_compact_kernel_receipt(
            receipt,
            expected_descriptor_sha256=descriptor_sha,
        )

    receipt = _dataset(tmp_path / "escape", descriptor, descriptor_sha)
    value = json.loads(receipt.read_text())
    value["outputs"]["train"]["path"] = "../train.jsonl"
    value.pop("receipt_sha256")
    value["receipt_sha256"] = training.document_sha256(value)
    _write_json(receipt, value)
    with pytest.raises(training.CandidateCortexTrainingError, match="path_invalid"):
        training.validate_compact_kernel_receipt(
            receipt,
            expected_descriptor_sha256=descriptor_sha,
        )


def test_checkpoint_requires_one_exact_cumulative_file(tmp_path: Path) -> None:
    root = tmp_path / "adapter"
    root.mkdir()
    with pytest.raises(training.CandidateCortexTrainingError, match="checkpoint_missing"):
        training.discover_exact_checkpoint(root, expected_cumulative_iterations=100)
    first = root / "0000100_adapters.safetensors"
    first.write_bytes(b"one")
    assert training.discover_exact_checkpoint(root, expected_cumulative_iterations=100)[
        "sha256"
    ] == _sha(first)
    (root / "100_adapters.safetensors").write_bytes(b"two")
    with pytest.raises(training.CandidateCortexTrainingError, match="checkpoint_ambiguous"):
        training.discover_exact_checkpoint(root, expected_cumulative_iterations=100)


def test_stage_progression_is_geometric_and_resume_is_exact(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    first = training.next_stage_plan(plan, observations=[], admissions=[])
    assert first["stage_iterations"] == 100
    assert first["cumulative_iterations"] == 100
    assert "--resume-adapter-file" not in first["command"]

    checkpoint = Path(plan["paths"]["adapter_root"]) / "0000100_adapters.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    binding = training.discover_exact_checkpoint(
        checkpoint.parent,
        expected_cumulative_iterations=100,
    )
    second = training.next_stage_plan(
        plan,
        observations=[_observation(0, 1.0, binding)],
        admissions=[_admission(0)],
    )
    assert second["stage_iterations"] == 200
    assert second["cumulative_iterations"] == 300
    index = second["command"].index("--resume-adapter-file")
    assert second["command"][index + 1] == str(checkpoint.resolve())


def test_convergence_requires_patience_and_admitted_metrics() -> None:
    policy = replace(training.StagePolicy(), min_stages=3, patience=2)
    observations = [_observation(0, 1.0), _observation(1, 0.999), _observation(2, 0.9985)]
    verdict = training.decide_after_stage(
        policy=policy,
        observations=observations,
        admissions=[_admission(index) for index in range(3)],
    )
    assert verdict == {
        "decision": "COMPLETE",
        "reason": "convergence_patience_pass",
        "stage": 2,
    }


def test_regression_rejects_without_weakening_thresholds() -> None:
    policy = training.StagePolicy()
    verdict = training.decide_after_stage(
        policy=policy,
        observations=[_observation(0, 1.0)],
        admissions=[_admission(0, retention=0.97)],
    )
    assert verdict["decision"] == "REJECT"
    assert verdict["reason"] == "admission_regression"
    loss_verdict = training.decide_after_stage(
        policy=policy,
        observations=[_observation(0, 1.0), _observation(1, 1.03)],
        admissions=[_admission(0), _admission(1)],
    )
    assert loss_verdict["reason"] == "validation_loss_regression"


def test_command_binds_every_requested_training_setting(tmp_path: Path) -> None:
    config = training.TrainingConfig(
        rank=16,
        scale=12.0,
        dropout=0.1,
        num_layers=48,
        targets=("self_attn.q_proj", "mlp.down_proj"),
        batch_size=2,
        gradient_accumulation_steps=8,
        max_seq_length=8192,
        learning_rate=2e-5,
        save_every=20,
        eval_every=10,
        report_every=2,
        val_batches=-1,
        seed=7,
    )
    plan = _plan(tmp_path, config=config)
    command = training.build_stage_command(plan, stage_index=0, resume_checkpoint=None)
    assert command[:4] == (
        str(Path(sys.executable).absolute()),
        "-m",
        "mlx_lm",
        "lora",
    )
    assert command[command.index("--model") + 1] == plan["model"]["canonical_path"]
    assert command[command.index("--data") + 1] == plan["dataset"]["data_root"]
    assert command[command.index("--num-layers") + 1] == "48"
    assert command[command.index("--batch-size") + 1] == "2"
    assert command[command.index("--max-seq-length") + 1] == "8192"
    assert command[command.index("--save-every") + 1] == "20"
    assert command[command.index("--steps-per-eval") + 1] == "10"
    assert command[command.index("--steps-per-report") + 1] == "2"
    assert command[command.index("--grad-accumulation-steps") + 1] == "8"
    assert "--mask-prompt" in command
    assert "--grad-checkpoint" in command
    assert "90000" not in command
    mlx_config = json.loads(Path(plan["paths"]["mlx_config"]).read_text())
    assert mlx_config == {
        "lora_parameters": {
            "rank": 16,
            "scale": 12.0,
            "dropout": 0.1,
            "keys": ["self_attn.q_proj", "mlp.down_proj"],
        }
    }


def test_venv_launcher_and_environment_are_preserved_and_reverified(
    tmp_path: Path,
) -> None:
    venv = tmp_path / "isolated"
    launcher = venv / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(Path(sys.executable))
    pyvenv = venv / "pyvenv.cfg"
    pyvenv.write_text("home = /frozen/python\n", encoding="ascii")

    plan_root = tmp_path / "plan"
    plan_root.mkdir()
    plan = _plan(plan_root, python_executable=launcher)
    assert plan["python"] == str(launcher)
    assert plan["python_binding"]["invocation_kind"] == "symlink"
    assert plan["python_binding"]["pyvenv"]["path"] == str(pyvenv)
    assert training.build_canary_command(plan)[0] == str(launcher)
    training.load_and_verify_plan(Path(plan["paths"]["run_root"]), verify_full_model=False)

    pyvenv.write_text("home = /mutated/python\n", encoding="ascii")
    with pytest.raises(
        training.CandidateCortexTrainingError,
        match="python_executable_binding_drift",
    ):
        training.load_and_verify_plan(
            Path(plan["paths"]["run_root"]),
            verify_full_model=False,
        )
def test_journal_tamper_and_adapter_identity_drift_fail_closed(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    journal = Path(plan["paths"]["journal"])
    key = b"k" * 32
    training.append_authenticated_event(
        journal,
        key=key,
        event_type="stage_started",
        payload={"trainer_pid": 10, "trainer_start_token": "token"},
    )
    assert len(training.read_authenticated_journal(journal, key=key)) == 1
    journal.write_bytes(journal.read_bytes().replace(b"token", b"taken"))
    with pytest.raises(training.CandidateCortexTrainingError, match="authentication"):
        training.read_authenticated_journal(journal, key=key)

    identity = Path(plan["paths"]["adapter_identity"])
    value = json.loads(identity.read_text())
    value["run_id"] = "wrong"
    _write_json(identity, value)
    with pytest.raises(training.CandidateCortexTrainingError, match="identity_mismatch"):
        training.load_and_verify_plan(Path(plan["paths"]["run_root"]), verify_full_model=False)


def test_non_finite_missing_eval_and_stale_process_are_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    policy = training.StagePolicy(**plan["stages"])
    checkpoint_path = Path(plan["paths"]["adapter_root"]) / "0000100_adapters.safetensors"
    checkpoint_path.write_bytes(b"checkpoint")
    checkpoint = training.discover_exact_checkpoint(
        checkpoint_path.parent,
        expected_cumulative_iterations=100,
    )
    identity_sha = training.document_sha256(
        json.loads(Path(plan["paths"]["adapter_identity"]).read_text())
    )
    base = {
        "schema": training.OBSERVATION_SCHEMA,
        "stage_index": 0,
        "cumulative_iterations": policy.cumulative_iterations(0),
        "validation_loss": 1.0,
        "eval_samples": 64,
        "checkpoint": checkpoint,
        "adapter_identity_sha256": identity_sha,
        "model_descriptor_sha256": plan["model"]["descriptor_sha256"],
        "dataset_receipt_sha256": plan["dataset"]["receipt_sha256"],
        "trainer_pid": 123,
        "trainer_start_token": "start-token",
        "trainer_exit_code": 0,
    }
    launched = {"trainer_pid": 123, "trainer_start_token": "start-token"}
    assert training.validate_stage_observation(
        base,
        plan=plan,
        expected_stage_index=0,
        launched_identity=launched,
    )["validation_loss"] == 1.0
    for field, value, reason in (
        ("validation_loss", float("nan"), "non_finite"),
        ("eval_samples", 0, "evidence_missing"),
        ("trainer_start_token", "reused-pid", "identity_stale"),
    ):
        broken = {**base, field: value}
        with pytest.raises(training.CandidateCortexTrainingError, match=reason):
            training.validate_stage_observation(
                broken,
                plan=plan,
                expected_stage_index=0,
                launched_identity=launched,
            )


def test_execution_requires_the_real_canary_before_adaptive_training(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    assert training.execution_admission(plan, execute=False) == {
        "status": "DRY_RUN",
        "execution_authorized": False,
    }
    assert training.execution_admission(plan, execute=True) == {
        "status": "CANARY_REQUIRED",
        "execution_authorized": False,
        "canary_launch_authorized": True,
    }


def _canary_log(config: training.TrainingConfig, policy: training.CanaryPolicy) -> bytes:
    lines = ["Loading pretrained model"]
    final_iteration = training.canary_micro_iterations(config, policy)
    validation_iterations = {1, final_iteration}
    interval = (
        config.gradient_accumulation_steps
        * policy.validation_interval_optimizer_steps
    )
    validation_iterations.update(range(interval, final_iteration, interval))
    for iteration in range(1, final_iteration + 1):
        if iteration in validation_iterations:
            loss = 2.0 if iteration == 1 else 1.8
            lines.append(f"Iter {iteration}: Val loss {loss:.3f}, Val took 1.250s")
        if (
            iteration % config.gradient_accumulation_steps == 0
            or iteration == final_iteration
        ):
            lines.append(
                f"Iter {iteration}: Train loss 1.750, Learning Rate 1.000e-05, "
                "It/sec 0.500, Tokens/sec 100.000, Trained Tokens 400, "
                "Peak mem 25.000 GB"
            )
    lines.append("Saved final weights to adapters.safetensors.")
    return ("\n".join(lines) + "\n").encode()


def test_canary_is_ten_optimizer_updates_with_post_update_validation(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    config = training.TrainingConfig(**plan["training"])
    policy = training.CanaryPolicy(**plan["canary"])
    command = training.build_canary_command(plan)
    assert "--mask-prompt" in command
    assert command[command.index("--iters") + 1] == "41"
    assert command[command.index("--save-every") + 1] == "40"
    assert command[command.index("--steps-per-report") + 1] == "4"
    parsed = training.parse_canary_training_log(
        _canary_log(config, policy),
        config=config,
        policy=policy,
    )
    assert parsed["optimizer_steps"] == 10
    assert [item["iteration"] for item in parsed["optimizer_update_reports"]] == list(
        range(4, 41, 4)
    )
    assert parsed["validation_reports"][-1]["iteration"] == 41


def test_canary_adjudication_binds_detached_resource_and_checkpoint_evidence(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    config = training.TrainingConfig(**plan["training"])
    policy = training.CanaryPolicy(**plan["canary"])
    adapter_root = Path(plan["paths"]["canary_adapter_root"])
    checkpoint_iteration = training.canary_checkpoint_iteration(config, policy)
    (adapter_root / f"{checkpoint_iteration:07d}_adapters.safetensors").write_bytes(
        b"checkpoint"
    )
    (adapter_root / "adapters.safetensors").write_bytes(b"final")
    log_path = Path(plan["paths"]["canary_execution_root"]) / "detached.log"
    log_path.write_bytes(_canary_log(config, policy))
    target_command = [str(Path(sys.executable).resolve()), "canary.py", "--run-root", plan["paths"]["run_root"]]
    receipt_body = {
        "command": target_command,
        "status": "passed",
        "passed": True,
        "returncode": 0,
        "restart_count": 0,
        "containment_verified": True,
        "process_group_empty": True,
        "lineage_empty": True,
        "duration_s": 120.0,
        "child_pid": 123,
        "child_start_token": "123:456",
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": training.document_sha256(receipt_body),
    }
    metrics = {
        "schema": training.CANARY_HOST_METRICS_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "model_descriptor_sha256": plan["model"]["descriptor_sha256"],
        "dataset_receipt_sha256": plan["dataset"]["receipt_sha256"],
        "training_command_sha256": training.document_sha256(
            list(training.build_canary_command(plan))
        ),
        "target_pid": 123,
        "started_at_unix": 100.0,
        "finished_at_unix": 220.0,
        "duration_seconds": 120.0,
        "sample_count": 240,
        "min_available_bytes": 10 * 1024**3,
        "max_used_percent": 84.0,
        "max_process_rss_bytes": 30 * 1024**3,
    }
    metrics_path = Path(plan["paths"]["canary_host_metrics"])
    _write_json(metrics_path, metrics)
    evidence = training.adjudicate_canary(
        plan,
        detached_receipt=receipt,
        expected_target_command=target_command,
        detached_log_path=log_path,
        host_metrics_path=metrics_path,
        journal_key=b"j" * 64,
        verify_full_model=False,
    )
    assert evidence["admission"]["status"] == "PASS"
    assert evidence["admission"]["optimizer_steps"] == 10
    assert (
        training.execution_admission(plan, execute=True)["status"]
        == "CANARY_AUTHENTICATION_REQUIRED"
    )
    events = training.read_authenticated_journal(
        Path(plan["paths"]["journal"]), key=b"j" * 64
    )
    assert len(events) == 2
    assert (
        training.execution_admission(
            plan,
            execute=True,
            authenticated_events=events,
        )["status"]
        == "CANARY_PASSED"
    )


def test_run_root_is_private_and_content_addressed(tmp_path: Path) -> None:
    first = _plan(tmp_path)
    run_root = Path(first["paths"]["run_root"])
    assert run_root.name == first["run_id"]
    assert os.stat(run_root).st_mode & 0o077 == 0
    descriptor = Path(first["model"]["descriptor_path"])
    receipt = Path(first["dataset"]["receipt_path"])
    second = training.prepare_training_run(
        descriptor_path=descriptor,
        expected_descriptor_sha256=first["model"]["descriptor_sha256"],
        dataset_receipt_path=receipt,
        output_root=tmp_path / "runs",
        python_executable=Path(sys.executable),
        admission_command=(sys.executable, "admit.py"),
        verify_full_model=False,
    )
    assert second["run_id"] == first["run_id"]
