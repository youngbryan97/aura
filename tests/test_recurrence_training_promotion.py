from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_launch_bundle import (
    inventory_root_sha256,
)
from tools import verify_recurrence_training_promotion as promotion


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _binding(path: Path, payload: bytes | None = None) -> dict[str, object]:
    raw = path.read_bytes() if payload is None else payload
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _safetensors(name: str, values: bytes = b"\x00\x00\x00\x00") -> bytes:
    header = json.dumps(
        {
            name: {
                "dtype": "F32",
                "shape": [len(values) // 4],
                "data_offsets": [0, len(values)],
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return len(header).to_bytes(8, "little") + header + values


class _FixtureTokenizer:
    eos_token_id = None

    def apply_chat_template(self, *_args, **_kwargs) -> list[int]:
        return [1, 2]

    def encode(self, *_args, **_kwargs) -> list[int]:
        return [3]


def _promotion_fixture(tmp_path: Path) -> dict:
    worktree = tmp_path / "training-worktree"
    tools = worktree / "tools"
    tools.mkdir(parents=True)
    venv = tmp_path / ".venv"
    launcher = venv / "bin/python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(Path(sys.executable).resolve())
    (venv / "pyvenv.cfg").write_text("home = /test/python\n", encoding="ascii")
    wrapper = tools / "run_recurrence_training_envelope.py"
    trainer = tools / "recurrence_native_train_v2.py"
    wrapper.write_bytes(b"# resource envelope\n")
    trainer.write_bytes(b"# recurrence trainer\n")
    curriculum = promotion._CURRICULUM_PATH.read_bytes()
    worktree_curriculum = worktree / "core/learning/recurrence_curriculum.py"
    worktree_curriculum.parent.mkdir(parents=True)
    worktree_curriculum.write_bytes(curriculum)
    model = tmp_path / "model"
    source_adapter = tmp_path / "source-adapter"
    frozen_adapter = tmp_path / "frozen-adapter"
    model.mkdir()
    source_adapter.mkdir()
    frozen_adapter.mkdir()

    resource_path = source_adapter / "launch_resource_envelope.json"
    resource = {
        "schema": promotion.RESOURCE_SCHEMA,
        "memory_limit_bytes": 40 * promotion._GIB,
        "cache_limit_bytes": 2 * promotion._GIB,
        "wired_limit_bytes": 48 * promotion._GIB,
        "cache_cleared_before_model_load": True,
        "device": {
            "architecture": "arm64",
            "memory_size": 64 * promotion._GIB,
            "max_recommended_working_set_size": 52 * promotion._GIB,
        },
        "mlx_version": "test",
        "wrapper_sha256": _binding(wrapper)["sha256"],
        "trainer_sha256": _binding(trainer)["sha256"],
    }
    _write_json(resource_path, resource)

    adapter_bytes = _safetensors("adapter")
    adapter_sha = hashlib.sha256(adapter_bytes).hexdigest()
    base_checkpoint = {
        "fingerprint": "a" * 64,
        "method": "sha256",
        "files": 4,
    }
    model_behavior = {
        "bundle_sha256": "f" * 64,
        "file_count": 0,
        "files": [],
    }
    training_runtime = {
        "identity_sha256": "1" * 64,
        "python": "test",
        "platform_system": "Darwin",
        "platform_release": "test",
        "platform_machine": "arm64",
        "dependencies": {},
    }
    execution_spec = {
        "schema": "aura.rlc_execution_spec.v1",
        "n_slots": 16,
        "branch_roles": ["constructive_solution", "counterexample_search"],
        "exchange_interval": 1,
        "alpha": 0.5,
        "alpha_schedule": "constant",
        "recurrent_steps": 4,
    }
    personality = {
        "present": False,
        "bundle_sha256": "",
        "file_count": 0,
        "files": [],
    }
    gradient_execution = {
        "schema": "aura.recurrence_streamed_depth_gradient.v1",
        "mode": "depth_serial_exact_sum",
        "concurrent_depth_graphs": 1,
        "optimizer_updates_per_sample": 1,
        "finite_loss_and_gradient_required_before_update": True,
    }
    config = {
        "schema": promotion.TRAINING_CONFIG_SCHEMA,
        "model_path": str(model),
        "base_checkpoint": base_checkpoint,
        "model_behavior_bundle": model_behavior,
        "personality_adapter_path": "",
        "personality_adapter": personality,
        "training_runtime": training_runtime,
        "execution_spec": execution_spec,
        "curriculum_depths": [1, 2, 4],
        "monotonicity_weight": 0.5,
        "lora": {
            "rank": 8,
            "targets": ["o_proj", "v_proj"],
            "wrapped_projections": ["model.layers.1.self_attn.o_proj"],
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
        },
        "gradient_execution": gradient_execution,
        "train_seed": 1777,
        "max_steps": 8,
    }
    tasks = promotion.task_battery(["khop", "boolean"], [2, 4], 2, seed=1777)
    curriculum_binding = {
        "path": "core/learning/recurrence_curriculum.py",
        "sha256": hashlib.sha256(curriculum).hexdigest(),
        "size_bytes": len(curriculum),
    }
    dataset = {
        "schema": promotion.TRAINING_DATASET_SCHEMA,
        "generator": curriculum_binding,
        "train_seed": 1777,
        "families": ["khop", "boolean"],
        "task_depths": [2, 4],
        "per_cell": 2,
        "examples": [
            {
                "family": task.family,
                "depth": task.depth,
                "seed": task.seed,
                "prompt": task.prompt,
                "answer": task.answer,
                "prompt_tokens": [1, 2],
                "answer_tokens": [3],
            }
            for task in tasks
        ],
    }
    loss_trail = [{"step": step, "mean_loss": round(1.0 / step, 6)} for step in range(1, 9)]
    receipt = {
        "schema": promotion.TRAINING_RECEIPT_SCHEMA,
        "complete": True,
        "halt_reason": "max_steps",
        "steps": 8,
        "final_checkpoint": "step-00000008-terminal",
        "epoch": 0,
        "cursor": 8,
        "elapsed_training_s": 12.5,
        "invocation_count": 1,
        "gradient_execution": gradient_execution,
        "loss_trail": loss_trail,
    }
    files = {
        "receipt": (source_adapter / "receipt.json", receipt),
        "config": (source_adapter / "training_config.json", config),
        "dataset": (source_adapter / "dataset_manifest.json", dataset),
        "execution_spec": (source_adapter / "execution_spec.json", execution_spec),
    }
    for path, document in files.values():
        _write_json(path, document)
    curriculum_snapshot = source_adapter / "source_snapshots/task_generator.py"
    curriculum_snapshot.parent.mkdir(parents=True)
    curriculum_snapshot.write_bytes(curriculum)
    manifest = {
        "schema": "aura.recurrence_adapter_manifest.v2",
        "adapter_id": "resident-test",
        "base_checkpoint": base_checkpoint,
        "model_behavior_bundle": model_behavior,
        "personality_adapter": personality,
        "training_runtime": training_runtime,
        "adapter": {
            "path": "adapters.safetensors",
            "sha256": adapter_sha,
            "size_bytes": len(adapter_bytes),
        },
        "training_receipt": {
            **_binding(files["receipt"][0]),
            "path": "receipt.json",
        },
        "training_config": {
            **_binding(files["config"][0]),
            "path": "training_config.json",
        },
        "dataset_manifest": {
            **_binding(files["dataset"][0]),
            "path": "dataset_manifest.json",
        },
        "execution_spec": {
            **_binding(files["execution_spec"][0]),
            "path": "execution_spec.json",
        },
        "sources": {
            "task_generator": {
                "origin_path": curriculum_binding["path"],
                "snapshot_path": "source_snapshots/task_generator.py",
                "sha256": curriculum_binding["sha256"],
                "size_bytes": curriculum_binding["size_bytes"],
            }
        },
    }
    manifest.update(
        {
            "config_sha256": manifest["training_config"]["sha256"],
            "dataset_sha256": manifest["dataset_manifest"]["sha256"],
            "execution_spec_sha256": manifest["execution_spec"]["sha256"],
        }
    )
    manifest_path = source_adapter / "recurrence_adapter_manifest.json"
    _write_json(manifest_path, manifest)
    completion = {
        "schema": promotion.TRAINING_COMPLETION_SCHEMA,
        "complete": True,
        "halt_reason": "max_steps",
        "step": 8,
        "adapter_sha256": adapter_sha,
        "receipt_sha256": manifest["training_receipt"]["sha256"],
        "manifest_sha256": _binding(manifest_path)["sha256"],
    }
    completion_path = source_adapter / "training_completion.json"
    _write_json(completion_path, completion)

    checkpoint = source_adapter / "checkpoints/step-00000008-terminal"
    checkpoint.mkdir(parents=True)
    checkpoint_adapter = checkpoint / "adapter.safetensors"
    checkpoint_optimizer = checkpoint / "optimizer.safetensors"
    checkpoint_adapter.write_bytes(adapter_bytes)
    checkpoint_optimizer.write_bytes(_safetensors("optimizer", b"\x01\x00\x00\x00"))
    order = sorted(
        range(len(dataset["examples"])),
        key=lambda index: hashlib.sha256(f"1777:0:{index}".encode("ascii")).digest(),
    )
    checkpoint_complete = {
        "schema": promotion.TRAINING_CHECKPOINT_SCHEMA,
        "checkpoint_id": checkpoint.name,
        "step": 8,
        "epoch": 0,
        "cursor": 8,
        "order": order,
        "config_sha256": manifest["config_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "execution_spec_sha256": manifest["execution_spec_sha256"],
        "elapsed_training_s": 12.0,
        "invocation_count": 1,
        "loss_trail": loss_trail,
        "pending_window_losses": [],
        "pending_window_cosines": [],
        "holdout_trail": [],
        "holdout_eval_count": 0,
        "sampler": "sha256_stateless_epoch_permutation.v1",
        "stochastic_state": "none_all_keys_explicit",
        "adapter": {
            **_binding(checkpoint_adapter),
            "path": checkpoint_adapter.name,
        },
        "optimizer": {
            **_binding(checkpoint_optimizer),
            "path": checkpoint_optimizer.name,
        },
    }
    checkpoint_complete_path = checkpoint / "complete.json"
    _write_json(checkpoint_complete_path, checkpoint_complete)
    latest = {
        "schema": promotion.TRAINING_POINTER_SCHEMA,
        "checkpoint": f"checkpoints/{checkpoint.name}",
        "complete_sha256": _binding(checkpoint_complete_path)["sha256"],
    }
    latest_path = source_adapter / "latest.json"
    _write_json(latest_path, latest)
    checkpoint_evidence = promotion._latest_checkpoint_evidence(source_adapter, latest)

    wrapper_options = [
        "--memory-limit-gb",
        "40",
        "--cache-limit-gb",
        "2",
        "--wired-limit-gb",
        "48",
        "--envelope-out",
        str(resource_path),
        "--trainer",
        str(trainer),
    ]
    trainer_options = [
        "--model",
        str(model),
        "--out-dir",
        str(source_adapter),
        "--adapter-id",
        "resident-test",
        "--personality-adapter",
        "none",
        "--train-seed",
        "1777",
        "--families",
        "khop,boolean",
        "--task-depths",
        "2,4",
        "--per-cell",
        "2",
        "--curriculum-depths",
        "1,2,4",
        "--n-slots",
        "16",
        "--branch-roles",
        "constructive_solution,counterexample_search",
        "--exchange-interval",
        "1",
        "--alpha",
        "0.5",
        "--alpha-schedule",
        "constant",
        "--lora-rank",
        "8",
        "--lora-targets",
        "o_proj,v_proj",
        "--learning-rate",
        "0.0001",
        "--monotonicity-weight",
        "0.5",
        "--max-minutes",
        "60",
        "--max-steps",
        "8",
        "--checkpoint-every",
        "2",
        "--log-every",
        "1",
    ]
    command = [str(launcher), str(wrapper), *wrapper_options, "--", *trainer_options]
    launcher_binding = promotion.detached._launcher_binding(launcher)
    plan = {
        "schema": "aura.detached_step.plan.v2",
        "name": "promotion-fixture",
        "plan_sha256": "b" * 64,
        "command_sha256": "c" * 64,
        "command": command,
        "executable_sha256": launcher_binding["resolved_sha256"],
        "executable_binding": launcher_binding,
        "cwd": str(worktree),
        "execution_environment_sha256": "2" * 64,
        "target_execution_manifest": {"manifest_sha256": "3" * 64},
        "timeout_s": 7200.0,
        "fork_policy": "kernel_denied",
        "restart_policy": "never",
        "resume_contract": "none",
        "broker_policy": [],
        "broker_policy_sha256": "4" * 64,
    }
    detached_receipt = {
        "schema": "aura.detached_step.receipt.v1",
        "receipt_sha256": "d" * 64,
        "plan_sha256": plan["plan_sha256"],
        "command_sha256": plan["command_sha256"],
        "command": command,
        "returncode": 0,
        "passed": True,
        "containment_verified": True,
        "lineage_empty": True,
        "process_group_empty": True,
        "timed_out": False,
        "supervisor_error": None,
        "restart_count": 0,
        "fork_policy": "kernel_denied",
        "started_at": 10.0,
        "finished_at": 30.0,
        "duration_s": 20.0,
    }
    source_inventory = [
        {
            "path": "recurrence_adapter_manifest.json",
            "sha256": _binding(manifest_path)["sha256"],
            "size_bytes": _binding(manifest_path)["size_bytes"],
        }
    ]
    freeze_certificate = {
        "adapter_id": "resident-test",
        "artifacts": source_inventory,
        "content_root_sha256": inventory_root_sha256(source_inventory),
        "identity_receipt": {
            "adapter_sha256": adapter_sha,
            "training_receipt_sha256": manifest["training_receipt"]["sha256"],
            "training_config_sha256": manifest["training_config"]["sha256"],
            "training_completion_sha256": _binding(completion_path)["sha256"],
        },
        "model_identity": {"fingerprint": "a" * 64},
    }
    artifact_bindings = {
        "completion": _binding(completion_path),
        "config": _binding(files["config"][0]),
        "dataset": _binding(files["dataset"][0]),
        "execution_spec": _binding(files["execution_spec"][0]),
        "latest": _binding(latest_path),
        "manifest": _binding(manifest_path),
        "receipt": _binding(files["receipt"][0]),
        "resource": _binding(resource_path),
        "trainer": _binding(trainer),
        "wrapper": _binding(wrapper),
    }
    contract_material = {
        "accepted_artifacts": {
            "training_config": {
                "sha256": artifact_bindings["config"]["sha256"],
                "size_bytes": artifact_bindings["config"]["size_bytes"],
            },
            "dataset_manifest": {
                "sha256": artifact_bindings["dataset"]["sha256"],
                "size_bytes": artifact_bindings["dataset"]["size_bytes"],
            },
            "execution_spec": {
                "sha256": artifact_bindings["execution_spec"]["sha256"],
                "size_bytes": artifact_bindings["execution_spec"]["size_bytes"],
                "semantic_sha256": promotion._sha256(execution_spec),
            },
            "resource_envelope": {
                "sha256": artifact_bindings["resource"]["sha256"],
                "size_bytes": artifact_bindings["resource"]["size_bytes"],
            },
        },
        "accepted_plan": {
            "broker_policy_sha256": plan["broker_policy_sha256"],
            "command_sha256": plan["command_sha256"],
            "executable_sha256": plan["executable_sha256"],
            "execution_environment_sha256": plan["execution_environment_sha256"],
            "fork_policy": plan["fork_policy"],
            "name": plan["name"],
            "plan_sha256": plan["plan_sha256"],
            "restart_policy": plan["restart_policy"],
            "resume_contract": plan["resume_contract"],
            "target_execution_manifest_sha256": plan["target_execution_manifest"][
                "manifest_sha256"
            ],
            "timeout_s": plan["timeout_s"],
        },
        "adapter_id": "resident-test",
        "claim_scope": "internal_mechanics_acceptance_only",
        "contract_id": "promotion-fixture",
        "evidence_limitations": [
            "training_started_before_contract_commit",
            "detached_training_terminal_does_not_bind_output_root",
            "external_attestation_not_present",
            "reasoning_and_frontier_gain_not_measured",
        ],
        "external_attestation_present": False,
        "gradient_execution": gradient_execution,
        "model_identity": {
            "base_checkpoint_fingerprint": base_checkpoint["fingerprint"],
            "base_checkpoint_files": base_checkpoint["files"],
            "model_behavior_bundle_sha256": model_behavior["bundle_sha256"],
            "training_runtime_identity_sha256": training_runtime["identity_sha256"],
        },
        "optimizer": config["optimizer"],
        "producer_sources": {
            "task_generator_sha256": curriculum_binding["sha256"],
            "trainer_sha256": artifact_bindings["trainer"]["sha256"],
            "wrapper_sha256": artifact_bindings["wrapper"]["sha256"],
        },
        "required_next_gates": [
            "terminal_artifact_generation_validation",
            "immutable_adapter_freeze",
            "resident_32b_frozen_adapter_mechanics_smoke",
            "fresh_hidden_task_pilot",
            "powered_external_frontier_campaign",
        ],
        "resource_envelope": {
            "memory_limit_bytes": resource["memory_limit_bytes"],
            "cache_limit_bytes": resource["cache_limit_bytes"],
            "wired_limit_bytes": resource["wired_limit_bytes"],
            "minimum_device_memory_bytes": resource["device"]["memory_size"],
            "minimum_recommended_working_set_bytes": resource["device"][
                "max_recommended_working_set_size"
            ],
        },
        "schema": "aura.recurrence_training_acceptance_contract.v1",
        "workload": {
            "alpha": 0.5,
            "alpha_schedule": "constant",
            "branch_roles": ["constructive_solution", "counterexample_search"],
            "checkpoint_every": 2,
            "curriculum_depths": [1, 2, 4],
            "exchange_interval": 1,
            "families": ["khop", "boolean"],
            "log_every": 1,
            "lora_rank": 8,
            "lora_targets": ["o_proj", "v_proj"],
            "max_minutes": 60.0,
            "max_steps": 8,
            "monotonicity_weight": 0.5,
            "n_slots": 16,
            "per_cell": 2,
            "personality_adapter": "none",
            "task_depths": [2, 4],
            "train_seed": 1777,
        },
    }
    contract = {
        **contract_material,
        "contract_sha256": promotion._sha256(contract_material),
    }
    contract_payload = canonical_json_bytes(contract) + b"\n"
    return {
        "detached_plan": plan,
        "detached_receipt": detached_receipt,
        "detached_evidence": {
            "attempt_event_count": 4,
            "attempt_journal_head_sha256": "e" * 64,
            "supervisor_attempt": 1,
        },
        "acceptance_contract": contract,
        "acceptance_contract_binding": {
            "path": "memory://promotion-fixture",
            "sha256": hashlib.sha256(contract_payload).hexdigest(),
            "size_bytes": len(contract_payload),
        },
        "resource_envelope": resource,
        "resource_path": resource_path,
        "training_manifest": manifest,
        "training_receipt": receipt,
        "training_config": config,
        "training_dataset": dataset,
        "execution_spec": execution_spec,
        "training_completion": completion,
        "latest": latest,
        "checkpoint_evidence": checkpoint_evidence,
        "artifact_bindings": artifact_bindings,
        "freeze_certificate": freeze_certificate,
        "source_inventory": source_inventory,
        "source_adapter": source_adapter,
        "frozen_adapter": frozen_adapter,
        "model": model,
        "adapter_id": "resident-test",
        "tokenizer": _FixtureTokenizer(),
    }


def _rebind_document(inputs: dict, document_key: str, binding_key: str) -> None:
    payload = canonical_json_bytes(inputs[document_key]) + b"\n"
    inputs["artifact_bindings"][binding_key].update(
        {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    )


def _tamper_dataset(inputs: dict) -> None:
    inputs["training_dataset"]["families"] = ["khop"]
    _rebind_document(inputs, "training_dataset", "dataset")


def _tamper_resource_limit(inputs: dict) -> None:
    inputs["resource_envelope"]["wired_limit_bytes"] = 47 * promotion._GIB
    _rebind_document(inputs, "resource_envelope", "resource")


def _tamper_personality_path(inputs: dict) -> None:
    inputs["training_config"]["personality_adapter_path"] = "/unexpected"
    _rebind_document(inputs, "training_config", "config")


def _wait_for_receipt(run_dir: Path, timeout_s: float = 10.0) -> dict:
    receipt_path = run_dir / promotion.detached.RECEIPT_FILE
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if receipt_path.is_file():
            try:
                return json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {receipt_path}")


def _wait_until_terminal(run_dir: Path, timeout_s: float = 20.0) -> dict:
    """Wait for what the reader is about to require, not just for the receipt.

    `_authoritative_detached_evidence` demands a terminal journal, a dead
    child AND a supervisor that has exited. The receipt lands before the
    supervisor finishes exiting, so a fixture that waits only for the receipt
    hands the reader a run that is still winding down — which it refuses, as
    `detached_training_journal_not_terminal`. Passes alone on an idle machine
    and fails inside a loaded chunk, which is what that race looks like from
    outside.
    """
    deadline = time.time() + timeout_s
    status: dict = {}
    while time.time() < deadline:
        status = promotion.detached._status(run_dir)
        if (
            status.get("terminal") is True
            and status.get("supervisor_alive") is False
            and status.get("child_state") == "dead"
        ):
            return status
        time.sleep(0.05)
    raise AssertionError(f"detached run never went terminal: {status}")


def _launch_detached_fixture(run_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(Path(promotion.detached.__file__).resolve()),
            "launch",
            "--run-dir",
            str(run_dir),
            "--name",
            "promotion-journal-test",
            "--cwd",
            str(run_dir.parent),
            "--timeout",
            "5",
            "--resume-contract",
            "none",
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    receipt = _wait_for_receipt(run_dir)
    assert receipt["passed"] is True
    _wait_until_terminal(run_dir)


def test_terminal_training_promotion_binds_complete_execution(tmp_path: Path) -> None:
    evidence = promotion._validate_documents(**_promotion_fixture(tmp_path))

    assert evidence["steps"] == 8
    assert evidence["latest_checkpoint"] == ("checkpoints/step-00000008-terminal")
    assert evidence["training_manifest_sha256"]
    assert evidence["training_wrapper_sha256"]


def test_public_verify_composes_terminal_promotion_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _promotion_fixture(tmp_path)
    detached_run = tmp_path / "detached-run"
    detached_run.mkdir()
    contract_path = tmp_path / "acceptance-contract.json"
    _write_json(contract_path, inputs["acceptance_contract"])
    validators = {
        "campaign_runner_sha256": promotion._file_binding(
            promotion.preparation.RUNNER_PATH,
            role="campaign_runner",
        )["sha256"],
        "freeze_contract_sha256": promotion._file_binding(
            promotion.preparation.FREEZE_PATH,
            role="freeze_contract",
        )["sha256"],
        "identity_validator_sha256": promotion._file_binding(
            promotion.preparation.IDENTITY_PATH,
            role="identity_validator",
        )["sha256"],
    }
    model_identity = {
        "fingerprint": "a" * 64,
        "files": 4,
        "model_behavior_bundle": {"bundle_sha256": "f" * 64},
        "runtime_bundle": {"bundle_sha256": "7" * 64},
        "runtime_environment": {"identity_sha256": "1" * 64},
        "personality_adapter": {"bundle_sha256": ""},
        "effective_stack_sha256": "8" * 64,
    }
    freeze_certificate = {
        **inputs["freeze_certificate"],
        "model_identity": promotion.preparation._selected_model_identity(model_identity),
        "validator_identity": validators,
        "certificate_sha256": "9" * 64,
    }
    monkeypatch.setattr(
        promotion,
        "_authoritative_detached_evidence",
        lambda _run: (
            inputs["detached_plan"],
            inputs["detached_receipt"],
            inputs["detached_evidence"],
        ),
    )
    monkeypatch.setattr(
        promotion,
        "_load_bound_tokenizer",
        lambda _model: inputs["tokenizer"],
    )
    monkeypatch.setattr(
        promotion,
        "adapter_artifact_inventory",
        lambda _adapter, reject_unplanned=False: inputs["source_inventory"],
    )
    monkeypatch.setattr(
        promotion,
        "verify_adapter_freeze",
        lambda _adapter: freeze_certificate,
    )
    monkeypatch.setattr(
        promotion.campaign_runner,
        "_identity_material",
        lambda _args: (
            model_identity,
            {"identity_receipt": freeze_certificate["identity_receipt"]},
        ),
    )
    monkeypatch.setattr(
        promotion.campaign_runner,
        "model_behavior_bundle_identity",
        lambda _model: model_identity["model_behavior_bundle"],
    )

    result = promotion.verify(
        SimpleNamespace(
            detached_run=str(detached_run),
            source_adapter=str(inputs["source_adapter"]),
            frozen_adapter=str(inputs["frozen_adapter"]),
            model=str(inputs["model"]),
            acceptance_contract=str(contract_path),
            resource_envelope=str(inputs["resource_path"]),
            adapter_id=inputs["adapter_id"],
        )
    )

    assert result["training_complete"] is True
    assert result["immutable_freeze_verified"] is True
    assert result["ready_for_mechanics_smoke"] is True
    assert result["pilot_eligible"] is False
    assert result["reasoning_gain_proven"] is False
    assert result["frontier_gain_proven"] is False
    assert result["training"]["training_dataset"]["example_count"] == 8
    assert result["training"]["tokenizer_model_behavior_bundle_sha256"] == "f" * 64
    assert result["promotion_sha256"] == promotion._sha256(
        {key: value for key, value in result.items() if key != "promotion_sha256"}
    )


def test_tokenizer_generation_stability_rejects_change() -> None:
    before = {"bundle_sha256": "f" * 64, "files": []}
    after = {**before, "bundle_sha256": "0" * 64}

    with pytest.raises(
        promotion.TrainingPromotionError,
        match="training_tokenizer_generation_changed",
    ):
        promotion._stable_tokenizer_behavior_sha256(before, after)


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (
            lambda item: item["detached_receipt"].update({"passed": False}),
            "detached_training_terminal_invalid",
        ),
        (
            _tamper_dataset,
            "training_acceptance_artifact_mismatch",
        ),
        (
            lambda item: item["detached_plan"]["command"].__setitem__(
                item["detached_plan"]["command"].index("--out-dir") + 1,
                str(item["frozen_adapter"]),
            ),
            "detached_training_identity_mismatch",
        ),
        (
            _tamper_resource_limit,
            "training_acceptance_artifact_mismatch",
        ),
        (
            lambda item: item["checkpoint_evidence"]["artifacts"]["adapter"].update(
                {"sha256": "e" * 64}
            ),
            "training_final_checkpoint_mismatch",
        ),
        (
            lambda item: item["freeze_certificate"].update({"artifacts": []}),
            "adapter_freeze_training_mismatch",
        ),
        (
            _tamper_personality_path,
            "training_acceptance_artifact_mismatch",
        ),
    ],
)
def test_terminal_training_promotion_rejects_broken_links(
    tmp_path: Path, mutator, error: str
) -> None:
    inputs = _promotion_fixture(tmp_path)
    mutator(inputs)

    with pytest.raises(promotion.TrainingPromotionError, match=error):
        promotion._validate_documents(**inputs)


def test_terminal_training_promotion_rejects_cross_read_generation_change(
    tmp_path: Path,
) -> None:
    inputs = _promotion_fixture(tmp_path)
    inputs["training_completion"]["step"] = 7

    with pytest.raises(
        promotion.TrainingPromotionError,
        match="training_completion_generation_changed",
    ):
        promotion._validate_documents(**inputs)


def test_terminal_training_promotion_rejects_noop_launcher(tmp_path: Path) -> None:
    inputs = _promotion_fixture(tmp_path)
    launcher = Path("/usr/bin/true")
    command = list(inputs["detached_plan"]["command"])
    command[0] = str(launcher)
    launcher_binding = promotion.detached._launcher_binding(launcher)
    inputs["detached_plan"].update(
        command=command,
        executable_binding=launcher_binding,
        executable_sha256=launcher_binding["resolved_sha256"],
    )
    inputs["detached_receipt"].update(command=command)

    with pytest.raises(
        promotion.TrainingPromotionError,
        match="training_launcher_not_pinned_python",
    ):
        promotion._validate_documents(**inputs)


def test_terminal_training_promotion_rejects_underpowered_declared_workload(
    tmp_path: Path,
) -> None:
    inputs = _promotion_fixture(tmp_path)
    command = inputs["detached_plan"]["command"]
    command[command.index("--max-steps") + 1] = "1"
    inputs["detached_receipt"]["command"] = list(command)

    with pytest.raises(
        promotion.TrainingPromotionError,
        match="training_acceptance_workload_mismatch",
    ):
        promotion._validate_documents(**inputs)


def test_training_dataset_replays_exact_bound_tokenizer(tmp_path: Path) -> None:
    inputs = _promotion_fixture(tmp_path)
    inputs["training_dataset"]["examples"][0]["prompt_tokens"] = [9]

    with pytest.raises(
        promotion.TrainingPromotionError,
        match="training_dataset_example_0_tokenization_mismatch",
    ):
        promotion._validate_training_dataset(
            inputs["training_dataset"],
            inputs["training_manifest"],
            source_adapter=inputs["source_adapter"],
            training_worktree=Path(inputs["detached_plan"]["cwd"]),
            families=["khop", "boolean"],
            task_depths=[2, 4],
            per_cell=2,
            train_seed=1777,
            tokenizer=_FixtureTokenizer(),
        )


def test_bound_tokenizer_uses_mlx_loader_and_model_eos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_json(model / "config.json", {"eos_token_id": [151645, 151643]})
    expected = object()
    observed: dict[str, object] = {}

    def fake_load_tokenizer(model_path: Path, *, eos_token_ids: object) -> object:
        observed.update(model_path=model_path, eos_token_ids=eos_token_ids)
        return expected

    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", fake_load_tokenizer)

    assert promotion._load_bound_tokenizer(model) is expected
    assert observed == {
        "model_path": model,
        "eos_token_ids": [151645, 151643],
    }


@pytest.mark.parametrize("eos_token_id", [None, True, -1, [], [151645, "151643"]])
def test_bound_tokenizer_rejects_invalid_model_eos(tmp_path: Path, eos_token_id: object) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_json(model / "config.json", {"eos_token_id": eos_token_id})

    with pytest.raises(
        promotion.TrainingPromotionError,
        match="training_tokenizer_eos_contract_invalid",
    ):
        promotion._load_bound_tokenizer(model)


@pytest.mark.parametrize(
    "loss_trail",
    [
        [{"step": 8, "mean_loss": 1.0}],
        [{"step": step, "mean_loss": 1.0} for step in [1, 2, 4, 3, 5, 6, 7, 8]],
        [
            *({"step": step, "mean_loss": 1.0} for step in range(1, 8)),
            {"step": 8, "mean_loss": float("nan")},
        ],
    ],
)
def test_loss_trail_requires_exact_producer_schedule(loss_trail: list[dict]) -> None:
    assert (
        promotion._validate_loss_trail(
            loss_trail,
            expected_steps=8,
            log_every=1,
        )
        is False
    )


def test_terminal_training_promotion_binds_training_to_detached_duration(
    tmp_path: Path,
) -> None:
    inputs = _promotion_fixture(tmp_path)
    inputs["detached_receipt"].update(finished_at=20.0, duration_s=10.0)

    with pytest.raises(
        promotion.TrainingPromotionError,
        match="training_final_checkpoint_mismatch",
    ):
        promotion._validate_documents(**inputs)


@pytest.mark.skipif(sys.platform != "darwin", reason="strong containment requires macOS")
def test_authoritative_detached_evidence_replays_real_journal(tmp_path: Path) -> None:
    run_dir = tmp_path / "detached-run"
    _launch_detached_fixture(run_dir)

    plan, receipt, evidence = promotion._authoritative_detached_evidence(run_dir)

    assert receipt["passed"] is True
    assert receipt["plan_sha256"] == plan["plan_sha256"]
    assert evidence["attempt_event_count"] >= 4
    assert len(evidence["attempt_journal_head_sha256"]) == 64
    assert evidence["supervisor_attempt"] == 1


@pytest.mark.skipif(sys.platform != "darwin", reason="strong containment requires macOS")
def test_authoritative_detached_evidence_rejects_orphan_receipt(tmp_path: Path) -> None:
    run_dir = tmp_path / "detached-run"
    _launch_detached_fixture(run_dir)
    (run_dir / promotion.detached.ATTEMPTS_FILE).unlink()

    with pytest.raises(
        promotion.detached.DetachedStepError,
        match="terminal receipt exists without an authoritative journal record",
    ):
        promotion._authoritative_detached_evidence(run_dir)


def test_latest_checkpoint_rejects_adapter_substitution(tmp_path: Path) -> None:
    inputs = _promotion_fixture(tmp_path)
    checkpoint = inputs["source_adapter"] / inputs["latest"]["checkpoint"] / "adapter.safetensors"
    checkpoint.write_bytes(b"substituted")

    with pytest.raises(
        promotion.TrainingPromotionError,
        match="training_checkpoint_adapter_binding_mismatch",
    ):
        promotion._latest_checkpoint_evidence(inputs["source_adapter"], inputs["latest"])


def test_safetensors_rejects_duplicate_header_keys(tmp_path: Path) -> None:
    header = b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    path = tmp_path / "duplicate.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00" * 4)

    with pytest.raises(
        promotion.TrainingPromotionError,
        match="test_safetensors_header_invalid",
    ):
        promotion._validate_safetensors(path, role="test")


def test_safetensors_accepts_explicit_null_metadata(tmp_path: Path) -> None:
    header = b'{"__metadata__":null,"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    path = tmp_path / "null-metadata.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00" * 4)

    evidence = promotion._validate_safetensors(path, role="test")

    assert evidence["tensor_count"] == 1


def test_safetensors_rejects_non_string_metadata(tmp_path: Path) -> None:
    header = b'{"__metadata__":{"step":576},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    path = tmp_path / "invalid-metadata.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00" * 4)

    with pytest.raises(
        promotion.TrainingPromotionError,
        match="test_safetensors_metadata_invalid",
    ):
        promotion._validate_safetensors(path, role="test")


def test_promotion_receipt_is_create_or_verify(tmp_path: Path) -> None:
    output = tmp_path / "promotion.json"
    document = {"schema": promotion.SCHEMA, "promotion_sha256": "a" * 64}

    promotion._write_create_or_verify(output, document)
    promotion._write_create_or_verify(output, document)
    assert output.read_bytes() == canonical_json_bytes(document) + b"\n"

    with pytest.raises(
        promotion.TrainingPromotionError,
        match="existing_promotion_receipt_differs",
    ):
        promotion._write_create_or_verify(output, {**document, "promotion_sha256": "b" * 64})
