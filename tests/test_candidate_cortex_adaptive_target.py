from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from core.learning.candidate_cortex_training import (
    StagePolicy,
    build_stage_command,
    read_authenticated_journal,
)
from tools import run_candidate_cortex_adaptive_target as target
from tools import run_detached_step as detached
from tools import verify_candidate_cortex_adaptive_resume as resume


def _plan(tmp_path: Path) -> dict[str, Any]:
    run = tmp_path / "run"
    adapter = run / "adapter"
    data = tmp_path / "data"
    model = tmp_path / "model"
    for path in (run, adapter, data, model):
        path.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "identity",
        "run_id": "run",
    }
    identity_path = run / "adapter_identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    config = {
        "lora_parameters": {
            "dropout": 0.0,
            "keys": ["self_attn.q_proj"],
            "rank": 2,
            "scale": 1.0,
        },
        "optimizer": "adafactor",
        "optimizer_config": {
            "adafactor": {
                "relative_step": False,
                "scale_parameter": False,
                "beta_1": None,
            }
        },
    }
    config_path = run / "mlx_lora_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return {
        "plan_sha256": "1" * 64,
        "run_id": "run",
        "python": "/usr/bin/python3",
        "model": {
            "canonical_path": str(model),
            "descriptor_sha256": "2" * 64,
        },
        "dataset": {
            "data_root": str(data),
            "receipt_sha256": "3" * 64,
        },
        "training": {
            "rank": 2,
            "scale": 1.0,
            "dropout": 0.0,
            "num_layers": -1,
            "targets": ["self_attn.q_proj"],
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 128,
            "learning_rate": 1e-5,
            "save_every": 10,
            "eval_every": 5,
            "report_every": 5,
            "val_batches": 1,
            "seed": 7,
        },
        "optimizer": {"name": "adafactor"},
        "stages": {
            "initial_iterations": 100,
            "growth_factor": 2,
            "max_stages": 5,
            "min_stages": 2,
            "patience": 2,
            "min_loss_improvement": 0.002,
            "max_loss_regression_fraction": 0.02,
            "persona_floor": 0.9,
            "retention_floor": 0.98,
            "no_regression_floor": 1.0,
            "min_eval_samples": 32,
        },
        "paths": {
            "run_root": str(run),
            "adapter_root": str(adapter),
            "checkpoint_root": str(adapter),
            "adapter_identity": str(identity_path),
            "mlx_config": str(config_path),
        },
    }


def test_publish_stage_moves_one_cumulative_checkpoint_and_removes_alias(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    policy = StagePolicy(**plan["stages"])
    local = target.stage_adapter_root(plan, 0)
    local.mkdir(parents=True)
    payload = b"exact-adapter"
    (local / "0000100_adapters.safetensors").write_bytes(payload)
    (local / "adapters.safetensors").write_bytes(payload)
    (local / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": -1,
                "lora_parameters": {
                    "dropout": 0.0,
                    "keys": ["self_attn.q_proj"],
                    "rank": 2,
                    "scale": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    command = build_stage_command(plan, stage_index=0, resume_checkpoint=None)

    completion = target._publish_checkpoint(
        plan,
        0,
        command,
        {"sample_count": 2, "min_available_bytes": 100},
    )

    canonical = (
        Path(plan["paths"]["adapter_root"])
        / f"{policy.cumulative_iterations(0):07d}_adapters.safetensors"
    )
    assert canonical.read_bytes() == payload
    assert not (local / "adapters.safetensors").exists()
    assert not (local / "0000100_adapters.safetensors").exists()
    assert completion["checkpoint"]["path"] == str(canonical.resolve())
    assert target._validated_completion(plan, 0) == completion


def test_reset_incomplete_stage_removes_only_unadmitted_stage_outputs(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    stage = target.stage_adapter_root(plan, 1)
    stage.mkdir(parents=True)
    (stage / "partial.safetensors").write_bytes(b"partial")
    canonical = Path(plan["paths"]["adapter_root"]) / "0000300_adapters.safetensors"
    canonical.write_bytes(b"uncommitted")
    previous = Path(plan["paths"]["adapter_root"]) / "0000100_adapters.safetensors"
    previous.write_bytes(b"admitted")

    target._reset_incomplete_stage(plan, 1)

    assert stage.is_dir() and not list(stage.iterdir())
    assert not canonical.exists()
    assert previous.read_bytes() == b"admitted"


def test_hybrid_stage_is_split_below_descriptor_failure_horizon(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    first = target._stage_segments(plan, 0)
    second = target._stage_segments(plan, 1)

    assert [(item.start_iteration, item.iterations) for item in first] == [
        (0, 48),
        (48, 48),
        (96, 4),
    ]
    assert [(item.start_iteration, item.iterations) for item in second] == [
        (0, 48),
        (48, 48),
        (96, 48),
        (144, 48),
        (192, 8),
    ]


def test_segment_command_binds_exact_range_and_resume_artifact(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    previous = Path(plan["paths"]["adapter_root"]) / "0000100_adapters.safetensors"
    previous.write_bytes(b"previous")
    binding = {
        "path": str(previous.resolve()),
        "size_bytes": previous.stat().st_size,
        "sha256": target.file_sha256(previous),
        "lines": 1,
    }
    segment = target._stage_segments(plan, 1)[0]

    command = target._segment_command(
        plan,
        stage_index=1,
        segment=segment,
        stage_resume_checkpoint=binding,
        actual_resume_checkpoint=binding,
    )

    assert command[command.index("--iters") + 1] == "48"
    assert command[command.index("--save-every") + 1] == "48"
    assert command[command.index("--steps-per-eval") + 1] == "48"
    assert command[command.index("--resume-adapter-file") + 1] == str(previous)
    assert command[command.index("--adapter-path") + 1] == str(
        target._segment_adapter_root(plan, 1, 0)
    )


def test_resumed_batch_iterator_matches_uninterrupted_order() -> None:
    from mlx_lm.tuner import trainer

    dataset = [([index, index + 1], 0) for index in range(24)]
    upstream = trainer.iterate_batches(
        dataset,
        batch_size=1,
        max_seq_length=32,
        loop=True,
        seed=19,
    )
    upstream_order = [int(next(upstream)[0][0, 0].item()) for _ in range(14)]
    uninterrupted = target._iterate_batches_from_stage_offset(
        dataset,
        batch_size=1,
        max_seq_length=32,
        loop=True,
        seed=19,
        start_iteration=0,
    )
    expected = [int(next(uninterrupted)[0][0, 0].item()) for _ in range(14)]
    resumed = target._iterate_batches_from_stage_offset(
        dataset,
        batch_size=1,
        max_seq_length=32,
        loop=True,
        seed=19,
        start_iteration=9,
    )

    assert expected == upstream_order
    assert [int(next(resumed)[0][0, 0].item()) for _ in range(5)] == expected[9:]


def test_optimizer_and_mlx_rng_state_round_trip(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    original = optim.Adafactor(
        learning_rate=1e-5,
        relative_step=False,
        scale_parameter=False,
        beta_1=None,
    )
    original.init(model.trainable_parameters())
    original.state["step"] = mx.array(7, mx.uint64)
    rng_before = [np.asarray(value) for value in mx.random.state]
    path = tmp_path / "optimizer_state.safetensors"

    target._save_optimizer_state(path, original)
    mx.random.seed(999)
    restored = optim.Adafactor(
        learning_rate=1e-5,
        relative_step=False,
        scale_parameter=False,
        beta_1=None,
    )
    target._restore_optimizer_state(path, restored)

    assert int(restored.state["step"].item()) == 7
    assert all(
        np.array_equal(before, np.asarray(after))
        for before, after in zip(rng_before, mx.random.state)
    )


def test_segmented_stage_publication_binds_every_durable_segment(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    segments = target._stage_segments(plan, 1)
    prior = Path(plan["paths"]["adapter_root"]) / "0000100_adapters.safetensors"
    prior.write_bytes(b"admitted-stage-zero")
    prior_binding = {
        "path": str(prior.resolve()),
        "size_bytes": prior.stat().st_size,
        "sha256": target.file_sha256(prior),
        "lines": 1,
    }
    completions = []
    for segment in segments:
        root = target._segment_adapter_root(plan, 1, segment.index)
        root.mkdir(parents=True)
        payload = f"adapter-{segment.index}".encode("ascii")
        (root / f"{segment.iterations:07d}_adapters.safetensors").write_bytes(
            payload
        )
        (root / "adapters.safetensors").write_bytes(payload)
        (root / "adapter_config.json").write_text(
            json.dumps(
                {
                    "fine_tune_type": "lora",
                    "num_layers": -1,
                    "lora_parameters": {
                        "dropout": 0.0,
                        "keys": ["self_attn.q_proj"],
                        "rank": 2,
                        "scale": 1.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        optimizer = target._segment_optimizer_path(plan, 1, segment.index)
        optimizer.write_bytes(f"optimizer-{segment.index}".encode("ascii"))
        completions.append(
            target._publish_segment_completion(
                plan,
                1,
                segment,
                ("python", "-m", "mlx_lm", str(segment.index)),
                {"sample_count": 1, "min_available_bytes": 100},
            )
        )

    completion = target._publish_checkpoint(
        plan,
        1,
        build_stage_command(
            plan,
            stage_index=1,
            resume_checkpoint=prior_binding,
        ),
        {"sample_count": len(segments), "min_available_bytes": 100},
        local_root=target._segment_adapter_root(plan, 1, segments[-1].index),
        local_iterations=segments[-1].iterations,
        segment_completions=tuple(completions),
    )

    assert completion["schema"] == target.SEGMENTED_STAGE_COMPLETION_SCHEMA
    assert target._validated_completion(plan, 1) == completion
    final_optimizer = target._segment_optimizer_path(plan, 1, segments[-1].index)
    final_optimizer.write_bytes(b"tampered")
    with pytest.raises(
        target.CandidateCortexTrainingError, match="segment_artifact_drift"
    ):
        target._validated_completion(plan, 1)


def test_phase_boundary_execs_bound_launcher_and_authenticates_restart(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _plan(tmp_path)
    journal = Path(plan["paths"]["run_root"]) / "training_journal.jsonl"
    journal_key = tmp_path / "journal.key"
    key = b"k" * 64
    journal_key.write_bytes(key)
    captured: dict[str, Any] = {}

    class ExecRequestedError(RuntimeError):
        pass

    def _execve(path: str, argv: list[str], environment: dict[str, str]) -> None:
        captured.update(path=path, argv=argv, environment=environment)
        raise ExecRequestedError

    monkeypatch.setattr(target.os, "execve", _execve)
    with pytest.raises(ExecRequestedError):
        target._restart_for_clean_model_phase(
            plan,
            run_root=Path(plan["paths"]["run_root"]),
            journal_key=journal_key,
            journal=journal,
            key=key,
            stage_index=0,
            next_phase="measure",
            execution_id="cp926-recovery",
        )

    assert captured["path"] == plan["python"]
    assert captured["argv"][0] == plan["python"]
    assert captured["argv"][-6:] == [
        "--run-root",
        str(Path(plan["paths"]["run_root"]).resolve()),
        "--journal-key",
        str(journal_key.resolve()),
        "--execution-id",
        "cp926-recovery",
    ]
    assert captured["environment"]["AURA_CANDIDATE_CORTEX_PHASE"] == "measure"
    events = read_authenticated_journal(journal, key=key)
    assert len(events) == 1
    assert events[0]["event_type"] == "phase_restart_requested"
    assert events[0]["payload"]["stage_index"] == 0
    assert events[0]["payload"]["next_phase"] == "measure"


def test_resume_verdict_matches_detached_consumer(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    plan = _plan(tmp_path)
    key = tmp_path / "key"
    key.write_bytes(b"k" * 64)
    monkeypatch.setattr(resume, "load_and_verify_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        resume,
        "read_authenticated_journal",
        lambda *_args, **_kwargs: [
            {"event_type": "canary_observed"},
            {"event_type": "canary_admitted"},
            {"event_type": "stage_admitted"},
        ],
    )
    monkeypatch.setenv("AURA_DETACHED_PLAN_SHA256", "a" * 64)
    monkeypatch.setenv("AURA_DETACHED_COMMAND_SHA256", "b" * 64)
    monkeypatch.setenv("AURA_DETACHED_PRIOR_ATTEMPT", "1")
    monkeypatch.setenv("AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256", "c" * 64)

    assert resume.main(
        ["--run-root", str(tmp_path / "run"), "--journal-key", str(key)]
    ) == 0
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["verdict"] == "safe_to_resume"
    assert verdict["checkpoint_sequence"] == 1
    detached.validate_resume_verdict(
        verdict,
        plan_sha256="a" * 64,
        command_sha256="b" * 64,
        prior_attempt=1,
        prior_journal_head_sha256="c" * 64,
    )
    assert verdict["evidence_sha256"] == detached._sha256(verdict["evidence"])
