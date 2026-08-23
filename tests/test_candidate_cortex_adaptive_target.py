from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.learning.candidate_cortex_training import (
    StagePolicy,
    build_stage_command,
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
