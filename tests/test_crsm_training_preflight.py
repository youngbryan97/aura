"""Safety checks for CRSM LoRA train/fuse execution."""
from __future__ import annotations

import hashlib
import json

import pytest

from training import model_basis, resume_training, run_unattended, train_and_fuse

GIB = 1024**3


def _patch_resources(
    resource_observer,
    *,
    available_gb=40.0,
    percent=50.0,
    free_disk_gb=200.0,
):
    resource_observer.configure_memory(
        total_bytes=64 * GIB,
        available_bytes=int(available_gb * GIB),
        percent=percent,
    )
    resource_observer.configure_disk(
        total_bytes=512 * GIB,
        free_bytes=int(free_disk_gb * GIB),
    )


def test_training_preflight_passes_with_headroom(monkeypatch, tmp_path, resource_observer):
    _patch_resources(resource_observer)
    monkeypatch.setattr(train_and_fuse, "_live_aura_processes", lambda **_kwargs: [])

    report = train_and_fuse.training_preflight(base_model=tmp_path / "Qwen2.5-32B-Instruct-4bit", skip_train=False)

    assert report["passed"] is True
    assert report["mode"] == "train_fuse_publish"
    assert report["requirements"]["min_available_gb"] == 28.0


def test_training_preflight_reports_crsm_delta_mode(monkeypatch, tmp_path, resource_observer):
    _patch_resources(resource_observer)
    monkeypatch.setattr(train_and_fuse, "_live_aura_processes", lambda **_kwargs: [])

    report = train_and_fuse.training_preflight(
        base_model=tmp_path / "Qwen2.5-32B-Instruct-4bit",
        skip_train=False,
        crsm_delta=True,
    )

    assert report["passed"] is True
    assert report["mode"] == "crsm_delta_train_fuse_publish"


def test_training_preflight_sizes_the_27b_from_artifact_metadata(
    monkeypatch,
    tmp_path,
    resource_observer,
):
    model = tmp_path / "opaque-cortex"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"measured-by-index")
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_parameters": 27_000_000_000,
                    "total_size": 15 * GIB,
                }
            }
        ),
        encoding="utf-8",
    )
    _patch_resources(resource_observer, available_gb=40.0)
    monkeypatch.setattr(train_and_fuse, "_live_aura_processes", lambda **_kwargs: [])

    report = train_and_fuse.training_preflight(
        base_model=model,
        skip_train=False,
    )

    assert report["size"] == "27B"
    assert report["requirements"]["min_available_gb"] > 44.0
    assert report["passed"] is False
    assert any("available_memory" in blocker for blocker in report["blockers"])


def test_training_preflight_blocks_low_memory(monkeypatch, tmp_path, resource_observer):
    _patch_resources(resource_observer, available_gb=9.0, percent=91.0)
    monkeypatch.setattr(train_and_fuse, "_live_aura_processes", lambda **_kwargs: [])

    report = train_and_fuse.training_preflight(base_model=tmp_path / "Qwen2.5-32B-Instruct-4bit", skip_train=False)

    assert report["passed"] is False
    assert any("available_memory" in blocker for blocker in report["blockers"])
    assert any("memory_pressure" in blocker for blocker in report["blockers"])


def test_training_preflight_blocks_live_aura_unless_explicitly_allowed(
    monkeypatch,
    tmp_path,
    resource_observer,
):
    _patch_resources(resource_observer)
    monkeypatch.setattr(
        train_and_fuse,
        "_live_aura_processes",
        lambda **_kwargs: [{"pid": 123, "cmdline": "aura_main.py"}],
    )

    blocked = train_and_fuse.training_preflight(base_model=tmp_path / "Qwen2.5-32B-Instruct-4bit", skip_train=False)
    assert blocked["passed"] is False
    assert "live_aura_processes:1" in blocked["blockers"]

    monkeypatch.setenv("AURA_TRAINING_ALLOW_LIVE_AURA", "1")
    allowed = train_and_fuse.training_preflight(base_model=tmp_path / "Qwen2.5-32B-Instruct-4bit", skip_train=False)
    assert allowed["passed"] is True


def test_run_unattended_accepts_resume_and_preflight_only_flags():
    args = run_unattended.parse_args(["--resume", "--preflight-only", "--crsm-delta", "--tag", "crsm-closeout"])

    assert args.resume is True
    assert args.preflight_only is True
    assert args.crsm_delta is True
    assert args.tag == "crsm-closeout"


def test_run_unattended_resume_fuse_publish_marks_crsm_consumed(monkeypatch):
    commands = []
    monkeypatch.setattr(run_unattended, "update_state", lambda **_kwargs: {})
    monkeypatch.setattr(run_unattended, "_spawn", lambda cmd, *, started_at: commands.append(cmd) or 0)
    args = run_unattended.parse_args(["--tag", "crsm-closeout"])

    rc = run_unattended.run_fuse_publish(args, started_at="now")

    assert rc == 0
    assert commands
    assert "--mark-crsm-consumed" in commands[0]


def test_run_unattended_passes_crsm_delta_to_train_and_fuse(monkeypatch):
    commands = []
    monkeypatch.setattr(run_unattended, "update_state", lambda **_kwargs: {})
    monkeypatch.setattr(run_unattended, "_spawn", lambda cmd, *, started_at: commands.append(cmd) or 0)
    args = run_unattended.parse_args(["--crsm-delta", "--tag", "crsm-closeout"])

    rc = run_unattended.run_train_and_fuse(args, started_at="now")

    assert rc == 0
    assert commands
    assert "--crsm-delta" in commands[0]


def test_run_unattended_crsm_delta_ignores_historical_partial_run(monkeypatch):
    calls = []
    monkeypatch.setattr(run_unattended, "_install_signal_handlers", lambda _started_at: None)
    monkeypatch.setattr(run_unattended, "update_state", lambda **_kwargs: {})
    monkeypatch.setattr(run_unattended, "has_partial_run", lambda: True)
    monkeypatch.setattr(
        run_unattended,
        "run_resume",
        lambda *, started_at: (_ for _ in ()).throw(AssertionError("resume should not run")),
    )
    monkeypatch.setattr(
        run_unattended,
        "run_train_and_fuse",
        lambda args, *, started_at: calls.append(args.crsm_delta) or 0,
    )

    rc = run_unattended.main(["--crsm-delta", "--tag", "crsm-closeout"])

    assert rc == 0
    assert calls == [True]


def test_build_crsm_delta_train_command_uses_real_resume_adapter(tmp_path):
    command = train_and_fuse.build_crsm_delta_train_command(
        base_model=tmp_path / "Qwen2.5-32B-Instruct-4bit",
        data_dir=tmp_path / "delta_data",
        adapter_dir=tmp_path / "delta_adapter",
        resume_adapter_file=tmp_path / "source_adapter" / "adapters.safetensors",
        iters=125,
        max_seq_length=1024,
        lora_config_path=tmp_path / "source_adapter" / "lora_config.yaml",
    )

    joined = " ".join(command)
    assert "mlx_lm lora" in joined
    assert "--resume-adapter-file" in command
    assert str(tmp_path / "source_adapter" / "adapters.safetensors") in command
    assert command[command.index("--iters") + 1] == "125"
    assert command[command.index("--max-seq-length") + 1] == "1024"


def test_crsm_delta_commits_basis_before_the_training_process_starts(
    monkeypatch,
    tmp_path,
):
    model = tmp_path / "model"
    source = tmp_path / "source-adapter"
    output = tmp_path / "delta-adapter"
    data = tmp_path / "data"
    model.mkdir()
    source.mkdir()
    data.mkdir()
    source_weights = b"source-adapter-weights"
    (source / "adapters.safetensors").write_bytes(source_weights)
    (source / "lora_config.yaml").write_text("lora_parameters: {}\n", encoding="utf-8")
    basis = model_basis.TrainingModelBasis(
        path=model,
        descriptor={"descriptor_sha256": "a" * 64},
        descriptor_sha256="a" * 64,
        source="test",
    )
    monkeypatch.setattr(train_and_fuse, "ADAPTER_DIR", source)
    monkeypatch.setattr(train_and_fuse, "assert_adapter_matches_basis", lambda *_a: None)

    def _run(_command, **_kwargs):
        config = json.loads((output / "training_config.json").read_text(encoding="utf-8"))
        assert config["training_basis"]["descriptor_sha256"] == "a" * 64
        assert config["source_adapter_sha256"] == hashlib.sha256(source_weights).hexdigest()
        (output / "adapters.safetensors").write_bytes(b"trained")
        return 0

    monkeypatch.setattr(train_and_fuse, "_run", _run)

    observed = train_and_fuse.train_crsm_delta_lora(
        base_model=model,
        model_basis=basis,
        data_dir=data,
        adapter_dir=output,
        iters=25,
        max_seq_length=512,
    )

    assert observed == output


def test_build_crsm_delta_dataset_adds_retention_and_provenance(monkeypatch, tmp_path):
    dataset = tmp_path / "synthetic" / "lora_dataset.jsonl"
    dataset.parent.mkdir()
    dataset.write_text(
        "\n".join(
            json.dumps({"text": f"User: remembered event {idx}\nAura: I integrated that event into future behavior."})
            for idx in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "training_data"
    data_dir.mkdir()
    retention = {
        "messages": [
            {"role": "system", "content": "Aura"},
            {"role": "user", "content": "keep skill"},
            {"role": "assistant", "content": "I keep the skill active."},
        ]
    }
    (data_dir / "train.jsonl").write_text(json.dumps(retention) + "\n", encoding="utf-8")
    (data_dir / "valid.jsonl").write_text(json.dumps(retention) + "\n", encoding="utf-8")
    monkeypatch.setattr(train_and_fuse, "CRSM_DATASET", dataset)
    monkeypatch.setattr(train_and_fuse, "DATA_DIR", data_dir)
    monkeypatch.setattr(train_and_fuse, "CRSM_DELTA_MANIFEST", data_dir / "crsm_delta_manifest.json")

    manifest = train_and_fuse.build_crsm_delta_dataset(
        output_dir=data_dir / "delta",
        max_crsm_examples=4,
        retention_examples=1,
        seed=7,
    )

    assert manifest["delta_mode"] is True
    assert manifest["accepted"] == 4
    assert manifest["output"]["crsm_examples"] == 4
    assert manifest["output"]["retention_examples"] == 1
    assert manifest["output"]["train"]["lines"] > 0
    assert manifest["output"]["valid"]["lines"] > 0
    assert (data_dir / "crsm_delta_manifest.json").exists()


def test_resume_training_parses_zenith_resume_log(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    base = adapter_dir / "0066000_adapters.safetensors"
    later = adapter_dir / "0066500_adapters.safetensors"
    base.write_text("base", encoding="utf-8")
    later.write_text("later", encoding="utf-8")
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\n--- Resume Zenith from 0066000_adapters.safetensors, "
        "24153 iters remaining, target_total=90153, seq=4096 ---\n"
        "Iter 500: Saved adapter weights to "
        f"{later}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(resume_training, "ADAPTER_PATH", adapter_dir)
    monkeypatch.setattr(resume_training, "LOG_PATH", log_path)

    checkpoint, remaining = resume_training._resume_state_from_log()

    assert checkpoint == later
    assert remaining == 23653


def test_resume_training_uses_the_checkpoint_model_basis(monkeypatch, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = tmp_path / "training_config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(resume_training, "TRAINING_CONFIG_PATH", config)
    monkeypatch.setattr(
        resume_training,
        "load_recorded_training_model_basis",
        lambda path, **kwargs: type("Basis", (), {"path": model})(),
    )

    assert resume_training._load_base_model() == model


def test_run_unattended_memory_guard_blocks_process_tree_rss(monkeypatch):
    monkeypatch.setenv("AURA_TRAINING_MAX_PROCESS_TREE_RSS_GB", "12")
    monkeypatch.setattr(run_unattended, "_process_tree_rss_gb", lambda _pid: 12.5)

    reason = run_unattended._memory_guard_reason(123)

    assert reason == "process_tree_rss:12.5GB/12.0GB"


def test_run_unattended_memory_guard_blocks_host_pressure(monkeypatch, resource_observer):
    monkeypatch.setenv("AURA_TRAINING_MAX_PROCESS_TREE_RSS_GB", "80")
    monkeypatch.setenv("AURA_TRAINING_MAX_HOST_MEMORY_PERCENT", "90")
    monkeypatch.setattr(run_unattended, "_process_tree_rss_gb", lambda _pid: 10.0)
    resource_observer.configure_memory(percent=93.0)

    reason = run_unattended._memory_guard_reason(123)

    assert reason == "host_memory_pressure:93.0%/90.0%"


def test_run_unattended_update_state_uses_current_started_at(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    checkpoint = adapter_dir / "0000100_adapters.safetensors"
    checkpoint.write_text("adapter", encoding="utf-8")
    state_file = adapter_dir / "training_state.json"
    monkeypatch.setattr(run_unattended, "ADAPTER_DIR", adapter_dir)
    monkeypatch.setattr(run_unattended, "STATE_FILE", state_file)

    first = run_unattended.update_state(started_at="first", phase="boot")
    second = run_unattended.update_state(started_at="second", phase="running")

    assert first["started_at"] == "first"
    assert second["started_at"] == "second"
    assert second["previous_started_at"] == "first"
    assert second["last_iter"] == 100


def test_record_crsm_delta_training_state_writes_receipt(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    state_path = adapter_dir / "training_state.json"
    state_path.write_text(json.dumps({"phase": "train_and_fuse_done"}), encoding="utf-8")
    manifest = tmp_path / "crsm_delta_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_lines": 10,
                "accepted": 8,
                "output": {
                    "retention_examples": 3,
                    "train": {"sha256": "train-hash"},
                    "valid": {"sha256": "valid-hash"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(train_and_fuse, "ADAPTER_DIR", adapter_dir)

    train_and_fuse.record_crsm_delta_training_state(
        adapter_dir=tmp_path / "delta_adapter",
        fused_path=tmp_path / "fused_model",
        manifest_path=manifest,
        iters=25,
        max_seq_length=1024,
    )

    state = json.loads(state_path.read_text())
    receipt = state["crsm_delta"]
    assert receipt["status"] == "fused_published_marker_ready"
    assert receipt["adapter_path"] == str(tmp_path / "delta_adapter")
    assert receipt["fused_model_path"] == str(tmp_path / "fused_model")
    assert receipt["source_lines"] == 10
    assert receipt["accepted"] == 8
    assert receipt["rejected"] == 2
    assert receipt["retention_examples"] == 3
    assert receipt["train_sha256"] == "train-hash"
    assert receipt["valid_sha256"] == "valid-hash"
    assert receipt["iters"] == 25
    assert receipt["max_seq_length"] == 1024


def test_record_crsm_delta_training_state_fails_when_receipt_is_not_durable(
    monkeypatch,
    tmp_path,
):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "training_state.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source_lines": 1, "accepted": 1}), encoding="utf-8")
    monkeypatch.setattr(train_and_fuse, "ADAPTER_DIR", adapter_dir)
    monkeypatch.setattr(
        train_and_fuse,
        "atomic_write_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk-full")),
    )

    with pytest.raises(RuntimeError, match="failed to record CRSM delta training state"):
        train_and_fuse.record_crsm_delta_training_state(
            adapter_dir=tmp_path / "delta",
            fused_path=tmp_path / "fused",
            manifest_path=manifest,
        )
