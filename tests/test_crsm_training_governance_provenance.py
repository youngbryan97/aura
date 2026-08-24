from __future__ import annotations

import json
import os

import pytest

from training import train_and_fuse


def _install_delegated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURA_DELEGATED_GOVERNANCE_RECEIPT_ID", "will-train-1")
    monkeypatch.setenv("AURA_DELEGATED_AUTHORITY_INTENT_ID", "intent-train-1")
    monkeypatch.setenv("AURA_DELEGATED_GOVERNANCE_DOMAIN", "semantic_weight_update")
    monkeypatch.setenv(
        "AURA_DELEGATED_GOVERNANCE_SOURCE",
        "system_maintenance:crsm_closure",
    )
    monkeypatch.setenv("AURA_DELEGATED_GOVERNANCE_PARENT_PID", str(os.getppid()))


def _write_current_manifest(dataset, manifest_path, *, rejected_by_reason=None) -> None:
    state = train_and_fuse._jsonl_file_stats(dataset)
    manifest_path.write_text(
        json.dumps(
            {
                "source_lines": state["lines"],
                "source_size": state["size"],
                "source_mtime": state["mtime"],
                "source_sha256": state["sha256"],
                "accepted": state["lines"],
                "rejected_by_reason": rejected_by_reason or {},
            }
        ),
        encoding="utf-8",
    )


def test_live_crsm_closeout_requires_parent_bound_semantic_receipt(monkeypatch) -> None:
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    for key in (
        "AURA_DELEGATED_GOVERNANCE_RECEIPT_ID",
        "AURA_DELEGATED_AUTHORITY_INTENT_ID",
        "AURA_DELEGATED_GOVERNANCE_DOMAIN",
        "AURA_DELEGATED_GOVERNANCE_SOURCE",
        "AURA_DELEGATED_GOVERNANCE_PARENT_PID",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(SystemExit, match="parent-bound semantic_weight_update"):
        train_and_fuse.enforce_live_delegated_authority(
            crsm_delta=True,
            tag="crsm-closeout",
        )

    _install_delegated_env(monkeypatch)
    train_and_fuse.enforce_live_delegated_authority(
        crsm_delta=True,
        tag="crsm-closeout",
    )


def test_operator_cli_lane_does_not_require_live_parent_receipt(monkeypatch) -> None:
    monkeypatch.delenv("AURA_LAUNCHED_FROM_APP", raising=False)
    train_and_fuse.enforce_live_delegated_authority(
        crsm_delta=True,
        tag="crsm-closeout",
    )


def test_training_claim_attributes_the_real_base_model() -> None:
    base_model = train_and_fuse.REPO_DIR / "models" / "candidate-cortex"
    claim = train_and_fuse._training_lane_claim(
        base_model,
        source="training_tooling:crsm_delta_lora",
    )

    assert claim.model_path == str(base_model)
    assert claim.purpose == "train"
    assert claim.request_gb > 0.0
    assert claim.metadata["pipeline"] == "train_and_fuse"


def test_delegated_worker_consumes_inherited_lane_immediately(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Lease:
        inherited = True

    def _acquire(**kwargs):
        captured.update(kwargs)
        return _Lease()

    for key, value in {
        "AURA_MODEL_LANE_INHERITED_OWNER_ID": "pipeline-owner",
        "AURA_MODEL_LANE_INHERITED_REQUEST_ID": "pipeline-request",
        "AURA_MODEL_LANE_INHERITED_MODEL_PATH": "/models/qwen-32b",
        "AURA_MODEL_LANE_INHERITED_PURPOSE": "compound",
        "AURA_MODEL_LANE_DELEGATION_TOKEN": "pipeline-token",
    }.items():
        monkeypatch.setenv(key, value)
    import core.runtime.model_lane_control as lane_module

    monkeypatch.setattr(lane_module, "acquire_standalone_model_lane", _acquire)

    assert train_and_fuse.consume_inherited_pipeline_lane() is True
    assert captured["model_path"] == "/models/qwen-32b"
    assert captured["purpose"] == "compound"


def test_published_model_manifest_preserves_governance_provenance(
    monkeypatch,
    tmp_path,
) -> None:
    _install_delegated_env(monkeypatch)
    fused_root = tmp_path / "fused-model"
    fused_path = fused_root / "Aura-32B-crsm"
    base_model = tmp_path / "Qwen2.5-32B-Instruct-4bit"
    fused_path.mkdir(parents=True)
    base_model.mkdir()
    monkeypatch.setattr(train_and_fuse, "FUSED_BASE_DIR", fused_root)
    descriptor = {"descriptor_sha256": "a" * 64}
    monkeypatch.setattr(
        train_and_fuse,
        "build_model_artifact_descriptor",
        lambda _path: descriptor,
    )
    captured: dict[str, object] = {}

    def _record(**kwargs):
        captured.update(kwargs)
        return {
            "candidate_receipt_path": str(fused_root / "candidates" / "candidate.json"),
            "active_pointer_unchanged": True,
        }

    monkeypatch.setattr(train_and_fuse, "record_upgrade_candidate", _record)

    receipt = train_and_fuse.publish_manifest(
        fused_path,
        tag="crsm-closeout",
        base_model=base_model,
    )

    assert receipt["active_pointer_unchanged"] is True
    assert captured["artifact_descriptor"] == descriptor
    assert captured["candidate_model_path"] == fused_path
    assert captured["base_model_path"] == base_model
    assert captured["source"] == "training.train_and_fuse"
    assert captured["governance"] == {
        "will_receipt_id": "will-train-1",
        "executive_intent_id": "intent-train-1",
        "domain": "semantic_weight_update",
        "source": "system_maintenance:crsm_closure",
    }


def test_consumed_marker_preserves_authority_receipts(monkeypatch, tmp_path) -> None:
    _install_delegated_env(monkeypatch)
    dataset = tmp_path / "captures.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    _write_current_manifest(dataset, manifest_path)
    monkeypatch.setattr(train_and_fuse, "CRSM_DATASET", dataset)
    captured: dict[str, object] = {}

    class _Monitor:
        def mark_dataset_consumed(self, **kwargs):
            captured.update(kwargs)
            return True

    import core.consciousness.crsm_loop_monitor as monitor_module

    monkeypatch.setattr(monitor_module, "get_crsm_loop_monitor", lambda: _Monitor())
    train_and_fuse.mark_crsm_loop_consumed_after_training(
        tmp_path / "fused",
        manifest_path=manifest_path,
        source="training.train_and_fuse.crsm_delta",
    )

    assert captured["governance_receipt_id"] == "will-train-1"
    assert captured["authority_intent_id"] == "intent-train-1"


def test_consumed_marker_failure_is_a_terminal_training_failure(monkeypatch, tmp_path) -> None:
    dataset = tmp_path / "captures.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    _write_current_manifest(dataset, manifest_path)
    monkeypatch.setattr(train_and_fuse, "CRSM_DATASET", dataset)

    class _Monitor:
        def mark_dataset_consumed(self, **_kwargs):
            return False

    import core.consciousness.crsm_loop_monitor as monitor_module

    monkeypatch.setattr(monitor_module, "get_crsm_loop_monitor", lambda: _Monitor())

    with pytest.raises(SystemExit, match="final consumed marker was not committed"):
        train_and_fuse.mark_crsm_loop_consumed_after_training(
            tmp_path / "fused",
            manifest_path=manifest_path,
            source="training.train_and_fuse.crsm_delta",
        )


def test_same_size_dataset_change_cannot_commit_consumed_marker(monkeypatch, tmp_path) -> None:
    dataset = tmp_path / "captures.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    _write_current_manifest(dataset, manifest_path)
    dataset.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(train_and_fuse, "CRSM_DATASET", dataset)

    with pytest.raises(SystemExit, match="dataset identity changed"):
        train_and_fuse.mark_crsm_loop_consumed_after_training(
            tmp_path / "fused",
            manifest_path=manifest_path,
            source="training.train_and_fuse.crsm_delta",
            required=True,
        )


def test_untrained_selection_overflow_cannot_close_loop(monkeypatch, tmp_path) -> None:
    dataset = tmp_path / "captures.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    _write_current_manifest(
        dataset,
        manifest_path,
        rejected_by_reason={"over_max_examples": 1},
    )
    monkeypatch.setattr(train_and_fuse, "CRSM_DATASET", dataset)

    with pytest.raises(SystemExit, match="remain untrained"):
        train_and_fuse.mark_crsm_loop_consumed_after_training(
            tmp_path / "fused",
            manifest_path=manifest_path,
            source="training.train_and_fuse.crsm_delta",
            required=True,
        )
