from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from core.learning import candidate_cortex_fusion as fusion
from core.learning import candidate_cortex_training as training
from tools import run_candidate_cortex_fusion_target as fusion_target


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _fixture_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    run_root = tmp_path / "training-run"
    adapter_root = run_root / "adapters"
    adapter_root.mkdir(parents=True)
    checkpoint_path = adapter_root / "0000700_adapters.safetensors"
    checkpoint_path.write_bytes(b"exact-admitted-checkpoint")
    checkpoint = {
        "path": str(checkpoint_path.resolve()),
        "sha256": training.file_sha256(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
    }
    adapter_config = adapter_root / "adapter_config.json"
    _write_json(adapter_config, {"rank": 32, "num_layers": 48})

    model_root = tmp_path / "Qwen3.8-27B-4bit"
    model_root.mkdir()
    descriptor = {
        "schema": "aura.model_artifact_descriptor.v1",
        "canonical_path": str(model_root.resolve()),
        "repository_id": "mlx-community/Qwen3.8-27B-4bit",
        "revision": "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
        "artifact_profile": {"weight_bytes": 16 * 1024**3},
    }
    descriptor["descriptor_sha256"] = training.document_sha256(descriptor)
    descriptor_path = tmp_path / "descriptor.json"
    _write_json(descriptor_path, descriptor)

    result_body = {
        "schema": training.ADAPTIVE_RESULT_SCHEMA,
        "plan_sha256": "1" * 64,
        "decision": {
            "decision": "COMPLETE",
            "reason": "convergence_patience_pass",
            "stage": 2,
        },
    }
    result = {
        **result_body,
        "result_sha256": training.document_sha256(result_body),
    }
    _write_json(run_root / "adaptive_result.json", result)
    plan = {
        "plan_sha256": result_body["plan_sha256"],
        "python": "/usr/bin/python3",
        "python_binding": {"path": "/usr/bin/python3"},
        "paths": {"adapter_root": str(adapter_root.resolve())},
        "model": {
            "canonical_path": str(model_root.resolve()),
            "descriptor_path": str(descriptor_path.resolve()),
            "descriptor_sha256": descriptor["descriptor_sha256"],
            "repository_id": descriptor["repository_id"],
            "revision": descriptor["revision"],
        },
    }
    authority = {
        "stage_index": 2,
        "cumulative_iterations": 700,
        "checkpoint": checkpoint,
    }
    monkeypatch.setattr(
        fusion,
        "_adaptive_authority",
        lambda *_args, **_kwargs: (plan, result, authority),
    )
    journal_key = tmp_path / "journal.key"
    journal_key.write_bytes(b"k" * 64)
    target_source = tmp_path / "fusion_target.py"
    target_source.write_text("raise SystemExit(0)\n", encoding="ascii")
    verifier_source = tmp_path / "fusion_verifier.py"
    verifier_source.write_text("raise SystemExit(0)\n", encoding="ascii")
    return plan, run_root, journal_key, target_source, verifier_source


def _fusion_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prime_future_descriptor: bool = False,
) -> tuple[dict[str, Any], Path, Path, Path]:
    training_plan, run_root, journal_key, target_source, verifier_source = _fixture_authority(
        tmp_path,
        monkeypatch,
    )
    fusion_root = tmp_path / "fusion-control"
    if prime_future_descriptor:
        fusion_root.mkdir()
        (fusion_root / "fused_model_descriptor.json").write_bytes(
            Path(training_plan["model"]["descriptor_path"]).read_bytes()
        )
    plan = fusion.prepare_fusion_plan(
        run_root=run_root,
        journal_key_path=journal_key,
        fusion_root=fusion_root,
        output_root=tmp_path / "fused-models",
        target_source=target_source,
        verifier_source=verifier_source,
        verify_full_model=False,
    )
    return plan, journal_key, target_source, verifier_source


def _redigest(plan: dict[str, Any]) -> None:
    material = dict(plan)
    material.pop("fusion_plan_sha256")
    plan["fusion_plan_sha256"] = training.document_sha256(material)


def test_fusion_plan_binds_complete_authority_and_exact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, journal_key, _, _ = _fusion_plan(tmp_path, monkeypatch)

    validated = fusion.validate_fusion_plan(
        plan,
        journal_key_path=journal_key,
        verify_full_model=False,
    )

    assert validated == plan
    assert plan["adaptive"]["stage_index"] == 2
    assert plan["adaptive"]["cumulative_iterations"] == 700
    assert plan["adaptive"]["checkpoint"]["sha256"] == training.file_sha256(
        Path(plan["adaptive"]["checkpoint"]["path"])
    )
    assert plan["output"]["generation_id"] in plan["output"]["path"]


def test_fusion_plan_binds_explicit_adaptive_result_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_plan, run_root, journal_key, target_source, verifier_source = (
        _fixture_authority(tmp_path, monkeypatch)
    )
    explicit = run_root / "adaptive-results" / "cp924-recovery.json"
    explicit.parent.mkdir()
    explicit.write_bytes((run_root / "adaptive_result.json").read_bytes())
    result = json.loads(explicit.read_text(encoding="utf-8"))
    authority = {
        "stage_index": 2,
        "cumulative_iterations": 700,
        "checkpoint": {
            "path": str(
                Path(training_plan["paths"]["adapter_root"])
                / "0000700_adapters.safetensors"
            ),
            "sha256": training.file_sha256(
                Path(training_plan["paths"]["adapter_root"])
                / "0000700_adapters.safetensors"
            ),
            "size_bytes": (
                Path(training_plan["paths"]["adapter_root"])
                / "0000700_adapters.safetensors"
            ).stat().st_size,
        },
    }
    observed_result_paths: list[Path | None] = []

    def _authority(*_args, adaptive_result_path=None, **_kwargs):
        observed_result_paths.append(adaptive_result_path)
        return training_plan, result, authority

    monkeypatch.setattr(fusion, "_adaptive_authority", _authority)
    plan = fusion.prepare_fusion_plan(
        run_root=run_root,
        journal_key_path=journal_key,
        fusion_root=tmp_path / "fusion-control",
        output_root=tmp_path / "fused-models",
        target_source=target_source,
        verifier_source=verifier_source,
        adaptive_result_path=explicit,
        verify_full_model=False,
    )
    assert plan["adaptive"]["result"]["path"] == str(explicit.resolve())
    assert fusion.validate_fusion_plan(
        plan,
        journal_key_path=journal_key,
        verify_full_model=False,
    ) == plan
    assert observed_result_paths == [explicit, explicit]


def test_fusion_plan_rejects_digest_and_adaptive_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, journal_key, _, _ = _fusion_plan(
        tmp_path,
        monkeypatch,
        prime_future_descriptor=True,
    )
    digest_tamper = copy.deepcopy(plan)
    digest_tamper["adaptive"]["cumulative_iterations"] = 701
    with pytest.raises(
        fusion.CandidateCortexFusionError,
        match="fusion_plan_digest_invalid",
    ):
        fusion.validate_fusion_plan(
            digest_tamper,
            journal_key_path=journal_key,
            verify_full_model=False,
        )

    identity_tamper = copy.deepcopy(plan)
    identity_tamper["adaptive"]["checkpoint"]["sha256"] = "f" * 64
    _redigest(identity_tamper)
    with pytest.raises(
        fusion.CandidateCortexFusionError,
        match="fusion_adaptive_checkpoint_mismatch",
    ):
        fusion.validate_fusion_plan(
            identity_tamper,
            journal_key_path=journal_key,
            verify_full_model=False,
        )


def test_fusion_plan_rejects_output_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, journal_key, _, _ = _fusion_plan(
        tmp_path,
        monkeypatch,
        prime_future_descriptor=True,
    )
    escaped = copy.deepcopy(plan)
    escaped["output"]["path"] = str(tmp_path / "outside" / "model")
    _redigest(escaped)

    with pytest.raises(
        fusion.CandidateCortexFusionError,
        match="fusion_output_path_escape",
    ):
        fusion.validate_fusion_plan(
            escaped,
            journal_key_path=journal_key,
            verify_full_model=False,
        )


def test_fusion_plan_rejects_bound_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, journal_key, target_source, _ = _fusion_plan(
        tmp_path,
        monkeypatch,
        prime_future_descriptor=True,
    )
    target_source.write_text("raise SystemExit(7)\n", encoding="ascii")

    with pytest.raises(
        fusion.CandidateCortexFusionError,
        match="fusion_source_target_binding_drift",
    ):
        fusion.validate_fusion_plan(
            plan,
            journal_key_path=journal_key,
            verify_full_model=False,
        )


def test_fusion_receipt_replays_provenance_and_descriptor_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _, _, _ = _fusion_plan(tmp_path, monkeypatch)
    model_path = Path(plan["output"]["path"])
    model_path.mkdir(parents=True)
    provenance = fusion.build_fusion_provenance(plan, fused_module_count=384)
    provenance_path = model_path / fusion.FUSION_PROVENANCE_FILE
    _write_json(provenance_path, provenance)
    descriptor_path = Path(plan["output"]["descriptor_path"])
    descriptor = {
        "schema": "aura.model_artifact_descriptor.v1",
        "canonical_path": str(model_path),
        "descriptor_sha256": "d" * 64,
    }
    _write_json(descriptor_path, descriptor)
    monkeypatch.setattr(
        fusion,
        "validate_model_artifact_descriptor",
        lambda raw, **_kwargs: raw,
    )
    body = {
        "schema": fusion.FUSION_RECEIPT_SCHEMA,
        "fusion_plan_sha256": plan["fusion_plan_sha256"],
        "generation_id": plan["output"]["generation_id"],
        "model_path": str(model_path),
        "descriptor": fusion._file_binding(descriptor_path),
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "provenance": fusion._file_binding(provenance_path),
    }
    receipt = {**body, "receipt_sha256": training.document_sha256(body)}

    assert fusion.validate_fusion_receipt(
        plan,
        receipt,
        verify_full_model=True,
    ) == receipt


def test_fusion_target_publishes_from_staging_under_exclusive_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _, _, _ = _fusion_plan(tmp_path, monkeypatch)
    Path(plan["output"]["root"]).mkdir(parents=True)
    observed: dict[str, Any] = {}

    def fake_fuse(_base: str, _adapter: Path, save_path: Path) -> int:
        observed["adapter"] = str(_adapter)
        save_path.mkdir()
        (save_path / "config.json").write_text("{}\n", encoding="ascii")
        (save_path / "model.safetensors").write_bytes(b"fused")
        return 384

    class _Lane:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(fusion_target, "standalone_model_lane", lambda **_kwargs: _Lane())
    monkeypatch.setattr(fusion_target, "_fuse_with_mlx", fake_fuse)
    monkeypatch.setattr(fusion_target, "_release_model_memory", lambda: None)

    final_path = fusion_target._fuse(plan)

    assert final_path == Path(plan["output"]["path"])
    assert final_path.is_dir()
    assert not Path(plan["output"]["staging_path"]).exists()
    assert observed["adapter"].endswith("adapter-input")
    assert fusion.validate_fusion_provenance(
        plan,
        json.loads(
            (final_path / fusion.FUSION_PROVENANCE_FILE).read_text(encoding="utf-8")
        ),
    )["fusion_plan_sha256"] == plan["fusion_plan_sha256"]
