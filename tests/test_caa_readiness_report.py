"""Tests for CAA readiness verification from on-disk provenance."""

from __future__ import annotations

import json

import numpy as np

from core.brain.llm.model_artifact_profile import build_model_artifact_descriptor
from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS as RUNTIME_DIMENSIONS
from core.consciousness.caa.readiness_report import scan_vector_files, verify_readiness
from training.extract_steering_vectors import AFFECTIVE_DIMENSIONS, ALL_AFFECTIVE_DIMENSIONS


def _vec(
    path,
    *,
    source,
    extracted,
    derived_at=1000.0,
    model_path=None,
    model_descriptor_sha256=None,
):
    np.savez(
        path,
        v=np.zeros(8, dtype=np.float32),
        source=source,
        extracted=extracted,
        derived_at=derived_at,
        requested_layer=11,
        selected_layer=11,
        selection_reason=source,
        model_path=model_path or "",
        model_descriptor_sha256=model_descriptor_sha256 or "",
    )


def _model(root, *, revision):
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 64,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "vocab_size": 128,
                    "max_position_embeddings": 4096,
                },
                "quantization": {"bits": 4, "group_size": 64},
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(f"weights-{revision}".encode())
    return build_model_artifact_descriptor(
        root,
        repository_id="test/cortex",
        revision=revision,
    )


def _activate(
    fdir,
    model,
    descriptor,
    *,
    fused_at=500.0,
    steering_status=None,
):
    migration_contract = None
    if steering_status is not None:
        migration_contract = {
            "components": {
                "steering": {
                    "status": steering_status,
                    "authority_kind": "model_basis_quarantine",
                }
            }
        }
    fdir.mkdir(exist_ok=True)
    (fdir / "active.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "active_model_path": str(model),
                "fused_at": fused_at,
                "artifact_descriptor": descriptor,
                "migration_contract": migration_contract,
            }
        ),
        encoding="utf-8",
    )


def _setup(tmp_path, specs):
    vdir = tmp_path / "vectors"
    fdir = tmp_path / "fused-model"
    vdir.mkdir()
    fdir.mkdir()
    for i, (source, extracted) in enumerate(specs):
        _vec(vdir / f"vec_layer{i}.npz", source=source, extracted=extracted)
    (fdir / "active.json").write_text(
        json.dumps({"active_model_path": "/m/active", "fused_at": 500.0})
    )
    return vdir, fdir


def test_scan_reads_provenance(tmp_path):
    vdir, _ = _setup(tmp_path, [("runtime_derived_caa", False), ("extracted_contrastive", True)])
    scan = scan_vector_files(vdir)
    assert scan["files"] == 2
    assert scan["extracted"] == 1
    assert scan["runtime_derived"] == 1


def test_runtime_derived_is_bootstrap_below_capacity_without_exact_identity(tmp_path):
    vdir, fdir = _setup(tmp_path, [("runtime_derived_caa", False)] * 6)
    r = verify_readiness(vectors_dir=vdir, fused_model_dir=fdir)
    assert r["level"] == "bootstrap"
    assert r["below_design_capacity"] is True
    assert r["steering_capacity_pct"] < 100
    assert "exact active-model identity unavailable" in r["detail"]


def test_unbound_extracted_vectors_are_not_production(tmp_path):
    vdir, fdir = _setup(tmp_path, [("extracted_contrastive", True)] * 6)
    r = verify_readiness(vectors_dir=vdir, fused_model_dir=fdir)
    assert r["level"] == "bootstrap"
    assert r["below_design_capacity"] is True
    assert r["active_model_identity_valid"] is False


def test_mixed_when_some_runtime_targets_are_exactly_bound(tmp_path):
    vdir = tmp_path / "vectors"
    fdir = tmp_path / "fused-model"
    active = fdir / "active-model"
    vdir.mkdir()
    descriptor = _model(active, revision="partial")
    _activate(fdir, active, descriptor)
    digest = str(descriptor["descriptor_sha256"])
    pairs = [
        (key, layer)
        for key in {spec["key"] for spec in RUNTIME_DIMENSIONS}
        for layer in (25, 30, 35)
    ]
    for key, layer in pairs[:8]:
        _vec(
            vdir / f"{key}_layer{layer}.npz",
            source="extracted_caa",
            extracted=True,
            model_path=str(active),
            model_descriptor_sha256=digest,
        )
    r = verify_readiness(vectors_dir=vdir, fused_model_dir=fdir)
    assert r["level"] == "mixed"
    assert 0.0 < r["extracted_ratio"] < 1.0


def test_no_vectors_is_bootstrap(tmp_path):
    vdir = tmp_path / "empty"
    vdir.mkdir()
    r = verify_readiness(vectors_dir=vdir, fused_model_dir=tmp_path)
    assert r["level"] == "bootstrap"


def test_signed_deferral_is_neutral_runtime_disposition(tmp_path):
    vdir = tmp_path / "vectors"
    fdir = tmp_path / "fused-model"
    active = fdir / "active-model"
    vdir.mkdir()
    descriptor = _model(active, revision="deferred")
    _activate(fdir, active, descriptor, steering_status="deferred")

    report = verify_readiness(vectors_dir=vdir, fused_model_dir=fdir)

    assert report["level"] == "deferred"
    assert report["steering_authority_status"] == "deferred"
    assert report["serving_authorized"] is False
    assert report["below_design_capacity"] is False
    assert report["steering_capacity_pct"] == 0.0


def test_extractor_defaults_to_live_runtime_dimensions():
    runtime_keys = {spec["key"] for spec in RUNTIME_DIMENSIONS}
    assert set(AFFECTIVE_DIMENSIONS) == runtime_keys
    assert {"valence_positive", "arousal", "curiosity", "frustration", "energy"} <= set(
        AFFECTIVE_DIMENSIONS
    )
    assert "confidence" in ALL_AFFECTIVE_DIMENSIONS
    assert "warmth" in ALL_AFFECTIVE_DIMENSIONS


def test_readiness_uses_runtime_contract_and_ignores_stale_nonruntime_vectors(tmp_path):
    vdir = tmp_path / "vectors"
    fdir = tmp_path / "fused-model"
    active = fdir / "active-model"
    vdir.mkdir()
    descriptor = _model(active, revision="active")
    _activate(fdir, active, descriptor)
    active_digest = str(descriptor["descriptor_sha256"])

    for key in {spec["key"] for spec in RUNTIME_DIMENSIONS}:
        for layer in (25, 30, 35):
            _vec(
                vdir / f"{key}_layer{layer}.npz",
                source="extracted_caa",
                extracted=True,
                model_path=str(active),
                model_descriptor_sha256=active_digest,
            )

    # This file is real directory drift from older derivation attempts; it
    # should be surfaced as ignored drift, not reduce production readiness.
    _vec(vdir / "warmth_layer99.npz", source="runtime_derived_caa", extracted=False)

    r = verify_readiness(vectors_dir=vdir, fused_model_dir=fdir)
    assert r["level"] == "production"
    assert r["below_design_capacity"] is False
    assert r["runtime_contract"]["expected_total"] == 15
    assert r["runtime_contract"]["expected_extracted"] == 15
    assert r["runtime_contract"]["ignored_file_count"] == 1
    assert r["runtime_contract"]["active_model_descriptor_sha256"] == active_digest


def test_extracted_vectors_from_previous_active_model_do_not_count_as_production(tmp_path):
    vdir = tmp_path / "vectors"
    fdir = tmp_path / "fused-model"
    active = fdir / "active-model"
    previous = fdir / "previous-model"
    vdir.mkdir()
    active_descriptor = _model(active, revision="new")
    previous_descriptor = _model(previous, revision="old")
    _activate(fdir, active, active_descriptor, fused_at=600.0)
    previous_digest = str(previous_descriptor["descriptor_sha256"])

    for key in {spec["key"] for spec in RUNTIME_DIMENSIONS}:
        for layer in (25, 30, 35):
            _vec(
                vdir / f"{key}_layer{layer}.npz",
                source="extracted_caa",
                extracted=True,
                model_path=str(previous),
                model_descriptor_sha256=previous_digest,
            )

    r = verify_readiness(vectors_dir=vdir, fused_model_dir=fdir)
    assert r["level"] == "bootstrap"
    assert r["below_design_capacity"] is True
    assert r["runtime_contract"]["expected_total"] == 15
    assert r["runtime_contract"]["expected_extracted"] == 0
    assert r["runtime_contract"]["expected_extracted_unbound"] == 15
