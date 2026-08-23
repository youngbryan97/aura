"""Model-identity contracts for CAA extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.extract_steering_vectors import (
    SteeringExtractionContractError,
    _load_model_identity,
    _publish_vector_generation,
    _representation_identity,
    _reserve_output_generation,
    _resolve_extraction_adapter,
)


def _adapter(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapters.safetensors").write_bytes(b"adapter")
    return path


def _fused_model(path: Path) -> Path:
    path.mkdir()
    (path / "aura_fusion_provenance.json").write_text(
        json.dumps(
            {
                "schema": "aura.candidate_cortex_fusion.provenance.v1",
                "representation_boundary": (
                    "fused weights define a new model identity; prior steering and "
                    "recurrent tensors are not representation-compatible"
                ),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_no_adapter_is_implicit_for_an_unfused_model(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _adapter(tmp_path / "training" / "adapters" / "aura-personality")

    assert _resolve_extraction_adapter(model, None) is None


def test_explicit_adapter_is_canonicalized_for_unfused_model(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    adapter = _adapter(tmp_path / "adapter")

    assert _resolve_extraction_adapter(model, str(adapter)) == str(adapter.resolve())


def test_fused_model_refuses_adapter_stacking(tmp_path: Path) -> None:
    model = _fused_model(tmp_path / "fused")
    adapter = _adapter(tmp_path / "adapter")

    with pytest.raises(
        SteeringExtractionContractError,
        match="adapter_stacking_on_fused_model",
    ):
        _resolve_extraction_adapter(model, str(adapter))


def test_malformed_fusion_provenance_fails_closed(tmp_path: Path) -> None:
    model = tmp_path / "fused"
    model.mkdir()
    (model / "aura_fusion_provenance.json").write_text(
        '{"schema":"wrong"}',
        encoding="utf-8",
    )

    with pytest.raises(
        SteeringExtractionContractError,
        match="fusion_provenance_invalid",
    ):
        _resolve_extraction_adapter(model, None)


@pytest.mark.parametrize("missing", ["adapter_config.json", "adapters.safetensors"])
def test_adapter_requires_complete_local_artifact(
    tmp_path: Path,
    missing: str,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    adapter = _adapter(tmp_path / "adapter")
    (adapter / missing).unlink()

    with pytest.raises(
        SteeringExtractionContractError,
        match="adapter_artifact_invalid",
    ):
        _resolve_extraction_adapter(model, str(adapter))


def test_model_descriptor_is_validated_against_exact_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from core.brain.llm import model_artifact_profile

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema": "aura.model_artifact_descriptor.v1",
                "descriptor_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    def _validate(raw, *, model_path, verify_full_hash):
        observed.update(
            raw=raw,
            model_path=model_path,
            verify_full_hash=verify_full_hash,
        )
        return raw

    monkeypatch.setattr(
        model_artifact_profile,
        "validate_model_artifact_descriptor",
        _validate,
    )

    identity = _load_model_identity(model, descriptor)

    assert identity["model_path"] == str(model.resolve())
    assert identity["model_descriptor_sha256"] == "a" * 64
    assert observed["model_path"] == model.resolve()
    assert observed["verify_full_hash"] is False


def test_invalid_model_descriptor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from core.brain.llm import model_artifact_profile

    model = tmp_path / "model"
    model.mkdir()
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        model_artifact_profile,
        "validate_model_artifact_descriptor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("descriptor_path_mismatch")
        ),
    )

    with pytest.raises(
        SteeringExtractionContractError,
        match="model_descriptor_invalid",
    ):
        _load_model_identity(model, descriptor)


def test_representation_identity_changes_with_adapter_evidence() -> None:
    model = {"model_descriptor_sha256": "a" * 64}
    first = _representation_identity(
        model_identity=model,
        adapter_identity={"files": [{"sha256": "b" * 64}]},
        fusion_provenance=None,
    )
    second = _representation_identity(
        model_identity=model,
        adapter_identity={"files": [{"sha256": "c" * 64}]},
        fusion_provenance=None,
    )

    assert first["representation_sha256"] != second["representation_sha256"]


def test_output_generation_is_reserved_once(tmp_path: Path) -> None:
    output = tmp_path / "vectors-generation"

    reservation = _reserve_output_generation(
        output,
        representation_sha256="a" * 64,
        extraction_contract_sha256="b" * 64,
    )

    assert b"aura.caa.extraction_reservation.v1" in reservation
    with pytest.raises(
        SteeringExtractionContractError,
        match="output_generation_already_reserved",
    ):
        _reserve_output_generation(
            output,
            representation_sha256="a" * 64,
            extraction_contract_sha256="b" * 64,
        )


def test_output_generation_refuses_stale_files(tmp_path: Path) -> None:
    output = tmp_path / "vectors-generation"
    output.mkdir()
    (output / "old-vector.npz").write_bytes(b"old")

    with pytest.raises(
        SteeringExtractionContractError,
        match="output_generation_not_empty",
    ):
        _reserve_output_generation(
            output,
            representation_sha256="a" * 64,
            extraction_contract_sha256="b" * 64,
        )


def test_vector_generation_publishes_with_metadata_last(tmp_path: Path) -> None:
    output = tmp_path / "vectors-generation"
    reservation = _reserve_output_generation(
        output,
        representation_sha256="a" * 64,
        extraction_contract_sha256="b" * 64,
    )

    _publish_vector_generation(
        output,
        vector_payloads={"warmth_layer1.npz": b"vector"},
        metadata={
            "method": "contrastive_activation_addition",
            "representation_identity": {"representation_sha256": "a" * 64},
        },
        reservation=reservation,
    )

    assert (output / "warmth_layer1.npz").read_bytes() == b"vector"
    metadata = json.loads(
        (output / "caa_steering_meta.json").read_text(encoding="ascii")
    )
    assert metadata["representation_identity"]["representation_sha256"] == "a" * 64
