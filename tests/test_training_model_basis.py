from __future__ import annotations

import json

import pytest

from training import model_basis


def _descriptor(digest: str = "a" * 64) -> dict[str, object]:
    return {"schema": "test", "descriptor_sha256": digest}


def _trust_descriptor(monkeypatch):
    import core.brain.llm.model_artifact_profile as profile

    monkeypatch.setattr(
        profile,
        "validate_model_artifact_descriptor",
        lambda descriptor, **_kwargs: descriptor,
    )


def test_active_training_basis_uses_the_promoted_artifact(monkeypatch, tmp_path):
    from core.brain.llm import model_registry

    _trust_descriptor(monkeypatch)
    artifact = tmp_path / "cortex"
    artifact.mkdir()
    descriptor = _descriptor()
    monkeypatch.delenv("AURA_LORA_BASE_MODEL", raising=False)
    monkeypatch.setattr(model_registry, "ACTIVE_MODEL", "Aura-Cortex")
    monkeypatch.setattr(model_registry, "get_runtime_model_path", lambda _name: str(artifact))
    monkeypatch.setattr(
        model_registry,
        "get_active_model_artifact_descriptor",
        lambda path: descriptor if path == artifact else None,
    )

    basis = model_basis.resolve_training_model_basis()

    assert basis.path == artifact
    assert basis.descriptor_sha256 == "a" * 64
    assert basis.source == "active_cortex:active_descriptor"


def test_training_basis_refuses_a_repository_fallback(monkeypatch):
    from core.brain.llm import model_registry

    monkeypatch.delenv("AURA_LORA_BASE_MODEL", raising=False)
    monkeypatch.setattr(model_registry, "ACTIVE_MODEL", "Aura-Cortex")
    monkeypatch.setattr(
        model_registry,
        "get_runtime_model_path",
        lambda _name: "mlx-community/unmaterialized-cortex",
    )

    with pytest.raises(model_basis.TrainingModelBasisError, match="must_be_local"):
        model_basis.resolve_training_model_basis()


def test_recorded_basis_rejects_a_resume_model_change(monkeypatch, tmp_path):
    _trust_descriptor(monkeypatch)
    trained = tmp_path / "trained-model"
    other = tmp_path / "other-model"
    trained.mkdir()
    other.mkdir()
    descriptor = _descriptor()
    config = tmp_path / "training_config.json"
    config.write_text(
        json.dumps(
            {
                "model": str(trained),
                "training_basis": {
                    "schema": model_basis.TRAINING_MODEL_BASIS_SCHEMA,
                    "model_path": str(trained),
                    "descriptor_sha256": "a" * 64,
                    "artifact_descriptor": descriptor,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(model_basis.TrainingModelBasisError, match="basis_change"):
        model_basis.load_recorded_training_model_basis(
            config,
            model_override=other,
            verify_full_hash=False,
        )


def test_legacy_training_config_cannot_claim_an_exact_resume(tmp_path):
    config = tmp_path / "training_config.json"
    config.write_text(json.dumps({"model": "/old/model"}), encoding="utf-8")

    with pytest.raises(model_basis.TrainingModelBasisError, match="model_basis_missing"):
        model_basis.load_recorded_training_model_basis(config)


def test_adapter_cannot_be_fused_on_another_exact_basis(monkeypatch, tmp_path):
    _trust_descriptor(monkeypatch)
    first = tmp_path / "first"
    second = tmp_path / "second"
    adapter = tmp_path / "adapter"
    first.mkdir()
    second.mkdir()
    adapter.mkdir()
    descriptor = _descriptor("a" * 64)
    (adapter / "training_config.json").write_text(
        json.dumps(
            {
                "model": str(first),
                "training_basis": {
                    "schema": model_basis.TRAINING_MODEL_BASIS_SCHEMA,
                    "model_path": str(first),
                    "descriptor_sha256": "a" * 64,
                    "artifact_descriptor": descriptor,
                },
            }
        ),
        encoding="utf-8",
    )
    expected = model_basis.TrainingModelBasis(
        path=second,
        descriptor=_descriptor("b" * 64),
        descriptor_sha256="b" * 64,
        source="test",
    )

    with pytest.raises(model_basis.TrainingModelBasisError, match="basis_change"):
        model_basis.assert_adapter_matches_basis(adapter, expected)
