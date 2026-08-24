from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.llm.model_bound_steering import (
    ModelBoundSteeringError,
    SteeringGenerationResolution,
    materialize_qualified_generation,
    resolve_active_generation,
)
from tests.support.cortex_migration_authority import (
    build_signed_migration_authorities,
)


def test_qualified_authority_materializes_exact_vector_bytes(tmp_path):
    descriptor_sha256 = "a" * 64
    private_state = tmp_path / "state"
    authorities = build_signed_migration_authorities(
        tmp_path / "fixture",
        descriptor_sha256=descriptor_sha256,
        state_root=private_state,
    )
    key_path = private_state / "private/cortex-upgrade/migration-authority.key"

    materialized = materialize_qualified_generation(
        authorities["steering"],
        descriptor_sha256=descriptor_sha256,
        model_cache_root=tmp_path / "runtime-cache",
        authority_key_path=key_path,
    )

    assert (materialized / "caa-vector.safetensors").read_bytes() == b"test-vector-basis"
    assert (materialized / "caa_steering_meta.json").is_file()
    assert (
        materialize_qualified_generation(
            authorities["steering"],
            descriptor_sha256=descriptor_sha256,
            model_cache_root=tmp_path / "runtime-cache",
            authority_key_path=key_path,
        )
        == materialized
    )


def test_materializer_refuses_existing_bytes_from_another_generation(tmp_path):
    descriptor_sha256 = "a" * 64
    private_state = tmp_path / "state"
    authorities = build_signed_migration_authorities(
        tmp_path / "fixture",
        descriptor_sha256=descriptor_sha256,
        state_root=private_state,
    )
    key_path = private_state / "private/cortex-upgrade/migration-authority.key"
    materialized = materialize_qualified_generation(
        authorities["steering"],
        descriptor_sha256=descriptor_sha256,
        model_cache_root=tmp_path / "runtime-cache",
        authority_key_path=key_path,
    )
    (materialized / "caa-vector.safetensors").write_bytes(b"drift")

    with pytest.raises(ModelBoundSteeringError, match="materialization_collision"):
        materialize_qualified_generation(
            authorities["steering"],
            descriptor_sha256=descriptor_sha256,
            model_cache_root=tmp_path / "runtime-cache",
            authority_key_path=key_path,
        )


def test_active_generation_preserves_signed_deferral(monkeypatch, tmp_path):
    descriptor_sha256 = "b" * 64
    authority = {"status": "deferred"}
    spec = SimpleNamespace(
        descriptor_sha256=descriptor_sha256,
        migration_contract=lambda: {"components": {"steering": authority}},
    )
    monkeypatch.setattr(
        "core.brain.llm.model_registry.get_active_cortex_spec",
        lambda **_kwargs: spec,
    )
    validated: list[dict] = []
    monkeypatch.setattr(
        "core.brain.llm.model_bound_steering.validate_component_authority",
        lambda value, **_kwargs: validated.append(value) or value,
    )

    resolution = resolve_active_generation(
        descriptor_sha256=descriptor_sha256,
        model_cache_root=tmp_path,
    )

    assert resolution == SteeringGenerationResolution(
        "deferred",
        reason="steering_generation_deferred",
    )
    assert validated == [authority]


def test_deferred_generation_cannot_attach_or_derive(monkeypatch, tmp_path):
    from core.consciousness.affective_steering import AffectiveSteeringEngine

    descriptor_sha256 = "c" * 64
    identity = {
        "descriptor_sha256": descriptor_sha256,
        "artifact_profile": {"num_hidden_layers": 2, "hidden_size": 513},
    }
    monkeypatch.setattr(
        "core.brain.llm.model_artifact_profile.validate_model_artifact_descriptor",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        "core.brain.llm.model_bound_steering.resolve_active_generation",
        lambda **_kwargs: SteeringGenerationResolution(
            "deferred",
            reason="steering_generation_deferred",
        ),
    )
    engine = AffectiveSteeringEngine()

    assert (
        engine.attach(
            SimpleNamespace(layers=[object(), object()]),
            SimpleNamespace(),
            model_path=tmp_path,
            model_identity=identity,
        )
        is False
    )
    assert engine.get_status()["model_info"]["attachment_error"] == ("steering_generation_deferred")
    assert engine.get_status()["vector_count"] == 0
