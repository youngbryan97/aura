from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from core.brain.llm.model_artifact_profile import (
    MODEL_ARTIFACT_DESCRIPTOR_SCHEMA,
    SERVING_PROFILE_SCHEMA,
    build_model_artifact_descriptor,
    build_model_serving_profile,
    get_model_artifact_profile,
    reset_model_artifact_profile_cache,
    validate_model_artifact_descriptor,
    validate_model_serving_profile,
)
from core.learning.cortex_generation_upgrade import (
    build_migration_contract,
    compare_batteries,
    normalize_active_pointer_identity,
)
from tests.support.cortex_migration_authority import build_signed_migration_authorities


@pytest.fixture(autouse=True)
def _isolated_state_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "state"))


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _artifact(tmp_path: Path, *, name: str = "Qwen3.8-27B-4bit") -> Path:
    root = tmp_path / name
    root.mkdir()
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16,
        },
        "quantization": {"bits": 4, "group_size": 64},
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    weights = b"candidate-weights"
    (root / "model.safetensors").write_bytes(weights)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_parameters": 27_000_000_000,
                    "total_size": len(weights),
                },
                "weight_map": {"model.embed_tokens.weight": "model.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    return root


def _qualification(
    descriptor_sha256: str,
    *,
    served_context_tokens: int = 262144,
    prefill_chunk_tokens: int = 2048,
) -> dict[str, object]:
    return {
        "schema": "aura.model_serving_qualification.v2",
        "verdict": "PASS",
        "model_descriptor_sha256": descriptor_sha256,
        "template_pass": True,
        "complete_answer_pass": True,
        "tool_contract_pass": True,
        "code_contract_pass": True,
        "context_pass": True,
        "latency_pass": True,
        "memory_pass": True,
        "served_context_tokens": served_context_tokens,
        "requested_context_tokens": served_context_tokens,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "evidence_sha256": _sha("serving-evidence"),
    }


def _limits() -> dict[str, dict[str, int]]:
    return {
        "foreground_simple": {"max_input_tokens": 32768, "max_output_tokens": 1024},
        "foreground_standard": {"max_input_tokens": 65536, "max_output_tokens": 3072},
        "foreground_extended": {"max_input_tokens": 131072, "max_output_tokens": 8192},
        "deep_reasoning": {"max_input_tokens": 131072, "max_output_tokens": 16384},
        "tool_execution": {"max_input_tokens": 65536, "max_output_tokens": 8192},
        "code": {"max_input_tokens": 131072, "max_output_tokens": 12288},
        "document": {"max_input_tokens": 196608, "max_output_tokens": 12288},
    }


def test_qwen35_profile_reads_the_text_architecture_not_just_width(tmp_path):
    artifact = _artifact(tmp_path)
    reset_model_artifact_profile_cache()

    profile = get_model_artifact_profile(str(artifact))

    assert profile.model_type == "qwen3_5_text"
    assert profile.architectures == ("Qwen3_5ForConditionalGeneration",)
    assert profile.hidden_size == 5120
    assert profile.num_hidden_layers == 64
    assert profile.vocab_size == 248320
    assert profile.native_context_window == 262144
    assert profile.linear_attention_layers == 48
    assert profile.full_attention_layers == 16


def test_descriptor_binds_full_weights_and_behavior_bytes(tmp_path):
    artifact = _artifact(tmp_path)
    first = build_model_artifact_descriptor(
        artifact,
        repository_id="mlx-community/Qwen3.8-27B-4bit",
        revision="3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
    )

    assert first["schema"] == MODEL_ARTIFACT_DESCRIPTOR_SCHEMA
    assert first["weight_identity"]["method"] == "sha256"
    assert len(first["descriptor_sha256"]) == 64
    validate_model_artifact_descriptor(first, model_path=artifact, verify_full_hash=True)

    (artifact / "model.safetensors").write_bytes(b"candidate-weightS")
    with pytest.raises(ValueError, match="descriptor_mismatch"):
        validate_model_artifact_descriptor(first, model_path=artifact, verify_full_hash=True)


def test_serving_profile_is_bound_to_model_and_measured_gates(tmp_path):
    artifact = _artifact(tmp_path)
    descriptor = build_model_artifact_descriptor(artifact)
    serving = build_model_serving_profile(
        descriptor,
        served_context_tokens=262144,
        prefill_chunk_tokens=2048,
        lane_limits=_limits(),
        qualification=_qualification(str(descriptor["descriptor_sha256"])),
    )

    assert serving["schema"] == SERVING_PROFILE_SCHEMA
    assert serving["model_descriptor_sha256"] == descriptor["descriptor_sha256"]
    assert serving["lanes"]["deep_reasoning"]["max_output_tokens"] == 16384
    validate_model_serving_profile(serving, descriptor)

    wrong = dict(descriptor)
    wrong["descriptor_sha256"] = _sha("other-model")
    with pytest.raises(ValueError, match="model_identity_mismatch"):
        validate_model_serving_profile(serving, wrong)


def test_serving_profile_rejects_a_qualification_from_another_model(tmp_path):
    first = _artifact(tmp_path, name="first")
    second = _artifact(tmp_path, name="second")
    (second / "model.safetensors").write_bytes(b"different-candidate-weights")
    first_descriptor = build_model_artifact_descriptor(first)
    second_descriptor = build_model_artifact_descriptor(second)

    with pytest.raises(ValueError, match="qualification_incomplete"):
        build_model_serving_profile(
            second_descriptor,
            served_context_tokens=262144,
            prefill_chunk_tokens=2048,
            lane_limits=_limits(),
            qualification=_qualification(
                str(first_descriptor["descriptor_sha256"]),
            ),
        )


def test_serving_profile_rejects_context_overcommit_and_unmeasured_expansion(tmp_path):
    artifact = _artifact(tmp_path)
    descriptor = build_model_artifact_descriptor(artifact)
    limits = _limits()
    limits["document"] = {
        "max_input_tokens": 260000,
        "max_output_tokens": 8192,
    }
    with pytest.raises(ValueError, match="context_overcommit"):
        build_model_serving_profile(
            descriptor,
            served_context_tokens=262144,
            prefill_chunk_tokens=2048,
            lane_limits=limits,
            qualification=_qualification(str(descriptor["descriptor_sha256"])),
        )

    failed = _qualification(str(descriptor["descriptor_sha256"]))
    failed["complete_answer_pass"] = False
    with pytest.raises(ValueError, match="qualification_incomplete"):
        build_model_serving_profile(
            descriptor,
            served_context_tokens=262144,
            prefill_chunk_tokens=2048,
            lane_limits=_limits(),
            qualification=failed,
        )


def test_active_manifest_exposes_identity_only_for_its_exact_model(tmp_path, monkeypatch):
    from core.brain.llm import model_registry

    artifact = _artifact(tmp_path)
    descriptor = build_model_artifact_descriptor(artifact)
    promotion_root = tmp_path / "promotion"
    promotion_root.mkdir()
    (promotion_root / "active.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "active_model_path": str(artifact),
                "artifact_descriptor": descriptor,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_registry, "get_fused_model_root", lambda: promotion_root)
    model_registry.reset_model_registry_caches_for_test()

    observed = model_registry.get_active_model_artifact_descriptor(artifact)
    assert observed == descriptor
    spec = model_registry.get_active_cortex_spec(force_refresh=True)
    assert spec is not None
    assert spec.exact_identity is True
    assert spec.promotion_qualified is False
    assert spec.model_path == artifact.resolve()
    assert spec.descriptor_sha256 == descriptor["descriptor_sha256"]

    legacy_limits = model_registry.get_active_cortex_serving_limits(artifact)
    assert legacy_limits is not None
    assert legacy_limits.qualified is False
    assert legacy_limits.source == "legacy_unqualified"
    assert legacy_limits.prefill_chunk_tokens == 0
    assert legacy_limits.lanes == ()

    observed["repository_id"] = "mutated-by-caller"
    assert spec.artifact_descriptor() == descriptor

    other = _artifact(tmp_path, name="same-width-other-model")
    (other / "model.safetensors").write_bytes(b"different-weights")
    resolution = model_registry.resolve_cortex_bound_artifact(other)
    assert resolution.status == "non_cortex_model"
    assert resolution.model_path == other.resolve()
    assert resolution.descriptor is None
    assert model_registry.get_active_model_artifact_descriptor(other) is None


def test_legacy_active_pointer_resolves_without_claiming_exact_identity(
    tmp_path,
    monkeypatch,
):
    from core.brain.llm import model_registry

    artifact = _artifact(tmp_path)
    promotion_root = tmp_path / "promotion"
    promotion_root.mkdir()
    (promotion_root / "active.json").write_text(
        json.dumps({"active_model_path": str(artifact)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_registry, "get_fused_model_root", lambda: promotion_root)
    model_registry.reset_model_registry_caches_for_test()

    spec = model_registry.get_active_cortex_spec(force_refresh=True)

    assert spec is not None
    assert spec.model_path == artifact.resolve()
    assert spec.exact_identity is False
    assert spec.promotion_qualified is False
    assert model_registry.get_active_model_artifact_descriptor(artifact) is None


def test_identity_transition_refuses_any_change_beyond_exact_normalization(
    tmp_path,
    monkeypatch,
):
    from core.brain.llm import model_registry

    artifact = _artifact(tmp_path)
    promotion_root = tmp_path / "promotion"
    promotion_root.mkdir()
    (promotion_root / "active.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "active_model_path": str(artifact),
                "base_model": "incumbent",
                "tag": "incumbent",
            }
        ),
        encoding="utf-8",
    )
    descriptor = build_model_artifact_descriptor(artifact)
    normalize_active_pointer_identity(
        artifact_descriptor=descriptor,
        fused_model_dir=promotion_root,
    )
    monkeypatch.setattr(model_registry, "get_fused_model_root", lambda: promotion_root)
    model_registry.reset_model_registry_caches_for_test()

    spec = model_registry.get_active_cortex_spec(force_refresh=True)
    assert spec is not None
    assert spec.identity_transition_verified is True

    pointer = json.loads((promotion_root / "active.json").read_text())
    pointer["unqualified_runtime_override"] = True
    (promotion_root / "active.json").write_text(json.dumps(pointer), encoding="utf-8")
    model_registry.reset_model_registry_caches_for_test()

    assert model_registry.get_active_cortex_spec(force_refresh=True) is None


def test_active_pointer_rejects_partial_promotion_contract(tmp_path, monkeypatch):
    from core.brain.llm import model_registry

    artifact = _artifact(tmp_path)
    descriptor = build_model_artifact_descriptor(artifact)
    promotion_root = tmp_path / "promotion"
    promotion_root.mkdir()
    (promotion_root / "active.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "active_model_path": str(artifact),
                "artifact_descriptor": descriptor,
                "serving_profile": {"schema": SERVING_PROFILE_SCHEMA},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_registry, "get_fused_model_root", lambda: promotion_root)
    model_registry.reset_model_registry_caches_for_test()

    assert model_registry.get_active_cortex_spec(force_refresh=True) is None
    assert model_registry._resolve_active_fused_model() is None


def test_active_pointer_rejects_malformed_complete_promotion_contract(
    tmp_path,
    monkeypatch,
):
    from core.brain.llm import model_registry

    artifact = _artifact(tmp_path)
    descriptor = build_model_artifact_descriptor(artifact)
    promotion_root = tmp_path / "promotion"
    promotion_root.mkdir()
    (promotion_root / "active.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "active_model_path": str(artifact),
                "artifact_descriptor": descriptor,
                "serving_profile": "not-an-object",
                "migration_contract": {},
                "evaluation": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_registry, "get_fused_model_root", lambda: promotion_root)
    model_registry.reset_model_registry_caches_for_test()

    assert model_registry.get_active_cortex_spec(force_refresh=True) is None


def test_active_pointer_accepts_one_complete_identity_bound_promotion(
    tmp_path,
    monkeypatch,
):
    from core.brain.llm import model_registry

    artifact = _artifact(tmp_path)
    descriptor = build_model_artifact_descriptor(artifact)
    evaluation = compare_batteries(
        {
            "label": "incumbent",
            "breadth_accuracy": 1.0,
            "reasoning_accuracy": 0.4,
            "identity_digests": ["incumbent"],
        },
        {
            "label": "candidate",
            "breadth_accuracy": 1.0,
            "reasoning_accuracy": 1.0,
            "identity_digests": ["candidate"],
        },
        candidate_descriptor=descriptor,
            critical_gates={
                "template": True,
                "complete_answer": True,
                "tool_contract": True,
                "code_contract": True,
                "context": True,
            "identity_migration": True,
            "latency": True,
            "memory": True,
        },
    )
    serving = build_model_serving_profile(
        descriptor,
        served_context_tokens=262144,
        prefill_chunk_tokens=2048,
        lane_limits=_limits(),
        qualification=_qualification(str(descriptor["descriptor_sha256"])),
    )
    migration = build_migration_contract(
        descriptor,
        components=build_signed_migration_authorities(
            tmp_path,
            descriptor_sha256=descriptor["descriptor_sha256"],
            state_root=Path(os.environ["AURA_STATE_ROOT"]),
        ),
    )
    promotion_root = tmp_path / "promotion"
    promotion_root.mkdir()
    (promotion_root / "active.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "active_model_path": str(artifact),
                "artifact_descriptor": descriptor,
                "serving_profile": serving,
                "migration_contract": migration,
                "evaluation": evaluation,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_registry, "get_fused_model_root", lambda: promotion_root)
    model_registry.reset_model_registry_caches_for_test()

    spec = model_registry.get_active_cortex_spec(force_refresh=True)

    assert spec is not None
    assert spec.exact_identity is True
    assert spec.promotion_qualified is True
    assert spec.serving_profile() == serving
    assert spec.migration_contract() == migration

    limits = model_registry.get_active_cortex_serving_limits(artifact)
    assert limits is not None
    assert limits.qualified is True
    assert limits.source == "qualified_profile"
    assert limits.model_path == artifact.resolve()
    assert limits.descriptor_sha256 == descriptor["descriptor_sha256"]
    assert limits.profile_sha256 == serving["profile_sha256"]
    assert limits.served_context_tokens == 262144
    assert limits.prefill_chunk_tokens == 2048
    assert {
        lane.name: (lane.max_input_tokens, lane.max_output_tokens)
        for lane in limits.lanes
    } == {
        name: (value["max_input_tokens"], value["max_output_tokens"])
        for name, value in _limits().items()
    }

    other = _artifact(tmp_path, name="same-width-unqualified-model")
    assert model_registry.get_active_cortex_serving_limits(other) is None
