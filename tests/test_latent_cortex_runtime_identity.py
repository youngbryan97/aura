from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

from core.brain.llm.latent_cortex import runtime_identity
from core.brain.llm.latent_cortex.worker_capture_identity import (
    build_worker_capture_identity,
)
from tests.fixtures.rlc_runtime_integrity import complete_serving_stack


def _source_identity():
    return {
        "source_root": "/repo",
        "commit_sha": "a" * 40,
        "branch": "main",
        "workspace_state_sha256": "b" * 64,
        "source_dirty": False,
        "source_change_count": 0,
        "shell_assets_sha256": "c" * 64,
    }


def _worker_identity():
    capture_identity = build_worker_capture_identity(
        worker_boot_id="d" * 32,
        worker_pid=1234,
    )
    return {
        "schema": runtime_identity.WORKER_IDENTITY_SCHEMA,
        "worker_boot_id": "d" * 32,
        "worker_pid": 1234,
        "worker_model_path": "/models/Aura-32B",
        "worker_model_parameter_count": 32_000_000_000,
        "worker_model_stored_parameter_element_count": 5_000_000_000,
        "worker_model_parameter_count_basis": "architecture_config_logical",
        "worker_source_sha256": "e" * 64,
        "worker_affective_steering_active": True,
        "worker_affective_steering_alpha": 0.30,
        "worker_action_capture_identity": capture_identity.public_identity,
        "worker_recurrent_adapter_activation": (
            runtime_identity.inactive_worker_recurrent_adapter_activation()
        ),
        **complete_serving_stack(),
    }


def test_worker_identity_rejects_every_mismatch():
    expected = _worker_identity()
    receipt = dict(expected)
    receipt["worker_model_parameter_count"] += 1
    receipt["worker_source_sha256"] = "tampered"

    errors = runtime_identity.worker_identity_errors(receipt, expected=expected)

    assert "worker_model_parameter_count_mismatch" in errors
    assert "invalid_worker_source_sha256" in errors
    assert "worker_source_sha256_mismatch" in errors


def test_current_worker_identity_requires_same_boot_capture_identity():
    missing = _worker_identity()
    missing.pop("worker_action_capture_identity")
    mismatched = _worker_identity()
    mismatched["worker_action_capture_identity"] = build_worker_capture_identity(
        worker_boot_id="c" * 32,
        worker_pid=1234,
    ).public_identity

    assert "invalid_worker_action_capture_identity" in (
        runtime_identity.worker_identity_errors(missing)
    )
    assert "worker_action_capture_identity_mismatch" in (
        runtime_identity.worker_identity_errors(mismatched)
    )


def test_v3_worker_identity_requires_positive_evidence_for_active_adapter():
    identity = _worker_identity()
    activation = dict(identity["worker_recurrent_adapter_activation"])
    activation.update(
        {
            "configured": True,
            "active": True,
            "reason": "certified_gain_proven",
            "receipt_sha256": "a" * 64,
            "activation_sha256": "b" * 64,
            "adapter_composite_identity_sha256": "c" * 64,
            "campaign_name": "resident-32b-role-v6",
            "claim_tier": "PROVEN",
            "verified_verdict": "gain_proven",
            "loaded_projection_count": 24,
        }
    )
    identity["worker_recurrent_adapter_activation"] = activation

    assert runtime_identity.worker_identity_errors(identity) == []

    activation["verified_verdict"] = "gain_preverified"
    assert "worker_recurrent_adapter_positive_evidence_incomplete" in (
        runtime_identity.worker_identity_errors(identity)
    )


def test_model_parameter_count_uses_native_nested_parameter_tree():
    model = SimpleNamespace(
        parameters=lambda: {
            "embed": SimpleNamespace(size=20),
            "layers": [
                {"q": SimpleNamespace(size=12), "k": SimpleNamespace(size=8)},
                (SimpleNamespace(size=5),),
            ],
        }
    )

    assert runtime_identity.model_parameter_count(model) == 45


def test_logical_qwen_parameter_count_uses_architecture_not_packed_elements(tmp_path):
    (tmp_path / "config.json").write_text(
        """{
          "model_type": "qwen2",
          "hidden_size": 5120,
          "intermediate_size": 27648,
          "num_hidden_layers": 64,
          "num_attention_heads": 40,
          "num_key_value_heads": 8,
          "vocab_size": 152064,
          "tie_word_embeddings": false
        }""",
        encoding="utf-8",
    )

    count, basis = runtime_identity.logical_model_parameter_count(
        tmp_path,
        stored_element_count=5_120_300_032,
    )

    assert count == 32_763_876_352
    assert basis == "architecture_config_logical"


def test_worker_identity_accepts_production_steering_coefficient():
    identity = _worker_identity()
    identity["worker_affective_steering_alpha"] = 5.525

    assert runtime_identity.worker_identity_errors(identity) == []


def test_worker_identity_zeroes_an_unattached_steering_coefficient(
    monkeypatch, tmp_path
):
    source = tmp_path / "worker.py"
    source.write_text("# worker\n", encoding="utf-8")
    capture = build_worker_capture_identity(
        worker_boot_id="d" * 32,
        worker_pid=os.getpid(),
    )
    monkeypatch.setattr(runtime_identity, "model_parameter_count", lambda _model: 10)
    monkeypatch.setattr(
        runtime_identity,
        "logical_model_parameter_count",
        lambda _path, *, stored_element_count: (
            stored_element_count,
            "stored_tensor_elements",
        ),
    )
    monkeypatch.setattr(
        runtime_identity,
        "serving_stack_identity",
        lambda *_args, **_kwargs: complete_serving_stack(),
    )

    identity = runtime_identity.build_worker_identity(
        object(),
        model_path=tmp_path,
        worker_boot_id="d" * 32,
        worker_source_path=source,
        worker_action_capture_identity=capture.public_identity,
        affective_steering_active=False,
        affective_steering_alpha=0.2,
    )

    assert identity["worker_affective_steering_active"] is False
    assert identity["worker_affective_steering_alpha"] == 0.0
    assert runtime_identity.worker_identity_errors(identity) == []


def test_worker_identity_rejects_alpha_without_active_steering():
    identity = _worker_identity()
    identity["worker_affective_steering_active"] = False

    assert "inactive_worker_affective_steering_has_alpha" in (
        runtime_identity.worker_identity_errors(identity)
    )


def test_representation_basis_survives_restart_but_not_neural_stack_drift():
    first = _worker_identity()
    restarted = _worker_identity()
    restarted["worker_boot_id"] = "f" * 32
    restarted["worker_pid"] = 5678
    restarted["worker_action_capture_identity"] = build_worker_capture_identity(
        worker_boot_id="f" * 32,
        worker_pid=5678,
    ).public_identity

    expected = runtime_identity.worker_representation_basis(first)
    assert runtime_identity.worker_representation_basis(restarted) == expected
    assert runtime_identity.worker_model_basis(restarted) != (
        runtime_identity.worker_model_basis(first)
    )

    for field, changed in (
        ("worker_source_sha256", "f" * 64),
        ("worker_affective_steering_alpha", 0.31),
        ("worker_adapter_stack_sha256", "f" * 64),
    ):
        drifted = _worker_identity()
        drifted[field] = changed
        assert runtime_identity.worker_representation_basis(drifted) != expected


def test_lora_identity_hashes_adapter_owned_tensors_not_wrapped_base():
    import mlx.core as mx
    from mlx_lm.tuner.lora import LoRALinear

    adapter = LoRALinear(16, 8, r=2)
    first, scope = runtime_identity._module_parameter_identity(adapter)
    assert scope == "adapter_owned_excluding_wrapped_base_v1"

    adapter.linear.weight = adapter.linear.weight + 1.0
    mx.eval(adapter.linear.weight)
    base_changed, _ = runtime_identity._module_parameter_identity(adapter)
    assert base_changed == first

    adapter.lora_a = adapter.lora_a + 1.0
    mx.eval(adapter.lora_a)
    adapter_changed, _ = runtime_identity._module_parameter_identity(adapter)
    assert adapter_changed != first


def test_worker_identity_rejects_parameter_count_basis_contradictions():
    architecture = _worker_identity()
    architecture["worker_model_parameter_count"] = 4_000_000_000
    stored = _worker_identity()
    stored["worker_model_parameter_count_basis"] = "stored_tensor_elements"

    assert "worker_model_parameter_count_basis_contradiction" in (
        runtime_identity.worker_identity_errors(architecture)
    )
    assert "worker_model_parameter_count_basis_contradiction" in (
        runtime_identity.worker_identity_errors(stored)
    )


def test_request_digest_binds_runtime_controls_and_is_order_stable():
    kwargs = {
        "prompt": "reason",
        "messages": None,
        "domain": "desktop_conversation",
        "config": {"decode_top_p": 0.8, "decode_max_tokens": 64},
        "budget": {"wall_clock_s": 30.0, "max_layer_apps": 1000},
        "runtime_controls": {
            "clean_user_surface_recurrent_loops": 2,
            "clean_user_surface_steering_alpha": 0.3,
        },
    }
    first = runtime_identity.latent_request_payload_sha256(**kwargs)
    reordered = runtime_identity.latent_request_payload_sha256(
        **{
            **kwargs,
            "config": {"decode_max_tokens": 64, "decode_top_p": 0.8},
        }
    )
    changed = runtime_identity.latent_request_payload_sha256(
        **{
            **kwargs,
            "runtime_controls": {
                **kwargs["runtime_controls"],
                "clean_user_surface_steering_alpha": 0.31,
            },
        }
    )
    contracted = runtime_identity.latent_request_payload_sha256(
        **kwargs,
        response_contract='{"answer":int}',
    )
    operation_bound = runtime_identity.latent_request_payload_sha256(
        **kwargs,
        operation_authority={"attempt_sha256": "a" * 64},
    )
    action_policy_bound = runtime_identity.latent_request_payload_sha256(
        **kwargs,
        action_policy_evidence={"snapshot_sha256": "b" * 64},
    )
    execution_offer_bound = runtime_identity.latent_request_payload_sha256(
        **kwargs,
        external_execution_offer={"offer_sha256": "c" * 64},
    )
    intervention_bound = runtime_identity.latent_request_payload_sha256(
        **kwargs,
        action_intervention={"intervention_sha256": "d" * 64},
    )

    assert first == reordered
    assert first != changed
    assert first != contracted
    assert first != operation_bound
    assert first != action_policy_bound
    assert first != execution_offer_bound
    assert first != intervention_bound
    assert operation_bound != action_policy_bound
    assert execution_offer_bound != action_policy_bound
    assert intervention_bound != action_policy_bound


def test_direct_runtime_identity_binds_exact_source(monkeypatch):
    from core.runtime import launch_provenance

    monkeypatch.setattr(
        launch_provenance,
        "collect_runtime_launch_provenance",
        lambda *_args, **_kwargs: {
            "required": False,
            "launch_mode": "direct",
            "source_verified": True,
            "source_root": "/repo",
            "issues": [],
        },
    )
    monkeypatch.setattr(
        launch_provenance,
        "collect_source_identity",
        lambda *_args, **_kwargs: _source_identity(),
    )

    receipt = runtime_identity.collect_latent_runtime_identity("/repo")

    assert receipt["identity_bound"] is True
    assert receipt["installed_app_required"] is False
    assert receipt["installed_app_verified"] is False
    assert receipt["source_commit"] == "a" * 40
    assert receipt["workspace_state_sha256"] == "b" * 64


def test_signed_app_runtime_identity_binds_executable_and_manifest(
    monkeypatch,
    tmp_path,
):
    from core.runtime import launch_provenance

    executable = tmp_path / "AuraLauncher"
    manifest_path = tmp_path / "aura-launch-provenance.json"
    executable.write_bytes(b"signed launcher bytes")
    manifest_path.write_bytes(b'{"schema":"aura.launch_provenance.v1"}')
    source = _source_identity()
    manifest = {
        **source,
        "schema": "aura.launch_provenance.v1",
        "bundle_identifier": "com.aura.desktop",
    }
    monkeypatch.setattr(
        launch_provenance,
        "collect_runtime_launch_provenance",
        lambda *_args, **_kwargs: {
            "required": True,
            "verified": True,
            "source_verified": True,
            "launch_mode": "signed_app",
            "actual": source,
            "manifest": manifest,
            "app_executable": str(executable),
            "manifest_path": str(manifest_path),
            "issues": [],
        },
    )

    receipt = runtime_identity.collect_latent_runtime_identity(tmp_path)

    assert receipt["identity_bound"] is True
    assert receipt["installed_app_required"] is True
    assert receipt["installed_app_verified"] is True
    assert receipt["app_executable_sha256"] == hashlib.sha256(executable.read_bytes()).hexdigest()
    assert (
        receipt["launch_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )


def test_signed_app_identity_fails_closed_when_bundle_hash_cannot_be_read(monkeypatch):
    from core.runtime import launch_provenance

    source = _source_identity()
    monkeypatch.setattr(
        launch_provenance,
        "collect_runtime_launch_provenance",
        lambda *_args, **_kwargs: {
            "required": True,
            "verified": True,
            "source_verified": True,
            "launch_mode": "signed_app",
            "actual": source,
            "manifest": {**source, "bundle_identifier": "com.aura.desktop"},
            "app_executable": "/missing/AuraLauncher",
            "manifest_path": "/missing/manifest.json",
            "issues": [],
        },
    )

    receipt = runtime_identity.collect_latent_runtime_identity("/repo")

    assert receipt["identity_bound"] is False
    assert receipt["installed_app_verified"] is False
    assert "installed_app_identity_unbound" in receipt["issues"]
