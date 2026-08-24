from __future__ import annotations

import copy
import hashlib
import json

from core.brain.llm import semantic_neural_serving
from core.brain.llm.semantic_neural_serving import (
    DEFAULT_ACTIVATION_PATH,
    MODEL_BOUND_ADJUDICATION_CLAIM,
    RECOVERY_PACKAGE_CAMPAIGN,
    _activation_claim_boundary,
    _recovery_descriptor,
    semantic_neural_activation_errors,
    semantic_neural_serving_status,
)
from core.learning.recovery_package_identity import DescriptorIdentity, package_id


def _activation():
    return json.loads(DEFAULT_ACTIVATION_PATH.read_text(encoding="utf-8"))


def _reseal(activation):
    body = {key: value for key, value in activation.items() if key != "activation_sha256"}
    activation["activation_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def test_materialized_semantic_activation_reopens_source_and_evidence():
    activation = _activation()
    assert semantic_neural_activation_errors(
        activation,
        verify_live_identity=False,
    ) == []


def test_semantic_activation_rejects_resealed_source_or_evidence_drift():
    activation = _activation()
    tampered = copy.deepcopy(activation)
    relative = next(iter(tampered["source_sha256s"]))
    tampered["source_sha256s"][relative] = "0" * 64
    _reseal(tampered)
    assert any(
        error.startswith("source_drift:")
        for error in semantic_neural_activation_errors(
            tampered,
            verify_live_identity=False,
        )
    )

    tampered = copy.deepcopy(activation)
    tampered["evidence"]["result_path"] = "../outside.json"
    _reseal(tampered)
    assert "evidence_invalid" in semantic_neural_activation_errors(
        tampered,
        verify_live_identity=False,
    )


def test_semantic_activation_rejects_resealed_model_substitution():
    activation = _activation()
    activation["model_identity"] = {
        **activation["model_identity"],
        "config_sha256": "0" * 64,
    }
    _reseal(activation)

    assert "evidence_drift" in semantic_neural_activation_errors(
        activation,
        verify_live_identity=False,
    )


def test_semantic_activation_rejects_resealed_authority_broadening():
    activation = _activation()
    activation["allowed_surface_profiles"].append("free_form")
    activation["promotion_mode"] = "unrestricted"
    activation["claim_boundary"] = "general reasoning is authorized"
    _reseal(activation)

    errors = semantic_neural_activation_errors(
        activation,
        verify_live_identity=False,
    )
    assert "allowed_surface_profiles" in errors
    assert "promotion_mode" in errors
    assert "claim_boundary" in errors


def test_semantic_activation_rejects_resealed_integration_contract_drift():
    activation = _activation()
    key = next(iter(activation["integration_contract_sha256s"]))
    activation["integration_contract_sha256s"][key] = "0" * 64
    _reseal(activation)

    assert any(
        error.startswith("integration_contract_drift:")
        for error in semantic_neural_activation_errors(
            activation,
            verify_live_identity=False,
        )
    )


def test_active_semantic_activation_requires_runtime_qualification():
    activation = _activation()
    activation.pop("runtime_qualification")
    _reseal(activation)

    assert "runtime_qualification" in semantic_neural_activation_errors(
        activation,
        verify_live_identity=False,
    )
    assert "runtime_qualification" not in semantic_neural_activation_errors(
        activation,
        verify_live_identity=False,
        require_runtime_qualification=False,
    )


def test_runtime_qualification_proves_the_full_serving_chain():
    activation = _activation()
    qualification = activation["runtime_qualification"]

    assert qualification["foreground_integration_count"] == 120
    assert qualification["service_integration_count"] == 120


def test_semantic_serving_kill_switch_is_fail_closed(monkeypatch):
    activation = _activation()
    model_path = activation["model_identity"]["path"]
    monkeypatch.setenv("AURA_SEMANTIC_NEURAL_SERVING", "0")
    status = semantic_neural_serving_status(model_path)
    assert status == {
        "active": False,
        "reason": "semantic_neural_serving_disabled",
    }


def test_recovery_activation_identity_is_descriptor_derived():
    expected = DescriptorIdentity(
        path="/models/target",
        config_sha256="a" * 64,
        weights_index_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        model_type="qwen3_5_text",
        num_hidden_layers=64,
        hidden_size=5120,
        vocab_size=248320,
        full_attention_layers=16,
        linear_attention_layers=48,
    )
    descriptor = _recovery_descriptor(expected.as_dict())
    # The fingerprint is part of the signed descriptor, not optional caller prose.
    assert package_id(descriptor, campaign=RECOVERY_PACKAGE_CAMPAIGN).startswith(
        "rlc-27b-recovery-"
    )


def test_model_bound_claim_boundary_does_not_relabel_a_checkpoint_size():
    boundary = _activation_claim_boundary(MODEL_BOUND_ADJUDICATION_CLAIM)
    assert "resident-model" in boundary
    assert "32B" not in boundary
    assert "27B" not in boundary


def test_operational_activation_precedes_but_does_not_replace_history(
    monkeypatch,
    tmp_path,
):
    operational = tmp_path / "semantic-neural-active.json"
    monkeypatch.setattr(semantic_neural_serving, "ACTIVE_ACTIVATION_PATH", operational)
    assert semantic_neural_serving.active_semantic_neural_activation_path() == (
        DEFAULT_ACTIVATION_PATH
    )

    operational.write_text("{}", encoding="utf-8")
    assert semantic_neural_serving.active_semantic_neural_activation_path() == operational


def test_qualification_candidate_does_not_replace_operational_activation(
    monkeypatch,
    tmp_path,
):
    historical = tmp_path / "historical.json"
    operational = tmp_path / "operational.json"
    candidate = tmp_path / "candidate.json"
    historical.write_text("{}", encoding="utf-8")
    operational.write_text("{}", encoding="utf-8")
    candidate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(semantic_neural_serving, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(semantic_neural_serving, "DEFAULT_ACTIVATION_PATH", historical)
    monkeypatch.setattr(semantic_neural_serving, "ACTIVE_ACTIVATION_PATH", operational)
    monkeypatch.setenv(
        "AURA_SEMANTIC_NEURAL_QUALIFICATION_ACTIVATION",
        str(candidate),
    )

    assert semantic_neural_serving.active_semantic_neural_activation_path() == operational

    monkeypatch.setenv("AURA_SEMANTIC_NEURAL_QUALIFICATION_CANDIDATE", "1")
    assert semantic_neural_serving.active_semantic_neural_activation_path() == candidate


def test_qualification_candidate_cannot_escape_or_symlink_the_repository(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(semantic_neural_serving, "REPO_ROOT", root)
    monkeypatch.setenv("AURA_SEMANTIC_NEURAL_QUALIFICATION_CANDIDATE", "1")
    monkeypatch.setenv(
        "AURA_SEMANTIC_NEURAL_QUALIFICATION_ACTIVATION",
        str(outside),
    )

    try:
        semantic_neural_serving.active_semantic_neural_activation_path()
    except RuntimeError as exc:
        assert "outside the repository" in str(exc)
    else:
        raise AssertionError("qualification accepted an activation outside the repository")

    linked = root / "linked.json"
    linked.symlink_to(outside)
    monkeypatch.setenv(
        "AURA_SEMANTIC_NEURAL_QUALIFICATION_ACTIVATION",
        str(linked),
    )
    try:
        semantic_neural_serving.active_semantic_neural_activation_path()
    except RuntimeError as exc:
        assert "cannot be a symlink" in str(exc)
    else:
        raise AssertionError("qualification accepted a symlink activation")
