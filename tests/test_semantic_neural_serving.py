from __future__ import annotations

import copy
import hashlib
import json

from core.brain.llm.semantic_neural_serving import (
    DEFAULT_ACTIVATION_PATH,
    semantic_neural_activation_errors,
    semantic_neural_serving_status,
)


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


def test_semantic_serving_kill_switch_is_fail_closed(monkeypatch):
    activation = _activation()
    model_path = activation["model_identity"]["path"]
    monkeypatch.setenv("AURA_SEMANTIC_NEURAL_SERVING", "0")
    status = semantic_neural_serving_status(model_path)
    assert status == {
        "active": False,
        "reason": "semantic_neural_serving_disabled",
    }
