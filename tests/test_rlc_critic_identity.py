from __future__ import annotations

import copy
import hashlib

import pytest

from core.brain.latent_cortex_service import LatentCortexService
from core.brain.llm.latent_cortex.critic_identity import (
    MIN_CHECKED_SAMPLES,
    MIN_GENERATOR_ERRORS,
    CriticBlindSpotLedger,
    audit_python_dependencies,
    build_critic_identity,
    build_shared_blind_spot_evidence,
    validate_critic_identity,
    validate_shared_blind_spot_evidence,
)
from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier

_WORKER = {
    "schema": "aura.latent_cortex.worker_identity.v1",
    "worker_boot_id": "1" * 32,
    "worker_pid": 4242,
    "worker_model_path": "/models/test-32b",
    "worker_model_parameter_count": 32_000_000_000,
    "worker_model_stored_parameter_element_count": 5_000_000_000,
    "worker_model_parameter_count_basis": "architecture_config_logical",
    "worker_source_sha256": "2" * 64,
    "worker_affective_steering_active": True,
    "worker_affective_steering_alpha": 0.30,
    "worker_adapters": [],
    "worker_adapter_stack_sha256": "3" * 64,
    "worker_tokenizer": {"tokenizer.json": "4" * 64},
    "worker_quantization": {"bits": 4, "group_size": 64},
    "worker_stack_identity_gaps": [],
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity():
    return build_critic_identity(
        EpisodeTaskVerifier("verify the exact answer"),
        worker_identity=_WORKER,
    )


def _rows(identity: dict, *, shared_errors: int) -> list[dict]:
    generator_sha = identity["generator_identity"]["function_sha256"]
    critic_sha = identity["critic_function_sha256"]
    rows = []
    error_count = max(MIN_GENERATOR_ERRORS, shared_errors)
    for index in range(MIN_CHECKED_SAMPLES):
        is_error = index < error_count
        rows.append(
            {
                "schema": "aura.rlc.checked_critic_outcome.v1",
                "bucket": "logic|heldout",
                "checked": True,
                "task_sha256": _digest(f"task-{index}"),
                "candidate_sha256": _digest(f"candidate-{index}"),
                "generator_function_sha256": generator_sha,
                "critic_function_sha256": critic_sha,
                "independent_grader_sha256": _digest(f"grader-{index % 2}"),
                "independent_receipt_sha256": _digest(f"receipt-{index}"),
                "generator_correct": not is_error,
                "critic_accepted": (not is_error) or index < shared_errors,
            }
        )
    return rows


def test_symbolic_critic_identity_is_disjoint_from_neural_generator():
    identity = _identity()
    validated = validate_critic_identity(identity, worker_identity=_WORKER)

    assert validated["function_identity_distinct"] is True
    assert validated["runtime_state_audit"]["trainable_parameter_count"] == 0
    assert validated["generator_identity"]["logical_parameter_count"] == 32_000_000_000
    assert validated["source_identity"]["dependency_audit"]["passed"] is True
    source_paths = {
        row["path"] for row in validated["source_identity"]["source_files"]
    }
    assert "core/brain/canonical_json.py" in source_paths
    assert "core/language/relational_request.py" in source_paths
    assert "core/brain/frontier_evidence_v5.py" not in source_paths
    assert (
        validated["critic_function_sha256"]
        != validated["generator_identity"]["function_sha256"]
    )


def test_source_and_generator_identity_tampering_is_rejected():
    identity = _identity()
    tampered = copy.deepcopy(identity)
    tampered["generator_identity"]["logical_parameter_count"] = 1

    with pytest.raises(ValueError):
        validate_critic_identity(tampered, worker_identity=_WORKER)

    changed_worker = dict(_WORKER, worker_adapter_stack_sha256="a" * 64)
    with pytest.raises(ValueError, match="independently proven"):
        validate_critic_identity(identity, worker_identity=changed_worker)


def test_dependency_audit_rejects_generator_runtime_imports():
    audit = audit_python_dependencies(
        {"critic.py": "import mlx.core\nfrom torch import nn\n"}
    )

    assert audit["passed"] is False
    assert audit["forbidden_imports"] == ["mlx.core", "torch"]


def test_shared_blind_spot_matrix_and_bound_are_reconstructed():
    identity = _identity()
    evidence = build_shared_blind_spot_evidence(
        bucket="logic|heldout",
        generator_function_sha256=identity["generator_identity"]["function_sha256"],
        critic_function_sha256=identity["critic_function_sha256"],
        checked_outcomes=_rows(identity, shared_errors=0),
    )

    assert evidence["evidence_state"] == "measured"
    assert evidence["generator_errors"] >= MIN_GENERATOR_ERRORS
    assert evidence["shared_blind_spot_rate"] == 0.0
    assert evidence["shared_blind_spot_wilson95"]["upper"] < 0.35
    assert evidence["critic_reliability_admitted"] is True
    validate_shared_blind_spot_evidence(
        evidence,
        generator_function_sha256=identity["generator_identity"]["function_sha256"],
        critic_function_sha256=identity["critic_function_sha256"],
    )

    tampered = copy.deepcopy(evidence)
    tampered["confusion_matrix"]["generator_error_critic_accept"] += 1
    with pytest.raises(ValueError, match="digest differs"):
        validate_shared_blind_spot_evidence(
            tampered,
            generator_function_sha256=identity["generator_identity"]["function_sha256"],
            critic_function_sha256=identity["critic_function_sha256"],
        )


def test_unpowered_evidence_is_honest_and_poor_powered_evidence_revokes_authority():
    identity = _identity()
    generator_sha = identity["generator_identity"]["function_sha256"]
    critic_sha = identity["critic_function_sha256"]
    bootstrap = build_shared_blind_spot_evidence(
        bucket="logic|bootstrap",
        generator_function_sha256=generator_sha,
        critic_function_sha256=critic_sha,
        checked_outcomes=[],
    )
    poor = build_shared_blind_spot_evidence(
        bucket="logic|heldout",
        generator_function_sha256=generator_sha,
        critic_function_sha256=critic_sha,
        checked_outcomes=_rows(identity, shared_errors=MIN_GENERATOR_ERRORS),
    )

    assert bootstrap["evidence_state"] == "bootstrap_unmeasured"
    assert bootstrap["shared_blind_spot_rate"] is None
    assert poor["evidence_state"] == "measured"
    assert poor["critic_reliability_admitted"] is False


def test_checked_critic_ledger_persists_and_rejects_replay(tmp_path):
    identity = _identity()
    generator_sha = identity["generator_identity"]["function_sha256"]
    critic_sha = identity["critic_function_sha256"]
    path = tmp_path / "critic.jsonl"
    ledger = CriticBlindSpotLedger(path)
    kwargs = {
        "bucket": "logic|durable",
        "task_sha256": _digest("task"),
        "candidate_sha256": _digest("candidate"),
        "generator_function_sha256": generator_sha,
        "critic_function_sha256": critic_sha,
        "independent_grader_sha256": _digest("grader"),
        "independent_receipt_sha256": _digest("receipt"),
        "generator_correct": False,
        "critic_accepted": False,
    }

    assert ledger.record_checked(**kwargs) is True
    with pytest.raises(ValueError, match="already recorded"):
        ledger.record_checked(**kwargs)
    restored = CriticBlindSpotLedger(path)
    evidence = restored.evidence(
        bucket="logic|durable",
        generator_function_sha256=generator_sha,
        critic_function_sha256=critic_sha,
    )
    assert evidence["checked_samples"] == 1
    assert restored.status()["restore_errors"] == 0


def test_ledger_rejects_generator_as_its_own_independent_grader(tmp_path):
    identity = _identity()
    generator_sha = identity["generator_identity"]["function_sha256"]
    with pytest.raises(ValueError, match="independent checked evidence"):
        CriticBlindSpotLedger(tmp_path / "critic.jsonl").record_checked(
            bucket="logic|heldout",
            task_sha256=_digest("task"),
            candidate_sha256=_digest("candidate"),
            generator_function_sha256=generator_sha,
            critic_function_sha256=identity["critic_function_sha256"],
            independent_grader_sha256=generator_sha,
            independent_receipt_sha256=_digest("receipt"),
            generator_correct=False,
            critic_accepted=True,
        )


def test_service_reconstructs_exact_critic_identity_and_evidence():
    identity = _identity()
    evidence = build_shared_blind_spot_evidence(
        bucket="logic|bootstrap",
        generator_function_sha256=identity["generator_identity"]["function_sha256"],
        critic_function_sha256=identity["critic_function_sha256"],
        checked_outcomes=[],
    )
    receipt = {
        **_WORKER,
        "critic_identity": identity,
        "shared_blind_spots": evidence,
        "verifier_guidance": {"requested": True, "available": True},
    }
    config = {"critic_blind_spot_evidence": evidence}
    errors = LatentCortexService._receipt_contract_errors(
        receipt,
        config,
        expected_worker_identity=_WORKER,
    )
    assert "disjoint_critic_authority_unproven" not in errors

    tampered = copy.deepcopy(receipt)
    tampered["shared_blind_spots"]["critic_reliability_admitted"] = False
    errors = LatentCortexService._receipt_contract_errors(
        tampered,
        config,
        expected_worker_identity=_WORKER,
    )
    assert "disjoint_critic_authority_unproven" in errors


def test_powered_shared_blind_spots_causally_revoke_worker_verifier(monkeypatch):
    from core.brain.llm.latent_cortex import worker_handler
    from core.brain.llm.latent_cortex.types import EpisodeReceipt, LatentReasoningResult

    identity = _identity()
    poor = build_shared_blind_spot_evidence(
        bucket="logic|heldout",
        generator_function_sha256=identity["generator_identity"]["function_sha256"],
        critic_function_sha256=identity["critic_function_sha256"],
        checked_outcomes=_rows(identity, shared_errors=MIN_GENERATOR_ERRORS),
    )
    captured: dict = {}

    class StubEngine:
        def __init__(self, *_args, **_kwargs):
            """Accept and ignore construction args; this stub holds no state."""

        def reason(self, **kwargs):
            captured.update(kwargs)
            return LatentReasoningResult(
                ok=True,
                text="bounded",
                receipt=EpisodeReceipt(episode_id="critic-revocation"),
            )

    class StubTokenizer:
        def encode(self, _text, **_kwargs):
            return [1]

        def decode(self, _ids):
            return "bounded"

    monkeypatch.setattr(worker_handler, "LatentCortexEngine", StubEngine)
    body = worker_handler.handle_latent_reason(
        {
            "prompt": "verify this",
            "domain": "logic",
            "verifier_guidance": True,
            "config": {"critic_blind_spot_evidence": poor},
        },
        model=object(),
        tokenizer=StubTokenizer(),
        model_path="/models/test-32b",
        worker_identity=dict(_WORKER),
    )

    assert captured["verifier"] is None
    assert body["receipt"]["verifier_guidance"]["available"] is False
    assert "shared_blind_spot_upper_bound_exceeded" in body["receipt"][
        "verifier_guidance"
    ]["reason"]
    assert "critic_authority_unproven" in body["receipt"]["honest_flags"]
