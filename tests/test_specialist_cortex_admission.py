from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.architecture_quality.attestation import attest_payload
from core.brain.llm.latent_cortex.exact_paired_grade import (
    ExactPairedObservation,
    grade_exact_paired_comparison,
)
from core.brain.llm.latent_cortex.exact_paired_statistics import Rational
from core.brain.llm.latent_cortex.resource_accounting import (
    ModelComputeProfile,
    ResourceLedger,
    build_information_receipt,
    certify_comparison_accounting,
)
from core.brain.llm.model_artifact_profile import (
    SERVING_QUALIFICATION_SCHEMA,
    build_model_artifact_descriptor,
    build_model_serving_profile,
)
from core.learning.cortex_serving_qualification import recommended_lane_limits
from core.learning.specialist_cortex_admission import (
    CERTIFICATE_SCHEMA,
    COMPARATIVE_SCHEMA,
    HOST_ENVELOPE_SCHEMA,
    INDEPENDENT_VERIFICATION_SCHEMA,
    SpecialistAdmissionError,
    build_source_closure,
    verify_specialist_qualification_certificate,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_model(root: Path) -> dict[str, object]:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 64,
                "max_position_embeddings": 32768,
                "quantization_config": {"bits": 4},
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"specialist-test-weights")
    return build_model_artifact_descriptor(root, repository_id="test/specialist", revision="r1")


def _serving_profile(descriptor: dict[str, object]) -> dict[str, object]:
    qualification = {
        "schema": SERVING_QUALIFICATION_SCHEMA,
        "verdict": "PASS",
        "model_descriptor_sha256": descriptor["descriptor_sha256"],
        "template_pass": True,
        "complete_answer_pass": True,
        "tool_contract_pass": True,
        "code_contract_pass": True,
        "context_pass": True,
        "latency_pass": True,
        "memory_pass": True,
        "served_context_tokens": 16384,
        "requested_context_tokens": 16384,
        "prefill_chunk_tokens": 512,
        "evidence_sha256": "e" * 64,
    }
    return build_model_serving_profile(
        descriptor,
        served_context_tokens=16384,
        prefill_chunk_tokens=512,
        lane_limits=recommended_lane_limits(16384),
        qualification=qualification,
    )


def _accounting() -> dict[str, object]:
    compute_profile = ModelComputeProfile(
        model_type="qwen2",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        head_dim=8,
    )
    treatment = ResourceLedger(profile=compute_profile)
    control = ResourceLedger(profile=compute_profile)
    counters = {
        "transformer_layer_apps": 1000,
        "attention_query_key_pairs": 2000,
        "output_head_tokens": 50,
        "verifier_calls": 30,
        "verifier_input_bytes": 600,
        "verifier_output_bytes": 60,
    }
    treatment.charge("generation", **counters)
    control.charge("generation", **counters)
    information = build_information_receipt(
        sources=[
            {
                "source_id": "held-out-tasks",
                "kind": "task_manifest",
                "content_sha256": "1" * 64,
                "byte_count": 1000,
                "token_count": 200,
            }
        ],
        policies={"decode": "2" * 64},
    )
    return certify_comparison_accounting(
        treatment_resource=treatment.to_receipt(),
        control_resource=control.to_receipt(),
        treatment_information=information,
        control_information=information,
    )


def _grade(*, resident_sha: str, specialist_sha: str) -> dict[str, object]:
    observations = {
        domain: [
            ExactPairedObservation(
                task_id=f"{domain}-{index:03d}",
                family=domain,
                treatment_success=True,
                control_success=False,
                treatment_compute=100,
                control_compute=100,
            )
            for index in range(30)
        ]
        for domain in ("coding", "mathematics")
    }
    return grade_exact_paired_comparison(
        experiment="specialist-vs-resident",
        statement="The specialist improves exact held-out results without regression.",
        treatment=specialist_sha,
        control=resident_sha,
        observations_by_family=observations,
        require_compute=True,
        compute_tolerance=Rational(0, 1),
        global_bound_family_count=3,
    )


def _certificate(tmp_path: Path, source_root: Path) -> dict[str, object]:
    resident_sha = "a" * 64
    resident_pointer = "b" * 64
    descriptor = _write_model(tmp_path / "specialist")
    specialist_sha = str(descriptor["descriptor_sha256"])
    grade = _grade(resident_sha=resident_sha, specialist_sha=specialist_sha)
    independent_body = {
        "schema": INDEPENDENT_VERIFICATION_SCHEMA,
        "verdict": "PASS",
        "claim": "specialist_over_resident",
        "subject_identity": specialist_sha,
        "verifier_identity": "independent-verifier",
        "verifier_execution": "separate_process",
        "verifier_code_sha256": "3" * 64,
        "raw_artifact_package_sha256": "4" * 64,
    }
    independent = {
        **independent_body,
        "receipt_sha256": _sha(independent_body),
    }
    accounting = _accounting()
    comparative_body = {
        "schema": COMPARATIVE_SCHEMA,
        "resident_descriptor_sha256": resident_sha,
        "specialist_descriptor_sha256": specialist_sha,
        "task_manifest_sha256": "5" * 64,
        "evaluator_sha256": "6" * 64,
        "grade": grade,
        "grade_sha256": _sha(grade),
        "accounting": accounting,
        "arm_bindings": {
            "treatment": {
                "model_descriptor_sha256": specialist_sha,
                "resource_receipt_sha256": accounting["treatment_resource_sha256"],
                "information_receipt_sha256": accounting[
                    "treatment_information_sha256"
                ],
            },
            "control": {
                "model_descriptor_sha256": resident_sha,
                "resource_receipt_sha256": accounting["control_resource_sha256"],
                "information_receipt_sha256": accounting[
                    "control_information_sha256"
                ],
            },
        },
        "independent_verification": independent,
        "admitted_domains": grade["evidence"]["positive_families"],
    }
    comparative = {
        **comparative_body,
        "comparative_sha256": _sha(comparative_body),
    }
    host_body = {
        "schema": HOST_ENVELOPE_SCHEMA,
        "resident_descriptor_sha256": resident_sha,
        "specialist_descriptor_sha256": specialist_sha,
        "topology": "exclusive_swap",
        "minimum_total_gb": 64.0,
        "minimum_available_gb": 12.0,
        "maximum_peak_gb": 36.0,
        "load_pass": True,
        "cancel_pass": True,
        "unload_pass": True,
        "resident_restore_pass": True,
    }
    host = {**host_body, "evidence_sha256": _sha(host_body)}
    issued = time.time()
    payload = {
        "schema": CERTIFICATE_SCHEMA,
        "issued_at": issued,
        "expires_at": issued + 86400.0,
        "source": build_source_closure(
            source_root,
            commit="c" * 40,
        ),
        "resident": {
            "descriptor_sha256": resident_sha,
            "pointer_sha256": resident_pointer,
        },
        "specialist_descriptor": descriptor,
        "serving_profile": _serving_profile(descriptor),
        "comparative": comparative,
        "host_envelope": host,
    }
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "signing.pem"
    public_path = tmp_path / "trust.pem"
    private_path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    attested = attest_payload(
        payload,
        digest_field="certificate_sha256",
        signing_key_path=private_path,
    )
    certificate_path = tmp_path / "admission.json"
    certificate_path.write_text(json.dumps(attested), encoding="utf-8")
    return {
        "certificate": attested,
        "certificate_path": certificate_path,
        "trust_path": public_path,
        "signing_path": private_path,
        "specialist_path": tmp_path / "specialist",
        "resident_sha": resident_sha,
        "resident_pointer": resident_pointer,
    }


def _verify(bundle: dict[str, object], source_root: Path, **overrides):
    arguments = {
        "trusted_public_key_path": bundle["trust_path"],
        "source_root": source_root,
        "current_source_commit": "c" * 40,
        "resident_descriptor_sha256": bundle["resident_sha"],
        "resident_pointer_sha256": bundle["resident_pointer"],
        "specialist_model_path": bundle["specialist_path"],
        "requested_domain": "coding",
    }
    arguments.update(overrides)
    return verify_specialist_qualification_certificate(
        bundle["certificate_path"],
        **arguments,
    )


def test_valid_certificate_admits_only_measured_domains(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    bundle = _certificate(tmp_path, source_root)

    admitted = _verify(bundle, source_root)
    refused = _verify(bundle, source_root, requested_domain="medical")
    unclassified = _verify(bundle, source_root, requested_domain=None)

    assert admitted.admitted is True
    assert admitted.reason == "qualified"
    assert admitted.admitted_domains == ("coding", "mathematics")
    assert refused.admitted is False
    assert refused.reason == "domain_not_qualified"
    assert unclassified.admitted is False
    assert unclassified.reason == "domain_not_qualified"


def test_resident_identity_drift_denies_admission(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    bundle = _certificate(tmp_path, source_root)

    with pytest.raises(SpecialistAdmissionError, match="resident_binding_stale"):
        _verify(bundle, source_root, resident_descriptor_sha256="d" * 64)


def test_source_revision_drift_denies_admission(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    bundle = _certificate(tmp_path, source_root)

    with pytest.raises(SpecialistAdmissionError, match="source_commit_stale"):
        _verify(bundle, source_root, current_source_commit="d" * 40)


def test_certificate_tamper_denies_before_semantic_admission(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    bundle = _certificate(tmp_path, source_root)
    raw = json.loads(Path(bundle["certificate_path"]).read_text(encoding="utf-8"))
    raw["comparative"]["admitted_domains"] = ["general"]
    Path(bundle["certificate_path"]).write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(SpecialistAdmissionError, match="attestation_invalid"):
        _verify(bundle, source_root)


def test_signed_certificate_cannot_misbind_resource_receipts(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    bundle = _certificate(tmp_path, source_root)
    raw = json.loads(Path(bundle["certificate_path"]).read_text(encoding="utf-8"))
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"certificate_sha256", "signature"}
    }
    payload["comparative"]["arm_bindings"]["treatment"][
        "model_descriptor_sha256"
    ] = "d" * 64
    comparative_body = {
        key: value
        for key, value in payload["comparative"].items()
        if key != "comparative_sha256"
    }
    payload["comparative"]["comparative_sha256"] = _sha(comparative_body)
    resigned = attest_payload(
        payload,
        digest_field="certificate_sha256",
        signing_key_path=Path(bundle["signing_path"]),
    )
    Path(bundle["certificate_path"]).write_text(json.dumps(resigned), encoding="utf-8")

    with pytest.raises(SpecialistAdmissionError, match="arm_binding_invalid"):
        _verify(bundle, source_root)


def test_signed_certificate_cannot_misbind_independent_verifier_subject(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    bundle = _certificate(tmp_path, source_root)
    raw = json.loads(Path(bundle["certificate_path"]).read_text(encoding="utf-8"))
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"certificate_sha256", "signature"}
    }
    verification = payload["comparative"]["independent_verification"]
    verification["subject_identity"] = "d" * 64
    verification_body = {
        key: value for key, value in verification.items() if key != "receipt_sha256"
    }
    verification["receipt_sha256"] = _sha(verification_body)
    comparative_body = {
        key: value
        for key, value in payload["comparative"].items()
        if key != "comparative_sha256"
    }
    payload["comparative"]["comparative_sha256"] = _sha(comparative_body)
    resigned = attest_payload(
        payload,
        digest_field="certificate_sha256",
        signing_key_path=Path(bundle["signing_path"]),
    )
    Path(bundle["certificate_path"]).write_text(json.dumps(resigned), encoding="utf-8")

    with pytest.raises(SpecialistAdmissionError, match="verification_invalid"):
        _verify(bundle, source_root)
