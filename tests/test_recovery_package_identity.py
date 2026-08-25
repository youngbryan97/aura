"""A verdict must not be able to migrate between checkpoints.

The tempting repair for a fail-closed package is to re-seal it against the new
model: every hash recomputes, the alarm goes green, and a claim measured on one
checkpoint serves on another. These tests hold the rules that make that
specific move impossible rather than merely discouraged.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.learning.recovery_package_identity import (
    LEGACY_EVIDENCE_ROOT,
    LEGACY_PACKAGE_ID,
    RECOVERY_CERTIFICATE_SCHEMA,
    RECOVERY_EVIDENCE_ROOT,
    DescriptorIdentity,
    RecoveryPackageError,
    build_package,
    canonical_sha256,
    certificate_errors,
    descriptor_from_manifest,
    evidence_namespace_errors,
    inherited_verdict_errors,
    package_id,
)

#: The checkout this test is running from. Derived rather than written
#: down: a literal install path reads another checkout's artifacts when
#: the suite runs in a worktree, and names one machine's account.
INSTALL = Path(__file__).resolve().parents[1]


def _descriptor(**overrides) -> DescriptorIdentity:
    base = {
        "path": "/models/target",
        "config_sha256": "a" * 64,
        "weights_index_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "vocab_size": 248320,
        "full_attention_layers": 16,
        "linear_attention_layers": 48,
    }
    base.update(overrides)
    return DescriptorIdentity(**base)


def _manifest() -> dict:
    return {
        "active_model_path": "/models/target",
        "artifact_descriptor": {
            "artifact_profile": {
                "path": "/models/target",
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "vocab_size": 248320,
                "full_attention_layers": 16,
                "linear_attention_layers": 48,
            },
            "behavior_identity": {
                "files": [
                    {"path": "config.json", "sha256": "a" * 64},
                    {
                        "path": "model.safetensors.index.json",
                        "sha256": "b" * 64,
                    },
                    {"path": "tokenizer.json", "sha256": "c" * 64},
                ]
            },
        },
    }


def test_the_id_is_derived_from_the_checkpoint_not_chosen():
    one = package_id(_descriptor(), campaign="rlc-27b-recovery")
    two = package_id(_descriptor(config_sha256="d" * 64), campaign="rlc-27b-recovery")
    assert one != two
    assert one.startswith("rlc-27b-recovery-")


def test_manifest_descriptor_uses_exact_signed_behavior_file_hashes():
    descriptor = descriptor_from_manifest(_manifest())

    assert descriptor.config_sha256 == "a" * 64
    assert descriptor.weights_index_sha256 == "b" * 64
    assert descriptor.tokenizer_sha256 == "c" * 64


@pytest.mark.parametrize(
    "missing",
    ("config.json", "model.safetensors.index.json", "tokenizer.json"),
)
def test_manifest_descriptor_refuses_a_missing_behavior_file(missing):
    manifest = _manifest()
    files = manifest["artifact_descriptor"]["behavior_identity"]["files"]
    manifest["artifact_descriptor"]["behavior_identity"]["files"] = [
        record for record in files if record["path"] != missing
    ]

    with pytest.raises(RecoveryPackageError, match="exactly once"):
        descriptor_from_manifest(manifest)


def test_manifest_descriptor_refuses_duplicate_or_malformed_hashes():
    manifest = _manifest()
    files = manifest["artifact_descriptor"]["behavior_identity"]["files"]
    files.append(dict(files[0]))
    with pytest.raises(RecoveryPackageError, match="exactly once"):
        descriptor_from_manifest(manifest)

    manifest = _manifest()
    manifest["artifact_descriptor"]["behavior_identity"]["files"][0][
        "sha256"
    ] = "not-a-hash"
    with pytest.raises(RecoveryPackageError, match="invalid config.json digest"):
        descriptor_from_manifest(manifest)


def test_moving_a_checkpoint_does_not_change_its_identity():
    # A claim that broke when somebody reorganised a models folder would teach
    # people to re-seal packages, which is the habit this file exists to stop.
    here = _descriptor(path="/models/a")
    there = _descriptor(path="/models/b")
    assert here.fingerprint() == there.fingerprint()


def test_every_behaviour_deciding_field_moves_the_fingerprint():
    base = _descriptor().fingerprint()
    for field, value in (
        ("config_sha256", "9" * 64),
        ("weights_index_sha256", "9" * 64),
        ("tokenizer_sha256", "9" * 64),
        ("model_type", "qwen2"),
        ("num_hidden_layers", 63),
        ("hidden_size", 4096),
        ("vocab_size", 152064),
        ("full_attention_layers", 64),
        ("linear_attention_layers", 0),
    ):
        assert _descriptor(**{field: value}).fingerprint() != base, field


def test_the_legacy_package_id_cannot_be_produced():
    # No descriptor yields the CP568 id, because the id is a function of the
    # checkpoint and CP568's checkpoint is not this one.
    assert package_id(_descriptor(), campaign="cp568") != LEGACY_PACKAGE_ID


def test_a_package_reusing_the_legacy_id_is_refused():
    errors = inherited_verdict_errors({"package_id": LEGACY_PACKAGE_ID})
    assert "package_reuses_the_legacy_id" in errors


def test_a_package_granting_the_legacy_claim_authority_is_refused():
    errors = inherited_verdict_errors(
        {
            "package_id": "fresh",
            "legacy_claim": {"authority_over_this_package": "supporting"},
        }
    )
    assert "legacy_claim_is_granted_authority" in errors


def test_a_package_that_does_not_name_what_it_supersedes_is_refused():
    errors = inherited_verdict_errors({"package_id": "fresh"})
    assert "package_does_not_name_the_legacy_claim_it_supersedes" in errors


def test_a_verdict_without_fresh_evidence_is_refused():
    errors = inherited_verdict_errors(
        {
            "package_id": "fresh",
            "legacy_claim": {"authority_over_this_package": "none"},
            "verdict": "BOUNDED_WOW_SIGNAL",
            "evidence": {"measured_on_this_checkpoint": False},
        }
    )
    assert "verdict_without_evidence_measured_on_this_checkpoint" in errors


def test_evidence_may_not_live_in_the_old_namespace():
    errors = evidence_namespace_errors(
        [f"{LEGACY_EVIDENCE_ROOT}/cp566_resident_mixed_multidomain_replication/result.json"]
    )
    assert errors and errors[0].startswith("evidence_in_legacy_namespace")


def test_evidence_outside_the_recovery_root_is_refused():
    errors = evidence_namespace_errors(["artifacts/current/whatever.json"])
    assert errors and errors[0].startswith("evidence_outside_recovery_namespace")


def test_evidence_inside_the_recovery_root_is_accepted():
    assert evidence_namespace_errors([f"{RECOVERY_EVIDENCE_ROOT}/canary.json"]) == []


def test_a_fresh_package_carries_no_verdict():
    package = build_package(_descriptor(), campaign="rlc-27b-recovery")
    assert package["verdict"] is None
    assert package["evidence"]["measured_on_this_checkpoint"] is False
    assert package["authorizes"] == []
    assert package["legacy_claim"]["authority_over_this_package"] == "none"


def test_a_fresh_package_never_authorizes_ordinary_chat():
    package = build_package(_descriptor(), campaign="rlc-27b-recovery")
    for forbidden in (
        "ordinary_chat_authorized",
        "arbitrary_reasoning_authorized",
        "global runtime promotion",
    ):
        assert forbidden in package["never_authorizes"]


def test_building_with_legacy_evidence_is_refused():
    with pytest.raises(RecoveryPackageError, match="legacy_namespace"):
        build_package(
            _descriptor(),
            campaign="rlc-27b-recovery",
            evidence_paths=[f"{LEGACY_EVIDENCE_ROOT}/cp566/result.json"],
        )


def test_a_campaign_label_must_be_a_plain_slug():
    with pytest.raises(RecoveryPackageError):
        package_id(_descriptor(), campaign="../../escape")


# ── Independent certificate verification ────────────────────────────────


def _certificate(descriptor: DescriptorIdentity, **overrides) -> dict:
    body = {
        "schema": RECOVERY_CERTIFICATE_SCHEMA,
        "package_id": package_id(descriptor, campaign="rlc-27b-recovery"),
        "descriptor_identity": descriptor.as_dict(),
        "evidence_paths": [f"{RECOVERY_EVIDENCE_ROOT}/canary.json"],
        "verdict": "BOUNDED_RECOVERY",
        "evidence": {"measured_on_this_checkpoint": True},
        "legacy_claim": {
            "package_id": LEGACY_PACKAGE_ID,
            "authority_over_this_package": "none",
        },
    }
    body.update(overrides)
    return {**body, "certificate_sha256": canonical_sha256(body)}


def test_a_well_formed_certificate_verifies():
    descriptor = _descriptor()
    assert certificate_errors(_certificate(descriptor), descriptor) == []


def test_a_certificate_for_another_checkpoint_is_refused():
    descriptor = _descriptor()
    other = _descriptor(config_sha256="f" * 64)
    assert "certificate_describes_a_different_checkpoint" in certificate_errors(
        _certificate(other), descriptor
    )


def test_a_tampered_certificate_is_refused():
    descriptor = _descriptor()
    certificate = _certificate(descriptor)
    certificate["verdict"] = "SOMETHING_BIGGER"
    assert (
        "certificate_digest_does_not_cover_the_certificate"
        in certificate_errors(certificate, descriptor)
    )


def test_a_certificate_citing_the_old_evidence_root_is_refused():
    descriptor = _descriptor()
    certificate = _certificate(
        descriptor, evidence_paths=[f"{LEGACY_EVIDENCE_ROOT}/cp566/result.json"]
    )
    errors = certificate_errors(certificate, descriptor)
    assert any(e.startswith("evidence_in_legacy_namespace") for e in errors)


def test_an_unrecognised_schema_stops_verification_immediately():
    descriptor = _descriptor()
    assert certificate_errors({"schema": "other"}, descriptor) == [
        "certificate_schema_unrecognised"
    ]


def test_the_live_manifest_produces_a_package_distinct_from_the_legacy_one():
    manifest = INSTALL / "training/fused-model/active.json"
    if not manifest.exists():
        pytest.skip("no active model manifest")
    descriptor = descriptor_from_manifest(json.loads(manifest.read_text()))
    package = build_package(descriptor, campaign="rlc-27b-recovery")
    assert package["package_id"] != LEGACY_PACKAGE_ID
    assert descriptor.linear_attention_layers == 48
    assert descriptor.full_attention_layers == 16
    assert package["verdict"] is None
