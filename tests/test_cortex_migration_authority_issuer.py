from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from core.learning.cortex_migration_authority import validate_component_authority
from core.learning.candidate_cortex_fusion import CandidateCortexFusionError
from core.learning.cortex_migration_authority_issuer import (
    issue_deferred_model_tissue_authority,
    issue_expert_retirement_authority,
    issue_persona_crsm_authority,
    issue_recurrence_authority,
    issue_steering_authority,
)
from tests.support.cortex_migration_authority import build_signed_migration_authorities


@pytest.fixture(autouse=True)
def _isolated_state_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "state"))


def _fixture_sources(tmp_path: Path, descriptor_sha256: str):
    return build_signed_migration_authorities(
        tmp_path / "sources",
        descriptor_sha256=descriptor_sha256,
        state_root=tmp_path / "state",
    )


def test_component_owned_issuers_retain_and_reopen_exact_evidence(tmp_path):
    descriptor_sha256 = "a" * 64
    sources = _fixture_sources(tmp_path, descriptor_sha256)
    steering_source = sources["steering"]["evidence"]
    expert_source = sources["expert_adapters"]["evidence"]
    recurrence_source = sources["recurrence_native"]["evidence"]

    steering = issue_steering_authority(
        metadata_path=Path(steering_source["metadata"]["path"]),
        causal_evaluation_path=Path(steering_source["causal_evaluation"]["path"]),
        independent_evidence_path=Path(steering_source["independent_verifier"]["path"]),
        descriptor_sha256=descriptor_sha256,
        custody_base=tmp_path / "issued",
        issued_at=2.0,
    )
    expert = issue_expert_retirement_authority(
        inventory_path=Path(expert_source["migration_inventory"]["path"]),
        descriptor_sha256=descriptor_sha256,
        custody_base=tmp_path / "issued",
        issued_at=2.0,
    )
    recurrence = issue_recurrence_authority(
        activation_path=Path(recurrence_source["activation"]["path"]),
        descriptor_sha256=descriptor_sha256,
        custody_base=tmp_path / "issued",
        issued_at=2.0,
    )

    for component, authority in {
        "steering": steering,
        "expert_adapters": expert,
        "recurrence_native": recurrence,
    }.items():
        assert (
            validate_component_authority(
                authority,
                component=component,
                descriptor_sha256=descriptor_sha256,
            )
            == authority
        )
        assert Path(authority["custody_root"]).is_relative_to(tmp_path / "issued")
        assert list(Path(authority["custody_root"]).glob("authority-*.json"))


@pytest.mark.parametrize(
    ("component", "family"),
    (("steering", "caa_steering"), ("recurrence_native", "recurrent_tissue")),
)
def test_deferred_tissue_authority_quarantines_old_basis(tmp_path, component, family):
    descriptor_sha256 = "e" * 64
    sources = _fixture_sources(tmp_path, descriptor_sha256)
    source = Path(sources["expert_adapters"]["evidence"]["migration_inventory"]["path"])
    inventory = json.loads(source.read_text(encoding="ascii"))
    for item in inventory["families"]:
        if item["family"] == family:
            item["outcome"] = "retrain"
            item["candidate_runtime_loadable"] = False
    material = dict(inventory)
    material.pop("inventory_sha256")
    inventory["inventory_sha256"] = hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    source.write_text(json.dumps(inventory), encoding="ascii")

    authority = issue_deferred_model_tissue_authority(
        component=component,
        inventory_path=source,
        descriptor_sha256=descriptor_sha256,
        custody_base=tmp_path / "issued",
        issued_at=2.0,
    )

    assert authority["status"] == "deferred"
    assert authority["authority_kind"] == "model_basis_quarantine"
    assert (
        validate_component_authority(
            authority,
            component=component,
            descriptor_sha256=descriptor_sha256,
        )
        == authority
    )


def test_issued_authority_survives_source_removal_but_not_retained_drift(tmp_path):
    descriptor_sha256 = "b" * 64
    sources = _fixture_sources(tmp_path, descriptor_sha256)
    evidence = sources["expert_adapters"]["evidence"]["migration_inventory"]
    source_path = Path(evidence["path"])
    authority = issue_expert_retirement_authority(
        inventory_path=source_path,
        descriptor_sha256=descriptor_sha256,
        custody_base=tmp_path / "issued",
        issued_at=3.0,
    )
    source_path.unlink()
    assert (
        validate_component_authority(
            authority,
            component="expert_adapters",
            descriptor_sha256=descriptor_sha256,
        )
        == authority
    )

    retained = Path(authority["evidence"]["migration_inventory"]["path"])
    retained.write_text("{}", encoding="ascii")
    with pytest.raises(ValueError, match="migration_inventory_binding_drift"):
        validate_component_authority(
            authority,
            component="expert_adapters",
            descriptor_sha256=descriptor_sha256,
        )


def test_steering_issuer_refuses_unbound_independent_evidence(tmp_path):
    descriptor_sha256 = "c" * 64
    sources = _fixture_sources(tmp_path, descriptor_sha256)
    evidence = sources["steering"]["evidence"]
    substituted = tmp_path / "substituted-independent-evidence.json"
    substituted.write_text(json.dumps({"verdict": "PASS"}), encoding="ascii")

    with pytest.raises(ValueError, match="steering_causal_evaluation_invalid"):
        issue_steering_authority(
            metadata_path=Path(evidence["metadata"]["path"]),
            causal_evaluation_path=Path(evidence["causal_evaluation"]["path"]),
            independent_evidence_path=substituted,
            descriptor_sha256=descriptor_sha256,
            custody_base=tmp_path / "issued",
        )


def test_persona_issuer_refuses_a_digest_shaped_fusion_story(tmp_path):
    descriptor_sha256 = "d" * 64
    sources = _fixture_sources(tmp_path, descriptor_sha256)
    evidence = sources["persona_crsm"]["evidence"]
    journal_key = tmp_path / "journal.key"
    journal_key.write_bytes(b"k" * 64)

    with pytest.raises(CandidateCortexFusionError):
        issue_persona_crsm_authority(
            fusion_plan_path=Path(evidence["fusion_plan"]["path"]),
            fusion_receipt_path=Path(evidence["fusion_receipt"]["path"]),
            journal_key_path=journal_key,
            descriptor_sha256=descriptor_sha256,
            custody_base=tmp_path / "issued",
        )


def test_no_generic_component_signer_is_public():
    import core.learning.cortex_migration_authority_issuer as issuer

    assert not hasattr(issuer, "issue_component_authority")
    assert "issue_component_authority" not in issuer.__all__
