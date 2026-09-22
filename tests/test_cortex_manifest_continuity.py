"""Exact manifest comparisons with real artifact and signed-authority validators."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from core.brain.llm.cortex_manifest_continuity import (
    manifest_snapshot_path,
    verify_manifest_component_continuity,
)
from core.brain.llm.semantic_neural_serving import (
    _identity_for_manifest,
    _manifest_identity_matches_activation,
    _qualification_manifest_identity,
)
from core.learning.cortex_generation_upgrade import (
    _governed_write,
    activate_upgrade,
    build_migration_contract,
    retain_cortex_manifest_snapshot,
    stage_upgrade,
)
from tests.support.cortex_migration_authority import build_signed_migration_authorities
from tests.test_cortex_generation_upgrade import _fused_dir, _upgrade_contracts


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


@pytest.fixture
def manifests(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("AURA_STATE_ROOT", str(state))
    fused, model = _fused_dir(tmp_path)
    descriptor, evaluation, profile, migration = _upgrade_contracts(model)
    stage_upgrade(
        candidate_model_path=model,
        base_model_path="test-base",
        tag="qualified",
        fused_model_dir=fused,
        evaluation=evaluation,
        serving_profile=profile,
        migration_contract=migration,
        artifact_descriptor=descriptor,
    )
    activate_upgrade(fused_model_dir=fused, authorized_by="test-operator", evaluation=evaluation)
    manifest = fused / "active.json"
    old = manifest.read_bytes()
    old_identity = _identity_for_manifest(manifest)
    refreshed = build_signed_migration_authorities(
        tmp_path / "refreshed",
        descriptor_sha256=descriptor["descriptor_sha256"],
        state_root=state,
    )
    components = {**migration["components"], "steering": refreshed["steering"]}
    current = json.loads(old)
    current["migration_contract"] = build_migration_contract(descriptor, components=components)
    _governed_write(manifest, _bytes(current), source="test.manifest_continuity")
    return manifest, model, old, old_identity, current, refreshed, state


def _verify(data, **kwargs):
    manifest, model, old, *_ = data
    return verify_manifest_component_continuity(
        manifest=manifest,
        predecessor_sha256=hashlib.sha256(old).hexdigest(),
        current_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        model_path=model,
        **kwargs,
    )


def test_signed_independent_update_keeps_exact_measurement_identity(manifests):
    manifest, model, old, expected, current, *_ = manifests
    proof = _verify(manifests, independent_components=frozenset({"steering"}))
    assert proof["changed_independent_components"] == ["steering"]
    assert proof["comparative_remeasurement"] is False
    assert proof["model_descriptor_sha256"] == current["artifact_descriptor"]["descriptor_sha256"]
    assert proof["current_pointer_sha256"] != expected["sha256"]
    assert manifest_snapshot_path(manifest, expected["sha256"]).read_bytes() == old
    assert _manifest_identity_matches_activation(
        expected=expected, current=_identity_for_manifest(manifest), selected_model=model
    )


def test_requalification_retains_measured_identity_after_independent_update(manifests):
    manifest, model, _, expected, *_ = manifests
    current = _identity_for_manifest(manifest)
    assert expected != current
    measured = _qualification_manifest_identity(
        expected=expected, current=current, selected_model=model
    )
    assert measured == expected
    measured["sha256"] = "altered by caller"
    assert measured != expected


def test_requalification_rejects_a_changed_dependent_component(manifests):
    manifest, model, _, expected, current, refreshed, _ = manifests
    components = copy.deepcopy(current["migration_contract"]["components"])
    components["recurrence_native"] = refreshed["recurrence_native"]
    current["migration_contract"] = build_migration_contract(
        current["artifact_descriptor"], components=components
    )
    manifest.write_bytes(_bytes(current))
    with pytest.raises(RuntimeError, match="identity differs"):
        _qualification_manifest_identity(
            expected=expected, current=_identity_for_manifest(manifest), selected_model=model
        )


def test_no_component_is_independent_by_default(manifests):
    with pytest.raises(ValueError, match="dependent_component_changed"):
        _verify(manifests)


@pytest.mark.parametrize("field", ["tag", "fused_at", "base_model", "schema_version", "new_field"])
def test_unrelated_pointer_edits_do_not_borrow_continuity(manifests, field):
    manifest, _, _, _, current, *_ = manifests
    current[field] = 4 if field == "schema_version" else "changed"
    manifest.write_bytes(_bytes(current))
    with pytest.raises(ValueError, match="not_narrow"):
        _verify(manifests, independent_components=frozenset({"steering"}))


@pytest.mark.parametrize("component", ["recurrence_native", "persona_crsm", "expert_adapters"])
def test_valid_changes_to_dependent_authorities_remain_rejected(manifests, component):
    manifest, model, _, expected, current, refreshed, _ = manifests
    components = copy.deepcopy(current["migration_contract"]["components"])
    components[component] = refreshed[component]
    current["migration_contract"] = build_migration_contract(
        current["artifact_descriptor"], components=components
    )
    manifest.write_bytes(_bytes(current))
    with pytest.raises(ValueError, match="dependent_component_changed"):
        _verify(manifests, independent_components=frozenset({"steering"}))
    assert not _manifest_identity_matches_activation(
        expected=expected, current=_identity_for_manifest(manifest), selected_model=model
    )


@pytest.mark.parametrize("field", ["serving_profile", "evaluation", "artifact_descriptor"])
def test_contract_edits_cannot_ride_a_steering_update(manifests, field):
    manifest, _, _, _, current, *_ = manifests
    current[field]["new_field"] = "changed"
    manifest.write_bytes(_bytes(current))
    with pytest.raises(ValueError):
        _verify(manifests, independent_components=frozenset({"steering"}))


def test_current_authority_is_reopened_and_signature_checked(manifests):
    manifest, _, _, _, current, *_ = manifests
    current["migration_contract"]["components"]["steering"]["authority_hmac_sha256"] = "0" * 64
    manifest.write_bytes(_bytes(current))
    with pytest.raises(ValueError, match="current_authority_invalid"):
        _verify(manifests, independent_components=frozenset({"steering"}))


def test_changed_weight_bytes_cannot_borrow_an_unchanged_index(manifests):
    _, model, *_ = manifests
    (model / "model.safetensors").write_bytes(b"unqualified replacement weights")
    with pytest.raises(ValueError, match="descriptor_mismatch"):
        _verify(manifests, independent_components=frozenset({"steering"}))


def test_full_hash_cache_rejects_same_size_replacement_with_restored_mtime(manifests):
    import os

    _verify(manifests, independent_components=frozenset({"steering"}))
    _, model, *_ = manifests
    weights = model / "model.safetensors"
    before = weights.stat()
    weights.write_bytes(b"x" * before.st_size)
    os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="descriptor_mismatch"):
        _verify(manifests, independent_components=frozenset({"steering"}))


def test_repeated_validation_reuses_full_hash_only_for_identical_file_versions(manifests):
    from core.brain.llm.model_artifact_profile import _validate_model_artifact_files_cached

    before = _validate_model_artifact_files_cached.cache_info()
    _verify(manifests, independent_components=frozenset({"steering"}))
    first = _validate_model_artifact_files_cached.cache_info()
    _verify(manifests, independent_components=frozenset({"steering"}))
    second = _validate_model_artifact_files_cached.cache_info()
    assert first.misses == before.misses + 1
    assert second.misses == first.misses
    assert second.hits == first.hits + 1


def test_explicit_verification_key_does_not_fall_back_to_test_state(
    manifests, monkeypatch, tmp_path
):
    manifest, model, _, expected, _, _, state = manifests
    key = state / "private/cortex-upgrade/migration-authority.key"
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "no-authority"))
    assert _manifest_identity_matches_activation(
        expected=expected,
        current=_identity_for_manifest(manifest),
        selected_model=model,
        authority_key_path=key,
    )
    assert not _manifest_identity_matches_activation(
        expected=expected, current=_identity_for_manifest(manifest), selected_model=model
    )


def test_snapshot_is_content_addressed_and_cannot_be_overwritten(manifests):
    manifest, _, old, expected, *_ = manifests
    snapshot = manifest_snapshot_path(manifest, expected["sha256"])
    assert snapshot.stat().st_mode & 0o777 == 0o400
    assert retain_cortex_manifest_snapshot(manifest, old) == snapshot
    snapshot.chmod(0o600)
    snapshot.write_bytes(b'{"corrupted": true}')
    with pytest.raises(ValueError, match="digest_mismatch"):
        _verify(manifests, independent_components=frozenset({"steering"}))
    with pytest.raises(ValueError, match="existing_bytes_mismatch"):
        retain_cortex_manifest_snapshot(manifest, old)


def test_snapshot_symlink_is_not_evidence(manifests, tmp_path):
    manifest, _, old, expected, *_ = manifests
    snapshot = manifest_snapshot_path(manifest, expected["sha256"])
    snapshot.unlink()
    alternate = tmp_path / "alternate.json"
    alternate.write_bytes(old)
    snapshot.symlink_to(alternate)
    with pytest.raises(ValueError, match="file_invalid"):
        _verify(manifests, independent_components=frozenset({"steering"}))


def test_history_name_cannot_escape_its_directory(tmp_path):
    with pytest.raises(ValueError, match="digest_invalid"):
        manifest_snapshot_path(tmp_path / "active.json", "../untrusted")


@pytest.mark.asyncio
async def test_boot_preparation_checks_artifacts_off_loop(monkeypatch):
    import threading

    from core.brain.llm import semantic_neural_serving

    caller_thread = threading.get_ident()

    def verify():
        assert threading.get_ident() != caller_thread
        return {"active": True, "reason": "test_verified"}

    monkeypatch.setattr(semantic_neural_serving, "semantic_neural_default_serving_status", verify)
    assert await semantic_neural_serving.prepare_semantic_neural_serving() == {
        "active": True,
        "reason": "test_verified",
    }
