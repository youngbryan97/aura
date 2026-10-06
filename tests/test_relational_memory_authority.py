from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.memory.shared_ground import SharedGroundBuffer
from core.memory.social_memory import SocialMemory
from core.runtime.memory_consent import (
    MemoryConsentMode,
    apply_relational_memory_command,
    is_delete_all_relational_memory_command,
    parse_consent_command,
)
from core.social.relational_memory import (
    LEGACY_UNSCOPED_AGENT,
    RelationalMemoryAuthority,
)


def _authority(tmp_path, *, now=None, legacy_paths=()):
    return RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"k" * 32,
        legacy_paths=legacy_paths,
        auto_provision_key=False,
        now_fn=now or (lambda: 100.0),
    )


def test_record_is_session_only_without_explicit_agent_consent(tmp_path):
    authority = _authority(tmp_path)

    record, receipt = authority.record(
        "bryan",
        kind="shared_ground",
        content="the verified release checklist",
    )

    assert record.durable is False
    assert receipt.durable is False
    assert authority.query("bryan", purpose="prompt") == []
    assert not (tmp_path / "relational.json").exists()


def test_consent_is_exact_agent_expiring_and_purpose_scoped(tmp_path):
    now = [100.0]
    authority = _authority(tmp_path, now=lambda: now[0])
    authority.grant_consent(
        "bryan",
        kinds=["shared_ground"],
        operations=["persist", "recall", "prompt"],
        receipt_id="user-consent-1",
        expires_at=200.0,
    )

    record, receipt = authority.record(
        "bryan",
        kind="shared_ground",
        content="the verified release checklist",
    )

    assert record.durable is True
    assert receipt.durable is True
    assert authority.query("alice", purpose="prompt") == []
    prompt = authority.prompt_block("bryan")
    assert "the verified release checklist" in prompt
    assert '"content":"the verified release checklist"' in prompt
    now[0] = 201.0
    assert authority.query("bryan", purpose="prompt") == []


def test_encrypted_envelope_contains_no_identity_or_memory_plaintext(tmp_path):
    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan",
        kinds=["boundary"],
        operations=["persist", "recall", "prompt"],
        receipt_id="user-consent-1",
    )
    authority.record(
        "bryan",
        kind="boundary",
        content="Never mention private-project-codename in public.",
    )

    payload = (tmp_path / "relational.json").read_text(encoding="utf-8")

    assert "bryan" not in payload
    assert "private-project-codename" not in payload
    assert '"encryption": "AES-256-GCM"' in payload
    assert json.loads(payload)["key_id"]


def test_restart_round_trip_and_wrong_key_locks_without_overwrite(tmp_path):
    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan",
        kinds=["milestone"],
        operations=["persist", "recall", "prompt"],
        receipt_id="user-consent-1",
    )
    authority.record("bryan", kind="milestone", content="shipped the verified parser")
    before = (tmp_path / "relational.json").read_bytes()

    restored = _authority(tmp_path)
    restored_prompt = restored.prompt_block("bryan")
    assert '"content":"shipped the verified parser"' in restored_prompt

    locked = RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"x" * 32,
        legacy_paths=(),
        auto_provision_key=False,
    )
    assert locked.status()["status"] == "locked"
    with pytest.raises(RuntimeError, match="store is locked"):
        locked.export_agent("bryan", authorization_receipt_id="export-1")
    with pytest.raises(RuntimeError, match="store is locked"):
        locked.delete_agent("bryan", authorization_receipt_id="delete-1")
    locked.record("bryan", kind="milestone", content="must remain session only")
    assert (tmp_path / "relational.json").read_bytes() == before


def test_a_store_under_another_key_is_named_by_both_fingerprints(tmp_path):
    import hashlib

    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan", kinds=["milestone"], operations=["persist"], receipt_id="user-consent-1"
    )
    authority.record("bryan", kind="milestone", content="shipped the verified parser")

    locked = RelationalMemoryAuthority(
        tmp_path / "relational.json", encryption_key=b"x" * 32, legacy_paths=(), auto_provision_key=False
    )
    stored = hashlib.sha256(b"k" * 32).hexdigest()[:16]
    current = hashlib.sha256(b"x" * 32).hexdigest()[:16]
    assert locked._locked_reason == f"encrypted_store_key_mismatch:{stored}!={current}"


def test_no_key_is_minted_over_a_store_on_disk(tmp_path, monkeypatch):
    """LIVE 2026-10-06: the July store met a different keychain key and stayed locked.

    A lookup that misses with a store on disk must not replace the key the
    store was written under.
    """
    from core.security import zenith_secrets

    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan", kinds=["milestone"], operations=["persist"], receipt_id="user-consent-1"
    )
    authority.record("bryan", kind="milestone", content="shipped the verified parser")
    minted = []
    monkeypatch.setattr(zenith_secrets, "get_secret", lambda *_a, **_k: None)
    monkeypatch.setattr(zenith_secrets, "set_secret", lambda *a, **_k: minted.append(a) or True)

    reopened = RelationalMemoryAuthority(tmp_path / "relational.json", legacy_paths=())
    assert minted == []
    assert reopened.persistence_available is False

    fresh = RelationalMemoryAuthority(tmp_path / "elsewhere" / "relational.json", legacy_paths=())
    assert len(minted) == 1
    assert fresh.persistence_available is True


def test_revoke_blocks_prompt_and_optional_delete_is_durable(tmp_path):
    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan",
        kinds=["boundary"],
        operations=["persist", "recall", "prompt"],
        receipt_id="grant-1",
    )
    authority.record("bryan", kind="boundary", content="keep this private")

    receipt = authority.revoke_consent(
        "bryan",
        receipt_id="revoke-1",
        delete_records=True,
    )

    assert receipt.operation == "revoke_consent"
    assert receipt.record_ids
    assert authority.export_agent(
        "bryan", authorization_receipt_id="export-after-delete"
    )["records"] == []
    assert authority.query("bryan", purpose="prompt") == []


def test_legacy_plaintext_is_encrypted_quarantined_and_requires_claim(tmp_path):
    social = tmp_path / "social_memory.json"
    shared = tmp_path / "shared_ground.json"
    social.write_text(
        json.dumps(
            {
                "milestones": [{"description": "legacy private milestone"}],
                "shared_keys": ["legacy callback"],
            }
        ),
        encoding="utf-8",
    )
    shared.write_text(
        json.dumps([{"reference": "old joke", "context": "private context"}]),
        encoding="utf-8",
    )

    authority = _authority(tmp_path, legacy_paths=(social, shared))

    assert not social.exists()
    assert not shared.exists()
    assert authority.status()["legacy_quarantine_count"] == 3
    assert "legacy private milestone" not in (
        tmp_path / "relational.json"
    ).read_text(encoding="utf-8")
    assert authority.query(LEGACY_UNSCOPED_AGENT, purpose="prompt") == []
    with pytest.raises(PermissionError):
        authority.claim_legacy_records(
            "bryan",
            confirmation_receipt_id="",
            confirmed=False,
        )

    authority.grant_consent(
        "bryan",
        kinds=["milestone", "shared_ground"],
        operations=["persist", "recall", "prompt"],
        receipt_id="grant-1",
    )
    receipt = authority.claim_legacy_records(
        "bryan",
        confirmation_receipt_id="claim-1",
        confirmed=True,
    )
    assert len(receipt.record_ids) == 3
    assert "legacy private milestone" in authority.prompt_block("bryan")


def test_delete_and_export_require_authorization_receipts(tmp_path):
    authority = _authority(tmp_path)
    with pytest.raises(PermissionError):
        authority.export_agent("bryan", authorization_receipt_id="")
    with pytest.raises(PermissionError):
        authority.delete_agent("bryan", authorization_receipt_id="")
    with pytest.raises(PermissionError):
        authority.delete_snapshot(
            "bryan",
            namespace="relationship_graph:v1",
            kind="shared_ground",
            authorization_receipt_id="",
        )


def test_targeted_snapshot_delete_preserves_other_namespaces_and_rolls_back(tmp_path, monkeypatch):
    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan",
        kinds=["shared_ground"],
        operations=["persist", "recall", "prompt"],
        receipt_id="grant-1",
    )
    authority.upsert_snapshot(
        "bryan",
        namespace="relationship_graph:v1",
        kind="shared_ground",
        payload={"node": {"interaction_count": 1}},
        confidence=0.4,
        provenance="test",
    )
    authority.upsert_snapshot(
        "bryan",
        namespace="shared_ground_adapter:v1",
        kind="shared_ground",
        payload={"entries": ["preserve"]},
        confidence=1.0,
        provenance="test",
    )

    receipt = authority.delete_snapshot(
        "bryan",
        namespace="relationship_graph:v1",
        kind="shared_ground",
        authorization_receipt_id="delete-graph",
    )

    assert receipt.operation == "delete_snapshot"
    assert receipt.record_ids
    assert authority.load_snapshot(
        "bryan",
        namespace="relationship_graph:v1",
        kind="shared_ground",
    ) is None
    assert authority.load_snapshot(
        "bryan",
        namespace="shared_ground_adapter:v1",
        kind="shared_ground",
    ) == {"entries": ["preserve"]}

    monkeypatch.setattr(authority, "_save_locked", lambda: False)
    with pytest.raises(RuntimeError, match="snapshot deletion could not be persisted"):
        authority.delete_snapshot(
            "bryan",
            namespace="shared_ground_adapter:v1",
            kind="shared_ground",
            authorization_receipt_id="delete-shared",
        )
    assert authority.load_snapshot(
        "bryan",
        namespace="shared_ground_adapter:v1",
        kind="shared_ground",
    ) == {"entries": ["preserve"]}


def test_failed_persistence_never_returns_a_durable_record_receipt(tmp_path, monkeypatch):
    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan",
        kinds=["milestone"],
        operations=["persist", "recall", "prompt"],
        receipt_id="grant-1",
    )
    monkeypatch.setattr(authority, "_save_locked", lambda: False)

    record, receipt = authority.record(
        "bryan",
        kind="milestone",
        content="write failed after consent",
    )

    assert record.durable is False
    assert receipt.durable is False
    assert receipt.reason == "persistence_failed_session_only"
    with pytest.raises(RuntimeError, match="revocation could not be persisted"):
        authority.revoke_consent("bryan", receipt_id="revoke-1")


def test_expired_record_is_rejected_before_mutating_state(tmp_path):
    authority = _authority(tmp_path)

    with pytest.raises(ValueError, match="expiry must be in the future"):
        authority.record(
            "bryan",
            kind="milestone",
            content="already expired",
            expires_at=99.0,
        )

    assert authority.status()["record_count"] == 0


def test_session_only_consent_does_not_survive_restart(tmp_path):
    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan",
        kinds=["milestone"],
        operations=["recall", "prompt"],
        receipt_id="session-grant",
    )
    authority.record("bryan", kind="milestone", content="session context")

    restored = _authority(tmp_path)

    assert authority.prompt_block("bryan")
    assert restored.prompt_block("bryan") == ""
    assert restored.status()["active_grant_count"] == 0


def test_failed_duplicate_update_preserves_prior_durable_record(tmp_path, monkeypatch):
    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan",
        kinds=["milestone"],
        operations=["persist", "recall", "prompt"],
        receipt_id="grant-1",
    )
    original, _ = authority.record(
        "bryan",
        kind="milestone",
        content="durable result",
        confidence=0.4,
    )
    monkeypatch.setattr(authority, "_save_locked", lambda: False)

    preserved, receipt = authority.record(
        "bryan",
        kind="milestone",
        content="durable result",
        confidence=0.9,
    )

    assert preserved.record_id == original.record_id
    assert preserved.durable is True
    assert preserved.confidence == 0.4
    assert receipt.durable is False
    assert receipt.reason == "persistence_failed_prior_preserved"


def test_failed_control_plane_mutations_roll_back_live_state(tmp_path, monkeypatch):
    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan",
        kinds=["milestone"],
        operations=["persist", "recall", "prompt"],
        receipt_id="grant-1",
    )
    record, _ = authority.record(
        "bryan", kind="milestone", content="must survive failed controls"
    )
    monkeypatch.setattr(authority, "_save_locked", lambda: False)

    with pytest.raises(RuntimeError, match="revocation could not be persisted"):
        authority.revoke_consent("bryan", receipt_id="revoke-1")
    assert authority.query("bryan", purpose="prompt")

    with pytest.raises(RuntimeError, match="deletion could not be persisted"):
        authority.delete_agent("bryan", authorization_receipt_id="delete-1")
    assert authority.query("bryan", purpose="prompt")
    assert authority.mark_used("bryan", record.record_id) is False
    restored = authority.query("bryan", purpose="recall")[0]
    assert restored.use_count == 0


def test_failed_consent_replacement_preserves_prior_policy(tmp_path, monkeypatch):
    authority = _authority(tmp_path)
    original = authority.grant_consent(
        "bryan",
        kinds=["milestone"],
        operations=["persist", "recall", "prompt"],
        receipt_id="grant-1",
    )
    authority.record("bryan", kind="milestone", content="still available")
    monkeypatch.setattr(authority, "_save_locked", lambda: False)

    with pytest.raises(RuntimeError, match="replacement could not be persisted"):
        authority.replace_consent(
            "bryan",
            kinds=["milestone"],
            operations=["recall", "prompt"],
            receipt_id="session-replacement",
        )

    assert authority.allows("bryan", "milestone", "persist") is True
    assert authority.prompt_block("bryan")
    exported = authority.export_agent(
        "bryan", authorization_receipt_id="verify-rollback"
    )
    original_export = next(
        grant for grant in exported["grants"] if grant["grant_id"] == original.grant_id
    )
    assert original_export["revoked_at"] is None


def test_structured_snapshot_upserts_one_authority_record_and_stays_out_of_generic_prompt(
    tmp_path,
):
    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan",
        kinds=["derived_profile"],
        operations=["persist", "recall", "prompt"],
        receipt_id="grant-profile",
    )

    first, _ = authority.upsert_snapshot(
        "bryan",
        namespace="conversational_profile:v1",
        kind="derived_profile",
        payload={"directness": 0.4, "turns": 3},
        confidence=0.3,
        provenance="conversational_profile.heuristics",
    )
    second, _ = authority.upsert_snapshot(
        "bryan",
        namespace="conversational_profile:v1",
        kind="derived_profile",
        payload={"directness": 0.8, "turns": 4},
        confidence=0.4,
        provenance="conversational_profile.heuristics",
    )

    assert second.record_id == first.record_id
    assert authority.status()["record_count"] == 1
    assert authority.load_snapshot(
        "bryan",
        namespace="conversational_profile:v1",
        kind="derived_profile",
        purpose="prompt",
    ) == {"directness": 0.8, "turns": 4}
    assert authority.prompt_block("bryan") == ""
    assert authority.load_snapshot(
        "alice",
        namespace="conversational_profile:v1",
        kind="derived_profile",
    ) is None


def test_legacy_profile_quarantine_claims_only_matching_exact_agent(tmp_path):
    legacy = tmp_path / "conversational_profiles.json"
    legacy.write_text(
        json.dumps(
            {
                "profiles": {
                    "bryan": {"user_id": "bryan", "interactions_analyzed": 7},
                    "alice": {"user_id": "alice", "interactions_analyzed": 9},
                },
                "phrase_counters": {"bryan": {"exact phrase": 3}},
            }
        ),
        encoding="utf-8",
    )
    authority = _authority(tmp_path)

    migrated = authority.quarantine_legacy_snapshot_file(
        legacy,
        namespace="conversational_profile:v1",
        kind="derived_profile",
    )

    assert migrated == 2
    assert not legacy.exists()
    authority.grant_consent(
        "bryan",
        kinds=["derived_profile"],
        operations=["persist", "recall", "prompt"],
        receipt_id="grant-bryan",
    )
    receipt = authority.claim_legacy_records(
        "bryan",
        confirmation_receipt_id="claim-bryan",
        confirmed=True,
    )
    assert len(receipt.record_ids) == 1
    snapshot = authority.load_snapshot(
        "bryan",
        namespace="conversational_profile:v1",
        kind="derived_profile",
    )
    assert snapshot == {
        "profile": {"interactions_analyzed": 7, "user_id": "bryan"},
        "phrase_counts": {"exact phrase": 3},
    }
    assert authority.status()["legacy_quarantine_count"] == 1


def test_session_snapshot_overlay_never_rewrites_prior_durable_value(tmp_path):
    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan",
        kinds=["derived_profile"],
        operations=["persist", "recall", "prompt"],
        receipt_id="durable-grant",
    )
    authority.upsert_snapshot(
        "bryan",
        namespace="conversational_profile:v1",
        kind="derived_profile",
        payload={"turns": 1},
        confidence=0.2,
        provenance="test",
    )
    authority.replace_consent(
        "bryan",
        kinds=["derived_profile"],
        operations=["recall", "prompt"],
        receipt_id="session-grant",
    )
    before_overlay = (tmp_path / "relational.json").read_bytes()

    overlay, receipt = authority.upsert_snapshot(
        "bryan",
        namespace="conversational_profile:v1",
        kind="derived_profile",
        payload={"turns": 2},
        confidence=0.3,
        provenance="test",
    )

    assert overlay.durable is False
    assert receipt.durable is False
    assert authority.load_snapshot(
        "bryan",
        namespace="conversational_profile:v1",
        kind="derived_profile",
    ) == {"turns": 2}
    assert (tmp_path / "relational.json").read_bytes() == before_overlay

    restored = _authority(tmp_path)
    restored.grant_consent(
        "bryan",
        kinds=["derived_profile"],
        operations=["recall", "prompt"],
        receipt_id="new-session-grant",
    )
    assert restored.load_snapshot(
        "bryan",
        namespace="conversational_profile:v1",
        kind="derived_profile",
    ) == {"turns": 1}


def test_legacy_dialogue_source_profiles_are_never_claimable_as_users(tmp_path):
    legacy = tmp_path / "dialogue_cognition.json"
    legacy.write_text(
        json.dumps(
            {
                "profiles": {
                    "bryan": {"user_id": "bryan", "interactions_analyzed": 4},
                    "source:edi": {
                        "user_id": "source:edi",
                        "interactions_analyzed": 30,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    authority = _authority(tmp_path)
    assert authority.quarantine_legacy_snapshot_file(
        legacy,
        namespace="dialogue_cognition:v1",
        kind="dialogue_preference",
    ) == 2
    authority.grant_consent(
        "bryan",
        kinds=["dialogue_preference"],
        operations=["persist", "recall", "prompt"],
        receipt_id="dialogue-grant",
    )

    receipt = authority.claim_legacy_records(
        "bryan",
        confirmation_receipt_id="claim-dialogue",
        confirmed=True,
    )

    assert len(receipt.record_ids) == 1
    assert authority.load_snapshot(
        "bryan",
        namespace="dialogue_cognition:v1",
        kind="dialogue_preference",
    ) == {"profile": {"interactions_analyzed": 4, "user_id": "bryan"}}
    assert authority.status()["legacy_quarantine_count"] == 1


def test_legacy_adapters_share_one_exact_agent_authority(tmp_path):
    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan",
        kinds=["milestone", "shared_ground"],
        operations=["persist", "recall", "prompt"],
        receipt_id="grant-1",
    )
    social = SocialMemory(authority=authority)
    shared = SharedGroundBuffer(authority=authority)
    ServiceContainer.clear()
    ServiceContainer.register_instance(
        "other_agent_model",
        SimpleNamespace(active_agent_id="bryan"),
        required=False,
    )

    try:
        milestone = social.record_milestone("verified the exact-agent migration", 0.8)
        entry = shared.record(
            "release checklist",
            context="Established after live verification",
            salience=0.7,
            tags=["work"],
        )
        social.relationship_depth = 1.0
        relationship_depth = social.relationship_depth
        social_context = social.get_social_context("bryan")
        shared_context = shared.get_context_injection(agent_id="bryan")
        alice_context = shared.get_context_injection(agent_id="alice")
    finally:
        ServiceContainer.clear()

    assert milestone.record_id
    assert entry.agent_id == "bryan"
    assert 0.0 < relationship_depth < 1.0
    assert "verified the exact-agent migration" in social_context
    assert "CONSENTED SHARED GROUND" in shared_context
    assert alice_context == ""
    assert not (tmp_path / "social_memory.json").exists()
    assert not (tmp_path / "shared_ground.json").exists()


def test_authority_has_no_synthetic_active_agent_without_estimator(tmp_path):
    ServiceContainer.clear()
    authority = _authority(tmp_path)

    assert authority.active_agent_id == ""


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("Aura, please remember always now.", MemoryConsentMode.REMEMBER_ALWAYS),
        ("session only please", MemoryConsentMode.SESSION_ONLY),
        ("Could you go private mode now?", MemoryConsentMode.PRIVATE_MODE),
        ("ask before remembering", MemoryConsentMode.ASK_BEFORE_REMEMBERING),
    ],
)
def test_relational_consent_parser_accepts_explicit_commands(command, expected):
    assert parse_consent_command(command) is expected


@pytest.mark.parametrize(
    "statement",
    [
        "Do not remember always.",
        "The documentation says 'remember always'.",
        "Remember always that my project is private.",
        "Can you explain what session only means?",
        "I will always remember this.",
    ],
)
def test_relational_consent_parser_rejects_mentions_and_negations(statement):
    assert parse_consent_command(statement) is None


def test_canonical_relational_command_applies_exact_agent_policy(tmp_path):
    authority = _authority(tmp_path)

    result = apply_relational_memory_command(
        authority,
        "bryan",
        "Aura, remember always.",
        receipt_id="authenticated-chat-command",
    )

    assert result is not None
    assert result["mode"] == "remember_always"
    assert result["persistence_allowed"] is True
    assert authority.allows("bryan", "derived_profile", "persist") is True
    assert authority.allows("alice", "derived_profile", "persist") is False


def test_canonical_private_mode_revokes_without_deleting_records(tmp_path):
    authority = _authority(tmp_path)
    apply_relational_memory_command(authority, "bryan", "remember always")
    authority.record("bryan", kind="boundary", content="private boundary")

    result = apply_relational_memory_command(authority, "bryan", "go private")

    assert result is not None
    assert result["mode"] == "private_mode"
    assert authority.allows("bryan", "boundary", "recall") is False
    assert authority.status()["record_count"] == 1


def test_canonical_delete_all_command_deletes_exact_agent_only(tmp_path):
    authority = _authority(tmp_path)
    apply_relational_memory_command(authority, "bryan", "remember always")
    apply_relational_memory_command(authority, "alice", "remember always")
    authority.record("bryan", kind="milestone", content="bryan record")
    authority.record("alice", kind="milestone", content="alice record")

    result = apply_relational_memory_command(
        authority,
        "bryan",
        "please forget everything about me",
    )

    assert is_delete_all_relational_memory_command(
        "please forget everything about me"
    )
    assert result is not None
    assert result["mode"] == "deleted"
    assert authority.query("bryan") == []
    assert authority.query("alice")[0].content == "alice record"
