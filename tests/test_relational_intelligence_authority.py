from __future__ import annotations

import json

import pytest

from core.social.relational_intelligence import RelationalIntelligence
from core.social.relational_memory import RelationalMemoryAuthority


def _authority(tmp_path) -> RelationalMemoryAuthority:
    return RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"r" * 32,
        legacy_paths=(),
        auto_provision_key=False,
    )


def _grant(authority: RelationalMemoryAuthority, *, durable: bool = True) -> None:
    operations = ["recall", "prompt"]
    if durable:
        operations.insert(0, "persist")
    # A session-only policy is one the person sets, and it refuses keeping; a durable one is a plain grant.
    (authority.grant_consent if durable else authority.replace_consent)(
        "bryan",
        kinds=["derived_profile"],
        operations=operations,
        receipt_id="relational-intelligence-consent",
    )


@pytest.mark.asyncio
async def test_relational_intelligence_does_not_infer_after_a_refusal(tmp_path):
    # Remembering needs no leave (Bryan, 13 November 2026); a person's refusal still holds.
    authority = _authority(tmp_path)
    authority.revoke_consent("bryan", receipt_id="bryan-refused")
    engine = RelationalIntelligence(
        tmp_path / "relational_intelligence.json",
        authority=authority,
    )

    updated = await engine.update_from_interaction(
        "bryan",
        "Honestly I feel scared and I disagree.",
        "I understand.",
    )

    assert updated is False
    assert engine.get_context_injection("bryan") == ""
    assert authority.status()["record_count"] == 0


@pytest.mark.asyncio
async def test_repeated_evidence_round_trips_as_calibrated_hypotheses(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority)
    engine = RelationalIntelligence(
        tmp_path / "relational_intelligence.json",
        authority=authority,
    )
    messages = [
        "Honestly I feel anxious. I disagree, but data shows AI supports autonomy. cobalt-private-phrase.",
        "Honestly I think this is an interesting debate. Data shows AI can support autonomy.",
        "I feel strongly and disagree, but this is a fun argument. Data shows autonomy matters.",
    ]
    for message in messages:
        assert await engine.update_from_interaction(
            "bryan",
            message,
            "Here is a bounded response.",
        )

    block = engine.get_context_injection("bryan")
    snapshot = authority.load_snapshot(
        "bryan",
        namespace="relational_intelligence:v1",
        kind="derived_profile",
    )

    assert block.startswith("## RELATIONAL HYPOTHESES")
    assert "never as identity, diagnosis" in block
    assert "Trust is building" not in block
    assert "Safe disclosure ceiling" not in block
    assert snapshot is not None
    assert snapshot["interactions_analyzed"] == 3
    assert snapshot["vulnerability"]["confidence"] >= 0.3
    assert snapshot["perspective"]["confidence"] >= 0.3
    assert "cobalt-private-phrase" not in json.dumps(snapshot)

    restored = RelationalIntelligence(
        tmp_path / "relational_intelligence.json",
        authority=_authority(tmp_path),
    )
    assert restored.get_context_injection("bryan").startswith(
        "## RELATIONAL HYPOTHESES"
    )


@pytest.mark.asyncio
async def test_continue_is_neutral_not_boredom_or_negative_engagement(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority, durable=False)
    engine = RelationalIntelligence(
        tmp_path / "relational_intelligence.json",
        authority=authority,
    )

    assert await engine.update_from_interaction("bryan", "continue", "Continuing.")
    snapshot = authority.load_snapshot(
        "bryan",
        namespace="relational_intelligence:v1",
        kind="derived_profile",
    )

    assert snapshot is not None
    assert snapshot["entertainment"]["evidence_count"] == 0
    assert snapshot["entertainment"]["what_bores"] == []
    assert snapshot["entertainment"]["confidence"] == 0.0
    assert authority.status()["durable_record_count"] == 0


@pytest.mark.asyncio
async def test_delete_and_cross_agent_access_invalidate_cached_projection(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority)
    engine = RelationalIntelligence(
        tmp_path / "relational_intelligence.json",
        authority=authority,
    )
    for _ in range(3):
        await engine.update_from_interaction(
            "bryan",
            "Honestly I feel anxious; I disagree, but this is interesting and data shows autonomy.",
            "A bounded response.",
        )

    assert engine.get_context_injection("alice") == ""
    authority.delete_agent("bryan", authorization_receipt_id="delete-relational")

    assert engine.get_context_injection("bryan") == ""
    assert engine.get_health()["profiles"] == 0
