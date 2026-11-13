from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.conversational.humor_engine import HumorEngine
from core.orchestrator.mixins.incoming_logic import IncomingLogicMixin
from core.social.relational_memory import RelationalMemoryAuthority


def _authority(tmp_path, now) -> RelationalMemoryAuthority:
    return RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"h" * 32,
        legacy_paths=(),
        auto_provision_key=False,
        now_fn=lambda: now[0],
    )


def _grant(authority: RelationalMemoryAuthority, user_id: str, *, durable: bool = True) -> None:
    operations = ["recall", "prompt"]
    if durable:
        operations.insert(0, "persist")
    # A session-only policy is one the person sets, and it refuses keeping; a durable one is a plain grant.
    (authority.grant_consent if durable else authority.replace_consent)(
        user_id,
        kinds=["style_preference"],
        operations=operations,
        receipt_id=f"humor-consent-{user_id}",
    )


def _engine(tmp_path, now, *, durable: bool = True):
    authority = _authority(tmp_path, now)
    _grant(authority, "bryan", durable=durable)
    return (
        HumorEngine(
            tmp_path / "humor_profiles.json",
            authority=authority,
            now_fn=lambda: now[0],
        ),
        authority,
    )


def _attempt(engine: HumorEngine, now, *, user_id: str = "bryan"):
    return engine.observe_delivered_response(
        user_id,
        "That parser has one job. It chose interpretive dance instead.",
        metadata={
            "response_metadata": {
                "provenance": "aura.response_generation",
                "response_humor_attempt": True,
                "humor_type": "dry_wit",
                "topic": "software parser",
            }
        },
        delivered_at=now[0],
        delivery_receipt_id=f"delivery-{user_id}-{now[0]}",
    )


def test_humor_does_not_record_or_prompt_after_a_refusal(tmp_path):
    # Remembering needs no leave (Bryan, 13 November 2026); a person's refusal still holds.
    now = [100.0]
    authority = _authority(tmp_path, now)
    authority.revoke_consent("bryan", receipt_id="bryan-refused")
    engine = HumorEngine(
        tmp_path / "humor_profiles.json",
        authority=authority,
        now_fn=lambda: now[0],
    )

    assert _attempt(engine, now) is None
    assert engine.record_reaction("bryan", "lol", timestamp=101.0) is None
    assert engine.get_humor_guidance("bryan") == ""
    assert authority.status()["record_count"] == 0


def test_delivered_attempt_and_reaction_persist_only_digests(tmp_path):
    now = [100.0]
    engine, authority = _engine(tmp_path, now)
    response = "That parser has one job. It chose interpretive dance instead."
    reaction = "lol that landed"

    attempt = _attempt(engine, now)
    result = engine.record_reaction("bryan", reaction, timestamp=101.0)
    snapshot = authority.load_snapshot(
        "bryan",
        namespace="humor_profile:v1",
        kind="style_preference",
    )

    assert attempt is not None
    assert result is True
    assert snapshot is not None
    encoded = json.dumps(snapshot)
    assert response not in encoded
    assert reaction not in encoded
    assert snapshot["attempts"][0]["response_digest"]
    assert snapshot["attempts"][0]["reaction_digest"]
    assert snapshot["profile"]["landing_rate"] == pytest.approx(2 / 3)
    assert "Evidence is sparse" in engine.get_humor_guidance("bryan")


def test_neutral_next_turn_consumes_attempt_without_scoring_later_praise(tmp_path):
    now = [100.0]
    engine, _ = _engine(tmp_path, now)
    assert _attempt(engine, now) is not None

    assert engine.record_reaction("bryan", "continue", timestamp=101.0) is None
    assert engine.record_reaction("bryan", "lol that was funny", timestamp=102.0) is None

    profile = engine.get_profile("bryan")
    assert profile is not None
    assert profile.total_attempts == 1
    assert profile.scored_attempts == 0
    assert engine.get_status()["pending_feedback_windows"] == 0


def test_stale_feedback_and_unmarked_responses_never_create_outcomes(tmp_path):
    now = [100.0]
    engine, _ = _engine(tmp_path, now)

    assert engine.observe_delivered_response(
        "bryan",
        "A plain factual response.",
        metadata={},
        delivery_receipt_id="plain-delivery",
    ) is None
    assert _attempt(engine, now) is not None
    assert engine.record_reaction("bryan", "lol", timestamp=100.0 + 1801.0) is None

    profile = engine.get_profile("bryan")
    assert profile is not None
    assert profile.scored_attempts == 0


def test_request_metadata_cannot_fabricate_a_humor_attempt(tmp_path):
    now = [100.0]
    engine, _ = _engine(tmp_path, now)

    assert engine.observe_delivered_response(
        "bryan",
        "The user described a joke and asked for its historical context.",
        metadata={"humor_type": "dry_wit", "humor_frame_active": True},
        delivery_receipt_id="plain-delivery",
    ) is None
    assert engine.get_status()["pending_feedback_windows"] == 0


def test_malformed_or_legacy_unpaired_snapshot_is_sanitized_and_unscored(tmp_path):
    now = [100.0]
    authority = _authority(tmp_path, now)
    _grant(authority, "bryan")
    authority.upsert_snapshot(
        "bryan",
        namespace="humor_profile:v1",
        kind="style_preference",
        payload={
            "attempts": [
                {
                    "id": {"untrusted": True},
                    "delivered_at": "not-a-time",
                    "expires_at": [],
                    "humor_type": {"not": "a type"},
                    "landed": True,
                    "reaction_at": 2.0,
                    "context_register": "private free-form transcript",
                }
            ],
            "profile": {
                "scored_attempts": "999999999999999999999",
                "landing_rate": "infinity",
                "type_scores": {"dry_wit": "not-a-number"},
            },
        },
        confidence=0.0,
        provenance="test.malformed_legacy",
    )
    engine = HumorEngine(
        tmp_path / "humor_profiles.json",
        authority=authority,
        now_fn=lambda: now[0],
    )

    profile = engine.get_profile("bryan")

    assert profile is not None
    assert profile.total_attempts == 1
    assert profile.scored_attempts == 0
    assert profile.landing_rate == 0.5
    assert "Evidence is sparse" in engine.get_humor_guidance("bryan")


def test_predelivery_feedback_cannot_consume_or_score_attempt(tmp_path):
    now = [100.0]
    engine, _ = _engine(tmp_path, now)
    assert _attempt(engine, now) is not None

    assert engine.record_reaction("bryan", "lol", timestamp=99.0) is None
    assert engine.get_status()["pending_feedback_windows"] == 1
    assert engine.record_reaction("bryan", "lol", timestamp=101.0) is True


def test_explicit_negative_language_wins_over_positive_word_overlap(tmp_path):
    now = [100.0]
    engine, _ = _engine(tmp_path, now)
    assert _attempt(engine, now) is not None

    assert engine.record_reaction("bryan", "That was not hilarious.", 101.0) is False


def test_three_paired_attempts_unlock_calibrated_type_guidance(tmp_path):
    now = [100.0]
    engine, _ = _engine(tmp_path, now)
    for index in range(3):
        now[0] = 100.0 + index * 2
        assert _attempt(engine, now) is not None
        assert engine.record_reaction("bryan", "lol that landed", now[0] + 1) is True

    profile = engine.get_profile("bryan")
    guidance = engine.get_humor_guidance("bryan")

    assert profile is not None
    assert profile.scored_attempts == 3
    assert profile.type_sample_counts["dry_wit"] == 3
    assert profile.type_scores["dry_wit"] == pytest.approx(0.8)
    assert "dry_wit (0.80, n=3)" in guidance
    assert "They're enjoying" not in guidance


def test_banter_state_is_exact_agent_and_not_global(tmp_path):
    now = [100.0]
    engine, authority = _engine(tmp_path, now)
    _grant(authority, "alice")
    for _index in range(2):
        now[0] += 1
        assert _attempt(engine, now) is not None
        assert engine.record_reaction("bryan", "lol", now[0] + 0.1) is True
    dynamics = SimpleNamespace(
        partner_frame="neutral",
        register="playful",
        humor_frame_active=True,
        escalation_invited=False,
    )

    engine.update_banter_state("lol", dynamics, user_id="bryan")

    assert engine.get_banter_state("bryan").active is True
    assert engine.get_banter_state("alice").active is False
    assert engine.get_banter_directive("alice") == ""


@pytest.mark.asyncio
async def test_emit_failure_cannot_open_humor_feedback_window(tmp_path):
    now = [100.0]
    engine, _ = _engine(tmp_path, now)

    class Harness(IncomingLogicMixin):
        user_identity = {"name": "bryan"}

    class Gate:
        def __init__(self, fail: bool) -> None:
            self.fail = fail

        async def emit(self, *_args, **_kwargs):
            if self.fail:
                raise RuntimeError("delivery failed")
            return "output-receipt-test"

    ServiceContainer.clear()
    ServiceContainer.register_instance("humor_engine", engine, required=False)
    harness = Harness()
    payload = {"user_id": "bryan", "humor_type": "dry_wit"}
    try:
        harness.output_gate = Gate(fail=True)
        with pytest.raises(RuntimeError, match="delivery failed"):
            await harness._emit_user_response(
                payload,
                "A delivered joke only if emit succeeds.",
                origin="user",
            )
        assert engine.get_status()["pending_feedback_windows"] == 0

        harness.output_gate = Gate(fail=False)
        await harness._emit_user_response(
            payload,
            "Just kidding, but only after emit succeeds.",
            origin="user",
        )
        assert engine.get_status()["pending_feedback_windows"] == 1
    finally:
        ServiceContainer.clear()


@pytest.mark.asyncio
async def test_incoming_turn_scores_previous_delivery_before_opening_next_window(
    tmp_path,
    monkeypatch,
):
    now = [100.0]
    engine, _ = _engine(tmp_path, now)
    monkeypatch.setattr(
        "core.orchestrator.mixins.incoming_logic.time.time",
        lambda: now[0],
    )

    class Harness(IncomingLogicMixin):
        user_identity = {"name": "bryan"}

    class Gate:
        async def emit(self, *_args, **_kwargs):
            return "output-receipt-test"

    ServiceContainer.clear()
    ServiceContainer.register_instance("humor_engine", engine, required=False)
    harness = Harness()
    harness.output_gate = Gate()
    try:
        assert _attempt(engine, now) is not None
        now[0] = 101.0
        harness._observe_social_turn({"user_id": "bryan"}, "lol", None)
        profile = engine.get_profile("bryan")
        assert profile is not None
        assert profile.scored_attempts == 1
        assert engine.get_status()["pending_feedback_windows"] == 0

        await harness._emit_user_response(
            {"user_id": "bryan", "humor_type": "sarcasm"},
            "Just kidding; the next response comes after the reaction was paired.",
            origin="user",
        )
        assert engine.get_status()["pending_feedback_windows"] == 1
        profile = engine.get_profile("bryan")
        assert profile is not None
        assert profile.scored_attempts == 1
    finally:
        ServiceContainer.clear()


def test_session_humor_overlay_is_non_durable_and_delete_invalidates_cache(tmp_path):
    now = [100.0]
    engine, authority = _engine(tmp_path, now, durable=False)
    assert _attempt(engine, now) is not None
    assert engine.record_reaction("bryan", "lol", 101.0) is True
    assert authority.status()["durable_record_count"] == 0

    authority.delete_agent("bryan", authorization_receipt_id="delete-humor")

    assert engine.get_banter_state("bryan").active is False
    assert engine.get_status()["profiles"] == 0
    assert engine.get_profile("bryan") is None
    assert engine.get_humor_guidance("bryan") == ""
