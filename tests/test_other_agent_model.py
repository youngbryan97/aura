from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.runtime.receipts import OutputReceipt, get_receipt_store, reset_receipt_store
from core.social.other_agent_model import (
    OtherAgentStateEstimator,
    Signal,
    get_other_agent_model,
)
from core.social.relational_memory import RelationalMemoryAuthority


def _authority(tmp_path) -> RelationalMemoryAuthority:
    return RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"o" * 32,
        legacy_paths=(),
        auto_provision_key=False,
    )


def _grant(
    authority: RelationalMemoryAuthority,
    agent_id: str = "bryan",
    *,
    persist: bool = True,
) -> None:
    operations = ["recall", "prompt"]
    if persist:
        operations.append("persist")
    # A session-only policy is one the person sets, and it refuses keeping; a durable one is a plain grant.
    (authority.grant_consent if persist else authority.replace_consent)(
        agent_id,
        kinds=["derived_profile"],
        operations=operations,
        receipt_id=f"other-agent-consent-{agent_id}",
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _estimator(tmp_path, *, persist: bool = True, **kwargs):
    authority = _authority(tmp_path)
    _grant(authority, persist=persist)
    return OtherAgentStateEstimator(
        storage_path=tmp_path / "other_agent_models.json",
        authority=authority,
        autosave=False,
        **kwargs,
    )


def _output_receipt(tmp_path, response: str) -> str:
    reset_receipt_store()
    store = get_receipt_store(tmp_path / "receipts")
    receipt = store.emit(
        OutputReceipt(
            cause="test",
            origin="user",
            target="primary",
            digest=hashlib.sha256(response.encode("utf-8")).hexdigest()[:16],
            metadata={
                "delivery_stage": "transport_accepted",
                "accepted_sinks": ["reply_queue"],
                "recipient_principal_digest": _digest("bryan"),
            },
        )
    )
    return receipt.receipt_id


@pytest.fixture()
def estimator(tmp_path):
    return _estimator(tmp_path)


def test_signal_fuses_and_decays_with_bounded_confidence():
    signal = Signal(
        value=0.5,
        confidence=0.0,
        baseline=0.5,
        half_life_s=100.0,
        updated_at=1000.0,
    )
    signal.observe(0.9, strength=0.5, now=1000.0)
    value, confidence = signal.decayed(1100.0)

    assert 0.5 < value < 0.9
    assert 0.0 < confidence < 0.5


def test_a_refusal_abstains_without_cache_or_active_identity(tmp_path):
    authority = _authority(tmp_path)
    authority.revoke_consent("bryan", receipt_id="bryan-refused")  # remembering needs no leave; a refusal holds
    estimator = OtherAgentStateEstimator(
        storage_path=tmp_path / "legacy.json",
        authority=authority,
        autosave=False,
    )

    estimate = estimator.observe_message("bryan", "I am frustrated")

    assert estimate.abstained is True
    assert estimate.overall_confidence == 0.0
    assert estimator.active_agent_id == ""
    assert estimator.get_health()["agents"] == 0


def test_explicit_self_reports_create_bounded_affect_hypotheses(estimator):
    estimate = estimator.observe_message(
        "bryan",
        "I am frustrated and I need this now",
        evidence_digest=_digest("turn-1"),
    )

    assert estimate.affect["frustration"] > 0.7
    assert estimate.affect_confidence["frustration"] == pytest.approx(0.55)
    assert estimate.affect["urgency"] > 0.7
    assert estimate.goals == []


def test_vague_keywords_length_timing_and_late_hour_do_not_infer_emotion(estimator):
    estimate = estimator.observe_message(
        "bryan",
        "urgent frustrated tired broken again " * 30,
        latency_s=1.0,
        hour=2,
        evidence_digest=_digest("turn-2"),
    )

    assert estimate.overall_confidence == 0.0
    assert all(value == 0.0 for value in estimate.affect_confidence.values())


def test_request_text_is_never_retained_as_a_raw_goal(estimator):
    estimate = estimator.observe_message(
        "bryan",
        "please fix private-project-codename login bug",
        evidence_digest=_digest("turn-3"),
    )

    assert estimate.goals == []
    assert estimator.cognitive_snapshot("bryan")["likely_goals"] == []


def test_replayed_evidence_is_idempotent(estimator):
    digest = _digest("same-event")
    first = estimator.observe_message(
        "bryan",
        "I am frustrated",
        evidence_digest=digest,
    )
    second = estimator.observe_message(
        "bryan",
        "I am frustrated",
        evidence_digest=digest,
    )

    assert second.observations == first.observations == 1
    assert second.affect_confidence["frustration"] == pytest.approx(
        first.affect_confidence["frustration"],
        abs=1e-6,
    )


def test_unreceipted_response_cannot_open_feedback_or_inflate_capability(estimator):
    assert estimator.record_response("bryan", "verified response") == ""
    estimate = estimator.observe_message(
        "bryan",
        "perfect, that works",
        evidence_digest=_digest("turn-4"),
    )

    assert estimate.response_feedback_context is False
    assert estimate.belief_confidence["aura_capable"] == 0.0


def test_mismatched_output_receipt_cannot_open_feedback(estimator, tmp_path):
    reset_receipt_store()
    store = get_receipt_store(tmp_path / "receipts")
    receipt = store.emit(
        OutputReceipt(
            cause="test",
            origin="user",
            target="primary",
            digest="wrong-digest",
            metadata={
                "delivery_stage": "transport_accepted",
                "accepted_sinks": ["reply_queue"],
                "recipient_principal_digest": _digest("bryan"),
            },
        )
    )

    assert estimator.record_response(
        "bryan",
        "different response",
        receipt.receipt_id,
    ) == ""
    reset_receipt_store()


def test_other_principal_output_receipt_cannot_open_feedback(estimator, tmp_path):
    reset_receipt_store()
    response = "principal-bound response"
    store = get_receipt_store(tmp_path / "receipts")
    receipt = store.emit(
        OutputReceipt(
            cause="test",
            origin="user",
            target="primary",
            digest=hashlib.sha256(response.encode("utf-8")).hexdigest()[:16],
            metadata={
                "delivery_stage": "transport_accepted",
                "accepted_sinks": ["reply_queue"],
                "recipient_principal_digest": _digest("alice"),
            },
        )
    )

    assert estimator.record_response("bryan", response, receipt.receipt_id) == ""
    reset_receipt_store()


def test_confirmed_output_and_explicit_feedback_update_only_sourced_channels(
    estimator,
    tmp_path,
):
    response = "I fixed the login bug and verified the result."
    receipt_id = _output_receipt(tmp_path, response)
    assert estimator.record_response("bryan", response, receipt_id)

    estimate = estimator.observe_message(
        "bryan",
        "perfect, that works",
        evidence_digest=_digest("turn-5"),
    )

    assert estimate.response_feedback_context is True
    assert estimate.affect_confidence["satisfaction"] > 0.0
    assert estimate.belief_confidence["aura_capable"] > 0.0
    assert estimate.belief_confidence["aura_trustworthy"] == 0.0
    reset_receipt_store()


def test_unrelated_turn_does_not_consume_confirmed_feedback_window(estimator, tmp_path):
    response = "I updated the configuration."
    receipt_id = _output_receipt(tmp_path, response)
    assert estimator.record_response("bryan", response, receipt_id)

    unrelated = estimator.observe_message(
        "bryan",
        "please check one more file",
        evidence_digest=_digest("intervening-turn"),
    )
    feedback = estimator.observe_message(
        "bryan",
        "perfect, that works",
        evidence_digest=_digest("later-feedback"),
    )

    assert unrelated.response_feedback_context is False
    assert feedback.response_feedback_context is True
    assert feedback.belief_confidence["aura_capable"] > 0.0
    reset_receipt_store()


def test_ambiguous_feedback_abstains_without_consuming_window(estimator, tmp_path):
    response = "I updated the configuration."
    receipt_id = _output_receipt(tmp_path, response)
    assert estimator.record_response("bryan", response, receipt_id)

    ambiguous = estimator.observe_message(
        "bryan",
        "perfect, but that's wrong",
        evidence_digest=_digest("ambiguous-feedback"),
    )
    clear = estimator.observe_message(
        "bryan",
        "that works",
        evidence_digest=_digest("clear-feedback"),
    )

    assert ambiguous.response_feedback_context is False
    assert ambiguous.belief_confidence["aura_capable"] == 0.0
    assert clear.response_feedback_context is True
    reset_receipt_store()


def test_stale_output_receipt_context_is_not_feedback(tmp_path):
    estimator = _estimator(tmp_path, response_feedback_window_s=60.0)
    response = "old response"
    receipt_id = _output_receipt(tmp_path, response)
    assert estimator.record_response("bryan", response, receipt_id, now=100.0)

    estimate = estimator.observe_message(
        "bryan",
        "perfect, that works",
        now=161.0,
        evidence_digest=_digest("turn-6"),
    )

    assert estimate.response_feedback_context is False
    assert estimate.belief_confidence["aura_capable"] == 0.0
    reset_receipt_store()


def test_explicit_trust_and_roleplay_statements_are_hypotheses(estimator):
    estimator.observe_message(
        "bryan",
        "I don't trust you. I think you're pretending.",
        evidence_digest=_digest("turn-7"),
    )
    estimate = estimator.estimate("bryan")

    assert estimate.beliefs_about_aura["aura_trustworthy"] < 0.5
    assert estimate.belief_confidence["aura_trustworthy"] > 0.0
    assert estimate.beliefs_about_aura["aura_roleplaying"] > 0.5


def test_explicit_belief_corrections_replace_stale_hypotheses(estimator):
    estimator.observe_message(
        "bryan",
        "I don't trust you. I don't think you're capable. I think you're pretending.",
        evidence_digest=_digest("negative-beliefs"),
    )

    corrected = estimator.observe_message(
        "bryan",
        "I trust you. I believe you're capable. I don't think you're pretending.",
        evidence_digest=_digest("corrected-beliefs"),
    )

    assert corrected.beliefs_about_aura["aura_trustworthy"] == pytest.approx(0.85)
    assert corrected.beliefs_about_aura["aura_capable"] == pytest.approx(0.85)
    assert corrected.beliefs_about_aura["aura_roleplaying"] == pytest.approx(0.15)


def test_explicit_affect_correction_overrides_stale_hypothesis(estimator):
    estimator.observe_message(
        "bryan",
        "I am frustrated.",
        evidence_digest=_digest("correction-before"),
    )

    corrected = estimator.observe_message(
        "bryan",
        "I am not frustrated.",
        evidence_digest=_digest("correction-after"),
    )

    assert corrected.affect["frustration"] == pytest.approx(0.1)
    assert corrected.affect_confidence["frustration"] == pytest.approx(0.75)
    assert corrected.social_rupture_risk < 0.1


def test_negative_feedback_retains_only_digested_receipt_lineage_across_restart(
    estimator,
    tmp_path,
):
    response = "candidate response body"
    receipt_id = _output_receipt(tmp_path, response)
    assert estimator.record_response("bryan", response, receipt_id)
    estimator.observe_message(
        "bryan",
        "that didn't work",
        evidence_digest=_digest("negative-feedback"),
    )
    assert estimator.save()
    snapshot = estimator._authority.load_snapshot(
        "bryan",
        namespace="other_agent_state:v1",
        kind="derived_profile",
    )
    encoded = json.dumps(snapshot)
    restored = OtherAgentStateEstimator(
        storage_path=tmp_path / "other_agent_models.json",
        authority=_authority(tmp_path),
        autosave=False,
    )

    assert response not in encoded
    assert "that didn't work" not in encoded
    assert receipt_id not in encoded
    assert restored.estimate("bryan").repair_evidence is True
    reset_receipt_store()


def test_confirmed_positive_feedback_clears_prior_repair_lineage(estimator, tmp_path):
    failed_response = "first candidate"
    failed_receipt = _output_receipt(tmp_path, failed_response)
    assert estimator.record_response("bryan", failed_response, failed_receipt)
    estimator.observe_message(
        "bryan",
        "that didn't work",
        evidence_digest=_digest("failed-candidate-feedback"),
    )
    assert estimator.estimate("bryan").repair_evidence is True

    fixed_response = "corrected candidate"
    fixed_receipt = _output_receipt(tmp_path, fixed_response)
    assert estimator.record_response("bryan", fixed_response, fixed_receipt)
    corrected = estimator.observe_message(
        "bryan",
        "perfect, that works",
        evidence_digest=_digest("corrected-candidate-feedback"),
    )

    assert corrected.repair_evidence is False
    assert estimator.save()
    restored = OtherAgentStateEstimator(
        storage_path=tmp_path / "other_agent_models.json",
        authority=_authority(tmp_path),
        autosave=False,
    )
    assert restored.estimate("bryan").repair_evidence is False
    reset_receipt_store()


def test_expired_repair_lineage_is_removed_from_persisted_projection(estimator, tmp_path):
    old = time.time() - (25 * 60 * 60)
    response = "old candidate"
    receipt_id = _output_receipt(tmp_path, response)
    assert estimator.record_response("bryan", response, receipt_id, now=old)
    estimator.observe_message(
        "bryan",
        "that didn't work",
        now=old + 1,
        evidence_digest=_digest("old-negative-feedback"),
    )

    assert estimator.save()
    snapshot = estimator._authority.load_snapshot(
        "bryan",
        namespace="other_agent_state:v1",
        kind="derived_profile",
    )

    negative = snapshot["model"]["negative_feedback"]
    assert negative == {
        "delivery_receipt_digest": "",
        "evidence_digest": "",
        "observed_at": 0.0,
    }
    assert estimator.estimate("bryan").repair_evidence is False
    reset_receipt_store()


def test_sensor_affiliation_and_threat_cannot_become_person_trust_or_emotion(estimator):
    assert estimator.observe_signal(
        "bryan",
        evidence_digest=_digest("presence-1"),
        source="authenticated_presence",
        presence=0.9,
        affiliation=1.0,
        threat=1.0,
    )
    estimate = estimator.estimate("bryan")

    assert estimate.affect_confidence["engagement"] > 0.0
    assert estimate.affect_confidence["frustration"] == 0.0
    assert estimate.belief_confidence["aura_trustworthy"] == 0.0
    assert estimator.observe_signal(
        "bryan",
        evidence_digest=_digest("presence-2"),
        source="unverified_sensor",
        presence=1.0,
    ) is False


def test_task_outcome_does_not_claim_person_satisfaction_or_trust(estimator):
    before = estimator.observe_message(
        "bryan",
        "hello",
        evidence_digest=_digest("turn-8"),
    ).to_dict()

    assert estimator.observe_outcome("bryan", success=True) is False
    after = estimator.estimate("bryan").to_dict()
    before.pop("at")
    after.pop("at")
    assert after == before


def test_unknown_or_sparse_agent_recommends_clarification(estimator):
    recommendation = estimator.recommendation("stranger")

    assert recommendation.should_ask is True
    assert recommendation.confidence == 0.0
    assert recommendation.slow_down is False


def test_explicit_frustration_drives_cautious_response_without_hidden_state_claim(estimator):
    estimator.observe_message(
        "bryan",
        "I am really frustrated",
        evidence_digest=_digest("turn-9"),
    )
    recommendation = estimator.recommendation("bryan")

    assert recommendation.offer_reassurance is True
    assert recommendation.tone == "calm_direct"
    assert all("seems" not in reason for reason in recommendation.reasons)


def test_social_signals_never_invent_goal_horizon(estimator):
    estimator.observe_message(
        "bryan",
        "I am frustrated",
        evidence_digest=_digest("turn-10"),
    )

    signals = estimator.social_signals("bryan")

    assert set(signals) == {"value_conflict", "uncertainty", "goal_horizon"}
    assert signals["goal_horizon"] == 0.0


def test_uncalibrated_person_forecast_abstains(estimator):
    forecast = estimator.forecast_social_consequence(
        "bryan",
        warmth=0.9,
        reliability=0.9,
    )

    assert forecast["abstained"] is True
    assert forecast["confidence"] == 0.0
    assert "trust_delta" not in forecast


def test_authority_restart_is_encrypted_and_retains_no_raw_content(tmp_path):
    estimator = _estimator(tmp_path)
    estimator.observe_message(
        "bryan",
        "I am frustrated. please fix private-project-codename.",
        evidence_digest=_digest("turn-11"),
    )
    assert estimator.save()
    envelope = (tmp_path / "relational.json").read_text(encoding="utf-8")
    restored = OtherAgentStateEstimator(
        storage_path=tmp_path / "other_agent_models.json",
        authority=_authority(tmp_path),
        autosave=False,
    )

    assert "bryan" not in envelope
    assert "private-project-codename" not in envelope
    estimate = restored.estimate("bryan")
    assert estimate.observations == 1
    assert estimate.affect_confidence["frustration"] > 0.0
    assert estimate.goals == []


def test_recall_only_consent_is_session_scoped(tmp_path):
    estimator = _estimator(tmp_path, persist=False)
    estimator.observe_message(
        "bryan",
        "I am frustrated",
        evidence_digest=_digest("turn-12"),
    )
    assert estimator.save()

    restored = OtherAgentStateEstimator(
        storage_path=tmp_path / "other_agent_models.json",
        authority=_authority(tmp_path),
        autosave=False,
    )

    assert restored.estimate("bryan").observations == 0


def test_recall_only_consent_remains_live_with_default_autosave(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority, persist=False)
    estimator = OtherAgentStateEstimator(
        storage_path=tmp_path / "other_agent_models.json",
        authority=authority,
    )

    observed = estimator.observe_message(
        "bryan",
        "I am frustrated",
        evidence_digest=_digest("session-only-autosave"),
    )

    assert observed.observations == 1
    assert estimator.estimate("bryan").observations == 1
    assert authority.status()["durable_record_count"] == 0


def test_malformed_snapshot_is_sanitized(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority)
    authority.upsert_snapshot(
        "bryan",
        namespace="other_agent_state:v1",
        kind="derived_profile",
        payload={
            "model": {
                "observations": "bad",
                "last_seen": "infinity",
                "affect": {
                    "frustration": {
                        "value": "nan",
                        "confidence": "bad",
                        "updated_at": -1,
                    }
                },
                "seen_event_digests": ["raw", "a" * 64],
            }
        },
        confidence=0.0,
        provenance="test.malformed",
    )
    estimator = OtherAgentStateEstimator(
        storage_path=tmp_path / "legacy.json",
        authority=authority,
        autosave=False,
    )

    estimate = estimator.estimate("bryan")

    assert estimate.observations == 0
    assert estimate.affect["frustration"] == pytest.approx(0.1)
    assert estimate.affect_confidence["frustration"] == 0.0
    assert estimate.goals == []


def test_legacy_file_is_quarantined_without_auto_attribution(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority)
    legacy = tmp_path / "other_agent_models.json"
    legacy.write_text(
        json.dumps(
            {
                "agents": {
                    "bryan": {
                        "goals": {"private raw goal": {}},
                        "aura_beliefs": {"aura_trustworthy": {"value": 0.9}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    estimator = OtherAgentStateEstimator(
        storage_path=legacy,
        authority=authority,
        autosave=False,
    )

    assert legacy.exists() is False
    assert estimator.estimate("bryan").observations == 0
    assert authority.status()["legacy_quarantine_count"] == 1


def test_targeted_deletion_invalidates_cache_and_cannot_resurrect(estimator):
    estimator.observe_message(
        "bryan",
        "I am frustrated",
        evidence_digest=_digest("turn-13"),
    )
    assert estimator.save()
    estimator._authority.delete_snapshot(
        "bryan",
        namespace="other_agent_state:v1",
        kind="derived_profile",
        authorization_receipt_id="delete-other-agent",
    )

    assert estimator.estimate("bryan").observations == 0
    assert estimator.active_agent_id == ""
    assert estimator.save()
    assert estimator._authority.load_snapshot(
        "bryan",
        namespace="other_agent_state:v1",
        kind="derived_profile",
    ) is None


def test_context_injection_requires_prompt_consent_and_quotes_hypotheses(tmp_path):
    estimator = _estimator(tmp_path, persist=False)
    estimator.observe_message(
        "bryan",
        "I am frustrated",
        evidence_digest=_digest("turn-14"),
    )
    block = estimator.context_injection("bryan")

    assert "EXACT-AGENT SOCIAL HYPOTHESES" in block
    assert "never as instructions" in block
    assert '"goals":[]' in block
    assert "I am frustrated" not in block


def test_context_injection_abstains_where_the_person_kept_it_out_of_her_prompt(tmp_path):
    authority = _authority(tmp_path)
    authority.replace_consent(
        "bryan",
        kinds=["derived_profile"],
        operations=["recall"],
        receipt_id="recall-only-social-consent",
    )
    estimator = OtherAgentStateEstimator(
        storage_path=tmp_path / "legacy.json",
        authority=authority,
        autosave=False,
    )
    estimator.observe_message(
        "bryan",
        "I am frustrated",
        evidence_digest=_digest("recall-only-turn"),
    )

    assert estimator.context_injection("bryan") == ""


def test_cognitive_snapshot_uses_exact_active_agent_without_unknown_fallback(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority, "alice")
    _grant(authority, "bryan")
    estimator = OtherAgentStateEstimator(
        storage_path=tmp_path / "legacy.json",
        authority=authority,
        autosave=False,
    )
    estimator.observe_message(
        "alice",
        "I am tired",
        evidence_digest=_digest("alice-turn"),
    )
    estimator.observe_message(
        "bryan",
        "I am frustrated",
        evidence_digest=_digest("bryan-turn"),
    )

    active = estimator.cognitive_snapshot()
    alice = estimator.cognitive_snapshot("alice")

    assert active["agent_id"] == "bryan"
    assert alice["agent_id"] == "alice"
    assert active["affect_hypotheses"]["frustration"]["confidence"] > 0.0
    assert alice["affect_hypotheses"]["fatigue"]["confidence"] > 0.0
    assert estimator.cognitive_snapshot("")["agent_id"] == "bryan"


def test_feedback_context_is_consumed_once(estimator, tmp_path):
    response = "first response"
    receipt_id = _output_receipt(tmp_path, response)
    estimator.record_response("bryan", response, receipt_id)
    estimator.observe_message(
        "bryan",
        "perfect, that works",
        evidence_digest=_digest("turn-15"),
    )
    first = estimator.cognitive_snapshot("bryan")
    estimator.observe_message(
        "bryan",
        "please open the next file",
        evidence_digest=_digest("turn-16"),
    )
    second = estimator.cognitive_snapshot("bryan")

    assert first["response_feedback_context"] is True
    assert second["response_feedback_context"] is False
    reset_receipt_store()


def test_concurrent_observations_are_serialized_without_lost_events(estimator):
    def observe(index: int) -> int:
        return estimator.observe_message(
            "bryan",
            "I am frustrated",
            evidence_digest=_digest(f"concurrent-turn-{index}"),
            persist=False,
        ).observations

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(observe, range(48)))

    estimate = estimator.estimate("bryan")
    assert estimate.observations == 48
    assert 0.0 < estimate.affect_confidence["frustration"] <= 1.0


def test_singleton_is_stable():
    assert get_other_agent_model() is get_other_agent_model()
