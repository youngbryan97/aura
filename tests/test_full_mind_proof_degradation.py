"""Contracts for the full-mind proof split: authorship vs state-completeness.

Live incident (2026-07-13): during a self-healing window the desktop gate
discarded REAL Cortex replies (think invoked, accepted, high confidence,
len=126) because ancillary state proofs (mind-snapshot readiness) weren't
green, and served fail-closed apologies on basic conversation — twice in one
exchange. Worse, the refusal log printed three flags that were all fine.

The contract now separates:
* SPEAKER-IDENTITY proofs (``authentic_cognitive_reply``) — the text came
  from her real cognitive engine, not repair/legacy machinery. Never waived.
* STATE-COMPLETENESS proofs — snapshot readiness, control receipts,
  subsystem health. When only these fail, the reply is SERVED with the
  degradation disclosed; ``full_mind_missing_proofs`` names each gap so no
  future incident is diagnosed blind.
"""
from __future__ import annotations

import pytest
from tests.chat_lane_support import patch_chat_lane

pytestmark = pytest.mark.unit


def _force_full_mind_runtime(monkeypatch, chat_routes):
    """None
    """
    # Sweep every lane: these probes are imported by several modules now, and
    # patching one leaves the rest running the real check.
    for name in (
        "_runtime_kernel_available",
        "_runtime_cognitive_engine_available",
        "_runtime_memory_available",
        "_runtime_tool_governance_available",
        "_runtime_substrate_voice_available",
    ):
        patch_chat_lane(monkeypatch, name, lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_inference_available", lambda *a, **k: True)


_READY_LANE = {
    "conversation_ready": True,
    "state": "ready",
    "desired_model": "Cortex (32B)",
    "foreground_endpoint": "Cortex",
}


def _green_trace() -> dict:
    return {
        "engine_think_invoked": True,
        "cognitive_engine_reply_accepted": True,
        "bounded_contract_used": False,
        "legacy_fallback_used": False,
        "live_mind_context_present": True,
        "live_mind_snapshot_present": True,
        "live_mind_snapshot_ready": True,
        "live_mind_required_subsystems_ok": True,
        "response_path": "cognitive_engine",
        "live_mind_controls_bound": True,
        "live_mind_generation_controls": {
            "temperature": 0.58,
            "top_p": 0.85,
            "clean_user_surface_recurrent_loops": 1,
            "clean_user_surface_steering_alpha": 0.3,
        },
        "live_mind_surface_control_receipt": {
            "enabled": False,
            "live_mind_controls_bound": True,
            "clean_user_surface_contract": True,
            "surface_quality_gate_enabled": False,
            "surface_quality_gate_passed": True,
            "surface_quality_gate_attempts": 0,
            "surface_quality_gate_reasons": [],
            "applied": True,
        },
        "live_mind_controls_worker_applied": True,
        # Single-owner generation proof: a live-mind reply consumes exactly one
        # foreground model generation (no double-generation, no zero-owner
        # theatre) — required for authentic_cognitive_reply.
        "foreground_model_generation_count": 1,
        "foreground_model_generation_consumed": True,
        "foreground_model_generation_transaction_id": "full-mind-test-transaction",
    }


def _payload(chat_routes, trace: dict, *, confidence: str = "high") -> dict:
    return chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status=dict(_READY_LANE),
        response_confidence=confidence,
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace=trace,
    )


def test_green_trace_proves_full_mind_with_no_missing_proofs(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    payload = _payload(chat_routes, _green_trace())
    assert payload["authentic_cognitive_reply"] is True
    assert payload["full_mind_path"] is True
    assert payload["full_mind_missing_proofs"] == []


def test_degraded_snapshot_keeps_authorship_and_names_the_gap(monkeypatch):
    """The live-incident anatomy: real reply, snapshot not ready. The
    contract must keep authorship TRUE (the gate serves + discloses) and
    name exactly what was missing."""
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _green_trace()
    trace["live_mind_snapshot_ready"] = False
    payload = _payload(chat_routes, trace)
    assert payload["authentic_cognitive_reply"] is True
    assert payload["full_mind_path"] is False
    assert "live_mind_snapshot_not_ready" in payload["full_mind_missing_proofs"]


def test_unbound_controls_keep_authorship_and_name_the_gap(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _green_trace()
    trace.pop("live_mind_surface_control_receipt")
    trace["live_mind_controls_worker_applied"] = False
    trace["live_mind_controls_bound"] = False
    payload = _payload(chat_routes, trace)
    assert payload["authentic_cognitive_reply"] is True
    assert payload["full_mind_path"] is False
    assert "live_mind_controls_unbound" in payload["full_mind_missing_proofs"]


def test_unavailable_subsystem_keeps_authorship_and_is_named(monkeypatch):
    """Self-healing window: a required subsystem is down. Authorship holds;
    the subsystem gap is named for the disclosure."""
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    patch_chat_lane(monkeypatch, "_runtime_memory_available", lambda: False)
    payload = _payload(chat_routes, _green_trace())
    assert payload["authentic_cognitive_reply"] is True
    assert payload["full_mind_path"] is False
    assert any(
        proof.startswith("subsystem:") for proof in payload["full_mind_missing_proofs"]
    )


def test_repair_text_is_never_authentic(monkeypatch):
    """The anti-theater guarantee is untouched: bounded-repair machinery
    can never author Aura speech, whatever else is green."""
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _green_trace()
    trace["bounded_contract_used"] = True
    payload = _payload(chat_routes, trace)
    assert payload["authentic_cognitive_reply"] is False
    assert payload["full_mind_path"] is False
    assert "bounded_repair_authored_text" in payload["full_mind_missing_proofs"]


def test_legacy_fallback_is_never_authentic(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _green_trace()
    trace["legacy_fallback_used"] = True
    payload = _payload(chat_routes, trace)
    assert payload["authentic_cognitive_reply"] is False
    assert "legacy_fallback_authored_text" in payload["full_mind_missing_proofs"]


def test_failed_engine_reply_is_never_authentic(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _green_trace()
    trace["cognitive_engine_reply_failed"] = True
    payload = _payload(chat_routes, trace)
    assert payload["authentic_cognitive_reply"] is False
    assert "engine_reply_failed" in payload["full_mind_missing_proofs"]


def test_low_confidence_downgrades_certification_not_authorship(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    payload = _payload(chat_routes, _green_trace(), confidence="low")
    assert payload["authentic_cognitive_reply"] is True
    assert payload["answer_delivery_proven"] is True
    assert payload["certification_complete"] is False
    assert "confidence:low" in payload["full_mind_missing_proofs"]


def test_semantic_completion_requires_positive_worker_receipt(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _green_trace()
    trace["semantic_completion_contract_expected"] = True

    payload = _payload(chat_routes, trace)

    assert payload["authentic_cognitive_reply"] is True
    assert payload["authored_answer_completion_proven"] is False
    assert payload["answer_delivery_proven"] is False
    assert "authored_answer_incomplete" in payload["full_mind_missing_proofs"]


def test_semantic_completion_accepts_positive_worker_receipt(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _green_trace()
    trace.update(
        {
            "semantic_completion_contract_expected": True,
            "semantic_completion_receipt_present": True,
            "semantic_completion_satisfied": True,
            "semantic_completion_incomplete": False,
        }
    )

    payload = _payload(chat_routes, trace)

    assert payload["authored_answer_completion_proven"] is True
    assert payload["answer_delivery_proven"] is True


def test_generation_ownership_requires_transaction_identity(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _green_trace()
    trace.pop("foreground_model_generation_transaction_id")

    payload = _payload(chat_routes, trace)

    assert payload["single_owner_model_generation_proven"] is False
    assert payload["authentic_cognitive_reply"] is False
    assert payload["answer_delivery_proven"] is False
    assert "foreground_model_generation_ownership_unproven" in payload[
        "full_mind_missing_proofs"
    ]


def test_unknown_response_path_is_never_authentic(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _green_trace()
    trace["response_path"] = "mystery_repair_lane"
    payload = _payload(chat_routes, trace)
    assert payload["authentic_cognitive_reply"] is False
    assert "response_path:mystery_repair_lane" in payload["full_mind_missing_proofs"]
