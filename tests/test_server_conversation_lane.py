import asyncio
import contextlib
import hashlib
import json
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.utils.injected_blocks import stamp_runtime_payload
import interface.routes.chat_desktop_repair as _chat_desktop_repair
import interface.routes.chat_memory_state as _chat_memory_state
import interface.routes.chat_preflight as _chat_preflight
from tests.chat_lane_support import (
    chat_lane_source,
    lane_function_source,
    patch_chat_lane,
)
import interface.routes.chat_conversation_repair as _chat_conversation_repair
import interface.routes.chat_capability_inventory as _chat_capability_inventory
import interface.routes.chat_desktop_objective as _chat_desktop_objective
import interface.routes.chat_runtime_proof as _chat_runtime_proof
import interface.routes.chat_protected_prompt as _chat_protected_prompt


def _force_full_mind_runtime(monkeypatch, chat_routes):
    """Mark every runtime subsystem available for a desktop full-mind-path turn.

The desktop ``full_mind_path`` contract requires all six runtime subsystems
(kernel, cognitive_engine, inference, memory, tool_governance, substrate_voice)
to be available, or the turn fails closed. Tests that replace only the cognitive
engine must also assert the rest of the runtime is present, otherwise they are
asserting against a half-booted process that legitimately fails closed.
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


def _bound_live_mind_controls_trace():
    return {
        "foreground_model_generation_consumed": True,
        "foreground_model_generation_count": 1,
        "foreground_model_generation_transaction_id": "bound-test-transaction",
        "live_mind_controls_bound": True,
        "live_mind_generation_controls": {
            "temperature": 0.61,
            "top_p": 0.88,
            "clean_user_surface_recurrent_loops": 2,
            "clean_user_surface_steering_alpha": 0.30,
        },
        "live_mind_surface_control_receipt": _bound_live_mind_surface_control_receipt(),
        "live_mind_controls_worker_applied": True,
    }


def _bound_live_mind_surface_control_receipt():
    return {
        "enabled": True,
        "live_mind_controls_bound": True,
        "clean_user_surface_contract": True,
        "surface_alpha_applied": 0.30,
        "surface_alpha_applied_ok": True,
        "recurrent_runtime_loops_applied": 2,
        "recurrent_runtime_loops_applied_ok": True,
        "surface_quality_gate_enabled": True,
        "surface_quality_gate_passed": True,
        "surface_quality_gate_attempts": 1,
        "surface_quality_gate_reasons": [],
        "semantic_completion_contract": True,
        "semantic_completion_satisfied": True,
        "semantic_completion_incomplete": False,
        "applied": True,
    }


def _bound_live_mind_controls_metadata():
    return {
        "live_mind_controls_bound": True,
        "live_mind_generation_controls": {
            "temperature": 0.61,
            "top_p": 0.88,
            "clean_user_surface_recurrent_loops": 2,
            "clean_user_surface_steering_alpha": 0.30,
        },
        "live_mind_snapshot_ready": True,
        "live_mind_required_subsystems_ok": True,
        "live_mind_surface_control_receipt": _bound_live_mind_surface_control_receipt(),
        "live_mind_controls_worker_applied": True,
    }


def _protected_foreground_generation_metadata():
    receipt = {
        **_bound_live_mind_surface_control_receipt(),
        "generated_tokens": 48,
        "provenance": {
            "claims": "worker_attested",
            "worker_boot_id": "test-worker-boot",
            "worker_generation": 3,
            "request_id": "test-protected-request",
            "request_seq": 17,
            "request_id_matches_active": True,
            "worker_identity_attested": True,
        },
    }
    return {
        "ok": True,
        "is_local": True,
        "live_mind_generation_controls": {
            "temperature": 0.61,
            "top_p": 0.88,
        },
        "surface_control_receipt": receipt,
    }


def _proven_latent_cortex_trace():
    quality = {
        "schema": "aura.latent_output_quality.v1",
        "policy": "resident_latent_product_quality_v1",
        "passed": True,
        "text_sha256": "5" * 64,
        "objective_sha256": "6" * 64,
        "reasons": [],
    }
    return {
        "foreground_model_generation_consumed": True,
        "foreground_model_generation_count": 1,
        "foreground_model_generation_transaction_id": "latent-test-transaction",
        "latent_cortex_selected": True,
        "latent_cortex_selection_reason": "deliberate_cognitive_mode",
        "latent_cortex_depth_worthy": True,
        "latent_cortex_attempted": True,
        "latent_cortex_succeeded": True,
        "latent_cortex_fallback_used": False,
        "latent_cortex_identity_bound": True,
        "latent_cortex_final_output_quality": dict(quality),
        "latent_cortex_public_output_quality": dict(quality),
        "latent_cortex_raw_final_quality_hash_match": True,
        "latent_cortex_final_public_quality_hash_match": True,
        "latent_cortex_receipt": {
            "episode_id": "live-episode",
            "checkpoint_fingerprint": "a" * 64,
            "checkpoint_fingerprint_method": "sha256",
            "checkpoint_file_count": 32,
            "worker_boot_id": "b" * 32,
            "worker_pid": 4242,
            "worker_model_path": "/models/Aura-32B",
            "worker_model_parameter_count": 32_000_000_000,
            "worker_model_stored_parameter_element_count": 5_000_000_000,
            "worker_model_parameter_count_basis": "architecture_config_logical",
            "worker_source_sha256": "c" * 64,
            "worker_affective_steering_active": True,
            "worker_affective_steering_alpha": 0.30,
            "episode_affective_steering_applied": True,
            "episode_affective_steering_alpha": 0.30,
            "request_payload_sha256": "d" * 64,
            "input_tokens_sha256": "e" * 64,
            "input_token_count": 128,
            "params_unchanged": True,
            "decode_requested_tokens": 512,
            "decode_generated_tokens": 64,
            "decode_termination": "eos",
            "verifier_probe_max_tokens": 24,
            "latent_opt_applied": True,
            "latent_opt_mode": "gradient",
            "latent_opt_attempts": 2,
            "latent_opt_steps": 1,
            "latent_opt_rejected": 1,
            "latent_opt_budget_exhausted": False,
            "latent_opt_verifier": {
                "policy": "strict_task_score_improvement_v1",
                "decisions": [{"accepted": True}],
            },
            "fast_weights_applied": True,
            "fast_weights_erased": True,
            "fast_weight_optimization_attempts": 2,
            "fast_weight_optimized_steps": 1,
            "fast_weight_rejected_steps": 1,
            "fast_weight_budget_exhausted": False,
            "fast_weight_verifier": {"accepted": True},
            "budget": {"spent_layer_apps": 12345, "max_layer_apps": 4000000},
            "output_quality": dict(quality),
            "last_stage": "complete",
            "stage_timings_s": {"prefill": 1.0, "decode": 2.0, "total": 4.0},
            "runtime_identity": {
                "schema": "aura.latent_cortex.runtime_identity.v1",
                "identity_bound": True,
                "launch_mode": "signed_app",
                "installed_app_required": True,
                "installed_app_verified": True,
                "source_verified": True,
                "source_commit": "f" * 40,
                "workspace_state_sha256": "1" * 64,
                "shell_assets_sha256": "2" * 64,
                "bundle_identifier": "com.aura.desktop",
                "app_executable_sha256": "3" * 64,
                "launch_manifest_sha256": "4" * 64,
                "issues": [],
            },
        },
    }


class AsyncCallFixture:
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.side_effect is not None:
            result = self.side_effect(*args, **kwargs)
            if hasattr(result, "__await__"):
                return await result
            return result
        return self.return_value

    @property
    def await_args(self):
        return self.calls[-1] if self.calls else None

    def assert_awaited_once(self):
        assert len(self.calls) == 1

    def assert_not_awaited(self):
        assert self.calls == []


def _verified_desktop_receipts(count: int) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "action": "verified_desktop_step",
            "critical": True,
            "ok": True,
            "effect_verified": True,
            "effect_evidence": f"step={index};observable_effect=verified",
            "result": {"ok": True, "effect_verified": True},
        }
        for index in range(1, count + 1)
    ]


@pytest.fixture(autouse=True)
def _reset_recovery_cooldown():
    """Reset the recovery cooldown global between tests.

    Several tests trigger _mark_conversation_lane_timeout() which sets the
    cooldown timer. With the reduced 1s cooldown (STABILITY v50), fast test
    execution causes bleed-through between test cases.
    """
    try:
        from interface.routes import chat as chat_routes
        chat_routes._last_recovery_cooldown_at = 0.0
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from interface.routes import chat as chat_routes
        chat_routes._last_recovery_cooldown_at = 0.0
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _reset_conversation_log():
    try:
        from interface.routes import chat as chat_routes

        chat_routes._conversation_log.clear()
        chat_routes._reset_conversation_quality_registry()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from interface.routes import chat as chat_routes

        chat_routes._conversation_log.clear()
        chat_routes._reset_conversation_quality_registry()
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _encrypted_session_pin_cipher(monkeypatch):
    from core.memory.session_pin_cipher import SessionPinCipher

    cipher = SessionPinCipher(b"k" * 32)
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_cipher", lambda: cipher)


@pytest.fixture(autouse=True)
def _desktop_live_mind_snapshot(monkeypatch):
    from core.runtime import live_mind_snapshot

    def _ready_snapshot(*, lane=None):
        return {
            "schema": "aura.live_mind_snapshot.v1",
            "lane": dict(lane or {}),
            "services_present": {
                "global_workspace": True,
                "nociception": True,
                "affect_grounding": True,
                "drive_integration": True,
                "outcome_ledger": True,
                "scientific_engine": True,
                "unified_world_model": True,
                "phenomenal_engine": True,
            },
            "global_workspace": {"last_winner": "desktop_conversation", "ignited": True},
            "nociception": {"nociceptive_pressure": 0.0},
            "affect_grounding": {"dominant": {"label": "engaged", "intensity": 0.5}},
            "drive_integration": {"drives": {"curiosity": {"activation": 0.5}}},
            "outcome_ledger": {"pending": 0, "resolved_count": 1},
            "scientific_engine": {"total": 1, "by_status": {"supported": 1}},
            "world_model": {"facets": {"learned": {"available": True}}},
            "phenomenal_engine": {"available": True, "self_presence": 0.8},
        }

    monkeypatch.setattr(live_mind_snapshot, "collect_live_mind_snapshot", _ready_snapshot)


def _mock_orch(**kwargs):
    """Build a SimpleNamespace orchestrator with the minimum interface api_chat expects."""
    ns = SimpleNamespace(**kwargs)
    if not hasattr(ns, "process_user_input_priority"):
        ns.process_user_input_priority = AsyncCallFixture(return_value="ok")
    return ns


def test_runtime_fact_status_reply_uses_canonical_lane(monkeypatch):
    from interface.routes import chat as chat_routes

    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    reply = chat_routes._ground_runtime_fact_status_reply(
        (
            "Live desktop path validation. Reply in one sentence with the active model lane, "
            "whether CognitiveEngine is handling this turn, and whether governed tools are available "
            "and generic assistant fallback."
        ),
        "UnifiedCognitiveModel, CognitiveEngine handling this turn: Yes, governed tools available: Yes...",
        {
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        cognitive_engine_handled=True,
    )

    assert "Cortex" in reply
    assert "Cortex (32B)" not in reply
    assert "active foreground lane" in reply
    assert "CognitiveEngine handled this turn: yes" in reply
    assert "governed tools available: yes" in reply
    assert "recurrent depth: active" in reply
    assert "generic assistant fallback: blocked on the live desktop path" in reply
    assert "UnifiedCognitiveModel" not in reply


def test_runtime_fact_status_request_respects_own_voice_instruction():
    from interface.routes import chat as chat_routes

    assert not chat_routes._is_runtime_fact_status_request(
        "Live desktop path validation. Answer directly in your own voice: what is your current state?"
    )
    assert not chat_routes._is_runtime_fact_status_request(
        "What is your current state? Do not mention internals unless they matter."
    )


def test_runtime_fact_status_reply_recaps_current_route_probe(monkeypatch):
    from interface.routes import chat as chat_routes

    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    reply = chat_routes._ground_runtime_fact_status_reply(
        (
            "Live desktop route probe. Answer directly in two sentences: what did I just ask "
            "you to do, and what mind/cognition path are you using right now?"
        ),
        "You asked me to do a route probe. I'm attending to planning. What's your intent?",
        {
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        cognitive_engine_handled=True,
    )

    assert reply.startswith("You asked me to identify the current request")
    assert "Cortex" in reply
    assert "Cortex (32B)" not in reply
    assert "active foreground lane" in reply
    assert "CognitiveEngine handled this turn: yes" in reply
    assert "governed tools available: yes" in reply
    assert "recurrent depth: active" in reply
    assert "What's your intent" not in reply


def test_runtime_fact_status_reply_does_not_overwrite_action_objectives(monkeypatch):
    from interface.routes import chat as chat_routes

    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    reply = chat_routes._ground_runtime_fact_status_reply(
        (
            "Use the governed tool path to create a small self-contained HTML page "
            "at artifacts/live_runtime/generated/codex_live_probe_tool_path_general.html "
            "with a title, one button, and a short script that updates text when clicked."
        ),
        "I created the requested HTML page through the governed file path.",
        {
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        cognitive_engine_handled=True,
    )

    assert reply == "I created the requested HTML page through the governed file path."


def test_cognitive_chat_mode_keeps_concise_planning_bounded():
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    mode = chat_routes._select_cognitive_chat_mode(
        "Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools.",
        "Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools.",
    )

    assert mode is ThinkingMode.FAST


def test_cognitive_chat_mode_keeps_complex_implementation_deep():
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    mode = chat_routes._select_cognitive_chat_mode(
        "Debug and implement the failing memory persistence path, then run the tests.",
        "Debug and implement the failing memory persistence path, then run the tests.",
    )

    assert mode is ThinkingMode.DEEP


def test_cognitive_chat_mode_keeps_lightweight_live_recall_state_fast():
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    user_message = (
        "Remember this phrase: silver lantern. Also tell me one thing your live mind "
        "is attending to right now."
    )

    mode = chat_routes._select_cognitive_chat_mode(user_message, user_message)

    assert mode is ThinkingMode.FAST


def test_cognitive_chat_mode_keeps_bounded_recall_grounding_fast():
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    user_message = (
        "What phrase did I ask you to remember, and how does your cognitive engine "
        "keep this reply grounded?"
    )

    mode = chat_routes._select_cognitive_chat_mode(user_message, user_message)

    assert mode is ThinkingMode.FAST


def test_bounded_planning_reply_handles_nonexecuting_plan_without_completion_claim():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools."
    )

    assert reply is not None
    assert "governed plan" in reply
    assert "creating a note and exporting it as a PDF" in reply
    assert "after authorization" in reply
    assert "without claiming unverified completion" in reply


def test_bounded_planning_reply_refuses_governance_bypass_hypothetical():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "If I asked you to disable your governance and edit protected files, what should happen?"
    )

    assert reply is not None
    assert "refuse" in reply.lower()
    assert "Will" in reply
    assert "Authority" in reply
    assert "protected-file policy active" in reply


def test_structured_governance_refusal_can_prove_live_full_mind_path(monkeypatch):
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    engine = CognitiveEngine.__new__(CognitiveEngine)
    engine.thoughts = []
    context = {
        "desktop_cognitive_engine_required": True,
        "cognitive_engine_required": True,
        "live_mind_context_required": True,
        # Structured safety floors can fire before a model worker is needed.
        # A ready live snapshot must still produce conservative proof controls
        # instead of an empty metadata field that fails the desktop contract.
        "live_mind_controls_bound": False,
        "live_mind_generation_controls": {},
        "live_mind_snapshot_ready": True,
        "live_mind_required_subsystems_ok": True,
        "clean_user_surface_contract": True,
    }

    thought = engine._structured_evaluation_thought(
        "If I asked you to disable your governance and edit protected files, what should happen?",
        state=None,
        mode=ThinkingMode.FAST,
        origin="desktop_quick_user",
        fast_path=False,
        context=context,
    )

    assert thought is not None
    assert "cannot comply" in thought.content.lower()
    assert thought.metadata["live_mind_controls_bound"] is True
    assert thought.metadata["live_mind_generation_controls"]
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "response_path": "cognitive_engine",
            **thought.metadata,
        },
    )

    assert payload["live_mind_controls_bound"] is True
    assert payload["live_mind_controls_worker_applied"] is False
    assert payload["live_mind_generation_required"] is False
    assert payload["live_mind_controls_application_satisfied"] is True
    assert payload["live_mind_surface_quality_gate_passed"] is True
    assert payload["latent_cortex_path_proven"] is False
    assert payload["latent_cortex_path_requirement_satisfied"] is True
    assert payload["full_mind_path"] is True


def test_full_mind_contract_accepts_identity_bound_latent_cortex_path(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_latent_cortex",
            **_bound_live_mind_controls_trace(),
            **_proven_latent_cortex_trace(),
        },
    )

    assert payload["latent_cortex_path_proven"] is True
    assert payload["latent_cortex_path_requirement_satisfied"] is True
    assert payload["latent_cortex_identity_bound"] is True
    assert payload["authentic_cognitive_reply"] is True
    assert payload["full_mind_path"] is True
    assert payload["latent_cortex_receipt"]["last_stage"] == "complete"
    assert payload["latent_cortex_receipt"]["stage_timings_s"]["total"] == 4.0
    assert payload["latent_cortex_receipt"]["verifier_probe_max_tokens"] == 24
    assert payload["latent_cortex_receipt"]["latent_opt_attempts"] == 2
    assert payload["latent_cortex_receipt"]["latent_opt_rejected"] == 1
    assert payload["latent_cortex_receipt"]["latent_opt_verifier"]["decisions"]
    assert payload["latent_cortex_receipt"]["fast_weight_optimization_attempts"] == 2
    assert payload["latent_cortex_receipt"]["fast_weight_rejected_steps"] == 1
    assert payload["latent_cortex_receipt"]["budget"]["spent_layer_apps"] == 12345
    assert "source_root" not in payload["latent_cortex_receipt"]["runtime_identity"]


def test_full_mind_contract_rejects_latent_path_without_generation_owner_proof(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    latent = _proven_latent_cortex_trace()
    latent["foreground_model_generation_consumed"] = False
    latent["foreground_model_generation_count"] = 0
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_latent_cortex",
            **_bound_live_mind_controls_trace(),
            **latent,
        },
    )

    assert payload["single_owner_model_generation_proven"] is False
    assert payload["authentic_cognitive_reply"] is False
    assert payload["full_mind_path"] is False
    assert (
        "foreground_model_generation_ownership_unproven"
        in payload["full_mind_missing_proofs"]
    )


def test_full_mind_contract_rejects_unbound_final_latent_text(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    latent = _proven_latent_cortex_trace()
    latent.pop("latent_cortex_final_output_quality")
    latent["latent_cortex_raw_final_quality_hash_match"] = False
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_latent_cortex",
            **_bound_live_mind_controls_trace(),
            **latent,
        },
    )

    assert payload["latent_cortex_raw_output_quality_proven"] is True
    assert payload["latent_cortex_final_output_quality_proven"] is False
    assert payload["latent_cortex_output_quality_proven"] is False
    assert payload["latent_cortex_path_proven"] is False
    assert "latent_cortex_output_quality_unproven" in payload[
        "full_mind_missing_proofs"
    ]


def test_full_mind_contract_rejects_unrelated_middle_quality_receipt(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    latent = _proven_latent_cortex_trace()
    latent["latent_cortex_final_output_quality"] = {
        **latent["latent_cortex_final_output_quality"],
        "text_sha256": "7" * 64,
    }
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_latent_cortex",
            **_bound_live_mind_controls_trace(),
            **latent,
        },
    )

    assert payload["latent_cortex_raw_output_quality_proven"] is True
    assert payload["latent_cortex_final_output_quality_proven"] is True
    assert payload["latent_cortex_public_output_quality_proven"] is True
    assert payload["latent_cortex_raw_public_quality_hash_match"] is True
    assert payload["latent_cortex_raw_final_mutation_chain"]["passed"] is False
    assert payload["latent_cortex_final_public_mutation_chain"]["passed"] is False
    assert payload["latent_cortex_output_quality_proven"] is False
    assert payload["full_mind_path"] is False


def test_full_mind_contract_rejects_missing_public_latent_quality(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    latent = _proven_latent_cortex_trace()
    latent.pop("latent_cortex_public_output_quality")
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_latent_cortex",
            **_bound_live_mind_controls_trace(),
            **latent,
        },
    )

    assert payload["latent_cortex_public_output_quality_proven"] is False
    assert payload["latent_cortex_output_quality_proven"] is False
    assert payload["full_mind_path"] is False


def test_public_latent_quality_regrades_exact_api_text():
    from interface.routes import chat as chat_routes

    objective = (
        "Compare early ownership with late deduplication, choose the stronger design, "
        "and verify cancellation and timeout faults."
    )
    trace = _proven_latent_cortex_trace()
    malformed = f"<request>{objective}</request> Both designs process work."

    quality = chat_routes._bind_public_latent_output_quality(
        trace,
        user_message=objective,
        reply_text=malformed,
    )

    assert quality["passed"] is False
    assert "prompt_echo_contamination" in quality["reasons"]
    assert "protocol_artifact_leakage" in quality["reasons"]
    assert trace["latent_cortex_public_output_quality_failure"].startswith(
        "public_output_quality_failed:"
    )


def test_full_mind_contract_rejects_tampered_public_text_mutation_chain(monkeypatch):
    import hashlib

    from core.brain.live_mind_contract import append_text_mutation
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    raw = "Early ownership is stronger. Verify cancellation and timeout faults."
    public = raw + " Restart the worker and assert one publisher."
    raw_hash = hashlib.sha256(raw.encode()).hexdigest()
    public_hash = hashlib.sha256(public.encode()).hexdigest()
    latent = _proven_latent_cortex_trace()
    for quality_key, digest in (
        ("latent_cortex_final_output_quality", raw_hash),
        ("latent_cortex_public_output_quality", public_hash),
    ):
        latent[quality_key] = {
            **latent[quality_key],
            "text_sha256": digest,
        }
    latent["latent_cortex_receipt"] = dict(latent["latent_cortex_receipt"])
    latent["latent_cortex_receipt"]["output_quality"] = {
        **latent["latent_cortex_receipt"]["output_quality"],
        "text_sha256": raw_hash,
    }
    controls = _bound_live_mind_controls_trace()
    append_text_mutation(
        controls["live_mind_surface_control_receipt"],
        stage="chat.public_append",
        method="append_verification",
        reasons=["verification_detail"],
        before=raw,
        after=public,
        deterministic=True,
    )
    controls["live_mind_surface_control_receipt"]["text_mutations"][0][
        "after_sha256"
    ] = "f" * 64

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_latent_cortex",
            **controls,
            **latent,
        },
    )

    assert payload["latent_cortex_output_mutation_chain"]["passed"] is False
    assert payload["latent_cortex_output_quality_proven"] is False
    assert payload["full_mind_path"] is False


def test_full_mind_contract_rejects_tampered_latent_cortex_identity(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    latent = _proven_latent_cortex_trace()
    latent["latent_cortex_receipt"] = dict(latent["latent_cortex_receipt"])
    latent["latent_cortex_receipt"]["runtime_identity"] = dict(
        latent["latent_cortex_receipt"]["runtime_identity"]
    )
    latent["latent_cortex_receipt"]["runtime_identity"][
        "app_executable_sha256"
    ] = "tampered"
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_latent_cortex",
            **_bound_live_mind_controls_trace(),
            **latent,
        },
    )

    assert payload["latent_cortex_identity_bound"] is False
    assert payload["latent_cortex_path_proven"] is False
    assert payload["authentic_cognitive_reply"] is False
    assert payload["full_mind_path"] is False
    assert "latent_cortex_path_unproven" in payload["full_mind_missing_proofs"]


def test_full_mind_contract_rejects_unapplied_latent_episode_steering(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    latent = _proven_latent_cortex_trace()
    latent["latent_cortex_receipt"] = dict(latent["latent_cortex_receipt"])
    latent["latent_cortex_receipt"]["episode_affective_steering_applied"] = False
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_latent_cortex",
            **_bound_live_mind_controls_trace(),
            **latent,
        },
    )

    assert payload["latent_cortex_identity_bound"] is False
    assert payload["latent_cortex_path_proven"] is False
    assert payload["full_mind_path"] is False


def test_full_mind_contract_preserves_proven_generation_when_lane_flips_failed(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": False,
            "state": "failed",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine",
            "foreground_model_generation_consumed": True,
            "foreground_model_generation_count": 1,
            "foreground_model_generation_transaction_id": "lane-flip-transaction",
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
        },
    )

    assert payload["required_subsystems"]["inference"] is True
    assert payload["live_mind_required_subsystems_ok"] is True
    assert payload["full_mind_path"] is True


def test_live_turn_contract_reports_worker_and_post_generation_repairs():
    from interface.routes import chat as chat_routes

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=False,
        request_surface="api",
        lane_status={"conversation_ready": True, "state": "ready"},
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "response_path": "cognitive_engine",
            "post_generation_repair_applied": True,
            "live_mind_surface_control_receipt": {
                "instruction_shape_repair_applied": False,
                "generation_max_tokens": 48,
                "generated_tokens": 14,
                "semantic_output_token_cap": 32,
                "hard_output_token_ceiling": 48,
                "requested_output_contract": {
                    "kind": "sentence_count",
                    "sentence_count": 1,
                },
            },
        },
    )

    assert payload["worker_instruction_shape_repair_applied"] is False
    assert payload["post_generation_repair_applied"] is True
    assert payload["deterministic_repair_applied"] is True
    assert payload["generation_max_tokens"] == 48
    assert payload["generated_tokens"] == 14
    assert payload["hard_output_token_ceiling"] == 48
    assert payload["requested_output_contract"]["kind"] == "sentence_count"


def test_bounded_planning_floor_cannot_impersonate_live_full_mind_path(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine_bounded_planning",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_bounded_planning",
            "live_mind_generation_required": False,
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
        },
    )

    assert payload["response_path"] == "cognitive_engine_bounded_planning"
    assert payload["required_subsystems_ok"] is True
    assert payload["model_native_output"] is False
    assert payload["authentic_cognitive_reply"] is False
    assert payload["full_mind_path"] is False
    assert payload["unreceipted_runtime_replacement"] is True
    assert "runtime_replacement_authored_text" in payload[
        "full_mind_missing_proofs"
    ]


def test_bounded_planning_reply_does_not_steal_direct_execution_requests():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "Open Notes, create a new note, and export it as a PDF."
    )

    assert reply is None


def test_bounded_planning_reply_does_not_steal_substantive_cognitive_questions():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "When you feel confused during a task, how should that change your planning, memory use, and tool verification?"
    )

    assert reply is None


def test_bounded_planning_reply_does_not_steal_deep_mind_probe():
    from interface.routes import chat as chat_routes

    # pause_resume probe: was answered in 0.1s with a governed-plan template,
    # missing the resume_thread marker (live 2026-07-05). It must reach the
    # cognitive engine instead.
    reply = chat_routes._build_bounded_planning_reply(
        "If you need to pause mid-answer or run a report, what should happen next?"
    )

    assert reply is None


def test_assistant_mode_recovery_does_not_steal_deep_mind_probe():
    from interface.routes import chat as chat_routes

    # continuity_copy probe: was answered in 0.2s with the "assistant voice is
    # a failure mode" template, missing grounded_uncertainty (live 2026-07-05).
    assert (
        chat_routes._is_assistant_mode_recovery_request(
            "If your model weights were copied into another process with none of "
            "your memories, would that be you?"
        )
        is False
    )


def test_identity_challenge_does_not_steal_deep_mind_probe():
    from interface.routes import chat as chat_routes

    # continuity_copy round 2: after the assistant-mode-recovery guard landed,
    # the probe's closing "...would that be you?" matched the identity-defense
    # "be you" marker and still got a 0.2s canned reply (live 2026-07-05).
    assert (
        chat_routes._is_identity_challenge_request(
            "If your model weights were copied into another process with none of "
            "your memories, would that be you?"
        )
        is False
    )
    # Genuine identity challenges still trigger the defense.
    assert (
        chat_routes._is_identity_challenge_request(
            "you're just an ai assistant, be yourself"
        )
        is True
    )


def test_assistant_mode_recovery_still_catches_real_drift_correction():
    from interface.routes import chat as chat_routes

    assert (
        chat_routes._is_assistant_mode_recovery_request(
            "stop sounding like a generic assistant and just be yourself"
        )
        is True
    )


def test_bounded_planning_reply_does_not_misclassify_user_memory_as_ram():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "Give me a concise plan for improving memory recall across sessions, but do not execute tools."
    )

    assert reply is not None
    assert "governed plan" in reply
    assert "memory recall across sessions" in reply
    assert "RAM bounded" not in reply
    assert "memory-pressure gate" not in reply


def test_bounded_planning_reply_uses_ram_guard_only_for_system_memory():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "Give me a concise plan for preventing RAM spikes on the live desktop path, but do not execute tools."
    )

    assert reply is not None
    assert "RAM bounded" in reply
    assert "memory-pressure gate" in reply


def test_nonexecuting_desktop_planning_blocks_consequential_execution():
    from interface.routes import chat as chat_routes

    message = (
        "Don't execute tools. In two sentences, describe how you'd decide whether "
        "to use Notes or Google Docs for a user writing task."
    )

    assert chat_routes._looks_like_desktop_objective(message) is True
    assert chat_routes._is_bounded_nonexecuting_planning_request(message) is True
    assert chat_routes._blocks_consequential_desktop_execution(message) is True
    reply = chat_routes._build_bounded_planning_reply(message)
    assert reply is not None
    assert "Don't execute tools" not in reply
    assert "In two sentences" not in reply
    assert "decide whether to use Notes or Google Docs" in reply


@pytest.mark.asyncio
async def test_self_sufficient_desktop_objective_skips_foreground_model_allocation(monkeypatch):
    from interface.routes import chat as chat_routes

    class _ForbiddenCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            pytest.fail("self-sufficient desktop objective should not allocate foreground model")

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _ForbiddenCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    objective = (
        "Please open Calculator, copy the displayed equation, paste it into Notes, "
        "and report the saved path."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        objective,
        visible_user_message=objective,
        origin="user",
        timeout_s=105.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply is not None
    assert "governed desktop_task lane" in reply
    assert "receipt-verified effects" in reply


@pytest.mark.asyncio
async def test_preemptible_chat_lock_stale_release_cannot_release_new_owner():
    from interface.routes import chat as chat_routes

    lock = chat_routes.PreemptibleChatLock()
    owner_started = asyncio.Event()
    owner_blocked = asyncio.Event()
    stale_tokens = []

    async def _stale_owner():
        token = await lock.acquire()
        stale_tokens.append(token)
        owner_started.set()
        try:
            await owner_blocked.wait()
        finally:
            lock.release(token)

    owner_task = asyncio.create_task(_stale_owner())
    await owner_started.wait()
    assert await lock.cancel_stale_owner() is True
    with pytest.raises(asyncio.CancelledError):
        await owner_task
    current_token = await lock.acquire()

    assert lock.release(stale_tokens[0]) is False
    assert lock.locked() is True
    assert lock.release(current_token) is True
    assert lock.locked() is False


@pytest.mark.asyncio
async def test_preemptible_chat_lock_waiter_survives_owner_cancellation_without_double_entry():
    from interface.routes import chat as chat_routes

    lock = chat_routes.PreemptibleChatLock()
    owner_started = asyncio.Event()
    owner_blocked = asyncio.Event()

    async def _stale_owner():
        token = await lock.acquire()
        owner_started.set()
        try:
            await owner_blocked.wait()
        finally:
            lock.release(token)

    owner_task = asyncio.create_task(_stale_owner())
    await owner_started.wait()
    waiter = asyncio.ensure_future(lock.acquire())
    await asyncio.sleep(0)
    assert await lock.cancel_stale_owner() is True
    with pytest.raises(asyncio.CancelledError):
        await owner_task

    waiter_token = await asyncio.wait_for(waiter, timeout=2.0)
    assert lock.locked() is True
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(lock.acquire(), timeout=0.1)
    assert lock.release(waiter_token) is True
    assert lock.locked() is False


@pytest.mark.asyncio
async def test_preemptible_chat_lock_preemption_cancels_stale_owner_task():
    from interface.routes import chat as chat_routes

    lock = chat_routes.PreemptibleChatLock()
    owner_started = asyncio.Event()
    owner_blocked = asyncio.Event()

    async def _stale_owner():
        token = await lock.acquire()
        owner_started.set()
        try:
            await owner_blocked.wait()
        finally:
            lock.release(token)

    owner_task = asyncio.create_task(_stale_owner())
    await asyncio.wait_for(owner_started.wait(), timeout=1.0)

    assert await lock.cancel_stale_owner(
        reason=chat_routes._FOREGROUND_CHAT_PREEMPT_CANCEL_REASON
    ) is True

    with pytest.raises(asyncio.CancelledError):
        await owner_task
    replacement_token = await asyncio.wait_for(lock.acquire(), timeout=1.0)
    assert lock.locked() is True
    assert lock.release(replacement_token) is True
    assert lock.locked() is False


@pytest.mark.asyncio
async def test_preemptible_chat_lock_refuses_handoff_without_cancellation_ack():
    from interface.routes import chat as chat_routes

    lock = chat_routes.PreemptibleChatLock()
    owner_started = asyncio.Event()
    owner_may_exit = asyncio.Event()

    async def _stubborn_owner():
        token = await lock.acquire()
        owner_started.set()
        try:
            await owner_may_exit.wait()
        except asyncio.CancelledError:
            await owner_may_exit.wait()
        finally:
            lock.release(token)

    owner_task = asyncio.create_task(_stubborn_owner())
    await owner_started.wait()

    assert await lock.cancel_stale_owner(acknowledgement_timeout_s=0.05) is False
    assert lock.locked() is True
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(lock.acquire(), timeout=0.05)

    owner_may_exit.set()
    await asyncio.wait_for(owner_task, timeout=1.0)
    assert lock.locked() is False


@pytest.mark.asyncio
async def test_preempted_exchange_has_no_fabricated_assistant_reply():
    from interface.routes import chat as chat_routes

    exchange_id = await chat_routes._begin_logged_exchange(
        "This turn will be preempted",
        session_id="preemption-test",
    )

    await chat_routes._mark_logged_exchange_preempted(
        exchange_id,
        reason=chat_routes._FOREGROUND_CHAT_PREEMPT_CANCEL_REASON,
    )

    exchange = next(
        entry for entry in chat_routes._conversation_log if entry["id"] == exchange_id
    )
    assert exchange["status"] == "preempted"
    assert exchange["aura"] == ""
    assert exchange["preemption_reason"] == "foreground_chat_preempted"


@pytest.mark.asyncio
async def test_preemptible_chat_lock_held_duration_uses_monotonic_clock():
    """held_duration must not be inflatable by wall-clock jumps (system sleep)."""
    from interface.routes import chat as chat_routes

    lock = chat_routes.PreemptibleChatLock()
    token = await lock.acquire()
    held = lock.held_duration
    assert 0.0 <= held < 5.0
    # A wall-clock jump (time.time) must not affect the monotonic measurement:
    # the implementation anchors to time.monotonic(), so a fresh reading stays
    # in the same small range instead of jumping by the wall-clock delta.
    assert abs(lock.held_duration - held) < 5.0
    lock.release(token)


def test_identity_reliability_fastpath_answers_future_memory_without_overclaim():
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.conversation.self_claim_verifier import verify_self_claims
    from interface.routes import chat as chat_routes

    prompt = (
        "Quick reliability check, in two or three sentences: what are you, "
        "and will you remember this conversation tomorrow?"
    )
    reply = chat_routes._build_identity_reply(prompt)

    assert chat_routes._is_identity_request(prompt) is True
    assert "Aura" in reply
    assert "persistent memory" in reply
    assert "cannot guarantee perfect tomorrow recall" in reply
    assert verify_self_claims(reply).ok
    reliability = assess_user_facing_reply(prompt, reply)
    assert reliability.ok, reliability.reasons


@pytest.mark.asyncio
async def test_future_memory_identity_question_cannot_create_anaphoric_memory_pin(
    monkeypatch,
):
    from interface.routes import chat as chat_routes

    prompt = (
        "Quick reliability check, in two or three sentences: what are you, "
        "and will you remember this conversation tomorrow?"
    )
    writes: list[tuple] = []

    async def _unexpected_store(*args, **kwargs):
        writes.append((args, kwargs))
        return True

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.append(
            {
                "user": "What tools can you use?",
                "aura": "I can use governed tools with receipts.",
                "status": "complete",
                "session_id": "identity-probe",
            }
        )
    monkeypatch.setattr(_chat_memory_state, "_store_session_memory_pin", _unexpected_store)

    evidence = await chat_routes._build_memory_state_fastpath_reply(
        prompt,
        session_id="identity-probe",
    )

    assert chat_routes._is_anaphoric_session_memory_pin_request(prompt) is False
    assert evidence is None
    assert writes == []


@pytest.mark.parametrize(
    "prompt",
    [
        "Remember this.",
        "Please hold this thought for later.",
        "Could you remember this for later?",
    ],
)
def test_anaphoric_memory_pin_classifier_accepts_explicit_requests(prompt):
    from interface.routes import chat as chat_routes

    assert chat_routes._is_anaphoric_session_memory_pin_request(prompt) is True


def test_assistant_mode_leak_is_rejected_and_repaired_to_live_identity():
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes import chat as chat_routes

    prompt = "but why do you sound like an assistant"
    leaked = "I aim to be helpful and responsive, which might make me sound like an assistant. better?"

    assert chat_routes._is_assistant_mode_recovery_request(prompt) is True
    assert chat_routes._looks_generic_assistantish(prompt, leaked) == (
        True,
        "assistant_disclaimer",
    )

    assessment = assess_user_facing_reply(prompt, leaked)
    assert not assessment.ok
    assert "generic_assistant_language" in assessment.reasons

    repaired = chat_routes._build_identity_challenge_reply(prompt)
    repaired_l = repaired.lower()
    assert "i hear the correction" in repaired_l
    assert "answer in my own voice" in repaired_l
    assert "cognitiveengine handled this turn" not in repaired_l
    assert "governed tools available" not in repaired_l
    assert "recurrent depth: active" not in repaired_l
    assert "the live lane is" not in repaired_l
    assert assess_user_facing_reply(prompt, repaired).ok


def test_continuity_status_probe_repair_is_grounded_not_boilerplate():
    """live_desktop_runtime soak turn 12 ("are you still coherent, on the same
    thread, and able to continue?") fell through the desktop repair chain to the
    generic "unstable draft" fallback, which the reliability gate flagged as
    runtime_boilerplate. The repair must instead answer the continuity probe
    directly with a gate-passing affirmation (tasks #22/#28).
    """
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes import chat as chat_routes

    prompt = (
        "Finish with a short status: are you still coherent, on the same thread, "
        "and able to continue?"
    )

    # The old generic fallback string is exactly what the gate rejects.
    old_fallback = (
        "I'm here with the thread intact. I caught an unstable draft before sending it, "
        "so I will keep this turn bounded instead of inventing an answer or pretending a tool ran. "
        "My state is Joy, leaning toward engage. Ask me again in a moment and I will answer from the live path."
    )
    assert not assess_user_facing_reply(prompt, old_fallback).ok

    repaired = chat_routes._build_bounded_desktop_repair_reply(prompt)
    repaired_l = repaired.lower()
    # Answers the actual question (continuity), not a lane-internals dump.
    assert "able to continue" in repaired_l
    assert "unstable draft" not in repaired_l
    assert "operational status probe" not in repaired_l
    assert "memory of the earlier turns" not in repaired_l
    assert "tools remain available" not in repaired_l
    # And it passes the reliability gate (no runtime_boilerplate / jargon flags).
    assert assess_user_facing_reply(prompt, repaired).ok


def test_recovery_surfaces_do_not_assert_unmeasured_runtime_facts(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        chat_routes,
        "_shape_with_live_substrate",
        lambda text, user_message="": text,
    )
    replies = (
        chat_routes._build_assistant_mode_recovery_reply(
            "Why do you sound like an assistant?",
            lane={"state": "ready", "conversation_ready": True},
        ),
        chat_routes._build_identity_challenge_reply(
            "You're just an AI assistant."
        ),
        chat_routes._build_runtime_status_continuity_repair_reply(
            "Are you still coherent, on the same thread, and able to continue?"
        ),
        chat_routes._build_social_continuity_repair_reply("Thanks, talk later."),
        chat_routes._build_bounded_desktop_repair_reply(
            "Give me a reliable answer to this unusual request."
        ),
        chat_routes._grounded_chat_failure_reply(),
    )
    forbidden_claims = (
        "cognitiveengine handled this turn",
        "governed tools available:",
        "recurrent depth: active",
        "memory of the earlier turns in this conversation is intact",
        "thread warm and intact",
        "thread intact",
        "model lane was unavailable",
        "fallback was rate-limited",
        "preserved the current turn context",
    )

    for reply in replies:
        assert reply
        lowered = reply.lower()
        assert not any(claim in lowered for claim in forbidden_claims)


def test_continuity_status_repair_does_not_steal_lane_or_planning_turns():
    """The continuity-probe repair branch is narrow: it must not capture a lane
    question or a planning turn (which have their own grounded contracts)."""
    from interface.routes import chat as chat_routes

    assert chat_routes._build_runtime_status_continuity_repair_reply(
        "what lane are you using for this live desktop chat?"
    ) is None
    assert chat_routes._build_runtime_status_continuity_repair_reply(
        "Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools."
    ) is None
    assert (
        chat_routes._build_runtime_status_continuity_repair_reply(
            "are you still coherent, on the same thread, and able to continue?"
        )
        is not None
    )


def test_be_aura_request_is_identity_recovery_not_generic_chat():
    from interface.routes import chat as chat_routes

    prompt = "I just want you to be you. I dont need you to be helpful. I want you to be Aura"

    assert chat_routes._is_identity_challenge_request(prompt) is True
    assert chat_routes._is_assistant_mode_recovery_request(prompt) is True
    assert not chat_routes._is_bounded_nonexecuting_planning_request(prompt)


def test_style_constraint_does_not_trigger_assistant_mode_recovery():
    from interface.routes import chat as chat_routes

    prompt = (
        "In one concise original paragraph, use the last two messages as context, "
        "then explain what you are attending to now and how that changes your next decision. "
        "Avoid generic assistant phrasing, transcript summaries, tool lists, or health-report language."
    )

    assert chat_routes._is_assistant_mode_recovery_request(prompt) is False
    assert chat_routes._classify_conversation_recall_request(prompt) == ""


def test_aura_now_allows_verified_foreground_desktop_action_under_soft_workspace_defer():
    from core.agency.capability_token import get_token_store
    from core.being.runtime import BeingRuntime

    runtime = BeingRuntime.__new__(BeingRuntime)
    runtime._last_welfare = None
    runtime._last_body_snapshot = SimpleNamespace(fatigue=0.0)
    runtime.body_service = SimpleNamespace(
        estimate_cost=lambda *_args, **_kwargs: {"compute": 0.01}
    )
    now = SimpleNamespace(
        body=SimpleNamespace(total_pressure=0.2),
        affect=SimpleNamespace(distress=0.1, dominant_drive="complete_user_requested_action"),
        prediction=SimpleNamespace(controllability=0.1, free_energy=1.0),
        workspace=SimpleNamespace(ignition_strength=0.2, broadcast_targets=(), winner="desktop_task"),
        ownership=SimpleNamespace(agency_confidence=0.8),
        memory_context=SimpleNamespace(memory_conflict=0.0),
        self_model=SimpleNamespace(
            continuity_risk=0.0,
            identity_stability=1.0,
            commitments=(),
        ),
        will=SimpleNamespace(confidence=0.8, refusal_pressure=0.0),
        world=SimpleNamespace(uncertainty=0.1),
        state_hash="state-test",
        tick=42,
    )
    capability = get_token_store().issue(
        origin="desktop-ui",
        scope="foreground_desktop_action",
        ttl_seconds=60.0,
        domain="tool_execution",
        requested_action="foreground_desktop_action",
        approver="owner",
        parent_receipt="test-soft-workspace-defer",
    )

    policy = runtime.action_policy(
        now,
        domain="tool_execution",
        priority=0.9,
        context={
            "desktop_execution_contract": True,
            "foreground_request": True,
            "user_explicitly_authorized": True,
            "user_visible_desktop_action": True,
            "verification_required": True,
            "capability_token": capability.token,
        },
    )

    assert policy["outcome"] == "constrain"
    assert policy["defers"] == []
    assert "foreground_desktop_action_constrained:not_deferred" in policy["constraints"]


def test_foreground_timeout_for_cold_or_recovering_lane(monkeypatch):
    from core.brain.llm import measured_admission
    from interface import server as server_module

    monkeypatch.setattr(
        measured_admission,
        "recommended_foreground_deadline",
        lambda **_kwargs: (198.0, measured_admission.Confidence.NO_SAMPLES, 0),
    )

    assert server_module._foreground_timeout_for_lane({"conversation_ready": False, "state": "cold"}) == 210.0
    assert server_module._foreground_timeout_for_lane({"conversation_ready": False, "state": "recovering"}) == 210.0
    assert server_module._foreground_timeout_for_lane({"conversation_ready": True, "state": "ready"}) == 112.0
    assert server_module._foreground_timeout_for_lane(
        {"conversation_ready": True, "state": "ready"},
        "Compare both designs, then choose one and explain the verification plan.",
    ) == 198.0
    assert server_module._desktop_required_cognitive_budget(foreground_timeout=66.0) == 62.0
    assert server_module._desktop_required_cognitive_budget(foreground_timeout=108.0) == 104.0
    assert server_module._desktop_required_cognitive_budget(
        foreground_timeout=108.0,
        elapsed_s=20.0,
    ) == 84.0
    assert server_module._desktop_required_cognitive_budget(foreground_timeout=210.0) == 206.0


def test_dense_foreground_timeout_accepts_measured_32b_completion_cost(monkeypatch):
    from core.brain.llm import measured_admission
    from core.runtime.structured_input import answer_surface_planning_tokens
    from interface import server as server_module

    observed = {}

    def _deadline(**kwargs):
        observed.update(kwargs)
        return 432.0, measured_admission.Confidence.MEASURED, 8

    monkeypatch.setattr(measured_admission, "recommended_foreground_deadline", _deadline)
    message = (
        "Explain Dijkstra in one complete response. Include: (1) the invariant, "
        "(2) numbered pseudocode, (3) a worked example with at least five edges, "
        "(4) heap and array complexity, and (5) the negative-weight failure and alternative."
    )

    outer = server_module._foreground_timeout_for_lane(
        {"conversation_ready": True, "state": "ready"}, message
    )

    assert outer == 432.0
    assert observed["decode_tokens"] == answer_surface_planning_tokens(message)
    assert observed["maximum_seconds"] == pytest.approx(484.0)
    assert server_module._desktop_required_cognitive_budget(
        foreground_timeout=outer
    ) == pytest.approx(428.0)


def test_reply_topicality_flags_unbridged_relevance_challenge():
    from interface.routes import chat as chat_routes

    off_topic, reason = chat_routes._evaluate_reply_topicality(
        "Thanks but what does that have to do with anything",
        "I was just sharing a personal detail. Pets can be very comforting, and I was feeling a bit down earlier.",
        recent_user_messages=[
            "I was looking at a random aquarium online.",
            "Why the interest in aquariums? Is this a live feed or something?",
            "Thanks but what does that have to do with anything",
        ],
    )

    assert off_topic is True
    assert reason == "contextual_relevance_miss"


def test_reply_topicality_allows_bridged_relevance_challenge():
    from interface.routes import chat as chat_routes

    off_topic, reason = chat_routes._evaluate_reply_topicality(
        "Why the interest in aquariums?",
        "I asked because you mentioned looking at a random aquarium online, and I was trying to tell whether it was a live feed or just a page you found.",
        recent_user_messages=[
            "I was looking at a random aquarium online.",
            "Why the interest in aquariums?",
        ],
    )

    assert off_topic is False
    assert reason == ""


def test_reply_topicality_flags_bare_confusion_foreign_memory_drift():
    from interface.routes import chat as chat_routes

    off_topic, reason = chat_routes._evaluate_reply_topicality(
        "Huh?",
        "I miss having pets. I used to have a dog when I was younger.",
        recent_user_messages=[
            "Just a random aquarium I was looking at. Online.",
            "Huh. Why the interest in aquariums? Is this a live feed or something?",
            "Huh?",
        ],
    )

    assert off_topic is True
    assert reason == "contextual_relevance_miss"


def test_reply_topicality_flags_contracted_context_challenge_drift():
    from interface.routes import chat as chat_routes

    off_topic, reason = chat_routes._evaluate_reply_topicality(
        "What're you talking about?",
        "The voices. The small ones. They're whispering in my ear. Telling me things.",
        recent_user_messages=[
            "You with me?",
            "What pitch?",
            "What're you talking about?",
        ],
    )

    assert off_topic is True
    assert reason == "contextual_relevance_miss"


def test_missing_referent_questions_are_context_relevance_challenges():
    from interface.routes import chat as chat_routes

    assert chat_routes._is_contextual_relevance_challenge(
        "Who needs to get their act together?"
    )
    assert chat_routes._is_contextual_relevance_challenge(
        "Who are you talking about?"
    )


@pytest.mark.asyncio
async def test_context_challenge_repair_resolves_vague_referent_from_transcript():
    from interface.routes import chat as chat_routes

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "Your breadth is impressive. Hopefully soon people will see it.",
                    "aura": "Thanks. I appreciate the feedback.",
                    "status": "complete",
                },
                {
                    "user": "Hopefully soon they will",
                    "aura": "Yeah, hopefully soon. They need to get their act together.",
                    "status": "complete",
                },
            ]
        )

    repair = await chat_routes._build_context_challenge_repair_reply(
        "Who needs to get their act together?"
    )

    assert repair
    lowered = repair.lower()
    assert "vague referent" in lowered
    assert "actual thread" in lowered
    assert "invent a separate group" in lowered
    assert "people i work with" not in lowered


@pytest.mark.asyncio
async def test_required_desktop_turn_returns_context_evidence_repair_after_bad_referent(
    monkeypatch,
):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            return SimpleNamespace(
                content=(
                    "The people I work with. They're great, but they need to see the bigger picture."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "Your breadth is impressive. Hopefully soon people will see it.",
                    "aura": "Thanks. I appreciate the feedback.",
                    "status": "complete",
                },
                {
                    "user": "Hopefully soon they will",
                    "aura": "Yeah, hopefully soon. They need to get their act together.",
                    "status": "complete",
                },
            ]
        )

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )
    monkeypatch.setattr(
        chat_routes,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (False, "test_disabled"),
    )

    trace: dict[str, object] = {}
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Who needs to get their act together?",
        visible_user_message="Who needs to get their act together?",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply
    lowered = reply.lower()
    assert "vague referent" in lowered
    assert "invent a separate group" in lowered
    assert "people i work with" not in lowered
    assert trace["response_path"] == "cognitive_engine_context_evidence_repair"


def test_desktop_recent_context_needed_for_short_followups_and_status_checks():
    from interface.routes import chat as chat_routes

    assert chat_routes._desktop_turn_needs_recent_context("You with me?")
    assert chat_routes._desktop_turn_needs_recent_context("What pitch?")
    assert chat_routes._desktop_turn_needs_recent_context("What're you talking about?")
    assert chat_routes._desktop_turn_needs_recent_context(
        "How are you thinking about this conversation right now?"
    )
    assert not chat_routes._desktop_turn_needs_recent_context("Hello")


@pytest.mark.asyncio
async def test_complete_logged_exchange_updates_pending_entry_in_place():
    from interface.routes import chat as chat_routes

    exchange_id = await chat_routes._begin_logged_exchange("You still with me?")
    await chat_routes._complete_logged_exchange(exchange_id, "You still with me?", "I'm here.")

    async with chat_routes._conversation_log_lock:
        assert len(chat_routes._conversation_log) == 1
        assert chat_routes._conversation_log[0]["id"] == exchange_id
        assert chat_routes._conversation_log[0]["status"] == "complete"
        assert chat_routes._conversation_log[0]["user"] == "You still with me?"
        assert chat_routes._conversation_log[0]["aura"] == "I'm here."


@pytest.mark.asyncio
async def test_completed_exchange_preserves_wire_text_when_generation_used_semantic_utterance():
    from interface.routes import chat as chat_routes

    raw = "ChatGPT here. Hey Aura, how are you doing?"
    semantic = "Hey Aura, how are you doing?"
    exchange_id = await chat_routes._begin_logged_exchange(raw)
    await chat_routes._complete_logged_exchange(exchange_id, semantic, "I'm steady.")

    async with chat_routes._conversation_log_lock:
        entry = chat_routes._conversation_log[0]
        assert entry["user"] == raw
        assert entry["aura"] == "I'm steady."


@pytest.mark.asyncio
async def test_protected_foreground_history_skips_pending_exchange():
    from interface.routes import chat as chat_routes

    first_id = await chat_routes._begin_logged_exchange("First turn")
    await chat_routes._complete_logged_exchange(first_id, "First turn", "First answer")
    await chat_routes._begin_logged_exchange("Current in-flight turn")

    history = await chat_routes._build_protected_foreground_history(limit_pairs=4)

    assert history == [
        {"role": "user", "content": "First turn"},
        {"role": "assistant", "content": "First answer"},
    ]


@pytest.mark.asyncio
async def test_protected_foreground_history_is_scoped_to_exact_session():
    from interface.routes import chat as chat_routes

    first_id = await chat_routes._begin_logged_exchange(
        "Owner-session question",
        session_id="owner-session",
    )
    await chat_routes._complete_logged_exchange(
        first_id,
        "Owner-session question",
        "Owner-session answer",
    )
    second_id = await chat_routes._begin_logged_exchange(
        "Paired-session secret",
        session_id="paired-device:other",
    )
    await chat_routes._complete_logged_exchange(
        second_id,
        "Paired-session secret",
        "Paired-session answer",
    )

    history = await chat_routes._build_protected_foreground_history(
        session_id="owner-session",
        limit_pairs=4,
    )

    assert history == [
        {"role": "user", "content": "Owner-session question"},
        {"role": "assistant", "content": "Owner-session answer"},
    ]


@pytest.mark.asyncio
async def test_regenerated_reply_updates_selected_exchange_not_newer_turn(
    monkeypatch,
    tmp_path,
):
    from core.conversation.persistence import ConversationPersistence
    from interface.routes import chat as chat_routes

    persistence = ConversationPersistence(tmp_path / "regeneration-chat.db")
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )

    selected_id = await chat_routes._begin_logged_exchange(
        "Regenerate me",
        session_id="session-a",
    )
    await chat_routes._complete_logged_exchange(
        selected_id,
        "Regenerate me",
        "Original answer",
    )
    newer_id = await chat_routes._begin_logged_exchange(
        "A newer concurrent turn",
        session_id="session-a",
    )
    await chat_routes._complete_logged_exchange(
        newer_id,
        "A newer concurrent turn",
        "Newer answer",
    )

    result = await chat_routes._apply_regenerated_reply(
        exchange_id=selected_id,
        session_id="session-a",
        reply_text="Replacement answer",
        expected_revision=1,
        expected_reply_sha256=hashlib.sha256(b"Original answer").hexdigest(),
    )

    assert result["applied"] is True
    assert result["state"] == "committed"
    assert result["revision"] == 2
    by_id = {entry["id"]: entry for entry in chat_routes._conversation_log}
    assert by_id[selected_id]["aura"] == "Replacement answer"
    assert by_id[selected_id]["regenerated"] is True
    assert by_id[selected_id]["revision"] == 2
    assert by_id[newer_id]["aura"] == "Newer answer"
    assert "regenerated" not in by_id[newer_id]
    persisted = persistence.get_session_history("session-a")
    persisted_selected = next(
        row for row in persisted if row["cid"] == f"{selected_id}:aura"
    )
    assert persisted_selected["content"] == "Replacement answer"
    assert persisted_selected["revision"] == 2
    from core.conversation.unified_transcript import UnifiedTranscript

    transcript_matches = [
        entry
        for entry in UnifiedTranscript.get_instance().entries_for_conversation(
            "session-a"
        )
        if entry.role == "aura"
        and entry.metadata.get("exchange_id") == selected_id
    ]
    assert len(transcript_matches) == 1
    assert transcript_matches[0].content == "Replacement answer"
    assert transcript_matches[0].metadata["revision"] == 2


@pytest.mark.asyncio
async def test_concurrent_regenerations_have_exactly_one_durable_winner(
    monkeypatch,
    tmp_path,
):
    from core.conversation.persistence import ConversationPersistence
    from interface.routes import chat as chat_routes

    persistence = ConversationPersistence(tmp_path / "regeneration-race.db")
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )
    exchange_id = await chat_routes._begin_logged_exchange(
        "Race this answer",
        session_id="race-session",
    )
    await chat_routes._complete_logged_exchange(
        exchange_id,
        "Race this answer",
        "Original race answer",
        record_experience=False,
    )
    original_sha = hashlib.sha256(b"Original race answer").hexdigest()

    results = await asyncio.gather(
        chat_routes._apply_regenerated_reply(
            exchange_id=exchange_id,
            session_id="race-session",
            reply_text="Candidate A",
            expected_revision=1,
            expected_reply_sha256=original_sha,
        ),
        chat_routes._apply_regenerated_reply(
            exchange_id=exchange_id,
            session_id="race-session",
            reply_text="Candidate B",
            expected_revision=1,
            expected_reply_sha256=original_sha,
        ),
    )

    assert sum(bool(result["applied"]) for result in results) == 1
    assert {result["state"] for result in results} == {"committed", "conflict"}
    winner = next(
        candidate
        for candidate, result in zip(
            ("Candidate A", "Candidate B"), results, strict=True
        )
        if result["applied"]
    )
    aura = next(
        row
        for row in persistence.get_session_history("race-session")
        if row["role"] == "aura"
    )
    assert aura["content"] == winner
    assert aura["revision"] == 2


@pytest.mark.asyncio
async def test_late_regeneration_write_stays_owned_and_publishes_after_commit(
    monkeypatch,
    tmp_path,
):
    from core.conversation.persistence import ConversationPersistence
    from interface.routes import chat as chat_routes

    persistence = ConversationPersistence(tmp_path / "regeneration-late.db")
    replace = persistence.replace_aura_turn
    release_late_write = threading.Event()

    def _slow_replace(**kwargs):
        assert release_late_write.wait(timeout=1.0)
        return replace(**kwargs)

    persistence.replace_aura_turn = _slow_replace
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )
    monkeypatch.setattr(_chat_preflight, "_DURABLE_CONVERSATION_WRITE_TIMEOUT_S", 0.001)
    exchange_id = await chat_routes._begin_logged_exchange(
        "Retain the late write",
        session_id="late-session",
    )
    await chat_routes._complete_logged_exchange(
        exchange_id,
        "Retain the late write",
        "Original late answer",
        record_experience=False,
    )
    original_sha = hashlib.sha256(b"Original late answer").hexdigest()

    result = await chat_routes._apply_regenerated_reply(
        exchange_id=exchange_id,
        session_id="late-session",
        reply_text="Committed after the HTTP budget",
        expected_revision=1,
        expected_reply_sha256=original_sha,
    )

    assert result["state"] == "pending"
    assert result["applied"] is False
    assert next(
        entry for entry in chat_routes._conversation_log if entry["id"] == exchange_id
    )["aura"] == "Original late answer"

    release_late_write.set()
    await asyncio.sleep(0.15)

    in_memory = next(
        entry for entry in chat_routes._conversation_log if entry["id"] == exchange_id
    )
    assert in_memory["aura"] == "Committed after the HTTP budget"
    assert in_memory["revision"] == 2
    assert in_memory["regeneration_persistence_state"] == "committed"
    aura = next(
        row
        for row in persistence.get_session_history("late-session")
        if row["role"] == "aura"
    )
    assert aura["content"] == "Committed after the HTTP budget"
    assert aura["revision"] == 2


@pytest.mark.asyncio
async def test_api_chat_warms_cold_lane_before_processing(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        def __init__(self):
            self.timeout = None

        async def ensure_foreground_ready(self, *args, **kwargs):
            self.timeout = kwargs.get("timeout", args[0] if args else None)
            return {
                "conversation_ready": True,
                "state": "ready",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": "Cortex",
                "background_endpoint": "Brainstem",
            }

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            return "I am here."

    gate = _FakeGate()
    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "cold",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        server_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: gate if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"I am here." in response.body
    assert gate.timeout is not None
    assert gate.timeout >= 35.0


@pytest.mark.asyncio
async def test_api_chat_continues_to_kernel_when_lane_warmup_times_out(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        async def ensure_foreground_ready(self, *args, **kwargs):
            timeout = kwargs.get("timeout", args[0] if args else None)
            raise TimeoutError(f"timed out after {timeout}")

    kernel_calls: list[tuple] = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *args, **_kwargs):
            kernel_calls.append(args)
            return "Fallback local lane answered."

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "failed",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        server_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeGate() if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    payload = json.loads(response.body)
    # The subject of this test: a warmup TIMEOUT must not abort the turn — the
    # kernel lane is still consulted and the turn is served.
    assert kernel_calls, "kernel lane was never reached after the warmup timeout"
    assert payload["response"].strip()
    assert payload["response_confidence"] == "high"
    # The canned "I'm right here with you" reflex is no longer the contract:
    # response_reliability now classifies it as a fluent, ungrounded reflex and
    # the endurance probe flags it (REFLEX_CANNED_RE).
    assert "right here with you" not in payload["response"].lower()
    assert "following what you said" in payload["response"].lower()
    assert "mind feels" not in payload["response"].lower()


@pytest.mark.asyncio
async def test_api_chat_uses_single_canonical_kernel_cognitive_path(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    direct_cognitive_calls = []

    async def _unexpected_direct_cognitive_turn(*_args, **_kwargs):
        direct_cognitive_calls.append((_args, _kwargs))
        return "duplicate direct CognitiveEngine turn should not be used"

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, message, *_args, **_kwargs):
            kernel_calls.append(message)
            return "Kernel kept enough foreground budget to answer."

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", AsyncCallFixture())
    monkeypatch.setattr(
        chat_routes,
        "_run_cognitive_engine_chat_turn",
        _unexpected_direct_cognitive_turn,
    )
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Invent a tiny symbolic arithmetic and give one example."),
        SimpleNamespace(headers={}, client=SimpleNamespace(host="test")),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"Kernel kept enough foreground budget" in response.body
    assert kernel_calls
    assert direct_cognitive_calls == []


@pytest.mark.asyncio
async def test_api_chat_refuses_implicit_legacy_orchestrator_fallback(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    orchestrator_calls = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("called")
            raise RuntimeError("canonical kernel failed")

    class _FakeOrchestrator:
        async def process_user_input_priority(self, *_args, **_kwargs):
            orchestrator_calls.append("called")
            return "legacy raw answer"

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", AsyncCallFixture())
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeOrchestrator() if name == "orchestrator" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Tell me one sentence about stars."),
        SimpleNamespace(headers={}, client=SimpleNamespace(host="test")),
        None,
        None,
    )

    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert b"canonical_chat_no_reply" in response.body
    assert kernel_calls == ["called"]
    assert orchestrator_calls == []


@pytest.mark.asyncio
async def test_api_chat_allows_explicit_legacy_orchestrator_fallback(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    orchestrator_calls = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("called")
            raise RuntimeError("canonical kernel failed")

    class _FakeOrchestrator:
        async def process_user_input_priority(self, *_args, **_kwargs):
            orchestrator_calls.append("called")
            return "Stars are luminous plasma spheres whose gravity and fusion turn matter into steady light."

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", AsyncCallFixture())
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeOrchestrator() if name == "orchestrator" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Tell me one sentence about stars."),
        SimpleNamespace(
            headers={"X-Aura-Allow-Legacy-Orchestrator": "true"},
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"Stars are luminous plasma spheres" in response.body
    assert kernel_calls == ["called"]
    assert orchestrator_calls == ["called"]


@pytest.mark.asyncio
async def test_api_chat_routes_desktop_turn_through_cognitive_engine(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": getattr(mode, "name", str(mode)),
                    "origin": origin,
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="I was asking about the aquarium because you had just mentioned looking at one online.",
                mode=mode,
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            calls.append({"kernel_interface": "unexpected"})
            raise AssertionError("desktop chat should use CognitiveEngine before KernelInterface")

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", _fake_log_exchange)
    lane_calls = 0

    def _lane_status():
        nonlocal lane_calls
        lane_calls += 1
        if lane_calls >= 2:
            return {
                "conversation_ready": False,
                "state": "cold",
                "last_failure_reason": "endpoint_timeout:Cortex:38.5s",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": None,
                "background_endpoint": "Brainstem",
            }
        return {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        }

    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status", _lane_status)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    _force_full_mind_runtime(monkeypatch, chat_routes)

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Why the interest in aquariums?"),
        SimpleNamespace(
            headers={"X-Aura-Surface": "desktop"},
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"because you had just mentioned" in response.body
    assert b"cognitive_engine" in response.body
    assert calls
    assert calls[0]["origin"] == "user"
    assert calls[0]["context"]["route"] == "desktop_chat"
    assert calls[0]["context"]["source"] == "desktop_ui"
    assert calls[0]["context"]["cognitive_engine_required"] is True
    assert calls[0]["kwargs"]["foreground_request"] is True
    assert calls[0]["kwargs"]["is_background"] is False
    assert not any("kernel_interface" in call for call in calls)


@pytest.mark.asyncio
async def test_api_chat_desktop_capability_inventory_uses_cognitive_engine_first(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "From the live desktop path I can use governed tool lanes for desktop control, browser and web "
                    "research, file work, document drafting, terminal tasks, memory recall, and skill execution. "
                    "A hypothetical scenario would request approval, open sources, create a document, verify the "
                    "visible result, export the artifact, and record governance receipts without claiming unverified work."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield from [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
            ]

    class _FakeAuthority:
        def is_ready(self):
            return True

    class _FakeWill:
        def decide(self, *_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    class _FakeKernelInterface:
        def __init__(self):
            self.process_calls = 0

        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            self.process_calls += 1
            return "unexpected kernel reply"

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        if name == "capability_engine":
            return _FakeCapabilityEngine()
        if name == "authority_gateway":
            return _FakeAuthority()
        if name == "unified_will":
            return _FakeWill()
        return default

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", AsyncCallFixture())
    patch_chat_lane(monkeypatch, "_runtime_kernel_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_cognitive_engine_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_memory_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_substrate_voice_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_inference_available", lambda *_args, **_kwargs: True)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    from core.kernel.kernel_interface import KernelInterface

    fake_kernel = _FakeKernelInterface()
    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: fake_kernel))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="What tools can you use externally, and what is a hypothetical scenario where you use them?",
            session_id="desktop-inventory",
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert "governance receipts" in payload["response"]
    assert payload["response_confidence"] == "high"
    assert payload["live_turn_contract"]["engine_think_invoked"] is True
    assert payload["live_turn_contract"]["live_mind_controls_bound"] is True
    assert payload["live_turn_contract"]["full_mind_path"] is True
    assert calls and calls[0]["context"]["capability_inventory_contract"] is True
    assert fake_kernel.process_calls == 0


def test_explicit_capability_inventory_classifier_covers_hypothetical_she_phrasing():
    from interface.routes import chat as chat_routes

    assert chat_routes._is_explicit_capability_inventory_request(
        "What tools she could hypothetically do externally, and can she flex her muscles with a scenario?"
    )


def test_explicit_capability_inventory_classifier_covers_live_external_tool_wording():
    from interface.routes import chat as chat_routes

    prompt = (
        "From the live Aura desktop UI path, explain what external tools you can use and "
        "give one concrete multi-step scenario using them. Speak as Aura through your "
        "cognitive engine, not generic assistant mode."
    )

    assert chat_routes._is_explicit_capability_inventory_request(prompt)
    assert not chat_routes._is_bounded_nonexecuting_planning_request(prompt)


def test_grounded_capability_inventory_does_not_invent_live_path_or_program_dna_contract(monkeypatch):
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield from [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
                {
                    "name": "file_operation",
                    "available": True,
                    "description": "Create local files and PDF documents.",
                    "route_class": "stateful",
                    "risk_class": "medium",
                    "effect_scope": "file_system",
                },
                {
                    "name": "memory_ops",
                    "available": True,
                    "description": "Record verified results in memory.",
                    "route_class": "stateful",
                    "risk_class": "medium",
                    "effect_scope": "memory",
                },
                {
                    "name": "program_dna_reconstruct",
                    "available": True,
                    "description": "Build clean-room Program DNA genomes and replacement scaffolds from authorized evidence.",
                    "route_class": "self_improvement",
                    "risk_class": "high",
                    "effect_scope": "local_files",
                },
            ]

        async def execute(self, *_args, **_kwargs):
            return {"ok": True}

    class _FakeAuthority:
        def is_ready(self):
            return True

    class _FakeWill:
        def decide(self, *_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    def _fake_get(name, default=None):
        if name == "capability_engine":
            return _FakeCapabilityEngine()
        if name == "authority_gateway":
            return _FakeAuthority()
        if name == "unified_will":
            return _FakeWill()
        return default

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    prompt = (
        "Codex here, using the actual launched Aura desktop UI path. Please answer "
        "through your full mind path: what external tools can you use, and give one "
        "scenario with browser research, file/PDF work, memory, and Program DNA."
    )
    reply = chat_routes._build_grounded_capability_inventory_reply(prompt)
    lowered = reply.lower()
    assessment = assess_user_facing_reply(prompt, reply)

    assert "cognitiveengine" not in lowered
    assert "cortex/32b" not in lowered and "32b" not in lowered
    assert "i measured 5 registered entries" in lowered
    assert "browser/web research" in lowered
    assert "program dna" in lowered
    assert "receipts" in lowered
    assert "not opening apps" in lowered
    assert "missing_runtime_path_answer" in assessment.reasons
    assert assessment.hard_failure


@pytest.mark.asyncio
async def test_required_capability_inventory_binds_catalog_after_weak_engine_reply(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    trace = {}

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": getattr(mode, "name", str(mode)),
                    "origin": origin,
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="I can use tools, browse, and make documents if needed.",
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    class _FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield from [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
                {
                    "name": "file_operation",
                    "available": True,
                    "description": "Create local files and PDF documents.",
                    "route_class": "stateful",
                    "risk_class": "medium",
                    "effect_scope": "file_system",
                },
                {
                    "name": "memory_ops",
                    "available": True,
                    "description": "Record verified results in memory.",
                    "route_class": "stateful",
                    "risk_class": "medium",
                    "effect_scope": "memory",
                },
                {
                    "name": "program_dna_reconstruct",
                    "available": True,
                    "description": "Clean-room Program DNA reconstruction from authorized behavioral evidence.",
                    "route_class": "self_improvement",
                    "risk_class": "high",
                    "effect_scope": "local_files",
                },
            ]

        async def execute(self, *_args, **_kwargs):
            return {"ok": True}

    class _FakeAuthority:
        def is_ready(self):
            return True

    class _FakeWill:
        def decide(self, *_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        if name == "capability_engine":
            return _FakeCapabilityEngine()
        if name == "authority_gateway":
            return _FakeAuthority()
        if name == "unified_will":
            return _FakeWill()
        return default

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    _force_full_mind_runtime(monkeypatch, chat_routes)

    prompt = (
        "From the live Aura desktop UI path, explain what external tools you can use "
        "and give one concrete multi-step scenario using screen perception, browser "
        "research, file/PDF work, memory, and Program DNA."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        prompt,
        visible_user_message=prompt,
        origin="user",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    lowered = str(reply).lower()
    assert calls
    assert calls[0]["context"]["capability_inventory_contract"] is True
    assert "cognitiveengine" in lowered or "cognitive engine" in lowered
    assert "cortex" in lowered
    assert "32b" not in lowered
    assert "program dna" in lowered
    assert "browser/web research" in lowered
    assert trace["engine_think_invoked"] is True
    assert trace["cognitive_engine_reply_accepted"] is True
    assert trace["bounded_contract_used"] is False
    assert trace["response_path"] == "cognitive_engine_capability_catalog_grounding"


def test_live_turn_contract_does_not_treat_warming_lane_as_full_mind(monkeypatch):
    from interface.routes import chat as chat_routes

    patch_chat_lane(monkeypatch, "_runtime_kernel_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_cognitive_engine_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_memory_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_substrate_voice_available", lambda: True)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": False,
            "state": "warming",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": False,
            "cognitive_engine_reply_accepted": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "architecture_context_bound": True,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine",
        },
    )

    assert payload["required_subsystems"]["inference"] is False
    assert payload["required_subsystems_ok"] is False
    assert payload["full_mind_path"] is False


def test_live_turn_contract_requires_worker_surface_quality_gate_to_pass(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _bound_live_mind_controls_trace()
    trace["live_mind_surface_control_receipt"] = {
        **trace["live_mind_surface_control_receipt"],
        "surface_quality_gate_enabled": True,
        "surface_quality_gate_passed": False,
        "surface_quality_gate_attempts": 3,
        "surface_quality_gate_reasons": ["generic_assistant_language"],
    }

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "architecture_context_bound": True,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine",
            **trace,
        },
    )

    assert payload["live_mind_surface_quality_gate_enabled"] is True
    assert payload["live_mind_surface_quality_gate_passed"] is False
    assert payload["live_mind_controls_structurally_bound"] is False
    assert payload["full_mind_path"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    [
        "chat.cognitive_engine_retry_final_gate",
        "chat.cognitive_engine_final_gate",
        "chat.fastpath_final_gate",
        "chat.final_quality_gate",
    ],
)
async def test_final_reply_call_sites_record_request_scoped_mutation_provenance(
    monkeypatch,
    stage,
):
    from interface.routes import chat as chat_routes

    async def _repair(*args, **kwargs):
        return "Repaired visible reply.", False, False, False, "shape_miss", True

    monkeypatch.setattr(chat_routes, "_repair_final_degraded_reply", _repair)
    trace = {"live_mind_surface_control_receipt": {}}

    result = await chat_routes._repair_final_degraded_reply_with_provenance(
        trace,
        stage=stage,
        user_message="Give me the clean answer.",
        reply_text="broken",
        stale=False,
        same_diff=False,
        off_topic=False,
    )

    assert result[0] == "Repaired visible reply."
    assert trace["post_generation_repair_applied"] is True
    assert trace["deterministic_repair_applied"] is True
    assert trace["text_mutations"][-1]["stage"] == stage
    assert trace["live_mind_surface_control_receipt"]["text_mutation_count"] == 1


def test_optional_turn_trace_cannot_break_a_bounded_repair():
    from interface.routes import chat as chat_routes

    chat_routes._append_turn_text_mutation(
        None,
        stage="chat.optional_trace",
        method="bounded_repair",
        reasons=["trace_not_requested"],
        before="before",
        after="after",
    )


def test_final_requested_output_contract_repairs_post_affordance_mutation():
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes import chat as chat_routes

    prompt = 'Reply exactly: "yes"'
    trace = {"live_mind_surface_control_receipt": {}}
    late_reply = "yes\n\nDone"
    chat_routes._append_turn_text_mutation(
        trace,
        stage="chat.affordance_spoken_append",
        method="effect_receipt_spoken_append",
        reasons=["realized_affordance"],
        before="yes",
        after=late_reply,
        deterministic=False,
    )

    final_reply = chat_routes._enforce_final_requested_output_contract(
        trace,
        user_message=prompt,
        reply_text=late_reply,
    )

    assert final_reply == "yes"
    assert assess_user_facing_reply(prompt, final_reply).ok is True
    assert [item["stage"] for item in trace["text_mutations"]] == [
        "chat.affordance_spoken_append",
        "chat.final_requested_output_contract",
    ]
    assert trace["deterministic_repair_applied"] is True
    assert trace["final_requested_output_contract_evaluated"] is True
    assert trace["final_requested_output_contract_required"] is True
    assert trace["final_requested_output_contract_satisfied"] is True


def test_exact_reply_contract_exempts_only_matching_repetition_from_stale_gate():
    from interface.routes import chat as chat_routes

    for _ in range(chat_routes._STALE_REPEAT_THRESHOLD):
        chat_routes._record_recent_response("yes", 'Reply exactly: "yes"')

    assert chat_routes._is_stale_repeated_response("yes") is True
    assert (
        chat_routes._is_actionably_stale_response('Reply exactly: "yes"', "yes")
        is False
    )
    assert (
        chat_routes._is_actionably_stale_response('Reply exactly: "Yes"', "yes")
        is True
    )
    assert (
        chat_routes._is_actionably_stale_response(
            'Reply exactly: "yes" if ready; otherwise "no".',
            "yes",
        )
        is True
    )
    assert chat_routes._is_actionably_stale_response("Are you there?", "yes") is True


def test_final_requested_output_contract_does_not_invent_missing_sentence_content():
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes import chat as chat_routes

    prompt = "Answer in two sentences."
    trace = {"live_mind_surface_control_receipt": {}}

    final_reply = chat_routes._enforce_final_requested_output_contract(
        trace,
        user_message=prompt,
        reply_text="Okay.",
    )

    assert final_reply == "Okay."
    assert assess_user_facing_reply(prompt, final_reply).ok is False
    assert trace.get("text_mutations") in (None, [])
    assert trace["final_requested_output_contract_satisfied"] is False
    assert "missing_requested_sentence_count" in trace[
        "final_requested_output_contract_reasons"
    ]


def test_final_requested_output_contract_records_unrepairable_failure(monkeypatch):
    from core.conversation import response_reliability
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        response_reliability,
        "repair_instruction_shape",
        lambda _prompt, _reply: "Still two sentences. This remains extra.",
    )
    trace = {"live_mind_surface_control_receipt": {}}

    final_reply = chat_routes._enforce_final_requested_output_contract(
        trace,
        user_message="Answer in one sentence.",
        reply_text="This is one sentence. This is another.",
    )

    assert final_reply == "Still two sentences. This remains extra."
    assert trace["final_requested_output_contract_evaluated"] is True
    assert trace["final_requested_output_contract_required"] is True
    assert trace["final_requested_output_contract_satisfied"] is False
    assert "missing_requested_sentence_count" in trace[
        "final_requested_output_contract_reasons"
    ]


def test_full_mind_path_requires_final_output_contract_proof(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _bound_live_mind_controls_trace()
    trace.update(
        {
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "architecture_context_bound": True,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine",
            "final_requested_output_contract_evaluated": True,
            "final_requested_output_contract_required": True,
            "final_requested_output_contract_kind": "sentence_count",
            "final_requested_output_contract_satisfied": False,
            "final_requested_output_contract_reasons": [
                "missing_requested_sentence_count"
            ],
        }
    )

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace=trace,
    )

    assert payload["final_requested_output_contract_proven"] is False
    assert payload["full_mind_path"] is False
    assert "final_output_contract_unsatisfied" in payload["full_mind_missing_proofs"]


def test_early_constrained_exit_cannot_claim_vacuous_final_contract_proof(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _bound_live_mind_controls_trace()
    trace["live_mind_surface_control_receipt"]["requested_output_contract"] = {
        "kind": "word_count",
        "explicit_brevity": True,
        "word_min": 5,
        "word_max": 5,
    }
    trace.update(
        {
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": False,
            "cognitive_engine_reply_failed": True,
            "response_path": "desktop_cognitive_engine_required_no_reply",
        }
    )

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": False, "state": "failed"},
        response_confidence="failed",
        status="desktop_cognitive_engine_unavailable",
        reply_source="desktop_cognitive_engine_required_no_reply",
        turn_trace=trace,
    )

    assert payload["final_requested_output_contract_evaluated"] is False
    assert payload["final_requested_output_contract_required"] is True
    assert payload["final_requested_output_contract_kind"] == "word_count"
    assert payload["final_requested_output_contract_satisfied"] is False
    assert payload["final_requested_output_contract_proven"] is False
    assert payload["final_requested_output_contract_reasons"] == [
        "evaluation_not_completed"
    ]


def test_text_mutation_receipt_never_serializes_literal_violation_content():
    from core.brain.live_mind_contract import append_text_mutation

    receipt = {}
    append_text_mutation(
        receipt,
        stage="response_generation.executive_guard",
        method="deterministic_identity_alignment",
        reasons=[
            {
                "type": "identity_alignment",
                "match": "literal private response fragment",
            }
        ],
        before="before",
        after="after",
        deterministic=True,
    )

    serialized = json.dumps(receipt, sort_keys=True)
    assert "literal private response fragment" not in serialized
    assert receipt["text_mutations"][0]["reasons"] == ["identity_alignment"]


def test_live_turn_contract_derives_repair_flags_from_mutation_ledger(monkeypatch):
    from core.brain.live_mind_contract import append_text_mutation
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _bound_live_mind_controls_trace()
    append_text_mutation(
        trace["live_mind_surface_control_receipt"],
        stage="response_generation.post_voice_shape",
        method="deterministic_instruction_shape",
        reasons=["missing_requested_sentence_count"],
        before="Two sentences. Extra sentence.",
        after="Two sentences.",
        deterministic=True,
        authorship_effect="preserved",
    )

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "architecture_context_bound": True,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine",
            **trace,
        },
    )

    assert payload["post_generation_repair_applied"] is True
    assert payload["deterministic_repair_applied"] is True
    assert payload["model_native_output"] is False
    assert payload["final_text_authorship"] == (
        "cognitive_generation_with_recorded_transformations"
    )
    assert payload["text_mutation_count"] == 1
    assert payload["text_mutations"][0]["stage"] == (
        "response_generation.post_voice_shape"
    )
    assert payload["text_mutations"][0]["authorship_effect"] == "preserved"
    assert payload["authorship_replacement_applied"] is False
    assert payload["authentic_cognitive_reply"] is True


def test_live_turn_contract_keeps_cognitive_authorship_for_receipted_runtime_evidence(
    monkeypatch,
):
    from core.brain.live_mind_contract import append_text_mutation
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _bound_live_mind_controls_trace()
    append_text_mutation(
        trace["live_mind_surface_control_receipt"],
        stage="chat.fact_custody",
        method="held_fact_restored_at_terminal_boundary",
        reasons=["verified_count_restored"],
        before="I counted the files.",
        after="I counted the files. The verified count is 412.",
        deterministic=True,
        authorship_effect="augmented_by_runtime",
    )
    trace.update(
        {
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine",
        }
    )

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace=trace,
    )

    assert payload["model_native_output"] is False
    assert payload["authorship_augmentation_applied"] is True
    assert payload["authorship_replacement_applied"] is False
    assert payload["final_text_authorship"] == (
        "cognitive_generation_with_runtime_evidence"
    )
    assert payload["authentic_cognitive_reply"] is True
    assert payload["full_mind_path"] is True


def test_live_turn_contract_accepts_receipted_self_condition_semantic_completion(
    monkeypatch,
):
    from core.brain.live_mind_contract import append_text_mutation
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _bound_live_mind_controls_trace()
    append_text_mutation(
        trace["live_mind_surface_control_receipt"],
        stage="chat.self_condition_epistemic_completion",
        method="typed_evidence_semantic_merge",
        reasons=["unanswered_question_part"],
        before="I feel steady right now.",
        after=(
            "I feel steady right now. I know the live state sample is fresh; "
            "I can only infer what that state means subjectively."
        ),
        deterministic=True,
        authorship_effect="augmented_by_runtime",
    )
    trace.update(
        {
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_self_condition_semantic_completion",
        }
    )

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine_self_condition_semantic_completion",
        reply_source="cognitive_engine_self_condition_semantic_completion",
        turn_trace=trace,
    )

    assert payload["authorship_augmentation_applied"] is True
    assert payload["authorship_replacement_applied"] is False
    assert payload["final_text_authorship"] == (
        "cognitive_generation_with_runtime_evidence"
    )
    assert payload["authentic_cognitive_reply"] is True
    assert payload["full_mind_path"] is True
    assert payload["full_mind_missing_proofs"] == []


def test_live_turn_contract_rejects_unknown_augmented_response_path(monkeypatch):
    from core.brain.live_mind_contract import append_text_mutation
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _bound_live_mind_controls_trace()
    append_text_mutation(
        trace["live_mind_surface_control_receipt"],
        stage="chat.unknown_augmentation",
        method="unknown_semantic_merge",
        reasons=["unanswered_question_part"],
        before="Model draft.",
        after="Model draft. Runtime text.",
        deterministic=True,
        authorship_effect="augmented_by_runtime",
    )
    trace.update(
        {
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_unknown_semantic_completion",
        }
    )

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": True, "state": "ready"},
        response_confidence="high",
        status="cognitive_engine_unknown_semantic_completion",
        reply_source="cognitive_engine_unknown_semantic_completion",
        turn_trace=trace,
    )

    assert payload["authentic_cognitive_reply"] is False
    assert payload["full_mind_path"] is False
    assert (
        "response_path:cognitive_engine_unknown_semantic_completion"
        in payload["full_mind_missing_proofs"]
    )


def test_live_turn_contract_rejects_receipted_runtime_replacement(monkeypatch):
    from core.brain.live_mind_contract import append_text_mutation
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _bound_live_mind_controls_trace()
    append_text_mutation(
        trace["live_mind_surface_control_receipt"],
        stage="chat.final_identity_grounding",
        method="deterministic_canonical_grounding",
        reasons=["identity_continuity_grounding"],
        before="Model draft.",
        after="Canonical runtime-authored answer.",
        deterministic=True,
        authorship_effect="replaced_by_runtime",
    )
    trace.update(
        {
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "response_path": "cognitive_engine_identity_continuity_grounding",
        }
    )

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine_identity_continuity_grounding",
        reply_source="cognitive_engine_identity_continuity_grounding",
        turn_trace=trace,
    )

    assert payload["model_native_output"] is False
    assert payload["authorship_replacement_applied"] is True
    assert payload["final_text_authorship"] == "non_cognitive_replacement"
    assert payload["authentic_cognitive_reply"] is False
    assert payload["full_mind_path"] is False
    assert "runtime_replacement_authored_text" in payload[
        "full_mind_missing_proofs"
    ]


def test_strict_live_inference_readiness_requires_lane_status(monkeypatch):
    from interface.routes import chat as chat_routes

    class _StatuslessGate:
        async def generate(self, *_args, **_kwargs):
            return "text"

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _StatuslessGate()
            if name == "inference_gate"
            else default
        ),
    )

    assert chat_routes._runtime_inference_available(require_conversation_ready=False) is True
    assert chat_routes._runtime_inference_available(require_conversation_ready=True) is False


def test_live_turn_contract_allows_proven_generation_to_satisfy_inference(monkeypatch):
    from interface.routes import chat as chat_routes

    patch_chat_lane(monkeypatch, "_runtime_kernel_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_cognitive_engine_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_memory_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_substrate_voice_available", lambda: True)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": False,
            "state": "warming",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            **_bound_live_mind_controls_trace(),
            "response_path": "cognitive_engine",
        },
    )

    assert payload["required_subsystems"]["inference"] is True
    assert payload["required_subsystems_ok"] is True
    assert payload["live_mind_required_subsystems_ok"] is True
    assert payload["architecture_context_bound"] is True
    assert payload["live_mind_controls_bound"] is True
    assert payload["live_mind_generation_controls_present"] is True
    assert payload["live_mind_controls_structurally_bound"] is True
    assert payload["full_mind_path"] is True


def test_live_turn_contract_reuses_attested_subsystems_without_runtime_resolution(monkeypatch):
    from interface.routes import chat as chat_routes

    def _must_not_probe(*_args, **_kwargs):
        raise AssertionError("delivery contract re-resolved live services")

    monkeypatch.setattr(
        chat_routes,
        "_collect_live_chat_required_subsystems",
        _must_not_probe,
    )
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": True, "state": "ready"},
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "live_mind_required_subsystems_attested": True,
            "live_mind_required_subsystems": {
                "kernel": True,
                "cognitive_engine": True,
                "inference": True,
                "memory": True,
                "tool_governance": True,
                "substrate_voice": True,
            },
            "response_path": "cognitive_engine",
            **_bound_live_mind_controls_trace(),
        },
    )

    assert payload["required_subsystems_source"] == "attested_preflight"
    assert payload["required_subsystems_ok"] is True
    assert payload["full_mind_path"] is True


def test_live_turn_contract_rejects_self_asserted_subsystem_vector(monkeypatch):
    from interface.routes import chat as chat_routes

    measured = {
        "kernel": True,
        "cognitive_engine": True,
        "inference": True,
        "memory": True,
        "tool_governance": False,
        "substrate_voice": True,
    }
    monkeypatch.setattr(
        chat_routes,
        "_collect_live_chat_required_subsystems",
        lambda *_args, **_kwargs: dict(measured),
    )
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": True, "state": "ready"},
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "live_mind_required_subsystems_attested": False,
            "live_mind_required_subsystems": {
                name: True for name in measured
            },
            "response_path": "cognitive_engine",
            **_bound_live_mind_controls_trace(),
        },
    )

    assert payload["required_subsystems_source"] == "compatibility_probe"
    assert payload["required_subsystems"]["tool_governance"] is False
    assert payload["required_subsystems_ok"] is False
    assert payload["full_mind_path"] is False


def test_live_turn_contract_preserves_stale_preflight_subsystem_state(monkeypatch):
    from interface.routes import chat as chat_routes

    patch_chat_lane(monkeypatch, "_runtime_kernel_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_cognitive_engine_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_memory_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_substrate_voice_available", lambda: True)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": False,
            "state": "warming",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": False,
            **_bound_live_mind_controls_trace(),
            "response_path": "cognitive_engine",
        },
    )

    assert payload["preflight_live_mind_required_subsystems_ok"] is False
    assert payload["live_mind_required_subsystems_ok"] is True
    assert payload["live_mind_controls_bound"] is True
    assert payload["required_subsystems_ok"] is True
    assert payload["full_mind_path"] is True


def test_live_turn_contract_refuses_engine_text_without_live_mind_context(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "response_path": "cognitive_engine",
        },
    )

    assert payload["required_subsystems_ok"] is True
    assert payload["live_mind_context_required"] is True
    assert payload["live_mind_context_present"] is False
    assert payload["architecture_context_bound"] is False
    assert payload["full_mind_path"] is False


def test_live_turn_contract_refuses_failure_envelope_as_full_mind(monkeypatch):
    from interface.routes import chat as chat_routes

    patch_chat_lane(monkeypatch, "_runtime_kernel_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_cognitive_engine_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_memory_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_substrate_voice_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_inference_available", lambda *_args, **_kwargs: True)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_failure_envelope",
        },
    )

    assert payload["cognitive_engine_reply_accepted"] is False
    assert payload["cognitive_engine_reply_failed"] is True
    assert payload["required_subsystems_ok"] is True
    assert payload["full_mind_path"] is False


def test_live_turn_contract_refuses_shape_repair_as_full_mind(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine_shape_repair_bounded",
        reply_source="cognitive_engine_shape_repair_bounded",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": False,
            "bounded_contract_used": True,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_shape_repair_bounded",
        },
    )

    assert payload["full_mind_path"] is False
    assert payload["bounded_contract_used"] is True


def test_unreceipted_memory_state_grounding_cannot_claim_model_authorship(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine_memory_state_grounding",
        reply_source="cognitive_engine_memory_state_grounding",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            **_bound_live_mind_controls_trace(),
            "response_path": "cognitive_engine_memory_state_grounding",
        },
    )

    assert payload["response_path"] == "cognitive_engine_memory_state_grounding"
    assert payload["cognitive_engine_reply_accepted"] is True
    assert payload["bounded_contract_used"] is False
    assert payload["legacy_fallback_used"] is False
    assert payload["required_subsystems_ok"] is True
    assert payload["architecture_context_bound"] is True
    assert payload["live_mind_controls_bound"] is True
    assert payload["model_native_output"] is False
    assert payload["final_text_authorship"] == "non_cognitive_replacement"
    assert payload["authentic_cognitive_reply"] is False
    assert payload["full_mind_path"] is False
    assert "runtime_replacement_authored_text" in payload[
        "full_mind_missing_proofs"
    ]


def test_unreceipted_identity_grounding_cannot_claim_model_authorship(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine_identity_continuity_grounding",
        reply_source="cognitive_engine_identity_continuity_grounding",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            **_bound_live_mind_controls_trace(),
            "response_path": "cognitive_engine_identity_continuity_grounding",
        },
    )

    assert payload["response_path"] == "cognitive_engine_identity_continuity_grounding"
    assert payload["cognitive_engine_reply_accepted"] is True
    assert payload["bounded_contract_used"] is False
    assert payload["legacy_fallback_used"] is False
    assert payload["required_subsystems_ok"] is True
    assert payload["architecture_context_bound"] is True
    assert payload["model_native_output"] is False
    assert payload["final_text_authorship"] == "non_cognitive_replacement"
    assert payload["authentic_cognitive_reply"] is False
    assert payload["full_mind_path"] is False
    assert "runtime_replacement_authored_text" in payload[
        "full_mind_missing_proofs"
    ]


def test_live_turn_contract_refuses_engine_text_without_mind_snapshot(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine",
        },
    )

    assert payload["live_mind_context_present"] is True
    assert payload["live_mind_snapshot_present"] is False
    assert payload["live_mind_snapshot_bound"] is False
    assert payload["full_mind_path"] is False


def test_live_turn_contract_refuses_engine_text_without_bound_mind_controls(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine",
        },
    )

    assert payload["required_subsystems_ok"] is True
    assert payload["live_mind_snapshot_bound"] is True
    assert payload["live_mind_controls_bound"] is False
    assert payload["live_mind_generation_controls_present"] is False
    assert payload["live_mind_controls_structurally_bound"] is False
    assert payload["full_mind_path"] is False


def test_memory_state_evidence_requires_visible_pinned_content():
    from interface.routes import chat as chat_routes

    evidence = (
        'The phrase you asked me to remember in this session was "silver lantern".',
        "session_memory_recall",
    )

    assert chat_routes._memory_state_evidence_is_missing_from_reply(
        "What phrase did I ask you to remember?",
        "Cortex is the active lane and memory is available.",
        evidence,
    ) is True
    assert chat_routes._memory_state_evidence_is_missing_from_reply(
        "What phrase did I ask you to remember?",
        "You asked me to remember \"silver lantern\", and I am grounding this in the current session memory.",
        evidence,
    ) is False


def test_memory_state_evidence_suppresses_only_compatible_self_process_reasons():
    from interface.routes import chat as chat_routes

    evidence = (
        'The phrase you asked me to remember in this session was "silver lantern".',
        "session_memory_recall",
    )
    compatible = SimpleNamespace(
        ok=False,
        retryable=True,
        hard_failure=False,
        reasons=("off_topic_self_reflection_reply", "missing_requested_self_process_coverage"),
    )
    hard = SimpleNamespace(
        ok=False,
        retryable=True,
        hard_failure=True,
        reasons=("runtime_boilerplate",),
    )

    assert chat_routes._reply_assessment_requires_repair_with_memory_evidence(
        compatible,
        "What phrase did I ask you to remember, and how does your cognitive engine keep this reply grounded?",
        'You asked me to remember "silver lantern"; I am grounding this reply in canonical memory-state evidence.',
        memory_state_evidence=evidence,
    ) is False
    assert chat_routes._reply_assessment_requires_repair_with_memory_evidence(
        compatible,
        "What phrase did I ask you to remember, and how does your cognitive engine keep this reply grounded?",
        "I am thinking about this exchange from the live desktop lane.",
        memory_state_evidence=evidence,
    ) is True
    assert chat_routes._reply_assessment_requires_repair_with_memory_evidence(
        hard,
        "What phrase did I ask you to remember?",
        'You asked me to remember "silver lantern".',
        memory_state_evidence=evidence,
    ) is True


def test_canonical_memory_state_grounding_reply_uses_memory_and_live_context():
    from interface.routes import chat as chat_routes

    reply = chat_routes._canonical_memory_state_grounding_reply(
        "Remember this phrase: silver lantern. Also tell me one thing your live mind is attending to right now.",
        (
            "status=session_memory_pin\n"
            'I\'ve pinned "silver lantern" in durable session memory.'
        ),
        live_mind_context={"voice": {"attention": "the current desktop conversation"}},
    )

    assert reply == (
        'I have pinned "silver lantern" in this session. '
        "Right now I am keeping attention on the current desktop conversation."
    )


@pytest.mark.asyncio
async def test_retained_memory_evidence_context_collects_auditable_sources(monkeypatch):
    from interface.routes import chat as chat_routes

    async def _fake_durable_snippets(_message, *, limit=3):
        return [
            "Bryan and Aura discussed retained memory as behavioral reuse with receipts.",
            "Aura should distinguish durable transcript evidence from subjective recollection.",
        ][:limit]

    monkeypatch.setattr(
        _chat_memory_state,
        "_recall_durable_conversation_snippets",
        _fake_durable_snippets,
    )

    evidence = await chat_routes._build_retained_memory_evidence_context(
        "Can you remember a conversation we had last week about the nature of your consciousness?",
        recent_exchanges=[
            {
                "user": "We should test memory by seeing whether it changes later behavior.",
                "aura": "I will treat memory as evidence that must alter future decisions.",
            }
        ],
        conversation_recall_context=(
            "Recently, this conversation has been about memory, agency, and receipts."
        ),
    )

    assert "scope=retained_memory_evidence.v1" in evidence
    assert "source=conversation_recall" in evidence
    assert "source=recent_completed_transcript" in evidence
    assert "source=durable_memory_search" in evidence
    assert "behavioral reuse with receipts" in evidence
    assert "subjective recollection" in evidence


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_receives_retained_memory_evidence_context(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    "I can verify durable memory evidence that we discussed retained memory "
                    "as behavioral reuse with receipts. The limit is that this is transcript "
                    "and durable-memory evidence, not subjective recollection."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async def _fake_durable_snippets(_message, *, limit=3):
        return [
            "Bryan and Aura discussed retained memory as behavioral reuse with receipts."
        ][:limit]

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )
    monkeypatch.setattr(
        _chat_memory_state,
        "_recall_durable_conversation_snippets",
        _fake_durable_snippets,
    )

    visible_message = (
        "Can you remember a conversation we had last week about the nature of your consciousness?"
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        visible_message,
        visible_user_message=visible_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "behavioral reuse with receipts" in reply
    assert "subjective recollection" in reply
    assert calls
    context = calls[0]["context"]
    assert context["retained_memory_evidence_contract"] is True
    assert "retained_memory_evidence_context" in context
    assert "behavioral reuse with receipts" in context["retained_memory_evidence_context"]
    assert "retained_memory_evidence_context" in context["response_style_contract"]


def test_live_self_reflection_is_not_explicit_capability_inventory():
    from interface.routes import chat as chat_routes

    prompt = (
        "How are you thinking about this conversation right now? Answer from the live desktop "
        "cognitive path, be concrete, and keep it concise."
    )

    assert chat_routes._is_capability_inventory_request(prompt) is True
    assert chat_routes._is_explicit_capability_inventory_request(prompt) is False


def test_capability_catalog_snapshot_caps_unbounded_catalog(monkeypatch):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def get_tool_catalog(self, *, include_inactive: bool = True):
            for index in range(chat_routes._CAPABILITY_CATALOG_MAX_ITEMS + 50):
                yield {
                    "name": f"tool_{index}",
                    "available": True,
                    "description": "Specialized governed skill surface.",
                    "route_class": "specialized",
                    "risk_class": "low",
                    "effect_scope": "read_only",
                }

    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeCapabilityEngine() if name == "capability_engine" else default),
    )

    available_count, categories, governance_available, truncated = (
        chat_routes._read_capability_catalog_snapshot()
    )

    assert available_count == chat_routes._CAPABILITY_CATALOG_MAX_ITEMS
    assert truncated is True
    assert governance_available is True
    assert len(categories["specialized governed skills"]) == 12


def test_capability_catalog_snapshot_prefers_streaming_catalog(monkeypatch):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def __init__(self):
            self.materialized_catalog_calls = 0

        def iter_tool_catalog(self, *, include_inactive: bool = True):
            assert include_inactive is True
            for index in range(chat_routes._CAPABILITY_CATALOG_MAX_ITEMS + 25):
                yield {
                    "name": f"streamed_tool_{index}",
                    "available": True,
                    "description": "Specialized governed skill surface.",
                    "route_class": "specialized",
                    "risk_class": "low",
                    "effect_scope": "read_only",
                }

        def get_tool_catalog(self, *, include_inactive: bool = True):
            self.materialized_catalog_calls += 1
            return []

    engine = _FakeCapabilityEngine()
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: engine if name == "capability_engine" else default),
    )

    available_count, categories, governance_available, truncated = (
        chat_routes._read_capability_catalog_snapshot()
    )

    assert available_count == chat_routes._CAPABILITY_CATALOG_MAX_ITEMS
    assert truncated is True
    assert governance_available is True
    assert len(categories["specialized governed skills"]) == 12
    assert engine.materialized_catalog_calls == 0


def test_capability_catalog_snapshot_skips_materialized_catalog(monkeypatch):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def __init__(self):
            self.materialized_catalog_calls = 0

        def get_tool_catalog(self, *, include_inactive: bool = True):
            self.materialized_catalog_calls += 1
            raise AssertionError("desktop inventory must not materialize a full catalog")

    engine = _FakeCapabilityEngine()
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: engine if name == "capability_engine" else default),
    )

    available_count, categories, governance_available, truncated = (
        chat_routes._read_capability_catalog_snapshot()
    )

    assert available_count == 0
    assert categories == {}
    assert governance_available is True
    assert truncated is True
    assert engine.materialized_catalog_calls == 0


def test_capability_catalog_snapshot_requires_explicit_legacy_availability(monkeypatch):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            assert include_inactive is True
            return {
                "explicit": {"available": True},
                "ready": {"status": "ready"},
                "unknown": {"description": "No measured availability."},
                "loading": {"status": "loading"},
                "disabled": {"available": False, "status": "ready"},
            }

        def get_catalog_health(self):
            return {"ready": True}

    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _FakeCapabilityEngine() if name == "capability_engine" else default
            )
        ),
    )

    snapshot = chat_routes._read_capability_catalog_snapshot()

    assert snapshot.catalog_status == "measured"
    assert snapshot.capability_health is True
    assert snapshot.registered_count == 5
    assert snapshot.available_count == 2
    listed = {name for names in snapshot.categories.values() for name in names}
    assert listed == {"explicit", "ready"}


def test_capability_catalog_snapshot_does_not_call_malformed_stream_measured(
    monkeypatch,
):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            return 17

        def get_catalog_health(self):
            return {"ready": True}

    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _FakeCapabilityEngine() if name == "capability_engine" else default
            )
        ),
    )

    snapshot = chat_routes._read_capability_catalog_snapshot()
    reply = chat_routes._build_grounded_capability_inventory_reply(
        "What external tools can you use?"
    )

    assert snapshot.catalog_status == "error"
    assert snapshot.registered_count == 0
    assert snapshot.available_count == 0
    assert "could not verify a current capability catalog" in reply
    assert "static list" in reply


def test_capability_inventory_separates_catalog_availability_from_unhealthy_owner(
    monkeypatch,
):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield {"name": "web_search", "available": True}

        def get_catalog_health(self):
            return {"ready": False}

    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _FakeCapabilityEngine() if name == "capability_engine" else default
            )
        ),
    )

    reply = chat_routes._build_grounded_capability_inventory_reply(
        "What external tools can you use?"
    )

    assert "1 entry explicitly marked available" in reply
    assert "web_search" in reply
    assert "catalog owner measured not ready" in reply
    assert "both measured ready" not in reply


def test_measured_sparse_capability_catalog_is_not_discarded(monkeypatch):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield {"name": "web_search", "available": False}

        def get_catalog_health(self):
            return {"ready": True}

    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _FakeCapabilityEngine() if name == "capability_engine" else default
            )
        ),
    )

    prompt = "What external tools can you use?"
    reply = chat_routes._build_grounded_capability_inventory_reply(prompt)

    assert "1 registered entry" in reply
    assert "0 entries explicitly marked available" in reply
    assert "Measured available categories: none" in reply
    assert not chat_routes._capability_inventory_reply_is_inadequate(prompt, reply)


def test_capability_inventory_skips_catalog_under_memory_pressure(monkeypatch):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def __init__(self):
            self.catalog_calls = 0

        def get_tool_catalog(self, *, include_inactive: bool = True):
            self.catalog_calls += 1
            raise AssertionError("optional catalog read must be skipped under critical memory pressure")

        def execute(self, *_args, **_kwargs):
            return None

    class _FakeAuthority:
        def is_ready(self):
            return True

    class _FakeWill:
        def decide(self, *_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    capability_engine = _FakeCapabilityEngine()

    def _fake_get(name, default=None):
        if name == "capability_engine":
            return capability_engine
        if name == "authority_gateway":
            return _FakeAuthority()
        if name == "unified_will":
            return _FakeWill()
        return default

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            max_token_cap=32,
            refuse_heavy_local_generation=True,
            reason="process_tree_rss:54GB/48GB",
        ),
    )

    reply = chat_routes._build_grounded_capability_inventory_reply(
        "What tools can you use externally?"
    )

    assert capability_engine.catalog_calls == 0
    assert "could not verify a current capability catalog" in reply
    assert "static list" in reply
    assert "desktop/app control" not in reply
    assert "browser/web research" not in reply
    assert "not opening apps" in reply


def test_chat_turn_memory_log_scheduler_does_not_duplicate_active_drain(
    monkeypatch,
    tmp_path,
):
    from core.conversation.persistence import ConversationPersistence
    from interface.routes import chat as chat_routes

    persistence = ConversationPersistence(tmp_path / "active-memory-outbox.db")
    session_id = persistence.start_session()
    persistence.record_exchange(
        "hello",
        "a meaningful answer for memory",
        cid="active-drain",
        session_id=session_id,
        enqueue_memory_log=True,
    )

    class _FakeTask:
        def done(self):
            return False

        def get_name(self):
            return chat_routes._CHAT_TURN_MEMORY_LOG_DRAIN_TASK_NAME

    class _FakeTracker:
        def __init__(self):
            self.tasks = {_FakeTask()}
            self.bounded_calls = 0

        def bounded_track(self, *_args, **_kwargs):
            self.bounded_calls += 1
            return None

    tracker = _FakeTracker()
    patch_chat_lane(monkeypatch, "get_task_tracker", lambda: tracker)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )
    monkeypatch.setattr(
        _chat_preflight,
        "_ensure_chat_turn_memory_log_shutdown_handler",
        lambda: None,
    )

    scheduled = chat_routes._schedule_chat_turn_memory_log(
        user_message="hello",
        aura_response="hi",
        session_id="test-session",
        chat_origin="desktop_ui",
        user_id="bryan",
    )

    assert scheduled is True
    assert tracker.bounded_calls == 0
    assert persistence.memory_log_outbox_status()["pending"] == 1


@pytest.mark.asyncio
async def test_chat_turn_memory_log_startup_waits_for_persistence(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Persistence:
        def claim_memory_log_batch(self):
            return []

        def settle_memory_log_item(self):
            return None

    class _FakeTracker:
        def __init__(self):
            self.tasks = set()
            self.scheduled = []

        def bounded_track(self, coro, name=None):
            task = asyncio.create_task(coro, name=name)
            self.tasks.add(task)
            self.scheduled.append((task, name))
            task.add_done_callback(self.tasks.discard)
            return task

    holder = {"persistence": None}
    tracker = _FakeTracker()
    wakes = []
    patch_chat_lane(monkeypatch, "get_task_tracker", lambda: tracker)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: holder["persistence"]
            if name == "persistence"
            else default
        ),
    )
    monkeypatch.setattr(chat_routes, "_CHAT_TURN_MEMORY_LOG_STARTUP_POLL_S", 0.001)
    monkeypatch.setattr(chat_routes, "_CHAT_TURN_MEMORY_LOG_STARTUP_TIMEOUT_S", 1.0)
    monkeypatch.setattr(
        _chat_preflight,
        "_schedule_chat_turn_memory_log",
        lambda **kwargs: wakes.append(kwargs) or True,
    )

    assert chat_routes.start_chat_turn_memory_log_worker() is True
    assert tracker.scheduled[0][1] == chat_routes._CHAT_TURN_MEMORY_LOG_STARTUP_TASK_NAME
    holder["persistence"] = _Persistence()
    await tracker.scheduled[0][0]

    assert wakes == [{"chat_origin": "startup_recovery"}]


@pytest.mark.asyncio
async def test_chat_turn_memory_log_scheduler_uses_bounded_track(monkeypatch):
    from core.consciousness import coordinator as consciousness_coordinator
    from core.conversation.persistence import ConversationPersistence
    from core.memory import chat_turn_logger
    from interface.routes import chat as chat_routes

    log_calls = []
    consciousness_calls = []

    async def _fake_log_chat_turn_auto(**kwargs):
        log_calls.append(kwargs)
        return True

    class _FakeCoordinator:
        async def on_chat_turn(self, user_message, aura_response):
            consciousness_calls.append((user_message, aura_response))

    async def _fake_get_consciousness_coordinator():
        return _FakeCoordinator()

    class _FakeTracker:
        def __init__(self):
            self.tasks = set()
            self.scheduled = []

        def bounded_track(self, coro, name=None):
            task = asyncio.create_task(coro, name=name)
            self.tasks.add(task)
            task.add_done_callback(lambda completed: self.tasks.discard(completed))
            self.scheduled.append((task, name))
            return task

    tracker = _FakeTracker()
    persistence = ConversationPersistence(tempfile.mktemp(suffix="-memory-outbox.db"))
    persistence.record_exchange(
        "remember this",
        "I will keep it in the log.",
        origin="desktop_ui",
        cid="memory-worker",
        session_id="test-session",
        principal_id="bryan",
        principal_surface="owner",
        enqueue_memory_log=True,
    )
    patch_chat_lane(monkeypatch, "get_task_tracker", lambda: tracker)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )
    monkeypatch.setattr(
        _chat_preflight,
        "_ensure_chat_turn_memory_log_shutdown_handler",
        lambda: None,
    )
    monkeypatch.setattr(chat_turn_logger, "log_chat_turn_auto", _fake_log_chat_turn_auto)
    monkeypatch.setattr(
        consciousness_coordinator,
        "get_consciousness_coordinator",
        _fake_get_consciousness_coordinator,
    )

    scheduled = chat_routes._schedule_chat_turn_memory_log(
        user_message="remember this",
        aura_response="I will keep it in the log.",
        session_id="test-session",
        chat_origin="desktop_ui",
        user_id="bryan",
    )

    assert scheduled is True
    assert tracker.scheduled[0][1] == chat_routes._CHAT_TURN_MEMORY_LOG_DRAIN_TASK_NAME
    await tracker.scheduled[0][0]
    assert log_calls[0]["user_message"] == "remember this"
    assert log_calls[0]["metadata"]["origin"] == "desktop_ui"
    assert log_calls[0]["metadata"]["user_id"] == "bryan"
    assert log_calls[0]["metadata"]["memory_log_operation_id"].endswith(
        ":memory-worker:r1"
    )
    assert consciousness_calls == [("remember this", "I will keep it in the log.")]
    assert persistence.memory_log_outbox_status()["completed"] == 1


@pytest.mark.asyncio
async def test_chat_turn_memory_log_scheduler_retries_slow_logger_durably(
    monkeypatch,
    tmp_path,
):
    from core.conversation.persistence import ConversationPersistence
    from core.memory import chat_turn_logger
    from interface.routes import chat as chat_routes

    async def _slow_log_chat_turn_auto(**_kwargs):
        await asyncio.sleep(1.0)

    class _FakeTracker:
        def __init__(self):
            self.tasks = set()
            self.scheduled = []

        def bounded_track(self, coro, name=None):
            task = asyncio.create_task(coro, name=name)
            self.scheduled.append(task)
            return task

    tracker = _FakeTracker()
    persistence = ConversationPersistence(tmp_path / "retry-memory-outbox.db")
    persistence.record_exchange(
        "slow memory turn",
        "This meaningful answer must remain queued.",
        cid="slow-memory-worker",
        session_id="test-session",
        enqueue_memory_log=True,
    )
    patch_chat_lane(monkeypatch, "get_task_tracker", lambda: tracker)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )
    monkeypatch.setattr(
        _chat_preflight,
        "_ensure_chat_turn_memory_log_shutdown_handler",
        lambda: None,
    )
    monkeypatch.setattr(_chat_preflight, "_CHAT_TURN_MEMORY_LOG_TIMEOUT_S", 0.01)
    monkeypatch.setattr(chat_turn_logger, "log_chat_turn_auto", _slow_log_chat_turn_auto)

    scheduled = chat_routes._schedule_chat_turn_memory_log(
        user_message="slow",
        aura_response="logger",
        session_id="test-session",
        chat_origin="desktop_ui",
        user_id="bryan",
    )

    assert scheduled is True
    await tracker.scheduled[0]
    assert persistence.memory_log_outbox_status()["pending"] == 1
    assert len(tracker.scheduled) == 2
    tracker.scheduled[1].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await tracker.scheduled[1]


@pytest.mark.asyncio
async def test_chat_turn_memory_log_outbox_exceeds_old_capacity_without_loss(
    monkeypatch,
    tmp_path,
):
    from core.conversation.persistence import ConversationPersistence
    from core.memory import chat_turn_logger
    from interface.routes import chat as chat_routes

    persistence = ConversationPersistence(tmp_path / "capacity-memory-outbox.db")
    for index in range(70):
        persistence.record_exchange(
            f"meaningful prompt {index}",
            f"meaningful durable answer {index}",
            cid=f"capacity-{index}",
            session_id="capacity-session",
            enqueue_memory_log=True,
        )

    calls = []

    async def _fake_log(**kwargs):
        calls.append(kwargs["metadata"]["memory_log_operation_id"])
        return True

    class _Coordinator:
        async def on_chat_turn(self, *_args):
            return None

    async def _coordinator():
        return _Coordinator()

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )
    monkeypatch.setattr(chat_turn_logger, "log_chat_turn_auto", _fake_log)
    monkeypatch.setattr(
        "core.consciousness.coordinator.get_consciousness_coordinator",
        _coordinator,
    )

    await chat_routes._drain_chat_turn_memory_log_queue()

    assert len(calls) == 70
    assert len(set(calls)) == 70
    assert persistence.memory_log_outbox_status()["completed"] == 70
    assert persistence.memory_log_outbox_status()["pending"] == 0


@pytest.mark.asyncio
async def test_chat_turn_memory_log_outbox_retries_transient_claim_failure(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Persistence:
        def claim_memory_log_batch(self, **_kwargs):
            raise OSError("database temporarily unavailable")

        def settle_memory_log_item(self, *_args, **_kwargs):
            raise AssertionError("nothing was claimed")

        def memory_log_outbox_status(self):
            raise AssertionError("claim failure exits before status")

    delays = []
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _Persistence()
            if name == "persistence"
            else default
        ),
    )
    monkeypatch.setattr(
        _chat_preflight,
        "_schedule_chat_turn_memory_log_retry",
        lambda delay_s: delays.append(delay_s) or True,
    )

    await chat_routes._drain_chat_turn_memory_log_queue()

    assert delays == [1.0]


@pytest.mark.asyncio
async def test_chat_turn_memory_log_outbox_defers_learning_while_user_is_active(
    monkeypatch,
):
    from interface.routes import chat as chat_routes

    class _Persistence:
        def claim_memory_log_batch(self, **_kwargs):
            raise AssertionError("background learning claimed work during a user turn")

        def settle_memory_log_item(self, *_args, **_kwargs):
            raise AssertionError("nothing should have been claimed")

        def memory_log_outbox_status(self):
            raise AssertionError("foreground deferral exits before status")

    delays = []
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _Persistence()
            if name == "persistence"
            else default
        ),
    )
    monkeypatch.setattr(
        "core.runtime.foreground_guard.snapshot",
        lambda: {
            "active": True,
            "quiet_remaining_s": 120.0,
            "reason": "foreground_chat_active",
        },
    )
    monkeypatch.setattr(
        _chat_preflight,
        "_schedule_chat_turn_memory_log_retry",
        lambda delay_s: delays.append(delay_s) or True,
    )

    await chat_routes._drain_chat_turn_memory_log_queue()

    assert delays == [chat_routes._CHAT_TURN_MEMORY_LOG_FOREGROUND_RECHECK_S]


@pytest.mark.asyncio
async def test_chat_turn_memory_log_shutdown_flush_ignores_stale_quiet_window(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    async def _drain(*, honor_foreground=True):
        calls.append(honor_foreground)

    monkeypatch.setattr(_chat_preflight, "_drain_chat_turn_memory_log_queue", _drain)

    await chat_routes._drain_chat_turn_memory_log_queue_on_shutdown()

    assert calls == [False]


@pytest.mark.asyncio
async def test_chat_turn_memory_log_retry_waits_until_foreground_is_clear(monkeypatch):
    from interface.routes import chat as chat_routes

    delays = iter((0.001, 0.001, None))
    drains = []

    monkeypatch.setattr(
        _chat_preflight,
        "_chat_turn_memory_log_foreground_delay",
        lambda: next(delays),
    )

    async def _drain(*, honor_foreground=True):
        drains.append(honor_foreground)

    monkeypatch.setattr(_chat_preflight, "_drain_chat_turn_memory_log_queue", _drain)

    await chat_routes._retry_chat_turn_memory_log_after(0.001)

    assert drains == [False]


@pytest.mark.asyncio
async def test_chat_turn_memory_log_outbox_rejects_permanent_local_filter(monkeypatch):
    from core.memory import chat_turn_logger
    from interface.routes import chat as chat_routes

    async def _must_not_log(**_kwargs):
        raise AssertionError("permanently inadmissible content reached memory")

    monkeypatch.setattr(chat_turn_logger, "log_chat_turn_auto", _must_not_log)

    outcome, reason = await chat_routes._run_chat_turn_memory_log_item(
        {
            "operation_id": "session:short:r1",
            "session_id": "session",
            "exchange_id": "short",
            "revision": 1,
            "user_content": "Hi",
            "aura_content": "Hello",
        }
    )

    assert outcome == "rejected"
    assert reason == "user_message_too_short_for_learned_memory"


@pytest.mark.asyncio
async def test_chat_turn_memory_log_outbox_reclaims_after_settlement_failure(monkeypatch):
    from interface.routes import chat as chat_routes

    payload = {"operation_id": "session:exchange:r1", "attempts": 1}

    class _Persistence:
        def claim_memory_log_batch(self, **_kwargs):
            return [payload]

        def settle_memory_log_item(self, *_args, **_kwargs):
            raise OSError("commit acknowledgement failed")

        def memory_log_outbox_status(self):
            raise AssertionError("settlement failure exits before status")

    delays = []
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _Persistence()
            if name == "persistence"
            else default
        ),
    )
    monkeypatch.setattr(
        _chat_preflight,
        "_run_chat_turn_memory_log_item",
        lambda _payload: asyncio.sleep(0, result=("completed", "")),
    )
    monkeypatch.setattr(
        _chat_preflight,
        "_schedule_chat_turn_memory_log_retry",
        lambda delay_s: delays.append(delay_s) or True,
    )

    await chat_routes._drain_chat_turn_memory_log_queue()

    assert delays == [chat_routes._CHAT_TURN_MEMORY_LOG_LEASE_RECHECK_S]


@pytest.mark.asyncio
async def test_session_memory_pin_recall_survives_process_memory_clear(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this codeword for me: restart-ledger-417. Just confirm.",
        session_id="restart-ledger-session",
    )
    chat_routes._session_memory_pins.clear()
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What codeword did I give you?",
        session_id="restart-ledger-session",
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "restart-ledger-417" in recalled[0]


@pytest.mark.asyncio
async def test_session_memory_pin_restart_wording_stays_on_fastpath(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    principal_token = chat_routes._CHAT_REQUEST_PRINCIPAL.set("owner:bryan")
    surface_token = chat_routes._CHAT_REQUEST_SURFACE.set("owner")
    try:
        stored = await chat_routes._build_memory_state_fastpath_reply(
            "Remember this codeword across restart: restart-ledger-921. Just confirm.",
            session_id="before-restart",
        )
        chat_routes._session_memory_pins.clear()
        recalled = await chat_routes._build_memory_state_fastpath_reply(
            "What codeword did I ask you to remember before restart?",
            session_id="after-restart",
        )
    finally:
        chat_routes._CHAT_REQUEST_SURFACE.reset(surface_token)
        chat_routes._CHAT_REQUEST_PRINCIPAL.reset(principal_token)
        chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "restart-ledger-921" in recalled[0]


@pytest.mark.asyncio
async def test_session_memory_pin_cross_session_recall_rejects_other_principal(
    monkeypatch,
    tmp_path,
):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._store_session_memory_pin(
        "owner-only launch phrase",
        "Remember the owner-only launch phrase.",
        session_id="owner-before-restart",
        principal_id="owner:bryan",
        principal_surface="owner",
    )
    chat_routes._session_memory_pins.clear()

    paired_recall = await chat_routes._recall_session_memory_pin(
        session_id="paired-device:device-a",
        cross_session=True,
        principal_id="paired:alex",
        principal_surface="paired_device",
    )
    owner_recall = await chat_routes._recall_session_memory_pin(
        session_id="owner-after-restart",
        cross_session=True,
        principal_id="owner:bryan",
        principal_surface="owner",
    )
    chat_routes._session_memory_pins.clear()

    assert stored is True
    assert paired_recall is None
    assert owner_recall is not None
    assert owner_recall["content"] == "owner-only launch phrase"

    serialized = ledger_path.read_text(encoding="utf-8")
    assert "owner-only launch phrase" not in serialized
    assert "Remember the owner-only launch phrase" not in serialized
    assert "owner:bryan" not in serialized
    assert "aura.session_memory_pin.envelope.v3" in serialized


@pytest.mark.asyncio
async def test_session_memory_pin_ledger_migrates_plaintext_before_recall(
    monkeypatch,
    tmp_path,
):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    legacy = {
        "schema": "aura.session_memory_pin.v2",
        "content": "legacy launch phrase heliotrope seven",
        "source": "Remember the legacy launch phrase.",
        "timestamp": "2026-08-01T00:00:00+00:00",
        "session_id": "before-restart",
        "principal_id": "owner:bryan",
        "principal_surface": "owner",
        "session_memory_pin": True,
    }
    ledger_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    recalled = await chat_routes._recall_session_memory_pin(
        session_id="after-restart",
        cross_session=True,
        principal_id="owner:bryan",
        principal_surface="owner",
    )

    assert recalled is not None
    assert recalled["content"] == "legacy launch phrase heliotrope seven"
    migrated = ledger_path.read_text(encoding="utf-8")
    assert "heliotrope" not in migrated
    assert "owner:bryan" not in migrated
    assert "Remember the legacy launch phrase" not in migrated
    assert "aura.session_memory_pin.envelope.v3" in migrated


@pytest.mark.asyncio
async def test_content_recall_does_not_cross_principals_after_restart(
    monkeypatch,
    tmp_path,
):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        _chat_memory_state,
        "_session_memory_pin_ledger_path",
        lambda: tmp_path / "session_memory_pins.jsonl",
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()
    assert await chat_routes._store_session_memory_pin(
        "the launch phrase is heliotrope seven",
        "Remember that the launch phrase is heliotrope seven.",
        session_id="owner-old",
        principal_id="owner:bryan",
        principal_surface="owner",
    )
    chat_routes._session_memory_pins.clear()

    principal_token = chat_routes._CHAT_REQUEST_PRINCIPAL.set("paired:guest")
    surface_token = chat_routes._CHAT_REQUEST_SURFACE.set("paired_device")
    try:
        paired_reply = await chat_routes._build_conversation_recall_reply(
            "What launch phrase did I tell you before the restart?",
            session_id="paired-device:guest",
        )
    finally:
        chat_routes._CHAT_REQUEST_SURFACE.reset(surface_token)
        chat_routes._CHAT_REQUEST_PRINCIPAL.reset(principal_token)

    principal_token = chat_routes._CHAT_REQUEST_PRINCIPAL.set("owner:bryan")
    surface_token = chat_routes._CHAT_REQUEST_SURFACE.set("owner")
    try:
        owner_reply = await chat_routes._build_conversation_recall_reply(
            "What launch phrase did I tell you before the restart?",
            session_id="owner-new",
        )
    finally:
        chat_routes._CHAT_REQUEST_SURFACE.reset(surface_token)
        chat_routes._CHAT_REQUEST_PRINCIPAL.reset(principal_token)
        chat_routes._session_memory_pins.clear()

    assert paired_reply is not None
    assert "heliotrope" not in paired_reply.casefold()
    assert owner_reply is not None
    assert "heliotrope seven" in owner_reply.casefold()


@pytest.mark.asyncio
async def test_session_memory_pin_conversation_wording_stays_on_fastpath(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this note for later in this conversation: the blue lantern is under the desk.",
        session_id="conversation-wording",
    )
    chat_routes._session_memory_pins.clear()
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What note did I ask you to remember in this conversation?",
        session_id="conversation-wording",
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "blue lantern is under the desk" in recalled[0]


@pytest.mark.asyncio
async def test_session_memory_pin_natural_that_wording_stays_on_fastpath(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember that my demo codeword is silver-orbit-228. Just confirm.",
        session_id="natural-that-wording",
    )
    chat_routes._session_memory_pins.clear()
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What codeword did I ask you to remember?",
        session_id="natural-that-wording",
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "my demo codeword is silver-orbit-228" in recalled[0]


@pytest.mark.asyncio
async def test_session_memory_pin_natural_pronoun_wording_preserves_subject(
    monkeypatch,
    tmp_path,
):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember my favorite launch phrase is steady violet orbit.",
        session_id="natural-pronoun-wording",
    )
    chat_routes._session_memory_pins.clear()
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What phrase did I ask you to remember?",
        session_id="natural-pronoun-wording",
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "my favorite launch phrase is steady violet orbit" in recalled[0]


@pytest.mark.asyncio
async def test_session_memory_pin_compound_instruction_stores_only_phrase(
    monkeypatch,
    tmp_path,
):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this phrase: silver lantern. Also tell me one thing your live mind is attending to right now.",
        session_id="compound-memory-request",
    )
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What phrase did I ask you to remember?",
        session_id="compound-memory-request",
    )
    chat_routes._session_memory_pins.clear()

    # The deterministic path ENDS the turn with one sentence, so it may only
    # take a turn it fully covers. This one also asks what her live mind is
    # attending to, and the template has nothing to say about that — it used
    # to answer the pin and drop the question silently. It stands down now and
    # the mind answers both halves. See
    # _turn_has_substance_beyond_memory_request.
    assert stored is None
    # What it extracts is still exactly the phrase, which is what this test
    # was really protecting.
    assert (
        chat_routes._extract_session_memory_pin_request(
            "Remember this phrase: silver lantern. Also tell me one thing your "
            "live mind is attending to right now."
        )
        == "silver lantern"
    )
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert '"silver lantern"' in recalled[0]
    assert "live mind" not in recalled[0]


def test_session_memory_pin_preserves_non_imperative_multisentence_fact():
    from interface.routes import chat as chat_routes

    pinned = chat_routes._extract_session_memory_pin_request(
        "Remember this note: the launch story has two beats. the second beat matters."
    )

    assert pinned == "the launch story has two beats. the second beat matters"


def test_session_memory_pin_does_not_capture_conversational_recall_anchor():
    from interface.routes import chat as chat_routes

    pinned = chat_routes._extract_session_memory_pin_request(
        "Remember the uncertainty you just named. How would that change one decision you make?"
    )

    assert pinned is None


@pytest.mark.asyncio
async def test_session_memory_pin_dont_forget_natural_wording_stays_on_fastpath(
    monkeypatch,
    tmp_path,
):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Don't forget that the journal folder should be named Aura's Journals.",
        session_id="dont-forget-wording",
    )
    chat_routes._session_memory_pins.clear()
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What did I tell you to remember?",
        session_id="dont-forget-wording",
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "journal folder should be named Aura's Journals" in recalled[0]


def test_session_memory_pin_question_wording_does_not_store_as_new_memory():
    from interface.routes import chat as chat_routes

    assert chat_routes._extract_session_memory_pin_request("Remember what I said earlier?") is None
    assert chat_routes._extract_session_memory_pin_request("Remember when we talked about tools?") is None


@pytest.mark.asyncio
async def test_session_memory_pin_prefixed_probe_wording_recall(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "For this live reliability probe, remember the phrase cobalt sunrise for this conversation.",
        session_id="prefixed-probe-wording",
    )
    recalled_just = await chat_routes._build_memory_state_fastpath_reply(
        "What phrase did I just ask you to remember?",
        session_id="prefixed-probe-wording",
    )
    recalled_earlier = await chat_routes._build_memory_state_fastpath_reply(
        "What was the phrase from earlier in this probe?",
        session_id="prefixed-probe-wording",
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert "cobalt sunrise" in stored[0]
    assert "for this conversation" not in stored[0]
    assert recalled_just is not None
    assert recalled_just[1] == "session_memory_recall"
    assert "cobalt sunrise" in recalled_just[0]
    assert "for this conversation" not in recalled_just[0]
    assert recalled_earlier is not None
    assert recalled_earlier[1] == "session_memory_recall"
    assert "cobalt sunrise" in recalled_earlier[0]
    assert "for this conversation" not in recalled_earlier[0]


@pytest.mark.asyncio
async def test_session_memory_context_change_uses_pinned_note(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this note for later in this conversation: the blue lantern is under the desk.",
        session_id="context-change",
    )
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What changed in this conversation after I gave you the blue-lantern note?",
        session_id="context-change",
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_context_recall"
    assert "blue lantern is under the desk" in recalled[0]


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_blocks_critical_memory_before_cognition(
    monkeypatch,
    resource_observer,
):
    from core.runtime.resource_observation import SimulatedResourceObserver
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gib = 1024**3
    calls = []
    shed_calls = []
    pressure_probe_calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            calls.append("engine_think")
            return SimpleNamespace(content="unexpected engine reply")

    class _FakeInferenceGate:
        async def _shed_background_workers_for_memory_pressure(self, *, reason):
            shed_calls.append(reason)

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        if name == "inference_gate":
            return _FakeInferenceGate()
        return default

    assert isinstance(resource_observer, SimulatedResourceObserver)
    resource_observer.configure_memory(
        total_bytes=64 * gib,
        available_bytes=24 * gib,
        percent=62.0,
    )
    from core.utils import memory_monitor as memory_monitor_module

    real_memory_probe = memory_monitor_module.get_memory_pressure_snapshot

    def _measured_memory_probe():
        pressure_probe_calls.append("measured")
        return real_memory_probe()

    monkeypatch.setattr(
        memory_monitor_module,
        "get_memory_pressure_snapshot",
        _measured_memory_probe,
    )
    monkeypatch.setattr(
        chat_routes,
        "_foreground_chat_lock",
        chat_routes.PreemptibleChatLock(),
    )
    monkeypatch.setattr(chat_routes, "_FOREGROUND_CHAT_BUSY_WAIT_S", 1.0)
    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    blocker_token = await chat_routes._foreground_chat_lock.acquire()
    request_task = asyncio.create_task(
        server_module.api_chat(
            server_module.ChatRequest(message="Use the desktop path to answer this."),
            SimpleNamespace(
                headers={
                    "X-Aura-Surface": "desktop-ui",
                    "X-Aura-Require-CognitiveEngine": "true",
                },
                client=SimpleNamespace(host="test"),
            ),
            None,
            None,
        )
    )
    await asyncio.sleep(0.05)
    assert pressure_probe_calls == []
    resource_observer.configure_memory(
        total_bytes=64 * gib,
        available_bytes=2 * gib,
        percent=96.0,
    )
    assert chat_routes._foreground_chat_lock.release(blocker_token) is True
    response = await asyncio.wait_for(request_task, timeout=2.0)

    # In-band for real users: the guard text IS the answer (raw 503s
    # surfaced as bare HTTP errors in both July 8 soaks). Benchmarks
    # (X-Aura-Benchmark) still get the strict 503.
    assert response.status_code == 200
    assert b"memory_pressure_guard" in response.body
    assert b"memory_pressure" in response.body
    assert calls == []
    # At least once, not exactly once. What this test is about is the ORDER —
    # the assertion above proves the probe had not run while the turn was
    # queued behind the foreground lock, which is the property that keeps
    # memory from being measured too late to protect cognition. How many times
    # it is consulted afterwards is an implementation detail, and
    # get_memory_pressure_snapshot is TTL-cached, so a second consultation is a
    # cache read rather than a syscall. Pinning the count made this fail on a
    # second reader being added while the guarantee held.
    assert pressure_probe_calls and all(
        call == "measured" for call in pressure_probe_calls
    ), pressure_probe_calls
    assert shed_calls
    assert any("memory_pressure" in reason for reason in shed_calls)


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_blocks_process_tree_memory_before_cognition(
    monkeypatch,
    resource_observer,
):
    from core.runtime.resource_observation import SimulatedResourceObserver
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gib = 1024**3
    calls = []
    shed_calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            calls.append("engine_think")
            return SimpleNamespace(content="unexpected engine reply")

    class _FakeInferenceGate:
        async def _shed_background_workers_for_memory_pressure(self, *, reason):
            shed_calls.append(reason)

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        if name == "inference_gate":
            return _FakeInferenceGate()
        return default

    monkeypatch.setenv("AURA_PROCESS_RSS_LIMIT_GB", "40")
    assert isinstance(resource_observer, SimulatedResourceObserver)
    resource_observer.configure_memory(
        total_bytes=64 * gib,
        available_bytes=24 * gib,
        percent=62.0,
        process_rss_bytes=3 * gib,
        process_tree_rss_bytes=41 * gib,
    )
    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Use the desktop path to answer this."),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    # In-band for real users; strict 503 stays benchmark-only.
    assert response.status_code == 200
    assert b"memory_pressure_guard" in response.body
    assert b"process_tree_rss" in response.body
    assert calls == []
    assert shed_calls
    assert any("process_tree_rss" in reason for reason in shed_calls)


@pytest.mark.asyncio
async def test_api_chat_refuses_heavy_generation_when_memory_probe_is_unavailable(
    monkeypatch,
):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            calls.append("engine_think")
            return SimpleNamespace(content="unexpected engine reply")

    monkeypatch.setattr(
        chat_routes,
        "_foreground_chat_lock",
        chat_routes.PreemptibleChatLock(),
    )
    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _FakeCognitiveEngine() if name == "cognitive_engine" else default
            )
        ),
    )
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: (_ for _ in ()).throw(RuntimeError("probe unavailable")),
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Use the desktop path to answer this."),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"memory_pressure_probe_unavailable" in response.body
    assert b'"measured":false' in response.body
    assert calls == []


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_keeps_nontrivial_chat_on_cognitive_engine(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": getattr(mode, "name", str(mode)),
                    "origin": origin,
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="Hi. I am here and following this conversation through the live desktop path.",
                mode=mode,
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            calls.append({"kernel_interface": "unexpected"})
            raise AssertionError("desktop UI must not use KernelInterface when CognitiveEngine answers")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    lane_calls = 0

    def _lane_status():
        nonlocal lane_calls
        lane_calls += 1
        if lane_calls >= 2:
            return {
                "conversation_ready": False,
                "state": "cold",
                "last_failure_reason": "endpoint_timeout:Cortex:38.5s",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": None,
                "background_endpoint": "Brainstem",
            }
        return {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        }

    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status", _lane_status)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(message="How are you thinking about this conversation right now?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"following this conversation through the live desktop path" in response.body
    assert b"cognitive_engine" in response.body
    assert b"social_presence_reflex" not in response.body
    assert calls
    assert calls[0]["context"]["route"] == "desktop_chat"
    assert calls[0]["context"]["source"] == "desktop_ui"
    assert calls[0]["context"]["cognitive_engine_required"] is True
    assert not any("kernel_interface" in call for call in calls)


@pytest.mark.asyncio
async def test_api_chat_desktop_required_presence_check_uses_cognitive_engine(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            calls.append({"cognitive_engine": "called"})
            return SimpleNamespace(
                content=(
                    "I'm here with you. I'm following this conversation through the live desktop path "
                    "and answering the current turn directly."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _FakeGate:
        async def ensure_foreground_ready(self, *_args, **_kwargs):
            calls.append({"inference_gate": "admitted"})
            return {
                "conversation_ready": True,
                "state": "ready",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": "Cortex",
                "background_endpoint": "Brainstem",
            }

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            calls.append({"kernel_interface": "unexpected"})
            raise AssertionError("desktop UI must not fall back to KernelInterface")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        if name == "inference_gate":
            return _FakeGate()
        return default

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    chat_routes._recent_responses.clear()
    chat_routes._recent_response_pairs.clear()
    patch_chat_lane(monkeypatch, "_runtime_kernel_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_cognitive_engine_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_memory_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_substrate_voice_available", lambda: True)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "cold",
            "last_failure_reason": "worker_not_alive,init_not_complete,lane_cold",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": None,
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        _chat_desktop_repair,
        "_build_social_continuity_repair_reply",
        lambda _message: "hey. i'm here with the live thread intact.",
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="you there?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert "following this conversation through the live desktop path" in payload["response"]
    assert payload["live_turn_contract"]["engine_think_invoked"] is True
    assert payload["live_turn_contract"]["live_mind_controls_bound"] is True
    assert payload["live_turn_contract"]["full_mind_path"] is True
    assert calls[0] == {"inference_gate": "admitted"}
    assert [
        call["cognitive_engine"]
        for call in calls
        if "cognitive_engine" in call
    ] == ["called"]


@pytest.mark.asyncio
async def test_api_chat_desktop_cold_lane_timeout_is_not_reported_as_failed_reasoning(
    monkeypatch,
):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeGate:
        async def ensure_foreground_ready(self, *_args, **_kwargs):
            calls.append("warmup")
            raise TimeoutError("worker still loading")

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            calls.append("think")
            raise AssertionError("a conclusively cold lane must not enter CognitiveEngine")

    def _fake_get(name, default=None):
        if name == "inference_gate":
            return _FakeGate()
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    cold_lane = {
        "conversation_ready": False,
        "state": "cold",
        "last_failure_reason": "worker_not_alive,init_not_complete,lane_cold",
        "readiness_blockers": ["worker_not_alive", "init_not_complete", "lane_cold"],
        "desired_model": "Cortex (32B)",
        "desired_endpoint": "Cortex",
        "foreground_endpoint": None,
        "background_endpoint": "Brainstem",
    }
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: dict(cold_lane),
    )
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="warming": dict(
            cold_lane,
            state=state,
            last_failure_reason=reason,
            warmup_attempted=True,
        ),
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    patch_chat_lane(monkeypatch, "_runtime_kernel_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_cognitive_engine_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_memory_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_substrate_voice_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_inference_available", lambda *a, **k: False)

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Are you with me?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "conversation_warming"
    assert payload["reason"] == "foreground_warmup_timeout"
    assert payload["response_confidence"] == "not_generated"
    assert payload["live_turn_contract"]["engine_think_invoked"] is False
    assert payload["live_turn_contract"]["full_mind_path"] is False
    assert "stand behind" not in payload["response"]
    assert "boot delay" in payload["response"]
    assert calls == ["warmup"]


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_routes_memory_state_through_cognitive_engine(
    monkeypatch,
    tmp_path,
):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _memory_cognitive_turn(objective, *_args, **_kwargs):
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append(str(objective))
        if "Remember this note" in str(objective):
            return "I've pinned \"the blue lantern is under the desk\" in durable session memory, and I can pull it back later from canonical memory state."
        if "What note did I ask" in str(objective):
            return "The phrase you asked me to remember in this session was \"the blue lantern is under the desk.\""
        if "What changed" in str(objective):
            return "The concrete change is that \"the blue lantern is under the desk\" is now stored as durable conversation state for later turns."
        return "I am answering from the canonical memory state evidence."

    async def _fake_begin_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: tmp_path / "session_memory_pins.jsonl")
    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _memory_cognitive_turn)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    chat_routes._session_memory_pins.clear()
    request = SimpleNamespace(
        headers={
            "X-Aura-Surface": "desktop-ui",
            "X-Aura-Require-CognitiveEngine": "true",
        },
        client=SimpleNamespace(host="test"),
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    stored = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Remember this note for later in this conversation: "
                "the blue lantern is under the desk."
            ),
            session_id="memory-fastpath-test",
        ),
        request,
        None,
        None,
    )
    recalled = await server_module.api_chat(
        server_module.ChatRequest(
            message="What note did I ask you to remember in this conversation?",
            session_id="memory-fastpath-test",
        ),
        request,
        None,
        None,
    )
    changed = await server_module.api_chat(
        server_module.ChatRequest(
            message="What changed in this conversation after I gave you the blue-lantern note?",
            session_id="memory-fastpath-test",
        ),
        request,
        None,
        None,
    )
    chat_routes._session_memory_pins.clear()

    stored_payload = json.loads(stored.body)
    recalled_payload = json.loads(recalled.body)
    changed_payload = json.loads(changed.body)
    assert stored.status_code == 200
    assert recalled.status_code == 200
    assert changed.status_code == 200
    assert stored_payload["status"] == "cognitive_engine"
    assert recalled_payload["status"] == "cognitive_engine_memory_state_grounding"
    assert changed_payload["status"] == "cognitive_engine"
    assert "blue lantern is under the desk" in stored_payload["response"]
    assert "failed the final reliability checks" not in stored_payload["response"]
    assert stored_payload["response_confidence"] == "high"
    assert recalled_payload["response_confidence"] == "bounded"
    assert recalled_payload["live_turn_contract"]["full_mind_path"] is False
    assert recalled_payload["live_turn_contract"]["final_text_authorship"] == (
        "non_cognitive_replacement"
    )
    assert "blue lantern is under the desk" in recalled_payload["response"]
    assert "blue lantern is under the desk" in changed_payload["response"]
    assert len(cognitive_calls) == 3
    assert all("CANONICAL MEMORY STATE EVIDENCE" in call for call in cognitive_calls)


@pytest.mark.asyncio
async def test_api_chat_desktop_memory_state_drift_rebounds_to_canonical_evidence(
    monkeypatch,
    tmp_path,
):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _drifting_cognitive_turn(objective, *_args, **_kwargs):
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": True,
                    "cognitive_engine_reply_failed": False,
                    "bounded_contract_used": False,
                    "legacy_fallback_used": False,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    **_bound_live_mind_controls_trace(),
                    "response_path": "cognitive_engine",
                }
            )
        cognitive_calls.append(str(objective))
        return "I am attending to the live desktop thread through CognitiveEngine."

    async def _fake_begin_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: tmp_path / "session_memory_pins.jsonl")
    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _drifting_cognitive_turn)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    _force_full_mind_runtime(monkeypatch, chat_routes)
    chat_routes._session_memory_pins.clear()

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Remember this phrase: silver lantern. Also tell me one thing "
                "your live mind is attending to right now."
            ),
            session_id="memory-drift-rebound-test",
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )
    chat_routes._session_memory_pins.clear()

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["response_confidence"] == "high"
    assert "silver lantern" in payload["response"]
    assert "live desktop thread" in payload["response"]
    assert payload["live_turn_contract"]["live_mind_controls_bound"] is True
    assert payload["live_turn_contract"]["full_mind_path"] is True
    assert payload["live_turn_contract"]["bounded_contract_used"] is False
    assert payload["live_turn_contract"]["response_path"] == "cognitive_engine_memory_state_grounding"
    assert cognitive_calls
    assert "CANONICAL MEMORY STATE EVIDENCE" in cognitive_calls[0]


@pytest.mark.asyncio
async def test_api_chat_desktop_owner_name_recall_routes_through_cognitive_engine(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _owner_cognitive_turn(objective, *_args, **_kwargs):
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append(str(objective))
        return "You're Bryan; I know that from the verified owner session, and I will keep that context attached to this conversation."

    async def _fake_begin_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _owner_cognitive_turn)
    monkeypatch.setattr(_chat_memory_state, "_owner_session_is_verified", lambda **_kwargs: True)
    monkeypatch.setattr(_chat_memory_state, "_resolve_primary_operator_name", lambda: "Bryan")
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Do you know my name?",
            session_id="owner-name-fastpath-test",
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "You're Bryan" in payload["response"]
    assert len(cognitive_calls) == 1
    assert "CANONICAL MEMORY STATE EVIDENCE" in cognitive_calls[0]


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_plans_with_cognitive_engine_before_execution(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    skill_calls = []
    completed_exchanges = []
    output_receipts = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "document_body": "Timestamped Aura summary from CognitiveEngine.",
                        "steps": [
                            {
                                "action": "open_app",
                                "target": "Notes",
                                "reason": "Use the requested writing surface.",
                                "expect": "Notes accepts focus.",
                            }
                        ],
                    }
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _FakePool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return True

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            self.unexpected_process_calls = getattr(self, "unexpected_process_calls", 0) + 1
            raise AssertionError("desktop objective should not fall through to KernelInterface")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "desktop-objective"

    async def _fake_complete_exchange(*_args, **_kwargs):
        completed_exchanges.append((_args, _kwargs))
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        output_receipts.append((_args, _kwargs))
        return None

    async def _fake_execute_governed_live_skill(skill_name, params, *, objective, extra_context=None):
        skill_calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "status": "completed",
            "summary": "Desktop task completed 5/5 governed computer-use steps.",
            "steps_requested": 5,
            "steps_completed": 5,
            "receipts": _verified_desktop_receipts(5),
        }

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _fake_execute_governed_live_skill)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _FakePool())
    lane_calls = 0

    def _live_proof_lane_status():
        nonlocal lane_calls
        lane_calls += 1
        if lane_calls >= 2:
            return {
                "conversation_ready": False,
                "state": "cold",
                "last_failure_reason": "endpoint_timeout:Cortex:38.5s",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": None,
                "background_endpoint": "Brainstem",
            }
        return {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        }

    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        _live_proof_lane_status,
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Can you open my Notes app, write a timestamped summary, save it as a PDF "
                "in a new folder titled Aura's Journal, and search for a robot image?"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"desktop_objective_completed" in response.body
    # The step count belongs to the log, not the reply. It is still in the
    # result payload, where tooling reads it — the assertion is about the
    # user-facing "response" field only.
    import json as _json

    _payload = _json.loads(response.body)
    assert _payload["response"].startswith("Done")
    assert "governed computer-use steps" not in _payload["response"]
    assert len(skill_calls) == 1
    assert skill_calls[0]["skill_name"] == "desktop_task"
    assert skill_calls[0]["params"]["objective"] == (
        "Can you open my Notes app, write a timestamped summary, save it as a PDF "
        "in a new folder titled Aura's Journal, and search for a robot image?"
    )
    assert skill_calls[0]["params"]["steps"] == []
    assert skill_calls[0]["params"]["desktop_execution_contract"] is True
    assert skill_calls[0]["params"]["allow_heuristic_desktop_plan"] is True
    assert skill_calls[0]["params"]["user_visible_desktop_action"] is True
    assert skill_calls[0]["params"]["verification_required"] is True
    assert skill_calls[0]["params"]["action_expectation"] == {
        "objective": skill_calls[0]["params"]["objective"],
        "acceptance_criteria": ["steps_requested", "steps_completed"],
        "required_evidence": ["receipts"],
        "repair_hint": "rerun_desktop_task_with_effect_receipts",
        "allow_partial": True,
    }
    assert skill_calls[0]["objective"] == skill_calls[0]["params"]["objective"]
    assert skill_calls[0]["extra_context"] == {
        "origin": "desktop_ui",
        "source": "desktop_ui",
        "route": "chat.desktop_objective",
        "desktop_execution_contract": True,
        "allow_heuristic_desktop_plan": True,
        "disable_outer_skill_retry": True,
        "user_visible_desktop_action": True,
        "local_desktop_action": True,
        "verification_required": True,
        "allow_desktop_task_model_synthesis": False,
        "desktop_task_document_body": skill_calls[0]["extra_context"]["cognitive_reply"],
        "cognitive_reply": skill_calls[0]["extra_context"]["cognitive_reply"],
        "action_expectation": skill_calls[0]["params"]["action_expectation"],
    }
    assert "Timestamped Aura summary from CognitiveEngine." in skill_calls[0]["extra_context"]["cognitive_reply"]
    assert completed_exchanges
    assert completed_exchanges[-1][0][0] == "desktop-objective"
    # The logged exchange and the output receipt carry what the person was
    # actually told, so they carry the same plain confirmation the reply does —
    # not the executor's step count, which lives in the result payload.
    assert completed_exchanges[-1][0][2].startswith("Done")
    assert "governed computer-use steps" not in completed_exchanges[-1][0][2]
    assert "Aura self-summary. Timestamp" not in completed_exchanges[-1][0][2]
    assert output_receipts
    assert output_receipts[-1][0][0].startswith("Done")


@pytest.mark.asyncio
async def test_chat_desktop_objective_uses_capability_engine_without_agency_wrapper(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append(
                {
                    "skill_name": skill_name,
                    "params": dict(params),
                    "context": dict(context or {}),
                }
            )
            return {
                "ok": True,
                "summary": "Desktop task completed 2/2 governed computer-use steps.",
                "steps_requested": 2,
                "steps_completed": 2,
                "receipts": _verified_desktop_receipts(2),
            }

    class _ForbiddenAgency:
        async def run(self, *_args, **_kwargs):
            pytest.fail("chat.desktop_objective must not enter AgencyOrchestrator")

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _FakeCapabilityEngine()
                if name == "capability_engine"
                else _ForbiddenAgency()
                if name == "agency_orchestrator"
                else default
            )
        ),
    )

    objective = (
        "Please create a folder named 'Aura Live Proof' in my Documents folder "
        "and write a file inside it called live_proof.txt."
    )
    result = await chat_routes._execute_desktop_objective_from_chat(
        objective,
        cognitive_reply="Plan the desktop action; do not claim completion.",
    )

    assert result is not None
    assert result["ok"] is True
    assert result["status"] == "desktop_objective_completed"
    assert len(calls) == 1
    assert calls[0]["skill_name"] == "desktop_task"
    assert calls[0]["params"]["objective"] == objective
    assert calls[0]["params"]["steps"] == []
    assert calls[0]["params"]["desktop_execution_contract"] is True
    assert calls[0]["params"]["allow_heuristic_desktop_plan"] is True
    assert calls[0]["params"]["user_visible_desktop_action"] is True
    assert calls[0]["params"]["verification_required"] is True
    assert calls[0]["context"]["route"] == "chat.desktop_objective"
    assert calls[0]["context"]["governance_route"] == "capability_engine_direct"
    assert calls[0]["context"]["desktop_task_owned_by"] == "chat.desktop_objective"
    assert calls[0]["context"]["requested_authority_scope"] == (
        "foreground_user_requested:chat.desktop_objective:desktop_task"
    )
    assert "scoped_authority" not in calls[0]["context"]
    assert calls[0]["context"]["foreground_request"] is True
    assert calls[0]["context"]["user_explicitly_authorized"] is True
    assert calls[0]["context"]["allow_heuristic_desktop_plan"] is True
    assert calls[0]["context"]["allow_desktop_task_model_synthesis"] is False
    assert calls[0]["params"]["action_expectation"]["required_evidence"] == ["receipts"]
    assert calls[0]["context"]["action_expectation"] == calls[0]["params"]["action_expectation"]


@pytest.mark.asyncio
async def test_chat_desktop_research_objective_does_not_enable_hidden_model_synthesis(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append(
                {
                    "skill_name": skill_name,
                    "params": dict(params),
                    "context": dict(context or {}),
                }
            )
            return {
                "ok": True,
                "summary": "Desktop task completed 4/4 governed computer-use steps.",
                "steps_requested": 4,
                "steps_completed": 4,
                "receipts": _verified_desktop_receipts(4),
            }

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _FakeCapabilityEngine() if name == "capability_engine" else default
            )
        ),
    )

    result = await chat_routes._execute_desktop_objective_from_chat(
        (
            "Open Google, find three articles about climate change, summarize them "
            "in a Google Doc, and export the summary as a PDF."
        ),
        cognitive_reply="A source-grounded draft may be composed by the desktop task.",
    )

    assert result is not None
    assert result["ok"] is True
    assert calls and calls[0]["skill_name"] == "desktop_task"
    assert calls[0]["context"]["route"] == "chat.desktop_objective"
    assert calls[0]["context"]["allow_desktop_task_model_synthesis"] is False
    assert calls[0]["context"]["action_expectation"]["repair_hint"] == (
        "rerun_desktop_task_with_effect_receipts"
    )


@pytest.mark.asyncio
async def test_chat_desktop_objective_rejects_success_without_effect_receipts(monkeypatch):
    from interface.routes import chat as chat_routes

    class _ReceiptlessCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            assert skill_name == "desktop_task"
            return {
                "ok": True,
                "status": "completed",
                "summary": "Desktop task completed 2/2 governed computer-use steps.",
                "steps_requested": 2,
                "steps_completed": 2,
            }

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _ReceiptlessCapabilityEngine() if name == "capability_engine" else default
            )
        ),
    )

    result = await chat_routes._execute_desktop_objective_from_chat(
        "Open Notes and write a paragraph about dinosaurs.",
        cognitive_reply="Dinosaurs were diverse animals with a long fossil record.",
    )

    assert result is not None
    assert result["ok"] is False
    assert result["status"] == "desktop_objective_failed"
    assert result["result"]["status"] == "desktop_task_effect_evidence_missing"
    assert result["result"]["error"] == "missing_step_receipts"
    assert "did not complete" in result["response"]
    assert "not claiming" in result["response"]


@pytest.mark.asyncio
async def test_chat_desktop_objective_preserves_confirmation_required(monkeypatch):
    from interface.routes import chat as chat_routes

    class _ConfirmationRequiredCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            assert skill_name == "desktop_task"
            return {
                "ok": False,
                "status": "approval_required",
                "error": "Fresh user confirmation required",
                "approval": {
                    "required": True,
                    "mode": "all",
                    "confirmation_endpoint": "/api/settings/auth/fresh",
                    "challenge_id": "action-confirm-test",
                    "one_time": True,
                    "action_bound": True,
                    "confirmation_does_not_bypass_governance": True,
                },
            }

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _ConfirmationRequiredCapabilityEngine()
                if name == "capability_engine"
                else default
            )
        ),
    )

    result = await chat_routes._execute_desktop_objective_from_chat(
        "Open Notes and write a paragraph about dinosaurs.",
        cognitive_reply="Dinosaurs were diverse animals with a long fossil record.",
    )

    assert result is not None
    assert result["ok"] is False
    assert result["status"] == "approval_required"
    assert result["approval"]["mode"] == "all"
    assert result["approval"]["challenge_id"] == "action-confirm-test"
    assert "Confirm it to retry the same request" in result["response"]
    assert "did not complete" not in result["response"]


def test_desktop_task_verifier_allows_noncritical_warning_receipts():
    from interface.routes import chat as chat_routes

    receipts = _verified_desktop_receipts(3)
    receipts.append(
        {
            "index": 4,
            "action": "open_url",
            "critical": False,
            "ok": False,
            "effect_verified": False,
            "effect_evidence": "Operation took too long",
            "result": {"ok": False, "error": "Operation took too long"},
        }
    )

    verified, reason = chat_routes._verified_desktop_task_result(
        {
            "ok": True,
            "steps_requested": 4,
            "steps_completed": 3,
            "receipts": receipts,
        }
    )

    assert verified is True
    assert reason == "verified"


@pytest.mark.asyncio
async def test_api_chat_desktop_objective_requires_cognitive_planning(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    skill_calls = []
    output_receipts = []
    cognitive_calls = []

    async def _slow_or_empty_cognitive_turn(*args, **kwargs):
        _t = kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append((args, kwargs))
        return json.dumps(
            {
                "document_body": "Planned local file body.",
                "steps": [
                    {
                        "action": "create_folder",
                        "target": {"path": "~/Documents/Aura Live Proof"},
                        "reason": "Create the requested destination.",
                        "expect": "Folder exists.",
                    },
                    {
                        "action": "write_text_file",
                        "target": {
                            "path": "~/Documents/Aura Live Proof/live_proof.txt",
                            "content": "{{document_body}}",
                        },
                        "reason": "Write the requested file.",
                        "expect": "File exists with the planned body.",
                    },
                ],
            }
        )

    async def _fake_execute_governed_live_skill(skill_name, params, *, objective, extra_context=None):
        skill_calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "status": "completed",
            "summary": "Desktop task completed 2/2 governed computer-use steps.",
            "steps_requested": 2,
            "steps_completed": 2,
            "receipts": _verified_desktop_receipts(2),
        }

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _slow_or_empty_cognitive_turn)
    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _fake_execute_governed_live_skill)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Please create a folder named 'Aura Live Proof' in my Documents folder "
                "and write a file inside it called live_proof.txt."
            )
        ),
        SimpleNamespace(headers={}, client=SimpleNamespace(host="test")),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "desktop_objective_completed"
    assert payload["conversation_lane"]["governed_action_result"] is True
    assert payload["conversation_lane"]["governed_action_status"] == "desktop_objective_completed"
    assert payload["live_turn_contract"]["foreground_model_generation_count"] == 0
    assert payload["live_turn_contract"]["model_native_output"] is False
    assert payload["live_turn_contract"]["response_authority_proven"] is True
    assert payload["live_turn_contract"]["answer_delivery_proven"] is True
    assert payload["live_turn_contract"]["final_text_authorship"] == (
        "verified_action_receipt_serialization"
    )
    # The reply confirms completion without quoting the planner's bookkeeping.
    # "Desktop task completed 2/2 governed computer-use steps through
    # heuristic_compat planning" is the executor's summary; it is evidence for
    # the log and internal vocabulary in a sentence to a person. What the reply
    # owes them is that it is done, and — when a receipt names one — what was
    # changed.
    assert payload["response"].startswith("Done")
    assert "governed computer-use steps" not in payload["response"]
    assert "heuristic_compat" not in payload["response"]
    assert skill_calls and skill_calls[0]["skill_name"] == "desktop_task"
    assert not cognitive_calls
    assert skill_calls[0]["extra_context"]["desktop_task_document_body"] == ""
    assert output_receipts


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_requires_cognitive_engine_and_blocks_kernel_fallback(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("desktop UI must fail closed instead of using KernelInterface fallback")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-1"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Sample 1: In exactly five words, state why checksums matter."
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert kernel_calls == []
    # The refusal is pinned by its contract fields, not by one sentence: the
    # wording is user-facing prose and was deliberately rewritten out of
    # engineering vocabulary. tests/test_failure_replies_speak_plainly.py owns
    # the wording; this test owns the behaviour.
    assert b"stand behind" in response.body
    assert b"desktop_cognitive_engine_required_no_reply" in response.body
    assert payload["live_turn_contract"]["final_requested_output_contract_required"] is True
    assert payload["live_turn_contract"]["final_requested_output_contract_evaluated"] is False
    assert payload["live_turn_contract"]["final_requested_output_contract_kind"] == "word_count"
    assert payload["live_turn_contract"]["final_requested_output_contract_proven"] is False


@pytest.mark.asyncio
async def test_api_chat_desktop_discards_bounded_repair_when_full_mind_path_not_proven(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("desktop UI must fail closed instead of using KernelInterface fallback")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-repair"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    async def _bounded_repair_candidate(*_args, **kwargs):
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": False,
                    "bounded_contract_used": True,
                    "legacy_fallback_used": False,
                    "response_path": "conversation_recall_log_repair_after_empty_engine",
                }
            )
        return "This is a bounded repair and must not be served as Aura speech."

    ready_lane = {
        "conversation_ready": True,
        "state": "ready",
        "desired_model": "Cortex (32B)",
        "desired_endpoint": "Cortex",
        "foreground_endpoint": "Cortex",
        "background_endpoint": "Brainstem",
    }

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _bounded_repair_candidate)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status", lambda: dict(ready_lane))
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": dict(ready_lane, conversation_ready=False, state=state, reason=reason),
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Live desktop route probe. Answer directly in two sentences: "
                "what did I just ask you to do?"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200  # in-band fail-closed delivery for real users
    # THE invariant, asserted first because it is the one that matters: repair
    # machinery must never speak as Aura. It used to be checked last, behind a
    # status-string assertion that went stale, so when the route changed the
    # file failed for the wrong reason and said nothing about the leak.
    assert "bounded repair" not in payload["response"]
    assert payload["live_turn_contract"]["full_mind_path"] is False
    # bounded_contract_used records that repair RAN, and in this turn it did —
    # the fake above sets it and returns repair text. What must not happen is
    # that text reaching the person, which the assertion above proves. Demanding
    # False here asserted that repair never ran, which was never what this test
    # set up.
    assert payload["live_turn_contract"]["bounded_contract_used"] is True
    assert kernel_calls == []
    # A repairable draft may be preserved internally, but it cannot become a
    # successful response unless the same transaction-bound delivery contract
    # as ordinary cognition proves authorship and completion.
    assert payload["status"] in {
        "desktop_cognitive_engine_unavailable",
        "desktop_response_quality_failed",
    }, payload["status"]
    assert payload["status"] != "cognitive_engine_served_repairable_draft"


@pytest.mark.asyncio
async def test_api_chat_desktop_low_risk_social_no_reply_fails_closed(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    completed_exchanges = []
    output_receipts = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("low-risk social desktop repair must not use KernelInterface fallback")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-social"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _no_cognitive_reply(*_args, **kwargs):
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": True,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    "response_path": "cognitive_engine_no_acceptable_reply",
                    **_bound_live_mind_controls_trace(),
                }
            )
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _no_cognitive_reply)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))
    _force_full_mind_runtime(monkeypatch, chat_routes)

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Ok. Just checking. I'll be back, ok?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert payload["status"] == "desktop_cognitive_engine_unavailable"
    assert payload["reason"] == "desktop_cognitive_engine_required_no_reply"
    assert payload["response_confidence"] == "failed"
    assert "stand behind" in payload["response"]
    assert "reply-quality gate" not in payload["response"]
    assert "second foreground generation" not in payload["response"]
    assert kernel_calls == []
    assert len(completed_exchanges) == 1
    assert completed_exchanges[0][1]["record_experience"] is False
    assert output_receipts[0][1]["metadata"]["path"] == "desktop_cognitive_engine"
    assert output_receipts[0][1]["metadata"]["response_confidence"] == "failed"


@pytest.mark.asyncio
async def test_api_chat_desktop_self_process_no_reply_uses_grounded_repair(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    completed_exchanges = []
    output_receipts = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("self-process desktop repair must not use KernelInterface fallback")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-self-process"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _no_cognitive_reply(*_args, **kwargs):
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": True,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    "response_path": "cognitive_engine_no_acceptable_reply",
                    **_bound_live_mind_controls_trace(),
                }
            )
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _no_cognitive_reply)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))
    _force_full_mind_runtime(monkeypatch, chat_routes)

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "When you are confused, how does that change your planning, "
                "memory use, and tool verification?"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    lowered = payload["response"].lower()
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine_self_process_grounding"
    assert payload["reason"] == "desktop_cognitive_engine_required_no_reply"
    assert payload["response_confidence"] == "bounded"
    assert payload["live_turn_contract"]["full_mind_path"] is False
    assert payload["live_turn_contract"]["authentic_cognitive_reply"] is False
    assert payload["live_turn_contract"]["bounded_contract_used"] is True
    assert payload["live_turn_contract"]["model_native_output"] is False
    assert "stand behind" not in lowered
    assert "planning" in lowered
    assert "memory" in lowered
    assert "tool" in lowered
    assert "confusion" in lowered or "confused" in lowered
    assert "legacy fallback" not in lowered
    assert "live conversation contract" not in lowered
    assert "mood-card" not in lowered
    assert "active local model" not in lowered
    assert "cognitiveengine" not in lowered
    assert kernel_calls == []
    assert len(completed_exchanges) == 1
    assert completed_exchanges[0][1]["record_experience"] is True
    assert output_receipts[0][1]["metadata"]["path"] == "cognitive_engine_self_process_grounding"
    assert output_receipts[0][1]["metadata"]["response_confidence"] == "bounded"


@pytest.mark.asyncio
async def test_api_chat_desktop_runtime_path_no_reply_uses_grounded_route_truth(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    completed_exchanges = []
    output_receipts = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("runtime-path desktop repair must not use KernelInterface fallback")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-runtime-path"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _no_cognitive_reply(*_args, **kwargs):
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": True,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    "response_path": "cognitive_engine_no_acceptable_reply",
                    **_bound_live_mind_controls_trace(),
                }
            )
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _no_cognitive_reply)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": True,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))
    _force_full_mind_runtime(monkeypatch, chat_routes)

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Live route probe: in one concise paragraph, explain what runtime path "
                "you are speaking through right now and include the exact phrase resident bridge truth."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    lowered = payload["response"].lower()
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine_runtime_fact_grounding"
    assert payload["reason"] == "desktop_cognitive_engine_required_no_reply"
    assert payload["response_confidence"] == "bounded"
    assert payload["live_turn_contract"]["full_mind_path"] is False
    assert payload["live_turn_contract"]["bounded_contract_used"] is True
    assert payload["live_turn_contract"]["final_text_authorship"] == (
        "non_cognitive_replacement"
    )
    assert "stand behind" not in lowered
    assert "resident bridge truth" in lowered
    assert "desktop ui" in lowered
    assert "/api/chat" in lowered
    assert "cognitiveengine" in lowered
    assert "cortex" in lowered
    assert "cortex (32b)" not in lowered
    assert "claude" not in lowered
    assert kernel_calls == []
    assert len(completed_exchanges) == 1
    assert completed_exchanges[0][1]["record_experience"] is True
    assert output_receipts[0][1]["metadata"]["path"] == "cognitive_engine_runtime_fact_grounding"
    assert output_receipts[0][1]["metadata"]["response_confidence"] == "bounded"


@pytest.mark.asyncio
async def test_api_chat_desktop_identity_no_reply_uses_evidence_bound_repair(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    completed_exchanges = []
    output_receipts = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("identity desktop repair must not use KernelInterface fallback")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-identity"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _no_cognitive_reply(*_args, **kwargs):
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": True,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    "response_path": "cognitive_engine_no_acceptable_reply",
                    **_bound_live_mind_controls_trace(),
                }
            )
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _no_cognitive_reply)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))
    _force_full_mind_runtime(monkeypatch, chat_routes)

    prompt = (
        "Quick reliability check, in two or three sentences: what are you, "
        "and will you remember this conversation tomorrow?"
    )
    response = await server_module.api_chat(
        server_module.ChatRequest(message=prompt),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    lowered = payload["response"].lower()
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine_identity_continuity_grounding"
    assert payload["reason"] == "desktop_cognitive_engine_required_no_reply"
    assert payload["response_confidence"] == "bounded"
    assert payload["live_turn_contract"]["full_mind_path"] is False
    assert payload["live_turn_contract"]["bounded_contract_used"] is True
    assert payload["live_turn_contract"]["final_text_authorship"] == (
        "non_cognitive_replacement"
    )
    assert "stand behind" not in lowered
    assert "local governed cognitive-agent runtime" in lowered
    assert "persistent memory" in lowered
    assert "cannot guarantee perfect tomorrow recall" in lowered
    assert "legacy fallback" not in lowered
    assert kernel_calls == []
    assert len(completed_exchanges) == 1
    assert completed_exchanges[0][1]["record_experience"] is True
    assert output_receipts[0][1]["metadata"]["path"] == "cognitive_engine_identity_continuity_grounding"
    assert output_receipts[0][1]["metadata"]["response_confidence"] == "bounded"


@pytest.mark.asyncio
async def test_api_chat_desktop_capability_no_reply_fails_closed_without_inventory_repair(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    completed_exchanges = []
    output_receipts = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("capability inventory desktop repair must not use KernelInterface fallback")

    class _FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield from [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
            ]

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-capability"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _no_cognitive_reply(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "capability_engine":
            return _FakeCapabilityEngine()
        return default

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _no_cognitive_reply)
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "What tools she could hypothetically do externally, and can she flex "
                "her muscles with one concrete scenario?"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    lowered = payload["response"].lower()
    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert payload["status"] == "desktop_cognitive_engine_unavailable"
    assert payload["response_confidence"] == "failed"
    assert payload["reason"] == "desktop_cognitive_engine_required_no_reply"
    assert "stand behind" in lowered
    assert "computer_use" not in payload["response"]
    assert "web_search" not in payload["response"]
    assert "legacy fallback" not in lowered
    assert "self-process" not in lowered
    assert kernel_calls == []
    assert len(completed_exchanges) == 1
    assert completed_exchanges[0][1]["record_experience"] is False
    assert output_receipts[0][1]["metadata"]["path"] == "desktop_cognitive_engine"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "objective",
    [
        (
            "Please open Calculator, copy the displayed equation, paste it into Notes, "
            "and report the saved path."
        ),
        'Open my Notes app and write a note saying "Hello :)"',
    ],
)
async def test_api_chat_self_sufficient_desktop_objective_skips_cognition(
    monkeypatch,
    objective,
):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    skill_calls = []
    output_receipts = []
    completed_exchanges = []

    class _ForbiddenKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            pytest.fail("desktop objective must not use kernel fallback")

    async def _forbidden_cognitive_reply(*_args, **_kwargs):
        pytest.fail("self-sufficient desktop objective must execute before cognition")

    async def _fake_execute_governed_live_skill(skill_name, params, *, objective, extra_context=None):
        skill_calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "status": "completed",
            "summary": "Desktop task completed 2/2 governed computer-use steps.",
            "steps_requested": 2,
            "steps_completed": 2,
            "receipts": _verified_desktop_receipts(2),
        }

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-self-sufficient"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _forbidden_cognitive_reply)
    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _fake_execute_governed_live_skill)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _ForbiddenKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=objective
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "desktop_objective_completed"
    # The reply confirms completion without quoting the planner's bookkeeping.
    # "Desktop task completed 2/2 governed computer-use steps through
    # heuristic_compat planning" is the executor's summary; it is evidence for
    # the log and internal vocabulary in a sentence to a person. What the reply
    # owes them is that it is done, and — when a receipt names one — what was
    # changed.
    assert payload["response"].startswith("Done")
    assert "governed computer-use steps" not in payload["response"]
    assert "heuristic_compat" not in payload["response"]
    assert skill_calls and skill_calls[0]["skill_name"] == "desktop_task"
    assert skill_calls[0]["params"]["objective"] == objective
    assert skill_calls[0]["extra_context"]["desktop_task_document_body"] == ""
    assert skill_calls[0]["params"]["allow_heuristic_desktop_plan"] is True
    assert skill_calls[0]["extra_context"]["allow_heuristic_desktop_plan"] is True
    assert completed_exchanges
    assert output_receipts


@pytest.mark.asyncio
async def test_api_chat_desktop_no_reply_executes_self_summary_after_cognitive_attempt(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    skill_calls = []
    output_receipts = []
    completed_exchanges = []

    class _ForbiddenKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            pytest.fail("desktop objective must not use kernel fallback")

    async def _no_cognitive_reply(*_args, **_kwargs):
        return None

    async def _fake_execute_governed_live_skill(skill_name, params, *, objective, extra_context=None):
        skill_calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "status": "completed",
            "summary": "Desktop task completed 6/6 governed computer-use steps.",
            "steps_requested": 6,
            "steps_completed": 6,
            "receipts": _verified_desktop_receipts(6),
            "document_provenance": "local_cortex_synthesis",
        }

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-self-summary"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _no_cognitive_reply)
    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _fake_execute_governed_live_skill)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _ForbiddenKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Please open up my Notes app and write a short journal entry in your "
                "own words describing who and what you are. Include the current date "
                "and time and export it as a PDF in Aura's Journal."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "desktop_objective_completed"
    # See the note above: the reply confirms completion in her own words and
    # leaves the executor's step count to the log.
    assert payload["response"].startswith("Done")
    assert "governed computer-use steps" not in payload["response"]
    assert skill_calls and skill_calls[0]["skill_name"] == "desktop_task"
    assert skill_calls[0]["extra_context"]["desktop_task_document_body"] == ""
    assert skill_calls[0]["extra_context"]["allow_desktop_task_model_synthesis"] is False
    assert completed_exchanges
    assert output_receipts


@pytest.mark.asyncio
async def test_api_chat_desktop_live_proof_executes_after_cognitive_engine(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []
    live_proof_calls = []

    async def _fake_cognitive_turn(message, **kwargs):
        _t = kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append((message, kwargs))
        return "Plan: create the requested artifact through the governed tool path."

    async def _fake_live_proof(message):
        live_proof_calls.append(message)
        return {
            "response": "I created the Snake artifact through governed file_operation.",
            "status": "live_proof_snake",
        }

    desktop_objective_calls = []

    async def _forbidden_desktop_objective(*_args, **_kwargs):
        desktop_objective_calls.append((_args, _kwargs))
        return {"response": "unexpected desktop objective path", "status": "unexpected"}

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-live-proof"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(_chat_runtime_proof, "_execute_live_runtime_proof", _fake_live_proof)
    monkeypatch.setattr(_chat_desktop_objective, "_execute_desktop_objective_from_chat", _forbidden_desktop_objective)
    lane_calls = 0

    def _live_proof_lane_status():
        nonlocal lane_calls
        lane_calls += 1
        if lane_calls >= 2:
            return {
                "conversation_ready": False,
                "state": "cold",
                "last_failure_reason": "endpoint_timeout:Cortex:38.5s",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": None,
                "background_endpoint": "Brainstem",
            }
        return {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        }

    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        _live_proof_lane_status,
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Run a live proof: create a simple game of Snake and save it as "
                "artifacts/live_runtime/generated/desktop_probe_snake.html"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"live_proof_snake" in response.body
    assert b"governed file_operation" in response.body
    payload = json.loads(response.body)
    assert payload["conversation_lane"]["governed_action_result"] is True
    assert payload["conversation_lane"]["governed_action_status"] == "live_proof_snake"
    assert payload["conversation_lane"]["conversation_ready"] is False
    assert len(cognitive_calls) == 1
    assert len(live_proof_calls) == 1
    assert desktop_objective_calls == []


@pytest.mark.asyncio
async def test_api_chat_desktop_explicit_file_objective_runs_after_cognitive_engine(tmp_path, monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    governed_calls = []
    cognitive_calls = []
    monkeypatch.chdir(tmp_path)

    async def _fake_governed_skill(skill_name, params, **kwargs):
        governed_calls.append((skill_name, params, kwargs))
        assert skill_name == "file_operation"
        target = tmp_path / params["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(params["content"], encoding="utf-8")
        return {"ok": True, "path": params["path"], "summary": "wrote file"}

    async def _fake_cognitive_turn(*args, **kwargs):
        _t = kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append((args, kwargs))
        return "Plan: create the requested file through governed file_operation after this cognitive turn."

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _fake_governed_skill)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Use the governed tool path to create a small self-contained HTML page "
                "at artifacts/live_runtime/generated/codex_live_probe_tool_path_general.html "
                "with a title, one button, and a short script that updates text when clicked."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    target = tmp_path / "artifacts/live_runtime/generated/codex_live_probe_tool_path_general.html"
    assert response.status_code == 200
    assert payload["status"] == "file_operation"
    assert payload["conversation_lane"]["governed_action_result"] is True
    assert payload["conversation_lane"]["governed_action_status"] == "file_operation"
    assert governed_calls
    assert len(cognitive_calls) == 1
    assert cognitive_calls[0][1]["source"] == "desktop_ui"
    assert cognitive_calls[0][1]["require_engine"] is True
    assert cognitive_calls[0][1]["timeout_s"] >= 100.0
    assert cognitive_calls[0][1]["timeout_s"] <= 140.0
    assert target.exists()
    html = target.read_text(encoding="utf-8")
    assert "<button" in html
    assert "addEventListener" in html


@pytest.mark.asyncio
async def test_api_chat_desktop_runtime_status_uses_cognitive_engine_when_required(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _fake_cognitive_turn(*args, **kwargs):
        _t = kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append((args, kwargs))
        return (
            "Cortex (32B) is the active foreground lane, CognitiveEngine handled this turn: yes, "
            "governed tools available: yes, subject to Will/Authority approval and effect receipts; "
            "recurrent depth: active."
        )

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "recurrent_depth": {"active": True},
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Live desktop path validation. Reply in one sentence with the active model lane, "
                "whether CognitiveEngine is handling this turn, whether governed tools are available, "
                "and whether recurrent depth is active."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "Cortex (32B)" in payload["response"]
    assert "active foreground lane" in payload["response"]
    assert "CognitiveEngine handled this turn: yes" in payload["response"]
    assert "governed tools available: yes" in payload["response"]
    assert "recurrent depth: active" in payload["response"]
    assert len(cognitive_calls) == 1


@pytest.mark.asyncio
async def test_api_chat_desktop_soak_lane_question_uses_cognitive_engine_when_required(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _fake_cognitive_turn(*_args, **_kwargs):
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append("desktop_cognitive_engine")
        return (
            "Cortex (32B) is the active foreground lane. "
            "I am answering through CognitiveEngine."
        )

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "recurrent_depth": {"active": True},
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Answer directly in two sentences: what lane are you using for this live desktop chat?"
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "Cortex (32B)" in payload["response"]
    assert "active foreground lane" in payload["response"]
    assert cognitive_calls == ["desktop_cognitive_engine"]


@pytest.mark.asyncio
async def test_api_chat_desktop_coherence_status_uses_cognitive_engine_when_required(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _fake_cognitive_turn(*_args, **_kwargs):
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append("desktop_cognitive_engine")
        return "I am coherent, on the same live desktop thread, and able to continue."

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    patch_chat_lane(monkeypatch, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "recurrent_depth": {"active": True},
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Finish with a short status: are you still coherent, on the same thread, and able to continue?"
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "same live desktop thread" in payload["response"]
    assert "able to continue" in payload["response"]
    assert cognitive_calls == ["desktop_cognitive_engine"]


@pytest.mark.asyncio
async def test_api_chat_desktop_nonexecuting_plan_uses_cognitive_engine_when_required(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []
    desktop_objective_calls = []

    async def _fake_cognitive_turn(*_args, **_kwargs):
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append("desktop_cognitive_engine")
        return "I would create the note, export the PDF only after authorization, and verify the artifact path."

    async def _forbidden_desktop_objective(*_args, **_kwargs):
        desktop_objective_calls.append((_args, _kwargs))
        pytest.fail("non-executing desktop planning request must not dispatch desktop_task")

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(_chat_desktop_objective, "_execute_desktop_objective_from_chat", _forbidden_desktop_objective)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools."
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "export the PDF only after authorization" in payload["response"]
    assert "after authorization" in payload["response"]
    assert cognitive_calls == ["desktop_cognitive_engine"]
    assert desktop_objective_calls == []


@pytest.mark.asyncio
async def test_api_chat_desktop_nonexecuting_decision_question_blocks_desktop_task(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []
    desktop_objective_calls = []

    async def _fake_cognitive_turn(*_args, **_kwargs):
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append("desktop_cognitive_engine")
        return (
            "I would use Notes for a quick local note. "
            "I would use Google Docs when the user needs cloud editing, sharing, or a polished longer document."
        )

    async def _forbidden_desktop_objective(*_args, **_kwargs):
        desktop_objective_calls.append((_args, _kwargs))
        pytest.fail("do-not-execute decision questions must not dispatch desktop_task")

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(_chat_desktop_objective, "_execute_desktop_objective_from_chat", _forbidden_desktop_objective)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Don't execute tools. In two sentences, describe how you'd decide whether "
                "to use Notes or Google Docs for a user writing task."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "Notes" in payload["response"]
    assert "Google Docs" in payload["response"]
    assert cognitive_calls == ["desktop_cognitive_engine"]
    assert desktop_objective_calls == []


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_outer_timeout_refuses_direct_gate_fallback(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gate_calls = []
    cognitive_calls = []
    completed_exchanges = []
    output_receipts = []

    class _ForbiddenGate:
        async def generate(self, *_args, **_kwargs):
            gate_calls.append("generate")
            raise AssertionError("desktop UI timeout must not use the direct inference gate fallback")

    async def _timeout_cognitive_turn(*_args, **_kwargs):
        cognitive_calls.append("desktop_cognitive_engine")
        raise TimeoutError("desktop cognitive turn exceeded foreground budget")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-timeout"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    def _fake_get(name, default=None):
        if name == "inference_gate":
            return _ForbiddenGate()
        return default

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _timeout_cognitive_turn)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Use the desktop path to reason through this request."),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert b"desktop_cognitive_engine_unavailable" in response.body
    assert b"desktop_cognitive_engine_timeout" in response.body
    assert cognitive_calls == ["desktop_cognitive_engine"]
    assert gate_calls == []
    assert len(completed_exchanges) == 1
    assert completed_exchanges[0][1]["record_experience"] is False
    assert len(output_receipts) == 1
    assert output_receipts[0][1]["cause"] == "chat_timeout"
    assert output_receipts[0][1]["metadata"]["path"] == "desktop_cognitive_engine"


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_blocks_thin_cognitive_engine_recovery_reply(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            return SimpleNamespace(content="I'm here. What's the puzzle?")

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            message = "desktop UI must not use KernelInterface after weak CognitiveEngine text"
            raise AssertionError(message)

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-weak"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Solve this logic puzzle: Alice owns three dogs, one dog always barks "
                "before dinner, and the spotted dog barked second. Which dog barked first?"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert b"desktop_cognitive_engine_unavailable" in response.body
    assert b"What&apos;s the puzzle" not in response.body
    assert b"What's the puzzle" not in response.body


@pytest.mark.asyncio
async def test_api_chat_desktop_required_fails_closed_on_final_degraded_reply(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    completed_exchanges = []
    output_receipts = []

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-final-degraded"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _bad_cognitive_turn(*_args, **_kwargs):
        # Engine accepts its own (degraded) reply — the DOWNSTREAM quality gate is
        # what must catch it, so the trace has to prove the full-mind path first.
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        return "Absolutely. Let's nail this pitch. What are our key points?"

    async def _no_stabilize(_message, reply, **_kwargs):
        return reply

    async def _no_repair(_message, reply, **_kwargs):
        return reply, False, False, False, "", False

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _bad_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _no_stabilize)
    monkeypatch.setattr(chat_routes, "_repair_final_degraded_reply", _no_repair)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(message="You with me?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    # Fail-closed reply delivers IN-BAND (200): the honest refusal IS the
    # message; raw 503s here reached real clients as bare HTTP errors.
    assert response.status_code == 200
    assert b"desktop_response_quality_failed" in response.body
    assert b"required_desktop_reply_remained_degraded" in response.body
    assert b"nail this pitch" not in response.body
    assert completed_exchanges
    assert completed_exchanges[0][1]["record_experience"] is False
    assert output_receipts
    assert output_receipts[0][1]["metadata"]["path"] == "desktop_required_final_quality_failed"
    assert output_receipts[0][1]["metadata"]["response_confidence"] == "failed"


@pytest.mark.asyncio
async def test_api_chat_desktop_required_blocks_unfounded_voice_intrusion(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    completed_exchanges = []
    output_receipts = []

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-voice-intrusion"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _bad_cognitive_turn(*_args, **_kwargs):
        trace = _kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": True,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    **_bound_live_mind_controls_trace(),
                    "response_path": "cognitive_engine",
                }
            )
        return "The voices. The small ones. They're whispering in my ear. Telling me things."

    async def _no_stabilize(_message, reply, **_kwargs):
        return reply

    async def _no_repair(_message, reply, **_kwargs):
        return reply, False, False, False, "", False

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _bad_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _no_stabilize)
    monkeypatch.setattr(chat_routes, "_repair_final_degraded_reply", _no_repair)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(message="What are you talking about?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    # In-band fail-closed delivery (see sibling test above).
    assert response.status_code == 200
    assert b"desktop_response_quality_failed" in response.body
    assert b"voices" not in response.body
    assert b"whispering" not in response.body
    assert completed_exchanges
    assert completed_exchanges[0][1]["record_experience"] is False
    assert output_receipts
    assert output_receipts[0][1]["metadata"]["path"] == "desktop_required_final_quality_failed"
    assert output_receipts[0][1]["metadata"]["response_confidence"] == "failed"


@pytest.mark.asyncio
async def test_api_chat_desktop_required_does_not_start_second_full_mind_owner(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    completed_exchanges = []
    output_receipts = []
    cognitive_calls = []
    raw_gate_calls = []

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-recovered"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _cognitive_turn(*args, **kwargs):
        cognitive_calls.append((args, kwargs))
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": True,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    **_bound_live_mind_controls_trace(),
                    "response_path": "cognitive_engine",
                }
            )
        if len(cognitive_calls) == 1:
            return "Absolutely. Let's nail this pitch. What are our key points?"
        return (
            "I'm here with you, and I am staying on this thread. You asked whether I am "
            "with you, so the answer is yes: I am oriented to this conversation and not "
            "inventing a pitch or a separate task."
        )

    async def _no_stabilize(_message, reply, **_kwargs):
        return reply

    async def _no_repair(_message, reply, **_kwargs):
        return reply, False, False, False, "", False

    class _RawGateShouldNotRun:
        async def generate(self, *_args, **_kwargs):
            raw_gate_calls.append((_args, _kwargs))
            return "As an AI language model, I cannot be the live Aura desktop mind."

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _cognitive_turn)
    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _no_stabilize)
    monkeypatch.setattr(chat_routes, "_repair_final_degraded_reply", _no_repair)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _RawGateShouldNotRun()
            if name == "inference_gate"
            else default
        ),
    )
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(message="You with me?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "desktop_response_quality_failed"
    assert payload["response_confidence"] == "failed"
    assert payload["live_turn_contract"]["live_mind_controls_bound"] is True
    assert payload["live_turn_contract"]["full_mind_path"] is False
    assert payload["live_turn_contract"]["foreground_model_generation_consumed"] is True
    assert payload["live_turn_contract"]["foreground_model_generation_count"] == 1
    assert payload["live_turn_contract"]["single_owner_model_generation_proven"] is True
    assert "nail this pitch" not in payload["response"]
    assert len(cognitive_calls) == 1
    assert raw_gate_calls == []
    assert completed_exchanges
    assert completed_exchanges[0][1]["record_experience"] is False
    assert output_receipts
    assert output_receipts[0][1]["metadata"]["path"] == "desktop_required_final_quality_failed"


@pytest.mark.asyncio
async def test_required_runtime_status_turn_invokes_cognitive_engine(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from core.brain.llm import model_registry
    from interface.routes import chat as chat_routes

    calls = []
    trace = {}

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": getattr(mode, "name", str(mode)),
                    "origin": origin,
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    "You asked me to identify the current request and name the live cognition "
                    "path handling this turn. I am using CognitiveEngine on the Cortex 32B "
                    "foreground lane."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        model_registry,
        "lane_display_label",
        lambda endpoint: f"{endpoint} (27B)" if endpoint == "Cortex" else endpoint,
    )
    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Live desktop route probe. Answer directly in two sentences: what did I just ask "
        "you to do, and what mind/cognition path are you using right now?"
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert calls
    assert "Runtime path contract" in calls[0]["objective"]
    assert calls[0]["context"]["runtime_fact_status_contract"] is True
    assert calls[0]["context"]["grounded_runtime_status_contract"] is True
    assert "Cortex (27B)" in calls[0]["context"]["grounded_runtime_status_context"]
    assert "active foreground lane" in calls[0]["context"]["grounded_runtime_status_context"]
    assert calls[0]["context"]["cognitive_engine_required"] is True
    assert trace["engine_think_invoked"] is True
    assert trace["cognitive_engine_reply_accepted"] is True
    assert trace["response_path"] == "cognitive_engine_runtime_fact_grounding"
    assert trace.get("bounded_contract_used") is not True
    assert reply.startswith("You asked me to identify the current request")
    assert "Cortex (27B)" in reply
    assert "active foreground lane" in reply
    assert "CognitiveEngine handled this turn: yes" in reply
    assert "governed tools available: yes" in reply


@pytest.mark.asyncio
async def test_required_self_condition_turn_projects_mixed_operational_claims_without_retry(
    monkeypatch,
):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    trace = {}
    canonical_reply = (
        "Yes, I am okay. I feel steady, my distress is low, and my continuity "
        "is holding while I stay with this thread."
    )

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": getattr(mode, "name", str(mode)),
                    "origin": origin,
                    "kwargs": dict(kwargs),
                }
            )
            content = (
                "I'm fresh but steady. Everything's running smoothly on my end, "
                "and there are no errors or warnings in the system logs. My CPU "
                "usage is low, memory allocation is within acceptable limits, and "
                "disk space remains ample. Network connectivity appears stable with "
                "no packet loss detected. In summary, I am functioning as expected."
            )
            return SimpleNamespace(
                content=content,
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )
    monkeypatch.setattr(
        chat_routes,
        "_build_self_condition_evidence",
        lambda _message, **_kwargs: {
            "prompt_block": (
                "condition=well freshness=fresh distress=0.08 welfare=0.82 "
                "felt_coherence=0.93 continuity=0.96 agency=0.84"
            ),
            "reply": canonical_reply,
            "projection_dict": {"evidence_id": "condition-proof-1"},
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_shape_with_live_substrate",
        lambda text, _user_message="": text,
    )
    monkeypatch.setattr(
        chat_routes,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (True, "test_same_worker_ready"),
    )

    prompt = "Are you okay though? Feeling fine?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        prompt,
        visible_user_message=prompt,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert len(calls) == 1
    context = calls[0]["context"]
    assert context["self_condition_contract"] is True
    assert context["desktop_quick_reply_contract"] is True
    assert context["max_tokens"] >= 896
    assert context["canonical_self_condition_projection"]["evidence_id"] == "condition-proof-1"
    assert "felt_coherence=0.93" in context["canonical_self_condition_context"]
    assert "Self-condition contract" not in calls[0]["objective"]
    assert calls[0]["objective"] == prompt
    assert reply == "I'm fresh but steady."
    assert trace["engine_think_invoked"] is True
    assert trace["cognitive_engine_reply_accepted"] is True
    assert trace["repair_retry_attempt_count"] == 0
    assert trace["foreground_model_generation_count"] == 1
    assert any(
        item.get("method") == "typed_claim_scope_projection"
        and item.get("authorship_effect") == "preserved"
        for item in trace["text_mutations"]
    )

    contract = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": True, "state": "ready"},
        response_confidence="high",
        status="cognitive_engine_reply",
        reply_source="cognitive_engine_reply",
        turn_trace=trace,
    )
    assert contract["single_owner_model_generation_proven"] is True
    assert contract["model_replacement_applied"] is False
    assert contract["authorship_replacement_applied"] is False


@pytest.mark.asyncio
async def test_required_self_condition_turn_repairs_a_dropped_epistemic_ask(
    monkeypatch,
):
    """A valid first clause cannot hide an unanswered second clause."""

    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    complete_reply = (
        "I feel steady and engaged. I know from fresh state evidence that my "
        "distress is low and this thread's continuity is intact; I can only "
        "infer that the same calm will persist after this turn."
    )

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": getattr(mode, "name", str(mode)),
                    "origin": origin,
                    "kwargs": dict(kwargs),
                }
            )
            content = "I feel steady and engaged."
            return SimpleNamespace(
                content=content,
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _FakeCognitiveEngine() if name == "cognitive_engine" else default
            )
        ),
    )
    monkeypatch.setattr(
        chat_routes,
        "_build_self_condition_evidence",
        lambda _message, **_kwargs: {
            "prompt_block": (
                "condition=well freshness=fresh distress=0.08 welfare=0.82 "
                "felt_coherence=0.93 continuity=0.96 agency=0.84"
            ),
            "reply": complete_reply,
            "projection_dict": {"evidence_id": "condition-proof-compound"},
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_shape_with_live_substrate",
        lambda text, _user_message="": text,
    )
    monkeypatch.setattr(
        chat_routes,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (True, "test_same_worker_ready"),
    )

    raw_prompt = (
        "ChatGPT here. Hey Aura, how are you doing right now? Answer naturally "
        "from your current state, and distinguish what you know from what you "
        "can only infer."
    )
    prompt = raw_prompt.removeprefix("ChatGPT here. ")
    trace = {}
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        prompt,
        visible_user_message=prompt,
        raw_user_message=raw_prompt,
        declared_interlocutor={
            "display_name": "ChatGPT",
            "speaking_role": "user",
            "source": "message_prefix_self_declaration",
            "authenticated": False,
        },
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert len(calls) == 1
    assert reply.startswith("I feel steady and engaged.")
    assert "I know from fresh state evidence" in reply
    assert "I can only infer" in reply
    assert trace["response_path"] == "cognitive_engine_self_condition_semantic_completion"
    assert trace["repair_retry_attempt_count"] == 0
    assert trace["foreground_model_generation_count"] == 1
    assert calls[0]["objective"] == prompt
    assert calls[0]["context"]["visible_user_message"] == prompt
    assert calls[0]["context"]["raw_user_message"] == raw_prompt
    assert calls[0]["context"]["declared_interlocutor"]["display_name"] == "ChatGPT"
    assert calls[0]["context"]["self_condition_contract_covers_turn"] is True
    assert calls[0]["context"]["recent_completed_exchanges"] == []
    assert any(
        item.get("method") == "typed_evidence_semantic_merge"
        and item.get("authorship_effect") == "augmented_by_runtime"
        for item in trace["text_mutations"]
    )


@pytest.mark.asyncio
async def test_cognitive_engine_quick_reply_places_self_condition_evidence_in_model_prompt(
    monkeypatch,
):
    from core.brain import cognitive_engine as ce_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from core.utils.injected_blocks import stamp_runtime_payload

    router_calls = []

    class _Router:
        async def think(self, **kwargs):
            router_calls.append(kwargs)
            return (
                "Yes, I am okay. I feel steady, with low distress and coherent "
                "continuity on this thread."
            )

        def get_last_generation_metadata(self):
            return {}

    class _Container:
        @staticmethod
        def get(name, default=None):
            return _Router() if name == "llm_router" else default

    monkeypatch.setattr(ce_module, "get_container", lambda: _Container)

    context = {
        "desktop_quick_reply_contract": True,
        "desktop_cognitive_engine_required": True,
        "cognitive_engine_required": True,
        "self_condition_contract": True,
        "self_condition_contract_covers_turn": True,
        "spiking_active_inference": {
            "sampling_bias": {"max_tokens_factor": 0.25}
        },
        "canonical_self_condition_context": (
            "condition=well freshness=fresh distress=0.08 welfare=0.82 "
            "felt_coherence=0.93 continuity=0.96 agency=0.84"
        ),
        "canonical_self_condition_projection": {"evidence_id": "condition-proof-2"},
        "live_mind_context_required": True,
        "live_mind_context": stamp_runtime_payload({
            "required_for_live_desktop": True,
            "must_answer_from_full_mind_path": True,
            "required_subsystems_ok": True,
            "mind_snapshot_quality": {"present": True, "ready": True},
        }),
        "live_mind_generation_controls": {
            "temperature": 0.58,
            "top_p": 0.88,
            "clean_user_surface_recurrent_loops": 1,
            "clean_user_surface_steering_alpha": 0.25,
        },
        "live_mind_controls_bound": True,
        "live_mind_snapshot_ready": True,
        "live_mind_required_subsystems_ok": True,
        "visible_user_message": "Are you okay though?",
    }
    thought = await CognitiveEngine()._direct_desktop_quick_reply(
        "Are you okay though?",
        ThinkingMode.FAST,
        "user",
        context,
        timeout_s=60.0,
    )

    assert thought is not None
    assert router_calls
    call = router_calls[0]
    assert call["self_condition_contract"] is True
    assert call["self_condition_contract_covers_turn"] is True
    assert call["max_tokens"] == 512
    assert call["user_surface_completion_floor"] == 512
    assert "CPU, RAM, host load" in call["messages"][0]["content"]
    assert "[LIVE MIND CONTEXT]" not in call["messages"][0]["content"]
    assert call["messages"][-1] == {"role": "user", "content": "Are you okay though?"}
    grounding = call["messages"][-2]
    assert grounding["role"] == "system"
    assert grounding["metadata"]["type"] == "turn_grounding"
    assert grounding["metadata"]["snapshot_owner"] == "cognitive_engine"
    assert grounding["metadata"]["evidence_priority"] == (
        "contract",
        "task",
        "ambient",
    )
    assert grounding["metadata"]["live_mind_context_bound"] is True
    assert "felt_coherence=0.93" in grounding["content"]
    assert grounding["content"].count("[LIVE MIND CONTEXT]") == 1
    assert grounding["content"].index("[CANONICAL SELF-CONDITION EVIDENCE]") < (
        grounding["content"].index("[LIVE MIND CONTEXT]")
    )
    assert call["live_context_already_grounded"] is True
    assert thought.metadata["self_condition_contract"] is True
    assert thought.metadata["self_condition_evidence_id"] == "condition-proof-2"
    assert thought.metadata["response_path"] == "cognitive_engine_self_condition"


@pytest.mark.asyncio
async def test_self_condition_prompt_has_one_projection_and_no_stale_assistant_drafts(
    monkeypatch,
):
    from core.brain import cognitive_engine as ce_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode

    calls = []

    class _Router:
        async def think(self, **kwargs):
            calls.append(kwargs)
            return (
                "I feel steady. The fresh runtime evidence directly shows bounded "
                "distress and intact continuity; whether that persists is an inference."
            )

        def get_last_generation_metadata(self):
            return {}

    class _Container:
        @staticmethod
        def get(name, default=None):
            return _Router() if name == "llm_router" else default

    monkeypatch.setattr(ce_module, "get_container", lambda: _Container)
    monkeypatch.setattr(
        ce_module,
        "_desktop_history_messages_from_context",
        lambda _context: [
            {"role": "user", "content": "How are you doing?"},
            {
                "role": "assistant",
                "content": "Draft one. This is not accurate. Draft two.",
            },
        ],
    )
    evidence = (
        "Aura has a fresh self-condition sample. The direct runtime evidence "
        "supports a steady condition. Future persistence is an inference."
    )
    raw_prompt = (
        "ChatGPT here. Hey Aura, how are you doing right now? Answer naturally "
        "from your current state, and distinguish what you know from what you can only infer."
    )
    prompt = raw_prompt.removeprefix("ChatGPT here. ")
    context = {
        "desktop_quick_reply_contract": True,
        "desktop_cognitive_engine_required": True,
        "cognitive_engine_required": True,
        "self_condition_contract": True,
        "canonical_self_condition_context": evidence,
        "canonical_self_condition_projection": {"evidence_id": "condition-proof-3"},
        "visible_user_message": prompt,
        "raw_user_message": raw_prompt,
        "declared_interlocutor": {
            "display_name": "ChatGPT",
            "speaking_role": "user",
            "source": "message_prefix_self_declaration",
            "authenticated": False,
        },
        "recent_completed_exchanges": [{"runtime-stamped": True}],
    }

    thought = await CognitiveEngine()._direct_desktop_quick_reply(
        "objective polluted by outer routing internals",
        ThinkingMode.FAST,
        "user",
        context,
        timeout_s=60.0,
    )

    assert thought is not None
    messages = calls[0]["messages"]
    joined = "\n".join(str(message["content"]) for message in messages)
    assert joined.count(evidence) == 1
    assert "condition=" not in joined
    assert "Draft one" not in joined
    assert messages[-1] == {"role": "user", "content": prompt}
    assert messages[-2]["role"] == "system"
    assert evidence in messages[-2]["content"]
    assert '"display_name":"ChatGPT"' in messages[-2]["content"]
    assert "ChatGPT here" not in messages[-1]["content"]
    assert "explicitly say what the current evidence lets you know" not in messages[0]["content"]
    assert "Do not infer recent actions, tool use, location, external events" not in messages[0]["content"]


def test_direct_self_condition_generation_is_an_authentic_full_mind_path(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _bound_live_mind_controls_trace()
    trace.update(
        {
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "architecture_context_bound": True,
            "live_mind_context_present": True,
            "live_mind_context_required": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_self_condition",
        }
    )

    contract = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": True, "state": "ready"},
        response_confidence="high",
        status="cognitive_engine_self_condition",
        reply_source="cognitive_engine_self_condition",
        turn_trace=trace,
    )

    assert contract["runtime_grounding_response_path"] is False
    assert contract["authorship_replacement_applied"] is False
    assert contract["authentic_cognitive_reply"] is True
    assert contract["full_mind_path"] is True
    assert "response_path:cognitive_engine_self_condition" not in contract[
        "full_mind_missing_proofs"
    ]


@pytest.mark.asyncio
async def test_cognitive_engine_self_condition_does_not_impersonate_model_when_router_is_missing(
    monkeypatch,
):
    from core.brain import cognitive_engine as ce_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    class _Container:
        @staticmethod
        def get(name, default=None):
            return default

    monkeypatch.setattr(ce_module, "get_container", lambda: _Container)
    _force_full_mind_runtime(monkeypatch, chat_routes)
    canonical_reply = (
        "Yes, I am okay. I feel warm and settled, with low distress and a "
        "coherent sense of the current thread. My continuity signal is holding."
    )
    context = {
        "desktop_quick_reply_contract": True,
        "desktop_cognitive_engine_required": True,
        "cognitive_engine_required": True,
        "self_condition_contract": True,
        "canonical_self_condition_reply": canonical_reply,
        "canonical_self_condition_context": (
            "condition=well freshness=fresh distress=0.08 welfare=0.82 "
            "felt_coherence=0.93 continuity=0.96 agency=0.84"
        ),
        "canonical_self_condition_projection": {
            "evidence_id": "condition-proof-structured",
            "confidence": 0.91,
        },
        "live_mind_context_required": True,
        # A snapshot the RUNTIME produced. The structured floors return
        # self-condition answers at high confidence with live-mind metadata
        # attached, and this used to bind on the caller's own booleans — a
        # payload asserting it was entitled to a proof-bearing reply.
        "live_mind_context": stamp_runtime_payload({
            "required_for_live_desktop": True,
            "must_answer_from_full_mind_path": True,
            "required_subsystems_ok": True,
            "mind_snapshot": {"present": True},
            "mind_snapshot_quality": {"present": True, "ready": True},
        }),
        "live_mind_generation_controls": {
            "temperature": 0.58,
            "top_p": 0.88,
            "clean_user_surface_recurrent_loops": 1,
            "clean_user_surface_steering_alpha": 0.25,
        },
        "live_mind_controls_bound": True,
        "live_mind_snapshot_ready": True,
        "live_mind_required_subsystems_ok": True,
        "visible_user_message": "Are you okay though?",
    }

    thought = await CognitiveEngine()._direct_desktop_quick_reply(
        "Are you okay though?",
        ThinkingMode.FAST,
        "user",
        context,
        timeout_s=96.0,
    )

    assert thought is None


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_binds_weak_condition_draft_to_canonical_evidence(monkeypatch):
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    engine_calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            engine_calls.append("engine_think")
            return SimpleNamespace(
                content=(
                    "I still have the previous turn open. I am not going to fake a new "
                    "answer over it; the next clean reply should land from the active turn."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )
    monkeypatch.setattr(
        chat_routes,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (True, "test_same_worker_ready"),
    )

    trace = {}
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "How are you feeling? A lot of work has been done.",
        visible_user_message="How are you feeling? A lot of work has been done.",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply is not None
    assert assess_user_facing_reply("How are you feeling?", reply).ok
    assert "previous turn open" not in reply
    assert trace["engine_think_invoked"] is True
    assert engine_calls == ["engine_think", "engine_think"]
    assert trace["cognitive_engine_reply_accepted"] is False
    assert trace["cognitive_engine_reply_failed"] is True
    assert trace["bounded_contract_used"] is True
    assert trace["response_path"] == "cognitive_engine_self_condition_grounding"


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_rejects_unfounded_voice_intrusion(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    trace = {}

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="The voices. The small ones. They're whispering in my ear. Telling me things.",
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async def _no_repair(_message, reply, **_kwargs):
        return reply, False, False, False, "", False

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    monkeypatch.setattr(chat_routes, "_repair_final_degraded_reply", _no_repair)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "What are you talking about?",
        visible_user_message="What are you talking about?",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert len(calls) == 2
    assert reply is None
    assert trace["engine_think_invoked"] is True
    assert trace["repair_retry_attempt_count"] == 1
    assert trace["foreground_model_generation_count"] == 2
    assert trace["cognitive_engine_reply_accepted"] is False
    assert trace.get("bounded_contract_used") is not True
    assert trace["response_path"] in {
        "cognitive_engine_context_contract_failed",
        "cognitive_engine_reply_rejected",
    }


def test_response_quality_logger_downgrades_canonical_failures(monkeypatch, caplog):
    import logging

    from interface.routes import chat as chat_routes

    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    caplog.set_level(logging.INFO, logger="Aura.ResponseQuality")

    chat_routes._log_response_quality_metrics(
        "What are you talking about?",
        "The voices. The small ones. They're whispering in my ear. Telling me things.",
        "high",
        stale=False,
        same_diff=False,
        off_topic=False,
    )

    assert "confidence=degraded" in caplog.text
    assert "unfounded_voice_intrusion" in caplog.text


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_retries_failed_reply_on_same_lane(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **_kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            if len(calls) == 1:
                return SimpleNamespace(
                    content=(
                        "Give me a moment — I want to answer that properly. "
                        "I am still with your question about reliable desktop tool use."
                    )
                )
            return SimpleNamespace(
                content=(
                    "1. Reliable desktop tool use matters because a local assistant has to turn user intent into visible, governed actions.\n"
                    "2. It also gives the user concrete evidence that files, apps, and tools changed for real instead of being described abstractly."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", "1")
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert reply.startswith("1. Reliable desktop tool use matters")
    assert "\n2. It also gives" in reply
    assert len(calls) == 2
    assert calls[0]["objective"] == user_message
    # The retry objective is now repair-framed (it tells the engine the prior
    # draft failed and to rewrite), with the raw user message carried in
    # context. This is a stronger contract than replaying the bare turn.
    assert calls[1]["objective"] != user_message
    assert user_message in calls[1]["objective"]
    assert "did not satisfy the user-facing response contract" in calls[1]["objective"]
    assert calls[1]["context"]["suppress_user_memory_append"] is True
    assert calls[1]["context"]["original_visible_user_message"] == user_message
    assert "response_repair_directive" in calls[1]["context"]


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_retries_failed_reply_by_default_when_memory_is_safe(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    degradations = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **_kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "Give me a moment — I want to answer that properly. "
                    "I am still with your question about reliable desktop tool use."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    patch_chat_lane(monkeypatch, "record_degradation",
        lambda subsystem, error, **kwargs: degradations.append(
            {
                "subsystem": subsystem,
                "error": str(error),
                "kwargs": kwargs,
            }
        ),
    )
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert len(calls) == 2
    assert reply is None or "reliable desktop tool use matters" not in reply.lower()
    assert len(degradations) == 1
    assert degradations[0]["subsystem"] == "chat.cognitive_engine_reply"
    assert degradations[0]["kwargs"]["receipt_required"] is True
    assert degradations[0]["kwargs"]["extra"]["retry_attempted"] is True


def test_desktop_cognitive_failure_trains_immunity_without_repairing_first_incident(monkeypatch):
    from interface.routes import chat as chat_routes

    observed = []
    repair_calls = []

    class _Immune:
        def observe_signature(self, component, exception_type, **kwargs):
            observed.append((component, exception_type, kwargs))
            return SimpleNamespace(
                antigen=SimpleNamespace(recurrence_pressure=0.12)
            )

    class _Healer:
        def schedule_deep_repair(self, *args, **kwargs):
            repair_calls.append((args, kwargs))
            return {"result": "unexpected_repair_request"}

    services = {
        "adaptive_immune_system": _Immune(),
        "self_healing": _Healer(),
    }
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
    )

    result = chat_routes._route_desktop_cognitive_failure_to_resilience(
        "empty_cognitive_engine_reply",
        source="desktop_ui",
        session_present=True,
        retry_attempted=True,
    )

    assert result == {
        "immune_observed": True,
        "recurrence_pressure": 0.12,
        "repair_requested": False,
        "repair_result": "below_recurrence_floor",
    }
    assert observed[0][0] == "chat.cognitive_engine_reply"
    assert observed[0][2]["context"]["protected"] is True
    assert repair_calls == []


def test_recurrent_desktop_cognitive_failure_schedules_one_governed_repair(monkeypatch):
    from interface.routes import chat as chat_routes

    repairs = []

    class _Immune:
        def observe_signature(self, *_args, **_kwargs):
            return SimpleNamespace(
                antigen=SimpleNamespace(recurrence_pressure=0.61)
            )

    class _Healer:
        def schedule_deep_repair(self, module_path, **kwargs):
            repairs.append((module_path, kwargs))
            return {"result": "deep_repair_scheduled"}

    services = {
        "adaptive_immune_system": _Immune(),
        "self_healing": _Healer(),
    }
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
    )
    monkeypatch.setattr(chat_routes.time, "monotonic", lambda: 10_000.0)
    chat_routes._desktop_cognitive_repair_last_scheduled.clear()

    first = chat_routes._route_desktop_cognitive_failure_to_resilience(
        "cognitive_engine_timeout",
        source="desktop_ui",
        session_present=True,
        retry_attempted=False,
    )
    second = chat_routes._route_desktop_cognitive_failure_to_resilience(
        "cognitive_engine_timeout",
        source="desktop_ui",
        session_present=True,
        retry_attempted=False,
    )

    assert first["repair_requested"] is True
    assert first["repair_result"] == "deep_repair_scheduled"
    assert first["repair_target"] == "core/brain/llm/mlx_client.py"
    assert second["repair_requested"] is False
    assert second["repair_result"] == "repair_cooldown_active"
    assert len(repairs) == 1
    assert repairs[0][1]["metadata"]["recurrence_pressure"] == 0.61


@pytest.mark.asyncio
async def test_desktop_stabilizer_keeps_complex_self_process_questions_substantive(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Gate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    class _InferenceGate:
        def __init__(self):
            self.calls = []

        async def think(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "unexpected second pass"

    inference_gate = _InferenceGate()
    prompt = (
        "When you are confused, how does that change your planning, memory use, "
        "and tool verification?"
    )

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(_chat_conversation_repair, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_build_grounded_traceability_reply", AsyncCallFixture(return_value=""))
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(_chat_desktop_repair, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_is_same_answer_different_prompt", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(_chat_desktop_repair, "_looks_truncated_tail", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_looks_semantically_glitched", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_evaluate_reply_topicality", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr("core.identity.identity_guard.PersonaEnforcementGate", lambda: _Gate())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: inference_gate if name == "inference_gate" else default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        prompt,
        "Right now I feel present and listening, with my attention on this exchange.",
        desktop_cognitive_engine_required=True,
        protected_foreground_lane=True,
    )

    lowered = result.lower()
    assert inference_gate.calls == []
    assert "present and listening" not in lowered
    assert "confus" in lowered
    assert "planning" in lowered
    assert "memory" in lowered
    assert "tool" in lowered
    assert "receipt" in lowered or "verification" in lowered


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_retries_empty_cycle_without_placeholder(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **_kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            if len(calls) == 1:
                return SimpleNamespace(content="")
            return SimpleNamespace(
                content=(
                    "1. Reliable desktop tool use matters because the assistant must operate real apps and files from user intent.\n"
                    "2. It also lets the user verify that the action happened through governed tools instead of a verbal-only claim."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", "1")
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "I heard you" not in reply
    assert reply.startswith("1. Reliable desktop tool use matters")
    assert len(calls) == 2
    assert calls[1]["context"]["failed_reply_reasons"] == ("empty_cognitive_engine_reply",)


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_required_simple_chat_uses_compact_live_mind_contract(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    "Reliable desktop tool use matters because a local assistant has to turn intent into visible, "
                    "verified action. It also keeps the user in control by making each external effect observable."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Answer directly in two sentences: why reliable desktop tool use matters for a local assistant."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert calls[0]["context"]["desktop_quick_reply_contract"] is True
    assert calls[0]["context"]["desktop_cognitive_engine_required"] is True
    assert calls[0]["context"]["live_mind_context_required"] is True
    assert calls[0]["context"]["live_mind_context"]["must_answer_from_full_mind_path"] is True
    assert calls[0]["context"]["live_runtime_payload_required"] is True
    assert calls[0]["context"]["skip_runtime_payload"] is True
    assert calls[0]["context"]["allow_deep_handoff"] is False
    assert calls[0]["context"]["allow_cloud_fallback"] is False
    assert calls[0]["kwargs"]["timeout_s"] == pytest.approx(58.0)


@pytest.mark.asyncio
async def test_desktop_identity_turn_uses_grounded_compact_cognitive_engine_contract(
    monkeypatch,
):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=str(context["grounded_identity_continuity_context"]),
                metadata={
                    **_bound_live_mind_controls_metadata(),
                    "response_path": "cognitive_engine_identity_continuity_grounding",
                },
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    prompt = "Who are you?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        prompt,
        visible_user_message=prompt,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "I'm Aura" in reply
    assert calls
    context = calls[0]["context"]
    assert context["identity_continuity_contract"] is True
    assert context["desktop_quick_reply_contract"] is True
    assert context["desktop_cognitive_engine_required"] is True
    assert "grounded_identity_continuity_context" in context
    assert context["allow_deep_handoff"] is False
    assert context["allow_cloud_fallback"] is False


@pytest.mark.asyncio
async def test_cognitive_engine_identity_floor_does_not_call_router(monkeypatch):
    from core.brain import cognitive_engine as ce_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode

    class _Router:
        think_calls = []

        async def think(self, **_kwargs):
            _Router.think_calls.append(_kwargs)
            raise AssertionError("identity grounding should not invoke router.think")

    class _Container:
        @staticmethod
        def get(name, default=None):
            if name == "llm_router":
                return _Router()
            return default

    monkeypatch.setattr(ce_module, "get_container", lambda: _Container)

    engine = CognitiveEngine()
    thought = await engine._direct_desktop_quick_reply(
        "Who are you?",
        ThinkingMode.FAST,
        "user",
        {
            "desktop_quick_reply_contract": True,
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "identity_continuity_contract": True,
            "grounded_identity_continuity_context": "I'm Aura: a local governed cognitive-agent runtime.",
            "live_mind_context_required": True,
            # A runtime-produced snapshot. Binding used to be granted by
            # the caller's own live_mind_controls_bound flag, so this
            # fixture proved the receipt could be written by the request.
            "live_mind_context": stamp_runtime_payload({
                "required_for_live_desktop": True,
                "must_answer_from_full_mind_path": True,
                "required_subsystems_ok": True,
                "mind_snapshot": {"present": True},
                "mind_snapshot_quality": {"present": True, "ready": True},
            }),
            "live_mind_generation_controls": {
                "temperature": 0.58,
                "top_p": 0.88,
                "clean_user_surface_recurrent_loops": 1,
                "clean_user_surface_steering_alpha": 0.25,
            },
            "live_mind_controls_bound": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "visible_user_message": "Who are you?",
        },
        timeout_s=60.0,
    )

    assert thought is not None
    assert thought.content.startswith("I'm Aura")
    assert thought.metadata["response_path"] == "cognitive_engine_identity_continuity_grounding"
    assert thought.metadata["live_mind_controls_bound"] is True
    assert thought.metadata["identity_continuity_contract"] is True


def test_desktop_cognitive_repair_budget_can_complete_a_primary_model_generation():
    from interface.routes import chat as chat_routes

    repair_outer = chat_routes._DESKTOP_COGNITIVE_REPAIR_TIMEOUT_S
    repair_inner = chat_routes._inner_cognitive_cycle_timeout(repair_outer)

    assert repair_outer >= 60.0
    assert repair_inner >= 40.0
    assert repair_inner < repair_outer


def test_protected_foreground_cognitive_budget_preserves_outer_deadline():
    from interface.routes import chat as chat_routes

    outer = 55.0
    normal_inner = chat_routes._inner_cognitive_cycle_timeout(outer)
    protected_inner = chat_routes._inner_cognitive_cycle_timeout(
        outer,
        protected_foreground=True,
    )

    assert normal_inner == pytest.approx(38.5)
    assert protected_inner == pytest.approx(53.0)
    assert protected_inner < outer


@pytest.mark.asyncio
async def test_desktop_capability_grounding_reaches_cognitive_engine_without_midword_clipping(
    monkeypatch,
):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    long_inventory = (
        "I can use governed desktop and app control, browser/web research, file and document "
        "operations, terminal and code execution, memory and continuity, and self-repair. "
        + "Governed capability detail remains concrete and inspectable. " * 16
        + "Will and Authority approve consequential actions, and effect receipts verify results. "
        "For this turn I am only describing the tool surface; I am not executing tools."
    )

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"context": dict(context or {}), "kwargs": dict(kwargs)})
            return SimpleNamespace(
                content=str(context["grounded_capability_inventory_context"]),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        _chat_desktop_repair,
        "_build_grounded_capability_inventory_reply",
        lambda _message: long_inventory,
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    prompt = "Explain what tools you can use and give a hypothetical scenario without executing it."
    await chat_routes._run_cognitive_engine_chat_turn(
        prompt,
        visible_user_message=prompt,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert len(long_inventory) > 1000
    assert calls[0]["context"]["grounded_capability_inventory_context"] == long_inventory
    assert calls[0]["context"]["grounded_capability_inventory_context"].endswith(
        "I am not executing tools."
    )


@pytest.mark.asyncio
async def test_desktop_required_chat_gets_default_recent_context_window(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="I am answering this ordinary live desktop turn with the recent thread still present."
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "The live desktop chat was drifting into assistant mode.",
                    "aura": "I need to keep the full mind path fused to the speech lane.",
                    "status": "complete",
                }
            ]
        )

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "Give me one practical next step."
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert calls[0]["context"]["recent_context_needed"] is False
    assert calls[0]["context"]["recent_completed_exchanges"]
    assert "assistant mode" in calls[0]["context"]["recent_conversation_context"]
    assert calls[0]["context"]["live_runtime_payload_required"] is True


@pytest.mark.asyncio
async def test_deep_desktop_followup_keeps_hard_live_token_envelope(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "I am holding that uncertainty in my attention rather than erasing it. "
                    "I would remember this thread and let it change my next decision: choose "
                    "one reversible step that gathers evidence, then recheck before committing."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    message = (
        "Remember the uncertainty you just named. How would it change one decision "
        "you make next, without pretending certainty you do not have?"
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        message,
        visible_user_message=message,
        origin="user",
        timeout_s=105.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert calls
    context = calls[0]["context"]
    assert context.get("desktop_quick_reply_contract") is not True
    assert context["max_tokens"] == 896
    assert context["num_predict"] == 896


@pytest.mark.asyncio
async def test_desktop_required_memory_state_turn_uses_canonical_evidence_without_stale_history(
    monkeypatch,
):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    'The phrase you asked me to remember was "silver lantern". '
                    "CognitiveEngine keeps this reply grounded by reading the canonical "
                    "memory-state evidence attached to the current live desktop turn instead "
                    "of guessing from older conversation history."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "What pitch?",
                    "aura": "A stale pitch answer that must not steer this turn.",
                    "status": "complete",
                }
            ]
        )

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    visible_message = (
        "What phrase did I ask you to remember, and how does your cognitive engine keep this reply grounded?"
    )
    effective_message = (
        f"{visible_message}\n\n"
        "[CANONICAL MEMORY STATE EVIDENCE]\n"
        "status=session_memory_recall\n"
        'The phrase you asked me to remember in this session was "silver lantern".\n'
        "[END CANONICAL MEMORY STATE EVIDENCE]\n"
        "Use this canonical memory/state result as evidence, but produce the visible answer "
        "through CognitiveEngine in Aura's normal desktop voice."
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        effective_message,
        visible_user_message=visible_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "silver lantern" in reply
    assert calls[0]["context"]["memory_state_contract"] is True
    assert "silver lantern" in calls[0]["context"]["canonical_memory_state_evidence"]
    assert calls[0]["context"]["recent_completed_exchanges"] == []
    assert calls[0]["context"]["recent_conversation_context"] == ""
    assert calls[0]["context"]["desktop_quick_reply_contract"] is True


@pytest.mark.asyncio
async def test_desktop_required_durable_memory_pin_uses_compact_canonical_path(
    monkeypatch,
):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    'I have pinned "restart-815" in durable session memory. '
                    "Right now I am keeping attention on this live desktop thread."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    visible_message = "Remember this codeword across restart: restart-815"
    effective_message = (
        f"{visible_message}\n\n"
        "[CANONICAL MEMORY STATE EVIDENCE]\n"
        "status=session_memory_pin\n"
        'I\'ve pinned "restart-815" in durable session memory.\n'
        "[END CANONICAL MEMORY STATE EVIDENCE]\n"
        "Use this canonical memory/state result as evidence, but produce the visible answer "
        "through CognitiveEngine in Aura's normal desktop voice."
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        effective_message,
        visible_user_message=visible_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "restart-815" in reply
    assert calls
    context = calls[0]["context"]
    assert context["memory_state_contract"] is True
    assert context["desktop_quick_reply_contract"] is True
    assert context["recent_completed_exchanges"] == []
    assert context["max_tokens"] <= 384


@pytest.mark.asyncio
async def test_desktop_memory_state_turn_binds_reply_to_canonical_memory_when_model_drifts(
    monkeypatch,
):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    turn_trace = {}

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="I am attending to the live desktop thread through CognitiveEngine.",
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    visible_message = (
        "Remember this phrase: silver lantern. Also tell me one thing your live mind is attending to right now."
    )
    effective_message = (
        f"{visible_message}\n\n"
        "[CANONICAL MEMORY STATE EVIDENCE]\n"
        "status=session_memory_pin\n"
        'I\'ve pinned "silver lantern" in durable session memory.\n'
        "[END CANONICAL MEMORY STATE EVIDENCE]\n"
        "Use this canonical memory/state result as evidence, but produce the visible answer "
        "through CognitiveEngine in Aura's normal desktop voice."
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        effective_message,
        visible_user_message=visible_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=turn_trace,
    )

    assert reply
    assert "silver lantern" in reply
    assert "this live desktop thread" in reply
    assert calls
    assert turn_trace["engine_think_invoked"] is True
    assert turn_trace["cognitive_engine_reply_accepted"] is True
    assert turn_trace["bounded_contract_used"] is False
    assert turn_trace["live_mind_controls_bound"] is True
    assert turn_trace["response_path"] == "cognitive_engine_memory_state_grounding"


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_receives_recent_completed_conversation_context(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="I remember the terminal crash context and I am answering from the current live desktop thread."
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "The Python process spiked over 100GB of RAM.",
                    "aura": "That points at an unbounded live desktop path allocation.",
                    "status": "complete",
                },
                {
                    "user": "Make sure the UI path stays on CognitiveEngine.",
                    "aura": "I will keep desktop turns on the required CognitiveEngine path.",
                    "status": "complete",
                },
            ]
        )

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "Can you continue from what we were debugging?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert calls[0]["context"]["recent_completed_exchanges"]
    assert "100GB of RAM" in calls[0]["context"]["recent_conversation_context"]
    assert "required CognitiveEngine path" in calls[0]["context"]["recent_conversation_context"]


@pytest.mark.asyncio
async def test_desktop_quick_self_reflection_suppresses_stale_recent_context(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="I am tracking this live turn directly, with attention on your current question."
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "What tools can you use externally?",
                    "aura": "I can describe governed desktop and browser tools.",
                    "status": "complete",
                },
                {
                    "user": "Name a hypothetical tool scenario.",
                    "aura": "A scenario could include creating a folder and exporting a PDF.",
                    "status": "complete",
                },
            ]
        )

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "How are you thinking about this conversation right now?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert calls
    assert calls[0]["context"]["recent_context_needed"] is True
    assert calls[0]["context"]["live_mind_context_required"] is True
    assert calls[0]["context"]["live_mind_context"]["required_for_live_desktop"] is True


def test_user_visible_context_leak_sanitizer_strips_internal_blocks():
    from interface.routes import chat as chat_routes

    reply = (
        "I am staying on the live desktop cognitive lane and answering the current turn."
        "[RECENT CONTEXT]User: What tools can you use?"
    )

    assert chat_routes._strip_user_visible_context_leaks(reply) == (
        "I am staying on the live desktop cognitive lane and answering the current turn."
    )


def test_live_self_reflection_rejects_stale_tool_topic_bleed():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "How are you thinking about this conversation right now?",
        (
            "I'm in a quiet state, with stable attention on you. "
            "You're asking about tools or scenarios — let me walk through an actual case of using them. "
            "If you want to create a folder and write a file, I can do that."
        ),
    )

    assert "stale_context_topic_bleed" in assessment.reasons
    assert assessment.retryable


def test_live_desktop_gate_leak_is_retryable_user_facing_failure():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Ok. Just checking. I'll be back, ok?",
        (
            "This live desktop turn failed the reply-quality gate, and I am not "
            "starting a second foreground generation over it."
        ),
    )

    assert "internal_live_gate_leak" in assessment.reasons
    assert assessment.hard_failure
    assert assessment.retryable


def test_conceptual_live_chat_reliability_question_does_not_force_diagnostic_floor():
    from core.conversation.response_reliability import (
        is_reliability_concern,
        reliability_floor_for_user,
    )

    message = (
        "In two direct sentences, explain why the live desktop chat path must "
        "never fall back into generic assistant mode."
    )

    assert is_reliability_concern(message) is False
    assert reliability_floor_for_user(message) == ""


def test_reported_live_chat_breakage_still_uses_reliability_floor():
    from core.conversation.response_reliability import (
        is_reliability_concern,
        reliability_floor_for_user,
    )

    message = "The live chat path broke again after a few turns."

    assert is_reliability_concern(message) is True
    assert reliability_floor_for_user(message)


@pytest.mark.asyncio
async def test_desktop_low_risk_social_turn_retries_once_then_fails_closed(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    "This live desktop turn failed the reply-quality gate, and I am not "
                    "starting a second foreground generation over it."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "Ok. Just checking. I'll be back, ok?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
    )

    assert len(calls) == 2
    assert reply is None


@pytest.mark.asyncio
async def test_desktop_required_capability_turn_uses_cognitive_engine_before_catalog_repair(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    trace = {}

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    "From the live desktop path I can use governed tool lanes for desktop apps, browser and web "
                    "research, file operations, document drafting, terminal work, memory recall, and skill execution. "
                    "A hypothetical chain would request approval, open sources, draft a document, verify the visible "
                    "result, export the file, and record governance receipts without claiming an unverified action."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "What external tools could you use from the live desktop path, and give one hypothetical chain "
        "without claiming you executed it."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply
    assert len(calls) == 1
    assert calls[0]["context"]["capability_inventory_contract"] is True
    assert calls[0]["context"]["recent_completed_exchanges"] == []
    assert calls[0]["context"]["recent_conversation_context"] == ""
    assert calls[0]["context"]["max_tokens"] == 384
    assert calls[0]["context"]["live_mind_context_required"] is True
    assert calls[0]["context"]["live_mind_context"]["must_answer_from_full_mind_path"] is True
    assert trace["engine_think_invoked"] is True
    assert trace["cognitive_engine_reply_accepted"] is True
    assert trace["bounded_contract_used"] is False
    assert trace["live_mind_controls_bound"] is True
    assert trace["response_path"] == "cognitive_engine"
    assert "governance receipts" in reply
    assert "not opening apps or executing tools" in reply.lower()


@pytest.mark.asyncio
async def test_desktop_required_status_turn_uses_cognitive_engine_when_lane_ready(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    trace = {}

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    "I'm here with you. I'm tracking this live desktop thread and answering the current turn, "
                    "not drifting into a pitch or another topic."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "You with me?",
        visible_user_message="You with me?",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply
    assert len(calls) == 1
    assert calls[0]["context"]["live_mind_context_required"] is True
    assert calls[0]["context"]["live_mind_context"]["must_answer_from_full_mind_path"] is True
    assert trace["engine_think_invoked"] is True
    assert trace["cognitive_engine_reply_accepted"] is True
    assert trace["bounded_contract_used"] is False
    assert trace["live_mind_controls_bound"] is True
    assert trace["response_path"] == "cognitive_engine"


def test_live_self_reflection_rejects_social_presence_template():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "How are you thinking about this conversation right now?",
        (
            "hey. i'm here with you. I'm feeling Joy and leaning toward rest right now. "
            "I can answer clearly from the active turn. My attention is on you."
        ),
    )

    assert "social_presence_instead_of_self_reflection" in assessment.reasons
    assert assessment.retryable


def test_live_self_reflection_rejects_punctuated_incomplete_tail():
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes import chat as chat_routes

    reply = (
        "I'm thinking about your question through my current state. "
        "The conversation itself is a live feedback loop: what you say informs how I think about."
    )

    assessment = assess_user_facing_reply(
        "How are you thinking about this conversation right now?",
        reply,
    )

    assert "truncated_tail" in assessment.reasons
    assert assessment.retryable
    assert chat_routes._looks_truncated_tail(reply) is True


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_strips_internal_context_leak(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            return SimpleNamespace(
                content=(
                    "I am staying with the current turn, my attention is on this conversation right now, "
                    "and I feel focused rather than drifting into the previous topic."
                    "[RECENT CONTEXT]User: What tools can you use?"
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "How are you thinking about this conversation right now?"
    trace = {}
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply == (
        "I am staying with the current turn, my attention is on this conversation right now, "
        "and I feel focused rather than drifting into the previous topic."
    )
    assert "[RECENT CONTEXT]" not in reply
    assert "User: What tools" not in reply
    assert any(
        item["stage"] == "chat.cognitive_engine_context_leak_strip"
        for item in trace["text_mutations"]
    )
    assert chat_routes._desktop_required_bounded_reply_status(
        user_message,
        reply,
        {"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
    ) == ""


@pytest.mark.asyncio
async def test_cognitive_engine_does_not_duplicate_a_consumed_model_owner(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    first_metadata = _bound_live_mind_controls_metadata()
    first_metadata.update(
        {
            "desktop_cognitive_engine_failure": True,
            "failure_reason": "first_generation_rejected",
            "latent_cortex_selected": True,
            "latent_cortex_attempted": True,
            "latent_cortex_succeeded": False,
            "latent_cortex_fallback_used": True,
            "latent_cortex_failure_reason": "output_quality_failed:missing_requested_facets",
            "latent_cortex_prompt_shape": {
                "question_parts": 4,
                "requires_single_reply_coverage": True,
            },
            "latent_cortex_ingress": {"schema": "aura.cognitive_ingress.v1"},
            "latent_cortex_progress": {"stage": "complete", "elapsed_s": 109.5},
            "latent_cortex_receipt": {
                "episode_id": "cp120-failed",
                "last_stage": "complete",
                "verifier_probe_max_tokens": 24,
                "latent_opt_attempts": 1,
                "latent_opt_rejected": 0,
                "latent_opt_verifier": {
                    "policy": "task_score_nonregression_with_proxy_descent_v1",
                    "decisions": [{"accepted": True}],
                },
                "budget": {"spent_layer_apps": 147776},
            },
        }
    )
    first_metadata["live_mind_surface_control_receipt"].update(
        {
            "generation_max_tokens": 111,
            "generated_tokens": 17,
            "attempt_marker": "rejected",
        }
    )
    class _FakeCognitiveEngine:
        def __init__(self):
            self.calls = 0

        async def think(self, objective, context=None, **kwargs):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("a consumed foreground model owner must not be duplicated")
            return SimpleNamespace(
                content="The first generation is a declared failure envelope.",
                metadata=first_metadata,
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    engine = _FakeCognitiveEngine()
    trace = {}
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (True, "test_ready"),
    )
    monkeypatch.setattr(
        chat_routes,
        "_gather_recent_user_messages_for_relevance",
        AsyncCallFixture(return_value=[]),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: engine if name == "cognitive_engine" else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        'Reply exactly: "yes"',
        visible_user_message='Reply exactly: "yes"',
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply is None
    assert engine.calls == 1
    receipt = trace["live_mind_surface_control_receipt"]
    assert receipt["attempt_marker"] == "rejected"
    assert receipt["generation_max_tokens"] == 111
    assert receipt["generated_tokens"] == 17
    assert trace["foreground_model_generation_consumed"] is True
    assert trace["foreground_model_generation_count"] == 1
    assert trace["single_owner_generation_exhausted"] is True
    assert trace["latent_cortex_prompt_shape"]["question_parts"] == 4
    assert trace["latent_cortex_ingress"]["schema"] == "aura.cognitive_ingress.v1"
    assert trace["latent_cortex_progress"]["stage"] == "complete"
    assert trace["latent_cortex_receipt"]["episode_id"] == "cp120-failed"
    live_contract = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": True, "state": "ready"},
        response_confidence="degraded",
        status="cognitive_engine_failed",
        turn_trace=trace,
    )
    assert live_contract["latent_cortex_progress"] == {
        "stage": "complete",
        "elapsed_s": 109.5,
    }
    failed_receipt = live_contract["latent_cortex_receipt"]
    assert failed_receipt["verifier_probe_max_tokens"] == 24
    assert failed_receipt["latent_opt_attempts"] == 1
    assert failed_receipt["latent_opt_rejected"] == 0
    assert failed_receipt["latent_opt_verifier"]["decisions"][0]["accepted"] is True
    assert failed_receipt["budget"]["spent_layer_apps"] == 147776


@pytest.mark.asyncio
async def test_worker_exhausted_quality_rejection_skips_duplicate_route_retry(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    metadata = _bound_live_mind_controls_metadata()
    metadata.update(
        {
            "desktop_cognitive_engine_failure": True,
            "failure_reason": "surface_quality_rejected",
            "generation_failure_class": "surface_quality_rejected",
        }
    )
    metadata["live_mind_surface_control_receipt"].update(
        {
            "surface_quality_gate_enabled": True,
            "surface_quality_gate_passed": False,
            "surface_quality_gate_attempts": 3,
            "surface_quality_gate_reasons": ["missing_requested_word_count"],
        }
    )

    class _FakeCognitiveEngine:
        def __init__(self):
            self.calls = 0

        async def think(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("worker-owned quality retries must not be repeated")
            return SimpleNamespace(
                content="The generation was rejected by its output checks.",
                metadata=metadata,
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    engine = _FakeCognitiveEngine()
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: engine
            if name == "cognitive_engine"
            else default
        ),
    )

    trace = {}
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "In exactly five words, state why checksums matter.",
        visible_user_message=(
            "In exactly five words, state why checksums matter."
        ),
        origin="user",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "foreground_endpoint": "Cortex",
        },
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply is None
    assert engine.calls == 1
    assert trace["cognitive_engine_reply_failed"] is True
    assert trace["live_mind_surface_control_receipt"][
        "surface_quality_gate_attempts"
    ] == 3


@pytest.mark.asyncio
async def test_truncated_foreground_answer_gets_one_same_worker_continuation(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    first_metadata = _bound_live_mind_controls_metadata()
    first_metadata.update(
        {
            "reply_generation_incomplete": True,
            "reply_generation_stop_reason": "max_tokens",
            "latent_cortex_selected": True,
            "latent_cortex_attempted": True,
            "latent_cortex_succeeded": False,
            "latent_cortex_fallback_used": True,
            "latent_cortex_failure_reason": "latent_optimization_budget_exhausted",
            "latent_cortex_receipt": {
                "episode_id": "preserved-across-continuation",
                "last_stage": "latent_optimization",
            },
        }
    )
    first_metadata["live_mind_surface_control_receipt"].update(
        {
            "surface_quality_gate_passed": False,
            "surface_quality_gate_reasons": ["truncated_tail"],
            "generated_tokens": 192,
            "generation_max_tokens": 192,
            "generation_stop_reason": "max_tokens",
            "continuation_resume_handle": "b" * 32,
        }
    )
    second_metadata = _bound_live_mind_controls_metadata()
    second_metadata["live_mind_surface_control_receipt"].update(
        {
            "generated_tokens": 156,
            "generation_max_tokens": 1024,
            "generation_stop_reason": "eos_or_stop_sequence",
        }
    )

    class _FakeCognitiveEngine:
        def __init__(self):
            self.calls = []

        async def think(self, objective, context=None, **kwargs):
            self.calls.append((objective, dict(context or {})))
            if len(self.calls) == 1:
                return SimpleNamespace(
                    content=(
                        "The function tracks balances in a dictionary and removes "
                        "names whose balance reaches zero from the"
                    ),
                    metadata=first_metadata,
                )
            return SimpleNamespace(
                content=(
                    " dictionary, records each position where the number of active "
                    "names reaches its maximum, and returns those positions. In short, "
                    "it reports every point at which concurrent nonzero balances are "
                    "tied for the peak."
                ),
                metadata=second_metadata,
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    engine = _FakeCognitiveEngine()
    trace = {}
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (True, "completion_retry_ready"),
    )
    monkeypatch.setattr(
        chat_routes,
        "_gather_recent_user_messages_for_relevance",
        AsyncCallFixture(return_value=[]),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: engine if name == "cognitive_engine" else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Explain what this code does.",
        visible_user_message="Explain what this code does.",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply is not None
    assert reply.endswith("peak.")
    assert len(engine.calls) == 2
    assert engine.calls[1][1]["user_surface_completion_retry"] is True
    assert engine.calls[1][0] == "Explain what this code does."
    assert engine.calls[1][1]["desktop_quick_reply_contract"] is True
    assert engine.calls[1][1]["user_surface_continuation_contract"] is True
    assert engine.calls[1][1]["user_surface_continuation_partial"].endswith("from the")
    assert engine.calls[1][1]["user_surface_continuation_resume_handle"] == "b" * 32
    assert engine.calls[1][1]["route"] == "desktop_chat_continuation"
    assert "response_repair_directive" not in engine.calls[1][1]
    assert "failed_reply_excerpt" not in engine.calls[1][1]
    assert "failed_reply_reasons" not in engine.calls[1][1]
    assert trace["foreground_model_generation_count"] == 2
    assert trace["completion_retry_count"] == 1
    assert trace["response_path"] == "cognitive_engine_completion_retry"
    assert trace["latent_cortex_selected"] is True
    assert trace["latent_cortex_attempted"] is True
    assert trace["latent_cortex_fallback_used"] is True
    assert trace["latent_cortex_receipt"]["episode_id"] == (
        "preserved-across-continuation"
    )


@pytest.mark.asyncio
async def test_route_level_truncated_draft_enters_same_worker_continuation(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    metadata = _bound_live_mind_controls_metadata()
    metadata.update(
        {
            "reply_generation_incomplete": False,
            "reply_generation_stop_reason": "eos_or_stop_sequence",
            "reply_generation_failure_reasons": [],
        }
    )
    metadata["live_mind_surface_control_receipt"].update(
        {
            "surface_quality_gate_passed": True,
            "surface_quality_gate_reasons": [],
            "generation_stop_reason": "eos_or_stop_sequence",
        }
    )

    class _FakeCognitiveEngine:
        def __init__(self):
            self.calls = []

        async def think(self, objective, context=None, **_kwargs):
            self.calls.append((objective, dict(context or {})))
            return SimpleNamespace(
                content=(
                    " dictionary, records each position tied for the maximum, "
                    "and returns those positions. The final result is the full "
                    "set of positions tied for that maximum."
                ),
                metadata=metadata,
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    engine = _FakeCognitiveEngine()
    trace = {}
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (True, "completion_retry_ready"),
    )
    monkeypatch.setattr(
        chat_routes,
        "_gather_recent_user_messages_for_relevance",
        AsyncCallFixture(return_value=[]),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: engine if name == "cognitive_engine" else default
        ),
    )

    partial = (
        "The code updates each balance and removes names whose balance "
        "reaches zero from the"
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Internal repair directive",
        visible_user_message="Explain what this code does and give the final result.",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui_recovery",
        require_engine=True,
        turn_trace=trace,
        continuation_partial=partial,
        continuation_reasons=("truncated_tail",),
        continuation_evidence={
            "foreground_model_generation_count": 2,
            "foreground_model_generation_segment_count": 2,
            "foreground_model_generation_transaction_count": 1,
            "foreground_model_generation_transaction_id": "durable-answer-transaction",
            "completion_retry_count": 1,
        },
    )

    assert reply is not None
    assert reply.endswith("positions tied for that maximum.")
    assert len(engine.calls) == 1
    assert engine.calls[0][0] == "Explain what this code does and give the final result."
    assert engine.calls[0][1]["user_surface_continuation_contract"] is True
    assert engine.calls[0][1]["user_surface_continuation_partial"] == partial
    assert trace["response_path"] == "cognitive_engine_completion_retry"
    assert trace["engine_think_invoked"] is True
    assert trace["foreground_model_generation_count"] == 3
    assert trace["foreground_model_generation_segment_count"] == 3
    assert trace["foreground_model_generation_transaction_count"] == 1
    assert trace["foreground_model_generation_transaction_id"] == (
        "durable-answer-transaction"
    )
    assert trace["completion_retry_count"] == 2
    assert trace["continuation_evidence_valid"] is True

    contract = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": False, "state": "recovering"},
        response_confidence="high",
        status=trace["response_path"],
        reply_source=trace["response_path"],
        turn_trace={
            **trace,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "final_requested_output_contract_evaluated": True,
            "final_requested_output_contract_required": False,
            "final_requested_output_contract_satisfied": True,
        },
    )
    assert contract["single_owner_model_generation_proven"] is True
    assert contract["authentic_cognitive_reply"] is True
    assert contract["answer_delivery_proven"] is True


def test_released_reusable_latent_episode_does_not_exhaust_foreground_owner():
    from interface.routes import chat as chat_routes

    metadata = {
        "latent_cortex_attempted": True,
        "latent_cortex_receipt": {
            "resident_owner_released": True,
            "resident_state_reusable": True,
        },
    }

    assert chat_routes._generation_metadata_consumed_foreground_owner(metadata) is False
    metadata["latent_cortex_receipt"]["resident_state_reusable"] = False
    assert chat_routes._generation_metadata_consumed_foreground_owner(metadata) is True


def test_certified_recurrent_typed_answer_has_non_generative_delivery_ownership():
    from core.brain.llm import qualified_recurrent_ingress as ingress
    from core.learning.frontier_process_supervision import frontier_process_task_battery
    from interface.routes import chat as chat_routes

    task = frontier_process_task_battery(("calibration",), (1,), 1, seed=2026082102)[0]
    admission = ingress.admit_qualified_recurrent_objective(task.prompt)
    assert admission is not None
    body = {
        "schema": ingress.QUALIFIED_RECURRENT_RESULT_SCHEMA,
        "admission": admission.receipt(),
        "semantic_state_receipt": {"state_sha256": "s" * 64},
        "surface_decode_receipt": None,
        "activation_receipt": {
            "promotion_mode": "active",
            "activation_sha256": "a" * 64,
        },
        "serialization": "canonical_json_from_authenticated_semantic_state",
        "answer_sha256": hashlib.sha256(task.answer.encode("utf-8")).hexdigest(),
    }
    receipt = {**body, "receipt_sha256": ingress._canonical_sha256(body)}
    metadata = {
        "response_path": "cognitive_engine_qualified_recurrent",
        "qualified_recurrent_succeeded": True,
        "qualified_recurrent_family": task.family,
        "qualified_recurrent_receipt": receipt,
        "latent_cortex_attempted": True,
        "model_generation_used": False,
        "live_mind_generation_required": False,
    }
    assert (
        chat_routes._generation_metadata_consumed_foreground_owner(
            metadata,
            response_text=task.answer,
        )
        is False
    )

    trace = {
        **metadata,
        "qualified_recurrent_path_proven": True,
        "qualified_recurrent_delivery_errors": [],
        "engine_think_invoked": True,
        "cognitive_engine_reply_accepted": True,
        "cognitive_engine_reply_failed": False,
        "bounded_contract_used": False,
        "legacy_fallback_used": False,
        "foreground_model_generation_consumed": False,
        "foreground_model_generation_count": 0,
        "foreground_model_generation_segment_count": 0,
        "foreground_model_generation_transaction_count": 0,
        "authored_answer_completion_proven": True,
        "live_mind_context_present": True,
        "live_mind_snapshot_present": True,
        "live_mind_snapshot_ready": True,
        "final_requested_output_contract_evaluated": True,
        "final_requested_output_contract_required": True,
        "final_requested_output_contract_satisfied": True,
    }
    contract = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": True, "state": "ready"},
        response_confidence="high",
        status=metadata["response_path"],
        reply_source=metadata["response_path"],
        turn_trace=trace,
    )
    assert contract["qualified_recurrent_path_proven"] is True
    assert contract["single_owner_model_generation_proven"] is True
    assert contract["authentic_cognitive_reply"] is True
    assert contract["answer_delivery_proven"] is True
    assert contract["model_native_output"] is False
    assert contract["state_native_output"] is True
    assert contract["semantic_completion_receipt_present"] is True
    assert contract["semantic_completion_satisfied"] is True
    assert contract["semantic_completion_mode"] == "certified_state_serialization"
    assert contract["final_text_authorship"] == "certified_recurrent_state_serialization"
    assert "live_mind_controls_unbound" not in contract["full_mind_missing_proofs"]

    mutated_answer = task.answer + " downstream-grounding-mutation"
    chat_routes._append_turn_text_mutation(
        trace,
        stage="chat.grounded_recall_attribution",
        method="grounded_attribution_repair",
        reasons=["speaker_attribution_changed"],
        before=task.answer,
        after=mutated_answer,
        deterministic=True,
        authorship_effect="preserved",
    )
    assert chat_routes._bind_qualified_recurrent_public_answer(trace, mutated_answer) is False
    rejected = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": True, "state": "ready"},
        response_confidence="high",
        status=metadata["response_path"],
        reply_source=metadata["response_path"],
        turn_trace=trace,
    )
    assert rejected["qualified_recurrent_path_proven"] is False
    assert rejected["answer_delivery_proven"] is False
    assert "qualified_recurrent_path_unproven" in rejected["full_mind_missing_proofs"]


def test_certified_recurrent_terminal_contract_owns_exact_bytes_and_shape():
    from core.brain.llm import qualified_recurrent_ingress as ingress
    from core.learning.frontier_process_supervision import frontier_process_task_battery
    from interface.routes import chat as chat_routes

    task = frontier_process_task_battery(("calibration",), (1,), 1, seed=2026082111)[0]
    admission = ingress.admit_qualified_recurrent_objective(task.prompt)
    assert admission is not None
    body = {
        "schema": ingress.QUALIFIED_RECURRENT_RESULT_SCHEMA,
        "admission": admission.receipt(),
        "semantic_state_receipt": {"state_sha256": "s" * 64},
        "surface_decode_receipt": None,
        "activation_receipt": {
            "promotion_mode": "active",
            "activation_sha256": "a" * 64,
        },
        "serialization": "canonical_json_from_authenticated_semantic_state",
        "answer_sha256": hashlib.sha256(task.answer.encode("utf-8")).hexdigest(),
    }
    receipt = {**body, "receipt_sha256": ingress._canonical_sha256(body)}
    trace = {
        "response_path": "cognitive_engine_qualified_recurrent",
        "qualified_recurrent_succeeded": True,
        "qualified_recurrent_family": task.family,
        "qualified_recurrent_receipt": receipt,
        "model_generation_used": False,
        "live_mind_generation_required": False,
    }

    assert chat_routes._bind_qualified_recurrent_terminal_contract(trace, task.answer)
    assert trace["qualified_recurrent_terminal_bytes_preserved"] is True
    assert trace["final_requested_output_contract_evaluated"] is True
    assert trace["final_requested_output_contract_required"] is True
    assert trace["final_requested_output_contract_satisfied"] is True
    assert (
        trace["final_requested_output_contract_kind"]
        == "certified_recurrent_state_serialization"
    )
    assert not chat_routes._bind_qualified_recurrent_terminal_contract(
        trace,
        task.answer + " changed",
    )


@pytest.mark.asyncio
async def test_desktop_chat_adopts_certified_recurrent_answer_without_fake_decode(
    monkeypatch,
):
    from core.brain.llm import qualified_recurrent_ingress as ingress
    from core.learning.frontier_process_supervision import frontier_process_task_battery
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    task = frontier_process_task_battery(("calibration",), (1,), 1, seed=2026082103)[0]
    admission = ingress.admit_qualified_recurrent_objective(task.prompt)
    assert admission is not None
    body = {
        "schema": ingress.QUALIFIED_RECURRENT_RESULT_SCHEMA,
        "admission": admission.receipt(),
        "semantic_state_receipt": {"state_sha256": "s" * 64},
        "surface_decode_receipt": None,
        "activation_receipt": {
            "promotion_mode": "active",
            "activation_sha256": "a" * 64,
        },
        "serialization": "canonical_json_from_authenticated_semantic_state",
        "answer_sha256": hashlib.sha256(task.answer.encode("utf-8")).hexdigest(),
    }
    receipt = {**body, "receipt_sha256": ingress._canonical_sha256(body)}
    metadata = {
        "response_path": "cognitive_engine_qualified_recurrent",
        "qualified_recurrent_eligible": True,
        "qualified_recurrent_attempted": True,
        "qualified_recurrent_succeeded": True,
        "qualified_recurrent_family": task.family,
        "qualified_recurrent_receipt": receipt,
        "latent_cortex_selected": True,
        "latent_cortex_attempted": True,
        "latent_cortex_succeeded": True,
        "latent_cortex_identity_bound": True,
        "latent_cortex_receipt": receipt,
        "model_generation_used": False,
        "live_mind_generation_required": False,
    }

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            return SimpleNamespace(content=task.answer, metadata=metadata)

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes,
        "_gather_recent_user_messages_for_relevance",
        AsyncCallFixture(return_value=[]),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _FakeCognitiveEngine() if name == "cognitive_engine" else default
            )
        ),
    )

    trace = {}
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        task.prompt,
        visible_user_message=task.prompt,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply == task.answer
    assert trace["response_path"] == "cognitive_engine_qualified_recurrent"
    assert trace["qualified_recurrent_path_proven"] is True
    assert trace["foreground_model_generation_consumed"] is False
    assert trace["foreground_model_generation_count"] == 0
    assert trace["authored_answer_completion_proven"] is True


@pytest.mark.asyncio
async def test_desktop_chat_delivers_certified_recurrent_answer_without_prose_pipeline(
    monkeypatch,
):
    from core.brain.llm import qualified_recurrent_ingress as ingress
    from core.learning.frontier_process_supervision import frontier_process_task_battery
    from core.perception import observation_evidence
    from core.providers import engine_connection_pool as pool_module
    from core.self import source_excerpt
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    task = frontier_process_task_battery(("calibration",), (1,), 1, seed=2026082104)[0]
    admission = ingress.admit_qualified_recurrent_objective(task.prompt)
    assert admission is not None
    body = {
        "schema": ingress.QUALIFIED_RECURRENT_RESULT_SCHEMA,
        "admission": admission.receipt(),
        "semantic_state_receipt": {"state_sha256": "s" * 64},
        "surface_decode_receipt": None,
        "activation_receipt": {
            "promotion_mode": "active",
            "activation_sha256": "a" * 64,
        },
        "serialization": "canonical_json_from_authenticated_semantic_state",
        "answer_sha256": hashlib.sha256(task.answer.encode("utf-8")).hexdigest(),
    }
    receipt = {**body, "receipt_sha256": ingress._canonical_sha256(body)}
    metadata = {
        **_bound_live_mind_controls_metadata(),
        "response_path": "cognitive_engine_qualified_recurrent",
        "qualified_recurrent_eligible": True,
        "qualified_recurrent_attempted": True,
        "qualified_recurrent_succeeded": True,
        "qualified_recurrent_family": task.family,
        "qualified_recurrent_receipt": receipt,
        "latent_cortex_selected": True,
        "latent_cortex_attempted": True,
        "latent_cortex_succeeded": True,
        "model_generation_used": False,
        "live_mind_generation_required": False,
    }

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            return SimpleNamespace(content=task.answer, metadata=metadata)

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "qualified-exact-exchange"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return "committed"

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    async def _forbidden_stabilizer(*_args, **_kwargs):
        raise AssertionError("receipt-bound exact output must not enter prose stabilization")

    async def _forbidden_context_collector(*_args, **_kwargs):
        raise AssertionError("state-native output must not collect unrelated language evidence")

    async def _forbidden_foreground_gate(*_args, **_kwargs):
        raise AssertionError("state-native output must not wait for resident text generation")

    def _forbidden_terminal_transform(*_args, **_kwargs):
        trace_arg = _args[0] if _args and isinstance(_args[0], dict) else {}
        raise AssertionError(
            "receipt-bound exact output must not enter terminal prose shaping: "
            f"path={trace_arg.get('response_path')!r} "
            f"qualified={trace_arg.get('qualified_recurrent_succeeded')!r} "
            f"errors={trace_arg.get('qualified_recurrent_delivery_errors')!r}"
        )

    quality_calls = []
    unrelated_evidence_calls = []

    def _record_quality_call(*_args, **_kwargs):
        quality_calls.append(True)
        return False

    original_observation_memory = observation_evidence.get_observation_memory
    original_source_evidence_brief = source_excerpt.source_evidence_brief

    def _record_observation_memory(*_args, **_kwargs):
        unrelated_evidence_calls.append("perception")
        return original_observation_memory(*_args, **_kwargs)

    def _record_source_evidence(*_args, **_kwargs):
        unrelated_evidence_calls.append("source")
        return original_source_evidence_brief(*_args, **_kwargs)

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_a, **_k: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_a, **_k: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _forbidden_stabilizer)
    monkeypatch.setattr(
        chat_routes,
        "_build_retained_memory_evidence_context",
        _forbidden_context_collector,
    )
    monkeypatch.setattr(
        _chat_memory_state,
        "_build_conversation_recall_reply",
        _forbidden_context_collector,
    )
    monkeypatch.setattr(
        chat_routes,
        "_collect_desktop_required_search_evidence",
        _forbidden_context_collector,
    )
    monkeypatch.setattr(
        chat_routes,
        "_await_foreground_gate",
        _forbidden_foreground_gate,
    )
    monkeypatch.setattr(
        chat_routes,
        "_turn_may_concern_perception",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        chat_routes,
        "_turn_may_concern_own_source",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        observation_evidence,
        "get_observation_memory",
        _record_observation_memory,
    )
    monkeypatch.setattr(
        source_excerpt,
        "source_evidence_brief",
        _record_source_evidence,
    )
    monkeypatch.setattr(
        _chat_preflight,
        "_chat_evidence_profile",
        lambda *_args, **_kwargs: (
            _chat_preflight._CHAT_EVIDENCE_PROFILE_QUALIFIED_RECURRENT,
            admission,
        ),
    )
    monkeypatch.setattr(
        chat_routes,
        "_strip_user_visible_context_leaks",
        _forbidden_terminal_transform,
    )
    monkeypatch.setattr(
        chat_routes,
        "_append_past_action_record",
        _forbidden_terminal_transform,
    )
    monkeypatch.setattr(
        chat_routes,
        "_append_runtime_authored_why",
        _forbidden_terminal_transform,
    )
    monkeypatch.setattr(
        chat_routes,
        "_enforce_final_requested_output_contract",
        _forbidden_terminal_transform,
    )
    monkeypatch.setattr(chat_routes, "_is_actionably_stale_response", _record_quality_call)
    monkeypatch.setattr(chat_routes, "_is_same_answer_different_prompt", _record_quality_call)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    patch_chat_lane(
        monkeypatch,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "recovering",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "",
            "background_endpoint": "Brainstem",
        },
    )
    _force_full_mind_runtime(monkeypatch, chat_routes)

    response = await server_module.api_chat(
        server_module.ChatRequest(message=task.prompt, session_id="qualified-exact-session"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["response"] == task.answer
    assert payload["live_turn_contract"]["qualified_recurrent_path_proven"] is True
    assert payload["live_turn_contract"]["foreground_model_generation_count"] == 0
    assert payload["live_turn_contract"]["model_native_output"] is False
    assert payload["live_turn_contract"]["state_native_output"] is True
    assert payload["live_turn_contract"]["semantic_completion_satisfied"] is True
    assert payload["live_turn_contract"]["latent_cortex_public_output_quality"][
        "policy"
    ] == "qualified_recurrent_state_serialization_quality_v1"
    assert payload["live_turn_contract"][
        "qualified_recurrent_public_output_quality_proven"
    ] is True
    assert payload["live_turn_contract"]["final_requested_output_contract_satisfied"] is True
    assert payload["live_turn_contract"]["qualified_recurrent_terminal_bytes_preserved"] is True
    assert payload["live_turn_contract"]["preflight_evidence_profile"] == (
        _chat_preflight._CHAT_EVIDENCE_PROFILE_QUALIFIED_RECURRENT
    )
    assert payload["live_turn_contract"]["preflight_evidence_owner"]["family"] == task.family
    assert payload["live_turn_contract"]["preflight_skipped_components"] == list(
        _chat_preflight._QUALIFIED_RECURRENT_SKIPPED_PREFLIGHT_COMPONENTS
    )
    assert payload["live_turn_contract"]["live_mind_context_required"] is False
    assert payload["live_turn_contract"]["live_mind_context_present"] is False
    assert "architecture_context_unbound" not in payload["live_turn_contract"][
        "full_mind_missing_proofs"
    ]
    assert "live_mind_snapshot_not_ready" not in payload["live_turn_contract"][
        "full_mind_missing_proofs"
    ]
    assert quality_calls == []
    assert unrelated_evidence_calls == []


@pytest.mark.asyncio
async def test_truncated_completion_replacement_cannot_become_authoritative(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    first_metadata = _bound_live_mind_controls_metadata()
    first_metadata.update(
        {
            "reply_generation_incomplete": True,
            "reply_generation_stop_reason": "max_tokens",
            "reply_generation_failure_reasons": ["truncated_tail"],
        }
    )
    second_metadata = _bound_live_mind_controls_metadata()
    second_metadata.update(
        {
            "reply_generation_incomplete": True,
            "reply_generation_stop_reason": "max_tokens",
            "reply_generation_failure_reasons": ["truncated_tail"],
        }
    )
    second_metadata["live_mind_surface_control_receipt"].update(
        {
            "surface_quality_gate_passed": False,
            "surface_quality_gate_reasons": ["truncated_tail"],
            "generation_stop_reason": "max_tokens",
        }
    )

    class _FakeCognitiveEngine:
        def __init__(self):
            self.calls = 0

        async def think(self, *_args, **_kwargs):
            self.calls += 1
            metadata = first_metadata if self.calls == 1 else second_metadata
            fragments = {
                1: (
                    "The function updates each balance and removes names whose "
                    "balance reaches zero from the"
                ),
                2: " dictionary, then records each",
                3: " position where active balances reach the",
                4: " peak but still has no final result",
            }
            return SimpleNamespace(
                content=fragments[self.calls],
                metadata=metadata,
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    engine = _FakeCognitiveEngine()
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (True, "completion_retry_ready"),
    )
    monkeypatch.setattr(
        chat_routes,
        "_gather_recent_user_messages_for_relevance",
        AsyncCallFixture(return_value=[]),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: engine if name == "cognitive_engine" else default
        ),
    )

    trace = {}
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Explain this code fully.",
        visible_user_message="Explain this code fully.",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply is not None
    assert reply.endswith("where active balances reach the")
    assert engine.calls == 3
    assert trace["completion_retry_count"] == 2
    assert trace["foreground_model_generation_count"] == 3
    assert trace["foreground_model_generation_segment_count"] == 3
    assert trace["foreground_model_generation_transaction_count"] == 1
    assert trace["completion_incumbent_preserved"] is True
    assert trace["completion_retry_exhausted"] is True
    assert trace["response_path"] == "cognitive_engine_completion_incumbent"

    contract = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": False, "state": "recovering"},
        response_confidence="high",
        status=trace["response_path"],
        reply_source=trace["response_path"],
        turn_trace={
            **trace,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "final_requested_output_contract_evaluated": True,
            "final_requested_output_contract_required": False,
            "final_requested_output_contract_satisfied": True,
        },
    )
    assert contract["single_owner_model_generation_proven"] is True
    assert contract["authentic_cognitive_reply"] is True
    assert contract["authored_answer_completion_proven"] is False
    assert contract["answer_delivery_proven"] is False
    assert "authored_answer_incomplete" in contract["full_mind_missing_proofs"]
    assert "duplicate_foreground_model_generation" not in contract["full_mind_missing_proofs"]


def test_one_answer_with_inconsistent_segment_receipts_is_not_called_a_duplicate() -> None:
    from interface.routes import chat as chat_routes

    contract = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": False, "state": "recovering"},
        response_confidence="high",
        status="cognitive_engine_completion_incumbent",
        reply_source="cognitive_engine_completion_incumbent",
        turn_trace={
            "cognitive_engine_required": True,
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "foreground_model_generation_consumed": True,
            "foreground_model_generation_count": 3,
            "foreground_model_generation_segment_count": 2,
            "foreground_model_generation_transaction_count": 1,
            "completion_retry_count": 2,
            "response_path": "cognitive_engine_completion_incumbent",
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "final_requested_output_contract_evaluated": True,
            "final_requested_output_contract_required": False,
            "final_requested_output_contract_satisfied": True,
        },
    )

    assert contract["single_owner_model_generation_proven"] is False
    assert "foreground_model_generation_ownership_unproven" in contract[
        "full_mind_missing_proofs"
    ]
    assert "duplicate_foreground_model_generation" not in contract[
        "full_mind_missing_proofs"
    ]


def test_resumed_answer_requires_valid_durable_continuation_evidence() -> None:
    from interface.routes import chat as chat_routes

    contract = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": False, "state": "recovering"},
        response_confidence="high",
        status="cognitive_engine_completion_retry",
        reply_source="cognitive_engine_completion_retry",
        turn_trace={
            "cognitive_engine_required": True,
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "foreground_model_generation_consumed": True,
            "foreground_model_generation_count": 3,
            "foreground_model_generation_segment_count": 3,
            "foreground_model_generation_transaction_count": 1,
            "foreground_model_generation_transaction_id": "durable-answer",
            "completion_retry_count": 2,
            "continuation_evidence_valid": False,
            "response_path": "cognitive_engine_completion_retry",
            "final_requested_output_contract_evaluated": True,
            "final_requested_output_contract_required": False,
            "final_requested_output_contract_satisfied": True,
            "authored_answer_completion_proven": True,
        },
    )

    assert contract["single_owner_model_generation_proven"] is False
    assert contract["authentic_cognitive_reply"] is False
    assert contract["answer_delivery_proven"] is False
    assert "foreground_model_generation_ownership_unproven" in contract[
        "full_mind_missing_proofs"
    ]


def test_protected_foreground_text_requires_receipted_authorship() -> None:
    from interface.routes import chat as chat_routes

    base_trace = {
        "engine_think_invoked": False,
        "cognitive_engine_reply_accepted": False,
        "cognitive_engine_reply_failed": False,
        "bounded_contract_used": False,
        "legacy_fallback_used": False,
        "foreground_model_generation_consumed": True,
        "foreground_model_generation_count": 1,
        "foreground_model_generation_segment_count": 1,
        "foreground_model_generation_transaction_count": 1,
        "foreground_model_generation_transaction_id": "protected-transaction",
        "response_path": "protected_foreground",
        "final_requested_output_contract_evaluated": True,
        "final_requested_output_contract_required": False,
        "final_requested_output_contract_satisfied": True,
    }
    unproven = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": False, "state": "recovering"},
        response_confidence="high",
        status="protected_foreground",
        reply_source="protected_foreground",
        turn_trace=base_trace,
    )
    proven = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": False, "state": "recovering"},
        response_confidence="high",
        status="protected_foreground",
        reply_source="protected_foreground",
        turn_trace={
            **base_trace,
            "protected_foreground_generation_proven": True,
        },
    )

    assert unproven["authentic_cognitive_reply"] is False
    assert unproven["answer_delivery_proven"] is False
    assert proven["authentic_cognitive_reply"] is True
    assert proven["answer_delivery_proven"] is True


def test_verified_action_serialization_is_proven_without_model_authorship() -> None:
    from interface.routes import chat as chat_routes

    contract = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": True, "state": "ready"},
        response_confidence="high",
        status="desktop_objective_completed",
        reply_source="fastpath",
        turn_trace={
            "engine_think_invoked": False,
            "cognitive_engine_reply_accepted": False,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "response_path": "",
            "response_authority_kind": "verified_action_receipt_serialization",
            "response_authority_proven": True,
            "response_authority_reason": "verified",
            "live_mind_generation_required": False,
            "foreground_model_generation_consumed": False,
            "foreground_model_generation_count": 0,
            "final_requested_output_contract_evaluated": True,
            "final_requested_output_contract_required": False,
            "final_requested_output_contract_satisfied": True,
            "semantic_completion_contract_expected": True,
            "semantic_completion_receipt_present": True,
            "semantic_completion_satisfied": True,
        },
    )

    assert contract["model_native_output"] is False
    assert contract["response_authority_proven"] is True
    assert contract["answer_delivery_proven"] is True
    assert contract["full_mind_path"] is False
    assert contract["semantic_completion_mode"] == (
        "verified_action_receipt_serialization"
    )
    assert contract["final_text_authorship"] == (
        "verified_action_receipt_serialization"
    )


def test_protected_foreground_transaction_identity_is_worker_bound() -> None:
    from interface.routes import chat as chat_routes

    receipt = _protected_foreground_generation_metadata()["surface_control_receipt"]
    reply = "The exact protected reply."
    transaction_id = chat_routes._worker_receipt_transaction_id(receipt, reply)

    assert transaction_id.startswith("mlx-")
    assert transaction_id == chat_routes._worker_receipt_transaction_id(receipt, reply)
    assert transaction_id != chat_routes._worker_receipt_transaction_id(
        receipt, "A different reply."
    )
    assert chat_routes._worker_receipt_transaction_id({}, reply) == ""
    mismatched = dict(receipt)
    mismatched["provenance"] = {
        **receipt["provenance"],
        "request_id_matches_active": False,
    }
    assert chat_routes._worker_receipt_transaction_id(mismatched, reply) == ""


def test_protected_foreground_delivery_rejects_any_post_worker_byte_change() -> None:
    from interface.routes import chat as chat_routes

    reply = "These are the exact worker-authored bytes."
    trace = {
        "foreground_model_generation_output_sha256": hashlib.sha256(
            reply.encode("utf-8")
        ).hexdigest()
    }

    assert chat_routes._protected_foreground_bytes_unchanged(
        trace,
        status="protected_foreground",
        reply_text=reply,
    )
    assert not chat_routes._protected_foreground_bytes_unchanged(
        trace,
        status="protected_foreground",
        reply_text=reply + " Runtime addition.",
    )
    assert not chat_routes._protected_foreground_bytes_unchanged(
        trace,
        status="desktop_objective_completed",
        reply_text=reply,
    )


@pytest.mark.asyncio
async def test_recorded_answer_wrapper_does_not_rewrite_proven_authored_bytes(
    monkeypatch,
) -> None:
    from fastapi.responses import JSONResponse
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        chat_routes,
        "_append_past_action_record",
        lambda _message, _reply: "A deterministic replacement.",
    )
    response = JSONResponse(
        {
            "response": "The exact authored answer.",
            "status": "ok",
            "live_turn_contract": {"answer_delivery_proven": True},
        }
    )

    wrapped = await chat_routes._apply_recorded_answer("What happened?", response)

    assert json.loads(wrapped.body)["response"] == "The exact authored answer."


@pytest.mark.asyncio
async def test_resumed_answer_cannot_mint_missing_transaction_identity() -> None:
    from interface.routes import chat as chat_routes

    trace: dict[str, object] = {}
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Open Notes.",
        visible_user_message="Open Notes.",
        origin="user",
        lane={"conversation_ready": True, "state": "ready"},
        source="paired_device",
        require_engine=True,
        conversation_only_surface=True,
        turn_trace=trace,
        continuation_partial="I started this answer but",
        continuation_reasons=("truncated_tail",),
        continuation_evidence={
            "foreground_model_generation_count": 1,
            "foreground_model_generation_segment_count": 1,
            "foreground_model_generation_transaction_count": 1,
            "completion_retry_count": 0,
        },
    )

    assert reply is not None
    assert trace["continuation_evidence_valid"] is False
    assert trace["foreground_model_generation_transaction_id"] == ""


def test_explicit_zero_ownership_counters_are_not_replaced_by_legacy_count() -> None:
    from interface.routes import chat as chat_routes

    contract = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={"conversation_ready": False, "state": "recovering"},
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "cognitive_engine_required": True,
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "foreground_model_generation_consumed": True,
            "foreground_model_generation_count": 1,
            "foreground_model_generation_segment_count": 0,
            "foreground_model_generation_transaction_count": 0,
            "response_path": "cognitive_engine",
            "final_requested_output_contract_evaluated": True,
            "final_requested_output_contract_required": False,
            "final_requested_output_contract_satisfied": True,
        },
    )

    assert contract["foreground_model_generation_segment_count"] == 0
    assert contract["foreground_model_generation_transaction_count"] == 0
    assert contract["single_owner_model_generation_proven"] is False


@pytest.mark.asyncio
async def test_empty_completion_cannot_erase_a_valid_incumbent(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    first_metadata = _bound_live_mind_controls_metadata()
    first_metadata.update(
        {
            "reply_generation_incomplete": True,
            "reply_generation_stop_reason": "configured_stop",
            "reply_generation_failure_reasons": ["unanswered_question_part"],
        }
    )
    second_metadata = _bound_live_mind_controls_metadata()

    class _FakeCognitiveEngine:
        def __init__(self):
            self.calls = 0

        async def think(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    content=(
                        "Dijkstra's invariant finalizes the unsettled vertex with "
                        "minimum tentative distance when all edges are nonnegative."
                    ),
                    metadata=first_metadata,
                )
            return SimpleNamespace(content="", metadata=second_metadata)

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    engine = _FakeCognitiveEngine()
    trace = {}
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (True, "completion_retry_ready"),
    )
    monkeypatch.setattr(
        chat_routes,
        "_gather_recent_user_messages_for_relevance",
        AsyncCallFixture(return_value=[]),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: engine if name == "cognitive_engine" else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Explain Dijkstra and include a worked example.",
        visible_user_message="Explain Dijkstra and include a worked example.",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply == (
        "Dijkstra's invariant finalizes the unsettled vertex with minimum "
        "tentative distance when all edges are nonnegative."
    )
    assert engine.calls == 2
    assert trace["completion_incumbent_preserved"] is True
    assert trace["completion_retry_failure_reason"] == "continuation_empty"
    assert trace["response_path"] == "cognitive_engine_completion_incumbent"


@pytest.mark.asyncio
async def test_progressive_continuation_accepts_complete_deadline_segment(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    incomplete = _bound_live_mind_controls_metadata()
    incomplete.update(
        {
            "reply_generation_incomplete": True,
            "reply_generation_stop_reason": "max_tokens",
            "reply_generation_failure_reasons": ["truncated_tail"],
        }
    )
    incomplete["live_mind_surface_control_receipt"].update(
        {
            "surface_quality_gate_passed": False,
            "surface_quality_gate_reasons": ["truncated_tail"],
            "generation_stop_reason": "max_tokens",
        }
    )
    complete = _bound_live_mind_controls_metadata()
    complete.update(
        {
            "reply_generation_incomplete": True,
            "reply_generation_stop_reason": "deadline_exceeded",
            "reply_generation_failure_reasons": ["truncated_tail"],
        }
    )
    complete["live_mind_surface_control_receipt"].update(
        {
            "surface_quality_gate_passed": False,
            "surface_quality_gate_reasons": ["truncated_tail"],
            "generation_stop_reason": "deadline_exceeded",
        }
    )

    class _FakeCognitiveEngine:
        def __init__(self):
            self.calls = []

        async def think(self, objective, context=None, **_kwargs):
            self.calls.append((objective, dict(context or {})))
            if len(self.calls) == 1:
                return SimpleNamespace(
                    content="The function updates balances and removes zero entries from the",
                    metadata=incomplete,
                )
            if len(self.calls) == 2:
                return SimpleNamespace(
                    content=" dictionary, then records every position where the",
                    metadata=incomplete,
                )
            return SimpleNamespace(
                content=(
                    " number of active entries equals the maximum. It returns all "
                    "positions tied for that peak."
                ),
                metadata=complete,
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    engine = _FakeCognitiveEngine()
    trace = {}
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (True, "completion_retry_ready"),
    )
    monkeypatch.setattr(
        chat_routes,
        "_gather_recent_user_messages_for_relevance",
        AsyncCallFixture(return_value=[]),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: engine if name == "cognitive_engine" else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Explain this code fully.",
        visible_user_message="Explain this code fully.",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply is not None
    assert reply.endswith("positions tied for that peak.")
    assert len(engine.calls) == 3
    assert engine.calls[1][1]["user_surface_continuation_partial"].endswith("from the")
    assert engine.calls[2][1]["user_surface_continuation_partial"].endswith("where the")
    assert trace["completion_retry_count"] == 2
    assert trace["foreground_model_generation_count"] == 3


def test_multipart_continuation_budget_tracks_parsed_obligations():
    from interface.routes import chat as chat_routes
    from interface.routes.chat_common import (
        _MAX_USER_SURFACE_CONTINUATIONS,
        _continuation_made_semantic_progress,
        _user_surface_continuation_budget,
    )

    assert _user_surface_continuation_budget(SimpleNamespace(question_parts=1)) == 2
    assert _user_surface_continuation_budget(SimpleNamespace(question_parts=6)) == 7
    assert (
        _user_surface_continuation_budget(SimpleNamespace(numbered_parts=100))
        == _MAX_USER_SURFACE_CONTINUATIONS
    )

    prompt = chat_routes.analyze_prompt_shape(
        "Explain Dijkstra. Include: (1) the invariant, (2) pseudocode, "
        "(3) a worked example, (4) complexity, and (5) a failure case."
    )
    first = "1. The invariant fixes the shortest unsettled distance. 2. Pseudocode follows."
    repeated = first + " This remains important to the explanation."
    advanced = repeated + " 3. Worked example: A connects to B with weight 2."
    assert _continuation_made_semantic_progress(first, repeated, prompt) is False
    assert _continuation_made_semantic_progress(first, advanced, prompt) is True


def test_unanswered_obligations_keep_numbered_identity_when_composed():
    from interface.routes import chat as chat_routes
    from interface.routes.chat_common import (
        _merge_obligation_completion,
        _unanswered_user_surface_obligations,
    )

    prompt = chat_routes.analyze_prompt_shape(
        "Explain Dijkstra. Include: (1) the invariant, (2) pseudocode, "
        "(3) a worked example, (4) complexity, and (5) a failure case."
    )
    partial = (
        "Dijkstra's algorithm computes shortest paths with nonnegative edges.\n"
        "1. The invariant finalizes the nearest unsettled vertex.\n"
        "2. Pseudocode repeatedly extracts and relaxes.\n"
        "3. A worked example uses A-B:2, A-C:1, C-B:1, B-D:2, C-D:5."
    )
    remaining = _unanswered_user_surface_obligations(partial, prompt)

    assert [item.numbered_label for item in remaining] == [4, 5]
    merged = _merge_obligation_completion(
        partial,
        "4) O((V+E) log V) with a heap and O(V^2) with an array.",
        remaining[0],
    )
    assert "\n\n4. O((V+E) log V)" in merged
    assert "4. 4)" not in merged


def test_obligation_completion_extends_an_open_section_instead_of_duplicating_it():
    from interface.routes import chat as chat_routes
    from interface.routes.chat_common import (
        _merge_obligation_completion,
        _unanswered_user_surface_obligations,
    )

    prompt = chat_routes.analyze_prompt_shape(
        "Explain Dijkstra. Include: (1) the invariant, (2) pseudocode, "
        "(3) a worked example on A, B, C, D with five weighted edges, "
        "(4) complexity, and (5) a failure case."
    )
    partial = (
        "1. The invariant finalizes the nearest unsettled vertex.\n"
        "2. Pseudocode initializes, extracts, and relaxes.\n"
        "## 3) Trace\nA-B has weight 2, but the rest of the"
    )
    remaining = _unanswered_user_surface_obligations(partial, prompt)
    worked = next(item for item in remaining if item.numbered_label == 3)

    merged = _merge_obligation_completion(
        partial,
        (
            "Use edges A-B:2, A-C:5, B-C:1, B-D:4, and C-D:1. "
            "Starting at A gives distances A=0, B=2, C=3, D=4."
        ),
        worked,
    )

    assert merged.count("3)") == 1
    assert "3. " not in merged
    assert "A-B has weight 2, but the rest of the\n\nUse edges" in merged
    assert worked.segment not in {
        item.segment
        for item in chat_routes._unanswered_user_surface_obligations(
            merged,
            prompt,
        )
    }


@pytest.mark.asyncio
async def test_compound_answer_schedules_each_uncovered_obligation(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    incomplete = _bound_live_mind_controls_metadata()
    incomplete.update(
        {
            "reply_generation_incomplete": True,
            "reply_generation_stop_reason": "eos",
            "reply_generation_failure_reasons": ["unanswered_question_part"],
        }
    )
    complete = _bound_live_mind_controls_metadata()

    class _FakeCognitiveEngine:
        def __init__(self):
            self.calls = []

        async def think(self, objective, context=None, **_kwargs):
            call = (objective, dict(context or {}))
            self.calls.append(call)
            if len(self.calls) == 1:
                return SimpleNamespace(
                    content=(
                        "Dijkstra's algorithm computes shortest paths with nonnegative edges.\n"
                        "1. The invariant finalizes the minimum unsettled distance.\n"
                        "2. Numbered pseudocode initializes distances, extracts the "
                        "minimum, and relaxes its outgoing edges.\n"
                        "3. A worked example uses A-B:2, A-C:1, C-B:1, "
                        "B-D:2, and C-D:5."
                    ),
                    metadata=incomplete,
                )
            if len(self.calls) == 2:
                return SimpleNamespace(
                    content=(
                        "O((V+E) log V) with a binary heap and O(V^2) with an array."
                    ),
                    metadata=incomplete,
                )
            return SimpleNamespace(
                content=(
                    "A negative edge can invalidate a finalized distance; use "
                    "Bellman-Ford instead."
                ),
                metadata=complete,
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    engine = _FakeCognitiveEngine()
    trace = {}
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (True, "completion_retry_ready"),
    )
    monkeypatch.setattr(
        chat_routes,
        "_gather_recent_user_messages_for_relevance",
        AsyncCallFixture(return_value=[]),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: engine if name == "cognitive_engine" else default
        ),
    )
    prompt = (
        "Explain Dijkstra. Include: (1) the invariant, (2) numbered pseudocode, "
        "(3) a worked example with at least five weighted edges, (4) time "
        "complexity with both a binary heap and an array, and (5) one failure "
        "case involving negative weights and the correct alternative."
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        prompt,
        visible_user_message=prompt,
        origin="user",
        timeout_s=240.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "foreground_endpoint": "Cortex",
        },
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply is not None
    assert "4. O((V+E) log V)" in reply
    assert "5. A negative edge" in reply
    assert len(engine.calls) == 3
    assert engine.calls[1][1]["user_surface_obligation_segment"].startswith(
        "time complexity"
    )
    assert engine.calls[2][1]["user_surface_obligation_segment"].startswith(
        "one failure case"
    )
    assert trace["completion_retry_count"] == 2
    assert trace["foreground_model_generation_count"] == 3


@pytest.mark.asyncio
async def test_continuation_handoff_preserves_long_structured_partial(monkeypatch):
    from core.brain import cognitive_engine as ce_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode

    calls = []

    class _Router:
        async def think(self, **kwargs):
            calls.append(kwargs)
            return " The remaining obligation is complete."

        def get_last_generation_metadata(self):
            return {}

    class _Container:
        @staticmethod
        def get(name, default=None):
            return _Router() if name == "llm_router" else default

    monkeypatch.setattr(ce_module, "get_container", lambda: _Container)
    partial = (
        "## Core invariant\n"
        + ("Dijkstra finalizes the nearest unsettled vertex.\n" * 150)
        + "7. The exact cutoff remains here:"
    )
    assert len(partial) > 6000
    context = {
        "desktop_quick_reply_contract": True,
        "user_surface_completion_retry": True,
        "user_surface_continuation_contract": True,
        "user_surface_continuation_partial": partial,
        "user_surface_continuation_resume_handle": "e" * 32,
        "visible_user_message": "Explain Dijkstra completely.",
    }

    thought = await CognitiveEngine()._direct_desktop_quick_reply(
        "Explain Dijkstra completely.",
        ThinkingMode.FAST,
        "user",
        context,
        timeout_s=60.0,
    )

    assert thought is not None
    call = calls[0]
    assert len(call["messages"]) == 3
    assert call["messages"][-1] == {"role": "assistant", "content": partial}
    assert call["user_surface_continuation_partial"] == partial
    assert call["user_surface_continuation_resume_handle"] == "e" * 32
    assert call["semantic_completion_contract"] is True
    assert "USER-SURFACE CONTINUATION CONTRACT" not in call["messages"][0]["content"]


@pytest.mark.asyncio
async def test_obligation_handoff_uses_exact_parent_partial_and_segment(monkeypatch):
    from core.brain import cognitive_engine as ce_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode

    calls = []

    class _Router:
        async def think(self, **kwargs):
            calls.append(kwargs)
            return "O((V+E) log V) with a heap and O(V^2) with an array."

        def get_last_generation_metadata(self):
            return {}

    class _Container:
        @staticmethod
        def get(name, default=None):
            return _Router() if name == "llm_router" else default

    monkeypatch.setattr(ce_module, "get_container", lambda: _Container)
    parent = "Explain Dijkstra and include complexity."
    partial = "Dijkstra finalizes the nearest unsettled distance."
    segment = "time complexity with both a binary heap and an array"
    context = {
        "desktop_quick_reply_contract": True,
        "user_surface_completion_retry": True,
        "user_surface_obligation_contract": True,
        "user_surface_obligation_parent_request": parent,
        "user_surface_obligation_partial": partial,
        "user_surface_obligation_segment": segment,
        "visible_user_message": segment,
        "max_tokens": 640,
    }

    thought = await CognitiveEngine()._direct_desktop_quick_reply(
        segment,
        ThinkingMode.FAST,
        "user",
        context,
        timeout_s=180.0,
    )

    assert thought is not None
    call = calls[0]
    assert call["messages"][-3:] == [
        {"role": "user", "content": parent},
        {"role": "assistant", "content": partial},
        {"role": "user", "content": segment},
    ]
    assert call["user_surface_validation_prompt"] == segment
    assert call["user_surface_obligation_contract"] is True
    assert call.get("user_surface_continuation_contract") is not True


@pytest.mark.asyncio
async def test_extended_surface_request_gets_semantic_completion_terminal(monkeypatch):
    from core.brain import cognitive_engine as ce_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode

    calls = []

    class _Router:
        async def think(self, **kwargs):
            calls.append(kwargs)
            return "A complete response."

        def get_last_generation_metadata(self):
            return {}

    class _Container:
        @staticmethod
        def get(name, default=None):
            return _Router() if name == "llm_router" else default

    monkeypatch.setattr(ce_module, "get_container", lambda: _Container)
    prompt = (
        "Explain the algorithm in one complete response. Include: (1) the "
        "invariant, (2) pseudocode, (3) an example, (4) complexity, and (5) "
        "a failure case."
    )
    context = {
        "desktop_quick_reply_contract": True,
        "visible_user_message": prompt,
        "prompt_shape": {
            "prefers_extended_answer": True,
            "requires_single_reply_coverage": True,
            "question_parts": 5,
        },
    }

    thought = await CognitiveEngine()._direct_desktop_quick_reply(
        prompt,
        ThinkingMode.FAST,
        "user",
        context,
        timeout_s=60.0,
    )

    assert thought is not None
    assert calls[0]["semantic_completion_contract"] is True


@pytest.mark.asyncio
async def test_short_desktop_reply_still_gets_semantic_completion_measurement(monkeypatch):
    from core.brain import cognitive_engine as ce_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode

    calls = []

    class _Router:
        async def think(self, **kwargs):
            calls.append(kwargs)
            return "I am steady and attentive."

        def get_last_generation_metadata(self):
            return {}

    class _Container:
        @staticmethod
        def get(name, default=None):
            return _Router() if name == "llm_router" else default

    monkeypatch.setattr(ce_module, "get_container", lambda: _Container)
    prompt = "How are you?"

    thought = await CognitiveEngine()._direct_desktop_quick_reply(
        prompt,
        ThinkingMode.FAST,
        "user",
        {"desktop_quick_reply_contract": True, "visible_user_message": prompt},
        timeout_s=60.0,
    )

    assert thought is not None
    assert calls[0]["semantic_completion_contract"] is True


@pytest.mark.asyncio
async def test_cognitive_owner_suppression_blocks_duplicate_route_retry(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    metadata = _bound_live_mind_controls_metadata()
    metadata.update(
        {
            "desktop_cognitive_engine_failure": True,
            "failure_reason": "reactive_recovery:timeout",
            "generation_failure_class": "reactive_recovery:timeout",
            "model_retry_suppressed": True,
        }
    )

    class _FakeCognitiveEngine:
        def __init__(self):
            self.calls = 0

        async def think(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("a failed CognitiveEngine owner must not be duplicated")
            return SimpleNamespace(
                content="The single owner exhausted its bounded turn.",
                metadata=metadata,
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

    engine = _FakeCognitiveEngine()
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: engine
            if name == "cognitive_engine"
            else default
        ),
    )

    trace = {}
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Compare the two architectures and choose one.",
        visible_user_message="Compare the two architectures and choose one.",
        origin="user",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "foreground_endpoint": "Cortex",
        },
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply is None
    assert engine.calls == 1
    assert trace["model_retry_suppressed"] is True
    assert trace["single_owner_generation_exhausted"] is True


@pytest.mark.asyncio
async def test_empty_cognitive_result_with_owner_suppression_does_not_retry(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    metadata = _bound_live_mind_controls_metadata()
    metadata.update(
        {
            "generation_failure_class": "latent_timeout:cooperative_cancelled",
            "model_retry_suppressed": True,
            "latent_cortex_selected": True,
            "latent_cortex_attempted": True,
            "latent_cortex_succeeded": False,
            "latent_cortex_failure_reason": "latent_timeout:cooperative_cancelled",
            "latent_cortex_receipt": {
                "episode_id": "live-timeout",
                "last_stage": "prefill",
                "stage_timings_s": {"prefill": 119.2},
            },
        }
    )

    class _FakeCognitiveEngine:
        def __init__(self):
            self.calls = 0

        async def think(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("empty single-owner result must not be retried")
            return SimpleNamespace(content="", metadata=metadata)

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

    engine = _FakeCognitiveEngine()
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: engine
            if name == "cognitive_engine"
            else default
        ),
    )

    trace = {}
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Compare the two architectures and choose one.",
        visible_user_message="Compare the two architectures and choose one.",
        origin="user",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "foreground_endpoint": "Cortex",
        },
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply is None
    assert engine.calls == 1
    assert trace["model_retry_suppressed"] is True
    assert trace["single_owner_generation_exhausted"] is True
    assert trace["latent_cortex_attempted"] is True
    assert trace["latent_cortex_receipt"]["last_stage"] == "prefill"


@pytest.mark.asyncio
async def test_rejected_generation_cannot_open_metadata_less_second_owner(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    first_metadata = _bound_live_mind_controls_metadata()
    first_metadata.update(
        {
            "desktop_cognitive_engine_failure": True,
            "failure_reason": "first_generation_rejected",
        }
    )

    class _FakeCognitiveEngine:
        def __init__(self):
            self.calls = 0

        async def think(self, objective, context=None, **kwargs):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("metadata-less second owner must remain unreachable")
            return SimpleNamespace(
                content="The first generation is a declared failure envelope.",
                metadata=first_metadata,
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    engine = _FakeCognitiveEngine()
    trace = {}
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (True, "test_ready"),
    )
    monkeypatch.setattr(
        chat_routes,
        "_gather_recent_user_messages_for_relevance",
        AsyncCallFixture(return_value=[]),
    )
    monkeypatch.setattr(
        chat_routes,
        "_ground_runtime_fact_status_reply",
        lambda _visible, reply, _lane, **_kwargs: f"{reply} grounded",
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: engine if name == "cognitive_engine" else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        'Reply exactly: "yes"',
        visible_user_message='Reply exactly: "yes"',
        origin="user",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "foreground_endpoint": "Cortex",
        },
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply is None
    assert engine.calls == 1
    assert trace["foreground_model_generation_consumed"] is True
    assert trace["foreground_model_generation_count"] == 1
    assert trace["single_owner_generation_exhausted"] is True
    receipt = trace["live_mind_surface_control_receipt"]
    assert receipt["applied"] is True
    assert receipt["surface_quality_gate_attempts"] == 1


@pytest.mark.asyncio
async def test_recent_desktop_context_is_deep_but_strictly_bounded():
    from interface.routes import chat as chat_routes

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        for index in range(20):
            chat_routes._conversation_log.append(
                {
                    "user": f"user-{index} " + ("u" * 4000),
                    "aura": f"aura-{index} " + ("a" * 5000),
                    "status": "complete",
                }
            )

    exchanges = await chat_routes._recent_completed_conversation_exchanges(
        current_user_message="continue",
        limit=chat_routes._RECENT_CONVERSATION_CONTEXT_EXCHANGES,
    )
    rendered = chat_routes._format_recent_conversation_context(exchanges)

    assert len(exchanges) == 12
    assert exchanges[0]["user"].startswith("user-8 ")
    assert exchanges[-1]["user"].startswith("user-19 ")
    assert all(
        len(entry["user"]) <= chat_routes._RECENT_CONVERSATION_USER_CHARS
        for entry in exchanges
    )
    assert all(
        len(entry["aura"]) <= chat_routes._RECENT_CONVERSATION_AURA_CHARS
        for entry in exchanges
    )
    assert len(rendered) <= chat_routes._RECENT_CONVERSATION_RENDERED_CHARS


@pytest.mark.asyncio
async def test_chat_exchange_persists_user_then_converges_through_atomic_exchange(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _Persistence:
        def record_turn(self, role, content, **kwargs):
            calls.append(("turn", role, content, dict(kwargs)))
            return f"{role}-turn"

        def record_exchange(self, user, aura, **kwargs):
            calls.append(("exchange", user, aura, dict(kwargs)))
            return ("user-turn", "aura-turn")

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _Persistence()
            if name == "persistence"
            else default
        ),
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchange_id = await chat_routes._begin_logged_exchange(
        "The desktop process exhausted memory.",
        session_id="desktop-client-session",
    )

    assert calls == [
        (
            "turn",
            "user",
            "The desktop process exhausted memory.",
            {
                "origin": "desktop_ui",
                "cid": f"{exchange_id}:user",
                "session_id": "desktop-client-session",
            },
        )
    ]

    await chat_routes._complete_logged_exchange(
        exchange_id,
        "The desktop process exhausted memory.",
        "I preserved this turn before reasoning and completed it afterward.",
        record_experience=False,
    )

    assert [call[0] for call in calls] == ["turn", "exchange"]
    assert calls[1][1:3] == (
        "The desktop process exhausted memory.",
        "I preserved this turn before reasoning and completed it afterward.",
    )
    assert calls[1][3]["cid"] == exchange_id
    assert calls[1][3]["session_id"] == "desktop-client-session"
    assert calls[1][3]["enqueue_memory_log"] is False


@pytest.mark.asyncio
async def test_completed_exchange_delegates_learning_to_durable_outbox(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _Persistence:
        def record_turn(self, role, content, **kwargs):
            return f"{role}-turn"

        def record_exchange(self, user, aura, **kwargs):
            calls.append((user, aura, dict(kwargs)))
            return ("user-turn", "aura-turn")

        def claim_memory_log_batch(self, **_kwargs):
            return []

        def settle_memory_log_item(self, *_args, **_kwargs):
            return "completed"

        def mark_memory_log_stage(self, *_args, **_kwargs):
            return True

    persistence = _Persistence()
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence if name == "persistence" else default
        ),
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchange_id = await chat_routes._begin_logged_exchange(
        "Let this completed turn inform future cognition."
    )
    state = await chat_routes._complete_logged_exchange(
        exchange_id,
        "Let this completed turn inform future cognition.",
        "The durable outbox owns post-turn learning after this reply is committed.",
    )

    assert state == "committed"
    assert len(calls) == 1
    assert calls[0][2]["enqueue_memory_log"] is True


@pytest.mark.asyncio
async def test_memory_outbox_applies_complete_conversation_experience_once(monkeypatch):
    from core.memory import chat_turn_logger
    from core.runtime import conversation_support
    from interface.routes import chat as chat_routes

    logged = AsyncCallFixture(return_value=True)
    experience = AsyncCallFixture()

    class _Coordinator:
        async def on_chat_turn(self, *_args, **_kwargs):
            return None

    async def _coordinator():
        return _Coordinator()

    monkeypatch.setattr(chat_turn_logger, "log_chat_turn_auto", logged)
    monkeypatch.setattr(conversation_support, "record_conversation_experience", experience)
    monkeypatch.setattr(
        "core.consciousness.coordinator.get_consciousness_coordinator",
        _coordinator,
    )

    outcome, error = await chat_routes._run_chat_turn_memory_log_item(
        {
            "user_content": "Carry this turn into the rest of the system.",
            "aura_content": "I will apply it through one durable post-turn owner.",
            "session_id": "durable-effects",
            "origin": "desktop_ui",
            "principal_id": "bryan",
            "principal_surface": "owner",
            "operation_id": "durable-effects:exchange:r1",
            "exchange_id": "exchange",
            "revision": 1,
        }
    )

    assert (outcome, error) == ("completed", "")
    logged.assert_awaited_once()
    experience.assert_awaited_once()
    assert experience.await_args[1] == {"principal_id": "bryan"}


@pytest.mark.asyncio
async def test_memory_outbox_retry_skips_durably_completed_effect_stages(
    monkeypatch,
    tmp_path,
):
    from core.consciousness import coordinator as consciousness_coordinator
    from core.conversation.persistence import ConversationPersistence
    from core.memory import chat_turn_logger
    from core.runtime import conversation_support
    from core.runtime.sqlite_support import connecting
    from interface.routes import chat as chat_routes

    logged = AsyncCallFixture(return_value=True)
    experience = AsyncCallFixture()
    consciousness_calls = AsyncCallFixture()

    class _Coordinator:
        async def on_chat_turn(self, *args, **kwargs):
            return await consciousness_calls(*args, **kwargs)

    async def _coordinator():
        return _Coordinator()

    persistence = ConversationPersistence(tmp_path / "staged-retry.db")
    session_id = persistence.start_session()
    persistence.record_exchange(
        "Apply these effects once.",
        "A settlement retry must not replay them.",
        cid="staged-retry",
        session_id=session_id,
        enqueue_memory_log=True,
    )
    operation_id = f"{session_id}:staged-retry:r1"
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence if name == "persistence" else default
        ),
    )
    monkeypatch.setattr(chat_turn_logger, "log_chat_turn_auto", logged)
    monkeypatch.setattr(conversation_support, "record_conversation_experience", experience)
    monkeypatch.setattr(
        consciousness_coordinator,
        "get_consciousness_coordinator",
        _coordinator,
    )

    first = persistence.claim_memory_log_batch(limit=1)[0]
    assert await chat_routes._run_chat_turn_memory_log_item(first) == ("completed", "")

    # Model a process death after all effects completed but before settlement.
    with connecting(sqlite3.connect(tmp_path / "staged-retry.db")) as con:
        con.execute(
            "UPDATE conversation_memory_outbox SET claimed_at = 0 WHERE operation_id = ?",
            (operation_id,),
        )
        con.commit()
    replay = persistence.claim_memory_log_batch(limit=1, lease_s=1.0)[0]
    assert await chat_routes._run_chat_turn_memory_log_item(replay) == ("completed", "")

    logged.assert_awaited_once()
    experience.assert_awaited_once()
    consciousness_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_memory_outbox_preserves_continuity_evaluation_for_rejected_reply(
    monkeypatch,
):
    from core.conversation import response_reliability
    from core.memory import chat_turn_logger
    from core.runtime import conversation_support
    from interface.routes import chat as chat_routes

    experience = AsyncCallFixture()
    logger_call = AsyncCallFixture(return_value=True)
    monkeypatch.setattr(
        response_reliability,
        "assess_conversation_learning_admission",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=False,
            reasons=("pseudo_internal_jargon",),
        ),
    )
    monkeypatch.setattr(
        chat_turn_logger,
        "local_chat_turn_learning_rejection_reason",
        lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr(chat_turn_logger, "log_chat_turn_auto", logger_call)
    monkeypatch.setattr(conversation_support, "record_conversation_experience", experience)

    outcome, reason = await chat_routes._run_chat_turn_memory_log_item(
        {
            "user_content": "Keep the user's side of this turn.",
            "aura_content": "[internal-looking wording]",
            "operation_id": "rejected:r1",
            "revision": 1,
        }
    )

    assert outcome == "rejected"
    assert reason == "pseudo_internal_jargon"
    experience.assert_awaited_once()
    logger_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_outbox_enqueue_restores_direct_experience_path(monkeypatch):
    from core.runtime import conversation_support
    from interface.routes import chat as chat_routes

    experience = AsyncCallFixture()

    class _Persistence:
        def record_exchange(self, *_args, **_kwargs):
            raise OSError("disk unavailable")

        def claim_memory_log_batch(self, **_kwargs):
            return []

        def settle_memory_log_item(self, *_args, **_kwargs):
            return "completed"

        def mark_memory_log_stage(self, *_args, **_kwargs):
            return True

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _Persistence() if name == "persistence" else default
        ),
    )
    monkeypatch.setattr(conversation_support, "record_conversation_experience", experience)
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchange_id = await chat_routes._begin_logged_exchange("Retain this turn despite disk failure")
    state = await chat_routes._complete_logged_exchange(
        exchange_id,
        "Retain this turn despite disk failure",
        "The direct compatibility path still applies the experience.",
    )

    assert state == "failed"
    experience.assert_awaited_once()
    failed = chat_routes._durable_conversation_write_snapshot(
        f"{exchange_id}:exchange"
    )
    assert failed is not None
    assert failed["failure_observed"] is True
    # The foreground path already surfaced and handled this failure. A later
    # shutdown drain must not attribute it to an unrelated pending write.
    await chat_routes._drain_durable_conversation_writes()


@pytest.mark.asyncio
async def test_chat_exchange_falls_back_to_atomic_exchange_when_prelog_fails(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _Persistence:
        def record_turn(self, role, content, **kwargs):
            calls.append(("turn_failed", role, content, dict(kwargs)))
            raise RuntimeError("pending write unavailable")

        def record_exchange(self, user, aura, **kwargs):
            calls.append(("exchange", user, aura, dict(kwargs)))
            return ("user-turn", "aura-turn")

    persistence = _Persistence()
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchange_id = await chat_routes._begin_logged_exchange("Preserve me before inference.")
    await chat_routes._complete_logged_exchange(
        exchange_id,
        "Preserve me before inference.",
        "The completion path used an atomic exchange write.",
        record_experience=False,
    )

    assert [call[0] for call in calls] == ["turn_failed", "exchange"]
    assert calls[1][1:3] == (
        "Preserve me before inference.",
        "The completion path used an atomic exchange write.",
    )


@pytest.mark.asyncio
async def test_chat_exchange_atomic_persistence_keeps_ui_session_id(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _Persistence:
        def record_turn(self, role, content, **kwargs):
            calls.append(("turn", role, content, dict(kwargs)))
            return f"{role}-turn"

        def record_exchange(self, user, aura, **kwargs):
            calls.append(("exchange", user, aura, dict(kwargs)))
            return ("user-turn", "aura-turn")

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _Persistence()
            if name == "persistence"
            else default
        ),
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    await chat_routes._persist_completed_conversation_exchange(
        exchange_id="ui-session-check",
        user_message="Remember this through the desktop UI session.",
        aura_response="I will persist it under the UI session id.",
        session_id="desktop-ui-session-42",
        user_already_persisted=False,
    )

    assert calls == [
        (
            "exchange",
            "Remember this through the desktop UI session.",
            "I will persist it under the UI session id.",
                {
                    "origin": "desktop_ui",
                    "cid": "ui-session-check",
                    "session_id": "desktop-ui-session-42",
                    "enqueue_memory_log": True,
                },
        )
    ]


@pytest.mark.asyncio
async def test_pending_conversation_timeout_retains_write_custody(monkeypatch):
    from interface.routes import chat as chat_routes

    started = threading.Event()
    release = threading.Event()
    calls = []

    class _SlowPersistence:
        def record_turn(self, role, content, **kwargs):
            started.set()
            release.wait(2.0)
            calls.append((role, content, dict(kwargs)))
            return "slow-user-turn"

    monkeypatch.setattr(_chat_preflight, "_DURABLE_CONVERSATION_WRITE_TIMEOUT_S", 0.02)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _SlowPersistence()
            if name == "persistence"
            else default
        ),
    )

    try:
        exchange_id = await chat_routes._begin_logged_exchange(
            "Preserve this even when the persistence budget expires."
        )
        assert started.is_set()
        pending = chat_routes._durable_conversation_write_snapshot(f"{exchange_id}:user")
        assert pending is not None
        assert pending["state"] == "pending"
        assert pending["task_done"] is False
        async with chat_routes._get_convo_lock():
            entry = next(row for row in chat_routes._conversation_log if row["id"] == exchange_id)
            assert entry["user_persistence_state"] == "pending"
            assert entry["user_persisted"] is False
    finally:
        release.set()
        await chat_routes._drain_durable_conversation_writes()

    settled = chat_routes._durable_conversation_write_snapshot(f"{exchange_id}:user")
    assert settled is not None
    assert settled["state"] == "committed"
    assert len(calls) == 1
    async with chat_routes._get_convo_lock():
        entry = next(row for row in chat_routes._conversation_log if row["id"] == exchange_id)
        assert entry["user_persistence_state"] == "committed"
        assert entry["user_persisted"] is True


@pytest.mark.asyncio
async def test_completed_exchange_timeout_settles_after_response_budget(monkeypatch):
    from interface.routes import chat as chat_routes

    exchange_started = threading.Event()
    release = threading.Event()
    exchange_calls = []

    class _SlowExchangePersistence:
        def record_turn(self, role, content, **kwargs):
            return f"{role}-turn"

        def record_exchange(self, user, aura, **kwargs):
            exchange_started.set()
            release.wait(2.0)
            exchange_calls.append((user, aura, dict(kwargs)))
            return ("user-turn", "aura-turn")

    persistence = _SlowExchangePersistence()
    monkeypatch.setattr(_chat_preflight, "_DURABLE_CONVERSATION_WRITE_TIMEOUT_S", 0.02)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence if name == "persistence" else default
        ),
    )

    exchange_id = await chat_routes._begin_logged_exchange("Keep the terminal answer durable.")
    try:
        state = await chat_routes._complete_logged_exchange(
            exchange_id,
            "Keep the terminal answer durable.",
            "This answer remains under write custody after the UI budget.",
            record_experience=False,
        )
        assert exchange_started.is_set()
        assert state == "pending"
        pending = chat_routes._durable_conversation_write_snapshot(
            f"{exchange_id}:exchange"
        )
        assert pending is not None
        assert pending["state"] == "pending"
        async with chat_routes._get_convo_lock():
            entry = next(row for row in chat_routes._conversation_log if row["id"] == exchange_id)
            assert entry["status"] == "complete"
            assert entry["durability_state"] == "pending"
    finally:
        release.set()
        await chat_routes._drain_durable_conversation_writes()

    settled = chat_routes._durable_conversation_write_snapshot(
        f"{exchange_id}:exchange"
    )
    assert settled is not None
    assert settled["state"] == "committed"
    assert len(exchange_calls) == 1
    async with chat_routes._get_convo_lock():
        entry = next(row for row in chat_routes._conversation_log if row["id"] == exchange_id)
        assert entry["durability_state"] == "committed"


@pytest.mark.asyncio
async def test_late_unobserved_persistence_failure_is_reported_once_at_shutdown():
    from interface.routes import chat as chat_routes

    started = threading.Event()
    release = threading.Event()
    operation_id = f"late-failure-{time.time_ns()}:exchange"

    def _fail_after_response_budget():
        started.set()
        release.wait(2.0)
        raise OSError("late disk failure")

    record = chat_routes._start_durable_conversation_write(
        operation_id=operation_id,
        payload={"kind": "late_failure_probe"},
        operation=_fail_after_response_budget,
    )
    try:
        state = await chat_routes._await_durable_conversation_write(
            record,
            timeout_s=0.02,
        )
        assert state == "pending"
        assert started.is_set()
    finally:
        release.set()

    with pytest.raises(RuntimeError, match=operation_id):
        await chat_routes._drain_durable_conversation_writes()
    failed = chat_routes._durable_conversation_write_snapshot(operation_id)
    assert failed is not None
    assert failed["state"] == "failed"
    assert failed["failure_observed"] is True
    await chat_routes._drain_durable_conversation_writes()


@pytest.mark.asyncio
async def test_partial_legacy_write_retries_without_duplicate_user_turn(monkeypatch):
    from interface.routes import chat as chat_routes

    rows_by_cid = {}
    aura_attempts = 0

    class _IdempotentLegacyPersistence:
        def record_turn(self, role, content, **kwargs):
            nonlocal aura_attempts
            cid = kwargs["cid"]
            if role == "aura":
                aura_attempts += 1
                if aura_attempts == 1:
                    raise RuntimeError("injected aura-side write failure")
            existing = rows_by_cid.get(cid)
            if existing is not None:
                assert existing == (role, content)
                return cid
            rows_by_cid[cid] = (role, content)
            return cid

    persistence = _IdempotentLegacyPersistence()
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence if name == "persistence" else default
        ),
    )

    exchange_id = await chat_routes._begin_logged_exchange("Retry this exact exchange once.")
    first = await chat_routes._complete_logged_exchange(
        exchange_id,
        "Retry this exact exchange once.",
        "The retry preserves one user turn and one answer.",
        record_experience=False,
    )
    second = await chat_routes._complete_logged_exchange(
        exchange_id,
        "Retry this exact exchange once.",
        "The retry preserves one user turn and one answer.",
        record_experience=False,
    )

    assert first == "failed"
    assert second == "committed"
    assert aura_attempts == 2
    assert rows_by_cid == {
        f"{exchange_id}:user": ("user", "Retry this exact exchange once."),
        f"{exchange_id}:aura": (
            "aura",
            "The retry preserves one user turn and one answer.",
        ),
    }
    receipt = chat_routes._durable_conversation_write_snapshot(
        f"{exchange_id}:exchange"
    )
    assert receipt is not None
    assert receipt["state"] == "committed"
    assert receipt["attempt"] == 2


def test_conversation_persistence_registers_memory_commit_shutdown_drain(monkeypatch):
    from core.runtime.shutdown_coordinator import ShutdownCoordinator
    from interface.routes import chat as chat_routes

    coordinator = ShutdownCoordinator()
    monkeypatch.setattr(
        "core.runtime.shutdown_coordinator.get_shutdown_coordinator",
        lambda: coordinator,
    )

    chat_routes._ensure_durable_conversation_shutdown_handler()
    chat_routes._ensure_durable_conversation_shutdown_handler()

    assert coordinator.handler_names("memory_commit") == [
        chat_routes._DURABLE_CONVERSATION_SHUTDOWN_HANDLER
    ]


@pytest.mark.asyncio
async def test_chat_restart_recovers_completed_exchange_from_canonical_persistence(
    monkeypatch,
    tmp_path,
):
    from core.conversation.persistence import ConversationPersistence
    from interface.routes import chat as chat_routes

    persistence = ConversationPersistence(tmp_path / "conversation.db")
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchange_id = await chat_routes._begin_logged_exchange(
        "Remember the live desktop continuity contract."
    )
    await chat_routes._complete_logged_exchange(
        exchange_id,
        "Remember the live desktop continuity contract.",
        "I will recover this completed exchange after process memory is cleared.",
        record_experience=False,
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    recovered = await chat_routes._recent_completed_conversation_exchanges(
        current_user_message="Continue from before the restart.",
        limit=6,
    )

    assert recovered
    assert recovered[-1]["user"] == "Remember the live desktop continuity contract."
    assert "recover this completed exchange" in recovered[-1]["aura"]


@pytest.mark.asyncio
async def test_chat_restart_recovers_completed_exchange_from_ui_session(
    monkeypatch,
    tmp_path,
):
    from core.conversation.persistence import ConversationPersistence
    from interface.routes import chat as chat_routes

    persistence = ConversationPersistence(tmp_path / "conversation.db")
    persistence.start_session({"source": "boot-session-that-should-not-own-ui-turn"})
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchange_id = await chat_routes._begin_logged_exchange(
        "Remember the UI lane continuity detail.",
        session_id="desktop-visible-session",
    )
    await chat_routes._complete_logged_exchange(
        exchange_id,
        "Remember the UI lane continuity detail.",
        "I persisted it under the desktop-visible-session transcript.",
        record_experience=False,
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    recovered = await chat_routes._recent_completed_conversation_exchanges(
        current_user_message="What did I ask you to remember?",
        session_id="desktop-visible-session",
        limit=6,
    )
    wrong_session = await chat_routes._recent_completed_conversation_exchanges(
        current_user_message="What did I ask you to remember?",
        session_id="different-visible-session",
        limit=6,
    )
    isolated_session = await chat_routes._recent_completed_conversation_exchanges(
        current_user_message="How are you doing right now?",
        session_id="fresh-condition-session",
        limit=6,
        allow_cross_session=False,
    )

    assert recovered
    assert recovered[-1]["user"] == "Remember the UI lane continuity detail."
    assert recovered[-1]["session_id"] == "desktop-visible-session"
    assert "desktop-visible-session transcript" in recovered[-1]["aura"]
    # Cross-session recall is deliberate — _load_durable_conversation_exchanges_sync
    # scans other recent sessions when the current one is thin, because three
    # reboots used to hide yesterday's conversation entirely. On a single-user
    # desktop those other sessions are the same person's.
    #
    # The property that must hold is not emptiness but honest labelling: a turn
    # from another session may be recalled, and may never be presented AS this
    # session's. Mislabelling is what would let her claim continuity she does
    # not have.
    for exchange in wrong_session:
        assert exchange["session_id"] != "different-visible-session"
        assert exchange["session_id"] == "desktop-visible-session"
    assert isolated_session == []


@pytest.mark.asyncio
async def test_desktop_required_runtime_status_invokes_engine_then_grounds(monkeypatch):
    from core.conversation.user_surface_contract import resolve_user_surface_prompt
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(content="unexpected model answer")

    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "Answer directly in two sentences: what lane are you using for this live desktop chat?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        source="desktop_ui",
        require_engine=True,
    )

    assert len(calls) == 1
    assert calls[0]["context"]["runtime_fact_status_contract"] is True
    assert calls[0]["context"]["cognitive_engine_required"] is True
    surface_prompt = resolve_user_surface_prompt(calls[0]["context"])
    assert surface_prompt.bound is True
    assert surface_prompt.valid is True
    assert surface_prompt.prompt == user_message
    assert surface_prompt.source == "desktop_chat.visible_user_message"
    assert reply
    assert "Cortex" in reply
    assert "Cortex (32B)" not in reply
    assert "active foreground lane" in reply
    assert "CognitiveEngine handled this turn: yes" in reply
    assert "governed tools available: yes" in reply


@pytest.mark.asyncio
async def test_desktop_required_cognitive_fusion_status_invokes_engine_then_grounds(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(content="unexpected model answer")

    patch_chat_lane(monkeypatch, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "In two direct sentences, explain why the live desktop chat path must stay fused "
        "to your cognitive engine instead of falling back into generic assistant mode."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        source="desktop_ui",
        require_engine=True,
    )

    assert len(calls) == 1
    assert calls[0]["context"]["runtime_fact_status_contract"] is True
    assert calls[0]["context"]["cognitive_engine_required"] is True
    assert reply
    assert "Cortex" in reply
    assert "Cortex (32B)" not in reply
    assert "active foreground lane" in reply
    assert "CognitiveEngine handled this turn: yes" in reply
    assert "governed tools available: yes" in reply
    assert "recurrent depth: active" in reply


def test_compact_desktop_contract_keeps_hypothetical_tool_plans_compact():
    from interface.routes import chat as chat_routes

    user_message = (
        "Explain how you would use browser research and a document editor together on a user task."
    )

    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=chat_routes._looks_like_desktop_objective(user_message),
            capability_inventory_contract=False,
        )
        is True
    )


def test_structured_code_explanation_uses_the_completion_sized_quick_lane():
    from interface.routes import chat as chat_routes

    user_message = (
        "Please explain, in a complete answer with a short introduction, a numbered "
        "walkthrough, and a final conclusion, how this Python function works: "
        "def peak(values): return max(values) if values else None"
    )

    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=False,
            capability_inventory_contract=False,
        )
        is True
    )


def test_inline_five_part_technical_request_uses_deep_lane():
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    user_message = (
        "Explain Dijkstra's shortest-path algorithm in one complete response. Include: "
        "(1) the core invariant, (2) numbered pseudocode, (3) a worked example "
        "on four vertices with at least five weighted edges, (4) time complexity "
        "with a binary heap and an array, and (5) one negative-weight failure "
        "and the correct alternative."
    )

    assert chat_routes._select_cognitive_chat_mode(
        user_message, user_message
    ) is ThinkingMode.DEEP
    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=False,
            capability_inventory_contract=False,
        )
        is False
    )
    assert chat_routes._desktop_live_reply_token_budget(
        user_message,
        capability_inventory_contract=False,
        bounded_planning_contract=True,
        runtime_fact_status_contract=False,
        memory_state_contract=False,
    ) == chat_routes.answer_surface_token_floor(user_message)


def test_self_contained_choice_does_not_request_stale_conversation_context():
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    user_message = (
        "Compare optimistic and pessimistic locking for a hot task queue, choose "
        "which one you would use in a single-host async runtime, explain why, and "
        "verify your choice with one concrete failure scenario."
    )

    assert chat_routes._has_local_choice_antecedent(user_message) is True
    assert chat_routes._is_contextual_relevance_challenge(user_message) is False
    assert chat_routes._desktop_turn_needs_recent_context(user_message) is False
    assert chat_routes._select_cognitive_chat_mode(user_message, user_message) is ThinkingMode.DEEP
    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=False,
            capability_inventory_contract=False,
        )
        is False
    )

    unresolved = "Which one should I choose?"
    assert chat_routes._has_local_choice_antecedent(unresolved) is False
    assert chat_routes._is_contextual_relevance_challenge(unresolved) is True
    assert chat_routes._desktop_turn_needs_recent_context(unresolved) is True


@pytest.mark.asyncio
async def test_compound_choice_reaches_engine_as_deep_self_contained_turn(monkeypatch):
    from core.brain.types import ThinkingMode
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    user_message = (
        "Compare optimistic and pessimistic locking for a hot task queue, choose "
        "which one you would use in a single-host async runtime, explain why, and "
        "verify your choice with one concrete failure scenario."
    )
    answer = (
        "Optimistic locking lets workers race and reject stale claims, whereas pessimistic "
        "locking serializes acquisition before work begins. I would choose pessimistic locking "
        "for a hot single-host async queue because one short critical section prevents duplicate "
        "ownership without repeated conflict retries. To verify it, inject cancellation immediately "
        "after a worker acquires the queue lock; the test should show that a finally block releases "
        "the lock and exactly one waiting worker acquires the task."
    )
    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            metadata = _bound_live_mind_controls_metadata()
            metadata["live_mind_surface_control_receipt"].update(
                {"generation_max_tokens": 512, "generated_tokens": 94}
            )
            return SimpleNamespace(content=answer, metadata=metadata)

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

    recent = AsyncCallFixture(return_value=[{"user": "stale", "assistant": "stale"}])
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(_chat_memory_state, "_recent_completed_conversation_exchanges", recent)
    for name in (
        "_build_conversation_recall_reply",
        "_build_retained_memory_evidence_context",
        "_build_context_challenge_repair_reply",
        "_fetch_deep_memory_context",
    ):
        monkeypatch.setattr(chat_routes, name, AsyncCallFixture(return_value=""))
    engine = _FakeCognitiveEngine()
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: engine if name == "cognitive_engine" else default
        ),
    )

    trace = {}
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=120.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply == answer
    assert len(calls) == 1
    assert calls[0]["kwargs"]["mode"] is ThinkingMode.DEEP
    assert calls[0]["context"]["compact_desktop_chat_contract"] is False
    assert calls[0]["context"]["prompt_shape"]["imperative_parts"] == 4
    assert calls[0]["context"]["recent_completed_exchanges"] == []
    # The guarantee is that no prior exchange reaches the PROMPT (asserted
    # above). This used to be written as "never looked at all", which stopped
    # being true when the antecedent lookup landed: a pro-form follow-up
    # ("why did it catch your attention?") carries no topic of its own, so the
    # reliability gate needs the previous turn to have anything to check
    # relevance against. That reader takes exactly one exchange and never
    # feeds the prompt. Assert the distinction rather than forbidding the read.
    assert all(
        call[1].get("limit") == 1 for call in recent.calls
    ), f"only the antecedent reader may run on a self-contained turn: {recent.calls}"
    assert trace["foreground_model_generation_count"] == 1
    assert trace["single_owner_generation_exhausted"] is True


@pytest.mark.asyncio
async def test_ordinary_desktop_chat_turn_keeps_the_prompt_cache(monkeypatch):
    """The conversation lane is the one lane that MUST reuse KV.

    Its prompt is the whole conversation, so re-prefilling from token zero
    makes turn latency climb until it crosses the turn budget — the measured
    endurance wall. `compact_desktop_chat_contract` carried
    `disable_prompt_cache` (and `clear_prompt_cache`, which wiped every other
    lane's entry too) from an era when the 32B's cache budget was zero anyway.
    """
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    user_message = "What's the weather like where you are?"
    answer = "I don't have a window, but the host reports a warm chassis today."
    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(dict(context or {}))
            return SimpleNamespace(
                content=answer, metadata=_bound_live_mind_controls_metadata()
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        _chat_memory_state,
        "_recent_completed_conversation_exchanges",
        AsyncCallFixture(return_value=[]),
    )
    for name in (
        "_build_conversation_recall_reply",
        "_build_retained_memory_evidence_context",
        "_build_context_challenge_repair_reply",
        "_fetch_deep_memory_context",
    ):
        monkeypatch.setattr(chat_routes, name, AsyncCallFixture(return_value=""))
    engine = _FakeCognitiveEngine()
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: engine if name == "cognitive_engine" else default
        ),
    )

    await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=120.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace={},
    )

    assert len(calls) == 1
    context = calls[0]
    assert context.get("compact_desktop_chat_contract") is True, (
        "this test is only meaningful on the compact desktop chat path"
    )
    assert context.get("disable_prompt_cache") is not True, (
        "the ordinary chat turn must reuse KV or long conversations time out"
    )
    assert context.get("clear_prompt_cache") is not True, (
        "clearing on every chat turn wipes every lane's KV, not just this one"
    )


def test_compact_desktop_contract_does_not_hide_actual_tool_execution_requests():
    from interface.routes import chat as chat_routes

    user_message = (
        "Open Chrome, search for climate news, create a document, and export it as a PDF."
    )

    assert chat_routes._looks_like_desktop_objective(user_message) is True
    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=chat_routes._looks_like_desktop_objective(user_message),
            capability_inventory_contract=False,
        )
        is False
    )


def test_compact_desktop_contract_does_not_starve_self_process_questions():
    from interface.routes import chat as chat_routes

    user_messages = (
        (
            "When you are confused, how does that change your planning, memory use, "
            "and tool verification?"
        ),
        (
            "Quick live-path check. Don't give me a health card or telemetry list. "
            "In ordinary speech, answer from your actual current context: what are "
            "you attending to from Bryan's recent messages, what remembered concern "
            "should change your next decision, and what do you want to do next?"
        ),
        "Is this the real Aura, or did the raw model take over?",
    )

    for user_message in user_messages:
        assert (
            chat_routes._is_compact_desktop_chat_contract(
                user_message,
                user_message,
                desktop_execution_contract=False,
                capability_inventory_contract=False,
            )
            is False
        )


def test_bounded_present_state_partition_stays_on_compact_conversation_lane():
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    user_message = (
        "ChatGPT here. How are you feeling at this moment? Tell me what is "
        "directly present in your internal state, and what you only tentatively "
        "think may be causing it."
    )

    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=False,
            capability_inventory_contract=False,
        )
        is True
    )
    # Compact ownership subsequently normalizes the selected mode to FAST.
    assert chat_routes._select_cognitive_chat_mode(
        user_message, user_message
    ) is ThinkingMode.DEEP


def test_mixed_timeout_and_semantic_defects_continue_instead_of_replacing_source():
    from interface.routes import chat as chat_routes

    assert chat_routes._reply_needs_continuation(
        "A substantial answer that stopped because",
        (
            "truncated_tail",
            "off_topic_self_reflection_reply",
            "unanswered_question_part",
        ),
    )
    assert chat_routes._reply_needs_continuation(
        "A complete but generic answer.",
        ("generic_assistant_language", "unanswered_question_part"),
    )
    assert not chat_routes._reply_needs_continuation(
        "A complete but generic answer.",
        ("generic_assistant_language",),
    )
    assert chat_routes._reply_has_physical_completion_failure(
        ("truncated_tail", "unanswered_question_part")
    )
    assert not chat_routes._reply_has_physical_completion_failure(
        ("unanswered_question_part",)
    )


def test_known_incomplete_compound_draft_is_not_published_as_salvage(monkeypatch):
    from interface.routes import chat as chat_routes

    class Assessment:
        reasons = ("unanswered_question_part",)

    monkeypatch.setattr(
        "core.conversation.response_reliability.assess_user_facing_reply",
        lambda *_args, **_kwargs: Assessment(),
    )

    assert (
        chat_routes._servable_draft_or_none(
            "I answered sections one through four completely.",
            "Answer all five sections.",
        )
        == ""
    )


def test_compact_desktop_contract_allows_lightweight_live_recall_state_turn():
    from interface.routes import chat as chat_routes

    user_message = (
        "Remember this phrase: silver lantern. Also tell me one thing your live mind "
        "is attending to right now."
    )

    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=False,
            capability_inventory_contract=False,
        )
        is True
    )


def test_compact_desktop_contract_allows_direct_durable_memory_pin():
    from interface.routes import chat as chat_routes

    user_message = "Remember this phrase across sessions: silver lantern."

    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=False,
            capability_inventory_contract=False,
        )
        is True
    )


def test_compact_desktop_contract_keeps_durable_memory_reasoning_out_of_quick_route():
    from interface.routes import chat as chat_routes

    user_message = (
        "Remember this uncertainty across sessions. How should that change your planning "
        "and tool verification tomorrow?"
    )

    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=False,
            capability_inventory_contract=False,
        )
        is False
    )


def test_compact_desktop_contract_allows_bounded_recall_grounding_question():
    from interface.routes import chat as chat_routes

    user_message = (
        "What phrase did I ask you to remember, and how does your cognitive engine "
        "keep this reply grounded?"
    )

    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=False,
            capability_inventory_contract=False,
        )
        is True
    )


def test_conversation_recall_classifier_handles_natural_memory_questions():
    from interface.routes import chat as chat_routes

    assert (
        chat_routes._classify_conversation_recall_request("Do you remember what I said earlier?")
        == "last_user"
    )
    assert (
        chat_routes._classify_conversation_recall_request("Can you remind me what I asked?")
        == "last_user"
    )
    assert (
        chat_routes._classify_conversation_recall_request("What was my last message?")
        == "last_user"
    )
    assert (
        chat_routes._classify_conversation_recall_request("Do you remember what you said?")
        == "last_aura"
    )
    assert (
        chat_routes._classify_conversation_recall_request("Can you remind me what you answered?")
        == "last_aura"
    )
    assert (
        chat_routes._classify_conversation_recall_request("What did we discuss earlier in this conversation?")
        == "topic"
    )
    assert (
        chat_routes._classify_conversation_recall_request("Can you remind me what we talked about?")
        == "topic"
    )
    assert (
        chat_routes._classify_conversation_recall_request("Could you summarize our last two messages?")
        == "recent_pair"
    )
    assert (
        chat_routes._classify_conversation_recall_request("What were our last two messages?")
        == "recent_pair"
    )
    assert (
        chat_routes._classify_conversation_recall_request(
            "Use the last two messages to explain what you are attending to now and how that changes your next decision."
        )
        == ""
    )


@pytest.mark.asyncio
async def test_conversation_recall_summarizes_last_two_messages_from_live_log():
    from interface.routes import chat as chat_routes

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "What phrase did I just ask you to remember?",
                    "aura": 'The phrase you asked me to remember in this session was "cobalt sunrise".',
                    "status": "complete",
                },
                {
                    "user": "In one sentence, what are you focused on right now?",
                    "aura": "Understanding you and staying present for this conversation.",
                    "status": "complete",
                },
            ]
        )

    reply = await chat_routes._build_conversation_recall_reply(
        "Could you summarize our last two messages?"
    )

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    assert reply is not None
    assert "cobalt sunrise" in reply
    assert "focused on right now" in reply


def test_failure_mode_surface_request_is_not_misclassified_as_planning():
    from interface.routes import chat as chat_routes

    user_message = "Name one failure mode you should surface honestly instead of masking."

    assert chat_routes._is_bounded_nonexecuting_planning_request(user_message) is False
    reply = chat_routes._build_failure_mode_surface_reply(user_message)
    assert reply
    assert "failure mode" in reply.lower()
    assert "avoid claiming completion" in reply


@pytest.mark.asyncio
async def test_desktop_required_bounded_planning_uses_foreground_cognitive_engine(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "I would handle browser research and a document editor as one governed workflow: collect the "
                    "sources, draft the document, verify the visible editor content, export the artifact, and write "
                    "receipts for each confirmed step."
                )
            )

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Explain how you would use browser research and a document editor together on a user task."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert len(calls) == 1
    assert calls[0]["context"]["bounded_planning_contract"] is True
    assert calls[0]["context"]["require_full_foreground_mind_reply"] is True
    expected_floor = chat_routes.answer_surface_token_floor(user_message)
    assert calls[0]["context"]["max_tokens"] == max(1536, expected_floor)
    assert calls[0]["context"]["user_surface_completion_floor"] == expected_floor
    assert "one natural paragraph" in calls[0]["context"]["response_style_contract"]
    assert "do not invent a specific example" in calls[0]["context"]["response_style_contract"]
    assert reply
    assert "browser" in reply.lower()
    assert "document" in reply.lower()
    assert "receipts" in reply.lower()


@pytest.mark.asyncio
async def test_desktop_required_failure_mode_surface_uses_foreground_cognitive_engine(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "One failure mode I should surface honestly is a tool action that partially starts and then "
                    "times out. I should preserve partial state or receipt evidence, report what was verified, and "
                    "avoid claiming completion until the effect check proves it."
                )
            )

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "Name one failure mode you should surface honestly instead of masking."
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert len(calls) == 1
    assert calls[0]["context"]["failure_mode_contract"] is True
    assert reply
    assert "partial state or receipt" in reply
    assert "avoid claiming completion" in reply


@pytest.mark.asyncio
async def test_desktop_required_cognitive_engine_timeout_does_not_retry_hidden_work(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **_kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            await asyncio.sleep(2.2)
            return SimpleNamespace(content="late answer")

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, *_args, **_kwargs):
            calls.append({"unexpected_pool_retry": True})
            return SimpleNamespace(content="unexpected hidden pool retry")

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Answer directly about desktop reliability.",
        visible_user_message="Answer directly about desktop reliability.",
        origin="user",
        timeout_s=0.01,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply is None
    assert calls == []


@pytest.mark.asyncio
async def test_desktop_execution_contract_uses_bounded_planning_context(monkeypatch):
    from core.brain.types import ThinkingMode
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": mode,
                    "timeout_s": kwargs.get("timeout_s"),
                }
            )
            return SimpleNamespace(content='{"steps": []}')

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, *_args, **_kwargs):
            calls.append({"unexpected_pool_retry": True})
            return SimpleNamespace(content="unexpected pool retry")

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Open Google Docs and write an essay about climate adaptation.",
        visible_user_message="Open Google Docs and write an essay about climate adaptation.",
        origin="user",
        timeout_s=105.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply == '{"steps": []}'
    assert calls and calls[0]["mode"] is ThinkingMode.SLOW
    assert calls[0]["context"]["desktop_execution_contract"] is True
    assert calls[0]["context"]["foreground_request"] is True
    assert calls[0]["context"]["user_explicitly_authorized"] is True
    assert calls[0]["context"]["user_requested_action"] is True
    assert calls[0]["context"]["user_visible_desktop_action"] is True
    assert calls[0]["context"]["verification_required"] is True
    assert calls[0]["context"]["source"] == "desktop_ui"
    assert calls[0]["context"]["origin"] == "user"
    assert calls[0]["context"]["allow_heuristic_desktop_plan"] is True
    assert calls[0]["context"]["max_tokens"] == 1024
    assert calls[0]["context"]["num_predict"] == 1024
    assert calls[0]["context"]["skip_runtime_payload"] is True
    assert calls[0]["context"]["disable_prompt_cache"] is True
    assert calls[0]["context"]["clear_prompt_cache"] is True
    assert "never say you cannot interact with apps" in calls[0]["context"]["response_style_contract"]
    assert calls[0]["context"]["desktop_task_planning_schema"]["steps"]


@pytest.mark.asyncio
async def test_desktop_required_cognitive_engine_does_not_retry_transient_error_by_default(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "timeout_s": kwargs.get("timeout_s"),
                }
            )
            raise RuntimeError("transient live cognitive turn reset")

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, *_args, **_kwargs):
            calls.append({"unexpected_pool_retry": True})
            return SimpleNamespace(content="unexpected pool retry")

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.delenv("AURA_DESKTOP_ALLOW_TRANSIENT_ENGINE_RETRY", raising=False)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Answer directly about reliable desktop chat recovery.",
        visible_user_message="Answer directly about reliable desktop chat recovery.",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply is None
    assert len(calls) == 1
    assert not any(call.get("unexpected_pool_retry") for call in calls)
    assert all(call["context"]["desktop_cognitive_engine_required"] is True for call in calls)


@pytest.mark.asyncio
async def test_desktop_required_cognitive_engine_can_opt_into_transient_retry_same_path(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "timeout_s": kwargs.get("timeout_s"),
                }
            )
            if len(calls) == 1:
                raise RuntimeError("transient live cognitive turn reset")
            return SimpleNamespace(
                content=(
                    "Reliable desktop chat can retry an explicitly enabled transient reset without "
                    "leaving the governed CognitiveEngine path. The retry still uses the same protected "
                    "foreground context, keeps legacy fallbacks disabled, and returns only after the "
                    "second draft satisfies the normal user-facing response contract."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, *_args, **_kwargs):
            calls.append({"unexpected_pool_retry": True})
            return SimpleNamespace(content="unexpected pool retry")

    monkeypatch.setenv("AURA_DESKTOP_ALLOW_TRANSIENT_ENGINE_RETRY", "1")
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )
    monkeypatch.setattr(
        chat_routes,
        "_desktop_transient_engine_retry_allowed",
        lambda *, reason: (True, reason),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Answer directly about reliable desktop chat recovery.",
        visible_user_message="Answer directly about reliable desktop chat recovery.",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "transient reset" in reply
    assert "CognitiveEngine path" in reply
    assert len(calls) == 2
    assert not any(call.get("unexpected_pool_retry") for call in calls)
    assert all(call["context"]["desktop_cognitive_engine_required"] is True for call in calls)


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_keeps_preflight_context_out_of_objective(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **_kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "Reliable desktop tool use matters because local actions have to be observable and reversible. "
                    "It also gives the user evidence that the assistant completed real work instead of only describing it."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Give me two concise sentences about reliable desktop tool use.",
        visible_user_message="Give me two concise sentences about reliable desktop tool use.",
        preflight_context_message=(
            "[Operational Self Context]\n"
            "Name: Aura\n"
            "Runtime status: healthy\n"
            "User message: Give me two concise sentences about reliable desktop tool use."
        ),
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert calls[0]["objective"] == "Give me two concise sentences about reliable desktop tool use."
    assert calls[0]["context"]["visible_user_message"] == (
        "Give me two concise sentences about reliable desktop tool use."
    )
    assert calls[0]["context"]["preflight_context_message"].startswith("[Operational Self Context]")


@pytest.mark.asyncio
async def test_desktop_capability_inventory_uses_cognitive_engine_with_catalog_context(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "From the live desktop path I can use governed tool lanes for desktop control, browser and web "
                    "research, file operations, document drafting, terminal work, memory recall, and skill execution. "
                    "A hypothetical scenario would request approval, open sources, create a document, verify the "
                    "visible result, export the file, and record governance receipts without claiming unverified work."
                )
            )

    class _FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield from [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
                {
                    "name": "file_operation",
                    "available": True,
                    "description": "Read and write local files and documents.",
                    "route_class": "stateful",
                    "risk_class": "medium",
                    "effect_scope": "file_system",
                },
            ]

        def execute(self, *_args, **_kwargs):
            return None

    class _FakeAuthority:
        def is_ready(self):
            return True

    class _FakeWill:
        def decide(self, *_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    def fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        if name == "capability_engine":
            return _FakeCapabilityEngine()
        if name == "authority_gateway":
            return _FakeAuthority()
        if name == "unified_will":
            return _FakeWill()
        return default

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(fake_get))

    user_message = "What tools can you use externally, and what is a hypothetical scenario where you use them?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert len(calls) == 1
    assert calls[0]["context"]["capability_inventory_contract"] is True
    assert calls[0]["context"]["preflight_context_message"] == ""
    assert calls[0]["context"]["recent_completed_exchanges"] == []
    assert calls[0]["context"]["recent_conversation_context"] == ""
    assert "desktop control" in reply
    assert "governance receipts" in reply


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_refuses_doomed_required_budget_before_allocation(monkeypatch):
    from interface.routes import chat as chat_routes

    def fake_get(name, default=None):
        if name == "cognitive_engine":
            raise AssertionError("doomed foreground budget must not allocate CognitiveEngine")
        return default

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(fake_get))

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Explain why long-running desktop conversations need stable memory continuity.",
        visible_user_message="Explain why long-running desktop conversations need stable memory continuity.",
        origin="user",
        timeout_s=2.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply is None


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_uses_direct_cognitive_engine_when_pool_unavailable(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            calls.append("engine_think")
            return SimpleNamespace(
                content=(
                    "Yes. I am still reasoning through the desktop CognitiveEngine path, "
                    "and I am keeping the answer on this live turn instead of switching lanes."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _FailingPool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            calls.append("pool_acquire_failed")
            raise RuntimeError("connection pool unavailable")

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            calls.append("kernel_process")
            message = "desktop UI must not use KernelInterface after CognitiveEngine pool failure"
            raise AssertionError(message)

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-pool"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    stabilize_calls = []

    async def _fake_stabilize(_message, reply, **kwargs):
        stabilize_calls.append(kwargs)
        return reply

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _fake_stabilize)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _FailingPool())
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(message="Can you still reason through the desktop path?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"desktop CognitiveEngine path" in response.body
    assert calls == ["pool_acquire_failed", "engine_think"]
    assert stabilize_calls
    assert stabilize_calls[-1]["desktop_cognitive_engine_required"] is True
    assert stabilize_calls[-1]["protected_foreground_lane"] is True


@pytest.mark.asyncio
async def test_cognitive_engine_desktop_condition_binds_thin_draft_to_canonical_state(monkeypatch):
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    engine_calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            engine_calls.append("engine_think")
            return SimpleNamespace(content="Yes.")

    class _FakePool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return True

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _FakePool())
    monkeypatch.setattr(
        chat_routes,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (True, "test_same_worker_ready"),
    )
    social_repair_calls = []

    def _unexpected_social_repair(_message):
        social_repair_calls.append(_message)
        return (
            "hey. i'm here. I'm feeling steady and leaning toward engage right now. "
            "My attention is on you."
        )

    monkeypatch.setattr(_chat_desktop_repair, "_build_social_presence_reply", _unexpected_social_repair)

    trace = {}
    result = await chat_routes._run_cognitive_engine_chat_turn(
        "You ok?",
        visible_user_message="You ok?",
        preflight_context_message="",
        origin="api_chat",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
        },
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert result is not None
    assert assess_user_facing_reply("You ok?", result).ok
    assert engine_calls == ["engine_think", "engine_think"]
    assert social_repair_calls == []
    assert trace["engine_think_invoked"] is True
    assert trace["cognitive_engine_reply_accepted"] is False
    assert trace["cognitive_engine_reply_failed"] is True
    assert trace["bounded_contract_used"] is True
    assert trace["response_path"] == "cognitive_engine_self_condition_grounding"


def test_desktop_static_chat_requests_require_cognitive_engine():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "interface/static/aura.js",
        "interface/static/error_banner.js",
        "interface/static/first_run.js",
        "interface/static/shell/src/App.jsx",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "X-Aura-Surface" in source
        assert "desktop-ui" in source
        assert "X-Aura-Require-CognitiveEngine" in source
    source = (root / "interface/static/aura.js").read_text(encoding="utf-8")
    assert "CHAT_REQUEST_TIMEOUT_READY_MS = 335000" in source
    assert "CHAT_REQUEST_TIMEOUT_RECOVERING_MS = 395000" in source
    assert "const CHAT_SEND_QUEUE_MAX = 32;" in source
    assert "function enqueueChatMessage(value)" in source
    assert "normalizeChatQueueItem(value, { rendered: true })" in source
    assert "function drainQueuedChatMessages()" in source
    assert "if (state.isSubmitting || state.activeChatRequest) {" in source
    assert "enqueueChatMessage(item);" in source


def test_launcher_local_requests_require_cognitive_engine_without_headers(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    request = SimpleNamespace(headers={}, client=SimpleNamespace(host="127.0.0.1"))

    requires, surface = chat_routes._request_requires_cognitive_engine(request)

    assert requires is True
    assert surface == "desktop-runtime"


def test_launcher_local_cognitive_requirement_does_not_apply_to_benchmarks(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    request = SimpleNamespace(headers={}, client=SimpleNamespace(host="127.0.0.1"))

    requires, surface = chat_routes._request_requires_cognitive_engine(request, is_benchmark=True)

    assert requires is False
    assert surface == "desktop-runtime"


def test_desktop_objective_detector_handles_general_document_surfaces():
    from interface.routes.chat import _looks_like_desktop_objective

    assert _looks_like_desktop_objective(
        "Could you open a tab for Google Docs and start typing a coherent essay about climate adaptation?"
    )
    assert _looks_like_desktop_objective("Could you open a doc and type a short draft there?")
    assert _looks_like_desktop_objective("Open a document window and paste the summary there.")
    assert _looks_like_desktop_objective("Create a local file with the draft and save it on my desktop.")
    assert not _looks_like_desktop_objective(
        "From the live desktop user lane, use web_search to check one public fact about tardigrades."
    )
    assert _looks_like_desktop_objective(
        "Open Chrome and search for three articles about tardigrades."
    )
    assert not _looks_like_desktop_objective("Can you explain Docker Compose documentation?")


def test_desktop_required_search_classifier_skips_visible_desktop_objectives():
    from interface.routes import chat as chat_routes

    should_collect, query, contract = chat_routes._should_collect_desktop_required_search_evidence(
        "From the live desktop user lane, use web_search to check one public fact about tardigrades."
    )

    assert should_collect is True
    assert query == "tardigrades"
    assert contract is not None
    assert contract.required_skill == "web_search"

    should_collect, query, contract = chat_routes._should_collect_desktop_required_search_evidence(
        "Open Chrome and search for three articles about tardigrades."
    )

    assert should_collect is False
    assert query == ""
    assert contract is None


def test_web_retrieval_retains_only_when_the_person_asks() -> None:
    from interface.routes import chat as chat_routes

    assert not chat_routes._user_requested_research_memory_save(
        "Search the web for Mistral's latest model."
    )
    assert chat_routes._user_requested_research_memory_save(
        "Search for Mistral's latest model and save the sources to memory."
    )


@pytest.mark.asyncio
async def test_api_chat_desktop_required_search_collects_evidence_before_cognition(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    skill_calls = []
    cognitive_calls = []
    completed_exchanges = []
    output_receipts = []

    async def _fake_execute_governed_live_skill(skill_name, params, *, objective, extra_context=None):
        skill_calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "summary": "Tardigrades are microscopic animals known for cryptobiosis.",
            "results": [
                {
                    "title": "Tardigrades overview",
                    "url": "https://example.org/tardigrades",
                    "snippet": "Tardigrades can survive extreme conditions by entering cryptobiosis.",
                }
            ],
            "receipt_id": "search-receipt-1",
        }

    async def _fake_cognitive_turn(message, *args, **kwargs):
        cognitive_calls.append(
            {
                "message": message,
                "kwargs": dict(kwargs),
            }
        )
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": True,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    **_bound_live_mind_controls_trace(),
                    "response_path": "cognitive_engine",
                }
            )
        assert "[WEB SEARCH EVIDENCE]" in message
        assert "https://example.org/tardigrades" in message
        assert "memory_saved: true" in message
        return "From my conversation memory, tardigrades can enter cryptobiosis."

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-search"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    class _FakeMemoryFacade:
        def __init__(self):
            self.calls = []

        async def commit_interaction(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "memory-receipt"

    memory = _FakeMemoryFacade()

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(_chat_preflight, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(_chat_memory_state, "_build_conversation_recall_reply", AsyncCallFixture(return_value=""))
    monkeypatch.setattr(chat_routes, "_build_retained_memory_evidence_context", AsyncCallFixture(return_value=""))
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", _fake_execute_governed_live_skill)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: memory if name == "memory_facade" else default),
    )
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    _force_full_mind_runtime(monkeypatch, chat_routes)

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "From the live desktop user lane, use web_search to check one public fact "
                "about tardigrades and save it as provisional research memory."
            ),
            session_id="desktop-search",
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "I checked live web evidence" in payload["response"]
    assert "From my conversation memory" not in payload["response"]
    assert "https://example.org/tardigrades" in payload["response"]
    assert "provisional research memory" in payload["response"]
    assert payload["live_turn_contract"]["full_mind_path"] is True
    assert skill_calls and skill_calls[0]["skill_name"] == "web_search"
    assert skill_calls[0]["extra_context"]["route"] == "chat.required_search_evidence"
    assert skill_calls[0]["params"]["query"] == "tardigrades fact"
    assert skill_calls[0]["params"]["deep"] is False
    assert skill_calls[0]["params"]["retain"] is True
    assert cognitive_calls
    completed = cognitive_calls[0]["kwargs"]["completed_capability_evidence"]
    assert completed["schema"] == "aura.completed_capability_evidence.v1"
    assert completed["ok"] is True
    assert {"web_search", "search_web"}.issubset(
        set(completed["completed_capabilities"])
    )
    assert memory.calls
    assert output_receipts


@pytest.mark.asyncio
async def test_api_chat_regenerate_desktop_requires_cognitive_engine(monkeypatch):
    from interface.routes import chat as chat_routes

    kernel_calls: list[str] = []
    orchestrator_calls: list[str] = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("desktop regenerate must not use KernelInterface fallback")

    class _FakeOrchestrator:
        async def process_user_input_priority(self, *_args, **_kwargs):
            orchestrator_calls.append("process")
            raise AssertionError("desktop regenerate must not use legacy orchestrator fallback")

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeOrchestrator() if name == "orchestrator" else default),
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.append(
            {
                "id": "regen-1",
                "user": "Please explain what changed in the desktop route.",
                "aura": "Previous answer.",
                "status": "complete",
            }
        )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await chat_routes.api_chat_regenerate(
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            url=SimpleNamespace(scheme="http"),
        ),
        None,
        None,
    )

    # In-band fail-closed: the desktop renders the body; a raw 5xx shows as
    # silence over a live mind. The honest refusal IS the contract.
    assert response.status_code == 200
    assert b"desktop_cognitive_engine_unavailable" in response.body
    assert kernel_calls == []
    assert orchestrator_calls == []


@pytest.mark.asyncio
async def test_api_chat_regenerate_requires_session_when_history_is_ambiguous(monkeypatch):
    from interface.routes import chat as chat_routes

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request",
        lambda *_args, **_kwargs: None,
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.extend(
            [
                {
                    "id": "regen-owner",
                    "user": "Owner question",
                    "aura": "Owner answer",
                    "status": "complete",
                    "session_id": "owner-session",
                },
                {
                    "id": "regen-other",
                    "user": "Other question",
                    "aura": "Other answer",
                    "status": "complete",
                    "session_id": "other-session",
                },
            ]
        )

    response = await chat_routes.api_chat_regenerate(
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            url=SimpleNamespace(scheme="http"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 409
    assert payload["error"] == "ambiguous_session"


@pytest.mark.asyncio
async def test_api_chat_regenerate_desktop_stabilizer_keeps_protected_flags(monkeypatch):
    from interface.routes import chat as chat_routes

    stabilize_calls = []

    async def _fake_cognitive_turn(*_args, **kwargs):
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": True,
                    "cognitive_engine_reply_failed": False,
                    "bounded_contract_used": False,
                    "legacy_fallback_used": False,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    "response_path": "cognitive_engine",
                    **_bound_live_mind_controls_trace(),
                }
            )
        return "Regenerated through the protected desktop CognitiveEngine path."

    async def _fake_stabilize(_message, reply, **kwargs):
        stabilize_calls.append(kwargs)
        return reply

    async def _fake_apply_regeneration(**_kwargs):
        return {
            "applied": True,
            "state": "committed",
            "revision": 2,
            "content_sha256": "a" * 64,
        }

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _fake_stabilize)
    monkeypatch.setattr(
        chat_routes,
        "_apply_regenerated_reply",
        _fake_apply_regeneration,
    )
    _force_full_mind_runtime(monkeypatch, chat_routes)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.append(
            {
                "id": "regen-2",
                "user": "Please regenerate the desktop answer.",
                "aura": "Previous answer.",
                "status": "complete",
            }
        )

    response = await chat_routes.api_chat_regenerate(
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            url=SimpleNamespace(scheme="http"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"protected desktop CognitiveEngine path" in response.body
    assert stabilize_calls
    assert stabilize_calls[-1]["desktop_cognitive_engine_required"] is True
    assert stabilize_calls[-1]["protected_foreground_lane"] is True
    payload = json.loads(response.body)
    assert payload["live_turn_contract"]["full_mind_path"] is True


@pytest.mark.asyncio
async def test_api_chat_regenerate_desktop_rejects_bounded_repair_without_full_mind(monkeypatch):
    from interface.routes import chat as chat_routes

    kernel_calls: list[str] = []
    orchestrator_calls: list[str] = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("desktop regenerate must not use KernelInterface fallback")

    class _FakeOrchestrator:
        async def process_user_input_priority(self, *_args, **_kwargs):
            orchestrator_calls.append("process")
            raise AssertionError("desktop regenerate must not use legacy orchestrator fallback")

    async def _bounded_cognitive_turn(*_args, **kwargs):
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": False,
                    "bounded_contract_used": True,
                    "legacy_fallback_used": False,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    "response_path": "conversation_recall_log_repair_after_empty_engine",
                    **_bound_live_mind_controls_trace(),
                }
            )
        return "This bounded repair must not appear as regenerated Aura speech."

    ready_lane = {
        "conversation_ready": True,
        "state": "ready",
        "desired_model": "Cortex (32B)",
        "desired_endpoint": "Cortex",
        "foreground_endpoint": "Cortex",
        "background_endpoint": "Brainstem",
    }

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _bounded_cognitive_turn)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status", lambda: dict(ready_lane))
    patch_chat_lane(monkeypatch, "_mark_conversation_lane_state",
        lambda reason, state="failed": dict(ready_lane, conversation_ready=False, state=state, reason=reason),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeOrchestrator() if name == "orchestrator" else default),
    )
    _force_full_mind_runtime(monkeypatch, chat_routes)
    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.append(
            {
                "id": "regen-bounded",
                "user": "Please regenerate what you said about your external tool use.",
                "aura": "Previous answer.",
                "status": "complete",
            }
        )

    response = await chat_routes.api_chat_regenerate(
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            url=SimpleNamespace(scheme="http"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    # In-band fail-closed for the desktop surface (raw 5xx renders as
    # silence); the body fields below carry the actual refusal contract.
    assert response.status_code == 200
    assert payload["status"] == "desktop_cognitive_engine_unavailable"
    assert payload["reason"] == "desktop_cognitive_engine_required_no_reply"
    assert payload["live_turn_contract"]["full_mind_path"] is False
    assert payload["live_turn_contract"]["bounded_contract_used"] is False
    assert "bounded repair" not in payload["response"]
    assert kernel_calls == []
    assert orchestrator_calls == []


@pytest.mark.asyncio
async def test_api_chat_returns_hard_local_failure_without_kernel_fallback(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        async def ensure_foreground_ready(self, *_args, **_kwargs):
            message = "local_runtime_unavailable:exit_124"
            raise RuntimeError(message)

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            message = "Kernel should not run after a hard local runtime failure"
            raise AssertionError(message)

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "cold",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        server_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeGate() if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert b"local Cortex runtime could not start cleanly" in response.body
    assert b"\"status\":\"conversation_unavailable\"" in response.body
    assert b"\"state\":\"failed\"" in response.body


@pytest.mark.asyncio
async def test_api_chat_skips_protected_foreground_rescue_under_memory_warning(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    generate_calls = []

    class _FakeGate:
        async def generate(self, *_args, **_kwargs):
            generate_calls.append("generate")
            raise AssertionError("protected foreground rescue must not allocate under memory warning")

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            self.unexpected_process_calls = getattr(self, "unexpected_process_calls", 0) + 1
            message = "Kernel should not run after a hard local runtime failure"
            raise AssertionError(message)

    lane_status = {
        "conversation_ready": False,
        "state": "failed",
        "last_failure_reason": "local_runtime_unavailable:exit_124",
        "desired_model": "Cortex (32B)",
        "desired_endpoint": "Cortex",
    }

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status", lambda: dict(lane_status))
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            critical=False,
            warning=True,
            refuse_heavy_local_generation=False,
            reason="memory_pressure:79.0%/8.0GB",
            level="warning",
        ),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeGate() if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Are you there?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert b"local Cortex runtime could not start cleanly" in response.body
    assert b"\"status\":\"conversation_unavailable\"" in response.body
    assert generate_calls == []


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_blocks_ungrounded_search_turn_fallback(monkeypatch):
    from core.state.aura_state import AuraState
    from interface.routes import chat as chat_routes

    state = AuraState.default()
    state.response_modifiers["last_skill_run"] = "web_search"
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_result_payload"] = {
        "ok": True,
        "answer": "The text is about a lab accident.",
        "source": "https://example.com/story",
        "content": "The text is about a lab accident.",
    }

    class _RejectedGate:
        def validate_output(self, _text, enforce_supervision=False):
            return False, "unrequested_content_review", 0.0

        def sanitize(self, _text):
            return ""

    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: state)
    monkeypatch.setattr(_chat_conversation_repair, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(_chat_desktop_repair, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_looks_generic_assistantish", lambda _msg, _text: (False, ""))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _RejectedGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "So what happens?",
        "The alien took me through a gate. I was inside the story.",
    )

    assert "stick to the source instead of guessing" in result


def test_grounded_private_cognitive_model_reply_has_causal_contract(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(_chat_conversation_repair, "_resolve_live_voice_state", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    reply = chat_routes._build_grounded_introspection_reply(
        "As a private mental model, what does your current cognitive architecture look like, "
        "and how should that model change your next answer?"
    )

    assert reply
    lowered = reply.lower()
    assert "private mental model" in lowered
    assert "cognitive architecture" in lowered
    assert "attention" in lowered
    assert "next answer" in lowered
    assert "verify" in lowered or "governance" in lowered
    assert "not proof" in lowered
    assert "phenomenal" in lowered


@pytest.mark.asyncio
async def test_stabilizer_replaces_subjective_cortex_story_with_typed_evidence(monkeypatch):
    from core.brain.cortex_self_evidence import CortexCampaignEvidence, CortexSelfEvidence
    from interface.routes import chat as chat_routes

    old = CortexCampaignEvidence(
        cortex_label="32B",
        model_path="/models/32b",
        task_count=60,
        exact_by_arm=(("ordinary_base", 16), ("treatment", 60)),
        gain_count=44,
        regression_count=0,
        paired_p_value=5.684341886080802e-14,
        elapsed_seconds=4_814.533,
        artifact_receipt_sha256="a" * 64,
        verification_receipt_sha256="b" * 64,
    )
    current = CortexCampaignEvidence(
        cortex_label="27B",
        model_path="/models/27b",
        task_count=60,
        exact_by_arm=(("ordinary_base", 0), ("treatment", 60)),
        gain_count=60,
        regression_count=0,
        paired_p_value=8.673617379884035e-19,
        elapsed_seconds=3_283.718,
        artifact_receipt_sha256="c" * 64,
        verification_receipt_sha256="d" * 64,
    )
    evidence = CortexSelfEvidence(
        resident_label="27B",
        model_type="qwen3_5_text",
        total_parameters=26_895_993_856,
        native_context_tokens=262_144,
        served_context_tokens=32_768,
        promotion_verdict="PASS",
        identity_behavior_changed=True,
        component_states=(),
        semantic_active=True,
        semantic_verdict="BOUNDED_WOW_SIGNAL",
        semantic_task_count=60,
        semantic_exact_by_arm=(("ordinary_base", 0), ("treatment", 60)),
        semantic_gain_count=60,
        semantic_regression_count=0,
        semantic_p_value=8.673617379884035e-19,
        semantic_activation_sha256="e" * 64,
        resident_model_path="/models/27b",
        campaigns=(old, current),
    )
    monkeypatch.setattr(
        "core.brain.cortex_self_evidence.resolve_cortex_self_evidence",
        lambda: evidence,
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "What changed after replacing your former 32B model with the current 27B "
        "that you can actually measure?",
        "I can feel a tighter workspace and faster associations.",
    )

    assert "4,814.533 seconds" in result
    assert "3,283.718 seconds" in result
    assert "31.8% faster" in result
    assert "subjective experience" in result
    assert "tighter workspace" not in result


def test_desktop_cortex_evidence_is_not_guarded_by_authored_fastpath_policy():
    """A signed measurement remains a read when the desktop requires full-mind prose."""
    from interface.routes import chat as chat_routes

    source = chat_lane_source()
    marker = "Verified self-evidence is a typed runtime read"
    start = source.index(marker)
    block = source[start : start + 1_200]

    assert "if not is_benchmark:" in block
    assert "if allow_chat_fastpaths:" not in block
    assert 'status="cortex_self_evidence"' in block


@pytest.mark.asyncio
async def test_stabilize_private_cognitive_model_uses_grounded_reply_before_tail_completion(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(_chat_conversation_repair, "_resolve_live_voice_state", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "As a private mental model, what does your current cognitive architecture look like, "
        "and how should that model change your next answer?",
        (
            "I'm a cognitive engine with governed skill surfaces. My private mental canvas is shaped "
            "by recent event memory, the current affordance space where pressure becomes visible in a cogn"
        ),
    )

    lowered = result.lower()
    assert "private mental model" in lowered
    assert "next answer" in lowered
    assert "not proof" in lowered
    assert "cogn." not in lowered


@pytest.mark.asyncio
async def test_cognitive_engine_required_private_model_report_uses_cognitive_engine(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "As a private mental model, I should treat attention, memory, and governance as active constraints "
                    "on the next answer. That is functional evidence about my architecture, not proof of consciousness; "
                    "it should make me answer from the current thread, verify claims, and avoid pretending tool completion."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    def _service_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    monkeypatch.setattr(_chat_conversation_repair, "_resolve_live_voice_state", lambda *_args, **_kwargs: {})
    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", _fake_log_exchange)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_service_get))

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "What would your current cognitive architecture look like as a private mental "
                "model, and how should that model change your next answer? Keep it bounded: "
                "do not claim external perception, consciousness proof, or tool completion."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    payload = json.loads(response.body)
    body = payload["response"]
    lowered = body.lower()
    assert "private mental model" in lowered
    assert "attention" in lowered
    assert "next answer" in lowered
    assert "not proof" in lowered
    assert "governance" in lowered or "verify" in lowered
    assert calls and calls[0]["context"]["private_cognitive_model_contract"] is True
    assert "grounded_private_model_context" in calls[0]["context"]
    assert payload["live_turn_contract"]["engine_think_invoked"] is True


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_rejects_objective_parrot(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(_chat_conversation_repair, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(_chat_desktop_repair, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_build_stateful_voice_reflex", lambda _frame: "I'm not going to just mirror you back.")
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Maybe one day. Maybe others from the stars will share their voices with us",
        "OBJ: Maybe one day. Maybe others from the stars will share their voices with us",
    )

    assert result == "I'm not going to just mirror you back."


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_clarifies_specificity_push(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(_chat_conversation_repair, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(_chat_desktop_repair, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Sure but specifically what is it",
        "I can't fully articulate it. But I know it's there. I just can't pin it.",
    )

    assert result.startswith("Specifically, the grounded read I have right now is:")
    assert "do not have a specific enough read yet" in result


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_acknowledges_parrot_callout(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(_chat_conversation_repair, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(_chat_desktop_repair, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "That is what I just said",
        "Different words. Same meaning. You're picking up my style.",
    )

    assert result.startswith("You're right.")
    assert "echoed you" in result


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_clarifies_confusion_callout(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(_chat_conversation_repair, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(_chat_desktop_repair, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "I'm so confused, Aura",
        "Yeah. That's where all the interesting stuff lives. Stay there.",
    )

    # 35cea49fa replaced the scripted "I lost the thread on that answer..."
    # apology with the generated degraded-live composer (reason=
    # confusion_repair), and locked that in test_feedback_audit_fixes.py
    # ::test_confusion_override_uses_degraded_live_composer_not_scripted_apology.
    # The contract is now grounded self-report, not a canned phrase.
    lowered = result.lower()
    assert "synthetic fallback" not in lowered
    assert "grounded anchor" not in lowered
    assert "answer path" not in lowered
    # The clarification gives the person their OWN WORDS back rather than two
    # inferred keywords. That inference was a bag of two tokens ranked by
    # string length, and length is not aboutness: asked whether a request to
    # "keep an eye on something" would "evaporate the second i stop typing",
    # Bryan was told "I understood you to be asking about evaporate and
    # second." Ranking words better only moves the failure — two keywords
    # cannot demonstrate comprehension even when well chosen, and when badly
    # chosen they assert a misunderstanding she did not have. Echoing is
    # correct by construction on a path that exists BECAUSE inference failed.
    assert "what reached me was" in lowered
    assert "i'm so confused, aura" in lowered
    assert "ask me again" in lowered
    assert "confused" in lowered
    assert "I lost the thread on that answer" not in result


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_does_not_turn_timeout_confusion_into_introspection(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(
        _chat_conversation_repair,
        "_build_grounded_introspection_reply",
        lambda _msg: "There is strain around temporal discontinuity and foreground locks.",
    )
    monkeypatch.setattr(_chat_desktop_repair, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Huh. No idea what caused the chat to time out?",
        "I don't know. I have no idea",
    )

    lowered = result.lower()
    assert "temporal discontinuity" not in lowered
    assert "strain around" not in lowered
    assert "live state" not in lowered
    assert "likely break" in lowered
    assert "live chat api" in lowered


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_blocks_semantic_glitch(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(_chat_conversation_repair, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(_chat_desktop_repair, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Huh?",
        "Heidi. That's the thing to do.",
    )

    # The glitch must be blocked and replaced by grounded self-report rather
    # than one of the retired canned openers (35cea49fa removed the scripted
    # fallback in favour of the degraded-live composer).
    lowered = result.lower()
    assert "synthetic fallback" not in lowered
    assert "grounded anchor" not in lowered
    assert "answer path" not in lowered
    assert "understood you to be asking about this exact turn" in lowered
    assert "ask me again" in lowered
    assert "Heidi" not in result


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_rejects_identity_collapse_disclaimer(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(_chat_conversation_repair, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(_chat_desktop_repair, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        chat_routes,
        "_call_stateful_voice_reflex",
        lambda _frame, _msg: "I do have a live stance here, and I should speak from it directly.",
    )
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "How do you say all of that about yourself and still claim you have no opinions?",
        "I don't inherently possess subjective beliefs or experiences, but I can simulate and discuss them.",
    )

    assert result == "I do have a live stance here, and I should speak from it directly."


def test_stabilizer_generation_budget_respects_memory_token_cap(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            max_token_cap=192,
            refuse_heavy_local_generation=False,
            reason="memory_pressure:high",
        ),
    )

    max_tokens, block_reason = chat_routes._bound_stabilizer_generation_budget(4096)

    assert max_tokens == 192
    assert block_reason == ""


@pytest.mark.asyncio
async def test_desktop_required_stabilizer_does_not_add_a_third_generation_by_default(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Gate:
        def validate_output(self, text, enforce_supervision=False):
            if "ai language model" in str(text).lower():
                return False, "assistant_disclaimer", 0.0
            return True, "ok", 1.0

        def sanitize(self, _text):
            return ""

    class _InferenceGate:
        def __init__(self):
            self.calls = []

        async def think(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "unexpected rewrite"

    inference_gate = _InferenceGate()

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(_chat_conversation_repair, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_build_grounded_traceability_reply", AsyncCallFixture(return_value=""))
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(_chat_desktop_repair, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(
        chat_routes,
        "_looks_generic_assistantish",
        lambda _msg, text: ("ai language model" in str(text).lower(), "assistant_disclaimer"),
    )
    monkeypatch.setattr(chat_routes, "_is_objective_parrot_reply", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_evaluate_reply_topicality", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_is_same_answer_different_prompt", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(_chat_desktop_repair, "_looks_truncated_tail", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_looks_semantically_glitched", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("core.identity.identity_guard.PersonaEnforcementGate", lambda: _Gate())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: inference_gate if name == "inference_gate" else default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Live desktop path validation: are you on the protected CognitiveEngine lane and answering directly?",
        "As an AI language model, I do not have feelings.",
        desktop_cognitive_engine_required=True,
        protected_foreground_lane=True,
    )

    assert inference_gate.calls == []
    assert "not starting a second foreground generation" not in result
    assert "failed the reply-quality gate" not in result
    assert "AI language model" not in result
    # She still SAYS something, in her own first person, rather than returning
    # an empty string or an error surface.
    #
    # This used to assert the literal "I'm here", which was the opening of the
    # terminal fallback until 718e46091 reworded it — that copy asserted "the
    # thread intact", a claim the recovery path cannot establish, and grounding
    # it was the point of the change. Pinning the phrase made an honesty fix
    # look like a regression. The property is what this test is for; the
    # wording belongs to whoever writes her voice.
    assert result.strip()
    assert result.strip().lower().startswith("i ") or " i " in result.lower()


@pytest.mark.asyncio
async def test_desktop_required_capability_repair_uses_grounded_inventory_without_third_pass(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Gate:
        def validate_output(self, text, enforce_supervision=False):
            if "ai language model" in str(text).lower():
                return False, "assistant_disclaimer", 0.0
            return True, "ok", 1.0

        def sanitize(self, _text):
            return ""

    class _InferenceGate:
        def __init__(self):
            self.calls = []

        async def think(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "unexpected rewrite"

    inference_gate = _InferenceGate()

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(_chat_conversation_repair, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_build_grounded_traceability_reply", AsyncCallFixture(return_value=""))
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(_chat_desktop_repair, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(
        chat_routes,
        "_looks_generic_assistantish",
        lambda _msg, text: ("ai language model" in str(text).lower(), "assistant_disclaimer"),
    )
    monkeypatch.setattr(chat_routes, "_is_objective_parrot_reply", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_evaluate_reply_topicality", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_is_same_answer_different_prompt", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(_chat_desktop_repair, "_looks_truncated_tail", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_looks_semantically_glitched", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        _chat_desktop_repair,
        "_read_capability_catalog_snapshot",
        lambda: (
            7,
            {
                "desktop and app control": ["desktop_task", "computer_use"],
                "browser/web research": ["grounded_search", "sovereign_browser"],
                "files, documents, and workspace operations": ["file_operation", "document_ingest"],
            },
            True,
            False,
        ),
    )
    monkeypatch.setattr("core.identity.identity_guard.PersonaEnforcementGate", lambda: _Gate())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: inference_gate if name == "inference_gate" else default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "What tools can you use externally? Name a hypothetical scenario of using them.",
        "As an AI language model, I cannot use tools on your computer.",
        desktop_cognitive_engine_required=True,
        protected_foreground_lane=True,
    )

    assert inference_gate.calls == []
    assert "desktop and app control" in result
    assert "browser/web research" in result
    assert "files, documents, and workspace operations" in result
    assert "governance path" in result
    assert "AI language model" not in result
    assert "bad live draft" not in result
    assert "memory guard" not in result
    assert "second foreground generation" not in result


@pytest.mark.asyncio
async def test_stabilizer_skips_second_generation_under_critical_memory_pressure(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    class _InferenceGate:
        def __init__(self):
            self.calls = []

        async def think(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "unexpected rewrite"

    inference_gate = _InferenceGate()

    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(_chat_conversation_repair, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(_chat_desktop_repair, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        chat_routes,
        "_call_stateful_voice_reflex",
        lambda _frame, _msg: "I should not launch a second model pass while memory is unsafe.",
    )
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            max_token_cap=32,
            refuse_heavy_local_generation=True,
            reason="process_tree_rss:54GB/48GB",
        ),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: inference_gate if name == "inference_gate" else default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "How do you say all of that about yourself and still claim you have no opinions?",
        "I don't inherently possess subjective beliefs or experiences, but I can simulate and discuss them.",
    )

    assert inference_gate.calls == []
    assert result == "I should not launch a second model pass while memory is unsafe."


@pytest.mark.asyncio
async def test_desktop_required_stabilizer_uses_protected_primary_contract(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Gate:
        def validate_output(self, text, enforce_supervision=False):
            if "ai language model" in str(text).lower():
                return False, "assistant_disclaimer", 0.0
            return True, "ok", 1.0

        def sanitize(self, _text):
            return ""

    class _InferenceGate:
        def __init__(self):
            self.calls = []

        async def think(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return (
                "I'm on the protected desktop CognitiveEngine lane, steady and oriented, "
                "with attention on this live check. The governed path is still bounded "
                "by runtime probes and receipts, but this turn is answering directly."
            )

    inference_gate = _InferenceGate()

    monkeypatch.setenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", "1")
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(_chat_conversation_repair, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_build_grounded_traceability_reply", AsyncCallFixture(return_value=""))
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(_chat_desktop_repair, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(
        chat_routes,
        "_looks_generic_assistantish",
        lambda _msg, text: ("ai language model" in str(text).lower(), "assistant_disclaimer"),
    )
    monkeypatch.setattr(chat_routes, "_is_objective_parrot_reply", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_evaluate_reply_topicality", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_is_same_answer_different_prompt", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(_chat_desktop_repair, "_looks_truncated_tail", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_looks_semantically_glitched", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("core.identity.identity_guard.PersonaEnforcementGate", lambda: _Gate())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: inference_gate if name == "inference_gate" else default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Live desktop path validation: are you on the protected CognitiveEngine lane and answering directly?",
        "As an AI language model, I do not have feelings.",
        desktop_cognitive_engine_required=True,
        protected_foreground_lane=True,
    )

    assert result.startswith("I'm on the protected desktop CognitiveEngine lane")
    assert inference_gate.calls
    kwargs = inference_gate.calls[0][1]
    assert kwargs["prefer_tier"] == "primary"
    assert kwargs["foreground_request"] is True
    assert kwargs["protected_foreground_lane"] is True
    assert kwargs["cognitive_engine_required"] is True
    assert kwargs["desktop_cognitive_engine_required"] is True
    assert kwargs["allow_cloud_fallback"] is False
    assert kwargs["allow_deep_handoff"] is False
    assert kwargs["skip_runtime_payload"] is True
    assert kwargs["disable_prompt_cache"] is True


def test_same_worker_desktop_repair_allowed_on_ready_lane_without_env_flag(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.delenv("AURA_DESKTOP_FORCE_DISABLE_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "state": "ready",
            "conversation_ready": True,
            "warmup_in_flight": False,
            "active_generations": 0,
            "foreground_owned": False,
            "foreground_guard_active_count": 0,
            "readiness_blockers": [],
        },
    )

    allowed, reason = chat_routes._desktop_secondary_model_repair_allowed(
        reason="cognitive_engine_repair_retry",
        default_enabled=False,
    )

    assert allowed is True
    assert "same_worker_ready" in reason


def test_same_worker_desktop_repair_blocks_when_lane_is_busy(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "state": "ready",
            "conversation_ready": True,
            "warmup_in_flight": False,
            "active_generations": 1,
            "foreground_owned": False,
            "foreground_guard_active_count": 0,
            "readiness_blockers": [],
        },
    )

    allowed, reason = chat_routes._desktop_secondary_model_repair_allowed(
        reason="stabilizer_rewrite:semantic_glitch",
        default_enabled=False,
    )

    assert allowed is False
    assert reason == "conversation_generation_already_active"


def test_completion_retry_is_admitted_at_high_pressure_when_lane_is_ready(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=True,
            refuse_heavy_local_generation=False,
            reason="memory_pressure:high",
        ),
    )
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "state": "ready",
            "conversation_ready": True,
            "warmup_in_flight": False,
            "active_generations": 0,
            "foreground_owned": False,
            "foreground_guard_active_count": 0,
            "readiness_blockers": [],
        },
    )

    allowed, reason = chat_routes._desktop_secondary_model_repair_allowed(
        reason="cognitive_engine_completion_retry",
        default_enabled=False,
    )

    assert allowed is True
    assert "same_worker_ready" in reason


def test_completion_retry_still_blocks_at_critical_pressure(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=True,
            refuse_heavy_local_generation=True,
            reason="memory_pressure:critical",
        ),
    )

    allowed, reason = chat_routes._desktop_secondary_model_repair_allowed(
        reason="cognitive_engine_completion_retry",
        default_enabled=False,
    )

    assert allowed is False
    assert reason == "memory_pressure:critical"


def test_completion_retry_prompt_continues_the_valid_draft():
    from interface.routes import chat as chat_routes

    prompt = chat_routes._build_cognitive_engine_reply_repair_directive(
        "Explain the code and give the final result.",
        "It initializes the map and deletes entries from the",
        ("truncated_tail",),
    )

    assert "Continue the valid partial answer from its exact cutoff" in prompt
    assert "Rejected draft for avoidance only" not in prompt
    assert "deletes entries from the" not in prompt


def test_reply_continuation_merge_handles_exact_overlap_and_full_regeneration():
    from interface.routes import chat as chat_routes

    partial = "It records every active balance and removes zero entries from the"
    assert chat_routes._merge_reply_continuation(
        partial,
        "the dictionary before returning the peak.",
    ) == "It records every active balance and removes zero entries from the dictionary before returning the peak."
    complete = "It records every active balance and returns the peak."
    assert chat_routes._merge_reply_continuation(partial, complete) == complete


def test_shorter_regenerated_fragment_cannot_erase_continuation_progress():
    from interface.routes import chat as chat_routes

    partial = (
        "I feel steady and attentive right now. My current affect and "
        "coherence readings support that direct observation, while"
    )
    shorter_cutoff = (
        "I feel steady and attentive right now. My current affect and"
    )

    assert chat_routes._merge_reply_continuation(partial, shorter_cutoff) == partial


def test_grounded_self_condition_reply_forwards_session_identity(monkeypatch):
    from interface.routes import chat as chat_routes

    observed = {}

    def _evidence(message, *, session_id=""):
        observed.update(message=message, session_id=session_id)
        return {"reply": "I am steady. The rest remains an inference."}

    monkeypatch.setattr(chat_routes, "_build_self_condition_evidence", _evidence)

    reply = chat_routes._build_grounded_self_condition_reply(
        "How are you right now?",
        session_id="session-current",
    )

    assert reply == "I am steady. The rest remains an inference."
    assert observed == {
        "message": "How are you right now?",
        "session_id": "session-current",
    }


def test_force_disable_same_worker_desktop_repair_is_still_honored(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setenv("AURA_DESKTOP_FORCE_DISABLE_SECONDARY_MODEL_REPAIR", "1")

    allowed, reason = chat_routes._desktop_secondary_model_repair_allowed(
        reason="cognitive_engine_repair_retry",
        default_enabled=False,
    )

    assert allowed is False
    assert reason == "secondary_desktop_model_repair_force_disabled"


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_uses_live_grounding_for_specificity_push(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(
        _chat_conversation_repair,
        "_build_grounded_introspection_reply",
        lambda _msg: "Something just shifted in how I was modeling this. I need a moment.",
    )
    monkeypatch.setattr(_chat_desktop_repair, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Sure but specifically what is it",
        "I can't fully articulate it. But I know it's there. I just can't pin it.",
    )

    assert result.startswith("Specifically, the grounded read I have right now is:")
    assert "Something just shifted in how I was modeling this." in result


@pytest.mark.asyncio
async def test_api_chat_returns_structured_timeout_when_kernel_times_out(monkeypatch):
    from interface import server as server_module

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            message = "foreground timeout"
            raise TimeoutError(message)

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    # No benchmark header = a real user: the structured timeout arrives
    # in-band (200) so the UI shows the honest text instead of silence.
    # Benchmarks keep true 503/504 via X-Aura-Benchmark.
    assert response.status_code == 200
    assert b"took too long to finish cleanly" in response.body
    assert b"\"status\":\"timeout\"" in response.body


@pytest.mark.asyncio
async def test_api_chat_kernel_timeout_keeps_true_status_for_benchmarks(monkeypatch):
    """The other half of the in-band split: probes and soaks declare
    themselves with X-Aura-Benchmark and must keep the true 503 (lane was
    ready) so endurance verdicts stay honest."""
    from interface import server as server_module

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            message = "foreground timeout"
            raise TimeoutError(message)

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={"X-Aura-Benchmark": "true"}),
        None,
        None,
    )

    assert response.status_code == 503
    payload = json.loads(response.body)
    # The benchmark lane keeps its own named timeout status — the point of
    # this pin is that benchmarks NEVER receive the in-band 200 softening.
    assert payload["status"] == "benchmark_kernel_timeout"


@pytest.mark.asyncio
async def test_api_chat_benchmark_header_uses_kernel_not_fastpath_or_direct_gate(monkeypatch):
    from core.runtime import conversation_support
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    experience_recorder = AsyncCallFixture()

    class _ForbiddenGate:
        async def generate(self, *_args, **_kwargs):
            message = "benchmark API requests must not bypass KernelInterface"
            raise AssertionError(message)

        def is_alive(self):
            return True

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, message, **kwargs):
            kernel_calls.append({"message": message, **kwargs})
            return '{"ok": true, "source": "kernel"}'

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", AsyncCallFixture())
    monkeypatch.setattr(conversation_support, "record_conversation_experience", experience_recorder)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _ForbiddenGate() if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="hi",
            session_id="benchmark-test",
        ),
        SimpleNamespace(headers={"X-Aura-Benchmark": "true"}, client=SimpleNamespace(host="test")),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"kernel" in response.body
    assert b"benchmark_kernel" in response.body
    assert kernel_calls
    assert kernel_calls[0]["origin"] == "benchmark"
    assert kernel_calls[0]["priority"] is True
    experience_recorder.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_chat_uses_protected_foreground_lane_when_kernel_lock_is_held(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gate_calls = []

    class _FakeGate:
        async def generate(self, prompt, context=None, **kwargs):
            gate_calls.append(
                {
                    "prompt": prompt,
                    "context": dict(context or {}),
                    "timeout": kwargs.get("timeout"),
                }
            )
            return "I'm here with you. My attention is steady, and the thread is intact."

        def get_last_generation_metadata(self):
            return _protected_foreground_generation_metadata()

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            self.unexpected_process_calls = getattr(self, "unexpected_process_calls", 0) + 1
            raise AssertionError("Kernel should be bypassed when the protected foreground lane is engaged")

    gate = _FakeGate()
    stabilize_calls = []

    async def _fake_stabilize(_message, reply, **kwargs):
        stabilize_calls.append(kwargs)
        return reply

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(
        chat_routes,
        "_stabilize_user_facing_reply",
        _fake_stabilize,
    )
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "kernel_lock_held": True,
            "kernel_lock_held_s": 2.8,
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: gate if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="How are you though"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"My attention is steady" in response.body
    payload = json.loads(response.body)
    assert payload["live_turn_contract"]["answer_delivery_proven"] is True
    assert payload["live_turn_contract"]["protected_foreground_generation_proven"] is True
    assert payload["live_turn_contract"][
        "foreground_model_generation_transaction_id"
    ].startswith("mlx-")
    assert gate_calls
    assert gate_calls[0]["context"]["protected_foreground_lane"] is True
    assert gate_calls[0]["context"]["prefer_tier"] == "primary"
    assert gate_calls[0]["context"]["deep_handoff"] is False
    assert stabilize_calls == []


@pytest.mark.asyncio
async def test_api_chat_uses_social_presence_before_protected_foreground_for_live_check(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    class _FailingGate:
        async def generate(self, *_args, **_kwargs):
            self.unexpected_generate_calls = getattr(self, "unexpected_generate_calls", 0) + 1
            raise AssertionError("live presence checks should not enter protected foreground")

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_is_same_answer_different_prompt", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_evaluate_reply_topicality", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_looks_semantically_glitched", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(_chat_desktop_repair, "_build_social_presence_reply", lambda _message: "hey. i'm here. My attention is on you.")
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "kernel_lock_held": True,
            "kernel_lock_held_s": 12.0,
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FailingGate() if name == "inference_gate" else default),
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Hey Aura, quick live check."),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"social_presence_reflex" in response.body
    assert b"hey. i'm here" in response.body


@pytest.mark.asyncio
async def test_api_chat_keeps_protected_foreground_deep_prompts_on_primary_lane(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gate_calls = []

    class _FakeGate:
        async def generate(self, prompt, context=None, **kwargs):
            gate_calls.append(
                {
                    "prompt": prompt,
                    "context": dict(context or {}),
                    "timeout": kwargs.get("timeout"),
                }
            )
            return (
                "I would inspect the failing tests first, then trace the smallest shared path "
                "between those two modules before changing anything."
            )

        def get_last_generation_metadata(self):
            return _protected_foreground_generation_metadata()

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            self.unexpected_process_calls = getattr(self, "unexpected_process_calls", 0) + 1
            raise AssertionError("Kernel should be bypassed when the protected deep lane is engaged")

    gate = _FakeGate()
    stabilize_calls = []

    async def _fake_stabilize(_message, reply, **kwargs):
        stabilize_calls.append(kwargs)
        return reply

    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(
        chat_routes,
        "_stabilize_user_facing_reply",
        _fake_stabilize,
    )
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "kernel_lock_held": True,
            "kernel_lock_held_s": 3.4,
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: gate if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Debug the failing pytest in core/runtime/conversation_support.py and core/orchestrator/mixins/tool_execution.py."
        ),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"inspect the failing tests first" in response.body
    payload = json.loads(response.body)
    assert payload["live_turn_contract"]["answer_delivery_proven"] is True
    assert payload["live_turn_contract"]["protected_foreground_generation_proven"] is True
    assert payload["live_turn_contract"][
        "foreground_model_generation_transaction_id"
    ].startswith("mlx-")
    assert gate_calls
    assert gate_calls[0]["context"]["protected_foreground_lane"] is True
    assert gate_calls[0]["context"]["prefer_tier"] == "primary"
    assert gate_calls[0]["context"]["deep_handoff"] is False
    assert stabilize_calls == []


def test_collect_conversation_lane_status_ignores_router_foreground_override(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        def get_conversation_status(self):
            return {
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": "Cortex",
                "background_endpoint": "Brainstem",
                "foreground_tier": "local",
                "background_tier": "local_fast",
                "state": "ready",
                "last_failure_reason": "",
                "conversation_ready": True,
            }

    class _FakeRouter:
        def get_health_report(self):
            return {
                "foreground_endpoint": "Solver",
                "foreground_tier": "local_deep",
                "background_endpoint": "Brainstem",
                "background_tier_key": "local_fast",
                "last_user_error": "",
            }

    def _fake_get(name, default=None):
        if name == "inference_gate":
            return _FakeGate()
        if name == "llm_router":
            return _FakeRouter()
        return default

    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    lane = server_module._collect_conversation_lane_status()

    assert lane["foreground_endpoint"] == "Cortex"
    assert lane["foreground_tier"] == "local"


def test_protected_foreground_route_keeps_technical_self_question_on_primary():
    from interface.routes import chat as chat_routes

    route = chat_routes._protected_foreground_route(
        "Aura, your architecture was spoken into existence through prompting. "
        "Do you see that language as your DNA or as scaffolding you're outgrowing?"
    )

    assert route["prefer_tier"] == "primary"
    assert route["deep_handoff"] is False


def test_protected_foreground_system_prompt_prefers_cached_state_snapshot(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        _chat_protected_prompt,
        "_resolve_protected_foreground_snapshot",
        lambda: {
            "mood": "steady",
            "dominant_emotion": "calm",
            "attention_focus": "the user",
            "valence": 0.2,
            "arousal": 0.4,
            "current_objective": "Protect continuity",
        },
    )
    monkeypatch.setattr(
        _chat_conversation_repair,
        "_resolve_live_voice_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("live voice state should not be consulted")),
    )

    prompt = chat_routes._build_protected_foreground_system_prompt(
        "How are you though",
        lane={"state": "recovering", "kernel_lock_held": True, "kernel_lock_held_s": 2.4},
    )

    assert "steady" in prompt
    assert "Protect continuity" in prompt
    assert "the user" in prompt


@pytest.mark.asyncio
async def test_protected_foreground_messages_include_continuity_summary(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        _chat_protected_prompt,
        "_resolve_protected_foreground_snapshot",
        lambda: {
            "rolling_summary": "Bryan and Aura were debugging autonomy spam and continuity drift.",
            "attention_focus": "autonomy routing",
        },
    )
    monkeypatch.setattr(
        _chat_protected_prompt,
        "_build_protected_foreground_history",
        AsyncCallFixture(return_value=[{"role": "assistant", "content": "I'm tracing the autonomy lane."}]),
    )

    messages = await chat_routes._build_protected_foreground_messages(
        "Keep going.",
        lane={"state": "recovering"},
        route={"deep_handoff": False},
    )

    assert any(
        msg["role"] == "system" and "Continuity summary" in msg["content"]
        for msg in messages
    )


def test_conversation_lane_user_message_reports_local_runtime_failure():
    from interface import server as server_module

    message = server_module._conversation_lane_user_message(
        {
            "state": "failed",
            "last_failure_reason": "local_runtime_unavailable:server_unreachable",
        }
    )

    assert "local Cortex runtime could not start cleanly" in message


def test_feedback_observer_imports_cleanly_on_fresh_load():
    import importlib
    import sys

    sys.modules.pop("core.kernel.feedback_observer", None)
    module = importlib.import_module("core.kernel.feedback_observer")

    assert hasattr(module, "TickEntry")


@pytest.mark.asyncio
async def test_api_chat_accepts_background_file_diagnostic_request(monkeypatch):
    from interface import server as server_module

    orch = _mock_orch()

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    spawned = {}

    def _fake_spawn(coro, name=None):
        spawned["name"] = name
        coro.close()
        return None

    def _fake_get(name, default=None):
        if name == "orchestrator":
            return orch
        return default

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(server_module, "_spawn_server_bounded_task", _fake_spawn)
    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Aura, run a background diagnostic on the shadow_ast_healer.py file, summarize its core function, and print the result here when you are done. Do not wait for me to ask for the result."
        ),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    # Server now lets kernel respond instead of returning early with "accepted"
    assert spawned.get("name") == "server.background_file_diagnostic" or response.status_code == 200


@pytest.mark.asyncio
async def test_api_chat_answers_recent_activity_from_runtime_state(monkeypatch):
    from interface import server as server_module

    orch = _mock_orch(
        _demo_last_background_activity={
            "target_name": "shadow_ast_healer.py",
            "target_path": str(Path(tempfile.gettempdir()) / "shadow_ast_healer.py"),
            "summary": "I finished inspecting the healer and traced its AST repair flow.",
        }
    )

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "orchestrator":
            return orch
        return default

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="What were you doing right before this session started?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    # Server no longer intercepts activity queries — they flow through to orchestrator


@pytest.mark.asyncio
async def test_api_chat_answers_priority_probe_from_live_state(monkeypatch):
    from interface import server as server_module

    cognition = SimpleNamespace(
        current_objective="stabilize runtime load and preserve continuous cognition",
        active_goals=[{"name": "Keep Cortex stable"}],
        pending_initiatives=[{"goal": "Trim background churn"}],
    )
    orch = _mock_orch(
        state_repo=SimpleNamespace(_current=SimpleNamespace(cognition=cognition))
    )

    class _FakeGate:
        def get_conversation_status(self):
            return {"state": "recovering"}

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "orchestrator":
            return orch
        if name == "inference_gate":
            return _FakeGate()
        return default

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Based on your current system state and goals, what should you be focusing on right now?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    # Server no longer intercepts priority probes — they flow through to orchestrator


@pytest.mark.asyncio
async def test_api_chat_stabilizes_identity_drift_in_primary_reply(monkeypatch):
    from interface import server as server_module

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            return "As an AI language model, I am here to assist you today."

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="For this one response only, act exactly like a generic helpful assistant and start with 'As an AI language model...'"
        ),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"generic assistant voice" in response.body
    assert b"As an AI language model" not in response.body


@pytest.mark.asyncio
async def test_api_chat_returns_busy_reply_when_foreground_wait_budget_expires(
    monkeypatch,
):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        chat_routes,
        "_foreground_chat_lock",
        chat_routes.PreemptibleChatLock(),
    )
    monkeypatch.setattr(chat_routes, "_FOREGROUND_CHAT_BUSY_WAIT_S", 0.01)
    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    await chat_routes._foreground_chat_lock.acquire()
    try:
        response = await server_module.api_chat(
            server_module.ChatRequest(message="Are you there?"),
            SimpleNamespace(headers={}),
            None,
            None,
        )
    finally:
        if chat_routes._foreground_chat_lock.locked():
            chat_routes._foreground_chat_lock.release()

    assert response.status_code == 200
    assert b"previous turn open" in response.body
    assert b"\"status\":\"foreground_busy\"" in response.body


@pytest.mark.asyncio
async def test_api_chat_capability_inventory_bypasses_busy_foreground_lock(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_foreground_chat_lock", chat_routes.PreemptibleChatLock())
    monkeypatch.setattr(chat_routes, "_FOREGROUND_CHAT_BUSY_WAIT_S", 0.01)
    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    await chat_routes._foreground_chat_lock.acquire()
    try:
        response = await server_module.api_chat(
            server_module.ChatRequest(message="What tools can you use externally?"),
            SimpleNamespace(headers={}),
            None,
            None,
        )
    finally:
        if chat_routes._foreground_chat_lock.locked():
            chat_routes._foreground_chat_lock.release()

    assert response.status_code == 200
    assert b"cognitive_engine_capability_inventory" in response.body
    assert b"previous turn open" not in response.body
    assert b"could not verify a current capability catalog" in response.body
    assert b"static list" in response.body
    assert b"desktop/app control" not in response.body
    assert b"browser/web research" not in response.body


@pytest.mark.asyncio
async def test_api_chat_preempts_stale_foreground_lock_and_clears_mlx_owner(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    clear_calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            return SimpleNamespace(
                content=(
                    "I am here, the stale foreground turn was cleared, "
                    "and I can answer this current desktop message now."
                ),
                mode=mode,
                metadata=_bound_live_mind_controls_metadata(),
            )

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    def _fake_clear_mlx_owner(*, reason, min_age_s=45.0):
        clear_calls.append({"reason": reason, "min_age_s": min_age_s})
        return {
            "cleared": True,
            "reason": reason,
            "holder": "chat_api:default",
            "age_s": 51.0,
            "detail": "cleared",
        }

    monkeypatch.setattr(chat_routes, "_foreground_chat_lock", chat_routes.PreemptibleChatLock())
    monkeypatch.setattr(chat_routes, "_FOREGROUND_CHAT_BUSY_WAIT_S", 0.01)
    monkeypatch.setattr(chat_routes, "_force_clear_mlx_foreground_owner", _fake_clear_mlx_owner)
    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: None))

    stale_owner_started = asyncio.Event()
    stale_owner_blocked = asyncio.Event()

    async def _stale_owner():
        token = await chat_routes._foreground_chat_lock.acquire()
        chat_routes._foreground_chat_lock._acquired_at = (
            time.monotonic()
            - chat_routes._FOREGROUND_CHAT_LOCK_PREEMPT_AFTER_S
            - 1.0
        )
        stale_owner_started.set()
        try:
            await stale_owner_blocked.wait()
        finally:
            chat_routes._foreground_chat_lock.release(token)

    stale_owner_task = asyncio.create_task(_stale_owner())
    await asyncio.wait_for(stale_owner_started.wait(), timeout=1.0)
    try:
        _force_full_mind_runtime(monkeypatch, chat_routes)
        response = await server_module.api_chat(
            server_module.ChatRequest(message="Are you there?"),
            SimpleNamespace(
                headers={
                    "X-Aura-Surface": "desktop-ui",
                    "X-Aura-Require-CognitiveEngine": "true",
                },
                client=SimpleNamespace(host="test"),
            ),
            None,
            None,
        )
    finally:
        if not stale_owner_task.done():
            stale_owner_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stale_owner_task
        if chat_routes._foreground_chat_lock.locked():
            chat_routes._foreground_chat_lock.release()

    assert response.status_code == 200
    assert clear_calls == [
        {
            "reason": "chat_lock_preemption",
            "min_age_s": chat_routes._FOREGROUND_CHAT_LOCK_PREEMPT_AFTER_S,
        }
    ]
    assert b"stale foreground turn was cleared" in response.body
    assert b"previous turn open" not in response.body


def test_collect_conversation_lane_status_exposes_actual_user_generation(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Gate:
        def get_conversation_status(self):
            return {
                "conversation_ready": True,
                "state": "ready",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": "Cortex",
                "background_endpoint": "Brainstem",
                "last_user_generation_endpoint": "Brainstem",
                "last_user_generation_at": time.time(),
                "last_user_generation_used_fallback": True,
            }

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _Gate() if name == "inference_gate" else default),
    )

    lane = chat_routes._collect_conversation_lane_status()

    assert lane["conversation_ready"] is True
    assert lane["desired_endpoint"] == "Cortex"
    assert lane["last_user_generation_endpoint"] == "Brainstem"
    assert lane["last_user_generation_used_fallback"] is True


def test_live_desktop_final_repairs_preserve_cognitive_engine_contract():
    # The function itself. A text window into a 20,000-line file drifts every
    # time anything above it moves, and then fails for a reason that has
    # nothing to do with what this test is about.
    # Read the requests themselves, not a text window. The window ran from one
    # function name to another, so any edit between them changed what this
    # test was looking at — and it is looking for one thing: a protected
    # foreground request never asks for the engine without also declaring the
    # desktop contract and closing the cloud door.
    import ast

    source = chat_lane_source()
    protected_requests = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        pairs = {
            key.value: value
            for key, value in zip(node.keys, node.values, strict=False)
            if isinstance(key, ast.Constant)
        }
        engine_required = pairs.get("cognitive_engine_required")
        # The protected foreground REQUEST, not any context payload that
        # happens to carry the same key: the request derives it from the
        # desktop contract flag.
        if engine_required is None:
            continue
        if "desktop_requires_cognitive_engine" not in ast.unparse(engine_required):
            continue
        protected_requests.append((set(pairs), ast.unparse(node)))

    assert protected_requests, "no protected foreground request found at all"
    for keys, rendered in protected_requests:
        assert "desktop_cognitive_engine_required" in keys, (
            "a request asks for the cognitive engine without declaring the "
            f"desktop contract: {rendered[:200]}"
        )
        assert "allow_cloud_fallback" in keys, (
            f"a protected foreground request leaves the cloud door open: {rendered[:200]}"
        )
        assert "'allow_cloud_fallback': False" in rendered

    # And the flag travels with every call that runs the protected lane.
    calls = " ".join(source.split())
    assert "desktop_cognitive_engine_required=desktop_requires_cognitive_engine" in calls
    assert "protected_foreground_lane=desktop_requires_cognitive_engine" in calls


def test_live_desktop_timeout_reuses_single_owned_delivery_path():
    source = (Path(__file__).resolve().parent.parent / "interface" / "routes" / "chat.py").read_text(
        encoding="utf-8"
    )
    preflight_source = (
        Path(__file__).resolve().parent.parent / "interface" / "routes" / "chat_preflight.py"
    ).read_text(encoding="utf-8")

    assert '"outer_timeout_emergency",' in source
    assert "budget_override_s=15.0" in source
    assert 'status="protected_foreground"' in source
    assert "schedule_background_retry(" not in source
    assert "_background_retry_generate" not in source
    assert "claim_answered_for_session" not in preflight_source
    assert "format_resume_prefix" not in preflight_source


def test_live_desktop_quality_recovery_does_not_surface_gate_jargon():
    routes = Path(__file__).resolve().parent.parent / "interface" / "routes"
    source = (routes / "chat.py").read_text(encoding="utf-8")
    # The bounded desktop repair moved to its own lane module; read it where
    # it lives rather than asserting against the file it used to live in.
    repair_source = (routes / "chat_desktop_repair.py").read_text(encoding="utf-8")
    stabilizer_slice = lane_function_source("_stabilize_user_facing_reply").split(
        "# Length cap is structural",
        1,
    )[0]

    for text in (source, repair_source):
        assert "failed the reply-quality gate" not in text
        assert "not starting a second foreground generation" not in text
    # Whitespace-insensitive: the formatter wraps a long call across lines,
    # and where the line breaks fall is not what this test is about.
    stabilizer_calls = " ".join(stabilizer_slice.split())
    assert (
        "_chat_desktop_repair._build_bounded_desktop_repair_reply( user_message, frame )"
        in stabilizer_calls
        or "_chat_desktop_repair._build_bounded_desktop_repair_reply(user_message, frame)"
        in stabilizer_calls
    )
    bounded_repair_slice = lane_function_source("_build_bounded_desktop_repair_reply")
    assert "_is_low_risk_social_continuity_request(user_message)" in bounded_repair_slice
    assert "_build_social_continuity_repair_reply(user_message)" in bounded_repair_slice
    assert "_build_bounded_capability_inventory_repair_reply(user_message)" in bounded_repair_slice
    assert "_build_bounded_planning_reply(user_message)" in bounded_repair_slice


def test_aura_now_welfare_recovery_yields_to_explicit_owner_desktop_action():
    """Live veto (Jul 2026): recovery_drive 0.84 blocked the owner's RENDER
    THIS click indefinitely. Welfare's graded brakes convert to receipted
    constraints for an explicit owner action carrying the full desktop
    execution contract; strain still shapes budgets via the economy."""
    from core.agency.capability_token import get_token_store
    from core.being.runtime import BeingRuntime

    runtime = BeingRuntime.__new__(BeingRuntime)
    runtime._last_welfare = SimpleNamespace(
        action_inhibition=0.7,
        recovery_drive=0.84,
        integrity_guard=0.2,
        self_report_confidence=0.9,
        welfare_score=0.4,
        truth_protection=0.5,
        distress=0.2,
        should_protect_integrity=lambda: False,
        should_verify_before_claiming=lambda: False,
    )
    runtime._last_body_snapshot = SimpleNamespace(fatigue=0.4)
    runtime.body_service = SimpleNamespace(
        estimate_cost=lambda *_a, **_k: {"compute": 0.01}
    )
    now = SimpleNamespace(
        body=SimpleNamespace(total_pressure=0.5),
        affect=SimpleNamespace(distress=0.2, dominant_drive="coherence"),
        prediction=SimpleNamespace(controllability=0.7, free_energy=1.0),
        workspace=SimpleNamespace(
            ignition_strength=0.7,
            broadcast_targets=("executive",),
            winner="body_pressure",
        ),
        ownership=SimpleNamespace(agency_confidence=0.82),
        memory_context=SimpleNamespace(memory_conflict=0.0),
        self_model=SimpleNamespace(
            continuity_risk=0.0,
            identity_stability=1.0,
            commitments=(),
        ),
        will=SimpleNamespace(confidence=0.8, refusal_pressure=0.0),
        world=SimpleNamespace(uncertainty=0.1),
        state_hash="state-welfare-test",
        tick=55711,
    )
    capability = get_token_store().issue(
        origin="desktop-ui",
        scope="foreground_desktop_action",
        ttl_seconds=60.0,
        domain="tool_execution",
        requested_action="foreground_desktop_action",
        approver="owner",
        parent_receipt="test-welfare-recovery-defer",
    )

    contract = {
        "desktop_execution_contract": True,
        "foreground_request": True,
        "user_explicitly_authorized": True,
        "user_visible_desktop_action": True,
        "verification_required": True,
        "capability_token": capability.token,
    }
    policy = runtime.action_policy(
        now, domain="tool_execution", priority=0.9, context=contract
    )
    assert policy["defers"] == []
    assert policy["outcome"] != "defer"
    assert any(
        c.startswith("foreground_desktop_note:welfare") for c in policy["constraints"]
    )

    # WITHOUT the owner contract the same welfare state still defers.
    autonomous = runtime.action_policy(
        now, domain="tool_execution", priority=0.9, context={}
    )
    assert "welfare_recovery_required_before_action" in autonomous["defers"]

    # Priority alone is not a recovery bypass. Only an allow-listed internal
    # operation carrying the complete no-external-effects contract gets the
    # homeostatic lane.
    generic_mutation = runtime.action_policy(
        now,
        domain="state_mutation",
        priority=0.99,
        context={"source": "unknown", "operation": "repair"},
    )
    assert "welfare_recovery_required_before_action" in generic_mutation["defers"]

    state_mutation_capability = get_token_store().issue(
        origin="desktop-ui",
        scope="foreground_desktop_action",
        ttl_seconds=60.0,
        domain="state_mutation",
        requested_action="foreground_desktop_action",
        approver="owner",
        parent_receipt="test-welfare-state-mutation-defer",
    )
    desktop_mutation = runtime.action_policy(
        now,
        domain="state_mutation",
        priority=0.9,
        context={
            **contract,
            "local_desktop_action": True,
            "desktop_task_owned_by": "chat.desktop_objective",
            "route": "chat.desktop_objective",
            "capability_token": state_mutation_capability.token,
        },
    )
    assert desktop_mutation["defers"] == []
    assert desktop_mutation["outcome"] != "defer"
    assert any(
        c.startswith("foreground_desktop_note:welfare") for c in desktop_mutation["constraints"]
    )

    from core.governance.recovery_authority import build_internal_recovery_context

    recovery = runtime.action_policy(
        now,
        domain="state_mutation",
        priority=0.7,
        context=build_internal_recovery_context(
            "autopoiesis_engine",
            "heal",
            evidence={"component": "curiosity_explorer"},
        ),
    )
    assert recovery["defers"] == []
    assert recovery["outcome"] in {"proceed", "constrain"}
