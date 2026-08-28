"""The receipt a desktop turn produces about its own path.

A turn that claims it ran the full mind has to be able to show which
subsystems were live, which controls were bound, and where the text came
from. These build that payload from the trace the turn actually recorded,
and refuse to assert a path nothing observed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from core.container import ServiceContainer
from interface.routes.chat_common import (  # noqa: E402
    _CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,  # noqa: F401
    _CHAT_RECOVERABLE_ERRORS,  # noqa: F401
    _CHAT_REQUEST_PRINCIPAL,  # noqa: F401
    _CHAT_REQUEST_SURFACE,  # noqa: F401
    _MAX_CONVERSATION_LOG_EXCHANGES,  # noqa: F401
    _conversation_log,  # noqa: F401
    _locks,  # noqa: F401
    logger,  # noqa: F401
)
from interface.routes import chat_desktop_repair as _chat_desktop_repair
from interface.routes import chat_preflight as _chat_preflight
from core.brain.live_mind_contract import (
    append_text_mutation,
    merge_text_mutations,
    normalize_live_mind_surface_control_receipt,
    summarize_text_mutation_authorship,
    verify_text_mutation_chain,
)
from core.runtime.errors import describe_error, record_degradation

from interface.routes.chat_common import (
    _MAX_USER_SURFACE_CONTINUATIONS,
    _ORGAN_ABSENCE_STREAKS,
)


def _runtime_cognitive_engine_available() -> bool:
    try:
        engine = ServiceContainer.get("cognitive_engine", default=None)
        return bool(engine is not None and callable(getattr(engine, "think", None)))
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Runtime CognitiveEngine status probe failed: %s", exc)
        return False


def _runtime_kernel_available() -> bool:
    try:
        kernel = ServiceContainer.get("aura_kernel", default=None)
        if kernel is not None:
            return True
        from core.kernel.kernel_interface import KernelInterface

        ki = KernelInterface.get_instance()
        return bool(ki and getattr(ki, "kernel", None) is not None)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Runtime kernel status probe failed: %s", exc)
        return False


def _runtime_memory_available() -> bool:
    try:
        for service_name in (
            "memory_system",
            "memory_service",
            "memory_write_gateway",
            "state_vault",
            "live_aura_state",
        ):
            if ServiceContainer.get(service_name, default=None) is not None:
                return True
        live_state = _chat_preflight._resolve_live_aura_state()
        return bool(
            live_state is not None
            and (
                getattr(live_state, "memory", None) is not None
                or getattr(getattr(live_state, "cognition", None), "working_memory", None)
                is not None
            )
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Runtime memory status probe failed: %s", exc)
        return False


def _runtime_inference_available(
    lane: dict[str, Any] | None = None,
    *,
    require_conversation_ready: bool = False,
) -> bool:
    try:
        gate = ServiceContainer.get("inference_gate", default=None)
        if gate is None:
            return False
        if hasattr(gate, "get_conversation_status"):
            lane = dict(lane or gate.get_conversation_status() or {})
            if lane.get("conversation_ready"):
                return True
            if require_conversation_ready:
                return False
            return str(lane.get("state") or "").strip().lower() in {
                "ready",
                "warming",
                "recovering",
            }
        if require_conversation_ready:
            return False
        return callable(getattr(gate, "generate", None))
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Runtime inference status probe failed: %s", exc)
        return False


def _runtime_substrate_voice_available() -> bool:
    try:
        from core.voice.substrate_voice_engine import get_substrate_voice_engine

        sve = get_substrate_voice_engine()
        return bool(sve is not None)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Runtime substrate voice status probe failed: %s", exc)
        return False


_EXPECTED_TURN_ORGANS: tuple[tuple[str, str], ...] = (
    ("personality_engine", "the voice that makes the reply hers rather than the base model's"),
    ("affect_engine", "the affect this turn is coloured by"),
    ("episodic_memory", "the turn is remembered, so tomorrow's answer knows about it"),
    ("semantic_memory", "what she knows, as opposed to what she just heard"),
    ("soul", "identity continuity across turns and restarts"),
    ("soma", "felt state, which disposition and prosody both read"),
    ("data_honesty_governor", "the honesty floor a claim has to clear"),
    ("knowledge_graph", "grounding a claim against what she already believes"),
    ("event_bus", "the turn is observable to the rest of the organism"),
)


def _collect_expected_turn_organs() -> dict[str, bool]:
    """Which conversational organs were actually present for this turn."""
    engaged: dict[str, bool] = {}
    for name, _why in _EXPECTED_TURN_ORGANS:
        try:
            engaged[name] = ServiceContainer.peek(name, default=None) is not None
        except _CHAT_RECOVERABLE_ERRORS:
            engaged[name] = False
    return engaged


def _absent_turn_organs(engaged: dict[str, bool] | None = None) -> list[str]:
    """The ones missing, with why they matter — for a log line or a receipt."""
    engaged = engaged if engaged is not None else _collect_expected_turn_organs()
    reasons = dict(_EXPECTED_TURN_ORGANS)
    return [
        f"{name} ({reasons.get(name, 'no stated purpose')})"
        for name, present in sorted(engaged.items())
        if not present
    ]


_CHRONIC_ABSENCE_TURNS = 12


def _note_organ_engagement(engaged: dict[str, bool]) -> list[str]:
    """Track absence across turns. Returns organs that just became chronic."""
    became_chronic: list[str] = []
    reasons = dict(_EXPECTED_TURN_ORGANS)
    for name, present in engaged.items():
        if present:
            _ORGAN_ABSENCE_STREAKS.pop(name, None)
            continue
        streak = _ORGAN_ABSENCE_STREAKS.get(name, 0) + 1
        _ORGAN_ABSENCE_STREAKS[name] = streak
        # Exactly at the threshold, so a permanent gap escalates once.
        if streak == _CHRONIC_ABSENCE_TURNS:
            became_chronic.append(name)
            record_degradation(
                "chat.turn_engagement",
                RuntimeError(
                    f"{name} absent for {streak} consecutive turns: "
                    f"{reasons.get(name, 'no stated purpose')}"
                ),
                severity="warning",
                action=(
                    "kept answering without it; a conversational organ missing "
                    "this persistently is a defect, not a warm-up"
                ),
            )
    return became_chronic


def _collect_live_chat_required_subsystems(
    lane: dict[str, Any] | None = None,
    *,
    generation_proven: bool = False,
) -> dict[str, bool]:
    lane = dict(lane or {})
    inference_ready = _runtime_inference_available(lane, require_conversation_ready=True)
    if generation_proven:
        inference_ready = True
    return {
        "kernel": _runtime_kernel_available(),
        "cognitive_engine": _runtime_cognitive_engine_available(),
        "inference": inference_ready,
        "memory": _runtime_memory_available(),
        "tool_governance": _chat_desktop_repair._runtime_tool_governance_available(),
        "substrate_voice": _runtime_substrate_voice_available(),
    }


_LIVE_CHAT_REQUIRED_SUBSYSTEMS = frozenset(
    {
        "kernel",
        "cognitive_engine",
        "inference",
        "memory",
        "tool_governance",
        "substrate_voice",
    }
)


def _attested_live_chat_required_subsystems(
    trace: dict[str, Any],
    *,
    generation_proven: bool = False,
) -> dict[str, bool] | None:
    """Reuse the runtime-stamped pre-generation snapshot without resolving services.

    Contract construction is observational. Calling ``ServiceContainer.get``
    here can initialize late services after an answer already exists, introducing
    seconds of latency and governance side effects. The foreground route already
    collects the complete subsystem vector inside a per-process stamped payload;
    accept only that exact, boolean vector. Older/non-stamped callers fall back to
    fresh compatibility probes instead of being trusted on a self-asserted flag.
    """

    if trace.get("live_mind_required_subsystems_attested") is not True:
        return None
    raw = trace.get("live_mind_required_subsystems")
    if not isinstance(raw, dict) or set(raw) != _LIVE_CHAT_REQUIRED_SUBSYSTEMS:
        return None
    if any(type(raw[name]) is not bool for name in _LIVE_CHAT_REQUIRED_SUBSYSTEMS):
        return None
    observed = {name: raw[name] for name in sorted(_LIVE_CHAT_REQUIRED_SUBSYSTEMS)}
    if generation_proven:
        # A successfully delivered cognitive answer is direct evidence that the
        # inference lane served this turn, even if the earlier lane snapshot was
        # still transitioning from warming to ready.
        observed["inference"] = True
    return observed


_RUNTIME_GROUNDING_RESPONSE_PATHS = frozenset(
    {
        "cognitive_engine_memory_state_grounding",
        "cognitive_engine_identity_continuity_grounding",
        "cognitive_engine_runtime_fact_grounding",
        "cognitive_engine_capability_tail_grounding",
        "cognitive_engine_capability_catalog_grounding",
        "cognitive_engine_self_process_grounding",
        "cognitive_engine_self_condition_grounding",
        "cognitive_engine_bounded_planning",
        "cognitive_engine_context_evidence_repair",
        "cognitive_engine_own_source_grounding",
        "verified_action_episode",
    }
)


def _build_live_turn_contract_payload(
    *,
    desktop_required: bool,
    request_surface: str,
    lane_status: dict[str, Any] | None,
    response_confidence: str,
    status: str,
    reply_source: str = "",
    turn_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Machine-readable evidence for whether a live desktop turn used the full mind path."""
    lane = dict(lane_status or {})
    trace = dict(turn_trace or {})
    response_path = str(trace.get("response_path") or reply_source or status or "").strip()
    qualified_recurrent_response_path = (
        response_path == "cognitive_engine_qualified_recurrent"
    )
    engine_think_invoked = bool(trace.get("engine_think_invoked"))
    engine_reply_failed = bool(trace.get("cognitive_engine_reply_failed"))
    engine_reply_accepted = (
        bool(trace.get("cognitive_engine_reply_accepted")) and not engine_reply_failed
    )
    bounded_contract_used = bool(trace.get("bounded_contract_used"))
    legacy_fallback_used = bool(trace.get("legacy_fallback_used"))
    latent_cortex_selected = bool(trace.get("latent_cortex_selected"))
    latent_cortex_attempted = bool(trace.get("latent_cortex_attempted"))
    latent_cortex_succeeded = bool(trace.get("latent_cortex_succeeded"))
    latent_cortex_fallback_used = bool(trace.get("latent_cortex_fallback_used"))
    latent_cortex_failure_reason = str(trace.get("latent_cortex_failure_reason") or "")[:500]
    raw_latent_receipt = trace.get("latent_cortex_receipt")
    raw_latent_receipt = dict(raw_latent_receipt) if isinstance(raw_latent_receipt, dict) else {}
    raw_runtime_identity = raw_latent_receipt.get("runtime_identity")
    raw_runtime_identity = (
        dict(raw_runtime_identity) if isinstance(raw_runtime_identity, dict) else {}
    )

    def _sha256(value: Any) -> bool:
        return bool(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    def _git_oid(value: Any) -> bool:
        return bool(
            isinstance(value, str)
            and len(value) in {40, 64}
            and all(character in "0123456789abcdef" for character in value)
        )

    installed_app_required = raw_runtime_identity.get("installed_app_required") is True
    runtime_identity_valid = bool(
        raw_runtime_identity.get("identity_bound") is True
        and raw_runtime_identity.get("source_verified") is True
        and _git_oid(raw_runtime_identity.get("source_commit"))
        and _sha256(raw_runtime_identity.get("workspace_state_sha256"))
        and _sha256(raw_runtime_identity.get("shell_assets_sha256"))
        and (
            not installed_app_required
            or (
                raw_runtime_identity.get("installed_app_verified") is True
                and bool(str(raw_runtime_identity.get("bundle_identifier") or "").strip())
                and _sha256(raw_runtime_identity.get("app_executable_sha256"))
                and _sha256(raw_runtime_identity.get("launch_manifest_sha256"))
            )
        )
    )
    latent_cortex_identity_bound = bool(
        trace.get("latent_cortex_identity_bound")
        and runtime_identity_valid
        and _sha256(raw_latent_receipt.get("checkpoint_fingerprint"))
        and _sha256(raw_latent_receipt.get("worker_source_sha256"))
        and _sha256(raw_latent_receipt.get("request_payload_sha256"))
        and _sha256(raw_latent_receipt.get("input_tokens_sha256"))
        and isinstance(raw_latent_receipt.get("worker_boot_id"), str)
        and len(raw_latent_receipt.get("worker_boot_id")) == 32
        and all(
            character in "0123456789abcdef"
            for character in raw_latent_receipt.get("worker_boot_id")
        )
        and type(raw_latent_receipt.get("worker_pid")) is int
        and raw_latent_receipt.get("worker_pid", 0) > 0
        and bool(str(raw_latent_receipt.get("worker_model_path") or "").strip())
        and type(raw_latent_receipt.get("worker_model_parameter_count")) is int
        and raw_latent_receipt.get("worker_model_parameter_count", 0) >= 20_000_000_000
        and raw_latent_receipt.get("worker_model_parameter_count_basis")
        == "architecture_config_logical"
        and type(raw_latent_receipt.get("worker_model_stored_parameter_element_count")) is int
        and raw_latent_receipt.get("worker_model_stored_parameter_element_count", 0) > 0
        and raw_latent_receipt.get("worker_affective_steering_active") is True
        and raw_latent_receipt.get("episode_affective_steering_applied") is True
        and isinstance(
            raw_latent_receipt.get("episode_affective_steering_alpha"),
            (int, float),
        )
        and not isinstance(raw_latent_receipt.get("episode_affective_steering_alpha"), bool)
        and 0.0 <= float(raw_latent_receipt.get("episode_affective_steering_alpha")) <= 1.0
    )
    raw_latent_output_quality = raw_latent_receipt.get("output_quality")
    raw_latent_output_quality = (
        dict(raw_latent_output_quality) if isinstance(raw_latent_output_quality, dict) else {}
    )
    latent_cortex_raw_output_quality_proven = bool(
        raw_latent_output_quality.get("schema") == "aura.latent_output_quality.v1"
        and raw_latent_output_quality.get("policy") == "resident_latent_product_quality_v1"
        and raw_latent_output_quality.get("passed") is True
        and _sha256(raw_latent_output_quality.get("text_sha256"))
        and _sha256(raw_latent_output_quality.get("objective_sha256"))
        and raw_latent_output_quality.get("reasons") == []
    )
    raw_final_output_quality = trace.get("latent_cortex_final_output_quality")
    latent_cortex_final_output_quality = (
        dict(raw_final_output_quality) if isinstance(raw_final_output_quality, dict) else {}
    )
    latent_cortex_final_output_quality_proven = bool(
        latent_cortex_final_output_quality.get("schema") == "aura.latent_output_quality.v1"
        and latent_cortex_final_output_quality.get("policy") == "resident_latent_product_quality_v1"
        and latent_cortex_final_output_quality.get("passed") is True
        and _sha256(latent_cortex_final_output_quality.get("text_sha256"))
        and _sha256(latent_cortex_final_output_quality.get("objective_sha256"))
        and latent_cortex_final_output_quality.get("objective_sha256")
        == raw_latent_output_quality.get("objective_sha256")
        and latent_cortex_final_output_quality.get("reasons") == []
    )
    raw_public_output_quality = trace.get("latent_cortex_public_output_quality")
    latent_cortex_public_output_quality = (
        dict(raw_public_output_quality) if isinstance(raw_public_output_quality, dict) else {}
    )
    qualified_recurrent_public_output_quality_proven = bool(
        qualified_recurrent_response_path
        and latent_cortex_public_output_quality.get("schema")
        == "aura.latent_output_quality.v1"
        and latent_cortex_public_output_quality.get("policy")
        == "qualified_recurrent_state_serialization_quality_v1"
        and latent_cortex_public_output_quality.get("passed") is True
        and latent_cortex_public_output_quality.get("state_serialization") is True
        and latent_cortex_public_output_quality.get("serialization")
        == "canonical_json_from_authenticated_semantic_state"
        and _sha256(latent_cortex_public_output_quality.get("text_sha256"))
        and _sha256(latent_cortex_public_output_quality.get("objective_sha256"))
        and _sha256(latent_cortex_public_output_quality.get("receipt_sha256"))
        and latent_cortex_public_output_quality.get("reasons") == []
    )
    latent_cortex_public_output_quality_proven = bool(
        qualified_recurrent_public_output_quality_proven
        or (
            latent_cortex_public_output_quality.get("schema")
            == "aura.latent_output_quality.v1"
            and latent_cortex_public_output_quality.get("policy")
            == "resident_latent_product_quality_v1"
            and latent_cortex_public_output_quality.get("passed") is True
            and _sha256(latent_cortex_public_output_quality.get("text_sha256"))
            and _sha256(latent_cortex_public_output_quality.get("objective_sha256"))
            and latent_cortex_public_output_quality.get("objective_sha256")
            == raw_latent_output_quality.get("objective_sha256")
            and latent_cortex_public_output_quality.get("reasons") == []
        )
    )
    raw_surface_receipt_for_quality = trace.get("live_mind_surface_control_receipt")
    raw_surface_receipt_for_quality = (
        dict(raw_surface_receipt_for_quality)
        if isinstance(raw_surface_receipt_for_quality, dict)
        else {}
    )
    raw_final_quality_hash_match = bool(
        trace.get("latent_cortex_raw_final_quality_hash_match")
        and raw_latent_output_quality.get("text_sha256")
        == latent_cortex_final_output_quality.get("text_sha256")
    )
    raw_public_quality_hash_match = bool(
        raw_latent_output_quality.get("text_sha256")
        and raw_latent_output_quality.get("text_sha256")
        == latent_cortex_public_output_quality.get("text_sha256")
    )
    final_public_quality_hash_match = bool(
        trace.get("latent_cortex_final_public_quality_hash_match")
        and latent_cortex_final_output_quality.get("text_sha256")
        == latent_cortex_public_output_quality.get("text_sha256")
    )
    quality_mutation_ledger = merge_text_mutations(
        raw_surface_receipt_for_quality.get("text_mutations"),
        trace.get("text_mutations"),
    )
    latent_cortex_raw_final_mutation_chain = verify_text_mutation_chain(
        raw_surface_receipt_for_quality.get("text_mutations"),
        before_sha256=raw_latent_output_quality.get("text_sha256"),
        after_sha256=latent_cortex_final_output_quality.get("text_sha256"),
    )
    latent_cortex_final_public_mutation_chain = verify_text_mutation_chain(
        quality_mutation_ledger,
        before_sha256=latent_cortex_final_output_quality.get("text_sha256"),
        after_sha256=latent_cortex_public_output_quality.get("text_sha256"),
    )
    latent_cortex_output_mutation_chain = verify_text_mutation_chain(
        quality_mutation_ledger,
        before_sha256=raw_latent_output_quality.get("text_sha256"),
        after_sha256=latent_cortex_public_output_quality.get("text_sha256"),
    )
    raw_final_quality_transition_proven = bool(
        raw_final_quality_hash_match
        or (
            latent_cortex_raw_final_mutation_chain.get("passed") is True
            and latent_cortex_raw_final_mutation_chain.get("chain_length", 0) > 0
        )
    )
    final_public_quality_transition_proven = bool(
        final_public_quality_hash_match
        or (
            latent_cortex_final_public_mutation_chain.get("passed") is True
            and latent_cortex_final_public_mutation_chain.get("chain_length", 0) > 0
        )
    )
    latent_cortex_output_quality_proven = bool(
        qualified_recurrent_public_output_quality_proven
        or (
            latent_cortex_raw_output_quality_proven
            and latent_cortex_final_output_quality_proven
            and latent_cortex_public_output_quality_proven
            and raw_final_quality_transition_proven
            and final_public_quality_transition_proven
            and (
                raw_public_quality_hash_match
                or latent_cortex_output_mutation_chain.get("passed") is True
            )
        )
    )
    latent_cortex_receipt = {
        key: raw_latent_receipt.get(key)
        for key in (
            "episode_id",
            "checkpoint_fingerprint",
            "checkpoint_fingerprint_method",
            "checkpoint_file_count",
            "worker_boot_id",
            "worker_pid",
            "worker_model_path",
            "worker_model_parameter_count",
            "worker_model_stored_parameter_element_count",
            "worker_model_parameter_count_basis",
            "worker_source_sha256",
            "worker_affective_steering_active",
            "worker_affective_steering_alpha",
            "episode_affective_steering_applied",
            "episode_affective_steering_alpha",
            "request_payload_sha256",
            "input_tokens_sha256",
            "input_token_count",
            "input_context_compaction",
            "params_unchanged",
            "schedule_hash",
            "n_slots",
            "n_branches",
            "steps_taken",
            "halting_reason",
            "decode_requested_tokens",
            "decode_generated_tokens",
            "decode_termination",
            "decode_temperature",
            "decode_top_p",
            "decode_newline_suppressions",
            "decode_repetition_penalty_applied",
            "decode_bridge_applied",
            "decode_bridge_policy",
            "decode_bridge_token_count",
            "decode_bridge_tokens_sha256",
            "decode_bridge_logits_digest",
            "output_quality",
            "verifier_guidance",
            "generative_verifier",
            "verifier_probe_max_tokens",
            "verifier_probe_contract",
            "contract_repair",
            "latent_opt_applied",
            "latent_opt_mode",
            "latent_opt_loss_trail",
            "latent_opt_attempts",
            "latent_opt_steps",
            "latent_opt_rejected",
            "latent_opt_budget_exhausted",
            "latent_opt_verifier",
            "fast_weights_applied",
            "fast_weights_erased",
            "fast_weights_layers",
            "fast_weight_optimization_attempts",
            "fast_weight_optimized_steps",
            "fast_weight_rejected_steps",
            "fast_weight_budget_exhausted",
            "fast_weight_optimizer",
            "fast_weight_loss_trail",
            "fast_weight_gradient_norm_trail",
            "fast_weight_accepted_step_sizes",
            "fast_weight_line_search_backtracks",
            "fast_weight_canaries",
            "fast_weight_verifier",
            "last_stage",
            "stage_timings_s",
            "budget",
            "honest_flags",
        )
        if key in raw_latent_receipt
    }
    latent_cortex_receipt["runtime_identity"] = {
        key: raw_runtime_identity.get(key)
        for key in (
            "schema",
            "identity_bound",
            "launch_mode",
            "installed_app_required",
            "installed_app_verified",
            "source_verified",
            "source_commit",
            "source_branch",
            "workspace_state_sha256",
            "source_dirty",
            "source_change_count",
            "shell_assets_sha256",
            "bundle_identifier",
            "app_executable_sha256",
            "launch_manifest_sha256",
            "issues",
        )
        if key in raw_runtime_identity
    }
    preflight_evidence_profile = str(trace.get("preflight_evidence_profile") or "")
    raw_preflight_evidence_owner = trace.get("preflight_evidence_owner_receipt")
    preflight_evidence_owner = (
        {
            key: raw_preflight_evidence_owner.get(key)
            for key in (
                "schema",
                "family",
                "task_depth",
                "parser_id",
                "public_source_sha256",
                "syntax_sha256",
                "receipt_sha256",
            )
            if key in raw_preflight_evidence_owner
        }
        if isinstance(raw_preflight_evidence_owner, dict)
        else {}
    )
    state_native_evidence_owner = bool(
        preflight_evidence_profile == "qualified_recurrent_state_serialization"
        and preflight_evidence_owner
    )
    live_mind_context_required = bool(desktop_required and not state_native_evidence_owner)
    live_mind_context_present = bool(trace.get("live_mind_context_present"))
    live_mind_snapshot_present = bool(trace.get("live_mind_snapshot_present"))
    live_mind_snapshot_ready = bool(trace.get("live_mind_snapshot_ready"))
    live_mind_snapshot_missing_services = list(
        trace.get("live_mind_snapshot_missing_services") or []
    )
    preflight_live_mind_required_subsystems_ok = bool(trace.get("live_mind_required_subsystems_ok"))
    architecture_context_bound = bool((not live_mind_context_required) or live_mind_context_present)
    live_mind_snapshot_bound = bool(
        (not live_mind_context_required)
        or (live_mind_snapshot_present and live_mind_snapshot_ready)
    )
    raw_live_mind_generation_controls = trace.get("live_mind_generation_controls")
    live_mind_generation_controls = (
        {
            key: raw_live_mind_generation_controls.get(key)
            for key in (
                "temperature",
                "top_p",
                "clean_user_surface_recurrent_loops",
                "clean_user_surface_steering_alpha",
            )
            if key in raw_live_mind_generation_controls
        }
        if isinstance(raw_live_mind_generation_controls, dict)
        else {}
    )
    live_mind_generation_controls_present = bool(live_mind_generation_controls)
    live_mind_controls_bound = bool(
        trace.get("live_mind_controls_bound") and live_mind_generation_controls_present
    )
    raw_surface_control_receipt = trace.get("live_mind_surface_control_receipt")
    live_mind_surface_control_receipt = (
        {
            key: raw_surface_control_receipt.get(key)
            for key in (
                "enabled",
                "live_mind_controls_bound",
                "clean_user_surface_contract",
                "surface_validation_prompt_present",
                "surface_alpha_applied",
                "surface_alpha_applied_ok",
                "recurrent_runtime_loops_applied",
                "recurrent_runtime_loops_applied_ok",
                "surface_quality_gate_enabled",
                "surface_quality_gate_passed",
                "surface_quality_gate_attempts",
                "surface_quality_gate_reasons",
                "surface_quality_gate_error",
                "latent_final_output_quality",
                "generation_max_tokens",
                "caller_requested_max_tokens",
                "adaptive_suggested_max_tokens",
                "output_contract_generation_floor",
                "generated_tokens",
                "semantic_output_token_cap",
                "hard_output_token_ceiling",
                "instruction_shape_repair_applied",
                "deterministic_repair_applied",
                "authorship_replacement_applied",
                "authorship_augmentation_applied",
                "model_replacement_applied",
                "text_mutations",
                "text_mutation_count",
                "exact_reply_token_count",
                "exact_reply_required_termination_headroom",
                "exact_reply_available_termination_headroom",
                "exact_reply_content_capacity_sufficient",
                "exact_reply_termination_headroom_sufficient",
                "exact_reply_token_ceiling_valid",
                "exact_reply_native_capacity_sufficient",
                "requested_output_contract",
                "applied",
            )
            if key in raw_surface_control_receipt
        }
        if isinstance(raw_surface_control_receipt, dict)
        else {}
    )
    text_mutations = merge_text_mutations(
        live_mind_surface_control_receipt.get("text_mutations"),
        trace.get("text_mutations"),
    )
    live_mind_surface_control_receipt["text_mutations"] = text_mutations
    live_mind_surface_control_receipt["text_mutation_count"] = len(text_mutations)
    live_mind_controls_worker_applied = bool(
        live_mind_surface_control_receipt.get("live_mind_controls_bound")
        and live_mind_surface_control_receipt.get("applied")
    )
    live_mind_generation_required = bool(
        trace.get(
            "live_mind_generation_required",
            live_mind_surface_control_receipt.get("generation_required", True),
        )
    )
    live_mind_controls_application_satisfied = bool(
        (not live_mind_generation_required) or live_mind_controls_worker_applied
    )
    live_mind_surface_quality_gate_enabled = bool(
        live_mind_surface_control_receipt.get("surface_quality_gate_enabled")
    )
    live_mind_surface_quality_gate_passed = bool(
        (not live_mind_surface_quality_gate_enabled)
        or live_mind_surface_control_receipt.get("surface_quality_gate_passed")
    )
    worker_instruction_shape_repair_applied = bool(
        live_mind_surface_control_receipt.get("instruction_shape_repair_applied")
        or any(
            str(item.get("stage") or "").startswith("mlx_worker.instruction_shape")
            for item in text_mutations
        )
    )
    legacy_post_generation_repair_applied = bool(trace.get("post_generation_repair_applied", False))
    post_generation_repair_applied = bool(text_mutations or legacy_post_generation_repair_applied)
    deterministic_repair_applied = bool(
        any(bool(item.get("deterministic")) for item in text_mutations)
        or worker_instruction_shape_repair_applied
        or trace.get("deterministic_repair_applied", False)
        or (legacy_post_generation_repair_applied and not text_mutations)
    )
    mutation_authorship_effects = [
        str(item.get("authorship_effect") or "replaced_by_runtime") for item in text_mutations
    ]
    response_authority_kind = str(trace.get("response_authority_kind") or "").strip()
    response_authority_proven = bool(
        response_authority_kind and trace.get("response_authority_proven") is True
    )
    unreceipted_runtime_replacement = bool(
        response_path in _RUNTIME_GROUNDING_RESPONSE_PATHS
        and not text_mutations
        and not response_authority_proven
    )
    authorship_replacement_applied = bool(
        "replaced_by_runtime" in mutation_authorship_effects or unreceipted_runtime_replacement
    )
    authorship_augmentation_applied = bool("augmented_by_runtime" in mutation_authorship_effects)
    model_replacement_applied = bool("replaced_by_model" in mutation_authorship_effects)
    requested_contract = live_mind_surface_control_receipt.get(
        "requested_output_contract"
    ) or trace.get("requested_output_contract")
    requested_contract = dict(requested_contract) if isinstance(requested_contract, dict) else {}
    requested_contract_kind = str(requested_contract.get("kind") or "").strip()
    inferred_contract_required = bool(requested_contract_kind and requested_contract_kind != "none")
    final_contract_evidence_present = "final_requested_output_contract_evaluated" in trace
    final_output_contract_evaluated = bool(
        trace.get(
            "final_requested_output_contract_evaluated",
            not inferred_contract_required,
        )
    )
    final_output_contract_required = bool(
        trace.get(
            "final_requested_output_contract_required",
            inferred_contract_required,
        )
    )
    final_output_contract_satisfied = bool(
        trace.get(
            "final_requested_output_contract_satisfied",
            not inferred_contract_required,
        )
    )
    final_output_contract_reasons = list(trace.get("final_requested_output_contract_reasons") or [])
    if inferred_contract_required and not final_contract_evidence_present:
        final_output_contract_reasons = ["evaluation_not_completed"]
    final_output_contract_proven = bool(
        final_output_contract_evaluated
        and (not final_output_contract_required or final_output_contract_satisfied)
    )
    # Model-native is a positive authorship claim. The absence of a repair or
    # state serializer cannot establish that a model generated the bytes.
    # Fast paths previously satisfied those negative conditions and were
    # mislabeled model-native even when their generation count was zero.
    model_native_output = bool(
        (
            (engine_think_invoked and engine_reply_accepted)
            or (
                response_path == "protected_foreground"
                and trace.get("protected_foreground_generation_proven") is True
            )
        )
        and not qualified_recurrent_response_path
        and not post_generation_repair_applied
        and not unreceipted_runtime_replacement
        and not response_authority_kind
    )
    confidence = str(response_confidence or "").strip().lower()
    qualified_recurrent_path_proven = bool(
        qualified_recurrent_response_path
        and trace.get("qualified_recurrent_path_proven") is True
        and trace.get("qualified_recurrent_succeeded") is True
        and trace.get("model_generation_used") is False
        and live_mind_generation_required is False
        and isinstance(trace.get("qualified_recurrent_receipt"), dict)
        and bool(trace.get("qualified_recurrent_receipt"))
        and not trace.get("qualified_recurrent_delivery_errors")
    )
    live_mind_controls_structurally_bound = bool(
        (not live_mind_context_required)
        or qualified_recurrent_path_proven
        or (
            live_mind_controls_bound
            and live_mind_controls_application_satisfied
            and live_mind_surface_quality_gate_passed
        )
    )
    accepted_full_mind_response_paths = {
        "cognitive_engine",
        "protected_foreground",
        "cognitive_engine_completion_retry",
        # A completion retry that ran and then correctly KEPT the original.
        #
        # chat.py sets this path when completion_incumbent_preserved is true —
        # the retry happened, was compared against the incumbent, and the
        # incumbent won. That is the retry machinery working at its best, and
        # it was in neither this set nor the single-owner clause below, so the
        # turn was refused and the person got "I couldn't get to an answer I'd
        # stand behind" instead of the answer that had already been judged the
        # better of two.
        #
        # Measured live 2026-08-18:
        #   missing: response_path:cognitive_engine_completion_incumbent,
        #            duplicate_foreground_model_generation
        #   path=cognitive_engine_completion_incumbent generations=3
        #   completion_retries=2 consumed=True
        # Three generations for one incumbent plus two retries is exactly
        # 1 + completion_retry_count — the arithmetic already agreed. Only the
        # name was unrecognised.
        "cognitive_engine_completion_incumbent",
        "cognitive_engine_repair_retry",
        "cognitive_engine_devocatived",
        "cognitive_engine_desktop_plan",
        "cognitive_engine_memory_state_grounding",
        "cognitive_engine_identity_continuity_grounding",
        "cognitive_engine_runtime_fact_grounding",
        "cognitive_engine_capability_tail_grounding",
        "cognitive_engine_capability_catalog_grounding",
        "cognitive_engine_self_process_grounding",
        "cognitive_engine_self_condition",
        "cognitive_engine_self_condition_grounding",
        "cognitive_engine_self_condition_semantic_completion",
        "cognitive_engine_bounded_planning",
        "cognitive_engine_latent_cortex",
        "cognitive_engine_qualified_recurrent",
    }
    latent_cortex_response_path = response_path == "cognitive_engine_latent_cortex"
    latent_cortex_path_proven = bool(
        latent_cortex_response_path
        and (
            latent_cortex_selected
            and latent_cortex_attempted
            and latent_cortex_succeeded
            and not latent_cortex_fallback_used
            and latent_cortex_identity_bound
            and latent_cortex_output_quality_proven
        )
    )
    latent_cortex_path_requirement_satisfied = bool(
        not latent_cortex_response_path or latent_cortex_path_proven
    )
    qualified_recurrent_path_requirement_satisfied = bool(
        not qualified_recurrent_response_path or qualified_recurrent_path_proven
    )
    foreground_model_generation_count = int(trace.get("foreground_model_generation_count") or 0)
    foreground_model_generation_segment_count = int(
        trace["foreground_model_generation_segment_count"]
        if "foreground_model_generation_segment_count" in trace
        else foreground_model_generation_count
    )
    foreground_model_generation_transaction_count = int(
        trace["foreground_model_generation_transaction_count"]
        if "foreground_model_generation_transaction_count" in trace
        else foreground_model_generation_count
    )
    foreground_model_generation_transaction_id = str(
        trace.get("foreground_model_generation_transaction_id") or ""
    ).strip()
    foreground_model_generation_consumed = bool(trace.get("foreground_model_generation_consumed"))
    completion_retry_count = int(trace.get("completion_retry_count") or 0)
    repair_retry_attempt_count = int(trace.get("repair_retry_attempt_count") or 0)
    continuation_evidence_valid = bool(trace.get("continuation_evidence_valid", True))
    single_owner_model_generation_proven = bool(
        (
            live_mind_generation_required
            and foreground_model_generation_consumed
            and continuation_evidence_valid
            and bool(foreground_model_generation_transaction_id)
            and foreground_model_generation_transaction_count == 1
            and foreground_model_generation_segment_count == 1
            and foreground_model_generation_count == 1
        )
        or (
            not live_mind_generation_required
            and not foreground_model_generation_consumed
            and foreground_model_generation_count == 0
        )
        or (
            live_mind_generation_required
            # Keeping the incumbent is the same OWNERSHIP story as adopting the
            # retry: one owner generated once, then continued up to
            # _MAX_USER_SURFACE_CONTINUATIONS times. Which of those answers won
            # the comparison changes nothing about who authored them, and
            # naming only the adopt-the-retry outcome meant the better outcome
            # could not be served.
            and response_path
            in {
                "cognitive_engine_completion_retry",
                "cognitive_engine_completion_incumbent",
            }
            and foreground_model_generation_consumed
            and continuation_evidence_valid
            and bool(foreground_model_generation_transaction_id)
            and foreground_model_generation_transaction_count == 1
            and 1 <= completion_retry_count <= _MAX_USER_SURFACE_CONTINUATIONS
            and foreground_model_generation_segment_count == 1 + completion_retry_count
            and foreground_model_generation_count == 1 + completion_retry_count
        )
        or (
            live_mind_generation_required
            and response_path == "cognitive_engine_repair_retry"
            and foreground_model_generation_consumed
            and bool(foreground_model_generation_transaction_id)
            and foreground_model_generation_transaction_count == 2
            and repair_retry_attempt_count == 1
            and foreground_model_generation_segment_count == 2
            and foreground_model_generation_count == 2
        )
    )
    # SPEAKER-IDENTITY proofs: did Aura's real cognitive engine author this
    # text (vs repair machinery / legacy fallback speaking in her voice)?
    # These are never waived — theater must never serve as Aura speech.
    protected_foreground_generation_proven = bool(
        trace.get("protected_foreground_generation_proven")
    )
    authored_generation_source_proven = bool(
        (engine_think_invoked and engine_reply_accepted)
        or (
            response_path == "protected_foreground"
            and protected_foreground_generation_proven
        )
    )
    # Whose words these are, apart from whether they were good enough.
    #
    # `authored_generation_source_proven` requires the engine to have ACCEPTED
    # the reply, so authorship and quality are one flag. They are different
    # facts, and the last-resort salvage site needs the first: it exists
    # because of the second.
    #
    # LIVE, 2026-08-28: twelve reasoning questions came back with the canned
    # apology. The engine had run, written a real partial answer, judged it not
    # good enough, and the salvage refused to serve it for want of a proof that
    # said "the engine accepted this" — at a site reached only when it did not.
    # `single_owner_model_generation_proven` is deliberately NOT required here,
    # and the reason is what that proof is for. It answers "which of several
    # answers is being served as the one" — it enumerates the retry paths and
    # their exact generation counts, and a gate-level retry matches none of
    # them, so an ordinary second attempt reads as two owners.
    #
    # The salvage site does not choose between answers. It serves the single
    # preserved draft or nothing. What it needs to know is that those words
    # came from the engine rather than from repair machinery, a legacy
    # fallback, or runtime substitution, and the conditions below say exactly
    # that.
    #
    # LIVE, 2026-08-28: eleven of twelve reasoning questions came back with the
    # canned apology, the last gate being duplicate_foreground_model_generation
    # on a turn whose only duplication was retrying once.
    engine_authored_the_text = bool(
        engine_think_invoked
        and not engine_reply_failed
        and not bounded_contract_used
        and not legacy_fallback_used
        and not authorship_replacement_applied
    )
    authentic_cognitive_reply = bool(
        authored_generation_source_proven
        and not engine_reply_failed
        and not bounded_contract_used
        and not legacy_fallback_used
        and response_path in accepted_full_mind_response_paths
        and latent_cortex_path_requirement_satisfied
        and qualified_recurrent_path_requirement_satisfied
        and single_owner_model_generation_proven
        and not authorship_replacement_applied
    )
    semantic_completion_expected = bool(
        trace.get("semantic_completion_contract_expected")
        or trace.get("semantic_completion_contract")
    )
    semantic_completion_receipt_present = bool(
        trace.get("semantic_completion_receipt_present")
        or qualified_recurrent_path_proven
    )
    semantic_completion_satisfied = bool(
        trace.get("semantic_completion_satisfied")
        or qualified_recurrent_path_proven
    )
    state_native_output = bool(
        (qualified_recurrent_path_proven or response_authority_proven)
        and not post_generation_repair_applied
        and not unreceipted_runtime_replacement
        and not authorship_replacement_applied
    )
    if "authored_answer_completion_proven" in trace:
        # A merged append-only answer is assessed as one semantic object after
        # the final segment, so it may carry a stronger route-level proof than
        # the receipt for that final segment alone.
        authored_answer_completion_proven = bool(
            trace["authored_answer_completion_proven"]
        )
    else:
        authored_answer_completion_proven = bool(
            not trace.get("completion_retry_exhausted")
            and not trace.get("semantic_completion_incomplete")
            and not trace.get("reply_generation_incomplete")
            and (
                not semantic_completion_expected
                or (
                    semantic_completion_receipt_present
                    and semantic_completion_satisfied
                )
            )
        )
    answer_delivery_proven = bool(
        (authentic_cognitive_reply or response_authority_proven)
        and authored_answer_completion_proven
        and final_output_contract_proven
    )
    accepted_cognitive_path = bool(
        authentic_cognitive_reply
        and answer_delivery_proven
        and confidence == "high"
        and architecture_context_bound
        and live_mind_snapshot_bound
        and live_mind_controls_structurally_bound
    )
    subsystems = _attested_live_chat_required_subsystems(
        trace,
        generation_proven=authentic_cognitive_reply,
    )
    required_subsystems_source = "attested_preflight"
    if subsystems is None:
        required_subsystems_source = "compatibility_probe"
        subsystems = _collect_live_chat_required_subsystems(
            lane,
            generation_proven=authentic_cognitive_reply,
        )
    _expected_organs = _collect_expected_turn_organs()
    _note_organ_engagement(_expected_organs)
    required_subsystems_ok = all(subsystems.values())
    live_mind_required_subsystems_ok = required_subsystems_ok
    full_mind_path = bool(desktop_required and accepted_cognitive_path and required_subsystems_ok)
    # STATE-COMPLETENESS proofs, named individually so a refusal (or a
    # degraded-disclosure delivery) can say exactly what was missing —
    # a live incident was diagnosed blind because the refusal log printed
    # three flags that all turned out to be fine.
    missing_proofs: list[str] = []
    if not engine_think_invoked:
        missing_proofs.append("engine_think_not_invoked")
    if engine_reply_failed:
        missing_proofs.append("engine_reply_failed")
    if not engine_reply_accepted:
        missing_proofs.append("engine_reply_not_accepted")
    if bounded_contract_used:
        missing_proofs.append("bounded_repair_authored_text")
    if legacy_fallback_used:
        missing_proofs.append("legacy_fallback_authored_text")
    if authorship_replacement_applied:
        missing_proofs.append("runtime_replacement_authored_text")
    if confidence != "high":
        missing_proofs.append(f"confidence:{confidence or 'unset'}")
    if response_path not in accepted_full_mind_response_paths:
        missing_proofs.append(f"response_path:{response_path or 'unset'}")
    if latent_cortex_response_path and not latent_cortex_path_proven:
        missing_proofs.append("latent_cortex_path_unproven")
    if latent_cortex_response_path and not latent_cortex_output_quality_proven:
        missing_proofs.append("latent_cortex_output_quality_unproven")
    if qualified_recurrent_response_path and not qualified_recurrent_path_proven:
        missing_proofs.append("qualified_recurrent_path_unproven")
    if not single_owner_model_generation_proven:
        missing_proofs.append(
            "duplicate_foreground_model_generation"
            if foreground_model_generation_transaction_count > 1
            else "foreground_model_generation_ownership_unproven"
        )
    if not authored_answer_completion_proven:
        missing_proofs.append("authored_answer_incomplete")
    if not architecture_context_bound:
        missing_proofs.append("architecture_context_unbound")
    if not live_mind_snapshot_bound:
        missing_proofs.append("live_mind_snapshot_not_ready")
    if not live_mind_controls_structurally_bound:
        missing_proofs.append("live_mind_controls_unbound")
    if not final_output_contract_evaluated:
        missing_proofs.append("final_output_contract_not_evaluated")
    elif final_output_contract_required and not final_output_contract_satisfied:
        missing_proofs.append("final_output_contract_unsatisfied")
    if not required_subsystems_ok:
        missing_proofs.extend(
            f"subsystem:{name}" for name, healthy in sorted(subsystems.items()) if not healthy
        )
    return {
        "desktop_cognitive_engine_required": bool(desktop_required),
        "request_surface": str(request_surface or ""),
        "response_confidence": str(response_confidence or ""),
        "status": str(status or ""),
        "response_path": response_path,
        "authentic_cognitive_reply": authentic_cognitive_reply,
        "authored_generation_source_proven": authored_generation_source_proven,
        "protected_foreground_generation_proven": (
            protected_foreground_generation_proven
        ),
        "authored_answer_completion_proven": authored_answer_completion_proven,
        "answer_delivery_proven": answer_delivery_proven,
        "response_authority_kind": response_authority_kind,
        "response_authority_proven": response_authority_proven,
        "response_authority_reason": str(trace.get("response_authority_reason") or ""),
        "certification_complete": full_mind_path,
        "full_mind_missing_proofs": missing_proofs,
        "engine_think_invoked": engine_think_invoked,
        "foreground_model_generation_consumed": foreground_model_generation_consumed,
        "foreground_model_generation_count": foreground_model_generation_count,
        "foreground_model_generation_segment_count": foreground_model_generation_segment_count,
        "foreground_model_generation_transaction_count": (
            foreground_model_generation_transaction_count
        ),
        "foreground_model_generation_transaction_id": (
            foreground_model_generation_transaction_id
        ),
        "foreground_model_generation_output_sha256": str(
            trace.get("foreground_model_generation_output_sha256") or ""
        ),
        "semantic_completion_contract_expected": semantic_completion_expected,
        "semantic_completion_receipt_present": semantic_completion_receipt_present,
        "semantic_completion_satisfied": semantic_completion_satisfied,
        "semantic_completion_mode": (
            "certified_state_serialization"
            if qualified_recurrent_path_proven
            else (
                response_authority_kind
                if response_authority_proven
                else ("model_generation" if model_native_output else "unproven_runtime_output")
            )
        ),
        "completion_retry_count": completion_retry_count,
        "continuation_evidence_valid": continuation_evidence_valid,
        "repair_retry_attempt_count": repair_retry_attempt_count,
        "single_owner_model_generation_proven": single_owner_model_generation_proven,
        "cognitive_engine_reply_accepted": engine_reply_accepted,
        "engine_authored_the_text": engine_authored_the_text,
        "cognitive_engine_reply_failed": engine_reply_failed,
        "bounded_contract_used": bounded_contract_used,
        "legacy_fallback_used": legacy_fallback_used,
        "latent_cortex_selected": latent_cortex_selected,
        "latent_cortex_selection_reason": str(trace.get("latent_cortex_selection_reason") or ""),
        "latent_cortex_depth_worthy": bool(trace.get("latent_cortex_depth_worthy")),
        "latent_cortex_prompt_shape": (
            dict(trace.get("latent_cortex_prompt_shape") or {})
            if isinstance(trace.get("latent_cortex_prompt_shape"), dict)
            else {}
        ),
        "latent_cortex_attempted": latent_cortex_attempted,
        "latent_cortex_succeeded": latent_cortex_succeeded,
        "latent_cortex_fallback_used": latent_cortex_fallback_used,
        "latent_cortex_failure_reason": latent_cortex_failure_reason,
        "latent_cortex_identity_bound": latent_cortex_identity_bound,
        "latent_cortex_output_quality_proven": latent_cortex_output_quality_proven,
        "latent_cortex_raw_output_quality_proven": (latent_cortex_raw_output_quality_proven),
        "latent_cortex_final_output_quality_proven": (latent_cortex_final_output_quality_proven),
        "latent_cortex_public_output_quality_proven": (latent_cortex_public_output_quality_proven),
        "qualified_recurrent_public_output_quality_proven": (
            qualified_recurrent_public_output_quality_proven
        ),
        "latent_cortex_raw_final_quality_hash_match": (raw_final_quality_hash_match),
        "latent_cortex_raw_public_quality_hash_match": (raw_public_quality_hash_match),
        "latent_cortex_final_public_quality_hash_match": (final_public_quality_hash_match),
        "latent_cortex_final_output_quality": latent_cortex_final_output_quality,
        "latent_cortex_public_output_quality": latent_cortex_public_output_quality,
        "latent_cortex_raw_final_mutation_chain": (latent_cortex_raw_final_mutation_chain),
        "latent_cortex_final_public_mutation_chain": (latent_cortex_final_public_mutation_chain),
        "latent_cortex_output_mutation_chain": latent_cortex_output_mutation_chain,
        "latent_cortex_path_proven": latent_cortex_path_proven,
        "latent_cortex_path_requirement_satisfied": (latent_cortex_path_requirement_satisfied),
        "qualified_recurrent_path_proven": qualified_recurrent_path_proven,
        "qualified_recurrent_terminal_bytes_preserved": bool(
            trace.get("qualified_recurrent_terminal_bytes_preserved")
        ),
        "preflight_evidence_profile": preflight_evidence_profile,
        "preflight_evidence_owner": preflight_evidence_owner,
        "preflight_skipped_components": list(
            trace.get("preflight_skipped_components") or []
        ),
        "qualified_recurrent_path_requirement_satisfied": (
            qualified_recurrent_path_requirement_satisfied
        ),
        "qualified_recurrent_family": str(trace.get("qualified_recurrent_family") or ""),
        "qualified_recurrent_delivery_errors": list(
            trace.get("qualified_recurrent_delivery_errors") or []
        ),
        "latent_cortex_final_text_transformed": bool(
            trace.get("latent_cortex_final_text_transformed")
        ),
        "latent_cortex_receipt": latent_cortex_receipt,
        "latent_cortex_ingress": (
            dict(trace.get("latent_cortex_ingress"))
            if isinstance(trace.get("latent_cortex_ingress"), dict)
            else {}
        ),
        "latent_cortex_progress": (
            dict(trace.get("latent_cortex_progress"))
            if isinstance(trace.get("latent_cortex_progress"), dict)
            else {}
        ),
        "live_mind_context_required": live_mind_context_required,
        "live_mind_context_present": live_mind_context_present,
        "live_mind_snapshot_present": live_mind_snapshot_present,
        "live_mind_snapshot_ready": live_mind_snapshot_ready,
        "live_mind_snapshot_bound": live_mind_snapshot_bound,
        "live_mind_snapshot_missing_services": live_mind_snapshot_missing_services,
        "live_mind_controls_bound": live_mind_controls_bound,
        "live_mind_generation_controls_present": live_mind_generation_controls_present,
        "live_mind_generation_controls": live_mind_generation_controls,
        "live_mind_surface_control_receipt": live_mind_surface_control_receipt,
        "live_mind_controls_worker_applied": live_mind_controls_worker_applied,
        "live_mind_generation_required": live_mind_generation_required,
        "live_mind_controls_application_satisfied": live_mind_controls_application_satisfied,
        "live_mind_surface_quality_gate_enabled": live_mind_surface_quality_gate_enabled,
        "live_mind_surface_quality_gate_passed": live_mind_surface_quality_gate_passed,
        "worker_instruction_shape_repair_applied": worker_instruction_shape_repair_applied,
        "post_generation_repair_applied": post_generation_repair_applied,
        "deterministic_repair_applied": deterministic_repair_applied,
        "authorship_replacement_applied": authorship_replacement_applied,
        "authorship_augmentation_applied": authorship_augmentation_applied,
        "model_replacement_applied": model_replacement_applied,
        "mutation_authorship_effects": mutation_authorship_effects,
        "unreceipted_runtime_replacement": unreceipted_runtime_replacement,
        "runtime_grounding_response_path": bool(response_path in _RUNTIME_GROUNDING_RESPONSE_PATHS),
        "model_native_output": model_native_output,
        "state_native_output": state_native_output,
        "final_text_authorship": (
            "non_cognitive_replacement"
            if authorship_replacement_applied
            else (
                response_authority_kind
                if response_authority_proven
                else (
                    "certified_recurrent_state_serialization"
                    if state_native_output
                    else (
                        "cognitive_generation_with_runtime_evidence"
                        if authorship_augmentation_applied
                        else (
                            "model_native"
                            if model_native_output
                            else (
                                "cognitive_generation_with_recorded_transformations"
                                if authored_generation_source_proven
                                else "unproven_runtime_output"
                            )
                        )
                    )
                )
            )
        ),
        "text_mutations": text_mutations,
        "text_mutation_count": len(text_mutations),
        "generation_max_tokens": live_mind_surface_control_receipt.get("generation_max_tokens"),
        "generated_tokens": live_mind_surface_control_receipt.get("generated_tokens"),
        "semantic_output_token_cap": live_mind_surface_control_receipt.get(
            "semantic_output_token_cap"
        ),
        "hard_output_token_ceiling": live_mind_surface_control_receipt.get(
            "hard_output_token_ceiling"
        ),
        "requested_output_contract": live_mind_surface_control_receipt.get(
            "requested_output_contract"
        ),
        "final_requested_output_contract_evaluated": final_output_contract_evaluated,
        "final_requested_output_contract_required": final_output_contract_required,
        "final_requested_output_contract_kind": str(
            trace.get("final_requested_output_contract_kind") or requested_contract_kind or ""
        ),
        "final_requested_output_contract_satisfied": final_output_contract_satisfied,
        "final_requested_output_contract_reasons": final_output_contract_reasons,
        "final_requested_output_contract_proven": final_output_contract_proven,
        "live_mind_controls_structurally_bound": live_mind_controls_structurally_bound,
        "live_mind_required_subsystems_ok": live_mind_required_subsystems_ok,
        "preflight_live_mind_required_subsystems_ok": preflight_live_mind_required_subsystems_ok,
        "architecture_context_bound": architecture_context_bound,
        "full_mind_path": full_mind_path,
        "required_subsystems": subsystems,
        "required_subsystems_source": required_subsystems_source,
        "required_subsystems_ok": required_subsystems_ok,
        # Reported, never fatal. A persistent absence here is why a reply can
        # be technically correct and not sound like her.
        "expected_organs": _expected_organs,
        "absent_expected_organs": _absent_turn_organs(_expected_organs),
        "recent_context_needed": bool(trace.get("recent_context_needed")),
        "recent_context_exchanges": int(trace.get("recent_context_exchanges") or 0),
        "compact_desktop_chat_contract": bool(trace.get("compact_desktop_chat_contract")),
        "desktop_execution_contract": bool(trace.get("desktop_execution_contract")),
        "capability_inventory_contract": bool(trace.get("capability_inventory_contract")),
    }
