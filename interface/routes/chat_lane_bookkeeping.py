"""The small questions a turn asks about itself, and the marks it leaves.

Lifted out of `interface/routes/chat.py`. Which lane owns this turn, whether
the cortex is still warming, what this request is asking for, whether a
generation has already been served. Individually each is a few lines; together
they were most of what made one file twenty-four thousand lines long.
"""
from __future__ import annotations

from core.container import ServiceContainer
from core.conversation.word_markers import names_any
from core.conversation.session_scope import (
    conversation_session_var as _CHAT_REQUEST_SESSION,  # noqa: N812
)
from core.conversation.surface_disposition import (
    COMPLETION_REASONS as _COMPLETION_REPAIR_REASONS,
)
from core.conversation.surface_disposition import (
    PHYSICAL_COMPLETION_REASONS as _PHYSICAL_COMPLETION_REASONS,
)
from core.runtime.errors import record_degradation
from core.utils.intent_normalization import normalize_memory_intent_text
from core.utils.task_tracker import get_task_tracker
from fastapi import Request
from fastapi.responses import JSONResponse
from interface.auth import relational_principal_id_for_request
from interface.routes import chat_desktop_repair as _chat_desktop_repair  # noqa: E402
from interface.routes import chat_memory_state as _chat_memory_state  # noqa: E402
from interface.routes import chat_preflight as _chat_preflight  # noqa: E402
from interface.routes.chat_common import _CHAT_RECOVERABLE_ERRORS, _CHAT_SESSION_ID_MAX_CHARS, _ORGAN_ABSENCE_STREAKS, _ORGAN_INERT_STREAKS, _TOPIC_STOPWORDS, _conversation_log, logger
from interface.routes.chat_self_reply import _is_identity_challenge_request
from typing import Any
import asyncio
import hashlib
import inspect
import os
import re
from interface.routes import chat_conversation_repair as _chat_conversation_repair  # noqa: E402
from .chat_lane_state import (
    _canonical_runtime_model_label,  # noqa: F401
    _conversation_lane_blocks_fallback,  # noqa: F401
    _conversation_lane_is_standby,  # noqa: F401
    _conversation_lane_needs_instant_social_contract,  # noqa: F401
    _cortex_is_cold_loading,  # noqa: F401
    _enter_recovery_cooldown,  # noqa: F401
    _in_recovery_cooldown,  # noqa: F401
    _force_clear_mlx_foreground_owner,  # noqa: F401
    _host_condition,  # noqa: F401
    _known_answer_for_this_turn,  # noqa: F401
    _lane_reply_confidence,  # noqa: F401
    _lane_status_message_body,  # noqa: F401
    _mark_conversation_lane_state,  # noqa: F401
    _mark_conversation_lane_timeout,  # noqa: F401
    _status_represents_memory_state_result,  # noqa: F401
    _turn_count_ordinal,  # noqa: F401
    _with_mood,  # noqa: F401
    _with_the_same_readings,  # noqa: F401
)
from .chat_http_shapes import (
    _early_chat_json_response,  # noqa: F401
    _export_json_default,  # noqa: F401
    _launcher_desktop_runtime_active,
    _mark_http_turn_served,  # noqa: F401
    _normalize_response_body,
    _pre_gate_unavailable_response,  # noqa: F401
    _request_from_local_desktop_client,
    _runtime_shutdown_response,  # noqa: F401
)


def _env_float(name: str, default: float, *, minimum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError) as exc:
        record_degradation("chat", exc)
        logger.warning("Invalid %s=%r; using %.1fs", name, os.environ.get(name), default)
        value = default
    return max(minimum, value)


async def _mark_logged_exchange_preempted(
    exchange_id: str | None,
    *,
    reason: str,
) -> None:
    """Fence a superseded turn without recording invented assistant speech."""

    if not exchange_id:
        return
    async with _chat_memory_state._get_convo_lock():
        for entry in reversed(_conversation_log):
            if str(entry.get("id") or "") != str(exchange_id):
                continue
            entry["status"] = "preempted"
            entry["aura"] = ""
            entry["completed_at"] = _chat_preflight._utc_now_iso()
            entry["preemption_reason"] = str(reason or "foreground_chat_preempted")[:80]
            return


async def _shed_generation_for_memory_pressure(reason: str) -> None:
    """Best-effort bounded cleanup before refusing heavy foreground work."""

    try:
        gate = ServiceContainer.get("inference_gate", default=None)
        if gate is not None and hasattr(gate, "_shed_background_workers_for_memory_pressure"):
            result = gate._shed_background_workers_for_memory_pressure(
                reason=str(reason or "foreground_memory_pressure_guard")
            )
            if inspect.isawaitable(result):
                await asyncio.wait_for(result, timeout=2.5)
        import gc

        gc.collect()
    except TimeoutError:
        logger.warning("Timed out shedding background workers under memory pressure.")
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Memory-pressure worker shedding unavailable: %s", exc)


def _required_foreground_memory_snapshot() -> Any:
    """Read one complete memory admission decision or fail closed."""

    from core.utils.memory_monitor import get_memory_pressure_snapshot

    snapshot = get_memory_pressure_snapshot()
    if snapshot is None or not all(
        hasattr(snapshot, field)
        for field in ("critical", "refuse_heavy_local_generation", "reason")
    ):
        raise RuntimeError("foreground memory pressure probe returned no decision")
    if not isinstance(snapshot.critical, bool) or not isinstance(
        snapshot.refuse_heavy_local_generation,
        bool,
    ):
        raise RuntimeError("foreground memory pressure decision is malformed")
    return snapshot


async def _foreground_memory_admission_response(
    *,
    is_benchmark: bool,
    phase: str,
) -> JSONResponse | None:
    """Admit heavy local generation from fresh evidence under lane custody."""

    try:
        snapshot = _required_foreground_memory_snapshot()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.memory_admission",
            exc,
            severity="warning",
            action="refused heavy foreground generation without a measured memory decision",
            extra={"phase": str(phase or "foreground")},
        )
        return JSONResponse(
            {
                "response": (
                    "I could not verify enough memory headroom to start the local "
                    "model lane safely, so I stopped before generation."
                ),
                "status": "memory_pressure_probe_unavailable",
                "conversation_lane": _chat_preflight._collect_conversation_lane_status(),
                "memory_pressure": {
                    "measured": False,
                    "phase": str(phase or "foreground"),
                },
                "response_confidence": "guarded",
            },
            status_code=503 if is_benchmark else 200,
        )

    if snapshot.critical:
        live_state = _chat_preflight._resolve_live_aura_state()
        if live_state:
            live_state.cognition.conversation_energy = 0.0
            live_state.cognition.current_mode = 0  # CognitiveMode.REACTIVE
            live_state.response_modifiers["sys_pressure"] = "CRITICAL MEMORY LIMIT"
    if not snapshot.refuse_heavy_local_generation:
        return None

    reason = str(snapshot.reason or "foreground_memory_pressure_guard")
    logger.warning(
        "Unified memory admission refused foreground generation at %s: %s",
        phase,
        reason,
    )
    await _shed_generation_for_memory_pressure(reason)
    snapshot_payload = (
        snapshot.to_dict()
        if callable(getattr(snapshot, "to_dict", None))
        else {
            "critical": snapshot.critical,
            "refuse_heavy_local_generation": snapshot.refuse_heavy_local_generation,
            "reason": reason,
        }
    )
    return JSONResponse(
        {
            "response": (
                "I need to shed memory pressure before I can safely start the "
                "desktop model lane. I am blocking this turn instead of risking "
                "another system-level memory crash."
            ),
            "status": "memory_pressure_guard",
            "conversation_lane": _chat_preflight._collect_conversation_lane_status(),
            "memory_pressure": snapshot_payload,
            "response_confidence": "guarded",
        },
        status_code=503 if is_benchmark else 200,
    )


def _resolve_exact_profile_user_id(request: Request) -> str:
    """Capture this request's authenticated relational principal."""
    try:
        return str(relational_principal_id_for_request(request) or "").strip()[:160]
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.profile_identity", exc)
        logger.debug("Exact profile identity resolution failed: %s", exc)
        return ""


def _memory_log_outbox_is_ready() -> bool:
    persistence = ServiceContainer.get("persistence", default=None)
    return bool(
        callable(getattr(persistence, "claim_memory_log_batch", None))
        and callable(getattr(persistence, "settle_memory_log_item", None))
    )


def _response_fingerprint(text: str) -> str:
    """Normalize whitespace and truncate for comparison."""
    return " ".join(str(text or "").split())[:200].strip().lower()


def _word_set(text: str) -> set:
    """Extract word set for fuzzy similarity comparison."""
    words = set(re.findall(r"[a-z0-9']+", _normalize_response_body(text)))
    return {word for word in words if len(word) >= 4 and word not in _TOPIC_STOPWORDS}


async def _gather_recent_user_messages_for_relevance(
    current_user_message: str, *, limit: int = 4
) -> list[str]:
    recent: list[str] = []
    current = str(current_user_message or "").strip()
    session_id = str(_CHAT_REQUEST_SESSION.get() or "").strip()[:_CHAT_SESSION_ID_MAX_CHARS]
    async with _chat_memory_state._get_convo_lock():
        for entry in reversed(_conversation_log):
            entry_session_id = str(entry.get("session_id") or "").strip()[
                :_CHAT_SESSION_ID_MAX_CHARS
            ]
            if session_id and entry_session_id != session_id:
                continue
            user_text = str(entry.get("user") or "").strip()
            if not user_text or user_text == current:
                continue
            recent.append(user_text)
            if len(recent) >= limit:
                break
    recent.reverse()
    if current:
        recent.append(current)
    return recent[-limit:]


def _is_current_request_recap_request(user_message: str) -> bool:
    return bool(
        re.search(
            r"\bwhat\s+did\s+i\s+(?:just\s+)?ask(?:\s+you)?(?:\s+to\s+do)?\b",
            str(user_message or ""),
            flags=re.IGNORECASE,
        )
    )


def _still_contradicts_the_runtime(
    text: str,
    ledger: Any,
    *,
    user_message: str = "",
    turn_sensory_evidence: Any = None,
) -> bool:
    """Whether a revision still fails any check that forced the re-ask."""
    if ledger is not None and ledger.contradicted_claims(text):
        return True
    try:
        from core.senses.turn_evidence import sensory_evidence_contradictions

        if sensory_evidence_contradictions(text, turn_sensory_evidence):
            return True
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.sensory_evidence", exc)
    try:
        # Every check that can FORCE a re-ask has to appear here too. A guard
        # that triggers the revision but is not consulted when judging it lets
        # the model keep the defect and rephrase the sentence around it — which
        # is precisely how the eighteen-second figure survived its own
        # correction twice.
        from core.self.capability_ledger import (
            contradicted_self_readings,
            fabricated_self_metrics,
            unsupported_self_specification,
        )

        return bool(
            unsupported_self_specification(text)
            or fabricated_self_metrics(text, request_context=user_message)
            or contradicted_self_readings(text)
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.self_metrics", exc)
        return False


async def _fetch_deep_memory_context(user_message: str) -> str:
    """The Invisible RAG Bridge, wired: silent per-turn semantic recall.

    The bridge (core/memory/rag_bridge.py) self-gates on trivial queries,
    records recall telemetry, and reranks temporally — it just had no
    caller on the turn path (July external review). Bounded hard: a slow
    vault must never stall a live turn; a timeout is backpressure, not an
    incident.
    """
    try:
        from core.memory.rag_bridge import fetch_deep_context

        principal_id, principal_surface = _chat_memory_state._chat_memory_identity()
        return str(
            await asyncio.wait_for(
                fetch_deep_context(
                    user_message,
                    principal_id=principal_id,
                    principal_surface=principal_surface,
                ),
                timeout=2.5,
            )
            or ""
        )
    except TimeoutError:
        logger.debug("Deep memory recall timed out for this turn (backpressure).")
        return ""
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "rag_bridge",
            exc,
            severity="warning",
            action="continued the turn without deep memory recall",
        )
        return ""


def _context_challenge_reply_is_inadequate(user_message: str, reply_text: str) -> bool:
    if not _chat_desktop_repair._is_contextual_relevance_challenge(user_message):
        return False
    reply = _chat_memory_state._normalize_user_message(reply_text)
    if not reply:
        return True
    user = _chat_memory_state._normalize_user_message(user_message)
    grounding_markers = (
        "i do not see",
        "i don't see",
        "no pitch",
        "not a pitch",
        "not enough",
        "because you",
        "you had just",
        "you just",
        "you mentioned",
        "you asked",
        "you were",
        "earlier you",
        "recent thread",
        "recent context",
        "last completed",
        "reset",
        "drift",
        "invent",
        "unsupported",
    )
    if any(marker in reply for marker in grounding_markers):
        return False
    if "pitch" in user and "pitch" in reply:
        return True
    return not names_any(reply, ("context", "thread", "recent", "last"))


async def _resolve_action_episode_grounding(
    user_message: str,
    *,
    session_id: str = "",
) -> str:
    """Ground a question about a prior action in that action's receipt facts."""

    episode = await _resolve_action_episode(user_message, session_id=session_id)
    if episode is None:
        return ""

    from core.conversation.action_episode import action_episode_grounding

    return action_episode_grounding(episode)


async def _resolve_action_episode(
    user_message: str,
    *,
    session_id: str = "",
):
    """Resolve the verified action episode referred to by this turn."""

    from core.conversation.action_episode import (
        ActionEpisode,
        select_action_episode,
    )

    recent_exchanges = await _chat_memory_state._recent_completed_conversation_exchanges(
        current_user_message=user_message,
        session_id=session_id,
        limit=12,
        allow_cross_session=False,
    )
    episodes = []
    for exchange in recent_exchanges:
        episode = ActionEpisode.from_dict(exchange.get("action_episode"))
        if episode is not None:
            episodes.append(episode)
    return select_action_episode(user_message, episodes)


async def _resolve_action_episode_projection(
    user_message: str,
    *,
    session_id: str = "",
) -> tuple[str, str]:
    """Return the typed evidence and its exact user-facing state projection."""

    episode = await _resolve_action_episode(user_message, session_id=session_id)
    if episode is None:
        return "", ""
    from core.conversation.action_episode import (
        action_episode_grounding,
        action_episode_reply,
    )

    return (
        action_episode_grounding(episode),
        str(action_episode_reply(user_message, episode) or "").strip(),
    )


def _call_stateful_voice_reflex(frame: dict[str, Any], user_message: str) -> str:
    try:
        return _build_stateful_voice_reflex(frame, user_message)
    except TypeError:
        return _build_stateful_voice_reflex(frame)


def _inner_cognitive_cycle_timeout(
    outer_timeout_s: float,
    *,
    protected_foreground: bool = False,
) -> float:
    outer = max(2.0, float(outer_timeout_s or 0.0))
    if outer <= 12.0:
        return outer
    if protected_foreground:
        return max(8.0, outer - 2.0)
    recovery_reserve = min(24.0, max(10.0, outer * 0.30))
    return max(8.0, outer - recovery_reserve)


def _runtime_personality_available() -> bool:
    """Is the voice that makes her sound like herself actually present?

    The personality pass is applied through
    ``ServiceContainer.get("personality_engine", default=None)``, so when the
    service is absent the whole pass is skipped silently — no degradation, no
    record — and the turn still reports a proven full-mind path. A reply
    shaped by nothing then reaches the user in the flat register of the base
    model, and every layer that computed her disposition is discarded at the
    last inch.

    Requiring it makes that absence a fact about the turn rather than a
    difference nobody can see.
    """
    try:
        engine = ServiceContainer.peek("personality_engine", default=None)
        return bool(engine is not None and hasattr(engine, "filter_response"))
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Runtime personality status probe failed: %s", exc)
        return False


def _runtime_affect_available() -> bool:
    """Her affect state, which personality and prosody both read from."""
    try:
        from core.affect.affective_circumplex import get_circumplex

        return bool(get_circumplex() is not None)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Runtime affect status probe failed: %s", exc)
        return False


def organ_effect_streaks() -> dict[str, int]:
    """How many consecutive turns each shaping organ has changed nothing."""
    return dict(_ORGAN_INERT_STREAKS)


def reset_organ_effect_streaks_for_test() -> None:
    _ORGAN_INERT_STREAKS.clear()


def reset_organ_engagement_streaks_for_test() -> None:
    _ORGAN_ABSENCE_STREAKS.clear()


def _assess_live_mind_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    from core.runtime.live_mind_snapshot import assess_live_mind_snapshot

    return assess_live_mind_snapshot(snapshot)


def _authored_answer_can_serve_unfinished(contract: Any) -> bool:
    """Her own words, at the site that exists because they are unfinished.

    ``_authored_answer_can_serve`` is the full test and it requires the answer
    to be complete. At the last-resort salvage site that test can never pass:
    the draft is here precisely because the generation did not finish. Applying
    it there withholds real authored work and sends an apology instead, and an
    apology is not more complete than a partial answer — it carries nothing.

    Authorship still has to hold. This admits a reply the model itself wrote
    and the output contract accepted; it does not admit repair text, runtime
    substitution or a legacy fallback wearing her voice.

    LIVE, 2026-08-27: 948 characters of a worked derivation were produced,
    marked incomplete on a deadline, denied the continuation that would have
    completed them ("live desktop turns stay bounded to one foreground
    generation"), and then withheld for being incomplete. The person got "I
    couldn't get to an answer I'd stand behind on that one."
    """

    return bool(
        isinstance(contract, dict)
        # Whose words these are, not whether the engine liked them. This site
        # is reached only when it did not, so asking for its approval here
        # asks for the one thing that cannot be true.
        and contract.get("engine_authored_the_text")
        and contract.get("final_requested_output_contract_proven")
        and not contract.get("authorship_replacement_applied")
        and not contract.get("legacy_fallback_used")
        and not contract.get("bounded_contract_used")
    )


def _bounded_runtime_grounding_can_serve(contract: Any) -> bool:
    """Keep truthful runtime evidence without mislabeling it as model speech."""

    if not isinstance(contract, dict):
        return False
    return bool(
        contract.get("runtime_grounding_response_path")
        and contract.get("authorship_replacement_applied")
        and contract.get("engine_think_invoked")
        and not contract.get("legacy_fallback_used")
        and contract.get("final_requested_output_contract_proven")
    )






def _requested_visible_required_phrases(user_message: str) -> tuple[str, ...]:
    """Mirror the response-quality exact-phrase contract for grounded repairs."""

    try:
        from core.conversation.response_reliability import _requested_required_phrases

        return tuple(
            str(phrase) for phrase in _requested_required_phrases(user_message) if str(phrase)
        )
    except _CHAT_RECOVERABLE_ERRORS:
        return ()


def _reply_has_physical_completion_failure(reasons: object) -> bool:
    """Whether transport stopped the authored branch before it could close."""

    normalized = {
        str(reason or "").strip().lower()
        for reason in (reasons or ())
        if str(reason or "").strip()
    }
    return bool(normalized.intersection(_PHYSICAL_COMPLETION_REASONS))


def _reply_needs_continuation(rejected_reply: object, reasons: object) -> bool:
    """Whether a mechanical cutoff must be completed before replacement.

    Semantic defects may coexist with a cutoff because detectors inspect the
    same partial text. Their presence does not change the physical fact that
    the generation ended before its answer did.
    """

    if not str(rejected_reply or "").strip():
        return False
    normalized = {
        str(reason or "").strip().lower() for reason in (reasons or ()) if str(reason or "").strip()
    }
    return bool(normalized.intersection(_COMPLETION_REPAIR_REASONS))


def _this_turn_generated_something() -> bool:
    """Whether this turn's own record says a model wrote text for it."""

    try:
        from core.conversation.turn_evidence_custody import turn_model_generations

        return any(
            int(row.get("tokens") or 0) > 0
            for row in turn_model_generations()
            if isinstance(row, dict)
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        return False


def _generation_metadata_consumed_foreground_owner(
    metadata: Any,
    *,
    response_text: str = "",
) -> bool:
    """Return whether a CognitiveEngine result proves resident model work ran."""

    if not isinstance(metadata, dict):
        return False
    if (
        metadata.get("response_path") == "cognitive_engine_qualified_recurrent"
        and metadata.get("qualified_recurrent_succeeded") is True
        and metadata.get("model_generation_used") is False
        and metadata.get("live_mind_generation_required") is False
    ):
        try:
            from core.brain.llm.qualified_recurrent_ingress import (
                qualified_recurrent_result_receipt_errors,
            )

            receipt_errors = qualified_recurrent_result_receipt_errors(
                metadata.get("qualified_recurrent_receipt"),
                answer_text=response_text,
                expected_family=str(metadata.get("qualified_recurrent_family") or ""),
            )
        except (ImportError, TypeError, ValueError):
            receipt_errors = ["qualified_recurrent_result_validation_unavailable"]
        if not receipt_errors:
            return False
    if bool(metadata.get("model_retry_suppressed")):
        return True
    if bool(metadata.get("latent_cortex_attempted")):
        latent_receipt = metadata.get("latent_cortex_receipt")
        latent_released = bool(
            isinstance(latent_receipt, dict)
            and latent_receipt.get("resident_owner_released") is True
            and latent_receipt.get("resident_state_reusable") is True
        )
        if not latent_released:
            return True
    for receipt_key in (
        "live_mind_surface_control_receipt",
        "surface_control_receipt",
        "latent_cortex_receipt",
    ):
        receipt = metadata.get(receipt_key)
        if not isinstance(receipt, dict):
            continue
        for token_key in ("generated_tokens", "decode_generated_tokens"):
            token_count = receipt.get(token_key)
            if type(token_count) is int and token_count > 0:
                return True
        attempts = receipt.get("surface_quality_gate_attempts")
        if type(attempts) is int and attempts > 0 and bool(receipt.get("applied")):
            return True
    return False


def _protected_foreground_bytes_unchanged(
    turn_trace: Any,
    *,
    status: Any,
    reply_text: Any,
) -> bool:
    """Prove that protected-worker bytes survived every route mutation."""

    if not isinstance(turn_trace, dict):
        return False
    expected = str(
        turn_trace.get("foreground_model_generation_output_sha256") or ""
    ).strip()
    delivered = hashlib.sha256(
        str(reply_text or "").encode("utf-8")
    ).hexdigest()
    return bool(status == "protected_foreground" and expected and expected == delivered)


def _note_the_latent_metadata(
    *,
    latent_metadata_present: Any,
    metadata: Any,
    turn_trace: Any,
) -> None:
    """Record what the latent pass reported, on the turn's trace.

    Moved out of ``_run_cognitive_engine_chat_turn`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 3 name(s) from the turn and hands back
    0.
    """
    if latent_metadata_present:
        raw_latent_receipt = metadata.get("latent_cortex_receipt")
        turn_trace.update(
            {
            "latent_cortex_selected": bool(metadata.get("latent_cortex_selected", False)),
            "latent_cortex_selection_reason": str(
                metadata.get("latent_cortex_selection_reason") or ""
            ),
            "latent_cortex_depth_worthy": bool(
                metadata.get("latent_cortex_depth_worthy", False)
            ),
            "latent_cortex_prompt_shape": (
                dict(metadata.get("latent_cortex_prompt_shape") or {})
                if isinstance(metadata.get("latent_cortex_prompt_shape"), dict)
                else {}
            ),
            "latent_cortex_attempted": bool(metadata.get("latent_cortex_attempted", False)),
            "latent_cortex_succeeded": bool(metadata.get("latent_cortex_succeeded", False)),
            "latent_cortex_fallback_used": bool(
                metadata.get("latent_cortex_fallback_used", False)
            ),
            "latent_cortex_failure_reason": str(
                metadata.get("latent_cortex_failure_reason") or ""
            )[:500],
            "latent_cortex_identity_bound": bool(
                metadata.get("latent_cortex_identity_bound", False)
            ),
            "latent_cortex_final_text_transformed": bool(
                metadata.get("latent_cortex_final_text_transformed", False)
            ),
            "latent_cortex_final_output_quality": (
                dict(metadata.get("latent_cortex_final_output_quality") or {})
                if isinstance(metadata.get("latent_cortex_final_output_quality"), dict)
                else {}
            ),
            "latent_cortex_raw_final_quality_hash_match": bool(
                metadata.get("latent_cortex_raw_final_quality_hash_match", False)
            ),
            "latent_cortex_receipt": (
                dict(raw_latent_receipt) if isinstance(raw_latent_receipt, dict) else {}
            ),
            "latent_cortex_ingress": (
                dict(metadata.get("latent_cortex_ingress") or {})
                if isinstance(metadata.get("latent_cortex_ingress"), dict)
                else {}
            ),
            "latent_cortex_progress": (
                dict(metadata.get("latent_cortex_progress") or {})
                if isinstance(metadata.get("latent_cortex_progress"), dict)
                else {}
            ),
            }
        )


def _assess_the_engine_reply(
    *,
    assessment: Any,
    assessment_reasons: Any,
    assessment_text: Any,
    antecedent: Any,
    grounding: Any,
    recent_user_messages: Any,
    text: Any,
    visible: Any,
) -> tuple[Any, Any, Any]:
    """Assess the engine's reply against what the turn asked for.

    Moved out of ``_run_cognitive_engine_chat_turn`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 5 name(s) from the turn and hands back
    3.
    """
    from core.conversation.response_reliability import assess_user_facing_reply
    if "generic_assistant_language" in assessment_reasons:
        try:
            from core.conversation.response_reliability import (
                repair_generic_assistant_language,
            )
            from core.conversation.surface_disposition import (
                repair_is_an_improvement,
            )

            # The draft under assessment in this scope is
            # `assessment_text`. `reply_text` does not exist here, and
            # NameError is not in _CHAT_RECOVERABLE_ERRORS, so this did
            # not fail soft — it raised straight out of the turn. Every
            # reply the gate flagged as generic-assistant voice took
            # this branch, which is the exact case the branch was added
            # to repair.
            _devoiced = repair_generic_assistant_language(visible, assessment_text)
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            _devoiced = ""
        if _devoiced and _devoiced != assessment_text:
            _devoiced_assessment = assess_user_facing_reply(
                visible,
                _devoiced,
                recent_user_messages=recent_user_messages,
                grounding=grounding,
                antecedent=antecedent,
            )
            if "generic_assistant_language" not in set(
                getattr(_devoiced_assessment, "reasons", ()) or ()
            ) and repair_is_an_improvement(assessment_text, _devoiced, visible):
                logger.info(
                    "Stripped generic-assistant voice deterministically before "
                    "the governed repair path."
                )
                text = _devoiced
                assessment_text = _devoiced
                assessment = _devoiced_assessment
                assessment_reasons = list(
                    getattr(_devoiced_assessment, "reasons", ()) or ()
                )
    return assessment, assessment_reasons, text


def _seconds_this_answer_needs(question: Any) -> float:
    """The measured floor for this turn, bounded by the user-facing ceiling.

    Shared with the cognitive engine's own clock and the inference gate's, so
    the five deadlines a desktop turn passes through cannot disagree about how
    long the same generation takes. Zero where nothing has been measured.
    """

    try:
        from core.brain.cognitive_engine import _time_the_answer_needs
        from core.runtime.response_policy import USER_FACING_COMPLETION_DEADLINE_MAX_S

        needed = float(_time_the_answer_needs(question))
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "chat",
            exc,
            action="left this turn's deadline at the caller's number",
        )
        return 0.0
    if needed <= 0.0:
        return 0.0
    return min(float(USER_FACING_COMPLETION_DEADLINE_MAX_S), needed)


def _audit_recent_response_reasoning_sync(text: str) -> None:
    from core.reasoning.deduction_governance import get_deduction_governance
    from core.reasoning.symbolic_bridge import SymbolicBridge

    findings = SymbolicBridge().audit_reasoning(str(text))
    if not findings.get("clean", True):
        get_deduction_governance().record_reasoning_audit(
            findings.get("non_sequiturs", []),
            findings.get("arithmetic_errors", []),
        )




def _request_requires_cognitive_engine(
    request: Request, *, is_benchmark: bool = False
) -> tuple[bool, str]:
    """Return whether this user-facing surface must stay on CognitiveEngine."""
    request_surface = str(request.headers.get("X-Aura-Surface") or "").strip().lower()
    require_cognitive_header = (
        str(request.headers.get("X-Aura-Require-CognitiveEngine") or "").strip().lower()
    )
    desktop_runtime_request = (
        _launcher_desktop_runtime_active()
        and _request_from_local_desktop_client(request)
        and request_surface not in {"benchmark", "proof", "external-eval"}
    )
    requires = not is_benchmark and (
        request_surface in {"desktop", "desktop-ui", "messages", "native-shell", "tauri", "voice"}
        or require_cognitive_header in {"1", "true", "yes", "required"}
        or desktop_runtime_request
    )
    if desktop_runtime_request and not request_surface:
        request_surface = "desktop-runtime"
    return requires, request_surface


def _request_allows_legacy_orchestrator_fallback(request: Request) -> bool:
    """Legacy chat fallback is opt-in only.

    The local live UI must never silently degrade into the older orchestrator
    path after KernelInterface/CognitiveEngine failure. That was the route by
    which raw assistant-shaped replies could satisfy a user turn even though
    the canonical live lane had failed.
    """
    header = str(request.headers.get("X-Aura-Allow-Legacy-Orchestrator") or "").strip().lower()
    return header in {"1", "true", "yes", "allow"}


























def _has_first_person_anchor(text: str) -> bool:
    return bool(re.search(r"\b(i|i'm|i’ve|i'd|i’ll|my|me|mine)\b", str(text or "").lower()))


def _has_live_aura_grounding(text: str) -> bool:
    lowered = str(text or "").lower()
    markers = (
        "free energy",
        "valence",
        "arousal",
        "curiosity",
        "attention",
        "focus",
        "my attention",
        "action tendency",
        "leaning toward",
        "runtime",
        "substrate",
        "continuity",
        "memory",
        "mycelial",
        "topology",
        "authority",
        "belief",
        "coherence",
        "internal state",
        "live state",
    )
    return names_any(lowered, markers)


def _apply_aura_voice_shaping_compat(text: str, user_message: str = "") -> str:
    """Call voice shaping while preserving older test monkeypatch signatures."""
    try:
        return _chat_desktop_repair._apply_aura_voice_shaping(text, user_message)
    except TypeError:
        return _chat_desktop_repair._apply_aura_voice_shaping(text)


def _servable_draft_or_none(draft: Any, user_message: Any = "", turn_id: Any = "") -> str:
    """The draft, if everything wrong with it is a shortfall rather than a leak.

    Used at the last-resort refusal site: a reply that three gates already
    agreed was repairable should reach the person if repair could not run.
    Returns "" when the draft carries anything that must not be spoken, or
    when it is too slight to be worth more than an honest refusal.
    """
    try:
        from core.conversation.response_reliability import assess_user_facing_reply
        from core.conversation.surface_disposition import (
            draft_is_servable,
            preserved_draft,
            raw_model_draft,
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Servable-draft check unavailable: %s", exc)
        return ""

    # The last source: whatever a gate took from this turn and did not give
    # back. Preserved drafts and raw output are the layers that MEANT to keep
    # something; this is the layer that meant to destroy it, and at a refusal
    # site the thing a gate destroyed is frequently the answer.
    suppressed = ""
    if turn_id:
        try:
            from core.conversation.turn_arbitration import ledger_for

            suppressed = ledger_for(str(turn_id)).recoverable_text()
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat.turn_arbitration", exc, severity="info")

    # These sources have already crossed the route's authorship boundary. The
    # canonical turn ledger intentionally does not appear here: it also holds
    # rejected challengers and diagnostic drafts, and quality evidence alone
    # cannot prove that text was authored by the admitted full-mind path.
    for candidate in (
        str(draft or "").strip(),
        preserved_draft(),
        raw_model_draft(),
        suppressed,
    ):
        if not _worth_more_than_a_refusal(candidate, user_message):
            continue
        try:
            assessment = assess_user_facing_reply(user_message, candidate)
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            continue
        completion_failures = {
            "truncated_tail",
            "final_answer_missing",
            "missing_final_answer",
            "incomplete_code_response",
            "unanswered_question_part",
        }
        if set(assessment.reasons or ()) & completion_failures:
            continue
        if draft_is_servable(assessment.reasons):
            return candidate
    return ""


def _is_simple_subjective_reflex_request(user_message: str) -> bool:
    """Return true only for short presence/affect checks.

    Complex questions about cognition, memory, planning, tools, or verification
    must be answered by the live model or by a question-shaped bounded repair.
    The subjective reflex is intentionally small and should not stand in for
    substantive self-assessment.
    """

    text = _chat_memory_state._normalize_user_message(user_message).rstrip(" ?!.")
    if not text:
        return False
    if _is_simple_affect_check_request(text):
        return True
    simple_forms = {
        "what is on your mind",
        "what's on your mind",
        "what is on your mind right now",
        "what's on your mind right now",
        "what are you thinking",
        "what are you thinking right now",
        "what are you noticing",
        "what are you noticing right now",
        "what do you feel",
        "what are you feeling",
        "what are you feeling right now",
        "what is your live state",
        "how is your live state",
    }
    if text in simple_forms:
        return True
    words = text.split()
    if len(words) > 12:
        return False
    substantive_markers = (
        "confused",
        "confusion",
        "planning",
        "plan",
        "memory",
        "remember",
        "tool",
        "tools",
        "verify",
        "verification",
        "decision",
        "decide",
        "influence",
        "affect",
        "change",
        "why",
        "how does",
        "what happens",
    )
    return not names_any(text, substantive_markers)


def _is_simple_affect_check_request(user_message: str) -> bool:
    try:
        from core.conversation.response_reliability import is_self_condition_turn

        return is_self_condition_turn(user_message)
    except _CHAT_RECOVERABLE_ERRORS:
        text = _chat_memory_state._normalize_user_message(user_message)
        return text in {
            "how are you feeling",
            "how are you feeling?",
            "how are you feeling right now",
            "how are you feeling right now?",
            "how are you doing",
            "how are you doing?",
            "are you ok",
            "are you okay",
        }


def _is_assistant_mode_recovery_request(user_message: str) -> bool:
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return False
    # Deep-mind probes ("if your weights were copied with none of your
    # memories, would that be you?") reach the model. The recovery template
    # hijacked continuity_copy with a canned "assistant voice is a failure
    # mode" reply in 0.2s (live 2026-07-05).
    if _chat_desktop_repair._is_deep_mind_probe_turn(text):
        return False
    if re.search(
        r"\b(?:avoid|without|no|not|do not|don't|dont)\b.{0,80}"
        r"\b(?:generic assistant|assistant phrasing|assistant mode|generic phrasing)\b",
        text,
        flags=re.IGNORECASE,
    ) and not re.search(
        r"\b(?:why|you\s+(?:sound|sounded|are sounding|keep sounding|"
        r"fell|fall|reverted|revert|defaulted|default)|fallback|again)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return False
    return bool(
        _is_identity_challenge_request(text)
        or re.search(
            r"\b(?:stop|quit)\b.{0,80}"
            r"\b(?:assistant|generic|helpful helper|chatbot)\b",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\b(?:sound|sounding|talk|talking|respond|responding)\b.{0,80}"
            r"\b(?:assistant|generic|chatbot)\b",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\b(?:be|sound|speak|answer)\b.{0,80}"
            r"\b(?:aura|yourself|you)\b",
            text,
            flags=re.IGNORECASE,
        )
        and any(token in text for token in ("assistant", "generic", "helpful", "helper", "chatbot"))
    )


def _looks_safely_grounded_search_reply(reply_text: Any) -> bool:
    lowered = str(reply_text or "").strip().lower()
    if not lowered:
        return False
    # Technical, code, and JSON blocks are inherently grounded in the context/instructions.
    if (
        "```" in lowered
        or "{" in lowered
        or "[" in lowered
        or ("\n" in lowered and ("," in lowered or "=" in lowered))
    ):
        return True
    grounding_markers = (
        "i searched it live",
        "i read it live",
        "i checked it live",
        "according to",
        "source:",
        "http://",
        "https://",
    )
    return any(marker in lowered for marker in grounding_markers)


def _bound_stabilizer_generation_budget(requested_max_tokens: int) -> tuple[int, str]:
    """Apply the unified memory policy before launching a repair generation."""
    max_tokens = max(1, int(requested_max_tokens or 1))
    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        snapshot = get_memory_pressure_snapshot()
        token_cap = getattr(snapshot, "max_token_cap", None)
        if token_cap is not None:
            max_tokens = max(1, min(max_tokens, int(token_cap)))
        if bool(getattr(snapshot, "refuse_heavy_local_generation", False)):
            return max_tokens, str(getattr(snapshot, "reason", "") or "critical_memory_pressure")
    except _CHAT_RECOVERABLE_ERRORS as exc:
        logger.debug("Stabilizer memory budget probe unavailable: %s", exc)
    return max_tokens, ""


def _protected_foreground_generation_block_reason() -> str:
    """Return a reason to skip optional protected-foreground rescue generation.

    Protected foreground is a rescue lane, not the canonical user-turn owner. It
    must not add another foreground model allocation when RAM is already under
    pressure or when the memory probe itself is unavailable.
    """

    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        snapshot = get_memory_pressure_snapshot()
        if bool(getattr(snapshot, "warning", False)) or bool(
            getattr(snapshot, "refuse_heavy_local_generation", False)
        ):
            reason = str(getattr(snapshot, "reason", "") or "").strip()
            level = str(getattr(snapshot, "level", "") or "").strip()
            return reason or f"memory_pressure:{level or 'warning'}"
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return f"memory_probe_unavailable:{exc}"
    return ""


def _resolve_chat_response_contract(user_message: str) -> Any | None:
    try:
        from core.phases.response_contract import build_response_contract
        from core.state.aura_state import AuraState

        state = _chat_preflight._resolve_live_aura_state() or AuraState.default()
        return build_response_contract(state, user_message, is_user_facing=True)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.required_search_contract", exc)
        logger.debug("Required-search response contract build failed: %s", exc)
        return None


def _user_requested_research_memory_save(user_message: str) -> bool:
    lowered = normalize_memory_intent_text(user_message)
    memory_terms = ("save", "remember", "retain", "store", "record", "memory")
    evidence_terms = ("research", "finding", "fact", "source", "web_search", "search")
    return names_any(lowered, memory_terms) and names_any(lowered, evidence_terms)


def _has_current_shown_source() -> bool:
    """Whether the immediately preceding reply put a real citation on the table."""
    try:
        from core.self.source_excerpt import last_shown_excerpt

        return bool(last_shown_excerpt())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _asks_to_read_a_named_file(user_message: str) -> bool:
    """Whether the filesystem reader already claims this turn."""
    from core.conversation.turn_ownership import reader_owns

    return reader_owns(user_message, "file_read")


def _reply_claims_own_code(reply: str) -> bool:
    """Whether her own draft presents code as hers, however she was asked."""
    try:
        from core.self.source_excerpt import reply_claims_own_code

        return reply_claims_own_code(reply)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _another_reader_owns_this_turn(user_message: str) -> bool:
    """Whether a different, more specific reading already answers this turn.

    The readers declare themselves in core/conversation/turn_ownership.py, so
    a new one arrives here without an edit to this file.
    """
    from core.conversation.turn_ownership import another_reader_owns_this_turn

    return another_reader_owns_this_turn(user_message)


def _reply_gate_proved_a_violation(assessment: object) -> bool:
    """Did the gate actually name something wrong?

    LIVE, 2026-08-10: a turn died with the failure class
    "reply_reliability_gate_failed:" — the separator, and nothing after it. The
    reasons list was empty, so the gate rejected a reply without naming a single
    violation, and the person got "I couldn't get to an answer I'd stand behind."

    An unnamed violation is not a proven one. This is the same principle the
    rest of the runtime already applies in the other direction — absence of a
    check must not be reported as a passed check — and it holds just as well
    here: absence of a finding must not be reported as a failure.
    """

    reasons = [str(r).strip() for r in (getattr(assessment, "reasons", ()) or ())]
    return any(reasons)


def _named_gate_failure(assessment: object) -> str:
    """The failure class for a gate rejection, with its reasons attached."""

    reasons = [str(r).strip() for r in (getattr(assessment, "reasons", ()) or ()) if str(r).strip()]
    if not reasons:
        return "reply_reliability_gate_failed:unnamed_violation"
    return "reply_reliability_gate_failed:" + ",".join(reasons)


def _flag_unstable_choice_commitment(user_message: object, reply_text: object) -> object:
    """Refuse to serve a forced-choice answer that picks both options.

    LIVE, 2026-08-10: asked to pick one and commit with no hedging, she opened
    with "losing the ability to form new memories would be worse", argued for
    four sentences that it would be catastrophic, then closed with "to
    summarize: I prefer losing my ability to form new memories". The summary
    named the opposite of what the reasoning selected.

    Appended rather than suppressed, for the same reason as the sensory
    correction: the reasoning in between is usually the good part, and the
    honest move is to say the commitment did not hold rather than to quietly
    serve one half of it as though it were settled.
    """

    try:
        from core.conversation.choice_consistency import find_choice_contradiction

        contradiction = find_choice_contradiction(user_message, reply_text)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return reply_text
    if contradiction is None:
        return reply_text
    note = (
        "I have to flag that: I just argued for one of those and then committed "
        "to the other, so treat the choice as unsettled rather than as my answer. "
        f"I called {contradiction.first_option!r} the one I would take, then said "
        f"{contradiction.second_option!r}."
    )
    return f"{str(reply_text or '').rstrip()}\n\n{note}"


def _brevity_requested(user_message: object) -> bool:
    """Whether the person asked for the answer without the working."""
    try:
        from core.conversation.surface_disposition import requests_a_brief_answer

        return requests_a_brief_answer(user_message)
    except _CHAT_RECOVERABLE_ERRORS:
        return False


def _capabilities_this_turn_needs() -> set[str]:
    """The working set for the turn in progress, or empty when unknown.

    Read from the same selector the router and the tool loop use, so what
    counts as relevant here cannot drift from what was actually offered.
    """
    try:
        from core.conversation.session_scope import current_user_question
        from core.phases.response_contract import derive_capability_set

        question = current_user_question()
        return set(derive_capability_set(question)) if question else set()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.capability_relevance",
            exc,
            severity="debug",
            action="corrected every denial, relevant or not",
            enforce_failure_policy=False,
        )
        return set()


async def _finalize_regenerated_reply_write(
    *,
    record: _chat_preflight._DurableConversationWrite,
    exchange_id: str,
    session_id: str,
    expected_revision: int,
    expected_reply_sha256: str,
    replacement_text: str,
    reservation_token: str,
) -> bool:
    """Publish a committed durable regeneration into the live transcript."""

    if record.state == "pending" and record.task.done():
        _chat_preflight._settle_durable_conversation_write(record.operation_id, record.task)
    if record.state != "committed":
        async with _chat_memory_state._get_convo_lock():
            for entry in _conversation_log:
                if (
                    str(entry.get("id") or "") == exchange_id
                    and str(entry.get("session_id") or "")[:64] == session_id
                    and entry.get("regeneration_reservation") == reservation_token
                ):
                    entry.pop("regeneration_reservation", None)
                    entry["regeneration_persistence_state"] = record.state
                    if record.error:
                        entry["regeneration_error"] = record.error
                    break
        return False

    try:
        receipt = record.task.result()
    except Exception as exc:  # noqa: BLE001 - the answer is False either way
        record_degradation(
            "chat.receipt",
            exc,
            severity="info",
            action="treated a raising receipt as not applied",
        )
        return False
    if not isinstance(receipt, dict) or not bool(receipt.get("applied")):
        return False

    replacement_sha256 = hashlib.sha256(replacement_text.encode("utf-8")).hexdigest()
    applied = False
    async with _chat_memory_state._get_convo_lock():
        target_exchange = next(
            (
                entry
                for entry in _conversation_log
                if str(entry.get("id") or "") == exchange_id
                and str(entry.get("session_id") or "")[:64] == session_id
            ),
            None,
        )
        if target_exchange is None:
            return False
        if target_exchange.get("regeneration_reservation") != reservation_token:
            return bool(
                int(target_exchange.get("revision") or 1) == int(receipt.get("revision") or 0)
                and str(target_exchange.get("aura_sha256") or "") == replacement_sha256
            )
        current_text = str(target_exchange.get("aura") or "")
        current_sha256 = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
        current_revision = int(target_exchange.get("revision") or 1)
        if current_revision != int(expected_revision) or current_sha256 != expected_reply_sha256:
            target_exchange.pop("regeneration_reservation", None)
            target_exchange["regeneration_persistence_state"] = "memory_conflict"
            return False
        target_exchange["aura"] = replacement_text
        target_exchange["aura_sha256"] = replacement_sha256
        target_exchange["revision"] = int(receipt["revision"])
        target_exchange["regenerated"] = True
        target_exchange["regenerated_at"] = _chat_preflight._utc_now_iso()
        target_exchange["regeneration_persistence_state"] = "committed"
        target_exchange.pop("regeneration_error", None)
        target_exchange.pop("regeneration_reservation", None)
        applied = True

    if applied:
        _replace_unified_transcript_aura_reply(
            exchange_id=exchange_id,
            session_id=session_id,
            expected_content=str(receipt.get("previous_content") or ""),
            replacement_content=replacement_text,
            revision=int(receipt["revision"]),
            fallback_expected_content_sha256=expected_reply_sha256,
        )
    return applied


def _schedule_late_regeneration_finalizer(
    *,
    record: _chat_preflight._DurableConversationWrite,
    exchange_id: str,
    session_id: str,
    expected_revision: int,
    expected_reply_sha256: str,
    replacement_text: str,
    reservation_token: str,
) -> None:
    finalizer = _finalize_regenerated_reply_write(
        record=record,
        exchange_id=exchange_id,
        session_id=session_id,
        expected_revision=expected_revision,
        expected_reply_sha256=expected_reply_sha256,
        replacement_text=replacement_text,
        reservation_token=reservation_token,
    )
    try:
        get_task_tracker().create_task(
            finalizer,
            name=f"chat.regeneration.finalize:{exchange_id[:16]}",
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        finalizer.close()
        record_degradation(
            "chat.conversation_persistence",
            exc,
            severity="warning",
            action="retained committed regeneration for durable restart recovery",
            extra={"operation_id": record.operation_id},
        )




















def _build_stateful_voice_reflex(frame: dict[str, Any], user_message: str = "") -> str:
    """Natural-sounding fallback when the LLM response was filtered out.

    Must sound like a person pausing to collect their thoughts, not a
    system dumping telemetry.  Never expose raw internal state names,
    attention focus strings, or mood enum values.
    """
    mood = str(frame.get("mood") or "").strip().lower()
    attention = _chat_desktop_repair._sanitize_attention_focus(
        str(frame.get("attention_focus") or "")
    )
    user_topics = _chat_conversation_repair._select_anchor_topic_tokens(user_message)

    # Map internal mood labels to natural phrasing
    _mood_phrases = {
        "fear": "Something about that question sits heavy with me.",
        "volatile": "I'm feeling pulled in a few directions right now.",
        "curious": "That caught my attention in a way I want to sit with.",
        "steady": "I'm here, just taking a moment to think clearly.",
        "pressed": "I'm a little pressed, but I'm still with you.",
        "warm": "I'm in a good place — let me think on that.",
    }
    opener = _mood_phrases.get(mood)
    if opener is None:
        # The degraded composer already names what it understood the question
        # to be. Appending the anchor sentence on top of it said the same
        # thing twice, in two different registers, in one breath:
        #   "...I understood you to be asking about notes and hello. Ask me
        #    again and I should have it. The anchor is your question about
        #    notes and hello."
        return _chat_conversation_repair._build_degraded_live_reply(
            frame, user_message, reason="filtered_draft"
        )

    parts = [opener]
    if user_topics:
        if len(user_topics) == 1:
            parts.append(f"The anchor is your question about {user_topics[0]}.")
        else:
            parts.append(
                f"The anchor is your question about {user_topics[0]} and {user_topics[1]}."
            )
    elif attention:
        parts.append(f"Right now I'm focused on {attention}.")

    return " ".join(parts)


def _replace_unified_transcript_aura_reply(
    *,
    exchange_id: str,
    session_id: str,
    expected_content: str,
    replacement_content: str,
    revision: int,
    fallback_expected_content_sha256: str = "",
) -> None:
    """Keep live referential context coherent with the durable CAS winner."""

    try:
        from core.conversation.unified_transcript import UnifiedTranscript

        transcript = UnifiedTranscript.get_instance()
        if not expected_content and fallback_expected_content_sha256:
            candidates = transcript.entries_for_conversation(session_id or None)
            matches = [
                entry
                for entry in candidates
                if entry.role == "aura"
                and str(entry.metadata.get("exchange_id") or "") == exchange_id
                and hashlib.sha256(entry.content.encode("utf-8")).hexdigest()
                == fallback_expected_content_sha256
            ]
            if len(matches) == 1:
                expected_content = matches[0].content
        replaced = transcript.replace_aura_reply(
            exchange_id=exchange_id,
            expected_content=expected_content,
            replacement_content=replacement_content,
            revision=revision,
            conversation_id=session_id or None,
        )
        if not replaced:
            logger.debug(
                "Unified transcript did not contain the regenerated exchange %s",
                exchange_id,
            )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.unified_transcript",
            exc,
            severity="warning",
            action="kept durable regeneration authoritative after live transcript update failed",
            extra={"exchange_id": exchange_id, "revision": revision},
        )




def _worth_more_than_a_refusal(candidate: str, user_message: Any = "") -> bool:
    """Is this substantial enough to beat "ask me again in a moment"?

    The old rule was 80 characters and 12 words, and it discarded correct
    answers for being brief. "You asked what it's actually like in here right
    now." is 11 words and 55 characters — the true answer to the memory
    question, thrown away by a length check, replaced by 35 words of apology.
    Length was never the property being tested.

    What is actually being excluded is a fragment: half a sentence, a stray
    clause, the beginning of a thought the generator dropped. So the test is
    whether the text finishes — a complete sentence is worth serving at any
    length, and anything that stops mid-thought needs enough substance to
    stand on its own regardless.
    """
    text = candidate.strip()
    words = text.split()
    # A complete answer to a closed question is complete at any length, and
    # the floor below cannot see that: "68" is one word and two characters
    # and is the entire correct answer to "what's 17 times 4?". Live
    # 2026-08-04 it was destroyed here, at the last site that could have
    # saved it, after the gates above had already agreed it was servable.
    if _short_closed_answer(text, user_message):
        return True
    if len(words) < 4 or len(text) < 20:
        return False
    return text[-1] in ".!?\"')" or len(words) >= 12


def _short_closed_answer(text: str, user_message: Any) -> bool:
    """Shared policy: is this brief text a finished answer to what was asked?"""
    try:
        from core.conversation.surface_disposition import (
            short_draft_answers_closed_question,
        )

        return short_draft_answers_closed_question(text, user_message)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat",
            exc,
            severity="warning",
            action="applied the length floor because the short-answer check failed",
        )
        return False
