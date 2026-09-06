"""The small questions a turn asks about itself, and the marks it leaves.

Lifted out of `interface/routes/chat.py`. Which lane owns this turn, whether
the cortex is still warming, what this request is asking for, whether a
generation has already been served. Individually each is a few lines; together
they were most of what made one file twenty-four thousand lines long.
"""
from __future__ import annotations

from core.container import ServiceContainer
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
from core.runtime.shutdown_coordinator import record_shutdown_admission_event
from core.utils.intent_normalization import normalize_memory_intent_text
from core.utils.task_tracker import get_task_tracker
from datetime import datetime
from fastapi import Request
from fastapi.responses import JSONResponse
from interface.auth import relational_principal_id_for_request
from interface.routes import chat_desktop_repair as _chat_desktop_repair  # noqa: E402
from interface.routes import chat_memory_state as _chat_memory_state  # noqa: E402
from interface.routes import chat_preflight as _chat_preflight  # noqa: E402
from interface.routes.chat_common import _CHAT_RECOVERABLE_ERRORS, _CHAT_SESSION_ID_MAX_CHARS, _ORGAN_ABSENCE_STREAKS, _ORGAN_INERT_STREAKS, _TOPIC_STOPWORDS, _conversation_log, logger
from interface.routes.chat_self_reply import _is_identity_challenge_request
from pathlib import Path
from typing import Any
import asyncio
import dataclasses
import hashlib
import inspect
import json
import os
import re
import time
from interface.routes import chat_conversation_repair as _chat_conversation_repair  # noqa: E402


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
    return not any(marker in reply for marker in ("context", "thread", "recent", "last"))


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


def _host_condition() -> dict[str, Any]:
    """The machine's own load, as the runtime already measures it."""

    try:
        from core.introspection.self_evidence import resolve_self_health

        readings = {
            reading.channel: reading for reading in resolve_self_health().readings
        }
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.host_condition",
            exc,
            severity="debug",
            action="left the machine's load out of her state snapshot",
        )
        return {}
    condition: dict[str, Any] = {}
    load = readings.get("host_load")
    if load is not None and load.present:
        values = dict(load.value or {})
        for name in ("processor_percent", "memory_percent"):
            try:
                condition[name] = round(float(values.get(name)), 1)
            except (TypeError, ValueError):
                continue
    thermal = readings.get("host_thermal")
    if thermal is not None and thermal.present:
        try:
            condition["thermal_pressure"] = round(float(thermal.value), 2)
        except (TypeError, ValueError):
            pass
    # Absent rather than zero: a load reported as 0% because nothing answered
    # is worse than one that is missing, and she can say she does not know.
    return condition


def _canonical_runtime_model_label(lane: dict[str, Any] | None) -> str:
    lane = dict(lane or {})
    candidates = [
        str(lane.get("desired_model") or ""),
        str(lane.get("last_user_generation_endpoint") or ""),
        str(lane.get("foreground_endpoint") or ""),
        str(lane.get("desired_endpoint") or ""),
        str(lane.get("model_path") or ""),
    ]
    joined = " ".join(candidates).lower()
    # Which LANE this is stays a name match -- the lane names are the words in
    # these fields. What the lane is CALLED comes from the registry, which
    # holds the signed resident descriptor. Reading the size out of these
    # strings could not match a 27B at all, and the "cortex" token was wired
    # to a literal "Cortex (32B)", so the label named a checkpoint that had
    # been replaced while its descriptor sat one call away.
    try:
        from core.brain.llm.model_registry import (
            BRAINSTEM_ENDPOINT,
            DEEP_ENDPOINT,
            FALLBACK_ENDPOINT,
            PRIMARY_ENDPOINT,
            lane_display_label,
        )
    except ImportError:
        lane_display_label = None
    if lane_display_label is not None:
        if "solver" in joined:
            return lane_display_label(DEEP_ENDPOINT)
        if "brainstem" in joined:
            return lane_display_label(BRAINSTEM_ENDPOINT)
        if "reflex" in joined:
            return lane_display_label(FALLBACK_ENDPOINT)
        if "cortex" in joined:
            return lane_display_label(PRIMARY_ENDPOINT)
    if lane.get("desired_model") or lane.get("foreground_endpoint"):
        return str(lane.get("desired_model") or lane.get("foreground_endpoint"))
    if lane_display_label is not None:
        return lane_display_label(PRIMARY_ENDPOINT)
    return "Cortex"


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


def _conversation_lane_is_standby(lane: dict[str, Any] | None) -> bool:
    lane = dict(lane or {})
    state = str(lane.get("state", "") or "").strip().lower()
    return (
        not bool(lane.get("conversation_ready", False))
        and state in {"cold", "closed", ""}
        and not bool(lane.get("warmup_attempted", False))
        and not bool(lane.get("warmup_in_flight", False))
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


def _mark_conversation_lane_timeout(reason: str = "foreground_timeout") -> dict[str, Any]:
    from core.brain.llm.model_registry import PRIMARY_ENDPOINT

    # Activate recovery cooldown so rapid follow-up messages are fast-rejected
    # instead of piling into the inference pipeline.
    _enter_recovery_cooldown()
    _force_clear_mlx_foreground_owner(reason=reason, min_age_s=45.0)

    try:
        gate = ServiceContainer.get("inference_gate", default=None)
        if gate and hasattr(gate, "note_foreground_timeout"):
            gate.note_foreground_timeout(reason)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Conversation lane timeout mark failed: %s", exc)

    lane = _chat_preflight._collect_conversation_lane_status()
    lane["state"] = "recovering"
    lane["conversation_ready"] = False
    lane["last_failure_reason"] = reason
    if not lane.get("foreground_endpoint"):
        lane["foreground_endpoint"] = PRIMARY_ENDPOINT
    return lane


def _force_clear_mlx_foreground_owner(
    *,
    reason: str,
    min_age_s: float = 45.0,
) -> dict[str, Any]:
    try:
        from core.brain.llm.mlx_client import force_clear_foreground_owner

        result = force_clear_foreground_owner(
            reason=reason,
            min_age_s=min_age_s,
        )
        if result.get("cleared"):
            logger.warning(
                "Cleared stale MLX foreground owner during chat recovery: %s",
                result,
            )
        return result
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("MLX foreground owner recovery hook unavailable: %s", exc)
        return {
            "cleared": False,
            "reason": reason,
            "holder": None,
            "age_s": 0.0,
            "detail": "unavailable",
        }


def _mark_conversation_lane_state(reason: str, *, state: str) -> dict[str, Any]:
    from core.brain.llm.model_registry import PRIMARY_ENDPOINT

    lane = _chat_preflight._collect_conversation_lane_status()
    lane["state"] = state
    lane["conversation_ready"] = False
    lane["last_failure_reason"] = reason
    lane["warmup_attempted"] = True
    if not lane.get("foreground_endpoint"):
        lane["foreground_endpoint"] = PRIMARY_ENDPOINT
    return lane


def _status_represents_memory_state_result(status: str | None) -> bool:
    return str(status or "").strip() in {
        "owner_identity_recall",
        "session_memory_pin",
        "session_memory_pin_transient",
        "session_memory_recall",
        "session_memory_context_recall",
        "conversation_recall",
    }


def _turn_count_ordinal(count: int) -> str:
    value = max(0, int(count))
    suffix = "th"
    if not 10 <= value % 100 <= 20:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _cortex_is_cold_loading(lane: object) -> bool:
    """True while the cortex is doing its one-time load for this process.

    The admission budget is the turn's remaining time minus a reserve, and the
    reserve normally includes 60s held back for producing an answer. During a
    COLD load that subtraction is backwards: it deducts time for answering from
    the time needed to become able to answer at all.

    LIVE 2026-08-17, measured four times: the first message after launch died
    at 15-16s every time. The foreground timeout is ~80s and the reserve is
    64s, so admission got ~16s — while a 32B cold load needs well over a
    minute. The turn could not have succeeded at any point during boot, and the
    person got "the live answer lane could not finish preparing", which reads
    as a fault rather than as a model still loading.

    A cold load is a one-time wait a person who just launched an app expects to
    pay. There is no answer to reserve for until the weights are up, so during
    that window only the response reserve is held back.
    """

    if not isinstance(lane, dict):
        return False
    try:
        if bool(lane.get("conversation_ready")):
            return False
        if bool(lane.get("has_generated_successfully")):
            return False  # served once already: this is a recovery, not a cold load
        return float(lane.get("last_ready_at") or 0.0) <= 0.0
    except (AttributeError, TypeError, ValueError):
        return False


def _with_the_same_readings(system_prompt: str, readings: list[str]) -> str:
    """The identity, with whatever was read put in front of the model."""
    if not readings:
        return system_prompt
    return "\n\n".join([system_prompt, *readings] if system_prompt else readings)


def _lane_reply_confidence(served: object, default: str) -> str:
    """How much to trust a reply the degraded path produced.

    A deterministic result is the most reliable answer the runtime can give —
    no model, no sampling, no lane. Live 2026-08-19 the exact product 50,420,273
    was served and badged "No answer", which is the opposite of true and
    exactly the kind of thing a person checking her work would catch.
    """
    body = str(served or "").strip()
    known = _known_answer_for_this_turn()
    return "computed" if known and body == known else str(default)


def _lane_status_message_body(
    lane: dict[str, Any],
    *,
    timed_out: bool = False,
    status_override: str = "",
) -> str:
    """Generate a personality-infused status message instead of a robotic error.

    [STABILITY v50] These messages now sound like Aura experiencing a
    momentary lapse rather than a system displaying error codes. Uses
    the live expression frame when available so Aura's current mood
    colours even her recovery messages.
    """
    state = str(lane.get("state", "warming") or "warming")
    failure_reason = str(lane.get("last_failure_reason", "") or "")
    status_override = str(status_override or "")

    # Hard infrastructure failures — keep these explicit for debugging
    if failure_reason.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:")):
        model_label = _canonical_runtime_model_label(lane)
        return (
            f"The local {model_label} runtime could not start cleanly. I should not "
            "fake a normal answer; the launcher logs have the failure details."
        )
    if (
        "memory_pressure_refused_worker_spawn" in failure_reason
        or "projected_process_tree_rss" in failure_reason
        or "model_load_headroom" in failure_reason
    ):
        return (
            "The local model lane was blocked by the unified-memory guard before loading. "
            "I am protecting the desktop from an unsafe RAM spike instead of pretending Cortex is merely warming."
        )

    # Build a mood-aware prefix for softer messages
    #
    # Every line below is written to continue one: "Mmm, that answer took too
    # long". With no mood to prefix, the sentence began lowercase and reached
    # the person as a fragment — "that answer took too long to finish
    # cleanly." LIVE 2026-08-26.
    _mood_prefix = ""
    try:
        _pe = ServiceContainer.peek("personality_engine", default=None)
        if _pe and hasattr(_pe, "get_emotional_context_for_response"):
            _emo = _pe.get_emotional_context_for_response() or {}
            _mood = str(_emo.get("mood", "") or "").lower()
            if _mood in {"frustrated", "irritated", "tense"}:
                _mood_prefix = "Ugh, "
            elif _mood in {"tired", "drowsy", "low"}:
                _mood_prefix = "Mmm, "
            elif _mood in {"curious", "playful", "amused"}:
                _mood_prefix = "Hmm — "
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Mood prefix unavailable for degraded reply: %s", exc)

    if status_override == "warming_timeout":
        return (
            _with_mood(_mood_prefix, "the live answer lane exceeded its warm-up budget before "
            "a reasoning turn began. I did not misclassify that boot delay as a failed answer.")
        )
    if status_override == "warming_failed":
        return (
            _with_mood(_mood_prefix, "the live answer lane could not finish preparing before "
            "a reasoning turn began. I recorded the readiness failure separately from Aura's answer quality.")
        )
    if timed_out:
        return _with_mood(_mood_prefix, "that answer took too long to finish cleanly. I logged the timeout and preserved the turn context.")
    if _conversation_lane_is_standby(lane):
        return _with_mood(_mood_prefix, "the local answer path is still preparing. I logged the cold lane instead of claiming Aura is ready.")
    if state == "recovering":
        return _with_mood(_mood_prefix, "the answer lane is recovering from the previous failure. I logged the degraded state instead of emitting a fragment.")
    if state == "failed":
        return _with_mood(_mood_prefix, "the local answer path failed before producing a coherent reply. I'm restarting it instead of pretending that was a real answer.")
    return _with_mood(_mood_prefix, "the answer path is not ready yet; the readiness state is recorded on the live lane.")


def _enter_recovery_cooldown() -> None:
    global _last_recovery_cooldown_at
    _last_recovery_cooldown_at = time.monotonic()


def _conversation_lane_blocks_fallback(lane: dict[str, Any]) -> bool:
    """Avoid hiding a hard local backend failure behind a generic fallback reply."""
    state = str(lane.get("state", "") or "").strip().lower()
    failure_reason = str(lane.get("last_failure_reason", "") or "")
    if state != "failed":
        return False
    return failure_reason.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:"))


def _conversation_lane_needs_instant_social_contract(lane: dict[str, Any]) -> bool:
    """Return whether a low-risk presence turn should avoid cold-warming Cortex."""

    state = str(lane.get("state", "") or "").strip().lower()
    if state in {"cold", "warming", "recovering", "failed", "unavailable"}:
        return True
    if lane.get("conversation_ready") is False:
        return True
    blockers = lane.get("readiness_blockers") or ()
    if isinstance(blockers, (list, tuple, set)) and blockers:
        return True
    if not str(lane.get("foreground_endpoint", "") or "").strip() and state not in {
        "ready",
        "healthy",
    }:
        return True
    return False


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
    return any(marker in lowered for marker in markers)


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
    return not any(marker in text for marker in substantive_markers)


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
    return any(term in lowered for term in memory_terms) and any(
        term in lowered for term in evidence_terms
    )


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


def _export_json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _early_chat_json_response(
    payload: dict[str, Any],
    *,
    status_code: int,
) -> JSONResponse:
    """Return an early payload; the paired boundary settles actual delivery."""

    return JSONResponse(payload, status_code=status_code)


def _pre_gate_unavailable_response(gate: str) -> JSONResponse:
    gate_label = {
        "defensive_runtime": "security",
        "conscience": "conscience",
    }.get(str(gate), "required")
    response = (
        f"I could not safely process that turn because my {gate_label} preflight "
        "is unavailable. I did not send the request into cognition or act on it. "
        "Please retry after the runtime recovers."
    )
    return _early_chat_json_response(
        {
            "response": response,
            "message": response,
            "error": "chat_preflight_unavailable",
            "status": "chat_preflight_unavailable",
            "gate": str(gate),
            "retryable": True,
            "processed": False,
            "response_confidence": "fail_closed",
        },
        status_code=503,
    )


def _mark_http_turn_served(outcome: Any, response: Any) -> None:
    """Record what the person actually received, from the response itself."""

    if outcome is None:
        return
    try:
        payload = getattr(response, "body", None)
        served = ""
        if payload is not None:
            data = json.loads(payload)
            if isinstance(data, dict):
                served = str(data.get("response") or "")
                live_contract = data.get("live_turn_contract")
                if isinstance(live_contract, dict):
                    outcome.record_receipt(
                        "served_response_authority",
                        {
                            # Use the turn ledger's redaction-safe evidence
                            # vocabulary. Raw HTTP contract names containing
                            # "auth" are intentionally redacted as possible
                            # credentials by record_receipt().
                            "evidence_kind": live_contract.get(
                                "response_authority_kind"
                            ),
                            "authority_verified": live_contract.get(
                                "response_authority_proven"
                            )
                            is True,
                            "evidence_reason": live_contract.get(
                                "response_authority_reason"
                            ),
                            "delivery_verified": live_contract.get(
                                "answer_delivery_proven"
                            )
                            is True,
                        },
                    )
        if served.strip():
            outcome.mark_served(served)
        else:
            from core.runtime.turn_outcome import UserVisibleState

            outcome.mark_served("", state=UserVisibleState.NOTHING_SERVED)
    except (_CHAT_RECOVERABLE_ERRORS, json.JSONDecodeError) as exc:
        record_degradation("chat.turn_outcome", exc, severity="info")


def _runtime_shutdown_response(
    checkpoint: str,
    *,
    slot_acquired: bool,
    error: BaseException | None = None,
) -> JSONResponse:
    """The 503 a turn returns when the runtime is going down under it.

    Module scope with `slot_acquired` passed in, rather than nested and
    closing over `foreground_slot_acquired`. That was the only thing it took
    from the turn, and one boolean is a cheaper contract than a closure.
    """
    outcome = "reaped" if slot_acquired else "suppressed"
    record_shutdown_admission_event(
        "chat.foreground_turn",
        resource_kind="foreground_turn",
        outcome=outcome,
        detail=f"checkpoint={checkpoint}",
    )
    logger.info(
        "Foreground chat stopped by runtime shutdown (checkpoint=%s error_type=%s).",
        checkpoint,
        type(error).__name__ if error is not None else "none",
    )
    return JSONResponse(
        {
            "response": (
                "The runtime is shutting down, so I stopped this turn cleanly "
                "before starting more cognitive work."
            ),
            "status": "runtime_shutdown",
            "checkpoint": checkpoint,
            "response_confidence": "not_generated",
        },
        status_code=503,
        headers={"Retry-After": "1"},
    )


def _launcher_desktop_runtime_active() -> bool:
    return any(
        str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}
        for name in ("AURA_LAUNCHED_FROM_APP", "AURA_EXTERNAL_GUI_OWNER", "AURA_GUI_PROXY")
    )


def _request_from_local_desktop_client(request: Request) -> bool:
    client = getattr(request, "client", None)
    host = str(getattr(client, "host", "") or "").strip().lower()
    if not host:
        return True
    return host in {"127.0.0.1", "::1", "localhost", "test", "local"}


def _normalize_response_body(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _with_mood(prefix: str, sentence: str) -> str:
    """A reply written to follow a mood prefix, said with or without one.

    Each of these lines continues something: "Mmm, that answer took too long".
    With no mood to continue from, the sentence has to start for itself —
    otherwise it reaches the person as a fragment, which is how "that answer
    took too long to finish cleanly." was read out lowercase. LIVE 2026-08-26.
    """
    said = str(sentence or "")
    lead = str(prefix or "")
    if lead:
        return f"{lead}{said}"
    return said[:1].upper() + said[1:] if said else said


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


def _known_answer_for_this_turn() -> str:
    """What the runtime can answer without the model, or empty.

    A lane that is warming, timed out or recovering says nothing about whether
    the answer is knowable. "what is 7919 * 6367?" has one exact answer, held
    by a deterministic form that needs no generation at all, and it was
    replaced by a sentence about the lane.
    """
    try:
        from core.conversation.arithmetic_check import requested_arithmetic_result
        from core.conversation.session_scope import current_user_question

        question = current_user_question()
        if not question:
            return ""

        # A seating problem the enumeration settles.
        #
        # LIVE, 2026-08-21: the solver produced the answer at 22:42:45 —
        # "took 1 reading(s): the seating, worked out" — and the turn then
        # spent another 105 seconds generating text that was replaced by that
        # same answer at the end. An exact answer is not an improvement on a
        # generated one, it is a reason not to generate.
        from core.reasoning.positional_constraints import (
            answer_positional_problem,
            describe_positional_answer,
        )

        seating = describe_positional_answer(answer_positional_problem(question))
        if seating:
            return seating

        # A game the preflight already enumerated. Worked out before anything
        # was generated, so there is nothing here to improve on.
        #
        # A repository diagnosis is different: it is an observation, and what
        # was asked for was an explanation of it. That one is composed with
        # the reply instead of replacing it, further down.
        from core.conversation.session_scope import solved_answers

        solved = solved_answers()
        settled = solved.get("finite_game", "")
        if settled:
            return settled
        # Anything else the runtime worked out this turn.
        #
        # This function is asked twice: before generating, where only a
        # preflight result exists, and again at the point of giving up, where
        # the comment beside the apology says to ask whether the runtime
        # already HOLDS the answer. A diagnosis is produced by a tool during
        # generation, so it can never skip generation — it can only stop the
        # turn ending in an apology while the finding sits in hand.
        for value in reversed(list(solved.values())):
            if value.strip():
                return value

        value = requested_arithmetic_result(question)
        if value is None:
            return ""
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        shown = f"{value:,}" if isinstance(value, int) else f"{value:,}"
        return f"{shown}."
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.known_answer",
            exc,
            severity="debug",
            action="served the lane status without checking for a computed answer",
            enforce_failure_policy=False,
        )
        return ""


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
