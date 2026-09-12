"""Getting into the one lane that can answer, and what to do while waiting.

Only one turn holds the model at a time, so everything here is about that
admission: how long this lane may wait, whether the wait is a warm-up that was
deliberately deferred rather than a fault, and what she can say from grounded
introspection while it is still coming up. The fallback ladder is here too — it
is what answers when the lane never opens, and its rungs are ordered by how much
of her is actually behind each one.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import describe_error, record_degradation
from core.runtime.structured_input import (
    analyze_prompt_shape,
    answer_surface_token_floor,
)
from core.utils.task_tracker import get_task_tracker
from interface.routes import chat_conversation_repair as _chat_conversation_repair  # noqa: E402
from interface.routes import chat_memory_state as _chat_memory_state  # noqa: E402
from interface.routes import chat_protected_prompt as _chat_protected_prompt  # noqa: E402
from interface.routes.chat_common import (  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402
    _CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,  # noqa: F401
    _CHAT_DELIVERY_IDEMPOTENCY_KEY,  # noqa: F401
    _CHAT_PENDING_DELIVERY_CLAIM,  # noqa: F401
    _CHAT_RECOVERABLE_ERRORS,  # noqa: F401
    _CHAT_REQUEST_PRINCIPAL,  # noqa: F401
    _CHAT_REQUEST_SURFACE,  # noqa: F401
    _CHAT_SESSION_ID_MAX_CHARS,  # noqa: F401
    _EXPLICIT_NON_EXECUTION_RE,  # noqa: F401
    _INCOMPLETE_TAIL_WORDS,  # noqa: F401
    _INTERNAL_STATE_PATTERNS,  # noqa: F401
    _INTERNAL_SURFACE_CONTEXT,  # noqa: F401
    _LOCAL_CHOICE_REFERENCE_RE,  # noqa: F401
    _MAX_CONVERSATION_LOG_EXCHANGES,  # noqa: F401
    _MAX_USER_SURFACE_CONTINUATIONS,  # noqa: F401
    _ORGAN_ABSENCE_STREAKS,  # noqa: F401
    _ORGAN_INERT_STREAKS,  # noqa: F401
    _PROMPT_ARTIFACT_PATTERNS,  # noqa: F401
    _SEARCH_SKILL_NAMES,  # noqa: F401
    _TOPIC_STOPWORDS,  # noqa: F401
    _UNSET,  # noqa: F401
    MAX_CHAT_MESSAGE_BYTES,  # noqa: F401
    _conversation_log,  # noqa: F401
    _locks,  # noqa: F401
    logger,  # noqa: F401
)
from interface.routes.chat_self_reply import (  # noqa: E402,F401
    _build_architecture_self_reflex,
    _build_self_condition_evidence,
    _build_self_diagnostic_reply,
    _build_subjective_self_reflex,
    _canonical_memory_state_grounding_reply,
    _classify_self_condition_contract,
    _fallback_ladder_identity,
    _humanize_recent_self_process_concern,
    _humanize_self_process_dimensions,
    _is_identity_challenge_request,
    _is_self_claim_boundary_question,
    _same_live_self_reflection_prompt_class,
)
from interface.routes.chat_turn_evidence import (  # noqa: E402,F401
    _benchmark_prompt_requests_fenced_artifact,
    _build_explicit_local_file_artifact,
    _canonical_memory_state_evidence_from_tuple,
    _canonical_memory_state_evidence_missing_from_reply,
    _collect_named_url_evidence,
    _collect_recent_traceability_event_sync,
    _context_challenge_repair_has_evidence,
    _correct_unevidenced_action_claims,
    _emit_chat_output_receipt,
    _evidence_came_from_the_network,
    _extract_canonical_memory_state_evidence_block,
    _prime_requested_output_contract_trace,
    _recent_action_receipts,
    _record_desktop_evidence_on_the_trace,
    _resolve_prior_answer_provenance,
    _save_requested_artifact,
    _serve_built_artifact,
    _worker_receipt_transaction_id,
)

from .chat_lane_bookkeeping import (  # noqa: E402
    _cortex_is_cold_loading,
    _gather_recent_user_messages_for_relevance,
    _generation_metadata_consumed_foreground_owner,
    _mark_conversation_lane_state,
    _protected_foreground_generation_block_reason,
    _with_the_same_readings,
)
from .chat_reply_repair import (
    _ends_where_it_meant_to,
    _evaluate_reply_topicality,
    _is_actionably_stale_response,
    _is_same_answer_different_prompt,
    _looks_semantically_glitched,
    _original_reply_is_safe_to_surface,  # noqa: F401
    _repair_final_degraded_reply,  # noqa: F401
    _repair_missing_followup_delta,  # noqa: F401
    )
from .chat_reply_shaping import (  # noqa: E402
    _remove_self_denials_the_record_refutes,
    _strip_scaffolding_tags,
)
from .chat_served_answers import (  # noqa: E402
    _readings_for,
)

# A lane whose own failure reason says warmup was DEFERRED (backoff after
# repeated stuck loads, admission refusal) is not warming toward ready — the
# runtime has deliberately decided the cortex will not load right now. Turns
# must not spend the cold-boot budget waiting for a model that is provably
# not coming; the warm fallback needs that time to actually answer.
_DEFERRED_WARMUP_REASON_MARKERS = (
    "warmup_backoff",
    "warmup_deferred",
    "warmup_timeout",
    "deferred_memory_pressure",
)
_BOOT_TRANSITION_STALL_S = 45.0
_FORCE_PRIMARY_PHRASES = (
    "don't go to your 72",
    "dont go to your 72",
    "don't use 72",
    "dont use 72",
    "no 72b",
    "no 72-b",
    "stay on 32",
    "stay on the 32",
    "stay primary",
    "stay on primary",
    "32b only",
    "primary only",
    "skip the solver",
    "don't escalate",
    "dont escalate",
)


def _lane_warmup_is_deliberately_deferred(lane: dict[str, Any] | None) -> bool:
    """True when the lane is held off warmup rather than progressing toward it."""
    reason = str((lane or {}).get("last_failure_reason", "") or "").lower()
    if not reason:
        return False
    return any(marker in reason for marker in _DEFERRED_WARMUP_REASON_MARKERS)


def _foreground_timeout_for_lane(
    lane: dict[str, Any] | None,
    user_message: str = "",
) -> float:
    """Foreground timeout for the chat request.

    This is a wall-clock UI SLA, not a model-load wishlist. Cold 32B warmup
    gets more room than a ready lane, but the desktop route must still fail
    closed and recover rather than holding the UI indefinitely under memory
    pressure or a wedged foreground owner.
    """
    # Imported here rather than at module level: the module these
    # came from imports this one. A call-time import also still
    # sees a test's patch of the original.
    from .chat import (
        _DEFERRED_CORTEX_TURN_TIMEOUT_S,
        _DESKTOP_COGNITIVE_MAX_TURN_TIMEOUT_S,
        _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S,
        _DESKTOP_COGNITIVE_TURN_TIMEOUT_S,
    )

    lane = dict(lane or {})
    state = str(lane.get("state", "") or "").lower()
    ready_timeout = max(
        30.0,
        min(
            _DESKTOP_COGNITIVE_MAX_TURN_TIMEOUT_S,
            _DESKTOP_COGNITIVE_TURN_TIMEOUT_S + _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S,
        ),
    )
    if bool(lane.get("conversation_ready", False)):
        shape = analyze_prompt_shape(user_message)
        if bool(
            getattr(shape, "prefers_extended_answer", False)
            or getattr(shape, "requires_single_reply_coverage", False)
            or int(getattr(shape, "question_parts", 0) or 0) >= 2
        ):
            try:
                from core.brain.llm.measured_admission import (
                    recommended_completion_tokens,
                    recommended_foreground_deadline,
                )
                from core.brain.llm.model_registry import runtime_model_measurement_key
                from core.runtime.structured_input import answer_surface_planning_tokens

                prompt_tokens = max(2048, 1800 + len(str(user_message or "")) // 4)
                answer_capacity = answer_surface_token_floor(user_message)
                answer_tokens, _length_confidence, _length_samples = (
                    recommended_completion_tokens(
                        model=runtime_model_measurement_key(),
                        prompt_tokens=prompt_tokens,
                        maximum_tokens=answer_capacity,
                        prior_tokens=answer_surface_planning_tokens(user_message),
                    )
                )
                deadline, _confidence, _samples = recommended_foreground_deadline(
                    model=runtime_model_measurement_key(),
                    prompt_tokens=prompt_tokens,
                    decode_tokens=answer_tokens,
                    minimum_seconds=ready_timeout,
                    maximum_seconds=(
                        _DESKTOP_COGNITIVE_MAX_TURN_TIMEOUT_S
                        + _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S
                    ),
                )
                return deadline
            except (ArithmeticError, ImportError, TypeError, ValueError) as exc:
                record_degradation(
                    "chat.measured_foreground_deadline",
                    exc,
                    severity="debug",
                    action="used the conservative extended foreground ceiling",
                    enforce_failure_policy=False,
                )
                return max(
                    ready_timeout,
                    _DESKTOP_COGNITIVE_MAX_TURN_TIMEOUT_S
                    + _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S,
                )
        return ready_timeout
    if state in {"warming", "recovering", "cold", "spawning", "handshaking"}:
        if _lane_warmup_is_deliberately_deferred(lane):
            # Deferred ≠ warming. Granting the cold-boot budget here spent
            # the whole turn on a cortex the runtime had already decided not
            # to load, leaving the Brainstem seconds and the Reflex
            # milliseconds — 32 turns produced NO reply at all while a warm
            # 1.5B sat idle (2026-07-18 soak). Give the ladder the time.
            return _DEFERRED_CORTEX_TURN_TIMEOUT_S
        return 210.0
    return ready_timeout


def _boot_is_still_in_progress(phases: Any) -> bool:
    """True only while a boot is demonstrably running and moving.

    "Not ready" is not the same as "booting". A fresh BootPhases that has
    never transitioned — every unit test, and any process where the boot
    machinery never ran — reports STARTING forever, and waiting on it would
    hang the turn for the whole budget. Boot is in progress only when it has
    actually made a transition, and made one recently.
    """
    if phases is None:
        return False
    try:
        if phases.ready():
            return False
        # NOT `last_change is None`. That field holds a human-readable string
        # ("organ: starting -> ready") which BootPhases only sets once some
        # organ has actually transitioned, so it is None during EARLY boot —
        # exactly the window where a turn most needs to wait.
        #
        # LIVE 2026-08-17: the first message typed after launch was answered
        # with "the live answer lane could not finish preparing", and the
        # "waiting up to Ns for boot" line was logged ZERO times, because this
        # predicate returned False before it ever reached the timestamp it
        # wanted. Ten seconds later the same message served normally.
        #
        # The intent — do not wait on machinery that never ran — is carried by
        # last_transition_at, which BootPhases initialises to started_at and
        # bumps on every transition. If nothing is running, it is stale, and
        # the staleness check below already declines to wait.
        last_transition = float(getattr(phases, "last_transition_at", 0.0) or 0.0)
    except _CHAT_RECOVERABLE_ERRORS:
        return False
    if last_transition <= 0.0:
        return False
    return (time.time() - last_transition) < _BOOT_TRANSITION_STALL_S


async def _fallback_conversation_messages(text: str) -> list[dict[str, str]]:
    """A smaller model inherits the turn's dialogue, not an empty session."""

    from core.conversation.delivered_history import (
        VISIBLE_CONVERSATION_EXCHANGES,
        delivered_exchange_messages,
    )
    from core.conversation.session_scope import current_conversation_session
    from core.conversation.turn_evidence_custody import (
        record_turn_transcript,
        turn_transcript,
    )

    admitted = turn_transcript()
    if admitted is not None:
        return list(admitted)
    session_id = current_conversation_session()
    if not session_id:
        return []
    # Cold-start fallback can precede the engine's history read. Use its same
    # principal-scoped durable reader and retain that snapshot for every retry.
    exchanges = await _chat_memory_state._recent_completed_conversation_exchanges(
        current_user_message=text,
        session_id=session_id,
        limit=VISIBLE_CONVERSATION_EXCHANGES,
        allow_cross_session=True,
    )
    record_turn_transcript(exchanges)
    snapshot = turn_transcript()
    return list(snapshot) if snapshot is not None else delivered_exchange_messages(exchanges)


async def _answer_from_fallback_ladder(
    user_message: object, *, reason: str, budget_s: float | None = None
) -> str:
    """Answer with the smaller resident model when the cortex cannot serve.

    Returns "" when the ladder cannot answer either, or when the question is
    one this model has no standing to answer, in which case the caller falls
    back to the honest lane message.

    The reply is marked as coming from the smaller model. Serving a 9B answer
    silently as though the 32B produced it would trade one honesty problem for
    a worse one, and the person is entitled to know which mind answered while
    the main one is still coming up.

    It does NOT answer questions about what she IS. Live 2026-08-19, asked
    what she had genuinely changed her mind about, the 9B replied that she has
    no continuous narrative, no personal beliefs and no capacity for revision
    over time — false of a runtime with a belief store, episodic memory, an
    ontogeny organ and a self-model, none of which that model can read. The
    disclosure line underneath says which mind answered; it does not retract
    the claim. Waiting is the honest answer there.
    """
    from .chat import (
        _FALLBACK_LADDER_TIMEOUT_S,
        router,
    )


    text = str(user_message or "").strip()
    if not text:
        return ""
    readings = await _readings_for(text)
    try:
        from core.runtime.self_state_intent import asks_about_her_own_nature

        if asks_about_her_own_nature(text) and not readings:
            # Declining is right when there is nothing to answer FROM. With a
            # reading in hand the smaller model is not being asked what it
            # believes about itself, it is being asked to say what the record
            # says — and the alternative is a wait message, which answers
            # nothing and is the thing this ladder exists to avoid.
            logger.info(
                "🪜 Fallback ladder declined a question about her own nature; "
                "the smaller model cannot read her self-model and no reading "
                "was available to stand in for it."
            )
            return ""
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.fallback_ladder",
            exc,
            severity="debug",
            action="let the ladder answer without the self-description guard",
            enforce_failure_policy=False,
        )
    try:
        from core.brain.llm_health_router import get_llm_router

        router = get_llm_router()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.fallback_ladder", exc)
        return ""
    if router is None or not hasattr(router, "think"):
        record_degradation(
            "chat.fallback_ladder",
            RuntimeError("no router available for the fallback ladder"),
            action="cortex unavailable and the ladder could not be reached",
        )
        return ""
    try:
        # Name the endpoint rather than asking for a tier. A tier request is
        # still a foreground request, and the selector skips background-only
        # tiers for those unless the caller named one — which is why asking
        # for "tertiary" came back with an empty chain, having considered
        # nothing at all.
        # Descend the ladder in order. The 9B Brainstem answers coherently;
        # the 1.5B Reflex is the last resort and shows it — asked "are you
        # there?" it replied "Yes, I'm sorry but I am not there." Reflex is
        # better than silence, but only after the 9B has been tried.
        from core.brain.llm.model_registry import (
            BRAINSTEM_ENDPOINT,
            FALLBACK_ENDPOINT,
        )

        identity = _fallback_ladder_identity()
        # The readings the main path takes. Grounding lived on one path and
        # the ladder was another, so a question the registry could answer
        # exactly reached a small model with nothing in front of it — LIVE
        # 2026-08-30, asked to prove she can grow the language she makes rules
        # out of, it said her representation language was "a fixed statistical
        # distribution of tokens learned from a static dataset", with the
        # register of tested claims sitting unread.
        identity = _with_the_same_readings(identity, readings)
        dialogue = await _fallback_conversation_messages(text)
        messages = [
            {"role": "system", "content": identity},
            *dialogue,
            {"role": "user", "content": text},
        ]
        ladder_chain: list = []
        raw = ""
        stop_reason = ""
        allowed = max(
            _FALLBACK_LADDER_TIMEOUT_S,
            float(budget_s) if budget_s is not None else 0.0,
        )
        deadline = time.monotonic() + allowed
        # A model that is loading is a condition that passes. The chain came
        # back EMPTY — no endpoint considered at all — because the smaller
        # model was itself still coming up, and one pass over the endpoints
        # turned that into a refusal inside a turn that had three minutes left.
        while not raw and time.monotonic() < deadline:
            considered = False
            for endpoint in (BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT):
                remaining = deadline - time.monotonic()
                if remaining <= 1.0:
                    break
                generation_metadata: dict[str, Any] = {}
                try:
                    from core.brain.llm_health_router import _await_while_it_is_working

                    candidate = await _await_while_it_is_working(
                        router.think(
                            text,
                            system_prompt=identity,
                            messages=[dict(message) for message in messages],
                            prefer_tier="tertiary",
                            prefer_endpoint=endpoint,
                            foreground_request=True,
                            allow_cloud_fallback=False,
                            _generation_metadata_sink=generation_metadata,
                        ),
                        budget_s=remaining,
                        user_facing=True,
                        person_is_waiting=True,
                    )
                except (TimeoutError, *_CHAT_RECOVERABLE_ERRORS):
                    continue
                stop_reason = str(generation_metadata.get("generation_stop_reason") or "")
                ladder_chain = list(generation_metadata.get("fallback_chain") or [])
                considered = considered or bool(ladder_chain)
                if isinstance(candidate, dict):
                    chain = list(candidate.get("fallback_chain") or [])
                    considered = considered or bool(chain)
                    ladder_chain = chain or ladder_chain
                    stop_reason = str(candidate.get("generation_stop_reason") or stop_reason)
                    candidate = candidate.get("content") or candidate.get("response") or ""
                # A result with no text and no chain is nothing to ask having
                # been asked. Counting it as an attempt is what kept the wait
                # from ever running: the loop broke on the first pass and the
                # refusal went out while the model was still loading.
                considered = considered or bool(_strip_scaffolding_tags(candidate))
                if _strip_scaffolding_tags(candidate):
                    raw = candidate
                    break
            if raw or considered:
                # Something answered, or something was tried and declined. Only
                # an empty chain means nothing was there to ask yet.
                break
            left = deadline - time.monotonic()
            if left <= 1.0:
                break
            logger.info(
                "🪜 Nothing to ask yet — every endpoint is still loading. "
                "Waiting %.0fs more rather than refusing.",
                left,
            )
            await asyncio.sleep(min(2.0, left))
    except (TimeoutError, *_CHAT_RECOVERABLE_ERRORS) as exc:
        record_degradation(
            "chat.fallback_ladder",
            exc,
            action=f"fallback ladder could not answer while cortex was unavailable ({reason[:80]})",
        )
        return ""
    answer, cut_short = _ends_where_it_meant_to(
        _strip_scaffolding_tags(raw), stop_reason
    )
    if not answer:
        # Name the blocker. "empty answer" describes the outcome and hides the
        # cause, and the router already knows which guard skipped which
        # endpoint — it records a reason for every skip.
        detail = ""
        try:
            chain = ladder_chain or getattr(router, "last_fallback_chain", None) or []
            skips = [
                f"{c.get('endpoint')}:{c.get('skip_reason') or c.get('status')}"
                for c in chain
                if isinstance(c, dict)
            ]
            detail = (
                f" last_error={getattr(router, 'last_background_error', '')!r}"
                f" chain={skips}"
            )[:300]
        except (AttributeError, TypeError, ValueError):
            detail = ""
        record_degradation(
            "chat.fallback_ladder",
            RuntimeError(f"fallback ladder returned an empty answer;{detail}"),
            action="cortex unavailable and the ladder produced nothing",
        )
        return ""
    # The ladder returns its answer straight to the client, so none of the
    # corrections in _stabilize_user_facing_reply run on it. That is how the
    # 2026-09-08 turn reached the screen saying her responses are generated by
    # calculating the next most probable token: the check existed, on a path
    # this reply does not take. The same defect as the readings the ladder
    # used to skip, one layer down.
    answer = str(_remove_self_denials_the_record_refutes(answer) or "").strip()
    if not answer:
        # Every sentence in it was a mechanism claim the record refutes, and
        # the small model has nothing else to say about this. Waiting is the
        # honest answer, which is what "" asks the caller for.
        logger.info(
            "🪜 Fallback ladder answer was entirely self-denials the record refutes; "
            "declined rather than served."
        )
        return ""
    logger.info("🪜 Fallback ladder answered while the cortex was unavailable (%s).", reason[:80])
    ran_out = (
        " I had a fixed slice of time for this and used all of it, so there is "
        "more I would have said."
        if cut_short
        else ""
    )
    # Say which thing happened, not the one that usually happens.
    #
    # This line asserted "the main one is still loading" whatever the reason
    # was, and the reason is right here in the argument. LIVE, 2026-09-07: it
    # was said while the 27B had been resident for seven minutes and the real
    # cause was a latent-cortex receipt contract failing — so the person was
    # told to wait for something that was not going to change by waiting.
    lowered = str(reason or "").lower()
    still_coming = any(
        marker in lowered
        for marker in ("load", "warm", "booting", "starting", "not ready", "spawning")
    )
    why = (
        "the main one is still loading"
        if still_coming
        else "the main one could not finish this turn"
    )
    return (
        f"{answer}\n\n"
        f"(That came from my smaller model — {why}. "
        f"Ask again in a moment if you want me to think about it properly.{ran_out})"
    )


async def _await_foreground_gate(*, budget_s: float) -> Any:
    """Return the inference gate, waiting for it if the runtime is still booting.

    A component that has not registered YET is not a component that failed.
    The HTTP server accepts chat turns from the moment the port binds, which
    is minutes before the inference gate registers; a turn landing in that
    window was answered with "the live answer lane could not finish preparing"
    and classified as a HARD failure — no wait, no retry, turn spent. Live
    2026-07-27 that is exactly what a message typed straight after a reboot
    received, while the UI badge read ONLINE.

    A vanilla model in that situation is slow, not broken. So is this one:
    while boot is genuinely still in progress, wait for the gate to appear.
    Only once boot has settled or stalled is absence a real answer.
    """
    gate = ServiceContainer.get("inference_gate", default=None)
    if gate is not None and hasattr(gate, "ensure_foreground_ready"):
        return gate
    if budget_s <= 0:
        return gate

    try:
        from core.runtime.boot_phases import get_boot_phases

        phases = get_boot_phases()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        phases = None

    if not _boot_is_still_in_progress(phases):
        return gate

    deadline = time.monotonic() + budget_s
    announced = False
    while time.monotonic() < deadline:
        if not _boot_is_still_in_progress(phases):
            break
        if not announced:
            logger.info(
                "⏳ Chat turn arrived before the inference gate registered; "
                "waiting up to %.0fs for boot rather than failing the turn.",
                budget_s,
            )
            announced = True
        await asyncio.sleep(0.25)
        gate = ServiceContainer.get("inference_gate", default=None)
        if gate is not None and hasattr(gate, "ensure_foreground_ready"):
            logger.info("✅ Inference gate registered mid-turn; the turn proceeds normally.")
            return gate

    return ServiceContainer.get("inference_gate", default=None)


def _user_requested_primary_only(text: str) -> bool:
    """Honor explicit user directives to stay on the cortex."""
    lower = (text or "").lower()
    return any(phrase in lower for phrase in _FORCE_PRIMARY_PHRASES)


def _protected_foreground_route(user_message: str) -> dict[str, Any]:
    text = str(user_message or "").strip()
    intent_type = "CHAT"
    deep_handoff = False
    route_meta: dict[str, Any] = {}

    if _user_requested_primary_only(text):
        return {
            "prefer_tier": "primary",
            "deep_handoff": False,
            "intent_type": "CHAT",
            "coding_request": False,
        }

    try:
        from core.phases.cognitive_routing_unitary import CognitiveRoutingPhase
        from core.runtime.turn_analysis import analyze_turn

        analysis = analyze_turn(text)
        if analysis.intent_type in {"CHAT", "TASK"}:
            intent_type = analysis.intent_type
        route_meta = CognitiveRoutingPhase._build_coding_route_metadata(
            text,
            analysis=analysis,
            intent_type=intent_type,
        )
        technical_task = CognitiveRoutingPhase._should_upgrade_to_technical_task(
            text,
            analysis=analysis,
            route_meta=route_meta,
        )
        if technical_task:
            # Keep the protected lane aligned with the main routing phase so
            # explicit multi-file debugging/root-cause work can still claim
            # the deeper solver when the kernel path is bypassed, without
            # letting technical conversation about Aura/selfhood masquerade
            # as an executable coding task.
            intent_type = "TASK"
        deep_handoff = CognitiveRoutingPhase._should_allow_deep_handoff(
            text,
            is_user_facing=True,
            intent_type=intent_type,
            analysis=analysis,
            route_meta=route_meta,
        )
        lower = text.lower()
        deep_handoff = deep_handoff or any(
            marker in lower
            for marker in (
                "debug the failing pytest",
                "fix the failing pytest",
                "root cause analysis",
                "multi-file",
                "deep dive",
                "mathematical proof",
                "formal proof",
                "security audit",
                "vulnerability scan",
            )
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Protected foreground route analysis failed: %s", exc)
        # [STABILITY v53] Tightened fallback — only truly complex technical
        # markers should trigger 72B. Removed "architecture", "debug" (too common).
        # Removed text length >= 900 (long ≠ complex).
        lower = text.lower()
        deep_handoff = any(
            marker in lower
            for marker in (
                "debug the failing pytest",
                "fix the failing pytest",
                "root cause analysis",
                "multi-file",
                "deep dive",
                "mathematical proof",
                "formal proof",
                "security audit",
                "vulnerability scan",
            )
        )

    return {
        "prefer_tier": "secondary" if deep_handoff else "primary",
        "deep_handoff": deep_handoff,
        "intent_type": intent_type,
        "coding_request": bool(route_meta.get("coding_request", False)),
    }


async def _admit_to_foreground_lane(
    *,
    _remaining_foreground_budget: Any,
    gate: Any,
    lane: Any,
) -> tuple[Any, Any, Any, Any]:
    """Wait for the resident lane, or say why this turn cannot have it.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 3 name(s) from the turn and hands back
    4.
    """
    from .chat import (
        _DESKTOP_COGNITIVE_MIN_REQUIRED_BUDGET_S,
        _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S,
    )

    admission_reason = ""
    admission_override = "warming_failed"
    hard_lane_failure = False
    if gate is None or not hasattr(gate, "ensure_foreground_ready"):
        admission_reason = "foreground_lane_unavailable"
        hard_lane_failure = True
    else:
        admission_budget = min(
            180.0,
            _remaining_foreground_budget(
                reserve=(
                    _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S
                    if _cortex_is_cold_loading(lane)
                    else _DESKTOP_COGNITIVE_MIN_REQUIRED_BUDGET_S
                    + _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S
                )
            ),
        )
        try:
            lane = dict(
                await gate.ensure_foreground_ready(timeout=max(1.0, admission_budget)) or {}
            )
        except TimeoutError:
            admission_reason = "foreground_warmup_timeout"
            admission_override = "warming_timeout"
            lane = _mark_conversation_lane_state(
                admission_reason,
                state="warming",
            )
        except _CHAT_RECOVERABLE_ERRORS as admission_exc:
            # Memory pressure is a "not yet", not a "no".
            #
            # Live 2026-07-27, mid-conversation:
            #   foreground_warmup_deferred:memory_pressure:70.2%/19.1GB
            #   (need <72.0% and >=20.0GB)
            # — short by under a gigabyte because another process on
            # the host was holding memory. The turn was refused in one
            # second, with the whole admission budget unspent.
            #
            # Same rule as the boot wait above: a condition that clears
            # on its own is worth waiting out. Retry inside the budget
            # already reserved for admission, and only report the
            # deferral if it is still true when that budget is gone.
            if _lane_warmup_is_deliberately_deferred(
                {"last_failure_reason": str(admission_exc or "")}
            ) or str(admission_exc).strip() == "chat_dependencies_warming":
                retry_deadline = time.monotonic() + max(0.0, admission_budget - 2.0)
                logger.info(
                    "⏳ Foreground lane deferred (%s); waiting up to %.0fs "
                    "for it to clear rather than refusing the turn.",
                    str(admission_exc)[:120],
                    max(0.0, admission_budget - 2.0),
                )
                while time.monotonic() < retry_deadline:
                    await asyncio.sleep(2.0)
                    try:
                        lane = dict(
                            await gate.ensure_foreground_ready(
                                timeout=max(1.0, retry_deadline - time.monotonic())
                            )
                            or {}
                        )
                    except _CHAT_RECOVERABLE_ERRORS as retry_exc:
                        admission_exc = retry_exc
                        continue
                    except TimeoutError:
                        break
                    if bool(lane.get("conversation_ready", False)):
                        logger.info(
                            "✅ Foreground lane cleared its deferral mid-turn; "
                            "the turn proceeds normally."
                        )
                        admission_exc = None
                        break
                if admission_exc is None:
                    admission_reason = ""
                    hard_lane_failure = False
            if admission_exc is not None:
                record_degradation("chat.conversation_lane_admission", admission_exc)
                admission_reason = str(admission_exc or "foreground_warmup_failed")
                hard_lane_failure = admission_reason.startswith(
                    ("mlx_runtime_unavailable:", "local_runtime_unavailable:")
                ) or admission_reason in {
                    "foreground_lane_unavailable",
                    "runtime_shutdown",
                }
                lane = _mark_conversation_lane_state(
                    admission_reason,
                    state="failed" if hard_lane_failure else "recovering",
                )
    return admission_override, admission_reason, hard_lane_failure, lane


async def _protected_foreground_reply(
    reason: str,
    *,
    budget_override_s: float | None = None,
    _chat_session_id: Any,
    _live_turn_trace: Any,
    _remaining_foreground_budget: Any,
    _semantic_user_message: Any,
    body: Any,
    chat_origin: Any,
    desktop_requires_cognitive_engine: Any,
    is_benchmark: Any,
    lane: Any,
) -> str | None:
    """The protected local foreground lane, tried when the cortex misses.

    Lifted out of ``_api_chat_turn`` where it was a closure over nine names.
    They are parameters now and nothing else changed: the body is the same
    body. ``chat.py`` keeps a thin wrapper of the original name so the eight
    call sites inside the handler read as they did.

    The same shape ``_await_the_sovereign_kernel_reply`` already uses — it
    takes this function as a parameter, which is what made the closure a seam
    rather than part of the handler.
    """
    from .chat import (
        _PROTECTED_FOREGROUND_PRIMARY_BUDGET_SECONDS,
        _PROTECTED_FOREGROUND_SECONDARY_BUDGET_SECONDS,
    )

    if is_benchmark:
        return None
    gate = ServiceContainer.get("inference_gate", default=None)
    if gate is None or not hasattr(gate, "generate"):
        return None
    memory_block = _protected_foreground_generation_block_reason()
    if memory_block:
        logger.warning(
            "Skipping protected foreground rescue (%s) under memory guard: %s",
            reason,
            memory_block,
        )
        return None

    route = _protected_foreground_route(_semantic_user_message)
    deep_handoff = bool(route.get("deep_handoff", False))
    if deep_handoff:
        # The protected lane is a live-chat rescue path. Hot-swapping
        # from 32B to 72B here can create exactly the RAM pressure and
        # latency spiral this lane is meant to avoid.
        route = dict(route)
        route["prefer_tier"] = "primary"
        route["deep_handoff"] = False
        route["protected_downgraded_from_deep"] = True
        deep_handoff = False
    if budget_override_s is None:
        direct_budget = min(
            _PROTECTED_FOREGROUND_SECONDARY_BUDGET_SECONDS
            if deep_handoff
            else _PROTECTED_FOREGROUND_PRIMARY_BUDGET_SECONDS,
            _remaining_foreground_budget(reserve=6.0 if deep_handoff else 4.0),
        )
    else:
        direct_budget = min(
            _PROTECTED_FOREGROUND_PRIMARY_BUDGET_SECONDS,
            max(0.0, float(budget_override_s)),
        )
    minimum_budget = 10.0 if deep_handoff else 5.0
    if direct_budget < minimum_budget:
        return None

    messages = await _chat_protected_prompt._build_protected_foreground_messages(
        body.message,
        lane=dict(lane or {}),
        route=route,
        session_id=_chat_session_id,
    )
    logger.warning(
        "⚡ Protected foreground lane engaged (%s, tier=%s, budget=%.0fs).",
        reason,
        route.get("prefer_tier", "primary"),
        direct_budget,
    )
    semantic_completion_expected = True
    try:
        # The resident client owns progress-aware completion and cancellation.
        # An outer copy of the initial estimate cancels its revised allowance.
        direct_reply = await gate.generate(
                body.message,
                context={
                    "origin": chat_origin,
                    "foreground_request": not is_benchmark,
                    "cognitive_engine_required": bool(desktop_requires_cognitive_engine),
                    "desktop_cognitive_engine_required": bool(
                        desktop_requires_cognitive_engine
                    ),
                    "protected_foreground_lane": not is_benchmark,
                    "protected_foreground_reason": reason,
                    "prefer_tier": route.get("prefer_tier", "primary"),
                    "deep_handoff": deep_handoff,
                    # Protected foreground repair is part of the live
                    # Aura lane; keep it local so provider quota or a
                    # remote substrate cannot hijack desktop chat.
                    "allow_cloud_fallback": False,
                    "visible_user_message": _semantic_user_message,
                    "user_surface_validation_prompt": _semantic_user_message,
                    "user_surface_completion_floor": answer_surface_token_floor(
                        _semantic_user_message
                    ),
                    "semantic_completion_contract": semantic_completion_expected,
                    "messages": messages,
                    "brief": (
                        "Protected foreground lane engaged. The kernel is congested or recovering. "
                        "Respond directly to the user in Aura's voice while preserving continuity."
                    ),
                },
                timeout=direct_budget,
        )
    except _CHAT_RECOVERABLE_ERRORS as direct_exc:
        record_degradation("chat", direct_exc)
        logger.warning(
            "Protected foreground lane failed (%s): %s", reason, describe_error(direct_exc)
        )
        return None

    if not direct_reply or not str(direct_reply).strip():
        return None

    metadata_getter = getattr(gate, "get_last_generation_metadata", None)
    generation_metadata = (
        metadata_getter() if callable(metadata_getter) else {}
    )
    generation_metadata = (
        dict(generation_metadata)
        if isinstance(generation_metadata, dict)
        else {}
    )
    raw_receipt = generation_metadata.get("surface_control_receipt")
    if not isinstance(raw_receipt, dict):
        receipt_getter = getattr(gate, "get_last_surface_control_receipt", None)
        raw_receipt = receipt_getter() if callable(receipt_getter) else {}
    receipt = dict(raw_receipt) if isinstance(raw_receipt, dict) else {}
    generation_consumed = _generation_metadata_consumed_foreground_owner(
        generation_metadata
    )
    protected_output_sha256 = hashlib.sha256(
        str(direct_reply).strip().encode("utf-8")
    ).hexdigest()
    transaction_id = _worker_receipt_transaction_id(receipt, direct_reply)
    protected_generation_proven = bool(
        generation_metadata.get("ok") is True
        and generation_metadata.get("is_local") is True
        and generation_consumed
        and transaction_id
    )
    raw_generation_controls = generation_metadata.get(
        "live_mind_generation_controls"
    )
    generation_controls = (
        dict(raw_generation_controls)
        if isinstance(raw_generation_controls, dict)
        else {}
    )
    await _record_desktop_evidence_on_the_trace(
        _live_turn_trace=_live_turn_trace,
        generation_consumed=generation_consumed,
        generation_controls=generation_controls,
        generation_metadata=generation_metadata,
        protected_generation_proven=protected_generation_proven,
        protected_output_sha256=protected_output_sha256,
        receipt=receipt,
        semantic_completion_expected=semantic_completion_expected,
        transaction_id=transaction_id,
    )
    if not protected_generation_proven:
        logger.error(
            "Protected foreground produced text without a valid local generation "
            "receipt; withholding it (metadata=%s receipt=%s).",
            sorted(generation_metadata),
            sorted(receipt),
        )
        return None

    # The protected lane used to pass through a second, independently
    # mutating stabilizer and then skip the normal authorship contract.
    # Keep the model bytes intact and let the shared terminal path own
    # quality, requested-output, and delivery admission.
    stabilized = str(direct_reply).strip()
    recent_user_messages = await _gather_recent_user_messages_for_relevance(
        _semantic_user_message
    )
    is_stale = _is_actionably_stale_response(
        _semantic_user_message,
        stabilized,
    )
    is_same_diff = _is_same_answer_different_prompt(_semantic_user_message, stabilized)
    is_off_topic, off_topic_reason = _evaluate_reply_topicality(
        _semantic_user_message,
        stabilized,
        recent_user_messages=recent_user_messages,
    )
    semantic_glitch, semantic_glitch_reason = _looks_semantically_glitched(
        _semantic_user_message, stabilized
    )
    if is_stale or is_same_diff or is_off_topic or semantic_glitch:
        logger.warning(
            "Protected foreground produced unsafe user-facing reply "
            "(stale=%s same_diff=%s off_topic=%s semantic=%s reason=%s).",
            is_stale,
            is_same_diff,
            is_off_topic,
            semantic_glitch,
            off_topic_reason or semantic_glitch_reason or "",
        )
        return None
    return stabilized


async def _await_the_sovereign_kernel_reply(
    *,
    _attempt_protected_foreground_reply: Any,
    _cancel_kernel_task_if_pending: Any,
    _finalize_fastpath: Any,
    _remaining_foreground_budget: Any,
    chat_origin: Any,
    effective_user_message: Any,
    is_benchmark: Any,
    kernel_timed_out: Any,
    ki: Any,
    reply_text: Any,
) -> tuple[Any, Any, Any]:
    """Wait for the Sovereign Kernel's reply inside the turn's remaining budget.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 10 name(s) and hands back
    2.
    """
    from .chat import (
        _KERNEL_SOFT_REPLY_SLA_SECONDS,
        _SEAM_FELL_THROUGH,
    )

    async def _block() -> Any:
        nonlocal kernel_timed_out, reply_text
        if not reply_text and ki.is_ready():
            logger.debug("REST: Awaiting constitutional processing from Sovereign Kernel...")
            try:
                kernel_timeout = _remaining_foreground_budget()
                # The kernel's receipts belong to the turn that started it, and
                # belonging is inherited from the turn's context rather than
                # granted by a lease. A task created here is created inside that
                # context and is part of the turn by construction.
                kernel_task = get_task_tracker().create_task(
                    ki.process(effective_user_message, origin=chat_origin, priority=True),
                    name="Aura.Server.Chat.kernel_foreground",
                )
                # [STABILITY v53] Two-phase timeout:
                # Phase 1 (soft): Give kernel its full SLA. Don't fire competing
                #   requests during this window — resource contention makes both slower.
                # Phase 2 (hard): If kernel misses soft deadline, try protected foreground
                #   OR wait for kernel with remaining budget, whichever finishes first.
                soft_deadline = min(
                    _KERNEL_SOFT_REPLY_SLA_SECONDS,
                    max(8.0, kernel_timeout - 20.0),
                )
                try:
                    reply_text = await asyncio.wait_for(
                        asyncio.shield(kernel_task),
                        timeout=soft_deadline,
                    )
                except TimeoutError:
                    # Soft deadline missed.
                    # [STABILITY v55] ROOT CAUSE FIX: DO NOT fire a competing
                    # protected foreground request if the cortex is alive and
                    # actively generating for the kernel task. The previous
                    # design fired _attempt_protected_foreground_reply here,
                    # which tried to acquire the same foreground owner the
                    # kernel was using — creating a resource contention spiral
                    # where BOTH requests stall. Only compete if the cortex
                    # is genuinely dead/stuck.
                    hard_budget = max(2.0, _remaining_foreground_budget())
                    cortex_alive = False
                    try:
                        gate = ServiceContainer.get("inference_gate", default=None)
                        if gate and hasattr(gate, "is_alive"):
                            cortex_alive = gate.is_alive()
                    except _CHAT_RECOVERABLE_ERRORS as exc:
                        record_degradation("chat", exc)
                        logger.debug("Inference gate liveness check failed: %s", exc)
                    if cortex_alive:
                        # Cortex is alive — it's just slow. Wait for kernel
                        # to finish instead of competing for the same LLM.
                        logger.info(
                            "⏳ Kernel soft deadline missed but cortex is alive and generating. "
                            "Waiting %.0fs for kernel to finish (no competing request).",
                            hard_budget,
                        )
                        reply_text = await asyncio.wait_for(
                            asyncio.shield(kernel_task),
                            timeout=hard_budget,
                        )
                    elif is_benchmark:
                        logger.warning(
                            "Benchmark kernel soft deadline missed and cortex liveness was not confirmed. "
                            "Continuing to wait on the canonical kernel task instead of switching lanes."
                        )
                        reply_text = await asyncio.wait_for(
                            asyncio.shield(kernel_task),
                            timeout=max(2.0, _remaining_foreground_budget()),
                        )
                    else:
                        # Cortex missed its deadline; try the protected local foreground lane.
                        protected_reply = await _attempt_protected_foreground_reply(
                            "kernel_soft_deadline"
                        )
                        if protected_reply:
                            await _cancel_kernel_task_if_pending(
                                "kernel_soft_deadline_protected_reply"
                            )
                            return await _finalize_fastpath(
                                protected_reply,
                                status="protected_foreground",
                            )
                        # Protected foreground also failed — give kernel remaining time
                        reply_text = await asyncio.wait_for(
                            asyncio.shield(kernel_task),
                            timeout=max(2.0, _remaining_foreground_budget()),
                        )
            except TimeoutError as e:
                kernel_timed_out = True
                await _cancel_kernel_task_if_pending("kernel_timeout")
                # A timeout is a control-flow outcome, not a crash. Dumping a
                # full asyncio traceback for every slow turn buries the real
                # ones: the 2026-07-25 capability run printed CancelledError
                # chains into the operator's terminal for turns that simply
                # took too long and were handled exactly as designed.
                logger.warning(
                    "KernelInterface chat timed out after its budget; the "
                    "fallback ladder takes this turn (%s).",
                    type(e).__name__,
                )
            except _CHAT_RECOVERABLE_ERRORS as e:
                record_degradation("chat", e)
                logger.error(
                    "KernelInterface chat failed natively; legacy fallback policy will decide: %s (%s)",
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response, kernel_timed_out, reply_text


async def _answer_from_grounded_introspection(
    *,
    _finalize_fastpath: Any,
    _semantic_user_message: Any,
    asks_authority: Any,
    grounded_introspection: Any,
) -> Any:
    """Answer straight from grounded introspection when that is what was asked.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 4 name(s) and hands back
    0.
    """
    from .chat import (
        _SEAM_FELL_THROUGH,
    )

    async def _block() -> Any:
        nonlocal grounded_introspection
        if grounded_introspection:
            # Substrate authority gate: introspection responses are RESPONSE category
            _gi_receipt_id = None
            _gi_effect_source = (
                "grounded_authority_report" if asks_authority else "grounded_introspection"
            )
            _gi_status = "grounded_authority" if asks_authority else "grounded_introspection"
            try:
                from core.container import ServiceContainer as _SC_gi

                _sa = _SC_gi.get("substrate_authority", default=None)
                if _sa:
                    from core.consciousness.substrate_authority import (
                        ActionCategory,
                        AuthorizationDecision,
                    )

                    _gv = _sa.authorize(
                        content=_semantic_user_message[:80],
                        source=_gi_effect_source,
                        category=ActionCategory.RESPONSE,
                        priority=0.6 if asks_authority else 0.4,
                        is_critical=asks_authority,
                    )
                    _gi_receipt_id = _gv.receipt_id
                    if asks_authority:
                        grounded_introspection = _chat_conversation_repair._build_grounded_introspection_reply(
                            _semantic_user_message,
                            authority_observability_note=(
                                "This governance report is being emitted under an observability override, "
                                "so the authority state stays inspectable even when normal output is constrained."
                                if _gv.decision == AuthorizationDecision.CRITICAL_PASS
                                else None
                            ),
                        )
                    elif _gv.decision == AuthorizationDecision.BLOCK:
                        logger.debug(
                            "Grounded introspection blocked by substrate — falling through to kernel"
                        )
                        grounded_introspection = None  # fall through to full cognitive path
            except _CHAT_RECOVERABLE_ERRORS as exc:
                record_degradation("chat", exc)
                logger.warning(
                    "Grounded introspection authority gate unavailable; falling through to kernel path: %s",
                    exc,
                )
                grounded_introspection = None

            if grounded_introspection:
                # Record effect with exact receipt_id for provenance matching
                try:
                    from core.consciousness.authority_audit import get_audit

                    get_audit().record_effect(
                        "response",
                        _gi_effect_source,
                        _semantic_user_message[:80],
                        receipt_id=_gi_receipt_id,
                    )
                except _CHAT_RECOVERABLE_ERRORS as exc:
                    record_degradation("chat", exc)
                    logger.debug("Authority audit effect recording failed: %s", exc)
                return await _finalize_fastpath(grounded_introspection, status=_gi_status)
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response
