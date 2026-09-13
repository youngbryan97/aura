"""Saying no, and saying what was missing.

A turn that cannot be served has to end in a sentence a person can act on, and
the contract it failed is the only thing that makes that possible: which proof
was absent, which benchmark was unmet, which part of the mind never answered.
A refusal that names none of those is indistinguishable from a fault, which is
how "I could not get to an answer" came to mean nothing.
"""
from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse

from core.runtime.errors import record_degradation
from interface.routes import chat_conversation_repair as _chat_conversation_repair  # noqa: E402
from interface.routes import chat_preflight as _chat_preflight  # noqa: E402
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
    _mark_conversation_lane_state,
)
from .chat_reply_shaping import (  # noqa: E402
    _append_turn_text_mutation,
)

#: Internal names for the things that keep her from answering, and what each
#: of them means to the person waiting.
_BLOCKER_IN_WORDS: tuple[tuple[str, str], ...] = (
    ("worker_not_alive", "My mind is still starting up."),
    ("model_not_loaded", "My mind is still loading."),
    ("warmup", "I am still warming up."),
    ("cortex", "My main reasoning is still coming online."),
    ("foreground_owner", "I am still finishing something else."),
    ("recovery", "I am recovering from a problem a moment ago."),
    ("memory", "I am still loading what I remember."),
)
#: Proofs that say the ANSWER is unfinished, as opposed to the bookkeeping.
#:
#: `chat_turn_contract` makes this distinction in a comment and nothing acted
#: on it: "one of them is not a statement about the answer at all". A draft cut
#: off mid-clause or judged semantically short is a reason to withhold what she
#: wrote. A retry counter reaching its limit, or a receipt nobody bound, is a
#: reason to say so — not to replace her answer with an apology.
_THE_ANSWER_ITSELF_IS_UNFINISHED = (
    "authored_answer_incomplete:generation_cut_off",
    "authored_answer_incomplete:semantically_short",
    "authored_answer_incomplete:semantic_contract_unmet",
    "authored_answer_incomplete",
    "final_output_contract_unsatisfied",
    "latent_cortex_output_quality_unproven",
    # Her mind did not answer, or its answer was refused. Whatever text is in
    # hand did not come from the turn this contract is about.
    "engine_think_not_invoked",
    "engine_reply_not_accepted",
    "engine_reply_failed",
)
#: Proofs about a RECEIPT — who owned the generation, whether a snapshot was
#: bound, whether anybody checked. None of them is a statement that the text is
#: wrong, so none is a reason to replace what she wrote with an apology.
#:
#: Prefixes, because three of these names carry a suffix naming which of
#: several conditions failed, and the set they were matched against held the
#: bare form. `live_mind_controls_unbound:not_applied` never matched
#: `live_mind_controls_unbound`, so it fell through to "withhold" — and
#: `live_mind_snapshot_unbound` was listed here while the contract emits
#: `live_mind_snapshot_not_ready`, so that entry had never matched anything at
#: all.
_A_PROOF_ABOUT_THE_BOOKKEEPING = (
    "authored_answer_incomplete:retry_exhausted",
    "authored_answer_incomplete:nobody_checked",
    "live_mind_controls_unbound",
    "architecture_context_unbound",
    "live_mind_snapshot_not_ready",
    # LIVE, 2026-09-08: this is the one that fired. A 2,826-character answer,
    # on topic, high confidence, `assessment=ok`, was replaced by "I couldn't
    # get my full attention onto that one" because nothing had recorded WHICH
    # lane owned the generation. That is a receipt about provenance and says
    # nothing about the text.
    "foreground_model_generation_ownership_unproven",
    "latent_cortex_path_unproven",
    "qualified_recurrent_path_unproven",
    "final_output_contract_not_evaluated",
)


def _in_plain_words(blocker: Any) -> str:
    """What a readiness blocker means to somebody waiting for an answer.

    A token is not a reason. LIVE 2026-08-26: "I am not ready to answer that
    yet — my mind is still coming up. I am still worker_not_alive." A person
    can do nothing with that word, and printing it is how internal vocabulary
    keeps arriving on the product surface. Anything without a plain reading
    is left out rather than shown raw: the sentence around it already says
    she is not ready.
    """
    token = str(blocker or "").strip().lower()
    if not token:
        return ""
    for name, said in _BLOCKER_IN_WORDS:
        if name in token:
            return said
    return ""


def _why_there_is_no_answer(lane: Any) -> str:
    """Say what actually stopped the turn, in the words that are true of it.

    The fixed line said "I couldn't put together an answer I'd stand behind",
    which claims a judgement she made about her own answer. Measured live:
    three identical questions refused in a row and the fourth answered
    normally, while the runtime was still warming its caches and foreground
    turns were averaging sixteen seconds. She had not weighed an answer and
    found it wanting. She ran out of time.

    A person can act on "I ran out of time, ask me again" and can do nothing
    with a statement about her standards. The lane already carries why, so
    the reply is built from it rather than from a constant.
    """
    state = lane if isinstance(lane, dict) else {}
    reason = str(state.get("last_failure_reason") or "").strip().lower()
    blockers = [str(item) for item in (state.get("readiness_blockers") or []) if item]
    # Still warming means a warm-up is actually happening.
    #
    # `conversation_ready` is False on this lane because THIS turn just
    # marked it failed, so reading that as "she is still coming up" told
    # people to wait for something that had already finished — measured
    # live, on a lane whose own health said ready with no blockers at all.
    warming = bool(state.get("warmup_in_flight")) or bool(state.get("readiness_blockers"))

    if "timeout" in reason or "deadline" in reason or "slow" in reason:
        return (
            "That one took longer than I had for it and I ran out of time before "
            "I had an answer. Ask me again and it should come back."
        )
    if warming or blockers:
        # The specific reason instead of the general one, where there is one.
        waiting_on = _in_plain_words(blockers[0]) if blockers else ""
        return (
            f"{waiting_on or 'I am not ready to answer that yet — my mind is still coming up.'}"
            " Give me a moment and ask again."
        )
    if "unavailable" in reason or "no_reply" in reason or not reason:
        return (
            "Nothing came back from my own reasoning on that one, so I have "
            "nothing to give you rather than something I made up. Ask me again."
        )
    return (
        f"I could not answer that one: {reason.replace('_', ' ')}. "
        "Ask me again in a moment."
    )


async def _anything_better_than_giving_up(
    message: object, *, reason: str, already: str, budget_s: float | None = None
) -> str:
    """One more thing to try before the honest failure goes out.

    The ladder is a resident model that is loaded, warm, and not the lane
    owner. Two paths reach the giving-up reply and only one of them asked it —
    so the same question that got a real answer through one route got "I
    couldn't get to an answer I'd stand behind" through the other, with the 9B
    idle and the readings unread.

    A refusal is the right answer when nothing better is known. It is the worst
    of the options available when something is.
    """
    # Imported here rather than at module level: the module these
    # came from imports this one. A call-time import also still
    # sees a test's patch of the original.
    from .chat import (
        _answer_from_fallback_ladder,
    )

    if already:
        return already
    try:
        said = await _answer_from_fallback_ladder(
            message, reason=reason, budget_s=budget_s
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.fallback_ladder",
            exc,
            severity="warning",
            action="gave up without asking the smaller model",
        )
        return ""
    if said:
        logger.info(
            "🪜 The ladder answered where the turn was about to give up (%s).",
            reason[:80],
        )
    return said


async def _refuse_an_empty_canonical_reply(
    *,
    _chat_session_id: Any,
    _original_user_message: Any,
    _semantic_user_message: Any,
    is_benchmark: Any,
    lane: Any,
    pending_exchange_id: Any,
    reply_text: Any,
) -> tuple[Any, Any, Any]:
    """Refuse the turn when the canonical path produced nothing to say.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 7 name(s) and hands back
    2.
    """
    from .chat import (
        _SEAM_FELL_THROUGH,
    )

    async def _block() -> Any:
        nonlocal lane, pending_exchange_id
        if not str(reply_text or "").strip():
            lane = _mark_conversation_lane_state(
                "canonical_chat_no_reply",
                state="failed",
            )
            # Human prose on the product surface; the machine identity lives
            # in the `status` field where tooling reads it. The old text
            # shipped 'canonical chat lane', 'implicit legacy fallback' and a
            # raw 'status=' token straight to the user — internal vocabulary
            # that explains nothing to the person waiting for an answer
            # (2026-07-18 soak: 32 turns of it).
            failure_reply = _why_there_is_no_answer(lane)
            # Same reading as the other two refusal sites. "The runtime logs
            # carry the detail" is the sharpest form of the defect: it tells the
            # person their answer exists somewhere she declined to look.
            evidenced_reply = _chat_conversation_repair._self_health_answer_or_empty(
                _semantic_user_message
            )
            if evidenced_reply:
                failure_reply = evidenced_reply
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    _semantic_user_message,
                    failure_reply,
                    record_experience=False,
                )
                pending_exchange_id = None
            else:
                await _chat_preflight._log_exchange(
                    _original_user_message,
                    failure_reply,
                    record_experience=False,
                    session_id=_chat_session_id,
                )
            await _emit_chat_output_receipt(
                failure_reply,
                cause="chat_response",
                metadata={
                    "response_confidence": "failed",
                    "path": "canonical_chat",
                    "status": "canonical_chat_no_reply",
                    "reason": "implicit_legacy_orchestrator_fallback_refused",
                },
            )
            return JSONResponse(
                {
                    "response": failure_reply,
                    "status": "canonical_chat_no_reply",
                    "conversation_lane": lane,
                    "response_confidence": "failed",
                },
                # In-band fail-closed delivery for real users.
                status_code=503 if is_benchmark else 200,
            )
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response, lane, pending_exchange_id


def _a_proof_that_says_the_answer_is_unfinished(missing: tuple[str, ...]) -> bool:
    """True when something in `missing` is about the text rather than a receipt.

    Unrecognised proofs count as being about the answer. A new proof nobody has
    classified must not silently become a reason to serve something —  and
    `tests/test_every_proof_is_classified.py` makes that a failing test rather
    than a silent apology, because three names had drifted out of this list
    without anything noticing.
    """

    for item in missing:
        if item in _THE_ANSWER_ITSELF_IS_UNFINISHED:
            return True
        if not item.startswith(_A_PROOF_ABOUT_THE_BOOKKEEPING):
            return True
    return bool(not missing)


def _fail_closed_on_an_unproven_full_mind_contract(
    *,
    _final_reply: Any,
    _full_mind_unproven: Any,
    _live_turn_contract: Any,
    _live_turn_trace: Any,
    final_live_turn_contract: Any,
    is_benchmark: Any,
    lane_status: Any,
) -> tuple[Any, Any]:
    """Refuse the turn when the required full-mind contract went unproven.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 7 name(s) and hands back
    1.
    """
    from .chat import (
        _SEAM_FELL_THROUGH,
    )

    def _block() -> Any:
        nonlocal final_live_turn_contract
        if _full_mind_unproven:
            missing = tuple(
                str(item)
                for item in (
                    final_live_turn_contract.get("full_mind_missing_proofs") or ()
                )
            )
            written = str(_final_reply or "").strip()
            if written and not _a_proof_that_says_the_answer_is_unfinished(missing):
                # Missing bookkeeping is not evidence the answer is wrong.
                #
                # `chat_turn_contract` already separates the proofs that say the
                # TEXT is unfinished — cut off mid-clause, semantically short —
                # from the ones that say a receipt was never bound. Only the
                # first kind is a reason to withhold what she wrote. LIVE
                # 2026-09-07: an on-topic, high-confidence, 175-character answer
                # to "what was the first thing I said" was replaced by "I
                # couldn't get my full attention onto that one" under
                # `retry_exhausted`, which is a fact about the retry counter.
                logger.warning(
                    "⚠️ Full-mind contract unproven on bookkeeping alone "
                    "(missing=%s); serving the answer she wrote, disclosed as "
                    "bounded rather than replaced.",
                    ",".join(missing) or "unrecorded",
                )
                _live_turn_trace.update(
                    {
                        "cognitive_engine_reply_accepted": True,
                        "response_path": "full_mind_contract_unproven_served",
                    }
                )
                final_live_turn_contract = _live_turn_contract(
                    lane_status=lane_status,
                    response_confidence="bounded",
                    status="full_mind_contract_unproven_served",
                    reply_source="full_mind_contract_unproven_served",
                )
                return JSONResponse(
                    {
                        "response": written,
                        "status": "full_mind_contract_unproven_served",
                        "reason": "full_mind_contract_unproven_on_bookkeeping",
                        "conversation_lane": lane_status,
                        "response_confidence": "bounded",
                        "live_turn_contract": final_live_turn_contract,
                    },
                    status_code=503 if is_benchmark else 200,
                )
            logger.warning(
                "⚠️ Required desktop full-mind contract was not proven; failing "
                "closed instead of serving partial/raw speech (path=%s, missing=%s).",
                final_live_turn_contract.get("response_path") or "",
                # WHICH proof failed is the whole diagnosis. Without it this
                # says only that something did, and the person gets "I couldn't
                # get my full attention onto that one" with no way to find out
                # why — which cost several rounds of guessing.
                ",".join(
                    str(item)
                    for item in (
                        final_live_turn_contract.get("full_mind_missing_proofs") or ()
                    )
                )
                or "unrecorded",
            )
            fail_closed_reply = (
                "I couldn't get my full attention onto that one, and I'd rather "
                "tell you that than answer from half of it. Try me again in a "
                "moment."
            )
            _append_turn_text_mutation(
                _live_turn_trace,
                stage="chat.full_mind_contract_fail_closed",
                method="deterministic_contract_failure_replacement",
                reasons=["desktop_full_mind_contract_not_proven"],
                before=_final_reply,
                after=fail_closed_reply,
                deterministic=True,
                authorship_effect="replaced_by_runtime",
            )
            _live_turn_trace.update(
                {
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": True,
                    "response_path": "desktop_full_mind_contract_not_proven",
                }
            )
            final_live_turn_contract = _live_turn_contract(
                lane_status=lane_status,
                response_confidence="failed_closed",
                status="desktop_full_mind_contract_not_proven",
                reply_source="desktop_full_mind_contract_not_proven",
            )
            return JSONResponse(
                {
                    "response": fail_closed_reply,
                    "status": "desktop_full_mind_contract_not_proven",
                    "reason": "desktop_full_mind_contract_not_proven",
                    "conversation_lane": lane_status,
                    "response_confidence": "failed_closed",
                    "live_turn_contract": final_live_turn_contract,
                },
                # In-band fail-closed delivery for real users.
                status_code=503 if is_benchmark else 200,
            )
        return _SEAM_FELL_THROUGH

    _seam_early_response = _block()
    return _seam_early_response, final_live_turn_contract


async def _serve_the_capability_inventory(
    *,
    _live_turn_contract: Any,
    _live_turn_trace: Any,
    _semantic_user_message: Any,
    capability_inventory: Any,
    lane: Any,
    pending_exchange_id: Any,
) -> tuple[Any, Any, Any]:
    """Serve the capability inventory when that is what the turn asked for.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 6 name(s) and hands back
    2.
    """
    from .chat import (
        _SEAM_FELL_THROUGH,
    )

    async def _block() -> Any:
        nonlocal lane, pending_exchange_id
        if capability_inventory:
            _live_turn_trace.update(
                {
                    "bounded_contract_used": True,
                    "response_path": "desktop_cognitive_engine_capability_inventory",
                }
            )
            lane = _mark_conversation_lane_state(
                "desktop_cognitive_engine_capability_inventory",
                state="recovering",
            )
            logger.warning(
                "Desktop CognitiveEngine produced no acceptable reply for a capability "
                "inventory turn; serving grounded governed-tool inventory instead of "
                "self-process repair or legacy fallback."
            )
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    _semantic_user_message,
                    capability_inventory,
                    record_experience=True,
                )
                pending_exchange_id = None
            await _emit_chat_output_receipt(
                capability_inventory,
                cause="chat_response",
                metadata={
                    "response_confidence": "bounded",
                    "path": "desktop_cognitive_engine_capability_inventory",
                    "status": "desktop_cognitive_engine_capability_inventory",
                    "reason": "desktop_cognitive_engine_required_no_reply",
                },
            )
            return JSONResponse(
                {
                    "response": capability_inventory,
                    "status": "desktop_cognitive_engine_capability_inventory",
                    "reason": "desktop_cognitive_engine_required_no_reply",
                    "conversation_lane": lane,
                    "response_confidence": "bounded",
                    "live_turn_contract": _live_turn_contract(
                        lane_status=lane,
                        response_confidence="bounded",
                        status="desktop_cognitive_engine_capability_inventory",
                        reply_source="desktop_cognitive_engine_capability_inventory",
                    ),
                }
            )
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response, lane, pending_exchange_id


def _fail_closed_on_an_unproven_output_contract(
    *,
    _final_reply: Any,
    _live_turn_contract: Any,
    _live_turn_trace: Any,
    final_live_turn_contract: Any,
    lane_status: Any,
) -> Any:
    """Refuse the turn when the requested-output contract went unproven.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 5 name(s) and hands back
    0.
    """
    from .chat import (
        _SEAM_FELL_THROUGH,
    )

    def _block() -> Any:
        if not bool(final_live_turn_contract.get("final_requested_output_contract_proven")):
            logger.error(
                "Final requested-output contract is unproven; failing closed (kind=%s reasons=%s).",
                final_live_turn_contract.get("final_requested_output_contract_kind") or "unknown",
                ",".join(
                    final_live_turn_contract.get("final_requested_output_contract_reasons") or []
                )
                or "unknown",
            )
            fail_closed_reply = (
                "I couldn't confirm that my answer matched the exact form you asked "
                "for, so I'd rather not send it than send something that only looks "
                "right. Ask again and I'll have another go."
            )
            _append_turn_text_mutation(
                _live_turn_trace,
                stage="chat.final_requested_output_contract_fail_closed",
                method="deterministic_contract_failure_replacement",
                reasons=["final_requested_output_contract_unproven"],
                before=_final_reply,
                after=fail_closed_reply,
                deterministic=True,
                authorship_effect="replaced_by_runtime",
            )
            _live_turn_trace.update(
                {
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": True,
                    "response_path": "requested_output_contract_not_proven",
                }
            )
            failed_contract = _live_turn_contract(
                lane_status=lane_status,
                response_confidence="failed_closed",
                status="requested_output_contract_not_proven",
                reply_source="requested_output_contract_not_proven",
            )
            return JSONResponse(
                {
                    "response": fail_closed_reply,
                    "status": "requested_output_contract_not_proven",
                    "reason": "requested_output_contract_not_proven",
                    "conversation_lane": lane_status,
                    "response_confidence": "failed_closed",
                    "live_turn_contract": failed_contract,
                },
                status_code=503,
            )
        return _SEAM_FELL_THROUGH

    _seam_early_response = _block()
    return _seam_early_response


async def _refuse_an_unmet_benchmark_contract(
    *,
    _chat_session_id: Any,
    _original_user_message: Any,
    _semantic_user_message: Any,
    contract_reason: Any,
    final_benchmark_text: Any,
    pending_exchange_id: Any,
) -> tuple[Any, Any]:
    """Refuse a benchmark turn whose artifact contract was not met.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 6 name(s) and hands back
    1.
    """
    from .chat import (
        _SEAM_FELL_THROUGH,
    )

    async def _block() -> Any:
        nonlocal pending_exchange_id
        if contract_reason:
            logger.warning(
                "Benchmark artifact contract unmet (%s): prompt_len=%d response_len=%d",
                contract_reason,
                len(_semantic_user_message),
                len(final_benchmark_text),
            )
            failed_reply = (
                "Benchmark request failed closed because the canonical kernel response "
                f"did not satisfy the requested artifact contract: {contract_reason}."
            )
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    _semantic_user_message,
                    failed_reply,
                    record_experience=False,
                )
                pending_exchange_id = None
            else:
                await _chat_preflight._log_exchange(
                    _original_user_message,
                    failed_reply,
                    record_experience=False,
                    session_id=_chat_session_id,
                )
            await _emit_chat_output_receipt(
                failed_reply,
                cause="chat_response",
                metadata={
                    "response_confidence": "failed",
                    "path": "kernel_benchmark",
                    "status": "benchmark_artifact_contract_unmet",
                    "reason": contract_reason,
                },
            )
            return JSONResponse(
                {
                    "response": failed_reply,
                    "status": "benchmark_artifact_contract_unmet",
                    "reason": contract_reason,
                    "conversation_lane": _chat_preflight._collect_conversation_lane_status(),
                    "response_confidence": "failed",
                },
                status_code=502,
            )
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response, pending_exchange_id


async def _refuse_an_empty_benchmark_reply(
    *,
    _chat_session_id: Any,
    _original_user_message: Any,
    _semantic_user_message: Any,
    final_benchmark_text: Any,
    pending_exchange_id: Any,
) -> tuple[Any, Any]:
    """Refuse a benchmark turn that produced no canonical response.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 5 name(s) and hands back
    1.
    """
    from .chat import (
        _SEAM_FELL_THROUGH,
    )

    async def _block() -> Any:
        nonlocal pending_exchange_id
        if not final_benchmark_text:
            empty_reply = "Benchmark request produced no canonical kernel response."
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    _semantic_user_message,
                    empty_reply,
                    record_experience=False,
                )
                pending_exchange_id = None
            else:
                await _chat_preflight._log_exchange(
                    _original_user_message,
                    empty_reply,
                    record_experience=False,
                    session_id=_chat_session_id,
                )
            await _emit_chat_output_receipt(
                empty_reply,
                cause="chat_response",
                metadata={
                    "response_confidence": "failed",
                    "path": "kernel_benchmark",
                    "status": "benchmark_no_response",
                },
            )
            return JSONResponse(
                {
                    "response": empty_reply,
                    "status": "benchmark_no_response",
                    "conversation_lane": _chat_preflight._collect_conversation_lane_status(),
                    "response_confidence": "failed",
                },
                status_code=502,
            )
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response, pending_exchange_id
