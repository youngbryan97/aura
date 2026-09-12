"""An answer she already gave, applied to this turn.

A regeneration and a recorded answer both hand the route a reply that was not
produced now, so both have to say where it came from before it is served. The
corrections are the part that matters: a recorded answer carried forward
unchanged will contradict something the turn has since established, and each
correction here names what it reconciled.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from typing import Any

from fastapi.responses import JSONResponse

from core.container import ServiceContainer
from core.conversation.persistence import ConversationRevisionConflictError
from core.runtime.errors import record_degradation
from interface.routes import chat_delivery as _chat_delivery  # noqa: E402
from interface.routes import chat_memory_state as _chat_memory_state  # noqa: E402
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

from .chat import _SERVED_COUNT_OPENING, _SERVED_FROM_RECORD_OPENINGS, _SERVED_TABULAR_OPENING
from .chat_lane_bookkeeping import (  # noqa: E402
    _finalize_regenerated_reply_write,
    _schedule_late_regeneration_finalizer,
)
from .chat_own_source import (
    _ASKS_TO_INSPECT_SHOWN_SOURCE_RE,  # noqa: F401
    _ASKS_WHERE_CODE_LIVES_RE,  # noqa: F401
    _OWN_SOURCE_ROUTE_MARGIN,  # noqa: F401
    _RECEIPTS_WORTH_READING,  # noqa: F401
    _REPO_PROBE_MAX_BYTES,  # noqa: F401
    _SELF_METRIC_CORRECTION_MARK,  # noqa: F401
    _SELF_PROCESS_ABOUT_HER_RE,  # noqa: F401
    _SELF_PROCESS_HYPOTHETICAL_RE,  # noqa: F401
    _attempt_generated_social_grounding_repair,  # noqa: F401
    _correct_unsourced_self_metrics,
    _explains_the_finding,  # noqa: F401
    _serve_repo_diagnosis,
    _turn_asks_where_that_came_from,  # noqa: F401
    )
from .chat_reply_shaping import (  # noqa: E402
    _correct_false_capability_denials,
)
from .chat_served_answers import (  # noqa: E402
    _serve_earlier_conversation,
    _serve_host_load,
    _serve_lifetime,
    _serve_measured_belief_history,
    _serve_measured_filesystem_count,
    _serve_positional_solution,
    _serve_queued_work,
    _serve_recent_activity,
    _serve_solved_game,
    _serve_tabular_answer,
    _serve_worked_out_sequence,
)


async def _apply_regenerated_reply(
    *,
    exchange_id: str,
    session_id: str,
    reply_text: str,
    expected_revision: int,
    expected_reply_sha256: str,
) -> dict[str, Any]:
    """Durably replace exactly the revision selected before model execution."""
    # Imported here rather than at module level: the module these
    # came from imports this one. A call-time import also still
    # sees a test's patch of the original.
    from .chat import (
        _never_an_ellipsis,
    )


    safe_exchange_id = str(exchange_id or "")[:64]
    safe_session_id = str(session_id or "")[:64]
    # A stored replacement that is only an ellipsis is a turn recorded as
    # answered when it was not.
    replacement_text = _never_an_ellipsis(reply_text)
    replacement_sha256 = hashlib.sha256(replacement_text.encode("utf-8")).hexdigest()
    reservation_token = uuid.uuid4().hex
    async with _chat_memory_state._get_convo_lock():
        target_exchange = next(
            (
                entry
                for entry in _conversation_log
                if str(entry.get("id") or "") == safe_exchange_id
                and str(entry.get("session_id") or "")[:64] == safe_session_id
            ),
            None,
        )
        if target_exchange is None:
            return {"applied": False, "state": "conflict", "reason": "target_missing"}
        current_text = str(target_exchange.get("aura") or "")
        current_sha256 = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
        current_revision = int(target_exchange.get("revision") or 1)
        if (
            current_revision != int(expected_revision)
            or current_sha256 != str(expected_reply_sha256 or "")
            or target_exchange.get("regeneration_reservation")
        ):
            return {"applied": False, "state": "conflict", "reason": "source_changed"}
        target_exchange["regeneration_reservation"] = reservation_token
        target_exchange["regeneration_persistence_state"] = "pending"

    persistence = ServiceContainer.get("persistence", default=None)
    replace_aura_turn = getattr(persistence, "replace_aura_turn", None)
    if not callable(replace_aura_turn):
        async with _chat_memory_state._get_convo_lock():
            target_exchange.pop("regeneration_reservation", None)
            target_exchange["regeneration_persistence_state"] = "failed"
        return {
            "applied": False,
            "state": "failed",
            "reason": "canonical_persistence_unavailable",
        }

    scope_kwargs = _chat_preflight._chat_principal_scope_kwargs()
    operation_id = (
        f"{safe_exchange_id}:regeneration:{int(expected_revision)}:{replacement_sha256[:16]}"
    )
    try:
        record = _chat_preflight._start_durable_conversation_write(
            operation_id=operation_id,
            payload={
                "kind": "regeneration",
                "exchange_id": safe_exchange_id,
                "session_id": safe_session_id,
                "expected_revision": int(expected_revision),
                "expected_reply_sha256": str(expected_reply_sha256 or ""),
                "replacement_sha256": replacement_sha256,
                "scope": scope_kwargs,
            },
            operation=lambda: replace_aura_turn(
                exchange_id=safe_exchange_id,
                session_id=safe_session_id or None,
                replacement_content=replacement_text,
                expected_revision=int(expected_revision),
                expected_content_sha256=str(expected_reply_sha256 or ""),
                origin="desktop_ui_regenerate",
                **scope_kwargs,
            ),
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        async with _chat_memory_state._get_convo_lock():
            target_exchange.pop("regeneration_reservation", None)
            target_exchange["regeneration_persistence_state"] = "failed"
            target_exchange["regeneration_error"] = f"{type(exc).__name__}:{exc}"
        return {"applied": False, "state": "failed", "reason": str(exc)}

    record.task.add_done_callback(
        lambda _completed: _schedule_late_regeneration_finalizer(
            record=record,
            exchange_id=safe_exchange_id,
            session_id=safe_session_id,
            expected_revision=int(expected_revision),
            expected_reply_sha256=str(expected_reply_sha256 or ""),
            replacement_text=replacement_text,
            reservation_token=reservation_token,
        )
    )
    state = await _chat_preflight._await_durable_conversation_write(record)
    if state == "committed":
        applied = await _finalize_regenerated_reply_write(
            record=record,
            exchange_id=safe_exchange_id,
            session_id=safe_session_id,
            expected_revision=int(expected_revision),
            expected_reply_sha256=str(expected_reply_sha256 or ""),
            replacement_text=replacement_text,
            reservation_token=reservation_token,
        )
        receipt = record.task.result()
        return {
            "applied": applied,
            "state": "committed" if applied else "conflict",
            "revision": int(receipt.get("revision") or 0) if isinstance(receipt, dict) else 0,
            "content_sha256": replacement_sha256,
        }
    if state == "pending":
        return {
            "applied": False,
            "state": "pending",
            "operation_id": record.operation_id,
        }
    error = record.error or "regeneration persistence failed"
    terminal_error: BaseException | None = None
    if not record.task.cancelled():
        try:
            terminal_error = record.task.exception()
        except (asyncio.InvalidStateError, asyncio.CancelledError):
            terminal_error = None
    conflict = isinstance(terminal_error, ConversationRevisionConflictError)
    return {
        "applied": False,
        "state": "conflict" if conflict else "failed",
        "reason": error,
    }


async def _recorded_answer_corrections(
    user_message: object,
    reply: object,
) -> tuple[str, bool]:
    """Every answer the runtime already holds, applied in order.

    Returns the text and whether a record replaced what the model wrote, so
    the caller does not have to recognise a record by how its first sentence
    opens. That inference is the same shape as the grounding "text marker"
    this codebase replaced with a stamp: it works until a reader phrases a
    true thing differently.
    """
    # Every route goes through `offer`, which counts what it was given and
    # what it did with it. A route that declines returns the reply unchanged,
    # which is right and also means a route that CANNOT fire looks exactly
    # like one that rarely applies — the difference only shows in the count.
    #
    # Each route takes the text so far rather than closing over it: the order
    # is the contract here, and a closure over a loop variable makes the order
    # something a reader has to work out from Python's scoping rules.
    from core.runtime.what_answered_this_turn import offer, offer_async

    body = str(reply or "")
    corrected = body

    the_routes: tuple[tuple[str, Any, bool], ...] = (
        ("measured_filesystem_count",
         lambda text: _serve_measured_filesystem_count(user_message, text), False),
        ("measured_belief_history",
         lambda text: _serve_measured_belief_history(text), False),
        ("earlier_conversation",
         lambda text: _serve_earlier_conversation(user_message, text), False),
        ("host_load", lambda text: _serve_host_load(user_message, text), False),
        ("queued_work", lambda text: _serve_queued_work(user_message, text), False),
        ("recent_activity",
         lambda text: _serve_recent_activity(user_message, text), False),
        ("saved_artifact",
         lambda text: _save_requested_artifact(user_message, text), True),
        ("positional_solution",
         lambda text: _serve_positional_solution(user_message, text), False),
        ("worked_out_sequence",
         lambda text: _serve_worked_out_sequence(user_message, text), False),
        ("lifetime", lambda text: _serve_lifetime(user_message, text), False),
        ("tabular_answer",
         lambda text: _serve_tabular_answer(user_message, text), False),
        ("solved_game",
         lambda text: _serve_solved_game(user_message, text), True),
        ("repo_diagnosis", lambda text: _serve_repo_diagnosis(text), False),
        ("built_artifact", lambda text: _serve_built_artifact(text), False),
    )

    for name, run, awaited in the_routes:
        here = corrected
        # Bound as defaults, not captured: the loop rebinds both on every
        # pass, and a closure over them would read whatever the last pass
        # left behind.
        if awaited:
            answer = await offer_async(
                name, here, lambda one=run, text=here: one(text)
            )
        else:
            answer = offer(name, here, lambda one=run, text=here: one(text))
        corrected = str(answer or here)
    return corrected, corrected.strip() != body.strip()


def _reply_was_served_from_a_record(reply: object) -> bool:
    body = str(reply or "").lstrip()
    if _SERVED_COUNT_OPENING.match(body) or _SERVED_TABULAR_OPENING.match(body):
        return True
    return any(body.startswith(opening) for opening in _SERVED_FROM_RECORD_OPENINGS)


async def _apply_recorded_answer(user_message: object, response: Any) -> Any:
    """Attach a recorded answer to whatever this turn ended up returning.

    Applied HERE, around the whole turn, because _api_chat_turn returns from
    many places and the ones that matter most are the failure branches.

    LIVE, 2026-08-10, five attempts at one defect. Inside
    _stabilize_user_facing_reply the correction was stripped as off-topic, then
    overwritten by a later repair. Moved to _final_reply it survived every
    repair — and the turn that needed it never reached _final_reply, because
    the cognitive engine failed closed with
    "retryable_error_and_nothing_served" and returned from a branch hundreds of
    lines earlier. She answered "I don't have that count because I never
    actually performed the action" about a read recorded five times over.

    Every one of those branches passes through here. Nothing else does.
    """
    from .chat import (
        _append_past_action_record,
    )


    try:
        payload = getattr(response, "body", None)
        if payload is None:
            return response
        data = json.loads(payload)
        if not isinstance(data, dict):
            return response
        reply = data.get("response")
        if not isinstance(reply, str) or not reply.strip():
            return response
        contract = data.get("live_turn_contract")

        # A typed assertion response has already been composed from verified
        # evidence and bound to these exact bytes. Generic prose repair cannot
        # add authority to it; it can only destroy that binding. Preserve any
        # assertion-backed response whose terminal hash still validates.
        try:
            from core.epistemics.assertion import (
                verified_assertion_response_matches,
            )

            assertion_authority = (
                contract.get("verified_assertion_response")
                if isinstance(contract, dict)
                else None
            )
            if verified_assertion_response_matches(reply, assertion_authority):
                return response
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat.assertion_response_authority", exc)

        # The exact response bytes have already been checked against the
        # qualified recurrent receipt at both cognition and terminal delivery.
        # A record lookup here is not stronger evidence for this task; it is a
        # different answer owner. Do not let the outer catch-all wrapper mutate
        # an authenticated state serialization after its final contract.
        if (
            isinstance(contract, dict)
            and contract.get("qualified_recurrent_path_proven") is True
            and contract.get("answer_delivery_proven") is True
        ):
            return response

        # A recorded answer outranks a proven one, and is typed as itself.
        #
        # LIVE, 2026-08-20. "what have you been up to tonight?" was answered
        # "I've been running in the background. The system was a little
        # sluggish earlier" while thirty-eight finished pieces of work sat in
        # the record. The reply had passed the answer contract, so the whole
        # correction chain was skipped by the gate below — every record
        # server with it. The comment there is right that a proof must not
        # come to refer to different text; what follows from that is
        # re-typing the response, not keeping the wrong answer.
        recorded, served_from_record = await _recorded_answer_corrections(
            user_message,
            reply,
        )
        if served_from_record:
            data["response"] = recorded
            if recorded != reply:
                _chat_delivery._invalidate_answer_proof_after_delivery_mutation(
                    data, original_text=reply, reason="recorded_answer_replacement",
                )
                contract = data.get("live_turn_contract")
            data["response_confidence"] = "computed"
            if isinstance(contract, dict):
                contract["response_confidence"] = "computed"
                contract["answer_delivery_proven"] = False
                contract["recorded_answer_served"] = True
            return JSONResponse(
                content=data, status_code=getattr(response, "status_code", 200)
            )

        if isinstance(contract, dict) and contract.get("answer_delivery_proven") is True:
            # Exact authored bytes have crossed the terminal answer contract,
            # and nothing above replaced them. A stylistic correction after
            # this point would make the proof refer to different text.
            return response
        corrected = str(_append_past_action_record(user_message, reply) or reply)
        # Self-metric honesty belongs here for the same reason the recorded
        # answer does. Applied on the repair branch alone it missed the lane
        # that actually served: LIVE 2026-08-17 the guard was wired at one of
        # forty-eight response sites, and "My memory stores are at 87%
        # capacity" shipped a second time — in a reply that opened by saying
        # she does not track memory at all.
        corrected = str(_correct_unsourced_self_metrics(corrected) or corrected)
        # A measured count outranks a generated one. Asking the model to use
        # the real number was tried and produced the wrong number a third time.
        # The same readers, through the one entry point. Evidence informs; it
        # does not enforce, so where the runtime holds the answer it composes
        # it rather than asking again.
        corrected = (await _recorded_answer_corrections(user_message, corrected))[0]
        corrected = str(_correct_false_capability_denials(corrected) or corrected)
        # Cut a reply that stopped mid-clause back to where it last made sense.
        #
        # The repair existed and served the desktop-task lane and the event
        # bridge, not the lane people type into. Live 2026-08-19 a correct
        # diagnosis — it quoted the exact wrong line out of the file — ended
        # "The correction would depend on whether" and was served that way.
        try:
            from core.conversation.response_reliability import complete_truncated_tail

            _whole = complete_truncated_tail(corrected)
            if _whole.strip() and _whole != corrected:
                logger.info("✂️ Trimmed a reply that stopped mid-clause.")
                corrected = _whole
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "chat.truncated_tail",
                exc,
                severity="debug",
                action="served the reply with its unfinished clause",
                enforce_failure_policy=False,
            )
        # Cut a hallucinated continuation of the transcript.
        #
        # LIVE 2026-08-18: "<tool_call> !user yes check it. Read the contents
        # back to me..." reached the person, appended to a real reply. The
        # artifact was flagged as repairable and nothing repaired it, so the
        # draft was served with the model's invention of their next message
        # still attached.
        try:
            from core.conversation.response_reliability import strip_prompt_artifacts

            _cut = strip_prompt_artifacts(corrected)
            if _cut.strip() and _cut != corrected:
                logger.warning(
                    "Cut a prompt artifact from a served reply (%d -> %d chars).",
                    len(corrected),
                    len(_cut),
                )
                corrected = _cut
        except _CHAT_RECOVERABLE_ERRORS as _artifact_exc:
            record_degradation("chat.prompt_artifact_strip", _artifact_exc)
        if corrected == reply:
            return response
        data["response"] = corrected
        _chat_delivery._invalidate_answer_proof_after_delivery_mutation(
            data, original_text=reply, reason="terminal_answer_correction",
        )
        # A served record is not the draft's confidence.
        #
        # Live 2026-08-19 the verbatim conversation history — five turns with
        # their times, read off disk — arrived badged "Partial", inherited from
        # the generated draft it replaced. The same way the exact product
        # 50,420,273 arrived badged "No answer". Whatever the model's attempt
        # scored says nothing about a fact that was looked up.
        if corrected != reply and _reply_was_served_from_a_record(corrected):
            data["response_confidence"] = "computed"
            contract = data.get("live_turn_contract")
            if isinstance(contract, dict):
                contract["response_confidence"] = "computed"
        return JSONResponse(content=data, status_code=getattr(response, "status_code", 200))
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return response
