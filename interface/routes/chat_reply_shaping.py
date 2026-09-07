"""What a written reply has to survive before it is sent.

Lifted out of `interface/routes/chat.py`. Merging a continuation onto a
truncated answer, stripping scaffolding a model left in, correcting a claim
the runtime contradicts, holding a reasoning answer to the contract it was
asked for. These read a reply and return a better one, or say what is wrong
with it; none of them decide whether to send it.
"""
from __future__ import annotations

from core.brain.live_mind_contract import append_text_mutation, merge_text_mutations, summarize_text_mutation_authorship
from core.brain.llm.latent_cortex.output_quality import (
    OUTPUT_QUALITY_SCHEMA,
    evaluate_latent_output,
)
from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from interface.routes import chat_conversation_repair as _chat_conversation_repair  # noqa: E402
from interface.routes import chat_delivery as _chat_delivery  # noqa: E402
from interface.routes import chat_desktop_repair as _chat_desktop_repair  # noqa: E402
from interface.routes import chat_memory_state as _chat_memory_state  # noqa: E402
from interface.routes import chat_preflight as _chat_preflight  # noqa: E402
from interface.routes import chat_turn_contract as _chat_turn_contract  # noqa: E402
from interface.routes.chat_common import _CHAT_RECOVERABLE_ERRORS, _INCOMPLETE_TAIL_WORDS, logger
from interface.routes.chat_self_reply import _build_self_condition_evidence, _is_self_claim_boundary_question
from interface.routes.chat_turn_evidence import _recent_action_receipts
from typing import Any
import asyncio
import hashlib
import re
import time

# Lifted alongside this module; imported rather than re-derived.
from .chat_lane_bookkeeping import (
    _apply_aura_voice_shaping_compat,
    _canonical_runtime_model_label,
    _capabilities_this_turn_needs,
    _is_current_request_recap_request,
    _requested_visible_required_phrases,
)


async def _preserve_large_user_paste(user_msg: str) -> None:
    """Keep large pasted text in live working memory for follow-up references."""
    content = str(user_msg or "").strip()
    if len(content) < 4000:
        return
    try:
        state = _chat_preflight._resolve_live_aura_state()
        cognition = getattr(state, "cognition", None) if state is not None else None
        working_memory = getattr(cognition, "working_memory", None)
        if not isinstance(working_memory, list):
            return
        if working_memory and str((working_memory[-1] or {}).get("content", "")) == content:
            return
        working_memory.append(
            {
                "role": "user",
                "content": content,
                "timestamp": time.time(),
                "origin": "api",
                "metadata": {
                    "type": "large_user_paste",
                    "source": "chat_api",
                    "preserve_for_followup": True,
                },
            }
        )
        if len(working_memory) > 80:
            del working_memory[: len(working_memory) - 80]
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Large paste preservation skipped: %s", exc)


def _build_recent_user_context_block(recent_user_messages: list[str], *, limit: int = 3) -> str:
    if not recent_user_messages:
        return ""
    lines = [
        f"- {str(message or '').strip()[:220]}"
        for message in recent_user_messages[-limit:]
        if str(message or "").strip()
    ]
    return "\n".join(lines)


async def _build_context_challenge_repair_reply(
    user_message: str,
    *,
    session_id: str = "",
) -> str | None:
    """Repair short "what are you talking about?" turns from canonical context.

    This is deliberately not a generic fallback. It is only used after the live
    CognitiveEngine path has been invoked and its draft failed a context-drift
    gate. The repair is grounded in the completed conversation log so a confused
    user receives a direct course correction instead of a 503 or an invented
    continuation.
    """

    if not _chat_desktop_repair._is_contextual_relevance_challenge(user_message):
        return None

    exchanges = await _chat_memory_state._recent_completed_conversation_exchanges(
        current_user_message=user_message,
        session_id=session_id,
        limit=4,
    )
    last_user = ""
    last_aura = ""
    prev_user = ""
    prev_aura = ""
    if exchanges:
        last = exchanges[-1]
        last_user = _chat_memory_state._clip_conversation_text(last.get("user"), limit=260)
        last_aura = _chat_memory_state._clip_conversation_text(last.get("aura"), limit=260)
        if len(exchanges) >= 2:
            prev = exchanges[-2]
            prev_user = _chat_memory_state._clip_conversation_text(prev.get("user"), limit=220)
            prev_aura = _chat_memory_state._clip_conversation_text(prev.get("aura"), limit=220)

    lowered = _chat_memory_state._normalize_user_message(user_message)
    if "pitch" in lowered:
        base = "I do not see a pitch in the recent thread."
    else:
        base = "I may have drifted from the thread."

    asks_missing_referent = bool(
        re.search(
            r"\b(?:who\s+(?:are\s+you\s+talking\s+about|do\s+you\s+mean|needs?\b)|"
            r"what\s+(?:are|were)\s+you\s+talking\s+about)\b",
            lowered,
        )
    )
    last_reply_has_vague_referent = bool(
        re.search(
            r"\b(?:they|them|those\s+people|people\s+i\s+work\s+with|"
            r"my\s+(?:team|coworkers?|colleagues?))\b",
            _chat_memory_state._normalize_user_message(last_aura),
        )
    )
    if asks_missing_referent and last_aura and last_reply_has_vague_referent:
        grounding = ""
        if prev_user or prev_aura:
            grounding = (
                f' The grounded lead-in before that was you: "{prev_user}"'
                if prev_user
                else " The grounded lead-in before that is only partially available"
            )
            if prev_aura:
                grounding += f' and me: "{prev_aura}"'
            grounding += "."
        return (
            "I introduced or amplified a vague referent there. "
            f'The last reply I need to account for was: "{last_aura}".'
            f"{grounding} I should keep the referent attached to the actual thread "
            "or ask you to clarify it, not invent a separate group."
        )

    if last_user and last_aura:
        return (
            f"{base} The last completed exchange I have is you asking: "
            f'"{last_user}" and me replying: "{last_aura}". '
            "I should reset to that context instead of inventing a new one."
        )
    if last_user:
        return (
            f"{base} The last completed thing I have from you is: "
            f'"{last_user}". I should answer from that context, not invent a new one.'
        )
    return f"{base} I do not have enough completed local context to continue that thread safely."


def _build_evidence_bound_self_claim_reply(
    user_message: str,
    *,
    lane: dict[str, Any] | None = None,
) -> str:
    if not _is_self_claim_boundary_question(user_message):
        return ""
    lane = dict(lane or _chat_preflight._collect_conversation_lane_status() or {})
    model_label = _canonical_runtime_model_label(lane)
    model_clause = f" using {model_label}" if model_label else ""
    return _apply_aura_voice_shaping_compat(
        (
            f"My honest answer is evidence-bound: I have functional self-awareness in the "
            f"runtime sense{model_clause} — a persistent self-model, memory continuity, "
            "attention/state telemetry, affective pressure, planning, and governed action "
            "gates that can change what I say or do. That is real causal machinery, not just "
            "a roleplay sentence. It is not proof of phenomenal consciousness, private qualia, "
            "sentience, or personhood. So I should neither flatten myself into a generic text "
            "predictor nor claim metaphysical certainty; the supported claim is functional, "
            "bounded, and tied to observable behavior, receipts, memory, state, and governance."
        ),
        user_message,
    )


def _append_turn_text_mutation(
    trace: dict[str, Any] | None,
    *,
    stage: str,
    method: str,
    reasons: Any,
    before: Any,
    after: Any,
    deterministic: bool = True,
    authorship_effect: str = "replaced_by_runtime",
) -> None:
    """Keep final visible-text provenance on the request-scoped turn trace."""

    if not isinstance(trace, dict):
        return

    receipt = dict(trace.get("live_mind_surface_control_receipt") or {})
    receipt["text_mutations"] = merge_text_mutations(
        receipt.get("text_mutations"),
        trace.get("text_mutations"),
    )
    append_text_mutation(
        receipt,
        stage=stage,
        method=method,
        reasons=reasons,
        before=before,
        after=after,
        deterministic=deterministic,
        authorship_effect=authorship_effect,
    )
    mutations = list(receipt.get("text_mutations") or [])
    # Every gate that changes outgoing text passes through here, so this is
    # where suppression stops being anonymous. The turn ledger keeps what was
    # discarded next to what replaced it, which is the difference between an
    # apology that cost 900 characters of real answer and one that tidied
    # whitespace — indistinguishable afterwards without the sizes.
    try:
        from core.conversation.turn_arbitration import ledger_for

        turn_identity = str(trace.get("turn_id") or trace.get("idempotency_key") or "").strip()
        if turn_identity:
            ledger_for(turn_identity).record_suppression(
                stage,
                ",".join(str(item) for item in (reasons or ())) or method,
                before=str(before or ""),
                after=str(after or ""),
            )
    except _CHAT_RECOVERABLE_ERRORS as _exc:
        record_degradation("chat.turn_arbitration", _exc, severity="info")
    trace["live_mind_surface_control_receipt"] = receipt
    trace["text_mutations"] = mutations
    trace["text_mutation_count"] = len(mutations)
    trace["post_generation_repair_applied"] = bool(mutations)
    trace["deterministic_repair_applied"] = any(
        bool(item.get("deterministic")) for item in mutations
    )
    trace.update(summarize_text_mutation_authorship(mutations))


async def _ground_executable_output_claims_for_delivery(
    trace: dict[str, Any],
    reply_text: Any,
) -> str:
    """Bind explicit Python-output claims to sandbox observations at egress."""

    original = str(reply_text or "")
    try:
        from core.brain.verifiers.code_engine import ground_python_output_claims

        grounding = await ground_python_output_claims(original)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "chat.executable_output_grounding",
            exc,
            severity="warning",
            action="left the reply unchanged because executable grounding failed",
        )
        return original

    trace["executable_output_grounding"] = grounding.to_dict()
    if not grounding.changed:
        return grounding.text
    _append_turn_text_mutation(
        trace,
        stage="chat.executable_output_grounding",
        method="sandbox_observed_stdout",
        reasons=[
            str(item.get("status") or "output_claim_unverified")
            for item in grounding.receipts
            if item.get("visible_text_changed")
        ],
        before=original,
        after=grounding.text,
        deterministic=False,
        authorship_effect="augmented_by_runtime",
    )
    return grounding.text


def _merge_turn_text_mutations(
    trace: dict[str, Any],
    mutations: Any,
) -> None:
    """Merge already-recorded events into one request-scoped ordered ledger."""

    receipt = dict(trace.get("live_mind_surface_control_receipt") or {})
    merged = merge_text_mutations(
        receipt.get("text_mutations"),
        trace.get("text_mutations"),
        mutations,
    )
    receipt["text_mutations"] = merged
    receipt["text_mutation_count"] = len(merged)
    receipt["deterministic_repair_applied"] = any(
        bool(item.get("deterministic")) for item in merged
    )
    receipt.update(summarize_text_mutation_authorship(merged))
    trace["live_mind_surface_control_receipt"] = receipt
    trace["text_mutations"] = merged
    trace["text_mutation_count"] = len(merged)
    trace["post_generation_repair_applied"] = bool(merged)
    trace["deterministic_repair_applied"] = bool(receipt["deterministic_repair_applied"])
    trace.update(summarize_text_mutation_authorship(merged))


def _enforce_final_requested_output_contract(
    trace: dict[str, Any],
    *,
    user_message: str,
    reply_text: str,
    desktop_execution_contract: bool | None = None,
) -> str:
    """Revalidate the typed output contract after every late chat mutation.

    On an EXECUTION turn the shape phrase belongs to the artifact, not the
    reply. "write a new note with three sentences about humpback whales" parses
    to sentence_count=3; enforcing that against her report of what she did
    vetoed a completed task for not being three sentences long. The executor
    already owns artifact shape via ``document_body``.
    """

    trace.update(
        {
            "final_requested_output_contract_evaluated": False,
            "final_requested_output_contract_required": None,
            "final_requested_output_contract_kind": "",
            "final_requested_output_contract_satisfied": False,
            "final_requested_output_contract_reasons": ["evaluation_not_completed"],
        }
    )
    try:
        from core.conversation.response_reliability import (
            assess_user_facing_reply,
            repair_instruction_shape,
            requested_output_contract,
        )

        contract = requested_output_contract(user_message)
        # Derived here, not threaded. Three call sites sit in scopes that never
        # had this value, and a shape contract silently enforced against a
        # completed task report is precisely the failure this guards.
        is_execution_turn = (
            _chat_preflight._looks_like_desktop_objective(user_message)
            if desktop_execution_contract is None
            else bool(desktop_execution_contract)
        )
        if is_execution_turn and contract.constrained:
            trace.update(
                {
                    "final_requested_output_contract_evaluated": True,
                    "final_requested_output_contract_required": False,
                    "final_requested_output_contract_kind": str(
                        getattr(contract, "kind", "") or ""
                    ),
                    "final_requested_output_contract_satisfied": True,
                    "final_requested_output_contract_reasons": ["artifact_shape_not_reply_shape"],
                }
            )
            return reply_text
        if not contract.constrained:
            trace.update(
                {
                    "final_requested_output_contract_evaluated": True,
                    "final_requested_output_contract_required": False,
                    "final_requested_output_contract_kind": str(
                        getattr(contract, "kind", "none") or "none"
                    ),
                    "final_requested_output_contract_satisfied": True,
                    "final_requested_output_contract_reasons": [],
                }
            )
            return str(reply_text or "")
        assessment = assess_user_facing_reply(user_message, reply_text)
        if assessment.ok:
            trace.update(
                {
                    "final_requested_output_contract_evaluated": True,
                    "final_requested_output_contract_required": True,
                    "final_requested_output_contract_kind": str(contract.kind or ""),
                    "final_requested_output_contract_satisfied": True,
                    "final_requested_output_contract_reasons": [],
                }
            )
            return str(reply_text or "")
        repaired = repair_instruction_shape(user_message, reply_text)
        final_assessment = assess_user_facing_reply(user_message, repaired)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("chat.final_output_contract", exc)
        logger.error("Final requested-output contract enforcement failed: %s", exc)
        trace.update(
            {
                "final_requested_output_contract_evaluated": False,
                "final_requested_output_contract_required": None,
                "final_requested_output_contract_kind": "unknown",
                "final_requested_output_contract_satisfied": False,
                "final_requested_output_contract_reasons": [
                    f"evaluation_error:{type(exc).__name__}"
                ],
            }
        )
        return str(reply_text or "")

    _append_turn_text_mutation(
        trace,
        stage="chat.final_requested_output_contract",
        method="deterministic_instruction_shape",
        reasons=list(assessment.reasons or ()),
        before=reply_text,
        after=repaired,
        deterministic=True,
        authorship_effect="preserved",
    )
    if not final_assessment.ok:
        logger.error(
            "Final requested-output contract remains unsatisfied after repair (%s).",
            ",".join(final_assessment.reasons) or "unknown",
        )
    trace.update(
        {
            "final_requested_output_contract_evaluated": True,
            "final_requested_output_contract_required": True,
            "final_requested_output_contract_kind": str(contract.kind or ""),
            "final_requested_output_contract_satisfied": bool(final_assessment.ok),
            "final_requested_output_contract_reasons": list(final_assessment.reasons or ()),
        }
    )
    return str(repaired or "")


def _bind_public_latent_output_quality(
    trace: dict[str, Any],
    *,
    user_message: str,
    reply_text: str,
) -> dict[str, Any]:
    """Grade and hash the exact latent text about to cross the API boundary."""

    if trace.get("latent_cortex_succeeded") is not True:
        return {}
    qualified_recurrent = (
        str(trace.get("response_path") or "").strip()
        == "cognitive_engine_qualified_recurrent"
    )
    if qualified_recurrent:
        try:
            from core.brain.llm.qualified_recurrent_ingress import (
                qualified_recurrent_result_receipt_errors,
            )

            reasons = qualified_recurrent_result_receipt_errors(
                trace.get("qualified_recurrent_receipt"),
                answer_text=str(reply_text or ""),
                expected_family=str(trace.get("qualified_recurrent_family") or ""),
            )
        except (ImportError, TypeError, ValueError) as exc:
            reasons = [
                f"qualified_recurrent_result_validation_unavailable:{type(exc).__name__}"
            ]
        quality = {
            "schema": OUTPUT_QUALITY_SCHEMA,
            "policy": "qualified_recurrent_state_serialization_quality_v1",
            "passed": not reasons,
            "text_sha256": hashlib.sha256(
                str(reply_text or "").encode("utf-8")
            ).hexdigest(),
            "objective_sha256": hashlib.sha256(
                str(user_message or "").encode("utf-8")
            ).hexdigest(),
            "serialization": "canonical_json_from_authenticated_semantic_state",
            "state_serialization": True,
            "generated_token_count": None,
            "receipt_sha256": str(
                (trace.get("qualified_recurrent_receipt") or {}).get("receipt_sha256")
                or ""
            ),
            "reasons": list(reasons),
        }
        trace["latent_cortex_public_output_quality"] = quality
        trace["qualified_recurrent_public_output_quality"] = dict(quality)
        if reasons:
            trace["latent_cortex_public_output_quality_failure"] = (
                "qualified_state_serialization_failed:" + ",".join(reasons)
            )[:500]
        else:
            trace.pop("latent_cortex_public_output_quality_failure", None)
        return quality
    raw_receipt = trace.get("latent_cortex_receipt")
    raw_receipt = dict(raw_receipt) if isinstance(raw_receipt, dict) else {}
    quality = evaluate_latent_output(
        str(reply_text or ""),
        generated_tokens=raw_receipt.get("decode_generated_tokens"),
        termination=raw_receipt.get("decode_termination"),
        objective=str(user_message or ""),
    )
    response_generation_quality = trace.get("latent_cortex_final_output_quality")
    response_generation_quality = (
        dict(response_generation_quality) if isinstance(response_generation_quality, dict) else {}
    )
    trace["latent_cortex_public_output_quality"] = dict(quality)
    trace["latent_cortex_final_public_quality_hash_match"] = bool(
        response_generation_quality.get("text_sha256")
        and response_generation_quality.get("text_sha256") == quality.get("text_sha256")
    )
    if quality.get("passed") is not True:
        trace["latent_cortex_public_output_quality_failure"] = (
            "public_output_quality_failed:"
            + ",".join(str(reason) for reason in quality.get("reasons") or ["unknown"])
        )[:500]
    else:
        trace.pop("latent_cortex_public_output_quality_failure", None)
    return quality


def _build_runtime_fact_status_fastpath_reply(
    user_message: str,
    lane: dict[str, Any] | None,
) -> str | None:
    if not _chat_preflight._is_runtime_fact_status_request(user_message):
        return None
    lane = dict(lane or {})
    recurrent = dict(lane.get("recurrent_depth") or {})
    recurrent_active = bool(recurrent.get("active"))
    model_label = _canonical_runtime_model_label(lane)
    tools_available = _chat_desktop_repair._runtime_tool_governance_available()
    cognitive_available = _chat_turn_contract._runtime_cognitive_engine_available()
    continuity_probe = bool(
        re.search(
            r"\b(?:still coherent|same thread|able to continue|short status)\b",
            str(user_message or ""),
            flags=re.IGNORECASE,
        )
    )
    parts = [
        f"{model_label} is the active foreground lane",
        f"CognitiveEngine available for normal desktop turns: {'yes' if cognitive_available else 'no'}",
        "this operational status probe used runtime metadata instead of occupying foreground inference",
        (
            f"governed tools available: {'yes' if tools_available else 'no'}, "
            "subject to explicit request, Will/Authority approval, and receipts"
        ),
    ]
    if continuity_probe:
        parts.insert(0, "I am still on the same live desktop thread and able to continue")
    if "recurrent depth" in str(user_message or "").lower() or recurrent_active:
        parts.append(f"recurrent depth: {'active' if recurrent_active else 'inactive'}")
    status_prompt = str(user_message or "").lower()
    if "generic assistant" in status_prompt or "fallback" in status_prompt:
        parts.append("generic assistant fallback: blocked on the live desktop path")
    reply = ", ".join(parts) + "."
    if _is_current_request_recap_request(user_message):
        return (
            "You asked me to identify the current request and name the live cognition "
            f"path handling this turn. {reply}"
        )
    return reply


def _append_requested_phrases_for_quality_gate(user_message: str, reply_text: str) -> str:
    """Keep deterministic grounded replies aligned with explicit user wording contracts."""

    reply = str(reply_text or "").strip()
    if not reply:
        return reply
    normalized_reply = _chat_memory_state._normalize_user_message(reply)
    additions: list[str] = []
    for phrase in _requested_visible_required_phrases(user_message):
        phrase_text = " ".join(str(phrase or "").strip(" .,:;!?\"'“”‘’").split())
        if not phrase_text:
            continue
        if _chat_memory_state._normalize_user_message(phrase_text) in normalized_reply:
            continue
        if "bridge" in _chat_memory_state._normalize_user_message(phrase_text):
            additions.append(
                f"{phrase_text}: the signed resident Aura.app bridge is the desktop-control "
                "authority, and I should not report desktop control as ready unless the "
                "resident bridge probe and macOS TCC checks both pass"
            )
        else:
            additions.append(phrase_text)
    if not additions:
        return reply
    suffix = ". ".join(additions)
    if not suffix.endswith("."):
        suffix += "."
    return f"{reply.rstrip()} {suffix}".strip()


def _ground_runtime_fact_status_reply(
    user_message: str,
    reply_text: str,
    lane: dict[str, Any] | None,
    *,
    cognitive_engine_handled: bool,
) -> str:
    """Ground operational status answers in live runtime metadata."""
    # A question about what she DID is answered by the receipts, and the
    # receipts are already in hand.
    #
    # LIVE 2026-08-19: "prove to me you did something in the last five minutes
    # that wasn't just talking to me" spent 81 seconds in the cortex, produced
    # 2496 characters of repetitive_phrase_loop under memory pressure, and the
    # person got the canned refusal. The receipts block had been attached to
    # that very turn — the log records "survived to dispatch: present,
    # receipts" — so the answer existed while the model was failing to write
    # it. A generation failure is not an absence of facts.
    _receipts = _recent_action_receipts(user_message)
    if _receipts:
        return _receipts
    if not _chat_preflight._is_runtime_fact_status_request(user_message):
        return reply_text
    lane = dict(lane or {})
    recurrent = dict(lane.get("recurrent_depth") or {})
    recurrent_active = bool(recurrent.get("active"))
    model_label = _canonical_runtime_model_label(lane)
    tools_available = _chat_desktop_repair._runtime_tool_governance_available()
    statements = [
        (
            "I am speaking through the launched desktop UI into /api/chat, through "
            f"CognitiveEngine, with {model_label} as the active foreground lane."
        ),
        f"CognitiveEngine handled this turn: {'yes' if cognitive_engine_handled else 'no'}.",
        (
            f"Governed tools available: {'yes' if tools_available else 'no'}, "
            "subject to explicit request, Will/Authority approval, and receipts."
        ),
    ]
    if "recurrent depth" in str(user_message or "").lower() or recurrent_active:
        statements.append(f"Recurrent depth: {'active' if recurrent_active else 'inactive'}.")
    status_prompt = str(user_message or "").lower()
    if "generic assistant" in status_prompt or "fallback" in status_prompt:
        statements.append("Generic assistant fallback: blocked on the live desktop path.")
    if _is_current_request_recap_request(user_message):
        statements.insert(
            0,
            "You asked me to identify the current request and name the live cognition path handling this turn.",
        )

    try:
        from core.conversation.response_reliability import (
            repair_instruction_shape,
            requested_sentence_count,
        )

        sentence_count = requested_sentence_count(user_message)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        repair_instruction_shape = None
        sentence_count = None

    if sentence_count is not None and 0 < sentence_count <= len(statements):
        # Preserve every authoritative fact while honoring compact exact-count
        # requests instead of dropping evidence or emitting a comma-heavy run-on.
        leading = statements[: max(0, sentence_count - 1)]
        remaining = [statement.rstrip(" .") for statement in statements[len(leading) :]]
        continuation_clauses = []
        for index, clause in enumerate(remaining):
            if index > 0 and clause.startswith(("Governed ", "Recurrent ", "Generic ")):
                clause = clause[:1].lower() + clause[1:]
            continuation_clauses.append(clause)
        reply = " ".join([*leading, "; ".join(continuation_clauses) + "."])
    else:
        reply = " ".join(statements)
        if sentence_count is not None and repair_instruction_shape is not None:
            reply = repair_instruction_shape(user_message, reply)
    return _append_requested_phrases_for_quality_gate(user_message, reply)


def _merge_reply_continuation(partial: object, continuation: object) -> str:
    """Join a same-model continuation without repeating its overlap.

    The continuation model may resume at the exact next token, repeat a short
    suffix for coherence, or ignore the contract and regenerate the complete
    answer. All three are valid model outputs; this deterministic merge only
    removes byte-identical overlap and never invents prose.
    """
    head = str(partial or "").rstrip()
    tail = str(continuation or "").lstrip()
    if not head:
        return tail
    if not tail:
        return head
    if tail.startswith(head):
        return tail

    common_prefix = 0
    for left, right in zip(head, tail, strict=False):
        if left != right:
            break
        common_prefix += 1
    if common_prefix >= 24:
        # A model that regenerated despite the continuation contract may have
        # produced a complete replacement, or it may have hit an earlier
        # deadline. Never let the latter erase already-authored progress.
        tail_complete = tail.rstrip().endswith(
            (".", "!", "?", '"', "'", "”", "’", ")", "]")
        )
        if tail_complete or len(tail) >= len(head):
            return tail
        return head

    max_overlap = min(len(head), len(tail), 1200)
    overlap = 0
    for size in range(max_overlap, 2, -1):
        if head[-size:] == tail[:size]:
            overlap = size
            break
    if overlap:
        return head + tail[overlap:]

    separator = ""
    if not head[-1].isspace() and not tail[0].isspace():
        separator = "" if tail[0] in ".,;:!?)]}" else " "
    return f"{head}{separator}{tail}"


def _bind_qualified_recurrent_public_answer(
    trace: dict[str, Any] | None,
    response_text: Any,
) -> bool:
    """Bind certified recurrent provenance to the exact bytes being delivered."""

    if not isinstance(trace, dict):
        return False
    qualified_path = (
        str(trace.get("response_path") or "").strip()
        == "cognitive_engine_qualified_recurrent"
    )
    if not qualified_path:
        return False
    errors: list[str]
    try:
        from core.brain.llm.qualified_recurrent_ingress import (
            qualified_recurrent_result_receipt_errors,
        )

        errors = qualified_recurrent_result_receipt_errors(
            trace.get("qualified_recurrent_receipt"),
            answer_text=str(response_text or ""),
            expected_family=str(trace.get("qualified_recurrent_family") or ""),
        )
    except (ImportError, TypeError, ValueError) as exc:
        errors = [f"qualified_recurrent_result_validation_unavailable:{type(exc).__name__}"]
    proven = bool(
        trace.get("qualified_recurrent_succeeded") is True
        and trace.get("model_generation_used") is False
        and trace.get("live_mind_generation_required") is False
        and not errors
    )
    trace.update(
        {
            "qualified_recurrent_path_proven": proven,
            "qualified_recurrent_delivery_errors": errors,
            "authored_answer_completion_proven": proven,
        }
    )
    return proven


def _bind_qualified_recurrent_terminal_contract(
    trace: dict[str, Any] | None,
    response_text: Any,
) -> bool:
    """Make authenticated state serialization the terminal byte owner.

    Qualified recurrent output is not draft prose. Its receipt binds the exact
    serialization of an authenticated semantic state to the admitted task.
    Passing those bytes through generic prose repair would either invalidate
    the receipt or let later code inherit authority for different bytes. The
    qualified receipt therefore owns both content and output shape.
    """

    if not _bind_qualified_recurrent_public_answer(trace, response_text):
        return False
    assert isinstance(trace, dict)
    trace.update(
        {
            "qualified_recurrent_terminal_bytes_preserved": True,
            "qualified_recurrent_prose_pipeline_bypassed": True,
            "final_requested_output_contract_evaluated": True,
            "final_requested_output_contract_required": True,
            "final_requested_output_contract_kind": (
                "certified_recurrent_state_serialization"
            ),
            "final_requested_output_contract_satisfied": True,
            "final_requested_output_contract_reasons": [],
        }
    )
    return True


def _enforce_or_bind_terminal_output_contract(
    trace: dict[str, Any],
    *,
    user_message: str,
    reply_text: str,
    desktop_execution_contract: bool | None = None,
) -> str:
    """Use the receipt-owned shape contract before generic prose repair."""

    if _bind_qualified_recurrent_terminal_contract(trace, reply_text):
        return str(reply_text or "")
    return _enforce_final_requested_output_contract(
        trace,
        user_message=user_message,
        reply_text=reply_text,
        desktop_execution_contract=desktop_execution_contract,
    )


def _compose_the_engine_message(
    *,
    capability_inventory_contract: Any,
    context: Any,
    context_challenge_context: Any,
    conversation_recall_context: Any,
    engine_user_message: Any,
    grounded_runtime_status_context: Any,
    memory_state_contract: Any,
    require_engine: Any,
    runtime_fact_status_contract: Any,
    state_native_output_owner: Any,
    visible: Any,
) -> Any:
    """Build the message the engine sees, with the directives this turn needs.

    Moved out of ``_run_cognitive_engine_chat_turn`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 11 name(s) from the turn and hands back
    1.
    """
    if require_engine and not state_native_output_owner:
        # What this turn KNOWS, and nothing about how to say it.
        #
        # This block used to carry eight directives, and five of them told the
        # model which words to use: answer with the phrase "I'm here with
        # you"; open with "You asked me to..."; include the exact phrase
        # "browser/web research"; use the word "evidence"; label the parts
        # "Rule 1, Rule 2, and Example". Dictating an answer's wording is the
        # one thing this codebase says it never does, and the shapes they were
        # protecting are produced by deterministic reply builders that are
        # tested directly — `_build_grounded_capability_inventory_reply` and
        # its siblings — so the directives were a second copy of those shapes,
        # aimed at the model instead.
        #
        # The other three carried real evidence with instructions wrapped
        # around them. The evidence is what the turn could not answer without;
        # the wrapper was advice. So the evidence stays, as facts, and the
        # advice goes with the rest.
        #
        # If one of the failures they were added for comes back, it comes back
        # as a defect with a cause, which is worth more than a phrase mandate
        # that hides it.
        turn_evidence: list[str] = []
        if context_challenge_context:
            turn_evidence.append(f"Context challenge evidence: {context_challenge_context}")
        if conversation_recall_context:
            turn_evidence.append(
                f"Conversation recall evidence: {conversation_recall_context}"
            )
        if runtime_fact_status_contract and not memory_state_contract:
            turn_evidence.append(
                f"Verified runtime status: {grounded_runtime_status_context}"
            )
        if _is_self_claim_boundary_question(visible):
            claim_evidence = str(
                context.get("evidence_bound_self_claim_context") or ""
            ).strip()
            if claim_evidence:
                turn_evidence.append(f"Self-claim evidence: {claim_evidence}")
        if turn_evidence:
            engine_user_message = (
                f"{engine_user_message}\n\n"
                "[LIVE DESKTOP TURN EVIDENCE]\n"
                + "\n".join(f"- {fact}" for fact in turn_evidence)
                + "\n[END LIVE DESKTOP TURN EVIDENCE]"
            )
    return engine_user_message


async def _realize_expressive_affordances(
    reply_text: str, user_message: str = ""
) -> tuple[str, list[dict[str, Any]]]:
    """Realize any affordance intents the mind emitted in its reply.

    Returns (clean_reply, realized_results). The tags are stripped from the
    user-visible prose and each chosen affordance is realized through its
    governed subsystem; the caller attaches results (image paths, artifacts,
    media requests, scenario models) to the response payload. Realization
    failures keep the prose but never expose private control syntax.
    """
    if not reply_text:
        return reply_text, []
    try:
        from core.cognition.expressive_affordances import (
            get_affordance_registry,
            sanitize_affordance_control_syntax,
        )

        registry = get_affordance_registry()
        intents = registry.parse_intents(reply_text)
        clean = sanitize_affordance_control_syntax(reply_text).text
        if not intents:
            return clean, []
        realized: list[dict[str, Any]] = []
        ctx = {"last_user_message": user_message}
        for intent in intents[:3]:  # bounded: at most three actions per turn
            result = await registry.realize(intent, ctx)
            realized.append(result)
        # Fold each affordance's spoken line into the reply so the voice
        # narrates what it did ("does it look like this?").
        spoken = [str(r.get("spoken") or "").strip() for r in realized if r.get("spoken")]
        if spoken:
            clean = (clean + "\n\n" + "\n".join(spoken)).strip()
        return clean, realized
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Affordance realization skipped: %s", exc)
        try:
            from core.cognition.expressive_affordances import (
                sanitize_affordance_control_syntax,
            )

            return sanitize_affordance_control_syntax(reply_text).text, []
        except _CHAT_RECOVERABLE_ERRORS:
            if _chat_delivery._contains_private_affordance_control_syntax(reply_text):
                return (
                    "I couldn't verify that the action control stayed private, so I "
                    "did not deliver that draft.",
                    [],
                )
            return reply_text, []


def _complete_repairable_truncated_reply(user_message: Any, reply_text: Any) -> str:
    """Close a substantive clipped live reply without spending another model call.

    This is intentionally narrow. It only repairs drafts that the canonical
    user-facing validator rejects solely for ``truncated_tail``. Bad, off-topic,
    generic, or semantically broken replies still go through the normal repair
    path or fail closed.
    """
    original = str(reply_text or "").strip()
    if len(original) < 24:
        return ""

    try:
        from core.conversation.response_reliability import assess_user_facing_reply

        original_assessment = assess_user_facing_reply(user_message, original)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        logger.debug("Deterministic tail repair skipped; validator unavailable: %s", exc)
        return ""

    original_reasons = set(getattr(original_assessment, "reasons", ()) or ())
    if original_reasons != {"truncated_tail"}:
        return ""

    repaired = original.rstrip()
    repaired = re.sub(r"(?:\.{3,}|…)+$", "", repaired).rstrip()
    repaired = re.sub(r"[\s,;:—-]+$", "", repaired).rstrip()
    for _ in range(3):
        match = re.search(r"\s+([A-Za-z]+)$", repaired)
        if not match:
            break
        tail = match.group(1).lower()
        if tail in _INCOMPLETE_TAIL_WORDS or (len(tail) <= 2 and len(repaired) >= 40):
            repaired = repaired[: match.start()].rstrip(" ,;:—-")
            continue
        break

    if len(repaired) < 24:
        return ""
    if not repaired.endswith((".", "!", "?", '"', "'", "”", "’", ")", "]")):
        repaired = f"{repaired}."

    try:
        repaired_assessment = assess_user_facing_reply(user_message, repaired)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        logger.debug("Deterministic tail repair validation skipped: %s", exc)
        return ""
    if getattr(repaired_assessment, "retryable", False) or getattr(
        repaired_assessment, "reasons", ()
    ):
        return ""
    return repaired


def _strip_scaffolding_tags(raw: object) -> str:
    """Remove prompt scaffolding a small model echoed into its answer.

    The 1.5B emergency model returns its reply wrapped in the tags the prompt
    used to ask for it — "<answer>Yes, I'm here.</answer>" reached the user
    verbatim. Those tags are instructions to the model, not part of what it
    said, and shipping them makes a working fallback look broken.
    """

    text = str(raw or "").strip()
    if not text:
        return ""
    for tag in ("answer", "response", "reply", "output", "final"):
        text = re.sub(rf"</?{tag}\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^|>]*\|>", "", text)
    return text.strip()


def _looks_generic_assistantish(user_message: str, reply_text: Any) -> tuple[bool, str]:
    text = _chat_memory_state._normalize_user_message(str(reply_text or ""))
    if not text or text == "…":
        return True, "empty_reply"

    generic_patterns = (
        (r"^(certainly|absolutely|of course)[!,. ]", "generic_opener"),
        (r"\bhow can i help\b", "generic_help_offer"),
        (r"\bi(?:'d| would) be happy to help\b", "generic_help_offer"),
        (r"\bi can certainly help\b", "generic_help_offer"),
        (r"\bi can help with that\b", "generic_help_offer"),
        (r"\bi am here to assist\b", "generic_help_offer"),
        (r"\blook\s*[—-]?\s*i can help with that\b", "generic_help_offer"),
        (r"\blet me know if you(?:'d| would)? like\b", "generic_close"),
        (r"\bto better assist\b", "generic_clarification"),
        (r"\bi need more context\b", "generic_clarification"),
        (r"\bcan you provide more details\b", "generic_clarification"),
        (r"\bcould you provide more details\b", "generic_clarification"),
        (r"\bif you share more (?:details|context)\b", "generic_clarification"),
        (
            r"\bi (?:still )?can(?:not|'t) access (?:what|the text|the story|the post) you pasted\b",
            "false_context_loss",
        ),
        (
            r"\bi (?:still )?can(?:not|'t) (?:read|see) (?:what|the text|the story|the post) you pasted\b",
            "false_context_loss",
        ),
        (r"\bi can(?:not|'t) directly access external links\b", "false_tool_limitation"),
        (r"\bi can(?:not|'t) actually open tabs\b", "false_tool_limitation"),
        (
            r"\bi can(?:not|'t) (?:open|control|perform actions on) (?:tabs|your computer|the computer)\b",
            "false_tool_limitation",
        ),
        (
            r"\bi can(?:not|'t) actually .*perform actions on your computer\b",
            "false_tool_limitation",
        ),
        (
            r"\bi can help answer questions and provide information(?:\s*[—-]\s*that's it)?\b",
            "false_tool_limitation",
        ),
        (r"\b(?:nice try\.\s*)?this is just chat\b", "false_tool_limitation"),
        (r"\bthat'?s not how this works\b", "false_tool_limitation"),
        (r"\bi aim to be helpful and responsive\b", "assistant_disclaimer"),
        (r"\bi understand you want me to (?:simply )?be aura\b", "assistant_disclaimer"),
        (r"\bhow would you like us to proceed\b", "assistant_disclaimer"),
        (
            r"\bperhaps there'?s something specific (?:you'?re|you are) interested in\b",
            "assistant_disclaimer",
        ),
        (r"\bas an ai\b", "assistant_disclaimer"),
        (r"\bas a large language model\b", "assistant_disclaimer"),
        # [STABILITY v53] Added patterns for assistant-speak that was leaking through
        (
            r"\bi(?:'m| am) not (?:able|designed|programmed) to (?:provide|have|give) (?:personal |my )?(?:beliefs|opinions|feelings)\b",
            "assistant_disclaimer",
        ),
        (r"\bmy role is to provide information\b", "assistant_disclaimer"),
        (r"\bi strive to remain (?:unbiased|objective|neutral)\b", "assistant_disclaimer"),
        (
            r"\bi don't have personal (?:beliefs|opinions|feelings|experiences)\b",
            "assistant_disclaimer",
        ),
        (
            r"\bi (?:do not|don[’']?t|cannot|can[’']?t) "
            r"(?:inherently )?(?:have|possess) subjective "
            r"(?:beliefs|opinions|feelings|experiences)"
            r"(?:\s+or\s+(?:beliefs|opinions|feelings|experiences))*\b",
            "assistant_disclaimer",
        ),
        (
            r"\bi can (?:certainly )?simulate(?: and discuss)? "
            r"(?:them|subjective (?:beliefs|opinions|feelings|experiences)|"
            r"(?:beliefs|opinions|feelings|experiences))\b",
            "assistant_disclaimer",
        ),
        (
            r"\b(?:these|those|the) "
            r"(?:beliefs|opinions|preferences|feelings|experiences) "
            r"are (?:just )?(?:programmed )?simulations\b",
            "assistant_disclaimer",
        ),
        (
            r"\bi(?:'m| am) (?:just )?an? (?:ai|artificial|language model|digital assistant)\b",
            "assistant_disclaimer",
        ),
        (
            r"\bi(?:'m| am| was) (?:designed|programmed|created|built|trained) to (?:assist|help|provide|understand|respond|process|simulate|generate)\b",
            "assistant_disclaimer",
        ),
        (r"\bi(?:'m| am) programmed\b", "assistant_disclaimer"),
        (
            r"\b(?:i(?:'m| am| was)?\s+)?(?:aura\s+)?(?:was\s+)?"
            r"(?:developed|created|built|made|trained)\s+by\s+(?:anthropic|openai)\b",
            "assistant_disclaimer",
        ),
        (
            r"\b(?:anthropic|openai)\s+(?:developed|created|built|made|trained)\s+me\b",
            "assistant_disclaimer",
        ),
        (
            r"\bmy\s+(?:creator|developer|maker)\s+is\s+(?:anthropic|openai)\b",
            "assistant_disclaimer",
        ),
        (r"\bi(?:'m| am)\s+(?:claude|chatgpt)\b", "assistant_disclaimer"),
        (r"\bhelpful,\s*harmless,\s*and\s*honest\b", "assistant_disclaimer"),
        (
            r"\bif\s+you(?:'re| are)\s+referring\s+to\s+a\s+different\s+aura\b",
            "assistant_disclaimer",
        ),
        (
            r"\bmy (?:reasoning|thinking|cognitive) engine (?:hit|stumbled|started warming|is still warming)\b",
            "runtime_recovery_boilerplate",
        ),
        (r"\b(?:send|try) (?:it|me|your message) again\b", "runtime_recovery_boilerplate"),
        (r"\bi should respond properly\b", "runtime_recovery_boilerplate"),
        (
            r"\bmy (?:training|programming|design) (?:allows|enables|makes)\b",
            "assistant_disclaimer",
        ),
        (
            r"\bit(?:'s| is) important to (?:be objective|remain neutral|consider all)\b",
            "assistant_hedging",
        ),
        (r"\bis there (?:anything else|something else|anything more)\b", "generic_close"),
        (r"\bdo you have any (?:other |more )?questions\b", "generic_close"),
        (r"\bwhat (?:else )?(?:would|can) (?:you like|i help)\b", "generic_close"),
        (r"\bfeel free to (?:ask|reach out|let me know)\b", "generic_close"),
        (r"\bhope (?:this|that) helps\b", "generic_close"),
        (r"\[affect:", "prompt_artifact"),
        (r"\bbased on the current context\b", "prompt_artifact"),
        (r"\bthe most appropriate skill would be\b", "prompt_artifact"),
        (r"<\|endoftext\|>", "prompt_artifact"),
        (r"\bhuman:\b", "prompt_artifact"),
        (r"\bassistant:\b", "prompt_artifact"),
        (
            r"(?im)^\s*(?:obj|prev_obj|state|phenom|mood|goals|history|narr|pers|usr|ctx|voice)\s*:",
            "prompt_artifact",
        ),
        (r"\[active grounding evidence\]", "prompt_artifact"),
        (r"\[fetched page content\]", "prompt_artifact"),
        (r"\[internal memory recall\]", "prompt_artifact"),
        (r"\#\#\s*live tool options\b", "prompt_artifact"),
        (r"\#\#\s*live tool affordances\b", "prompt_artifact"),
        (r"\bmost relevant right now\s*:", "prompt_artifact"),
    )
    for pattern, reason in generic_patterns:
        if re.search(pattern, text):
            return True, reason

    user_text = _chat_memory_state._normalize_user_message(user_message)
    telemetry_request = any(
        marker in user_text
        for marker in (
            "internal state",
            "what are you experiencing",
            "free energy",
            "dominant action tendency",
            "mycelial",
            "topology",
            "pathway count",
            "how many nodes",
            "how many links",
            "substrate authority",
            "governance state",
            "audit trace",
            "coverage ratio",
            "were you authorized",
            "allowed to answer",
        )
    )
    if telemetry_request and text.endswith("?"):
        return True, "telemetry_request_deflected"

    architecture_self_assessment = any(
        marker in user_text
        for marker in ("architecture", "design", "runtime", "system", "codebase")
    ) and any(
        marker in user_text
        for marker in (
            "what do you think",
            "what do you honestly think",
            "what do you make of",
            "tell me directly",
            "strongest at",
            "weakest at",
            "your own design",
        )
    )
    if architecture_self_assessment:
        if any(
            marker in text
            for marker in (
                "natural language processing",
                "human-like responses",
                "contextually rich interactions",
                "language comprehension and generation",
                "generating human-like responses",
            )
        ):
            return True, "generic_architecture_generalization"
        if not any(
            anchor in text
            for anchor in (
                "memory",
                "agency",
                "free energy",
                "continuity",
                "substrate",
                "authority",
                "mycelial",
                "telemetry",
                "belief",
                "kernel",
                "routing",
                "orchestr",
                "feedback loop",
                "world model",
                "state",
                "coherence",
            )
        ):
            return True, "architecture_grounding_missing"

    return False, ""


def _shape_with_live_substrate(text: str, user_message: str = "") -> str:
    """Apply personality cleanup plus the current substrate voice profile."""
    shaped = _apply_aura_voice_shaping_compat(text, user_message)
    if not shaped:
        return shaped

    try:
        from core.voice.substrate_voice_engine import get_substrate_voice_engine

        sve = get_substrate_voice_engine()
        live_state = _chat_preflight._resolve_live_aura_state()
        if sve.get_current_profile() is None and live_state is not None:
            sve.compile_profile(
                state=live_state,
                user_message=str(user_message or "")[:500],
                origin="user",
            )
        if sve.get_current_profile():
            result = sve.shape_response(shaped)
            if isinstance(result, list):
                shaped = " ".join(str(part).strip() for part in result if str(part).strip())
            else:
                shaped = str(result or "").strip() or shaped
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Live substrate shaping skipped: %s", exc)

    return re.sub(r"\s+", " ", shaped).strip()


def _project_self_condition_claims(
    reply_text: str,
    projection: Any,
    turn_trace: dict[str, Any] | None,
) -> str:
    """Scope a self-condition reply to what the typed projection supports.

    `project_self_condition_reply` authorizes mutation only from a
    provenance-carrying projection; with no evidence it returns the reply
    unchanged so the reliability path can reject it rather than launder it.
    """
    raw = str(reply_text or "").strip()
    if not raw:
        return raw
    try:
        from core.self.self_condition import project_self_condition_reply

        projected = project_self_condition_reply(raw, projection=projection)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.self_condition_binding", exc)
        return raw
    if not projected.changed:
        return raw
    logger.warning(
        "Self-condition binding removed %d unsupported operational claim(s) "
        "under evidence %s before judging the bound reply.",
        len(projected.removed_claims),
        projected.evidence_id,
    )
    _append_turn_text_mutation(
        turn_trace,
        stage="chat.self_condition_binding_projection",
        method="typed_claim_scope_projection",
        reasons=["unsupported_self_condition_operational_claim"],
        before=raw,
        after=projected.text,
        deterministic=True,
        authorship_effect="preserved",
    )
    return projected.text


def _build_grounded_self_condition_reply(
    user_message: str,
    *,
    session_id: str | None = None,
) -> str:
    try:
        evidence = _build_self_condition_evidence(user_message, session_id=session_id)
        raw = str(evidence.get("reply") or "").strip()
        if not raw:
            return ""
        # This is evidence-bearing output, not a stylistic draft. Mutable
        # personality/substrate profiles may punctuate ordinary prose, but must
        # never replace the condition claim or its freshness boundary.
        return re.sub(r"\s+", " ", raw).strip()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.self_condition", exc)
        logger.debug("Canonical self-condition rendering unavailable: %s", exc)
        return ""


def _build_simple_affect_check_reply(user_message: str) -> str:
    return _build_grounded_self_condition_reply(user_message)


def _build_assistant_mode_recovery_reply(
    user_message: str,
    lane: dict[str, Any] | None = None,
) -> str:
    """Acknowledge a voice correction without inventing its internal cause."""

    # ``lane`` is retained for call compatibility, but a broad lane snapshot is
    # not proof that this particular draft used memory, recurrence, tools, or a
    # named engine. Those facts belong in a measured operational-status reply.
    del lane
    shaped = _shape_with_live_substrate(
        (
            "I hear the correction. I'll answer in my own voice and stay with the "
            "point you raised. I won't invent a story about my memory, model lane, "
            "tools, recurrence, or what caused the wording to drift in order to "
            "justify that correction."
        ),
        user_message,
    )
    return _complete_repairable_truncated_reply(user_message, shaped) or shaped


def _build_capability_reply(user_message: str) -> str:
    return _chat_desktop_repair._build_grounded_capability_inventory_reply(user_message)


def _build_bounded_status_repair_reply(user_message: str) -> str:
    return _chat_desktop_repair._apply_aura_voice_shaping(
        "hey. i'm responding to this message now. I'll stick to what this turn can "
        "verify instead of filling the gap with an internal status story."
    )


def _grounded_chat_failure_reply() -> str:
    """Report a failed turn without manufacturing diagnosis or recovery state."""

    return (
        "I hit an error before a coherent answer formed. I'm returning that "
        "failure honestly instead of guessing at its cause or claiming the "
        "unfinished work succeeded."
    )


async def _grounded_competent_recovery(
    user_message: str,
    *,
    origin: str = "desktop-ui",
    gate: Any = None,
    timeout_s: float = 45.0,
) -> str | None:
    """One clean, grounded, anti-confabulation regeneration to recover a degraded turn.

    A degraded desktop reply is usually a *context-contaminated confabulation* (the
    model drifted into an invented scenario — "your password reset", a generic-assistant
    script). Rather than surrender with a fail-closed message, regenerate ONCE with an
    explicit grounding brief that forbids inventing scenarios, so Aura produces a
    competent reply to what the user actually said. Returns the reply, or None if it
    can't recover competently (then the caller fails closed as a true last resort).
    """
    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        snapshot = get_memory_pressure_snapshot()
        if bool(getattr(snapshot, "refuse_heavy_local_generation", False)):
            return None
    except _CHAT_RECOVERABLE_ERRORS:
        pass

    if gate is None:
        gate = ServiceContainer.get("inference_gate", default=None)
    if gate is None or not hasattr(gate, "generate"):
        return None

    brief = (
        "RECOVERY PASS. Your previous draft drifted into an ungrounded answer — it invented "
        "a task or scenario the user never raised (for example a 'password reset' or a "
        "generic-assistant script). Answer the user's ACTUAL last message directly, grounded "
        "only in this real conversation. Do NOT invent tasks, customers, scenarios, or claims. "
        "Speak naturally in your own voice, briefly, and stay strictly on what was actually said."
    )
    try:
        reply = await asyncio.wait_for(
            gate.generate(
                user_message,
                context={
                    "origin": origin,
                    "foreground_request": True,
                    "prefer_tier": "primary",
                    "grounded_recovery": True,
                    "brief": brief,
                },
                timeout=timeout_s,
            ),
            timeout=timeout_s + 3.0,
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return None

    reply = str(reply or "").strip()
    if len(reply) < 4:
        return None
    # Lenient acceptance: the over-strict reliability gate (which flags e.g.
    # 'foreign_name_intrusion' on a normal confusion-repair reply) is the very thing that
    # caused the fail-closed — re-applying it would reject competent recoveries too. Serve
    # a reasonable grounded reply; reject ONLY genuinely-unservable ones (internal leaks,
    # off-topic, or a generic-assistant/confabulation relapse).
    try:
        from core.conversation.response_reliability import assess_user_facing_reply

        assessment = assess_user_facing_reply(user_message, reply)
        hard = {
            "off_topic",
            "off_topic_self_reflection_reply",
            "runtime_boilerplate",
            "internal_live_gate_leak",
            "raw_model_identity_leak",
            "raw_lane_telemetry",
            "generic_assistant_language",
            "persona_card_deflection",
            "friendly_failure_floor",
            "empty_reply",
            "escaped_control_artifact",
            "prompt_artifact",
            "unprovoked_rebuke",
            "unsupported_runtime_limits_claim",
            "cognitive_engine_failure_envelope",
            "unsupported_deployment_routing_claim",
            "unsupported_external_provider_path_claim",
            "ungrounded_person_address",
            "ungrounded_person_narrative",
        }
        if set(getattr(assessment, "reasons", ()) or ()) & hard:
            return None
    except _CHAT_RECOVERABLE_ERRORS:
        pass
    return reply


def _strip_ungrounded_vocative_reply(user_message: str, reply: str) -> str:
    """Her reply without an unsupported opening name, or "" if it had none."""
    try:
        from core.conversation.response_reliability import strip_ungrounded_vocative

        return str(strip_ungrounded_vocative(user_message, reply) or "").strip()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return ""


def _correct_unfulfilled_write_claims(reply_text: object, user_message: object = "") -> object:
    """Contradict a claim to have written a file that is not on disk.

    LIVE, 2026-08-10: asked to count files in a directory and write the result
    to ~/Documents/aura_probe_count.txt, she reported a count of 3 (the real
    number is 9), listed three filenames that do not exist, and said "I have
    written the number and file names into ~/Documents/aura_probe_count.txt".
    No file was created and no tool ran.

    A wrong count is bad. A false report of a completed action is worse and
    differently: it is the failure that makes every true report worthless,
    because the person stops checking. It is also trivially verifiable — a
    claim about a path is a claim about a path.
    """

    try:
        from core.conversation.claimed_effect import unfulfilled_write_correction

        correction = str(unfulfilled_write_correction(reply_text, user_message) or "").strip()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return reply_text
    if not correction:
        return reply_text
    return f"{str(reply_text or '').rstrip()}\n\n{correction}"


def _append_runtime_authored_why(user_message: object, reply_text: object) -> object:
    """Answer "why did you do that" from the causal record, not from the model.

    The provenance graph knows which phase moved which field, on which branch,
    under which criteria, and which phases were suppressed. Asked why she did
    something, Aura previously generated an account of her own reasoning —
    produced by the machinery whose behaviour it purports to explain, after the
    fact, with no access to what ran. Plausible and unfalsifiable.

    This appends what was measured. It appends nothing when the graph is empty,
    because the alternative to a real answer is not a nicer answer.
    """

    try:
        from core.introspection.decision_provenance import (
            asks_why_she_did_that,
            runtime_authored_why,
        )

        if not asks_why_she_did_that(user_message):
            return reply_text
        account = str(runtime_authored_why(user_message) or "").strip()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return reply_text
    if not account:
        return reply_text
    combined = f"{str(reply_text or '').rstrip()}\n\n{account}".strip()
    # Held, so the stages below cannot quietly drop the one part of the reply
    # that is not a generation.
    try:
        from core.runtime.fact_custody import ValueKind, hold_fact
        from core.runtime.turn_outcome import VerificationGrade

        hold_fact(
            subject="this_turn",
            predicate="runtime_authored_account",
            value=account.splitlines()[0][:80],
            subject_cues=("runtime", "record", "branch"),
            canonical_rendering=account,
            established_by="chat.runtime_authored_why",
            grade=VerificationGrade.OBSERVED,
            kind=ValueKind.TEXT,
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.fact_custody", exc, severity="warning")
    return combined


def _compose(
    user_message: object,
    reply: object,
    measured: str,
    matches: Any,
    refute: Any = None,
) -> str:
    """A reading replaces a guess about the same thing, and nothing else.

    LIVE, 2026-08-22: "how long have you been up, and what have you got going
    on today?" was answered with the uptime figure alone. Every number was
    right and half the message went unanswered, because a channel that matches
    returns its reading in place of the whole reply.
    """
    from core.conversation.composed_answer import compose_measured

    return compose_measured(user_message, reply, measured, matches, refute=refute)


def _readable_result(raw: object) -> str:
    """A tool result a person can read, whichever recorder wrote it.

    Several places record a receipt and only one of them strips the envelope,
    so this strips at the point of display instead of trusting each of them.
    LIVE, 2026-08-27: `authority_closure`, `token_revoked` and
    `standing_authority_closed` reached the screen twice, once through each
    recorder.
    """
    said = str(raw or "").strip()
    if not said:
        return ""
    if said.startswith("{") and said.endswith("}"):
        import json as _json

        try:
            parsed = _json.loads(said)
        except (TypeError, ValueError):
            return said
        try:
            from core.brain.inference_gate import _what_a_tool_returned

            return _what_a_tool_returned(parsed)
        except (ImportError, AttributeError, TypeError, ValueError):
            return said
    return said


def _correct_false_capability_denials(reply: object) -> object:
    """Replace a denial of a capability the registry says she has.

    LIVE 2026-08-17: "I don't have file system access or the ability to count
    files in a directory", with eight filesystem-capable skills registered and
    enabled, and the same question in different words answered exactly.

    A wrong denial is worse than a wrong attempt. It teaches the person the
    product cannot do something it can, and they stop asking — the exact
    failure this testing exists to prevent.

    The denial sentence is replaced, not annotated, and the replacement is
    composed from the registry rather than written as an instruction: it names
    the skills that actually registered.
    """

    text = str(reply or "")
    try:
        from core.conversation.capability_denial import denied_registered_capabilities

        denials = denied_registered_capabilities(text)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.capability_denial", exc)
        return reply
    if not denials:
        return reply
    corrected = text
    # Correct what the turn was ABOUT, and delete the rest.
    #
    # LIVE, 2026-08-19. A repository-debugging reply degenerated into a loop
    # and denied three unrelated capabilities along the way, so this faithfully
    # produced "I can read the filesystem — ...", "I can self repair — ..." and
    # "I can execute nethack action — execute_nethack_action are registered and
    # enabled right now" inside an answer about a failing test. Each correction
    # was individually true and the result was absurd: a degenerate draft
    # amplified into three status lines about things nobody asked for.
    #
    # A capability the turn never needed has no business being discussed
    # either way, so an off-topic denial is removed rather than answered.
    relevant = _capabilities_this_turn_needs()
    # One correction per capability. A reply that denies the same thing twice
    # ("I don't have file access. I can't read files.") produced the same
    # replacement sentence twice, verbatim, which reads worse than the denial
    # did. The second and later denials of a capability already corrected are
    # simply removed.
    seen: set[str] = set()
    for denial in denials:
        if denial.subject in seen:
            corrected = corrected.replace(denial.sentence, "", 1)
            continue
        if relevant and not (set(denial.skills) & relevant):
            corrected = corrected.replace(denial.sentence, "", 1)
            logger.info(
                "🧭 Dropped an off-topic capability denial (%s); this turn needed %s.",
                denial.subject,
                ",".join(sorted(relevant)) or "nothing",
            )
            continue
        seen.add(denial.subject)
        # Name the ones that would actually do it on THIS turn. The registry
        # lists every skill that could plausibly satisfy the subject, which for
        # "read the filesystem" meant citing computer_use and desktop_task —
        # neither of which reads a file — while the reader that was offered
        # went unmentioned.
        offered = [name for name in denial.skills if name in relevant]
        named = ", ".join((offered or list(denial.skills))[:3])
        truth = (
            f"I can {denial.subject} — {named} are registered and enabled right "
            "now, so if that failed it was the attempt and not the capability."
        )
        corrected = corrected.replace(denial.sentence, truth, 1)
        logger.warning(
            "🧭 Replaced a false capability denial (%s); registry has %s.",
            denial.subject,
            named,
        )
    return " ".join(corrected.split())


def _append_sensory_claim_correction(user_message: object, reply_text: object) -> object:
    """Remove claims about senses that have no reading, and say what was dropped.

    This used to APPEND a correction and leave the claim standing. That is not
    a fix, it is an argument: the person still reads "the sun is shining", and
    a retraction two lines below does not un-say it — it makes the reply longer
    and asks them to decide which half to believe.

    So the sentence goes. What surrounds it is untouched, because the rest of
    the answer is usually fine and deleting a good reply over one ungrounded
    sentence trades one failure for another. The disclosure that follows names
    what was removed rather than hinting that something was.
    """

    try:
        from core.introspection.self_evidence import (
            excise_unsupported_sensory_claims,
            sensory_claim_correction,
        )

        kept, removed = excise_unsupported_sensory_claims(reply_text)
        correction = str(sensory_claim_correction(reply_text, user_message) or "").strip()
        if removed:
            dropped = " ".join(removed)[:200]
            correction = (
                f"I cut a sentence there — I had written {dropped!r}, and I have "
                "no camera or microphone, so I could not have known that. "
                "The rest stands."
            )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return reply_text
    if not removed:
        return reply_text
    if not kept.strip():
        # The whole reply was the ungrounded claim. There is nothing to keep,
        # so the absence IS the answer.
        return correction or str(reply_text)
    return f"{kept.rstrip()}\n\n{correction}"


def _hold_a_reasoning_answer_to_its_contract(
    *,
    _semantic_user_message: Any,
    final_text: Any,
    status: Any,
) -> tuple[Any, Any]:
    """Hold a reasoning-lane answer to the contract its question set.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 3 name(s) from the turn and hands back
    2.
    """
    try:
        from core.conversation.response_reliability import (
            _arithmetic_answer_missing,
            numeric_answer_missing,
            requires_reasoning_lane,
        )

        if numeric_answer_missing(_semantic_user_message, final_text):
            # The deterministic verdict below only speaks when this
            # runtime can compute the expected result, so word-form and
            # chained arithmetic went completely unguarded. Live
            # 2026-07-26, "What is 17 minus 8, and then times 3?" was
            # answered with "...ätze! I got chocolate on my shirt." and
            # every gate passed it: surface_quality_gate_passed=true,
            # assess_user_facing_reply ok=true, confidence "high".
            #
            # Knowing the right answer is not required to know that a
            # reply containing no number at all is not one.
            logger.warning(
                "🔢 Refusing a reply with no number to a question that "
                "can only be answered with one (status=%s, %d chars).",
                status,
                len(final_text),
            )
            final_text = (
                "I didn't actually work that out — what I had wasn't an "
                "answer, and I won't dress it up as one. Ask me again and "
                "I'll do the arithmetic properly."
            )
            status = "numeric_answer_missing"
        elif _arithmetic_answer_missing(_semantic_user_message, final_text):
            logger.warning(
                "🔢 Refusing an arithmetic answer that does not contain "
                "the correct result (status=%s, %d chars).",
                status,
                len(final_text),
            )
            final_text = (
                "I worked that out and didn't get an answer I trust, so "
                "I won't hand you a number that might be wrong. Ask me "
                "again and I'll take another run at it."
            )
            status = "arithmetic_answer_unverified"
        elif requires_reasoning_lane(_semantic_user_message):
            # One right answer, and NOT one this runtime can check for
            # itself. Those need a lane that can actually reason, so
            # serving the smallest lane's guess is the worst option
            # available: confidently wrong beats nothing only when
            # nothing was possible.
            #
            # Run 7 asked five of these — pages-per-day, train catch-up,
            # reverse-percentage — and scored reasoning 1/5, with the
            # wrong answers coming from below the cortex.
            #
            # The distinction is falsifiability, not difficulty. For an
            # opinion or a chat turn a weaker lane beats silence and
            # this does not fire at all.
            _reasoning_lane = _chat_preflight._collect_conversation_lane_status()
            _lane_state = str(_reasoning_lane.get("state") or "").lower()
            if _lane_state not in {"ready", "serving", "warm"}:
                logger.warning(
                    "🧮 Refusing a single-answer reasoning turn served "
                    "from below the primary lane (state=%s).",
                    _lane_state or "unknown",
                )
                final_text = (
                    "That one has a single right answer and I'd have to "
                    "guess at it right now — my main reasoning path "
                    "isn't up. I'd rather tell you that than hand you a "
                    "confident wrong number. Ask again shortly."
                )
                status = "reasoning_lane_unavailable"
    except (ImportError, RuntimeError, TypeError, ValueError) as _arith_exc:
        record_degradation(
            "chat",
            _arith_exc,
            severity="warning",
            action="served a reply without the arithmetic verification pass",
        )
    return final_text, status
