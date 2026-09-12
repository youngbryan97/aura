"""A turn that refers back, and the evidence that lets it.

"That one", "the thing you just read", "where did you save it" — each of those
is a question about an earlier turn, and each is answerable only from what was
written down at the time. What is here decides whether a reply that claims to
remember actually carried the evidence, and builds the context a follow-up is
answered from.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from core.runtime.errors import record_degradation
from interface.routes import chat_memory_state as _chat_memory_state  # noqa: E402
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
from interface.routes.chat_quality import (  # noqa: E402
    _reply_assessment_requires_repair,
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
    _gather_recent_user_messages_for_relevance,
)

_MEMORY_STATE_COMPATIBLE_ASSESSMENT_REASONS = frozenset(
    {
        "off_topic_self_reflection_reply",
        "missing_requested_self_process_coverage",
        "too_thin_for_operational_status_turn",
        "too_thin_for_status_turn",
    }
)
_RETAINED_MEMORY_EVIDENCE_REQUEST_RE = re.compile(
    r"\b(?:"
    r"remember|recall|memory|memories|retained|retention|across\s+sessions?|"
    r"last\s+(?:week|month|session|time)|previous\s+(?:session|conversation|chat)|"
    r"earlier\s+(?:conversation|session|chat)|persistent\s+context|conversation\s+continuity"
    r")\b",
    re.IGNORECASE,
)
_REFERENTIAL_FOLLOWUP_MARKERS = (
    "can you answer it",
    "you gonna answer",
    "answer the question",
    "answer it",
    "the last question",
    "that question",
    "what specifically",
    "what's the actual thing you need",
    "whats the actual thing you need",
)
_REFERENTIAL_FOLLOWUP_RE = re.compile(
    r"(?:"
    r"\b(?:how|why|when|where|what)\s+"
    r"(?:does|do|did|is|are|was|were|would|will|could|should)\s+"
    r"(?:that|this|it|those|these)\b"
    r"|\bwhat\s+about\s+(?:that|this|it|then|now)\b"
    r"|\b(?:is|are|was|were|does|did|has|have)\s+"
    r"(?:that|this|it|those|these)\b"
    r")",
    re.IGNORECASE,
)
#: Words that make a turn possibly about something she looked at: the
#: surface itself, the act of seeing, or a reference back to a thing rather
#: than a topic ("that repo", "those tabs").
_PERCEPTION_RELEVANCE_RE = re.compile(
    r"\b(?:screen|screens|window|windows|tab|tabs|monitor|display|desktop|"
    r"app|apps|application|page|browser|see|seeing|saw|seen|look|looking|"
    r"looked|watch|watching|showing|shown|visible|open|onscreen|"
    r"front|foreground)\b"
    r"|\b(?:that|those|this|these|it)\s+"
    r"(?:one|thing|repo|repository|video|title|file|window|tab|page|app|"
    r"article|error|message|name)\b"
    # Asking how she knows something is a question ABOUT the perception:
    # she cannot source what she saw if the seeing was not carried in.
    r"|\bhow (?:did|do) you know\b"
    r"|\bhow (?:can|could) you tell\b"
    r"|\bwhere did (?:that|it|this|they) come from\b"
    r"|\bwhich one\b",
    re.IGNORECASE,
)
#: A reply asserting she did not do the thing. When receipts say otherwise, the
#: record has to arrive before the denial rather than after it.
_DENIES_THE_ACTION_RE = re.compile(
    r"\b(?:i\s+(?:did\s?n[o']?t|never|have\s+not|haven[\u2019']t)\s+"
    r"(?:actually\s+)?(?:do|did|run|ran|execute|perform|count|read|write|wrote)"
    r"|i\s+don[\u2019']?t\s+have\s+(?:that|the|it)"
    r"|no\s+record\s+of\s+(?:that|it))\b",
    re.IGNORECASE,
)


def _memory_state_evidence_is_missing_from_reply(
    user_message: str,
    reply_text: str,
    memory_state_evidence: tuple[str, str] | None,
) -> bool:
    """Return True when canonical memory evidence was not honored visibly."""
    # Reached at call time: chat.py imports this module, so the other
    # direction cannot be a module-level import.
    from .chat import _conversation_recall_reply_is_inadequate


    del user_message  # Reserved for future status-specific diagnostics.
    if not memory_state_evidence:
        return False

    memory_reply, memory_status = memory_state_evidence
    status = str(memory_status or "").strip()
    reply = str(reply_text or "").lower()
    if not reply:
        return True

    expected_content = _chat_memory_state._extract_session_memory_pin_request(
        str(memory_reply or "")
    )
    if not expected_content:
        match = re.search(r'"([^"]{1,240})"', str(memory_reply or ""))
        expected_content = match.group(1) if match else ""
    expected_content = str(expected_content or "").strip()

    if status in {
        "session_memory_pin",
        "session_memory_pin_transient",
        "session_memory_recall",
        "session_memory_context_recall",
    }:
        if not expected_content:
            return True
        return expected_content.lower() not in reply

    if status == "session_memory_miss":
        return not (
            "don't have" in reply
            or "do not have" in reply
            or "no pinned" in reply
            or "not pinned" in reply
        )

    if status in {"owner_identity_recall", "conversation_recall"}:
        return _conversation_recall_reply_is_inadequate(
            "",
            reply_text,
            str(memory_reply or ""),
        )

    return False


def _memory_state_reply_satisfies_canonical_evidence(
    user_message: str,
    reply_text: str,
    *,
    memory_state_evidence: tuple[str, str] | None = None,
    canonical_memory_state_evidence: str = "",
) -> bool:
    """True only when visible prose honors the canonical memory/state evidence."""

    if memory_state_evidence:
        return not _memory_state_evidence_is_missing_from_reply(
            user_message,
            reply_text,
            memory_state_evidence,
        )
    if canonical_memory_state_evidence:
        return not _canonical_memory_state_evidence_missing_from_reply(
            canonical_memory_state_evidence,
            reply_text,
        )
    return False


def _reply_assessment_requires_repair_with_memory_evidence(
    assessment: Any,
    user_message: str,
    reply_text: str,
    *,
    memory_state_evidence: tuple[str, str] | None = None,
    canonical_memory_state_evidence: str = "",
) -> bool:
    """Keep hard failures, but do not reject honored memory/state replies as self-process misses."""
    # Reached at call time: chat.py imports this module, so the other
    # direction cannot be a module-level import.
    from .chat import _conversation_recall_reply_is_inadequate


    if not _reply_assessment_requires_repair(assessment):
        return False
    reasons = set(getattr(assessment, "reasons", ()) or ())
    # A complete short answer is not a defect. "68" fails every thinness
    # heuristic there is and is still the whole correct answer to "what's
    # 17 times 4?" — live 2026-08-04 this gate turned it into "I couldn't
    # get to an answer I'd stand behind".
    try:
        from core.conversation.surface_disposition import (
            draft_is_servable,
            short_draft_answers_closed_question,
        )

        if (
            reasons
            and draft_is_servable(reasons)
            and short_draft_answers_closed_question(reply_text, user_message)
        ):
            return False
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat",
            exc,
            severity="warning",
            action="kept the standard repair path after the short-answer check failed",
        )
    if (
        reasons
        and reasons.issubset(_MEMORY_STATE_COMPATIBLE_ASSESSMENT_REASONS)
        and _memory_state_reply_satisfies_canonical_evidence(
            user_message,
            reply_text,
            memory_state_evidence=memory_state_evidence,
            canonical_memory_state_evidence=canonical_memory_state_evidence,
        )
    ):
        return False
    return True


def _is_retained_memory_evidence_request(user_message: str) -> bool:
    text = str(user_message or "")
    if not text.strip():
        return False
    if _chat_memory_state._is_session_memory_recall_request(
        text
    ) or _chat_memory_state._classify_conversation_recall_request(text):
        return True
    return bool(_RETAINED_MEMORY_EVIDENCE_REQUEST_RE.search(text))


async def _build_retained_memory_evidence_context(
    user_message: str,
    *,
    session_id: str = "",
    recent_exchanges: list[dict[str, str]] | None = None,
    conversation_recall_context: str = "",
) -> str:
    """Return auditable evidence for broad retained-memory questions.

    This is deliberately evidence, not prose. The visible reply still comes
    from CognitiveEngine, but it must choose from transcript/durable-memory
    records or admit the gap instead of treating plausible continuity as proof.
    """

    if not _is_retained_memory_evidence_request(user_message):
        return ""

    lines: list[str] = [
        "scope=retained_memory_evidence.v1",
        "rule=Use only the evidence below for remembered-session claims. If it does not support the claim, say the memory is not verified.",
    ]

    if conversation_recall_context:
        lines.append("source=conversation_recall")
        lines.append(
            _chat_memory_state._clip_conversation_text(conversation_recall_context, limit=900)
        )

    exchanges = list(recent_exchanges or [])
    if not exchanges:
        exchanges = await _chat_memory_state._recent_completed_conversation_exchanges(
            current_user_message=user_message,
            session_id=session_id,
            limit=4,
        )
    if exchanges:
        lines.append("source=recent_completed_transcript")
        for idx, entry in enumerate(exchanges[-4:], start=1):
            user_text = _chat_memory_state._clip_conversation_text(entry.get("user"), limit=220)
            aura_text = _chat_memory_state._clip_conversation_text(entry.get("aura"), limit=260)
            if user_text:
                lines.append(f"turn_{idx}.user={user_text}")
            if aura_text:
                lines.append(f"turn_{idx}.aura={aura_text}")

    durable = await _chat_memory_state._recall_durable_conversation_snippets(user_message, limit=4)
    if durable:
        lines.append("source=durable_memory_search")
        for idx, snippet in enumerate(durable, start=1):
            lines.append(
                f"memory_{idx}={_chat_memory_state._clip_conversation_text(snippet, limit=320)}"
            )

    if len(lines) <= 2:
        lines.append("source=none")
        lines.append(
            "No matching canonical transcript or durable memory record was available for this request."
        )

    return "\n".join(lines)[:3200]


def _is_referential_followup_request(user_message: str) -> bool:
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text or len(text) > 120:
        return False
    try:
        from core.conversation.action_episode import is_action_episode_question

        if is_action_episode_question(user_message):
            return True
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.action_outcome_language", exc)
    try:
        from core.runtime.turn_analysis import looks_like_deep_mind_probe

        if looks_like_deep_mind_probe(user_message):
            return False
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Deep mind probe classifier unavailable: %s", exc)
    if any(marker in text for marker in _REFERENTIAL_FOLLOWUP_MARKERS):
        return True
    if _REFERENTIAL_FOLLOWUP_RE.search(text):
        return True
    tokens = set(re.findall(r"\b\w+\b", text))
    return ("question" in tokens or "answer" in tokens) and bool(tokens & {"it", "that", "last"})


async def _resolve_referential_followup_anchor(
    user_message: str,
    *,
    session_id: str = "",
) -> str | None:
    if not _is_referential_followup_request(user_message):
        return None

    # Referential ownership is session-local. The general continuity loader is
    # allowed to reach into an earlier session after a restart, but doing that
    # here can bind "that" to an unrelated old question. Resolve from the
    # explicit session ledger first; the ambient ContextVar is only a fallback
    # for direct internal callers that do not own a session id.
    recent_exchanges = await _chat_memory_state._recent_completed_conversation_exchanges(
        current_user_message=user_message,
        session_id=session_id,
        limit=8,
        allow_cross_session=False,
    )
    recent = [
        str(exchange.get("user") or "").strip()
        for exchange in recent_exchanges
        if str(exchange.get("user") or "").strip()
    ]
    if not recent:
        recent = await _gather_recent_user_messages_for_relevance(user_message, limit=8)
    current = str(user_message or "").strip()
    for candidate in reversed(recent):
        candidate_text = str(candidate or "").strip()
        if not candidate_text or candidate_text == current:
            continue
        if _is_referential_followup_request(candidate_text):
            continue
        if len(candidate_text) < 24:
            continue
        return candidate_text
    return None


def _format_traceability_reply(
    *,
    anchor_message: str,
    event: dict[str, Any] | None,
    reason_category: str,
) -> str:
    # Reached at call time: chat.py imports this module, so the other
    # direction cannot be a module-level import.
    from .chat import _classify_traceability_request

    _asks_traceability, asks_reason, asks_example = _classify_traceability_request(anchor_message)

    if event is None:
        if reason_category == "governance rule blocks disclosure":
            return "Reason: governance rule blocks disclosure. I can see recent private traces, but I do not have a safe non-private one I should expose."
        if reason_category == "do not have access":
            return "Reason: I do not have access to a safe live trace for that right now."
        if reason_category == "uncertain":
            return "Reason: I am uncertain which live trace would be the honest one to cite, so I should not invent one."
        return "Reason: the data does not exist in my current rolling trace window."

    timestamp = float(event.get("timestamp") or 0.0)
    timestamp_iso = (
        datetime.fromtimestamp(timestamp, tz=UTC).isoformat() if timestamp > 0.0 else "unknown"
    )
    trace_line = (
        f"Timestamp: {timestamp_iso} | "
        f"Subsystem: {event.get('subsystem') or 'unknown'} | "
        f"EventID: {event.get('event_id') or 'unavailable'} | "
        f"Action: {event.get('action') or 'unknown'} | "
        f"Result: {event.get('result') or 'unknown'} | "
        f"FutureBehavior: {'yes' if bool(event.get('changed_future_behavior')) else 'no'}"
    )

    if asks_example and not asks_reason:
        return trace_line

    preface = (
        "Access scope: I have a rolling runtime trace, not a full lifetime ledger. "
        "I can inspect recent receipts and audit trails, but I should not invent history outside that window."
    )
    return f"{preface}\n{trace_line}"


def _turn_may_concern_perception(user_message: str) -> bool:
    """Whether her recent looking could bear on this turn at all.

    Deliberately generous — being asked about the screen and having no
    perception of it is the blindness this exists to prevent, so the cost
    of a false positive (a few lines of notes she ignores) is much lower
    than a false negative. It is not unconditional: a screen reading has no
    business riding along on arithmetic.
    """
    text = str(user_message or "").strip()
    if not text:
        return False
    try:
        from core.cognition.evidence_relevance import SCREEN_PERCEPTION, wants_evidence

        return wants_evidence(
            text,
            SCREEN_PERCEPTION,
            lexical_floor=lambda candidate: bool(_PERCEPTION_RELEVANCE_RE.search(candidate)),
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "chat",
            exc,
            severity="warning",
            action="routed perception by pattern after semantic relevance failed",
        )
        return bool(_PERCEPTION_RELEVANCE_RE.search(text))


def _append_past_action_record(user_message: object, reply_text: object) -> object:
    """Attach her own receipts when a recall answer does not match them.

    LIVE, 2026-08-10: "Earlier today I asked you to count files in one of your
    own directories ... Without guessing: what was the count? If you don't
    actually have it, say so." — "The count of files in the directory was
    seventeen, if I recall correctly."

    The count was 9, recorded in four verified tool_execution receipts. She was
    told to say so if she did not have it. She had it, and nothing read it.

    Appended only when the reply states none of the recorded numbers, so a
    correct recall passes untouched and a wrong one is answered by the record
    rather than argued with.
    """

    try:
        from core.introspection.self_evidence import (
            asks_about_past_actions,
            concise_past_action_answer,
        )

        if not asks_about_past_actions(user_message):
            return reply_text
        # One line of recorded fact, not the ledger. The full record was
        # 3,300 characters and never reached the person: reply shaping reads a
        # wall of unrelated-looking text as off-topic and strips it, which is
        # the right instinct — the answer to "what was the count" is a number.
        record = str(concise_past_action_answer(user_message) or "").strip()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return reply_text
    if not record:
        return reply_text
    reply = str(reply_text or "")
    recorded_numbers = set(re.findall(r"\d+", record))
    reply_numbers = set(re.findall(r"\d+", reply))
    if recorded_numbers and recorded_numbers & reply_numbers:
        # She already quoted something the receipts support.
        return reply_text
    # When the reply DENIES doing the thing the receipts record, the correction
    # leads. Trailing it produced "I didn't actually execute the file counting
    # command earlier ... From my own receipts, the count was 9" — the true
    # part arriving after the false one, as a footnote to it.
    #
    # Leading also rescues a truncated reply: that same turn was cut off at "If
    # you", and an answer appended to a severed sentence reads as debris.
    denies = _DENIES_THE_ACTION_RE.search(reply) is not None
    truncated = bool(reply.strip()) and reply.strip()[-1] not in ".!?\"')]}"
    if denies or truncated:
        return f"{record}\n\n{reply.strip()}"
    return f"{reply.rstrip()}\n\n{record}"
