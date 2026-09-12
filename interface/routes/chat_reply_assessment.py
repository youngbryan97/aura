"""Whether a draft is worth serving, said as a named fault rather than a score.

Is it about what was asked, is it the same answer to a different question, does
it end where it meant to, is there context in it that was never meant to be
read. Each check answers one of those and nothing else, because a single
quality number is a thing a repair cannot act on — and every repair in
chat_reply_repair.py is written against one of these names.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only; chat.py imports this module
    from .chat import _ReplyQualitySnapshot

import re
from collections.abc import Sequence
from typing import Any

from core.runtime.errors import record_degradation
from interface.routes import chat_conversation_repair as _chat_conversation_repair  # noqa: E402
from interface.routes import chat_desktop_repair as _chat_desktop_repair  # noqa: E402
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

from .chat_lane_bookkeeping import (  # noqa: E402
    _gather_recent_user_messages_for_relevance,
    _normalize_response_body,
    _response_fingerprint,
)


def _evaluate_reply_topicality(
    user_message: str,
    reply_text: str,
    *,
    recent_user_messages: list[str] | None = None,
) -> tuple[bool, str]:
    # Imported here rather than at module level: the module these
    # came from imports this one. A call-time import also still
    # sees a test's patch of the original.
    from .chat import (
        _CONTEXTUAL_RELEVANCE_BRIDGE_MARKERS,
        _CONTEXTUAL_RELEVANCE_DRIFT_MARKERS,
        _asked_about_her_own_workings,
        _looks_like_unrequested_content_review,
    )

    reply = str(reply_text or "").strip()
    if not reply:
        return False, ""

    review_drift, review_reason = _looks_like_unrequested_content_review(user_message, reply)
    if review_drift:
        return True, review_reason

    anchors = set()
    for message in recent_user_messages or [user_message]:
        anchors.update(_chat_conversation_repair._extract_topic_tokens(message))

    reply_tokens = _chat_conversation_repair._extract_topic_tokens(reply)
    lowered_reply = _chat_memory_state._normalize_user_message(reply)
    if _chat_desktop_repair._is_contextual_relevance_challenge(user_message):
        if any(marker in lowered_reply for marker in _CONTEXTUAL_RELEVANCE_BRIDGE_MARKERS):
            return False, ""
        if any(marker in lowered_reply for marker in _CONTEXTUAL_RELEVANCE_DRIFT_MARKERS):
            return True, "contextual_relevance_miss"
        if anchors and reply_tokens and not anchors.intersection(reply_tokens):
            return True, "contextual_relevance_miss"

    if not anchors or len(reply_tokens) < 16:
        return False, ""

    if anchors & reply_tokens:
        return False, ""

    concrete_reply_tokens = {token for token in reply_tokens if len(token) >= 5}
    if len(concrete_reply_tokens) < 12:
        return False, ""

    # Everything above this line measures ABSENCE — the reply borrowed no
    # vocabulary from the question. That is not evidence of drift, and treating
    # it as evidence cost two correct answers live on 2026-08-04:
    #
    #   "I've always wanted to teach myself physics…"
    #     -> "Start with the basics — kinematics, Newtonian mechanics. Khan
    #         Academy has some great free resources…"     BLOCKED
    #   "Do you ever get tired of being asked how you are?"
    #     -> "Not really. The question is a social lubricant, and I enjoy the
    #         interaction…"                                BLOCKED
    #
    # Both are in the durable transcript; the person got "I couldn't get a
    # clear enough answer together" instead. An expert answer names the
    # subfields rather than echoing the word, and a polar answer answers by
    # saying "not really" — the better the reply, the less it repeats.
    #
    # So the block now needs PRESENCE of the failure it exists for: a reply
    # about the runtime's own operation instead of about anything asked.
    try:
        from core.conversation.reply_subject import (
            answers_polar_question,
            assess_subject_alignment,
            assess_subject_drift,
        )
    except ImportError as exc:  # pragma: no cover - import wiring failure
        # A check that could not run has not passed. Blocking here would
        # discard the reply on the strength of a measurement nobody made.
        record_degradation("chat.reply_subject", exc, severity="warning")
        return False, ""

    if answers_polar_question(user_message, reply):
        return False, ""

    if _asked_about_her_own_workings(user_message):
        # A reply about the runtime's own operation is what this block exists
        # to catch, and it is the answer when the question was about her own
        # operation. LIVE 2026-08-26: "what are you actually able to do on
        # this machine that you could not do a month ago" — she wrote five
        # hundred and ninety characters about exactly that, the block called
        # it a foreign topic, and the person got "I couldn't get a clear
        # enough answer together."
        #
        # The drift check is given the reply alone and cannot know what was
        # asked, so the question has to be read here or not at all.
        return False, ""

    drift = assess_subject_drift(reply)
    if not drift.drifted:
        return False, ""
    alignment = assess_subject_alignment(
        user_message,
        reply,
        recent_thread=recent_user_messages,
    )
    if alignment.aligned:
        return False, ""

    return True, "foreign_topic_burst"


def _is_actionably_stale_response(user_message: str, text: str) -> bool:
    """Return whether repetition is stale for this request's output contract.

    An explicit exact-reply contract can legitimately produce identical bytes on
    consecutive turns. Exempt only a response that exactly satisfies the parsed
    target; ordinary repetition and conditional/disjunctive prompts retain the
    normal stale-response protection.
    """
    from .chat import (
        _is_stale_repeated_response,
    )

    if not _is_stale_repeated_response(text):
        return False
    try:
        from core.conversation.response_reliability import requested_exact_reply_target

        exact_target = requested_exact_reply_target(user_message)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.output_contract", exc)
        return True
    if exact_target:
        # An explicit exact-reply contract keeps its original, stricter rule:
        # only the response that exactly satisfies the parsed target is exempt.
        return str(text or "").strip() != exact_target

    # Saying the same thing twice about the same subject is CONSISTENCY, not
    # staleness. The failure this guard exists for is a reply that repeats even
    # though the question moved on — and the runtime already computes exactly
    # that, as `same_answer_diff_prompt`, right beside this. It just was not
    # allowed to settle anything.
    #
    # MEASURED live 2026-08-04. She held a real 8-turn conversation with
    # ChatGPT, wrote the memory record, summarised it — and then, asked "so
    # what did you two actually talk about?", produced the same account again
    # and was degraded for it:
    #
    #   Response confidence: degraded (stale=True, same_answer_diff_prompt=False,
    #   off_topic=False, semantic_glitch=False, assessment=, streak=2)
    #
    # `assessment=` is empty: nothing was wrong with the reply. The person
    # asked a follow-up about the thing she had just done and got "I couldn't
    # get a clear enough answer together" over a memory she was holding.
    if not _is_same_answer_different_prompt(user_message, text):
        logger.debug("Repetition is consistent with the question asked; not treating it as stale.")
        return False
    return True


def _is_same_answer_different_prompt(user_message: str, text: str) -> bool:
    """Detect when different user prompts are getting the same response."""
    from .chat import (
        _conversation_quality_lock,
        _conversation_quality_state_locked,
        _fuzzy_similar,
        _is_referential_followup_request,
        _same_repair_prompt_class,
    )

    if _is_referential_followup_request(user_message):
        return False
    try:
        from core.conversation.response_reliability import is_operational_status_turn

        if is_operational_status_turn(user_message):
            return False
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Operational-status same-answer bypass unavailable: %s", exc)
    user_fp = _response_fingerprint(user_message)
    response_body = _normalize_response_body(text)
    if not user_fp or not response_body:
        return False
    with _conversation_quality_lock:
        response_pairs = tuple(_conversation_quality_state_locked().recent_response_pairs)
    for prev_user, prev_resp in response_pairs:
        if prev_user == user_fp:
            continue
        if _is_referential_followup_request(prev_user):
            continue
        if _same_repair_prompt_class(prev_user, user_fp):
            continue
        if _same_live_self_reflection_prompt_class(prev_user, user_fp):
            continue
        # Near-paraphrase follow-ups can legitimately receive the same answer.
        if _fuzzy_similar(prev_user, user_fp):
            continue
        prev_tokens = _chat_conversation_repair._extract_topic_tokens(prev_user)
        current_tokens = _chat_conversation_repair._extract_topic_tokens(user_fp)
        if prev_tokens and current_tokens:
            overlap = len(prev_tokens & current_tokens)
            smaller = min(len(prev_tokens), len(current_tokens))
            if smaller >= 4 and overlap >= 4 and (overlap / smaller) >= 0.8:
                continue
            response_tokens = _chat_conversation_repair._extract_topic_tokens(response_body)
            prev_response_tokens = _chat_conversation_repair._extract_topic_tokens(prev_resp)
            current_specific = current_tokens - prev_tokens
            prev_specific = prev_tokens - current_tokens
            if (
                len(current_specific & response_tokens) >= 2
                and len(prev_specific & prev_response_tokens) >= 2
            ):
                continue
        if prev_resp == response_body or _fuzzy_similar(prev_resp, response_body):
            return True
    return False


def _ends_where_it_meant_to(said: str, stop_reason: str = "") -> tuple[str, bool]:
    """The reply back to its last finished sentence, and whether it was cut.

    The ladder has twenty-five seconds and the model spends them, so a long
    answer stops wherever the clock did — LIVE 2026-08-30, a good answer about
    a memory leak ended at "2. **Acc". Serving that is worse than serving less
    of it: the reader cannot tell a truncated thought from a confused one, and
    the half-word says the machine broke rather than that time ran out.

    The generator says why it stopped, and that is the fact. Reading it from
    the shape of the text instead guesses, and it guesses wrong on the two
    things people write most: a list whose last item has no full stop, and a
    line of code.
    """
    from .chat import (
        _A_DANGLING_MARKER,
        _A_SENTENCE_ENDS,
        _FINISHED,
    )

    text = str(said or "").rstrip()
    if not text:
        return text, False
    said_why = str(stop_reason or "").strip().lower()
    if said_why in _FINISHED:
        return text, False
    if _A_DANGLING_MARKER.search(text):
        # A reply ending on a bare "1." ends in a full stop and is not
        # finished: that is the number of a point she never got to.
        trimmed = _A_DANGLING_MARKER.sub("", text).rstrip()
        return (trimmed or text), True
    if not said_why and text.endswith(_A_SENTENCE_ENDS):
        # Nothing said why it stopped, so the shape is all there is.
        return text, False
    # Back up to the last place a sentence actually finished. Code fences and
    # list items count: a fenced block that closed is a finished thought.
    ends = [text.rfind(mark) for mark in (".", "!", "?", "\u2026", "```")]
    at = max(ends)
    if at <= 0:
        return text, True
    whole = text[: at + (3 if text[at:].startswith("```") else 1)].rstrip()
    # A trailing "3." is the number of a point she never got to, not a
    # sentence. Cutting to the last full stop lands on one whenever the answer
    # was a numbered list, which is what a long answer usually is.
    whole = _A_DANGLING_MARKER.sub("", whole).rstrip()
    if len(whole) < len(text) // 2:
        # Trimming would throw away most of what she said. Keep it and say so.
        return text, True
    return whole, True


def _looks_semantically_glitched(user_message: str, reply_text: Any) -> tuple[bool, str]:
    """Catch short, visibly derailed replies that pass surface identity checks."""
    from .chat import (
        _SOFT_REPAIRABLE_REPLY_SHAPE_REASONS,
    )

    try:
        from core.conversation.response_reliability import assess_user_facing_reply

        assessment = assess_user_facing_reply(user_message, reply_text)
        if _reply_assessment_requires_repair(assessment):
            hard_reasons = [
                reason
                for reason in (assessment.reasons or ())
                if reason not in _SOFT_REPAIRABLE_REPLY_SHAPE_REASONS
            ]
            if hard_reasons:
                return True, hard_reasons[0]
    except (ImportError, RuntimeError, TypeError, ValueError, AttributeError) as exc:
        logger.debug("Conversation reliability assessment unavailable: %s", exc)

    user_text = _chat_memory_state._normalize_user_message(user_message)
    reply = _chat_memory_state._normalize_user_message(str(reply_text or ""))
    if not reply or reply == "…":
        return True, "empty_reply"

    if "heidi" in reply and "heidi" not in user_text:
        return True, "foreign_name_intrusion"
    if re.search(r"\bm'?lol\b", reply) and "lol" not in user_text:
        return True, "corrupted_social_fragment"

    try:
        from core.phases.dialogue_policy import contains_corrupted_language

        if contains_corrupted_language(str(reply_text or "")):
            return True, "corrupted_language"
    except (ImportError, RuntimeError, TypeError, ValueError, AttributeError) as exc:
        logger.debug("Dialogue corruption check unavailable: %s", exc)

    return False, ""


def _strip_user_visible_context_leaks(reply_text: Any) -> str:
    """Remove internal conversation/context protocol blocks from user-visible text."""
    from .chat import (
        _USER_VISIBLE_CONTEXT_LEAK_MARKERS,
        _USER_VISIBLE_CONTEXT_LEAK_RE,
        _drop_context_leak_lines,
    )


    text = str(reply_text or "").strip()
    if not text:
        return ""
    lower = text.lower()
    cut_at = len(text)
    for marker in _USER_VISIBLE_CONTEXT_LEAK_MARKERS:
        index = lower.find(marker.lower())
        if index >= 0:
            cut_at = min(cut_at, index)
    if cut_at < len(text):
        kept = text[:cut_at].strip()
        if kept:
            return kept
        # The leak was at position 0, so truncating removed the whole reply.
        #
        # LIVE 2026-08-17: "what's on my screen right now?" was served as a
        # bare "…". The cortex had produced a 172-character answer, the screen
        # reading had reached it, and this stripper cut from index 0 — the
        # caller's `or "…"` then turned an authored answer into an ellipsis.
        # An ellipsis is not an answer; it is the shape of one.
        #
        # Cutting from the marker is right when there is text before it. When
        # there is not, drop the marker LINE and keep the rest, which preserves
        # the answer while still removing the protocol block.
        salvaged = _drop_context_leak_lines(text)
        if salvaged:
            logger.warning(
                "Context-leak strip would have emptied a %d-char reply; kept %d "
                "chars by dropping the marker lines instead.",
                len(text),
                len(salvaged),
            )
            return salvaged
        record_degradation(
            "chat.context_leak_strip",
            RuntimeError("stripping context leaks emptied a non-empty reply"),
            action="served nothing rather than a reply that was entirely protocol",
        )
        return ""
    cleaned = _USER_VISIBLE_CONTEXT_LEAK_RE.sub("", text).strip()
    return cleaned


def _strip_unexpected_cjk_artifacts(user_message: str, reply_text: Any) -> str:
    from .chat import (
        _CJK_PUNCT_RE,
        _CJK_SCRIPT_RE,
        _has_unexpected_cjk,
    )

    reply = str(reply_text or "").strip()
    if not reply or not _has_unexpected_cjk(user_message, reply):
        return reply

    def _cleanup_fragment(text: str) -> str:
        cleaned = _CJK_SCRIPT_RE.sub(" ", text)
        cleaned = _CJK_PUNCT_RE.sub(" ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
        return cleaned.strip(" -—")

    sentence_parts = [
        part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", reply) if part.strip()
    ]
    filtered_parts = []
    for part in sentence_parts:
        if not _CJK_SCRIPT_RE.search(part):
            filtered_parts.append(part)
            continue
        cleaned_part = _cleanup_fragment(part)
        if len(cleaned_part) >= 18 and re.search(r"[A-Za-z]{3}", cleaned_part):
            filtered_parts.append(cleaned_part)
    cleaned = " ".join(filtered_parts).strip()
    if len(cleaned) >= max(24, int(len(reply) * 0.45)):
        return re.sub(r"\s+", " ", cleaned).strip()

    cleaned_chars = _cleanup_fragment(reply)
    return cleaned_chars if len(cleaned_chars) >= 24 else reply


def _original_reply_is_safe_to_surface(
    user_message: str,
    text: str,
    *,
    identity_collapse: bool = False,
    unexpected_cjk: bool,
    objective_parrot: bool,
    off_topic: bool,
    truncated_tail: bool,
    semantic_glitch: bool,
) -> bool:
    """Check whether the original model text is safer than another generation."""
    from .chat import (
        _SEARCH_SNIPPET_PATTERNS,
    )


    candidate = str(text or "").strip()
    if len(candidate) < 16 or candidate == "…":
        return False
    if _INTERNAL_STATE_PATTERNS.search(candidate) or _PROMPT_ARTIFACT_PATTERNS.search(candidate):
        return False
    if _SEARCH_SNIPPET_PATTERNS.search(candidate):
        return False
    if (
        identity_collapse
        or unexpected_cjk
        or objective_parrot
        or off_topic
        or truncated_tail
        or semantic_glitch
    ):
        return False
    return True


async def _measure_reply_quality_candidate(
    user_message: str,
    reply_text: Any,
    *,
    recent_user_messages: Sequence[str] | None = None,
) -> _ReplyQualitySnapshot:
    """Measure one candidate once inside its request.

    Stabilization and terminal admission examine the same unchanged candidate
    back to back. Each pass used to repeat transcript acquisition, semantic
    integrity checks, and the full reliability classifier. Besides delaying
    delivery, the duplicate classifier wrote the same candidate to the turn
    ledger twice. A request-scoped snapshot keeps every check while making the
    candidate identity the unit of work.
    """
    from .chat import (
        _CHAT_REPLY_QUALITY_SNAPSHOTS,
        _ReplyQualitySnapshot,
    )


    user = str(user_message or "")
    reply = str(reply_text or "")
    key = (user, reply)
    cache = _CHAT_REPLY_QUALITY_SNAPSHOTS.get()
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    recent = tuple(
        recent_user_messages
        if recent_user_messages is not None
        else await _gather_recent_user_messages_for_relevance(user)
    )

    is_stale = _is_actionably_stale_response(user, reply)
    is_same_diff = _is_same_answer_different_prompt(user, reply)
    is_off_topic, off_topic_reason = _evaluate_reply_topicality(
        user,
        reply,
        recent_user_messages=list(recent),
    )
    semantic_glitch, semantic_glitch_reason = _looks_semantically_glitched(
        user,
        reply,
    )
    try:
        from core.conversation.response_reliability import assess_user_facing_reply

        reply_assessment = assess_user_facing_reply(
            user,
            reply,
            recent_user_messages=recent,
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Conversation reliability assessment unavailable: %s", exc)
        reply_assessment = None

    snapshot = _ReplyQualitySnapshot(
        recent_user_messages=recent,
        is_stale=is_stale,
        is_same_diff=is_same_diff,
        is_off_topic=is_off_topic,
        off_topic_reason=off_topic_reason,
        semantic_glitch=semantic_glitch,
        semantic_glitch_reason=semantic_glitch_reason,
        reply_assessment=reply_assessment,
    )
    if cache is not None:
        cache[key] = snapshot
    return snapshot
