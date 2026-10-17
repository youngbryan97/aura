"""Response quality metrics, consistency checks, and commitment extraction.

Extracted from interface/routes/chat.py (decomposition pattern: cohesive,
low-entanglement sections move to sibling modules; chat.py re-exports the
names so existing monkeypatch targets keep working — external callers
resolve through chat.py's globals, and intra-section calls live here).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.API.Chat")

_CHAT_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

# ── Response Quality Metrics ─────────────────────────────────
_quality_logger = logging.getLogger("Aura.ResponseQuality")


def _reply_assessment_requires_repair(assessment: Any) -> bool:
    """Return True when canonical chat reliability says the reply is not final."""
    if assessment is None:
        return False
    return bool(
        getattr(assessment, "retryable", False)
        or getattr(assessment, "hard_failure", False)
        or not getattr(assessment, "ok", True)
    )


def _log_response_quality_metrics(
    user_message: str,
    reply_text: str,
    confidence: str,
    stale: bool,
    same_diff: bool,
    off_topic: bool,
) -> None:
    """Log structured quality metrics for every response for offline analysis.

    Writes to a dedicated logger so these can be routed to a file/store
    independently of the main application log.

    Delivery settlement is owned by the outer paired-response boundary after
    journal sealing. Metrics must not claim an answer was delivered before the
    route knows which payload actually reached the client.
    """
    try:
        assessment_reasons: tuple[str, ...] = ()
        try:
            from core.conversation.response_reliability import assess_user_facing_reply

            assessment = assess_user_facing_reply(user_message, reply_text)
            assessment_reasons = tuple(getattr(assessment, "reasons", ()) or ())
            if str(confidence).lower() == "high" and _reply_assessment_requires_repair(assessment):
                confidence = "degraded"
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("Quality metric reliability assessment skipped: %s", exc)

        # Lazy import: chat.py imports this module at load; resolving the
        # live-state helper at call time avoids the cycle while that
        # helper's deep entanglement keeps it in chat.py.
        from interface.routes.chat import _resolve_live_aura_state

        live_state = _resolve_live_aura_state()
        wm_size = 0
        has_summary = False
        coherence = 1.0
        recent_thread: list[str] = []
        ledger_entries = 0
        if live_state:
            cognition = getattr(live_state, "cognition", None)
            wm = getattr(cognition, "working_memory", None)
            wm_size = len(wm) if isinstance(wm, list) else 0
            has_summary = bool(getattr(cognition, "rolling_summary", ""))
            coherence = float(getattr(cognition, "coherence_score", 1.0) or 1.0)
            if isinstance(wm, list):
                recent_thread = [
                    str(m.get("content", "") or "")
                    for m in wm[-8:]
                    if isinstance(m, dict)
                    and str(m.get("role", "")).lower() in ("user", "assistant")
                ]
            ledger = getattr(cognition, "continuity_ledger", None)
            if isinstance(ledger, dict):
                ledger_entries = len(ledger.get("entries", []) or [])

        # `coherence` above is state.cognition.coherence_score — an internal
        # state score that never looks at the conversation. It read 0.814
        # "ok" on the turn before a total non-sequitur. Log it still, but log
        # beside it something that actually measures the thread.
        thread_metrics: dict[str, object] = {}
        try:
            from core.conversation.thread_continuity import assess_thread_continuity

            thread_metrics = assess_thread_continuity(
                user_message, reply_text, recent_thread=recent_thread
            ).as_metrics()
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat_quality.thread_continuity", exc, severity="warning",
                               action="logged quality metrics without a thread-continuity reading",
                               enforce_failure_policy=False)

        _quality_logger.info(
            "📊 quality_metrics | confidence=%s | stale=%s | same_diff=%s | off_topic=%s | "
            "reply_len=%d | user_len=%d | wm_size=%d | has_summary=%s | ledger_entries=%d | "
            "state_coherence=%.3f | thread_abandoned=%s | overlap_turn=%s | overlap_thread=%s | "
            "assessment=%s",
            confidence, stale, same_diff, off_topic,
            len(reply_text or ""), len(user_message or ""),
            wm_size, has_summary, ledger_entries, coherence,
            thread_metrics.get("thread_abandoned", "unknown"),
            thread_metrics.get("overlap_turn", "-"),
            thread_metrics.get("overlap_thread", "-"),
            ",".join(assessment_reasons) or "ok",
        )

        if thread_metrics.get("thread_abandoned"):
            # Loud, because this is the shape a person actually notices: a
            # fluent reply about something else entirely.
            logger.warning(
                "🧵 Reply abandoned the thread (overlap_turn=%s overlap_thread=%s) — "
                "user=%r reply=%r",
                thread_metrics.get("overlap_turn"),
                thread_metrics.get("overlap_thread"),
                (user_message or "")[:120],
                (reply_text or "")[:160],
            )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Quality metric logging skipped: %s", exc)


# ── Consistency Check ─────────────────────────────────────────
_INABILITY_PATTERNS = re.compile(
    r"(i can't|i cannot|i'm unable to|i don't have access to|"
    r"i'm not able to|i lack the ability to)(?:\s+\w+){0,3}\s+"
    r"(browse|search|access|look up|check|find|open|read|visit|navigate|download|fetch)",
    re.IGNORECASE,
)

_COMMITMENT_NEGATION_CLAUSE = re.compile(
    r"\bi\s+(?:do\s+not|don['’]t|have\s+not|haven['’]t|did\s+not|didn['’]t|"
    r"can\s+not|cannot|can['’]t|am\s+not\s+able\s+to|was\s+not\s+able\s+to)\b"
    r"[^.!?\n]{0,220}",
    re.IGNORECASE,
)
_COMMITMENT_WORD = re.compile(r"[a-z0-9][a-z0-9'-]*", re.IGNORECASE)
_COMMITMENT_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "but",
    "can",
    "cannot",
    "could",
    "did",
    "does",
    "doing",
    "done",
    "for",
    "from",
    "have",
    "into",
    "just",
    "later",
    "might",
    "need",
    "not",
    "now",
    "only",
    "should",
    "that",
    "the",
    "their",
    "then",
    "there",
    "they",
    "this",
    "through",
    "until",
    "was",
    "were",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def _commitment_terms(text: str) -> set[str]:
    """Return stable content terms for clause-local commitment comparison."""
    return {
        token
        for token in (match.group(0).lower() for match in _COMMITMENT_WORD.finditer(text or ""))
        if len(token) >= 3 and token not in _COMMITMENT_STOPWORDS
    }


def _negated_clause_contradicts_commitment(reply_text: str, description: str) -> bool:
    """Require a negated clause to refer specifically to the commitment.

    Comparing a whole reply to a persistent commitment made unrelated planning
    answers fail whenever they contained any first-person negation.  Keep the
    guard semantic and local: the actual negated clause must substantially
    overlap the commitment's content terms.
    """
    commitment_terms = _commitment_terms(description)
    if len(commitment_terms) < 2:
        return False
    for match in _COMMITMENT_NEGATION_CLAUSE.finditer(reply_text or ""):
        clause_terms = _commitment_terms(match.group(0))
        if len(clause_terms) < 2:
            continue
        overlap = commitment_terms & clause_terms
        smaller_side = min(len(commitment_terms), len(clause_terms))
        required_overlap = 2 if smaller_side <= 3 else 3
        if len(overlap) >= required_overlap and len(overlap) / smaller_side >= 0.6:
            return True
    return False


def _check_response_consistency(reply_text: str, user_message: str) -> tuple[bool, str]:
    """Check response against known capabilities and commitments.

    Returns (is_consistent, reason).
    """
    # 1. Check for false inability claims — Aura has web access via skills
    if _INABILITY_PATTERNS.search(reply_text or ""):
        try:
            from core.container import ServiceContainer
            cap = ServiceContainer.get("capability_engine", default=None)
            catalog = {}
            if cap and hasattr(cap, "get_catalog"):
                catalog = cap.get_catalog() or {}
            elif cap and hasattr(cap, "get_tool_catalog"):
                tool_catalog = cap.get_tool_catalog(include_inactive=True) or []
                catalog = {
                    str(item.get("name") or ""): {
                        "status": "unavailable" if not bool(item.get("available")) else str(item.get("state") or "ready").lower(),
                    }
                    for item in tool_catalog
                    if isinstance(item, dict) and item.get("name")
                }

            if catalog:
                available = {
                    k for k, v in catalog.items()
                    if isinstance(v, dict) and v.get("status") != "unavailable"
                }
                web_skills = {"web_search", "search_web", "free_search", "grounded_search", "sovereign_browser"}
                desktop_skills = {"computer_use", "os_manipulation", "sovereign_terminal", "sovereign_vision"}
                mentions_web = bool(re.search(r"\b(browse|search|look up|check online|find online|open (?:a )?(?:site|page|url|website))\b", reply_text or "", re.IGNORECASE))
                mentions_desktop = bool(re.search(r"\b(open|control|click|type|use|navigate)\b.{0,40}\b(tab|browser|computer|desktop|screen)\b", reply_text or "", re.IGNORECASE))
                if (mentions_web and web_skills & available) or (mentions_desktop and desktop_skills & available):
                    return False, "false_inability_claim"
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("Skill registry availability check failed: %s", exc)

    # 2. Check for self-contradiction against active commitments
    try:
        from core.agency.commitment_engine import get_commitment_engine
        ce = get_commitment_engine()
        if hasattr(ce, "get_active_commitments"):
            active = ce.get_active_commitments()
            for commitment in (active or [])[:5]:
                if isinstance(commitment, dict):
                    desc = str(commitment.get("description", "") or "").lower()
                else:
                    desc = str(getattr(commitment, "description", "") or "").lower()
                if desc and len(desc) > 10 and _negated_clause_contradicts_commitment(
                    reply_text,
                    desc,
                ):
                    return False, "commitment_contradiction"
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Commitment contradiction check skipped: %s", exc)

    return True, ""


# ── Commitment Extraction ─────────────────────────────────────
#: First-person undertakings — she is saying she will do this thing.
#:
#: Accepting a commitment someone else proposed is the commonest way to make
#: one, and none of these matched it. Live 2026-08-10: asked to hold a deferred
#: task and act on it unprompted, she answered "I accept the commitment and I am
#: able to act on it without further prompting." Nothing was registered, nothing
#: acted, and thirteen minutes later the file did not exist.
_UNDERTAKING_PATTERNS = re.compile(
    r"(?:^|\. |\n)"
    r"(i'll |i will |i'm going to |i won't forget |i promise |"
    r"i'll remember |"
    r"i accept |i've noted |i have noted |noted[,.] |consider it done|"
    r"will do|i'll take care of |i am able to act|i can do that|"
    r"yes[,.]? i'll |yes[,.]? i will |on it[,.]? )"
    r"([^.!?\n]{0,120})",
    re.IGNORECASE | re.MULTILINE,
)

#: Deliberately NOT registered. "we should", "i should" and "next step is"
#: propose a course of action; they do not undertake one, and a proposal that
#: goes unactioned is not a broken promise.
#:
#: These three sat in the registration pattern for as long as it existed, which
#: was harmless only because the function was dead code (it called a method that
#: does not exist). Repairing the function makes them live, and every hedge in
#: conversation — "we should check the logs first" — would become a user-facing
#: commitment that breaks 24 hours later and marks down the reliability figure
#: she reports about herself. That is precisely the flood this ledger was just
#: cleaned of, refilled from a different tap.
#:
#: Nor is "let me": it narrates what the reply then does. Live 17 October every
#: date difference she worked opened "Let me work this out step by step", and
#: each became a commitment due in a day, 49 such rows in her ledger, all
#: "broken" or about to be, and the active ones quoted back to her as things she
#: had undertaken to do.
_PROPOSAL_PATTERNS = re.compile(
    r"(?:^|\. |\n)(next step is |we should |i should )([^.!?\n]{0,120})",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_and_register_commitments(reply_text: str, user_message: str) -> None:
    """Scan the final response for commitment/promise language and register.

    This closes the open-loop tracking gap: when Aura says 'I'll remember X'
    or 'next step is Z', it actually gets tracked.

    LIVE DEFECT, 2026-08-10. That docstring was the whole of it. The function
    called ``ce.add_commitment(...)`` behind ``if not hasattr(ce,
    "add_commitment"): return`` — and CommitmentEngine has no such method. Its
    API is ``commit(description, outcome, deadline_hours, commitment_type)``. So
    the guard was true on every call, the function returned immediately every
    time, and not one commitment has ever been registered from a conversation.
    The gap it claims to close was open the entire time, silently, because a
    hasattr guard on a method that exists nowhere looks exactly like a
    capability that is merely unavailable.

    Found by asking her to hold a deferred task: "some time in the next few
    minutes, without me asking again, write READY into a file." She answered "I
    accept the commitment and I am able to act on it without further prompting."
    Nothing was registered, nothing acted, and the file did not exist thirteen
    minutes later.
    """
    if not reply_text or len(reply_text) < 20:
        return
    try:
        matches = _UNDERTAKING_PATTERNS.findall(reply_text)
        if not matches:
            return
        from core.agency.commitment_engine import (
            CommitmentStatus,
            CommitmentType,
            get_commitment_engine,
        )

        ce = get_commitment_engine()
        request = str(user_message or "").strip()
        # Restating a promise is not making a second one. Without this, holding
        # one open loop across a conversation ("I'll write that file" asked
        # about three times) accrues three rows that must each be fulfilled
        # separately, so two of them break and the reliability figure falls for
        # a promise that was kept.
        open_descriptions = {
            " ".join(str(c.description or "").lower().split())
            for c in ce._commitments.values()
            if c.status == CommitmentStatus.ACTIVE
        }
        for prefix, body in matches[:3]:  # cap at 3 per response
            description = f"{prefix.strip()} {body.strip()}".strip()
            if len(description) < 8:
                continue
            normalized = " ".join(description.lower().split())
            if normalized in open_descriptions:
                logger.debug("Commitment already open, not duplicating: %s", description[:60])
                continue
            open_descriptions.add(normalized)
            # The engine requires an OUTCOME — what would count as this being
            # done. For a promise made in conversation the only honest answer is
            # the request it was made against, so that is what is recorded
            # rather than a restatement of the promise.
            ce.commit(
                description=description,
                outcome=(
                    f"The request this was promised against is satisfied: {request[:240]}"
                    if request
                    else "The promise made in this reply is satisfied."
                ),
                commitment_type=CommitmentType.USER_FACING,
            )
            logger.info("📝 Registered commitment from reply: %s", description[:80])
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation('chat', exc)
        logger.debug("Commitment extraction skipped: %s", exc)


# ── Post-response confidence downgrades ──────────────────────────────


@dataclass(frozen=True)
class ConfidenceDowngrade:
    """A post-response reason to stop calling a reply high-confidence."""

    confidence: str
    reason: str = ""
    lane_warning: str = ""

    @property
    def downgraded(self) -> bool:
        return bool(self.reason)


def assess_post_response_confidence(
    confidence: str,
    *,
    self_consistent: bool,
    inconsistency_reason: str = "",
    used_fallback_lane: bool = False,
    generation_happened_this_turn: bool = False,
    actual_endpoint: str = "",
    desired_endpoint: str = "",
) -> ConfidenceDowngrade:
    """Decide whether a delivered reply may still be called high-confidence.

    Lifted out of ``api_chat`` — a 4,488-line function with 633 branches, where
    this decision had no test of its own and could only be exercised by driving
    a whole HTTP turn. Two independent reasons to downgrade, in the order they
    were applied:

    1. the reply contradicts itself (false inability claims, commitment
       contradictions);
    2. the generation that produced it came from a FALLBACK lane in this turn,
       so the answer is real but not from the mind that was asked.

    Only ``high`` is downgraded. A reply already marked degraded stays degraded;
    nothing here promotes anything.
    """
    if str(confidence) != "high":
        return ConfidenceDowngrade(str(confidence))

    if not self_consistent:
        return ConfidenceDowngrade(
            "degraded", inconsistency_reason or "self_inconsistent"
        )

    if used_fallback_lane and generation_happened_this_turn:
        warning = (
            f"last accepted user generation used {actual_endpoint or 'fallback'} "
            f"instead of desired {desired_endpoint or 'primary'}"
        )
        return ConfidenceDowngrade("degraded", "fallback_lane_generation", warning)

    return ConfidenceDowngrade("high")
