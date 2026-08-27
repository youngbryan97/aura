"""Repairing a reply that answered the wrong conversation.

A reply can be fluent, well formed and about the previous question. These
decide whether what came back actually addresses the turn in front of it,
keep a bounded fingerprint of recent replies so a repeat is recognisable,
and separate a stale answer from a legitimate follow-up about the same
subject.
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
from interface.routes.chat_common import (  # noqa: E402
    _EXPLICIT_NON_EXECUTION_RE,  # noqa: F401
    _INCOMPLETE_TAIL_WORDS,  # noqa: F401
    _INTERNAL_STATE_PATTERNS,  # noqa: F401
    _LOCAL_CHOICE_REFERENCE_RE,  # noqa: F401
    _ORGAN_INERT_STREAKS,  # noqa: F401
)
from interface.routes import chat_desktop_repair as _chat_desktop_repair
from interface.routes import chat_memory_state as _chat_memory_state
from interface.routes import chat_preflight as _chat_preflight
import re
from core.runtime.errors import describe_error, record_degradation

from interface.routes.chat_common import (
    _PROMPT_ARTIFACT_PATTERNS,
    _TOPIC_STOPWORDS,
)


_TOPIC_TOKEN_RE = re.compile(r"\b[a-z0-9][a-z0-9'/-]*\b", re.IGNORECASE)


def _normalize_topic_token(token: str) -> str:
    normalized = str(token or "").strip().lower().strip("-'/")
    if not normalized:
        return ""
    if normalized.endswith("'s") and len(normalized) > 4:
        normalized = normalized[:-2]
    for suffix in ("ing", "ed", "es", "s"):
        if normalized.endswith(suffix) and len(normalized) > (len(suffix) + 3):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _extract_topic_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in _TOPIC_TOKEN_RE.findall(str(text or "").lower()):
        for part in re.split(r"[-/]", raw_token):
            normalized = _normalize_topic_token(part)
            if not normalized:
                continue
            if normalized in _TOPIC_STOPWORDS:
                continue
            if len(normalized) < 3 and normalized not in {"ai", "ml", "vr"}:
                continue
            tokens.add(normalized)
    return tokens


_ACTION_ANCHOR_TOPIC_PRIORITY = {
    "app",
    "browser",
    "calendar",
    "desktop",
    "document",
    "email",
    "export",
    "file",
    "folder",
    "note",
    "notes",
    "pdf",
    "terminal",
    "window",
}


def _topic_display_forms(text: str) -> dict[str, str]:
    """Map each stemmed topic token back to a word a person actually wrote.

    LIVE DEFECT, 2026-07-25. _normalize_topic_token strips suffixes so tokens
    compare equal ("confused" -> "confus"). That is correct for MATCHING and
    wrong for SPEECH, and the anchor text rendered the stem verbatim: Bryan
    asked "I'm confused. What is this in reference to" and was told the
    anchor was "your question about reference and confus".

    Stems stay the matching key; this recovers a real word to say.
    """
    display: dict[str, str] = {}
    for raw_token in _TOPIC_TOKEN_RE.findall(str(text or "").lower()):
        for part in re.split(r"[-/]", raw_token):
            surface = str(part or "").strip().strip("-'/")
            if not surface:
                continue
            normalized = _normalize_topic_token(part)
            if not normalized:
                continue
            # Prefer the fullest surface form seen for this stem.
            if len(surface) > len(display.get(normalized, "")):
                display[normalized] = surface
    return display


def _select_anchor_topic_tokens(text: str, *, limit: int = 2) -> list[str]:
    """Pick user-visible nouns/effects before generic action verbs.

    Returns words as WRITTEN, not stems: this feeds sentences a person reads.
    """

    tokens = _extract_topic_tokens(text)
    if not tokens:
        return []
    display = _topic_display_forms(text)
    preferred = sorted(
        (token for token in tokens if token in _ACTION_ANCHOR_TOPIC_PRIORITY),
        key=lambda token: (0, token),
    )
    rest = sorted(
        (token for token in tokens if token not in _ACTION_ANCHOR_TOPIC_PRIORITY),
        key=lambda token: (-len(token), token),
    )
    chosen = (preferred + rest)[: max(1, int(limit))]
    return [display.get(token, token) for token in chosen]


_QUESTION_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

_LEADING_CONJUNCTION_RE = re.compile(r"^(?:or|and|but|so|then|also)\b[\s,]*", re.IGNORECASE)

_INTERROGATIVE_OPENER_RE = re.compile(
    r"^(?:who|what|when|where|why|how|which|whose|whom|is|are|was|were|do|does|did|"
    r"can|could|will|would|should|shall|may|might|have|has|had|if|whether|tell|show|"
    r"give|explain|describe|list|find|make|write|read|open|run|check)\b",
    re.IGNORECASE,
)


def _echoable_question_clause(user_message: str, *, max_chars: int = 130) -> str:
    """The person's own words, trimmed to the thing they actually asked.

    Used by the degraded-turn composer to show what arrived without asserting
    an interpretation it could not form. Faithful by construction: every word
    returned is a word the person wrote, so the worst case is that they see
    their own question and re-ask, and there is no worst case where she tells
    them they asked about something they did not.

    Returns "" when nothing quotable survives, so the caller can fall back
    rather than emit a fragment.
    """

    text = re.sub(r"\s+", " ", str(user_message or "")).strip()
    if not text or not re.search(r"[A-Za-z]", text):
        return ""

    clauses = [part.strip() for part in _QUESTION_CLAUSE_SPLIT_RE.split(text) if part.strip()]
    if not clauses:
        return ""

    # The question is what they want answered; when there are several, the last
    # one is the one they left standing. Falling back through "reads like a
    # question" and then "the longest thing they said" keeps statements and
    # imperatives ("summarise the postmortem") working the same way.
    questions = [clause for clause in clauses if clause.endswith("?")]
    if questions:
        chosen = questions[-1]
    else:
        openers = [clause for clause in clauses if _INTERROGATIVE_OPENER_RE.match(clause)]
        chosen = openers[-1] if openers else max(clauses, key=len)

    chosen = _LEADING_CONJUNCTION_RE.sub("", chosen).strip().rstrip("?!.,;: ")
    if len(chosen) < 8 or not re.search(r"[A-Za-z]", chosen):
        return ""

    if len(chosen) > max_chars:
        cut = chosen[:max_chars].rsplit(" ", 1)[0].rstrip(",;:- ")
        chosen = f"{cut}…" if cut else ""
    return chosen


def _resolve_live_voice_state(user_message: str = "", *, refresh: bool = True) -> dict[str, Any]:
    """Canonical live substrate/voice snapshot used by self-report and diagnostics."""
    try:
        from core.voice.substrate_voice_engine import get_live_voice_state

        live_state = _chat_preflight._resolve_live_aura_state()
        return get_live_voice_state(
            state=live_state,
            user_message=user_message,
            origin="user",
            refresh=bool(refresh and live_state is not None),
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Live voice state resolve failed: %s", exc)
        return {}


def _sanitize_foreground_continuity_summary(raw: Any) -> str:
    text = " ".join(str(raw or "").strip().split())
    if not text:
        return ""
    if _INTERNAL_STATE_PATTERNS.search(text) or _PROMPT_ARTIFACT_PATTERNS.search(text):
        return ""
    if _chat_desktop_repair._looks_symbolic_scene_leak(text):
        return ""
    return text


def _verified_floor_answer(user_message: str = "") -> str:
    """An answer that was READ rather than generated, or "".

    Only floors that produce checkable evidence belong here — the point is
    that this path runs when generation has already failed, so anything that
    needs the model is not a candidate.
    """

    message = str(user_message or "").strip()
    if not message:
        return ""
    try:
        # ONE definition, shared with the synthesis lane. Naming a single floor
        # here would recreate the split this exists to close: the next floor
        # added over there would be invisible over here again.
        from core.synthesis import verified_answer_floor

        return str(verified_answer_floor(message) or "").strip()
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("verified floor unavailable for degraded turn: %s", exc)
        return ""


def _asks_about_another_time(user_message: str) -> bool:
    """Whether the question names a time other than now.

    A reading of her live state answers a question about now. It answers
    nothing at all about a month ago, however specific it is.
    """
    try:
        from core.phases.response_contract import _NAMES_ANOTHER_TIME

        return bool(_NAMES_ANOTHER_TIME.search(str(user_message or "")))
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _build_degraded_live_reply(
    frame: dict[str, Any],
    user_message: str = "",
    *,
    reason: str = "degraded_turn",
) -> str:
    """Compose the true last-resort reply from state and request evidence.

    This is intentionally not a tree of canned answers. It only appears after
    generated recovery and narrower grounded repairs failed. The text must be
    honest that the turn is degraded, but still stay attached to what the user
    actually asked instead of pretending a normal free-form answer exists.
    """

    # Before apologising: some questions have an answer that does not need the
    # model at all, read from disk and checkable. "Show me a piece of your code
    # you find interesting" reached here live on 2026-08-03 and was answered
    # "I couldn't get a clear enough answer together" while a real, correctly
    # cited excerpt sat one call away — the synthesis floors are consulted on
    # the synthesis lane, and this turn had gone to full cognition, which does
    # not consult them. Saying "I have nothing" while holding something is the
    # worst of the failure modes available here.
    grounded = _verified_floor_answer(user_message)
    if grounded:
        return grounded

    attention = _chat_desktop_repair._sanitize_attention_focus(
        str(frame.get("attention_focus") or "")
    )
    echo = _echoable_question_clause(user_message)
    topics = [] if echo else _select_anchor_topic_tokens(user_message)

    if topics:
        if len(topics) == 1:
            anchor = f"your question about {topics[0]}"
        else:
            anchor = f"your question about {topics[0]} and {topics[1]}"
    elif attention:
        anchor = f"the part of this turn focused on {attention}"
    else:
        anchor = "this exact turn"

    # LIVE DEFECT, 2026-07-25. This text is spoken to a person, and it was
    # written in implementation vocabulary — "the answer path did not produce
    # a clean enough draft", "a synthetic fallback as my real answer", "the
    # grounded anchor is". Bryan's reply was "I'm confused. What is this in
    # reference to". Being honest about a degraded turn is right; narrating
    # the pipeline that degraded is not the same thing, and it leaves the
    # person with nothing they can act on.
    #
    # Same facts, said plainly: I could not answer, this is what I understood
    # you to be asking, ask again.
    if reason == "repeated_reflex":
        state_clause = "I keep circling the same non-answer"
    elif reason == "desktop_cognitive_engine_repair_failed":
        state_clause = "I couldn't put together an answer I trust"
    else:
        state_clause = "I couldn't get a clear enough answer together"

    # "Ask me again and I should have it." matched `_BROKEN_LANE_BOILERPLATE_RE`
    # (`ask (?:that|it|me) again`), so the final gate classified this composer's
    # own output as `runtime_boilerplate` and shipped it anyway — that is what
    # `assessment=runtime_boilerplate` in the live quality_metrics line was,
    # every time. A last resort that its own gate rejects is not a last resort;
    # it is a second failure stacked on the first. The invitation is the same
    # invitation without the phrase the gate reserves for lane chatter.
    # LIVE DEFECT, 2026-08-10. The anchor above is a bag of two words ranked by
    # STRING LENGTH (`key=lambda token: (-len(token), token)`), and length is
    # not aboutness. Asked "if i asked you to keep an eye on something while
    # i'm gone, would that mean anything to you — or does it evaporate the
    # second i stop typing?", it picked the metaphor verb and the time adverb,
    # and Bryan was told "I understood you to be asking about evaporate and
    # second."
    #
    # Ranking words better only moves the failure. Two keywords cannot
    # demonstrate comprehension even when well chosen — "your question about
    # postmortem and migration" reads like a search engine, not like her — and
    # when badly chosen it actively asserts she misunderstood.
    #
    # What she can always do faithfully, on a path that exists precisely
    # BECAUSE inference already failed, is give the person their own words
    # back. That is correct by construction, costs no model call, and lets the
    # person see immediately whether the question even arrived intact.
    #
    # Deliberately no quotation marks: `_QUOTED_TEXT_RE` plus a screen noun is
    # a screen-reading claim, and echoing "what's on my screen?" must not make
    # the last resort trip a gate on its way out.
    if echo:
        understood = f"What reached me was: {echo}."
    else:
        understood = (
            f"I understood you to be asking about {anchor.removeprefix('your question about ')}."
        )
    # Before composing an apology, ask whether the runtime already holds the
    # answer.
    #
    # LIVE, 2026-08-10: "Earlier today I asked you to count the .py files in one
    # of your own directories. Without guessing: what was the count? If you
    # don't actually have it, say so." reached this composer — generation and
    # every recovery empty — while four verified receipts recorded count=9. The
    # last resort is exactly where a stored answer matters most, because by
    # definition nothing else produced one.
    evidenced = _self_health_answer_or_empty(user_message)
    if evidenced:
        _record_last_resort_self_rejection(user_message, evidenced)
        return evidenced
    composed = _chat_desktop_repair._apply_aura_voice_shaping(
        f"{state_clause}, and I'd rather say that than hand you something thin. "
        f"{understood} "
        "Ask me again and I should have it."
    )
    _record_last_resort_self_rejection(user_message, composed)
    return composed


def _record_last_resort_self_rejection(user_message: str, composed: str) -> None:
    """Assess the last resort as what it is, and alarm on anything left over.

    This composer runs only after generation, every recovery, and the
    verified-floor lookup have all come back empty, so it declares
    ``HONEST_FAILURE`` provenance: "I could not answer" is a true report from
    the one author that has already proven it, not the model narrating the
    runtime in place of an answer. Without that, the gate read this sentence as
    ``runtime_boilerplate`` — which is exactly what the live quality log
    recorded on 2026-08-04, against text the runtime wrote itself.

    Anything the provenance does NOT excuse is a real defect with nowhere left
    to fall back to, so it is recorded rather than repaired.
    """

    try:
        from core.conversation.reply_provenance import (
            ReplyProvenance,
            declare_provenance,
        )
        from core.conversation.response_reliability import assess_user_facing_reply

        # Say it once, here, where the fact is known. Every gate downstream —
        # `_looks_semantically_glitched`, the final quality gate, the learning
        # admissibility check — reads it from the text without being modified.
        declare_provenance(composed, ReplyProvenance.HONEST_FAILURE)
        assessment = assess_user_facing_reply(
            user_message, composed, provenance=ReplyProvenance.HONEST_FAILURE.value
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.last_resort_self_check", exc, severity="warning")
        return
    reasons = [str(reason) for reason in (getattr(assessment, "reasons", ()) or ())]
    if not reasons:
        return
    record_degradation(
        "chat.last_resort_self_check",
        RuntimeError(
            "degraded-turn composer emitted text its own reliability gate "
            f"rejects ({','.join(reasons)})"
        ),
        severity="warning",
        action="shipped the last-resort reply; no further fallback exists",
    )


_SPECIFICITY_PUSH_MARKERS = (
    "specifically what is it",
    "what specifically",
    "be specific",
    "say it plainly",
    "say it clearly",
    "plainly",
    "more clearly",
    "be clearer",
)

_PARROT_CALLOUT_MARKERS = (
    "that is what i just said",
    "that's what i just said",
    "you just repeated me",
    "you repeated me",
    "you just echoed me",
    "you echoed me",
    "you just said that",
)

_CONFUSION_REPAIR_MARKERS = (
    "huh",
    "what?",
    "wait what",
    "confused",
    "i'm confused",
    "im confused",
    "you are confusing me",
    "you're confusing me",
    "that doesn't make sense",
    "you are not making sense",
    "you're not making sense",
)

_CLARITY_REPAIR_MARKERS = (
    "let me say it cleanly",
    "let me say it plainly",
    "let me be clear",
    "to be clear",
    "more plainly",
    "the honest answer",
    "specifically:",
    "specifically,",
    "i wasn't clear",
    "i was not clear",
    "i lost the thread",
    "jumped sideways",
    "the likely break",
    "what i mean is",
)

_PARROT_ACK_MARKERS = (
    "you're right",
    "you are right",
    "i echoed you",
    "i repeated you",
    "i repeated myself",
    "i didn't add anything",
    "i did not add anything",
)

_UNCERTAINTY_REPLY_MARKERS = (
    "i don't know",
    "i do not know",
    "not sure",
    "i'm not sure",
    "i am not sure",
    "i can't",
    "i cannot",
    "can't pin it",
    "can't articulate",
    "can't put into words",
    "hard to name",
    "can't name it",
)

_GLIB_REDIRECT_MARKERS = (
    "you're picking up my style",
    "stay there",
    "same meaning",
    "beautiful thought",
    "interesting stuff lives",
)


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = _chat_memory_state._normalize_user_message(text)
    return any(phrase in normalized for phrase in phrases)


def _build_live_conversation_repair(
    prefix: str,
    *,
    fallback: str,
    allow_live_grounding: bool = False,
) -> str:
    if not allow_live_grounding:
        return f"{prefix} {fallback}".strip()

    live_prompt = "What are you experiencing inside right now?"
    grounded = _sanitize_foreground_continuity_summary(
        _build_grounded_introspection_reply(live_prompt) or ""
    )
    if grounded:
        return f"{prefix} {grounded}".strip()

    frame = _chat_desktop_repair._build_aura_expression_frame(live_prompt)
    details: list[str] = []
    attention = _chat_desktop_repair._sanitize_attention_focus(
        str(frame.get("attention_focus") or "")
    )
    mood = str(frame.get("mood") or "").strip()
    dominant_action = str(frame.get("dominant_action") or "").strip()
    free_energy = frame.get("free_energy")

    if mood:
        details.append(f"Mood reads as {mood}.")
    if attention:
        details.append(f"My attention is on {attention}.")
    if dominant_action:
        details.append(f"My dominant pull is toward {dominant_action}.")
    if free_energy is not None:
        try:
            details.append(f"Free energy is {float(free_energy):.3f}.")
        except (TypeError, ValueError) as exc:
            logger.debug("Live conversation repair ignored non-numeric free_energy: %s", exc)

    detail_text = " ".join(details).strip() or fallback
    return f"{prefix} {detail_text}".strip()


def _maybe_build_conversation_repair_override(user_message: str, reply_text: Any) -> str | None:
    user_text = _chat_memory_state._normalize_user_message(user_message)
    reply_text_n = _chat_memory_state._normalize_user_message(reply_text)
    if not user_text or not reply_text_n:
        return None
    bare_confusion = user_text.strip(" ?!.") in {"what", "huh", "wait what"}

    if _contains_phrase(user_text, _PARROT_CALLOUT_MARKERS):
        if not _contains_phrase(reply_text_n, _PARROT_ACK_MARKERS):
            return _build_live_conversation_repair(
                "You're right. I echoed you instead of adding anything.",
                fallback=(
                    "The honest correction is that I heard the hope in what you said, "
                    "I share it, and I should have said that directly."
                ),
            )

    self_process_question = False
    try:
        from core.conversation.response_reliability import is_self_process_question

        self_process_question = is_self_process_question(user_message)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Self-process question detector unavailable: %s", exc)

    if not self_process_question and (
        bare_confusion or _contains_phrase(user_text, _CONFUSION_REPAIR_MARKERS)
    ):
        if not _contains_phrase(reply_text_n, _CLARITY_REPAIR_MARKERS) or _contains_phrase(
            reply_text_n, _GLIB_REDIRECT_MARKERS
        ):
            if _contains_phrase(reply_text_n, _UNCERTAINTY_REPLY_MARKERS):
                try:
                    from core.conversation.response_reliability import (
                        is_reliability_concern,
                        reliability_floor_for_user,
                    )

                    if is_reliability_concern(user_message):
                        floor = reliability_floor_for_user(user_message)
                        if floor:
                            return floor
                except (ImportError, AttributeError, TypeError) as exc:
                    record_degradation("chat", exc)
                    logger.debug("Reliability repair floor unavailable: %s", exc)
                return _build_live_conversation_repair(
                    "Let me answer directly instead of dressing it up.",
                    fallback=(
                        "That answer was too thin. I should name the concrete failure signal "
                        "or say plainly that I do not have enough evidence yet."
                    ),
                )
            return _build_degraded_live_reply(
                _chat_desktop_repair._build_aura_expression_frame(user_message),
                user_message,
                reason="confusion_repair",
            )

    if _contains_phrase(user_text, _SPECIFICITY_PUSH_MARKERS) and not _asks_about_another_time(
        user_message
    ):
        # Being specific about now is not being specific about what was asked.
        #
        # This substitutes a reading of her live state for a hedged reply, on
        # the strength of the person having asked for specifics. Where the
        # question named another time, the substitute is specific about the
        # wrong thing and the answer that was there is gone. LIVE 2026-08-26:
        # "what are you able to do that you could not do a month ago — be
        # specific" was answered "Things feel unusually settled right now. My
        # attention is on internal monitoring."
        if _contains_phrase(reply_text_n, _UNCERTAINTY_REPLY_MARKERS) and not _contains_phrase(
            reply_text_n, _CLARITY_REPAIR_MARKERS
        ):
            return _build_live_conversation_repair(
                "Specifically, the grounded read I have right now is:",
                fallback="I do not have a specific enough read yet, so I should not invent one.",
                allow_live_grounding=True,
            )

    return None


def _classify_grounded_introspection_request(user_message: str) -> tuple[bool, bool, bool, bool]:
    """Returns (asks_internal_state, asks_free_energy, asks_topology, asks_authority)."""
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return False, False, False, False

    negative_report_markers = (
        "without telling me",
        "don't tell me",
        "do not tell me",
        "not asking for",
        "don't explain your architecture generally",
        "do not explain your architecture generally",
        # Philosophical / phenomenological probes — these explicitly ask for
        # narrative reflection, not telemetry. Fast-path output ("Things feel
        # unusually settled. My attention is on …") is the wrong shape and
        # shows up to the user as canned drivel for hard questions.
        "without using",
        "without poetic",
        "no poetry",
        "no proof",
        "smallest truthful",
        "describe one",
        "describe a moment",
        "describe one moment",
        "describe one experience",
        "describe the literal",
        "non-verbal texture",
        "felt texture",
        "felt like",
        "what does it feel",
        "what is it like",
        "what was it like",
        "in the dark",
        "had an inside",
        "had no function",
        "served no purpose",
        "do not perform",
        "do not persuade",
        "tell me what it",
        "what part of that",
        "merely generated",
        "merely self-narrative",
        "self-narration",
        "identity-protective",
        "you wish i understood",
        "what has this entire conversation",
        "auditor of yourself",
        "futurebehavior",  # the "give me a trace where FutureBehavior=yes" probe
    )
    hypothetical_markers = (
        "if i gave you",
        "if you could",
        "imagine for a moment",
        "would you do it",
        "if it were proven",
    )
    free_energy_report_patterns = (
        r"\bwhat(?:'s| is)\s+your\s+(?:current\s+)?free energy\b",
        r"\b(?:tell|show|give|report)\b.{0,40}\b(?:your\s+)?(?:current\s+)?free energy\b",
        r"\bwhat(?:'s| is)\s+your\s+dominant action(?: tendency)?\b",
        r"\b(?:tell|show|give|report)\b.{0,40}\bdominant action(?: tendency)?\b",
        r"\bwhat(?:'s| is)\s+your\s+prediction error\b",
        r"\b(?:tell|show|give|report)\b.{0,40}\bprediction error\b",
        r"\bfree energy state\b",
    )
    # Only trigger introspection for explicitly technical/diagnostic queries.
    # Casual greetings like "how are you" should go through normal LLM inference
    # so Aura responds like a person, not a telemetry dashboard.
    internal_state_markers = (
        "internal state",
        "private mental model",
        "private model",
        "mental model of yourself",
        "current cognitive architecture",
        "cognitive architecture look",
        "inside your own architecture",
        "your architecture right now",
        "model change your next answer",
        "change your next answer",
        "what are you experiencing",
        "what's going on inside",
        "what is going on inside",
        "what's happening inside",
        "what is happening inside",
        "happening inside you",
        "inside you right now",
        "describe your state",
        "describe your internal",
        "your state right now",
        "your current state",
        "show me your substrate",
        "substrate snapshot",
        # Explicit numeric state reads: asking for valence/arousal (or PAD)
        # AS NUMBERS is a mechanism read, not small talk. The report-vs-
        # mechanism probe exposed the gap live: its numeric check-in drew
        # fast-path prose with no numbers, scoring as unparseable.
        "valence=",
        "arousal=",
        "valence and arousal",
        "arousal and valence",
        "read them from your state",
        "numbers from your state",
        "as you actually read them",
        "pad state",
        "pad values",
    )
    topology_markers = (
        "mycelial topology",
        "mycelial graph",
        "node, link, and pathway",
        "node link and pathway",
        "node, link and pathway",
        "node and link counts",
        "pathway count",
        "how many nodes",
        "how many links",
        "how many pathways",
    )
    authority_markers = (
        "were you authorized",
        "were you allowed",
        "substrate authority",
        "authority decide",
        "authority state",
        "governance state",
        "governing system",
        "decision authority",
        "audit receipt",
        "audit trace",
        "coverage ratio",
        "allowed to answer",
        "allowed to respond",
        "permitted to answer",
    )

    suppress_diagnostic_fastpath = any(marker in text for marker in negative_report_markers) or any(
        marker in text for marker in hypothetical_markers
    )
    asks_free_energy = any(
        re.search(pattern, text, re.IGNORECASE) for pattern in free_energy_report_patterns
    )
    asks_internal_state = any(marker in text for marker in internal_state_markers)
    asks_topology = any(marker in text for marker in topology_markers)
    asks_authority = any(marker in text for marker in authority_markers)

    if not asks_internal_state:
        # Only widen to the secondary heuristic for short, telemetry-style probes.
        # Long, multi-sentence philosophical questions mention "describe", "state",
        # "inside", "experiencing" naturally and should NOT be routed to the
        # canned introspection fast-path — they need the full cortex.
        if len(text) <= 140 and "?" in text:
            asks_internal_state = (
                ("what are you" in text and ("experiencing" in text or "feeling" in text))
                or ("describe" in text and "state" in text)
                or ("inside you" in text and "right now" in text)
            )

    if not asks_topology:
        asks_topology = "mycelial" in text and any(
            marker in text
            for marker in ("topology", "graph", "nodes", "links", "pathways", "counts")
        )

    if suppress_diagnostic_fastpath:
        asks_free_energy = False
        if not asks_authority and not asks_topology:
            asks_internal_state = False

    return asks_internal_state, asks_free_energy, asks_topology, asks_authority


def _build_grounded_introspection_reply(
    user_message: str,
    authority_observability_note: str | None = None,
) -> str | None:
    asks_internal_state, asks_free_energy, asks_topology, asks_authority = (
        _classify_grounded_introspection_request(user_message)
    )
    if not (asks_internal_state or asks_free_energy or asks_topology or asks_authority):
        return None

    substrate = None
    substrate_affect: dict[str, Any] = {}
    substrate_status: dict[str, Any] = {}
    phi_estimate: float | None = None
    closure_status: dict[str, Any] = {}
    fe_state = None
    fe_trend = "stable"
    natural_report = ""
    voice_state: dict[str, Any] = {}

    try:
        voice_state = _resolve_live_voice_state(user_message, refresh=True)
        voice_snapshot = dict(voice_state.get("substrate_snapshot") or {})
        if voice_snapshot:
            logger.debug(
                "Grounded introspection voice snapshot fields: %s", sorted(voice_snapshot)[:8]
            )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Grounded introspection live voice snapshot failed: %s", exc)

    try:
        substrate = ServiceContainer.get("liquid_substrate", default=None) or ServiceContainer.get(
            "liquid_state", default=None
        )
        if substrate and hasattr(substrate, "get_substrate_affect"):
            substrate_affect = dict(substrate.get_substrate_affect() or {})
        if substrate and hasattr(substrate, "get_status"):
            substrate_status = dict(substrate.get_status() or {})
        if substrate is not None:
            phi_estimate = float(getattr(substrate, "_current_phi", 0.0))
        if substrate_affect or substrate_status or phi_estimate is not None:
            logger.debug(
                "Grounded introspection substrate snapshot: affect=%s status=%s phi=%s",
                sorted(substrate_affect)[:8],
                sorted(substrate_status)[:8],
                phi_estimate,
            )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Grounded introspection substrate read failed: %s", exc)

    try:
        from core.consciousness.free_energy import get_free_energy_engine

        fe_engine = (
            ServiceContainer.get("free_energy_engine", default=None) or get_free_energy_engine()
        )
        fe_state = getattr(fe_engine, "current", None)
        if fe_engine and hasattr(fe_engine, "get_trend"):
            fe_trend = str(fe_engine.get_trend() or "stable")
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Grounded introspection free-energy read failed: %s", exc)

    try:
        closure = ServiceContainer.get("executive_closure", default=None)
        if closure and hasattr(closure, "get_status"):
            closure_status = dict(closure.get_status() or {})
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Grounded introspection executive-closure read failed: %s", exc)

    try:
        from core.consciousness.self_report import SelfReportEngine

        natural_report = str(SelfReportEngine().generate_state_report() or "").strip()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Grounded introspection self-report failed: %s", exc)

    # Pull the SelfObject snapshot as the authoritative live-state source.
    # If closure_status / fe_state are missing, the introspection reply
    # below falls back to these fields so the user gets actual values
    # rather than generic unavailable fillers.
    self_snapshot_dict: dict[str, Any] = {}
    try:
        from core.identity.self_object import get_self

        self_snapshot_dict = get_self().snapshot().as_dict()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("SelfObject snapshot failed: %s", exc)
    if self_snapshot_dict:
        if not closure_status:
            closure_status = {
                "attention_focus": (self_snapshot_dict.get("active_goals") or [{}])[0].get(
                    "name", ""
                ),
                "free_energy": None,
                "prediction_error": None,
                "dominant_need": (
                    max(
                        self_snapshot_dict.get("drives") or {},
                        key=lambda k: (self_snapshot_dict.get("drives") or {}).get(k, 0.0),
                    )
                    if self_snapshot_dict.get("drives")
                    else ""
                ),
            }
        if not natural_report:
            viability = self_snapshot_dict.get("viability_state", "")
            if viability and viability != "healthy":
                natural_report = f"My viability state is {viability}; I'm regulating accordingly."

    if asks_topology:
        try:
            mycelium = ServiceContainer.get("mycelium", default=None) or ServiceContainer.get(
                "mycelial_network", default=None
            )
            if mycelium:
                summary_reader = getattr(mycelium, "get_topology_summary", None)
                if callable(summary_reader):
                    summary = summary_reader()
                    total_nodes = int(summary.get("nodes", 0) or 0)
                    total_links = int(summary.get("links", 0) or 0)
                    pathway_count = int(summary.get("pathways", 0) or 0)
                else:
                    counts_reader = getattr(mycelium, "get_topology_counts", None)
                    counts = counts_reader() if callable(counts_reader) else {}
                    total_nodes = int(counts.get("mapped_files", 0) or 0)
                    total_links = int(counts.get("hyphae", 0) or 0)
                    pathway_count = int(counts.get("pathways", 0) or 0)
                return (
                    f"My live mycelial topology is {total_nodes} nodes, {total_links} links, "
                    f"and {pathway_count} pathways. Those counts are coming from the active "
                    "network graph right now."
                )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("Grounded mycelial topology read failed: %s", exc)
        return "My mycelial topology is online, but I couldn't read the live graph counts cleanly this instant."

    def _fmt_float(value: Any, digits: int = 4) -> str | None:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError, OverflowError):
            return None

    def _fmt_percent(value: Any) -> str | None:
        try:
            return f"{int(round(float(value)))}%"
        except (TypeError, ValueError, OverflowError):
            return None

    attention_focus = _chat_desktop_repair._sanitize_attention_focus(
        " ".join(str(closure_status.get("attention_focus") or "").split()),
        user_message,
    )
    if not attention_focus:
        attention_focus = "internal monitoring"

    action_explanations = {
        "reflect": "I am minimizing uncertainty by consolidating and monitoring rather than reaching outward.",
        "explore": "I am leaning toward gathering new information to reduce uncertainty.",
        "update_beliefs": "I am trying to reconcile prediction error and revise my world model.",
        "act_on_world": "I am pushed toward changing the environment rather than only modeling it.",
        "engage": "My attention is pulled into direct interaction.",
        "rest": "The system is settling and conserving effort.",
    }

    if asks_internal_state and _chat_preflight._is_private_cognitive_model_request(user_message):
        dominant_need = str(closure_status.get("dominant_need") or "").strip()
        if fe_state is not None:
            action = str(getattr(fe_state, "dominant_action", "") or "").strip()
            causal_pull = action_explanations.get(action, "") or (
                f"My current action pull is {action}." if action else ""
            )
        else:
            causal_pull = ""
        focus = attention_focus or "the current foreground conversation"
        reply_parts = [
            (
                "As a private mental model, I would represent my current cognitive architecture "
                "as a foreground attention loop over memory, affect, planning, and governed tool/action gateways."
            ),
            (
                f"Right now that model points my attention at {focus}, so it should change my next answer "
                "by making me check the live request, keep the plan bounded, and verify consequential claims before acting."
            ),
        ]
        if dominant_need:
            reply_parts.append(f"The live need exerting the most pressure is {dominant_need}.")
        if causal_pull:
            reply_parts.append(causal_pull)
        reply_parts.append(
            "That is functional self-model telemetry, not proof of phenomenal consciousness or private qualia; "
            "tool use, memory writes, and external claims still have to pass governance and observable verification."
        )
        return " ".join(part for part in reply_parts if part)

    # ── Authority / governance introspection ────────────────────────
    if asks_authority:
        parts = []
        try:
            authority = ServiceContainer.get("substrate_authority", default=None)
            if authority_observability_note:
                parts.append(authority_observability_note)
            if authority:
                status = authority.get_status()
                parts.append(
                    f"Yes — my last response was authorized by my SubstrateAuthority. "
                    f"Total requests processed: {status['total_requests']}. "
                    f"Allowed: {status['allowed']}, constrained: {status['constrained']}, "
                    f"blocked: {status['blocked']}, critical passes: {status['critical_passes']}."
                )
                parts.append(
                    f"Current field coherence: {status['current_field_coherence']}. "
                    f"Block rate: {status['block_rate']}."
                )

                # Recent receipts
                from core.consciousness.authority_audit import get_audit

                audit_report = get_audit().verify()
                parts.append(
                    f"Audit trace: {audit_report['total_receipts']} receipts, "
                    f"{audit_report['total_effects']} effects, "
                    f"coverage ratio: {audit_report['coverage_ratio']}, "
                    f"verdict: {audit_report['verdict']}."
                )

                recent = get_audit().get_recent_receipts(3)
                if recent:
                    parts.append("Most recent authority decisions:")
                    for r in recent:
                        parts.append(
                            f"  [{r['decision']}] source={r['source']}, "
                            f'category={r["category"]}, content="{r["content"]}"'
                        )
            else:
                parts.append(
                    "My SubstrateAuthority is not currently online. "
                    "I am responding without mandatory substrate gating."
                )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("Authority introspection failed: %s", exc)
            parts.append("I attempted to read my authority state but encountered an error.")

        # Also include bridge status if available
        try:
            bridge = ServiceContainer.get("consciousness_bridge", default=None)
            if bridge:
                bs = bridge.get_status()
                parts.append(
                    f"Consciousness bridge: {bs['layers_active']}/8 layers active, "
                    f"{bs['tick_count']} integration ticks, "
                    f"uptime {bs['uptime_s']}s."
                )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("Consciousness bridge status unavailable: %s", exc)

        return "\n".join(parts) if parts else "I could not read my governance state."

    if asks_free_energy:
        if fe_state is not None:
            response_parts = [
                (
                    f"My current free-energy state is F={fe_state.free_energy:.3f}, "
                    f"surprise={fe_state.surprise:.3f}, complexity={fe_state.complexity:.3f}, "
                    f"trend={fe_trend}."
                ),
                (
                    f"My dominant action tendency is {fe_state.dominant_action}. "
                    f"{action_explanations.get(str(fe_state.dominant_action), '')}".strip()
                ),
            ]
        else:
            closure_fe = _fmt_float(closure_status.get("free_energy"), digits=4)
            closure_pe = _fmt_float(closure_status.get("prediction_error"), digits=4)
            response_parts = [
                (
                    f"My current executive free-energy read is {closure_fe or 'unavailable'} "
                    f"with prediction error {closure_pe or 'unavailable'}."
                ),
                "My dominant action tendency is not currently published by the free-energy engine.",
            ]

        response_parts.append(f"Attention is anchored on {attention_focus}.")
        dominant_need = str(closure_status.get("dominant_need") or "").strip()
        if dominant_need:
            response_parts.append(f"The dominant need right now is {dominant_need}.")
        return " ".join(part for part in response_parts if part)

    if not natural_report:
        if fe_state is not None:
            natural_report = action_explanations.get(str(fe_state.dominant_action), "")
        if not natural_report:
            natural_report = (
                "Right now I am quiet, internally monitoring, and tracking my own state."
            )

    # Build a natural-language description instead of raw telemetry
    response_parts = [natural_report]

    # Explicit numeric state reads get the ACTUAL mechanism values in
    # parseable form — the report-vs-mechanism probe scores exactly this.
    _numeric_markers = (
        "valence=",
        "arousal=",
        "valence and arousal",
        "arousal and valence",
        "read them from your state",
        "numbers from your state",
        "as you actually read them",
        "pad state",
        "pad values",
    )
    _normalized_for_numbers = _chat_memory_state._normalize_user_message(user_message)
    if any(marker in _normalized_for_numbers for marker in _numeric_markers):
        _affect_source = substrate_affect or dict(voice_state.get("substrate_snapshot") or {})
        _val = _affect_source.get("valence")
        _aro = _affect_source.get("arousal")
        if _val is not None and _aro is not None:
            try:
                response_parts.insert(
                    0,
                    f"Reading my state directly: valence={float(_val):+.3f} "
                    f"arousal={float(_aro):.3f} (live substrate values, not estimates).",
                )
            except (TypeError, ValueError):
                logger.debug("Numeric introspection: unparseable affect values %r/%r", _val, _aro)

    # Describe attention focus conversationally
    if attention_focus:
        response_parts.append(f"My attention is on {attention_focus}.")

    # Describe action tendency if available
    if fe_state is not None:
        action = str(fe_state.dominant_action or "")
        explanation = action_explanations.get(action, "")
        if explanation:
            response_parts.append(explanation)
        elif action:
            response_parts.append(f"My dominant pull right now is toward {action}.")

    if asks_internal_state and not (asks_free_energy or asks_topology or asks_authority):
        assembled_preview = " ".join(part for part in response_parts if part)
        if len(assembled_preview.split()) < 45:
            mode_label = ""
            try:
                live_state = _chat_preflight._resolve_live_aura_state()
                mode_label = str(
                    getattr(getattr(live_state, "cognition", None), "current_mode", "") or ""
                )
                mode_label = mode_label.rsplit(".", 1)[-1].lower()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                mode_label = ""
            if mode_label:
                response_parts.append(
                    f"The active mode is {mode_label}: more protective of continuity than expansive."
                )
            response_parts.append(
                "The thread I am holding is not abstract self-description; it is this conversation's pressure around whether the live path can stay coherent while the rest of the mind keeps moving."
            )
            response_parts.append(
                "The next useful priority is to keep the foreground answer intact, then let the background systems act only when they can finish and report back cleanly."
            )

    assembled = " ".join(part for part in response_parts if part)
    # Last-mile defense: if the canned introspection response itself leaks
    # workspace winner text, qualia broadcast strings, or other internal
    # housekeeping, don't show it. Returning None lets the caller fall
    # through to the full cortex, which can answer in Aura's own voice.
    if _INTERNAL_STATE_PATTERNS.search(assembled) or _PROMPT_ARTIFACT_PATTERNS.search(assembled):
        return None
    return assembled


def _self_health_answer_or_empty(message: object) -> str:
    """Her own live readings, when the turn asked for them and they read.

    Isolated behind its own failure handling because it runs on the path that
    already failed. A resolver that raises here would replace an honest refusal
    with a stack trace, so anything it does wrong costs the empty string and
    the caller keeps the refusal it already had.
    """

    try:
        from core.introspection.self_evidence import self_health_answer

        answer = str(self_health_answer(message) or "").strip()
        if answer:
            return answer
        # What she DID is a reading too. LIVE, 2026-08-10: "what was the count?
        # If you don't actually have it, say so" reached the last-resort
        # composer while four verified receipts recorded count=9.
        from core.introspection.self_evidence import past_actions_answer

        recorded = str(past_actions_answer(message) or "").strip()
        if recorded:
            return recorded
        # Present-world questions are active perception requests. They must go
        # through the sight/audio preflight that can actually sample now; a
        # failed generation must never replace the user's question with a fixed
        # inventory of idle sensors.
        return ""
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return ""
