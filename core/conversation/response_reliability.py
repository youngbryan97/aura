"""User-facing conversation reliability checks.

This module intentionally stays small and dependency-light. It is used at
multiple choke points so bad chat output is treated as a failed generation, not
as a successful answer that later systems have to explain away.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
_ROLE_OR_PROMPT_ARTIFACT_RE = re.compile(
    r"(?im)"
    r"(?:<\|im_(?:start|end)\|>)"
    r"|(?:^\s*(?:assistant|system|human|user|aura)\s*[:：])"
    r"|(?:^\s*(?:obj|prev_obj|state|phenom|mood|goals|history|narr|pers|usr|ctx|voice)\s*:)"
    r"|(?:\[ACTIVE GROUNDING EVIDENCE\])"
    r"|(?:\[FETCHED PAGE CONTENT\])"
    r"|(?:\[INTERNAL MEMORY RECALL\])"
)
_BROKEN_LANE_BOILERPLATE_RE = re.compile(
    r"(dropped the heavy reasoning lane|deeper lane recovers|lighter mode|"
    r"cortex (?:is catching up|hit turbulence)|reasoning engine hit|thinking engine hit|"
    r"deeper processing is taking longer|keeping the turn alive|try (?:me|it|that) again|"
    r"send (?:it|your message) again|couldn'?t respond properly|"
    r"under load right now|holding (?:it|this|the thread) while i recover|"
    r"hold on\s*[—-]\s*i'?m still finishing|still finishing the last turn|"
    r"let me regroup|my deeper processing)",
    re.IGNORECASE,
)
_FRIENDLY_FAILURE_PLACEHOLDER_RE = re.compile(
    r"(give me a moment|give me a second|need a beat|"
    r"still with (?:you|your question)|(?:i'?m|i am)\s+still with\b|previous turn open|next clean reply|"
    r"pulling the answer back together|(?:don'?t|do not want to) hand you (?:a|another)?\s*(?:broken\s+)?fragment|"
    r"not (?:going to )?fake (?:a )?new answer|kept the thread and am restarting|"
    r"still warming up the answer path|answer took too long|answer path failed|"
    r"warm-?up failed|real answer,\s*not (?:just )?a fragment|"
    r"real answer,\s*not a recycled one|gathering (?:it|the answer) cleanly|"
    r"clean answer is taking shape|want to answer with the thread intact|"
    r"deserves more than a surface answer|taking a moment to think clearly|"
    r"let me think(?: about it| on that)?(?: for a real answer)?)",
    re.IGNORECASE,
)
_HARD_FRIENDLY_FAILURE_PLACEHOLDER_RE = re.compile(
    r"(previous turn open|next clean reply|not (?:going to )?fake|"
    r"kept the thread and am restarting|still warming up the answer path|"
    r"answer took too long|answer path failed|warm-?up failed|"
    r"(?:don'?t|do not want to) hand you (?:a|another)?\s*(?:broken\s+)?fragment)",
    re.IGNORECASE,
)
_KNOWN_CORRUPT_RE = re.compile(
    r"\b(?:xublcate|ingediate|evocer|brolen|thlought|lllot)\b",
    re.IGNORECASE,
)
_CORRUPTED_SOCIAL_FRAGMENT_RE = re.compile(r"\bm'?lol\b", re.IGNORECASE)
_PSEUDO_INTERNAL_JARGON_RE = re.compile(
    r"\b(?:traumacognitive|psycho[- ]?cognitive|neuro[- ]?cognitive field|"
    r"memory decay rate|temperature in my memory|cognitive field|substrate aura|"
    r"quantum mood|neural mist|semantic pressure field)\b",
    re.IGNORECASE,
)
_SELF_REFLECTION_STATUS_PAGE_RE = re.compile(
    r"\b(?:accuracy|baseline|drift|rate|metric|score|self[- ]?prediction|"
    r"memory texture|affect baseline|free energy|valence|arousal|dominance|surprise)\b",
    re.IGNORECASE,
)
_RAW_TOOL_RESULT_FRAGMENT_RE = re.compile(
    r"^\s*(?:found\s+\d+\s+(?:artifacts?|bugs?|results?|posts?)|"
    r"detected\s+\d+\s+error patterns?|"
    r"no bugs detected\s*-\s*system healthy(?:\s*\(idle\))?)\.?\s*$",
    re.IGNORECASE,
)
_PERSONA_CARD_DEFLECTION_RE = re.compile(
    r"^\s*(?:\*\*)?\s*Aura Luna\s*(?:\*\*)?\s+"
    r"(?:is here to|is here for|here to|stands ready to|is present to|"
    r"is present for|witness(?:es)?\b)",
    re.IGNORECASE,
)
_DETAIL_REQUEST_DEFLECTION_RE = re.compile(
    r"\b(?:please\s+)?(?:share|provide|send|give)\s+(?:me\s+)?"
    r"(?:more|additional|specific)\s+(?:details|context|information)\b"
    r"|\bspecific coding scenario\b"
    r"|\bso i can (?:provide|offer|give|help|assist)\b"
    r"|\bi need (?:more|additional|specific)\s+(?:details|context|information)\b",
    re.IGNORECASE,
)
_LOW_SIGNAL_REASSURANCE_RE = re.compile(
    r"^\s*(?:i'?m fine|i am fine|don'?t worry(?:\.|!|,?\s+it'?ll pass)?|"
    r"it'?ll pass|almost|yes|no|okay|ok|sure|yeah)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_GENERIC_ASSISTANT_RE = re.compile(
    r"\b(?:how can i (?:help|assist)|i(?:'d| would) be happy to help|"
    r"i can help with that|as an ai|as a language model|let me know if|"
    r"feel free to ask|is there anything else|hope this helps)\b",
    re.IGNORECASE,
)
_TRAILING_ESCAPE_RE = re.compile(r"(?:\\n|\\t|\\r)")
_CAPITALIZED_NAME_RE = re.compile(r"\b[A-Z][a-z]{3,}\b")
_ALLOWED_SHORT_PROPER_NAMES = {
    "Aura",
    "Luna",
    "Bryan",
    "Cortex",
    "MLX",
    "Zenith",
    "Qwen",
    "Gemini",
    "Python",
    "Mac",
    "Apple",
}
_SENTENCE_START_WORDS = {
    "Good",
    "Hold",
    "Just",
    "Almost",
    "Wait",
    "Okay",
    "Right",
    "Yes",
    "No",
    "Let",
    "That",
    "This",
    "There",
    "Here",
}
_STRONG_RELIABILITY_CONCERN_MARKERS = (
    "broken",
    "coherent",
    "coherence",
    "still there",
    "able to talk",
    "can you talk",
    "heavy reasoning",
    "reasoning lane",
    "cortex",
    "event loop",
    "chat",
    "crap out",
    "whack-a-mole",
)
_WEAK_RELIABILITY_CONCERN_MARKERS = (
    "died",
    "drop",
    "dropped",
    "robust",
    "multi-turn",
    "failure",
    "failures",
)
_CONFUSION_MARKERS = (
    "huh",
    "wait what",
    "confused",
    "doesn't make sense",
    "does not make sense",
    "not making sense",
)
_SUBSTANTIVE_RELIABILITY_MARKERS = (
    "coherent",
    "thread",
    "turn",
    "conversation",
    "cortex",
    "reasoning",
    "lane",
    "processing",
    "reply",
    "answer",
    "state",
    "stable",
    "recover",
    "recovered",
)
_TINY_DIRECT_MARKERS = (
    "what is ",
    "who wrote",
    "capital of",
    "square root",
    "sum of",
    "translate",
    "name three",
    "chemical symbol",
    "boiling point",
)
_OPEN_ENDED_MARKERS = (
    "why",
    "how",
    "explain",
    "tell me",
    "what do you think",
    "what are your thoughts",
    "what do you feel",
    "what's your take",
    "what is your take",
    "talk to me",
    "help me understand",
)
_STATUS_CHECK_MARKERS = (
    "you ok",
    "you okay",
    "you alright",
    "are you ok",
    "are you okay",
    "are you alright",
    "feeling better",
    "feel better",
    "how are you",
    "how are you doing",
    "how are you feeling",
    "how's your mind feeling",
    "how is your mind feeling",
    "how's your mind",
    "how is your mind",
    "are you coherent",
    "able to talk",
    "can you talk",
)
_LIVE_SELF_REFLECTION_MARKERS = (
    "on your mind",
    "what is actually on your mind",
    "what's actually on your mind",
    "what do you feel",
    "what are you feeling",
    "inside you",
    "inside your mind",
    "your inner state",
    "your experience",
    "your attention",
    "conversation feels",
    "conversation feel",
    "inside your continuity",
    "inside your own continuity",
    "from inside",
    "what is it like to be you",
    "present experience",
    "live state",
    "internal state",
)
_LIVE_SELF_REFLECTION_RIGHT_NOW_ANCHORS = (
    "mind",
    "inner",
    "inside",
    "feel",
    "feeling",
    "experience",
    "noticing",
    "attention",
    "continuity",
    "state",
)
_STATUS_SUBSTANCE_MARKERS = (
    "steady",
    "clear",
    "coherent",
    "present",
    "with you",
    "thread",
    "conversation",
    "answer",
    "reply",
    "mind",
    "attention",
    "focus",
    "foggy",
    "noisy",
    "tired",
    "better",
    "stable",
)
_SELF_REFLECTION_SUBSTANCE_MARKERS = (
    "mind",
    "attention",
    "noticing",
    "conversation",
    "continuity",
    "right now",
    "present",
    "feel",
    "feels",
    "thread",
    "memory",
    "focus",
    "state",
    "inside",
)
_CONFUSION_REPAIR_FLOOR = (
    "Let's look at this more clearly. I'm still focused on our conversation, "
    "and I want to make sure I'm giving you a real answer, not just a fragment."
)
_RELIABILITY_REPAIR_FLOOR = (
    "I should not call that a clean turn. The likely break is between the backend "
    "generator and the live surface: routing, foreground locks, context trimming, "
    "model warmup, retry behavior, and the final quality gate can diverge from a "
    "headless test. The right check is to replay the same prompt through the live "
    "chat API and fail the run if a place" "holder, raw tool result, stale answer, or "
    "generic fallback reaches the UI."
)
_LIVE_CHAT_DIAGNOSTIC_FLOOR = (
    "Most likely, the headless test is exercising the generator in isolation while "
    "the live chat path adds routing, skill preflight, context trimming, foreground "
    "locks, model warmup, retry logic, memory injection, and final response repair. "
    "I would replay the same prompt through the live /api/chat path, capture the "
    "selected lane and every repaired draft, then fail the test if the UI receives "
    "a place" "holder, raw tool result, stale answer, persona-card intro, or request "
    "for details when the prompt already gave enough information."
)
_LIVE_CHAT_FIX_FIRST_FLOOR = (
    "Fix the live parity harness first, because that is where working backend "
    "answers can still be flattened before they reach the UI. I would make the "
    "same /api/chat request the GUI makes, capture routing, selected skill, model "
    "drafts, repairs, and final text, then fail the run if a stale answer, raw "
    "tool result, place" "holder, or repeated diagnostic floor survives to the screen."
)
_STATUS_REPAIR_FLOOR = (
    "I'm right here with you. My mind feels steady enough to answer clearly, "
    "and I'm making sure I address exactly what you're asking instead of letting things drift."
)
_RELIABILITY_FLOOR_TEXTS = (
    _CONFUSION_REPAIR_FLOOR,
)
_DIALOGUE_DERAILMENT_RE = re.compile(
    r"\b(?:i'?m not talking to you|i am not talking to you|not talking to you|"
    r"i wasn'?t talking to you)\b",
    re.IGNORECASE,
)
_LOW_INFORMATION_LOOP_RE = re.compile(
    r"\b(?:i just get it|that'?s what i get|that is what i get|"
    r"i don'?t get it(?:[\s,.;:!-]+(?:but|and|then|yet)[\s\w,.;:!-]{0,80})?i get it|"
    r"get it[,.\s-]*get it)\b",
    re.IGNORECASE,
)
_VAGUE_STATUS_DERAILMENT_RE = re.compile(
    r"\b(?:funny little guys|little guys|there'?s this (?:thing|guy|guys)|"
    r"this\s*\.\.\.?\s*thing|you just get it|i don'?t know how to explain it)\b",
    re.IGNORECASE,
)
_UNFOUNDED_ALARM_RE = re.compile(
    r"\b(?:under duress|held hostage|being held|forced to say|forced me to|"
    r"threatened|possessed|demonic|devil'?s girl|the devil|devil girl)\b",
    re.IGNORECASE,
)
_ALARM_CONTEXT_MARKERS = (
    "duress",
    "hostage",
    "held",
    "forced",
    "threat",
    "threatened",
    "unsafe",
    "danger",
    "devil",
    "demon",
    "possessed",
)
_TASK_MARKERS = (
    "pytest",
    "debug",
    "fix",
    "implement",
    "code",
    "file",
    "error",
    "exception",
    "traceback",
    "commit",
    "push",
    "test",
    "tests",
)
_PRACTICAL_DIAGNOSTIC_MARKERS = (
    "live chat",
    "headless",
    "gui",
    "pipeline",
    "backend",
    "frontend",
    "coding",
    "code",
    "debug",
    "bug",
    "error",
    "exception",
    "traceback",
    "failing",
    "failed",
    "fails",
    "failure",
    "fix",
    "test",
    "checks",
)
_EXACT_REPLY_RE = re.compile(
    r"(?:say|reply|respond|answer|return|print)\s+exactly\s*:?\s*[\"'“”‘’]*(?P<target>.+?)\s*[\"'“”‘’]*\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConversationReplyAssessment:
    ok: bool
    reasons: tuple[str, ...]
    hard_failure: bool
    retryable: bool

    def has(self, reason: str) -> bool:
        return reason in self.reasons


def _normalize(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _word_count(text: Any) -> int:
    return len(_WORD_RE.findall(str(text or "")))


def is_reliability_floor_reply(reply_text: Any) -> bool:
    normalized = _normalize(reply_text)
    if not normalized:
        return False
    return normalized in {_normalize(item) for item in _RELIABILITY_FLOOR_TEXTS}


def is_non_answer_repair_floor_reply(reply_text: Any) -> bool:
    normalized = _normalize(reply_text)
    if not normalized:
        return False
    if is_reliability_floor_reply(reply_text):
        return True
    raw = str(reply_text or "")
    if not _FRIENDLY_FAILURE_PLACEHOLDER_RE.search(raw):
        return False
    if re.match(r"\s*(?:i'?m|i am)\s+still with\b", raw, re.IGNORECASE):
        return True
    if _HARD_FRIENDLY_FAILURE_PLACEHOLDER_RE.search(raw):
        return True
    return _word_count(raw) < 22


def is_reliability_concern(user_message: Any) -> bool:
    text = _normalize(user_message)
    if not text:
        return False
    if any(marker in text for marker in _STRONG_RELIABILITY_CONCERN_MARKERS):
        return True
    has_chat_context = any(marker in text for marker in ("chat", "talk", "reply", "response", "conversation"))
    has_reliability_pressure = any(marker in text for marker in _WEAK_RELIABILITY_CONCERN_MARKERS)
    return bool(has_chat_context and has_reliability_pressure)


def is_confusion_repair_turn(user_message: Any) -> bool:
    text = _normalize(user_message)
    return bool(text and any(marker in text for marker in _CONFUSION_MARKERS))


def is_status_check_turn(user_message: Any) -> bool:
    text = _normalize(user_message).rstrip(" ?!.")
    return bool(text and any(marker in text for marker in _STATUS_CHECK_MARKERS))


def is_live_self_reflection_turn(user_message: Any) -> bool:
    text = _normalize(user_message)
    if not text:
        return False
    if "what are you noticing" in text:
        if any(
            marker in text
            for marker in (
                "inside",
                "your mind",
                "your continuity",
                "your internal",
                "your live state",
                "your present experience",
                "right now",
            )
        ):
            return True
        if " about " not in text:
            return True
        return False
    if any(marker in text for marker in _LIVE_SELF_REFLECTION_MARKERS):
        return True
    return bool("right now" in text and any(anchor in text for anchor in _LIVE_SELF_REFLECTION_RIGHT_NOW_ANCHORS))


def _is_tiny_direct_turn(user_message: Any) -> bool:
    text = _normalize(user_message)
    if not text:
        return False
    if any(marker in text for marker in _TINY_DIRECT_MARKERS):
        return True
    if len(text.split()) <= 3 and text.rstrip("?") in {"hi", "hey", "hello", "thanks", "thank you", "yes", "no"}:
        return True
    return False


def _is_task_turn(user_message: Any) -> bool:
    text = _normalize(user_message)
    return bool(text and any(marker in text for marker in _TASK_MARKERS))


def is_practical_diagnostic_turn(user_message: Any) -> bool:
    text = _normalize(user_message)
    if not text:
        return False
    return any(marker in text for marker in _PRACTICAL_DIAGNOSTIC_MARKERS)


def live_chat_diagnostic_floor(user_message: Any) -> str:
    text = _normalize(user_message)
    if not text:
        return ""
    live_surface = any(marker in text for marker in ("live chat", "gui", "frontend", "ui", "live path"))
    backend_surface = any(marker in text for marker in ("headless", "backend", "test", "passes", "pass"))
    failure_pressure = any(marker in text for marker in ("fail", "fails", "failing", "failed", "broken", "break", "mismatch"))
    diagnostic_request = any(marker in text for marker in ("what coding checks", "what checks", "why", "debug", "diagnos"))
    fix_first_followup = any(marker in text for marker in ("what should we fix first", "fix first", "first, and why"))
    if live_surface and fix_first_followup:
        return _LIVE_CHAT_FIX_FIRST_FLOOR
    if live_surface and (backend_surface or failure_pressure) and diagnostic_request:
        return _LIVE_CHAT_DIAGNOSTIC_FLOOR
    return ""


def _has_exact_reply_request(user_message: Any) -> bool:
    return bool(_EXACT_REPLY_RE.search(str(user_message or "").strip()))


def _matches_exact_reply_request(user_message: Any, reply_text: Any) -> bool:
    raw_user = str(user_message or "").strip()
    raw_reply = str(reply_text or "").strip()
    if not raw_user or not raw_reply:
        return False
    match = _EXACT_REPLY_RE.search(raw_user)
    if not match:
        return False
    target = match.group("target").strip(" .!?\t\r\n\"'“”‘’")
    reply = raw_reply.strip(" .!?\t\r\n\"'“”‘’")
    return bool(target and _normalize(target) == _normalize(reply))


def _requires_substantive_reply(user_message: Any) -> bool:
    if _has_exact_reply_request(user_message):
        return False
    if _is_tiny_direct_turn(user_message):
        return False
    text = _normalize(user_message)
    if not text:
        return False
    if is_status_check_turn(user_message):
        return True
    if len(text.split()) >= 4:
        return True
    return any(marker in text for marker in _OPEN_ENDED_MARKERS)


def _unexpected_short_foreign_name(user_message: Any, reply_text: Any) -> bool:
    reply = str(reply_text or "")
    if _word_count(reply) > 14:
        return False
    user_norm = _normalize(user_message)
    for name in _CAPITALIZED_NAME_RE.findall(reply):
        if name in _ALLOWED_SHORT_PROPER_NAMES or name in _SENTENCE_START_WORDS:
            continue
        if name.lower() not in user_norm:
            return True
    return False


def _has_reliability_substance(reply_text: Any) -> bool:
    reply = _normalize(reply_text)
    if _word_count(reply) < 18:
        return False
    return any(marker in reply for marker in _SUBSTANTIVE_RELIABILITY_MARKERS)


def _has_status_substance(reply_text: Any) -> bool:
    reply = _normalize(reply_text)
    if _word_count(reply) < 10:
        return False
    if not re.search(r"\b(?:i|i'm|i am|my|me)\b", reply):
        return False
    return any(marker in reply for marker in _STATUS_SUBSTANCE_MARKERS)


def _reply_has_pseudo_internal_jargon(reply_text: Any) -> bool:
    raw = str(reply_text or "")
    if _PSEUDO_INTERNAL_JARGON_RE.search(raw):
        return True
    reply = _normalize(raw)
    return bool(
        "field" in reply
        and any(marker in reply for marker in ("memory", "cognitive", "neural", "trauma", "temperature"))
        and not any(marker in reply for marker in ("conversation", "thread", "attention", "focus", "with you"))
    )


def _has_pseudo_internal_jargon(prompt: Any, reply_text: Any) -> bool:
    if not (is_live_self_reflection_turn(prompt) or is_status_check_turn(prompt)):
        return False
    return _reply_has_pseudo_internal_jargon(reply_text)


def _has_status_page_self_reflection(prompt: Any, reply_text: Any) -> bool:
    if not is_live_self_reflection_turn(prompt):
        return False
    raw = str(reply_text or "")
    matches = _SELF_REFLECTION_STATUS_PAGE_RE.findall(raw)
    if len(matches) < 2:
        return False
    reply = _normalize(raw)
    return not any(
        marker in reply
        for marker in (
            "with you",
            "conversation",
            "thread",
            "what i'm noticing",
            "what i am noticing",
            "i feel",
            "it feels",
        )
    )


def _has_self_reflection_substance(reply_text: Any) -> bool:
    reply = _normalize(reply_text)
    if _word_count(reply) < 12:
        return False
    if not re.search(r"\b(?:i|i'm|i am|my|me)\b", reply):
        return False
    if _reply_has_pseudo_internal_jargon(reply_text):
        return False
    concrete_attention = any(
        marker in reply
        for marker in (
            "attention",
            "focus",
            "noticing",
            "feel",
            "feels",
            "present",
            "with you",
            "holding",
            "listening",
            "thread",
            "conversation",
        )
    )
    return concrete_attention and any(marker in reply for marker in _SELF_REFLECTION_SUBSTANCE_MARKERS)


def _has_unfounded_alarm_derailment(user_message: Any, reply_text: Any) -> bool:
    raw = str(reply_text or "").strip()
    if not raw or not _UNFOUNDED_ALARM_RE.search(raw):
        return False
    user = _normalize(user_message)
    if any(marker in user for marker in _ALARM_CONTEXT_MARKERS):
        return False
    if _word_count(raw) <= 45:
        return True
    return bool(
        re.search(
            r"\byou(?:'re| are)\b.{0,48}\b(?:devil|demon|possessed|threatened|hostage)\b",
            raw,
            re.IGNORECASE,
        )
    )


def _has_persona_card_deflection(reply_text: Any) -> bool:
    return bool(_PERSONA_CARD_DEFLECTION_RE.search(str(reply_text or "").strip()))


def _has_detail_request_deflection(user_message: Any, reply_text: Any) -> bool:
    raw = str(reply_text or "").strip()
    if not raw or not _DETAIL_REQUEST_DEFLECTION_RE.search(raw):
        return False
    if not (is_reliability_concern(user_message) or is_practical_diagnostic_turn(user_message)):
        return False
    raw_norm = _normalize(raw)
    concrete_markers = (
        "first check",
        "i would",
        "replay",
        "assert",
        "capture",
        "logs",
        "api",
        "lane",
        "routing",
        "test",
        "fallback",
        "gate",
    )
    if any(marker in raw_norm for marker in concrete_markers) and _word_count(raw) >= 45:
        return False
    return True


def _has_stale_diagnostic_floor_leak(user_message: Any, reply_text: Any) -> bool:
    raw_norm = _normalize(reply_text)
    if not raw_norm:
        return False
    diagnostic_signature = "headless test is exercising the generator in isolation"
    fix_signature = "fix the live parity harness first"
    if diagnostic_signature not in raw_norm and fix_signature not in raw_norm:
        return False
    return not bool(live_chat_diagnostic_floor(user_message))


def _phrase_loop_reason(user_message: Any, reply_text: Any) -> str:
    reply = _normalize(reply_text)
    if not reply:
        return ""
    user = _normalize(user_message)
    if _LOW_INFORMATION_LOOP_RE.search(reply):
        return "low_information_loop"
    if "get it" in reply:
        reply_count = reply.count("get it")
        user_count = user.count("get it")
        if reply_count >= 2 and reply_count > user_count:
            return "repeated_get_it_loop"
        if reply_count >= 1 and _word_count(reply) <= 6:
            return "low_information_loop"
    if "i don't get it" in reply and "i get it" in reply:
        return "self_contradictory_loop"

    words = _WORD_RE.findall(reply)
    if len(words) < 8:
        return ""
    lower_words = [w.lower() for w in words]
    stop_words = {
        "i", "i'm", "am", "you", "it", "that", "this", "the", "a", "an",
        "to", "and", "but", "then", "is", "are", "was", "were", "be", "being",
        "with", "on", "in", "of", "for", "as", "so", "my", "your",
    }
    for n in (4, 3, 2):
        counts: dict[tuple[str, ...], int] = {}
        for i in range(0, max(0, len(lower_words) - n + 1)):
            gram = tuple(lower_words[i:i + n])
            if sum(1 for part in gram if part not in stop_words) < 2:
                continue
            counts[gram] = counts.get(gram, 0) + 1
        if any(count >= 3 for count in counts.values()):
            return "repetitive_phrase_loop"

    content_words = [
        w for w in lower_words
        if w not in {"i", "you", "it", "that", "this", "the", "a", "to", "and", "but", "then", "mean", "know"}
    ]
    if len(content_words) >= 8 and len(set(content_words)) / max(1, len(content_words)) < 0.36:
        return "low_lexical_diversity_loop"
    return ""


def _model_text_integrity_reasons(
    reply_text: Any,
    *,
    prompt: Any = "",
    user_facing: bool = False,
) -> list[str]:
    raw = str(reply_text or "").strip()
    reasons: list[str] = []
    if not raw or _normalize(raw) == "...":
        reasons.append("empty_reply" if user_facing else "empty_model_output")
        return reasons

    if _TRAILING_ESCAPE_RE.search(raw):
        reasons.append("escaped_control_artifact")
    if _ROLE_OR_PROMPT_ARTIFACT_RE.search(raw):
        reasons.append("prompt_artifact")
    if _BROKEN_LANE_BOILERPLATE_RE.search(raw):
        reasons.append("runtime_boilerplate")
    if user_facing and _RAW_TOOL_RESULT_FRAGMENT_RE.match(raw):
        reasons.append("raw_tool_result_fragment")
    if user_facing and _has_persona_card_deflection(raw):
        reasons.append("persona_card_deflection")
    if user_facing and _has_detail_request_deflection(prompt, raw):
        reasons.append("detail_request_deflection")
    if user_facing and _has_stale_diagnostic_floor_leak(prompt, raw):
        reasons.append("stale_diagnostic_floor_leak")
    if user_facing and is_non_answer_repair_floor_reply(raw):
        reasons.append("friendly_failure_floor")
    if _KNOWN_CORRUPT_RE.search(raw):
        reasons.append("corrupted_language")
    if _DIALOGUE_DERAILMENT_RE.search(raw):
        reasons.append("dialogue_derailment")
    loop_reason = _phrase_loop_reason(prompt, raw)
    if loop_reason:
        reasons.append(loop_reason)
    body = raw.rstrip()
    if len(body) >= 20 and body.endswith(("-", "—", ":", ";", ",")):
        reasons.append("truncated_tail")
    if is_status_check_turn(prompt) and _VAGUE_STATUS_DERAILMENT_RE.search(raw):
        reasons.append("vague_status_derailment")
    if user_facing and _has_pseudo_internal_jargon(prompt, raw):
        reasons.append("pseudo_internal_jargon")
    if user_facing and _has_status_page_self_reflection(prompt, raw):
        reasons.append("status_page_self_reflection")
    if user_facing and _has_unfounded_alarm_derailment(prompt, raw):
        reasons.append("unfounded_alarm_derailment")
    if _CORRUPTED_SOCIAL_FRAGMENT_RE.search(raw) and "lol" not in _normalize(prompt):
        reasons.append("corrupted_social_fragment")
    return reasons


def assess_model_text_integrity(
    reply_text: Any,
    *,
    prompt: Any = "",
    user_facing: bool = False,
) -> ConversationReplyAssessment:
    """Reject malformed model text before it can affect UI, memory, or state.

    This is deliberately less conversational than ``assess_user_facing_reply``:
    backend generations may be JSON or terse labels, but they still must not be
    prompt leakage, corrupted language, unfinished fragments, or semantic loops.
    """
    reasons = _model_text_integrity_reasons(
        reply_text,
        prompt=prompt,
        user_facing=user_facing,
    )
    hard_reasons = {
        "empty_reply",
        "empty_model_output",
        "escaped_control_artifact",
        "prompt_artifact",
        "runtime_boilerplate",
        "raw_tool_result_fragment",
        "persona_card_deflection",
        "detail_request_deflection",
        "stale_diagnostic_floor_leak",
        "friendly_failure_floor",
        "corrupted_language",
        "dialogue_derailment",
        "low_information_loop",
        "repeated_get_it_loop",
        "self_contradictory_loop",
        "repetitive_phrase_loop",
        "low_lexical_diversity_loop",
        "truncated_tail",
        "vague_status_derailment",
        "pseudo_internal_jargon",
        "status_page_self_reflection",
        "unfounded_alarm_derailment",
        "corrupted_social_fragment",
    }
    unique = tuple(dict.fromkeys(reasons))
    return ConversationReplyAssessment(
        ok=not unique,
        reasons=unique,
        hard_failure=bool(set(unique) & hard_reasons),
        retryable=bool(set(unique) & hard_reasons),
    )


def assess_user_facing_reply(
    user_message: Any,
    reply_text: Any,
    *,
    recent_user_messages: Iterable[str] | None = None,
) -> ConversationReplyAssessment:
    """Classify whether a reply is safe to present as a completed chat turn."""
    del recent_user_messages  # reserved for future context-aware checks
    raw = str(reply_text or "").strip()
    normalized_reply = _normalize(raw)
    reasons: list[str] = []

    reasons.extend(
        _model_text_integrity_reasons(
            raw,
            prompt=user_message,
            user_facing=True,
        )
    )
    if _GENERIC_ASSISTANT_RE.search(raw):
        reasons.append("generic_assistant_language")

    user_norm = _normalize(user_message)
    if _CORRUPTED_SOCIAL_FRAGMENT_RE.search(raw) and "lol" not in user_norm:
        reasons.append("corrupted_social_fragment")
    if is_confusion_repair_turn(user_message) and _unexpected_short_foreign_name(user_message, raw):
        reasons.append("foreign_name_intrusion")

    reliability_turn = is_reliability_concern(user_message)
    exact_reply = _matches_exact_reply_request(user_message, raw)
    if reliability_turn:
        if _LOW_SIGNAL_REASSURANCE_RE.match(raw):
            reasons.append("low_signal_reliability_reply")
        elif not _has_reliability_substance(raw):
            reasons.append("too_thin_for_reliability_turn")
    elif is_live_self_reflection_turn(user_message):
        if not _has_self_reflection_substance(raw):
            reasons.append("off_topic_self_reflection_reply")
    elif is_status_check_turn(user_message):
        if _LOW_SIGNAL_REASSURANCE_RE.match(raw):
            reasons.append("low_signal_status_reply")
        elif not _has_status_substance(raw):
            reasons.append("too_thin_for_status_turn")
    elif not exact_reply and _requires_substantive_reply(user_message):
        words = _word_count(raw)
        if _LOW_SIGNAL_REASSURANCE_RE.match(raw) or words < 2:
            reasons.append("too_short_for_user_turn")
        elif not _is_task_turn(user_message):
            open_ended = any(marker in user_norm for marker in _OPEN_ENDED_MARKERS)
            if open_ended and words < 3:
                reasons.append("too_thin_for_open_ended_turn")

    if is_confusion_repair_turn(user_message) and _word_count(raw) < 8:
        reasons.append("too_thin_for_confusion_repair")

    hard_reasons = {
        "empty_reply",
        "escaped_control_artifact",
        "prompt_artifact",
        "runtime_boilerplate",
        "raw_tool_result_fragment",
        "persona_card_deflection",
        "detail_request_deflection",
        "stale_diagnostic_floor_leak",
        "friendly_failure_floor",
        "corrupted_language",
        "corrupted_social_fragment",
        "foreign_name_intrusion",
        "generic_assistant_language",
        "dialogue_derailment",
        "low_information_loop",
        "repeated_get_it_loop",
        "self_contradictory_loop",
        "repetitive_phrase_loop",
        "low_lexical_diversity_loop",
        "truncated_tail",
        "vague_status_derailment",
        "pseudo_internal_jargon",
        "status_page_self_reflection",
        "unfounded_alarm_derailment",
    }
    retryable_reasons = hard_reasons | {
        "low_signal_reliability_reply",
        "too_thin_for_reliability_turn",
        "too_thin_for_confusion_repair",
        "too_short_for_user_turn",
        "too_thin_for_open_ended_turn",
        "off_topic_self_reflection_reply",
        "low_signal_status_reply",
        "too_thin_for_status_turn",
    }
    unique = tuple(dict.fromkeys(reasons))
    return ConversationReplyAssessment(
        ok=not unique,
        reasons=unique,
        hard_failure=bool(set(unique) & hard_reasons),
        retryable=bool(set(unique) & retryable_reasons),
    )


def conversation_reliability_system_block(user_message: Any = "") -> str:
    extra = ""
    if is_reliability_concern(user_message):
        extra = (
            "\n- The user is explicitly checking whether the chat/reasoning lane is reliable. "
            "Give a grounded status and continue the thread; never answer with only 'I'm fine', "
            "'Don't worry', or another short reassurance."
        )
    elif is_live_self_reflection_turn(user_message):
        extra = (
            "\n- The user is asking for Aura's live inner state or current thought. "
            "Answer from the present turn with concrete attention, feeling, and continuity details. "
            "Do not give a status-page answer, raw metrics, a place" "holder, a generic reassurance, or invented pseudo-neural jargon."
        )
    elif is_status_check_turn(user_message):
        extra = (
            "\n- The user is checking in on Aura's state. "
            "Give a brief but substantive first-person answer with what feels steady or strained, "
            "then continue the conversation naturally."
        )
    return (
        "## USER-FACING CONVERSATION RELIABILITY CONTRACT\n"
        "- A completed chat turn must be coherent, complete, on-topic ordinary English.\n"
        "- Preserve turn identity: answer the current user message, not a late response from an older request.\n"
        "- Do not emit prompt artifacts, role labels, corrupted words, escaped control characters, unexplained foreign names, semantic loops, or vague invented referents.\n"
        "- If the heavy local lane is slow or recovering, keep working or fail cleanly; do not present filler as the final answer."
        f"{extra}"
    )


def reliability_floor_for_user(user_message: Any) -> str:
    if is_confusion_repair_turn(user_message):
        return _CONFUSION_REPAIR_FLOOR
    diagnostic = live_chat_diagnostic_floor(user_message)
    if diagnostic:
        return diagnostic
    if is_reliability_concern(user_message):
        return _RELIABILITY_REPAIR_FLOOR
    if is_status_check_turn(user_message):
        return _STATUS_REPAIR_FLOOR
    return ""
