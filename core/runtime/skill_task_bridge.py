from __future__ import annotations

import re
from collections.abc import Iterable

from core.utils.intent_normalization import normalize_memory_intent_text

_ACTION_VERBS = (
    "open",
    "launch",
    "run",
    "execute",
    "click",
    "tap",
    "press",
    "type",
    "write",
    "enter",
    "search",
    "look up",
    "read",
    "inspect",
    "check",
    "navigate",
    "visit",
    "save",
    "download",
    "remember",
    "store",
    "recall",
    "report",
    "return",
    "come back",
    "use",
    # Changing something is an action, and so is being told not to.
    #
    # LIVE, 2026-08-22: "...tell me what's actually wrong? don't just make the
    # test pass." routed to the desktop actuation lane, which spent
    # thirty-seven seconds failing to compile AppleScript for a Python
    # question. The mutation detector reads `make` as a change; this list,
    # which decides whether a change was NEGATED, did not have it — nor fix,
    # change, edit, patch or modify. So a prohibition read as an instruction,
    # and the one sentence telling her not to touch the code was the sentence
    # that sent her to the lane that touches things.
    #
    # Two vocabularies answering "is this an action" and "was it negated" will
    # always drift apart. Everything the mutation detectors recognise belongs
    # here.
    "make",
    "fix",
    "change",
    "edit",
    "modify",
    "patch",
    "update",
    "create",
    "delete",
    "remove",
    "rename",
    "move",
    "copy",
    "append",
    "install",
    "send",
    "post",
    "submit",
    "buy",
    "book",
)

_ACTION_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(verb).replace(r"\ ", r"\s+") for verb in _ACTION_VERBS)
    + r")\b",
    re.IGNORECASE,
)

_CHAIN_PATTERNS = (
    r"\band then\b",
    r"\bthen\b",
    r"\bafter that\b",
    r"\bafterwards\b",
    r"\bstep by step\b",
    r"\bmultiple steps?\b",
    r"\bseries of\b",
    r"\bkeep using\b",
    r"\bcontinue\b",
    r"\bfrom there\b",
    r"\bso that\b",
)

_REPORT_PATTERNS = (
    r"\bcome back\b",
    # "report back"-style chain markers only. A bare \breport\b hijacked
    # "Report your current valence…" into the TaskEngine (observed live).
    r"\breport\s+back\b",
    r"\breport\b[^.?!\n]{0,40}\b(?:what happened|results?|findings|outcome)\b",
    r"\btell me what happened\b",
    r"\blet me know\b",
    r"\bshow me\b",
    r"\bconfirm\b",
    r"\bverify\b",
    r"\bmake sure\b",
    r"\bactually interact\b",
    r"\bon your own\b",
)

_DESKTOP_PATTERNS = (
    r"\bon my computer\b",
    r"\bon my screen\b",
    r"\bdesktop\b",
    r"\bnotes\b",
    r"\bterminal\b",
    r"\bapp\b",
    r"\bapplication\b",
    r"\bwindow\b",
    r"\btab\b",
    r"\bmouse\b",
    r"\bkeyboard\b",
)

_SINGLE_STEP_PATTERNS = (
    r"^\s*(?:search|look up|google|browse|read|inspect|check)\b",
    r"^\s*(?:open|launch)\s+(?:https?://|\w+\.\w+)",
    r"^\s*(?:what(?:'s| is)\s+the\s+time|what(?:'s| is)\s+today(?:'s)? date)\b",
)

_REPORTBACK_PATTERNS = (
    r"\bthis is what i did\b",
    r"\bhere(?:'s| is) what i did\b",
    r"\b(?:here(?:'s| is)|this is) what i changed\b",
    r"\bmade some fixes\b",
    r"\bmade a few fixes\b",
    r"\bcommitted as [0-9a-f]{7,40}\b",
    r"\b(?:summary|status update)\s*:",
)

_REPORTBACK_VERBS = (
    "fixed",
    "patched",
    "updated",
    "changed",
    "committed",
    "verified",
    "ran",
    "tested",
    "implemented",
    "completed",
    "finished",
    "added",
    "removed",
)

_REPORTBACK_VERB_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(verb).replace(r"\ ", r"\s+") for verb in _REPORTBACK_VERBS)
    + r")\b",
    re.IGNORECASE,
)

_FIRST_PERSON_REPORT_RE = re.compile(r"\b(?:i|we)\b", re.IGNORECASE)
# An execution directive does not have to open the sentence.
#
# The prefix form only sees a command at position zero, so "Do this for
# real now: open Chrome, take a screenshot..." read as an inventory
# question, the desktop router bailed on that basis, the turn fell through
# to chat, and she narrated a screen she never looked at — inventing
# Notepad++ and File Explorer on a Mac. Measured live 2026-07-27.
#
# A clause boundary is enough of an anchor to stay conservative: this must
# still not fire on "what tools can you use to open apps", where the verb
# is the object of the question rather than an instruction.
_DIRECT_EXECUTION_CLAUSE_RE = re.compile(
    r"(?:^|[.!?;:\n]|\b(?:now|then|first|also|and)\b)\s*"
    r"(?:please\s+|go\s+ahead\s+and\s+|go\s+)?"
    r"(?:open|launch|run|execute|click|tap|press|type|write|download|save|\
create|build|set\s+up|automate|organize)\s+"
    r"(?:the\s+|a\s+|an\s+|up\s+|my\s+|your\s+)?"
    r"(?:chrome|safari|firefox|finder|terminal|browser|tab|window|app|"
    r"application|document|doc|file|folder|note|screenshot|screen\s*shot)\b",
    re.IGNORECASE,
)
_DIRECT_EXECUTION_PREFIX_RE = re.compile(
    r"^\s*(?:please\s+|can you\s+|could you\s+|would you\s+|i need you to\s+|"
    r"help me\s+|go\s+)?(?:open|launch|run|execute|click|tap|press|type|"
    r"write|search|look up|read|inspect|download|save|fix|implement|create|build|"
    r"set up|automate|organize)\b",
    re.IGNORECASE,
)
_EXPLANATORY_PREFIX_RE = re.compile(
    r"^\s*(?:answer|explain|describe|tell me|walk me through|give me your take)\b",
    re.IGNORECASE,
)
_ANSWER_SURFACE_REQUEST_RE = re.compile(
    r"(?:^|[.!?;:]\s+)"
    r"(?:(?:please|kindly)\s+|(?:can|could|would|will)\s+you\s+)?"
    r"(?:answer|explain|describe|tell\s+me|walk\s+me\s+through|compare|"
    r"contrast|summari[sz]e|state|name|give\s+me|show\s+me)\b",
    re.IGNORECASE,
)
_QUESTION_WORD_RE = re.compile(r"\b(?:what|why|how|when|where|which)\b", re.IGNORECASE)
_CONCEPTUAL_SHOULD_RE = re.compile(
    r"\b(?:what|why|how|when)\s+(?:should|would|is|are|do|does|can|could)\b",
    re.IGNORECASE,
)
_ASKS_INVENTORY_DIRECTLY_RE = re.compile(
    r"\bhow\s+many\s+(?:tools?|skills?|capabilit(?:y|ies)|things?)\b"
    r"|\bwhat\s+(?:can|could)\s+you\s+do\b"
    r"|\bwhat\s+are\s+you\s+(?:able\s+to\s+do|capable\s+of)\b"
    r"|\bwhat(?:'s| is)\s+in\s+your\s+(?:toolkit|inventory|repertoire)\b",
    re.IGNORECASE,
)
_CAPABILITY_INVENTORY_RE = re.compile(
    r"\bhow\s+many\b.{0,60}\b(?:tools?|skills?|capabilit(?:y|ies))\b|"
    r"\b(?:what|which|list|tell me|describe|explain|show)\b.{0,100}"
    r"\b(?:tools?|skills?|capabilit(?:y|ies)|things? you can do|what you can do)\b|"
    r"\b(?:can|could|do|does|are|is|have|has)\b.{0,100}\b(?:you|aura)\b.{0,100}"
    r"\b(?:tools?|skills?|capabilit(?:y|ies)|external(?:ly)?|desktop|computer|browser|files?|apps?|notes?|pdf|search|web|terminal)\b|"
    r"\b(?:whether|if)\s+(?:you|aura|she)\s+(?:can|could|would)\b.{0,120}"
    r"\b(?:tools?|skills?|capabilit(?:y|ies)|external(?:ly)?|desktop|computer|browser|files?|apps?|notes?|pdf|search|web|terminal)\b",
    re.IGNORECASE,
)
# Explicit reply contracts: the user is telling us where the answer goes.
_INLINE_REPLY_CONTRACT_RE = re.compile(
    r"\bnot as a (?:background )?task\b|"
    r"\bwithout (?:starting|creating|filing|opening|spawning) a task\b|"
    r"\bdon'?t (?:start|create|file|open|spawn) a (?:background )?task\b|"
    r"\bin this (?:very )?reply\b|"
    r"\bin your (?:next )?reply\b|"
    r"\banswer(?:ed)?(?:\s+\S+){0,2}\s+right here\b",
    re.IGNORECASE,
)

# First-person state check-ins — these want Aura's live voice, never a ticket.
_INTROSPECTIVE_STATE_RE = re.compile(
    r"\bvalence\b|\barousal\b|"
    r"\byour (?:internal|current|felt|emotional|affective) state\b|"
    r"\bfrom your state\b|"
    r"\bhow (?:are you|do you) feel(?:ing)?\b|"
    r"\byour mood\b|"
    r"\bthe two numbers\b|"
    r"\bfelt[- ]sense\b",
    re.IGNORECASE,
)

# An action verb aimed at something outside the reply (files, apps, web,
# desktop). Presence of one means the turn genuinely wants execution.
_EXTERNAL_EFFECT_RE = re.compile(
    r"\b(?:open|launch|run|execute|install|download|save|create|build|make|"
    r"change|set|adjust|enable|disable|turn\s+(?:on|off)|put|copy|"
    r"write|draft|generate|compose|organize|automate|set\s+up|fix|refactor|"
    r"rename|move|delete|remove|clean\s+up|sort|schedule|send|email|browse|"
    r"navigate|visit|click|type|edit|update|research|search|find|look\s+up|"
    r"check|read)\b"
    r"[^.?!\n]{0,50}?"
    r"\b(?:files?|folders?|director(?:y|ies)|apps?|applications?|scripts?|"
    r"notes?|documents?|docs?|repos?|repositor(?:y|ies)|projects?|"
    r"screenshots?|downloads?|desktop|computer|browsers?|tabs?|windows?|"
    r"emails?|calendar|terminal|websites?|pages?|urls?|spreadsheets?|"
    r"presentations?|reminders?|web|internet|online|articles?|sources?)\b",
    re.IGNORECASE,
)
_EXTERNAL_MEDIUM_RE = re.compile(
    r"\b(?:using|via|with)\s+(?:the\s+)?(?:web|internet|browser|web\s+search|"
    r"online\s+(?:search|sources?))\b",
    re.IGNORECASE,
)

_NEGATED_ACTION_SPAN_RE = re.compile(
    r"\b(?:do\s+not|don't|dont|never|without|no)\b"
    r"(?:\s+[a-z0-9_'/-]+){0,10}\s+"
    r"\b(?:"
    + "|".join(
        re.escape(verb).replace(r"\ ", r"\s+")
        for verb in (*_ACTION_VERBS, "tool", "tools", "app", "apps")
    )
    + r")\b"
    r"(?:\s+[a-z0-9_'/-]+){0,8}",
    re.IGNORECASE,
)


def normalize_matched_skills(matched_skills: object) -> list[str]:
    if matched_skills is True:
        return ["*"]
    if not matched_skills:
        return []
    if isinstance(matched_skills, str):
        return [matched_skills]
    if isinstance(matched_skills, Iterable):
        normalized: list[str] = []
        for item in matched_skills:
            text = str(item or "").strip()
            if text:
                normalized.append(text)
        return normalized
    return [str(matched_skills)]


def strip_negated_action_spans(text: str) -> str:
    """Remove local negative action clauses before execution-intent matching."""

    normalized = normalize_memory_intent_text(text)
    if not normalized:
        return ""
    return re.sub(r"\s+", " ", _NEGATED_ACTION_SPAN_RE.sub(" ", normalized)).strip()


def looks_like_capability_inventory_dialogue_request(text: str) -> bool:
    normalized = normalize_memory_intent_text(text)
    if not normalized:
        return False
    # An address is not a sentence.
    #
    # LIVE, 2026-08-27: "ok this is driving me nuts. /private/tmp/claude-501/
    # -Users-bryan--aura-live-source/.../invoice-tools — clean run, nothing
    # raises... what's the actual cause, and what do I change?" read as a
    # question about her capabilities, so routing skipped the skill block
    # entirely, no tool was offered, and the model guessed the answer from the
    # symptom instead of looking. "aura" is in the path.
    #
    # This is the second reader with this bug; the first was fixed in
    # chat_preflight the same way, which is what two functions answering one
    # question always costs.
    try:
        from core.intent.opaque_spans import without_opaque_spans

        normalized = without_opaque_spans(normalized)
    except (ImportError, TypeError, ValueError):
        pass
    if not normalized.strip():
        return False
    if len(normalized.split()) > 80:
        return False
    sanitized = strip_negated_action_spans(normalized).lower()
    direct_execution = bool(
        _DIRECT_EXECUTION_PREFIX_RE.search(sanitized)
        or _DIRECT_EXECUTION_CLAUSE_RE.search(sanitized)
    )
    asks_inventory = bool(
        _CAPABILITY_INVENTORY_RE.search(normalized)
        or _ASKS_INVENTORY_DIRECTLY_RE.search(normalized)
    )
    if not asks_inventory or direct_execution:
        return False

    # Modal requests are ordinary English requests, not automatically
    # capability questions. The old broad inventory pattern read any
    # ``can/could ... you ... desktop/computer/browser`` sentence as asking
    # what Aura can do, so "Can you change my desktop background?" was
    # intercepted before the action router. Keep explicit answer surfaces
    # ("tell me whether", "explain whether") in dialogue; otherwise let the
    # shared grammatical substrate decide whether this is a present directive
    # whose object is a concrete external effect.
    if not _ANSWER_SURFACE_REQUEST_RE.search(sanitized):
        try:
            from core.conversation.request_mood import assess_request_mood

            mood = assess_request_mood(normalized)
            if mood.asks_for_action and _EXTERNAL_EFFECT_RE.search(sanitized):
                return False
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return True


def looks_like_execution_report(text: str) -> bool:
    normalized = normalize_memory_intent_text(text)
    if not normalized:
        return False

    lowered = normalized.lower()
    if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in _REPORTBACK_PATTERNS):
        return True

    report_verbs = {match.group(0).lower() for match in _REPORTBACK_VERB_RE.finditer(lowered)}
    if not report_verbs:
        return False

    if _FIRST_PERSON_REPORT_RE.search(lowered) and len(report_verbs) >= 2 and not lowered.endswith("?"):
        return True

    return False


def looks_like_explanatory_dialogue_request(text: str) -> bool:
    """Return true for questions about how an action should work, not a request to do it.

    Tool-heavy explanations often contain words like "run", "launch", "tool",
    "report", and "trace". Those are evidence terms in a question, not always
    an instruction to enter the AutonomousTaskEngine. Direct imperatives still
    route to SKILL/TASK.
    """
    normalized = normalize_memory_intent_text(text)
    if not normalized:
        return False
    if looks_like_capability_inventory_dialogue_request(normalized):
        return True
    lowered = normalized.lower()
    answer_surface = bool(
        _EXPLANATORY_PREFIX_RE.search(lowered)
        or _ANSWER_SURFACE_REQUEST_RE.search(lowered)
    )
    has_question_shape = "?" in lowered or answer_surface
    if not has_question_shape:
        return False
    if _DIRECT_EXECUTION_PREFIX_RE.search(lowered) and not answer_surface:
        return False
    if _CONCEPTUAL_SHOULD_RE.search(lowered):
        return True
    if answer_surface:
        return True
    if (
        "operational evidence" in lowered
        and any(term in lowered for term in ("personhood", "consciousness", "proof"))
    ):
        return True
    return False


#: Asks whose deliverable is speech, not an external effect. "Tell me a
#: story" is a request; what it requests is words.
_CONVERSATIONAL_ASK_RE = re.compile(
    r"\b(?:"
    r"tell\s+me\s+(?:about\s+(?:yourself|you)|something|a\s+story|more|"
    r"what|how|why|when|whether|if)"
    r"|talk\s+to\s+me\s+about"
    r"|say\s+something"
    r"|give\s+me\s+(?:your\s+)?(?:take|opinion|thoughts?|read)"
    r"|what'?s\s+(?:your\s+)?(?:take|opinion|read)"
    r")\b",
    re.IGNORECASE,
)


def looks_like_inline_answer_request(text: str) -> bool:
    """True when the turn's deliverable is words in the current reply.

    A question — a math problem, a feeling check-in, a "what do you
    think" — wants an answer here and now. Routing such a turn into the
    background TaskEngine answers a question with a ticket receipt
    (observed live, July 2026: a numeric introspection check-in and a
    train catch-up problem both got "Task accepted into governed
    background execution"). A demand for external effects — files, apps,
    web, desktop — is never inline; an explicit reply contract like
    "not as a task" always is.
    """
    normalized = normalize_memory_intent_text(text)
    if not normalized:
        return False
    lowered = normalized.lower()
    if _INLINE_REPLY_CONTRACT_RE.search(lowered):
        return True
    sanitized = strip_negated_action_spans(normalized).lower()
    if (
        _EXTERNAL_EFFECT_RE.search(sanitized)
        or _EXTERNAL_MEDIUM_RE.search(sanitized)
        or _DIRECT_EXECUTION_PREFIX_RE.search(sanitized)
        or any(re.search(pattern, sanitized, re.IGNORECASE) for pattern in _DESKTOP_PATTERNS)
    ):
        return False
    if _INTROSPECTIVE_STATE_RE.search(lowered):
        return True
    if looks_like_explanatory_dialogue_request(normalized):
        return True
    # "Tell me about yourself", "tell me something", "tell me a story".
    # The deliverable is words in THIS reply. Without this the last-resort
    # asks_for_action branch in turn_analysis classified them TASK, and a
    # person asking Aura to say something about herself got "Task accepted
    # into governed background execution" — the exact receipt-for-an-answer
    # failure this function's docstring was written about.
    if _CONVERSATIONAL_ASK_RE.search(lowered):
        return True
    return "?" in lowered and bool(_QUESTION_WORD_RE.search(lowered))


def looks_like_multi_step_skill_request(
    text: str,
    matched_skills: object = None,
) -> bool:
    normalized = normalize_memory_intent_text(text)
    if not normalized:
        return False
    if looks_like_execution_report(normalized):
        return False
    if looks_like_explanatory_dialogue_request(normalized):
        return False
    if looks_like_inline_answer_request(normalized):
        return False

    lowered = strip_negated_action_spans(normalized).lower()
    skills = normalize_matched_skills(matched_skills)

    action_hits = {match.group(0).lower() for match in _ACTION_RE.finditer(lowered)}
    action_count = len(action_hits)
    has_chain_marker = any(re.search(pattern, lowered, re.IGNORECASE) for pattern in _CHAIN_PATTERNS)
    has_report_marker = any(re.search(pattern, lowered, re.IGNORECASE) for pattern in _REPORT_PATTERNS)
    has_desktop_marker = any(re.search(pattern, lowered, re.IGNORECASE) for pattern in _DESKTOP_PATTERNS)
    single_step_like = any(re.search(pattern, lowered, re.IGNORECASE) for pattern in _SINGLE_STEP_PATTERNS)

    if action_count >= 3:
        return True
    if action_count >= 2 and (has_chain_marker or has_report_marker or has_desktop_marker):
        return True
    if len(set(skills)) >= 2 and (action_count >= 1 or has_chain_marker):
        return True
    if has_report_marker and (action_count >= 1 or bool(skills)):
        return True
    if has_chain_marker and has_desktop_marker:
        return True
    if has_desktop_marker and " and " in lowered and action_count >= 1:
        return True
    if single_step_like:
        return False
    return False
