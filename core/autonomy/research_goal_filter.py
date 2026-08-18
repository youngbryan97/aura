from __future__ import annotations

import re
from typing import Any

_RESEARCH_PREFIXES = (
    "research and learn something new about ",
    "research ",
    "learn about ",
    "explore ",
    "investigate ",
    "look into ",
    "find out about ",
    "self-directed exploration of ",
)

_STALE_OR_RECEIPT_MARKERS = (
    "unresolved: stalled goal:",
    "stalled goal:",
    "desktop task receipt",
    "canonical computer-use gateway",
    "governed desktop actuators",
    "artifact references:",
    "[retained memory evidence]",
    "scope=retained_memory_evidence.v1",
    "source=durable_memory_search",
)

_PROMPT_SCAFFOLD_MARKERS = (
    "subconscious synthesis",
    "concept a:",
    "concept b:",
    "task:",
    "strategic heuristic",
    "universal principle",
    "predict how self will react if i take this action",
    "{'type': 'autonomous_goal'",
    '"type": "autonomous_goal"',
    "mastery of: user asked about:",
    "reply \"no_connection\"",
    "reply 'no_connection'",
    "json schema",
    "system prompt",
    "you are an ai",
)

_DESKTOP_ACTION_MARKERS = (
    "create a folder",
    "write a file",
    "write a note",
    "open notes",
    "notes app",
    "google docs",
    "export",
    "pdf",
    "wallpaper",
    "desktop folder",
    "documents folder",
    "open chrome",
    "open safari",
    "type out",
    "keyboard",
    "mouse",
)

_RESEARCHABLE_HINTS = (
    "why ",
    "how ",
    "what ",
    "which ",
    "research",
    "learn",
    "explore",
    "investigate",
    "study",
    "papers",
    "evidence",
    "sources",
)


def normalize_goal_text(value: Any) -> str:
    text = str(value or "").replace("\x00", " ")
    return " ".join(text.split()).strip(" -:;,.?!")


#: A goal she chose is ABOUT THE WORLD. One that addresses her in the second
#: person, or instructs her to produce output over some supplied text, is a
#: fragment of a prompt that got promoted into volitional state.
#:
#: LIVE DEFECT, 2026-08-18. The integrity monitor reported 22 cognitive_engine
#: degradations in 30 minutes, all of one shape:
#:
#:     objective repeatedly unresolved: You are writing Aura
#:     objective repeatedly unresolved: Summarize the follow
#:
#: Both are clipped system instructions. They can never be resolved, because
#: there is no "the following" in an objective and no task in "you are" — so
#: they generate sustained objective friction forever.
#:
#: The marker list above could not catch them and structurally cannot: it needs
#: TWO markers, or one plus 500+ characters, and truncation is exactly what
#: removes both. A clipped prompt is short and carries few markers, so the
#: shorter and more mangled the fragment, the more certainly it slipped
#: through. An enumeration of phrasings is always one phrasing behind; this
#: matches the GRAMMATICAL FORM of an instruction addressed to a model, which
#: does not vary with topic or length.
_INSTRUCTION_OPENER_RE = re.compile(
    r"^\s*(?:"
    # Addressed to her: "You are writing Aura", "Your task is ..."
    r"you\s+(?:are|will|shall|should|must|can|may|have\s+to|need\s+to)\b"
    r"|your\s+(?:task|job|role|goal|instruction|instructions|response|reply|output)\b"
    # Told to operate on supplied text: "Summarize the following ..."
    # `follow` bare catches the clipped form; the lookahead keeps a real goal
    # about "the follow-up notes" from being mistaken for one.
    r"|(?:summari[sz]e|rewrite|paraphrase|translate|classify|extract|list|"
    r"generate|produce|output|respond\s+to|reply\s+to|answer|continue|"
    r"complete|format|analyz|analys)\w*\s+the\s+"
    r"(?:follow(?:ing)?(?!-)|above|below|text|passage|conversation|transcript)\b"
    r"|(?:given|based\s+on|using)\s+the\s+(?:follow(?:ing)?(?!-)|above|below)\b"
    r")",
    re.IGNORECASE,
)


def is_instruction_shaped_goal(value: Any) -> bool:
    """True when the text is an instruction to a model, not a goal of her own."""
    text = normalize_goal_text(value)
    if not text:
        return False
    return bool(_INSTRUCTION_OPENER_RE.search(text))


def is_prompt_shaped_goal(value: Any) -> bool:
    text = normalize_goal_text(value)
    if not text:
        return False
    lowered = text.casefold()
    marker_hits = sum(1 for marker in _PROMPT_SCAFFOLD_MARKERS if marker in lowered)
    if marker_hits >= 2:
        return True
    if len(text) > 500 and marker_hits >= 1:
        return True
    # Independent of length and marker count: truncation defeats both.
    if is_instruction_shaped_goal(text):
        return True
    return bool(re.search(r"\btask:\s*\d+\.\s", lowered))


def is_stale_or_prompt_scaffold_goal(value: Any) -> bool:
    text = normalize_goal_text(value)
    if not text:
        return False
    lowered = text.casefold()
    if any(marker in lowered for marker in _STALE_OR_RECEIPT_MARKERS):
        return True
    return is_prompt_shaped_goal(text)


def is_desktop_action_goal(value: Any) -> bool:
    lowered = normalize_goal_text(value).casefold()
    if not lowered:
        return False
    hits = sum(1 for marker in _DESKTOP_ACTION_MARKERS if marker in lowered)
    if hits >= 2:
        return True
    return lowered.startswith(("open ", "create ", "write ", "export ", "change ")) and hits >= 1


def is_unresearchable_goal(value: Any) -> bool:
    """True when a pending initiative should not be fed to background research.

    This keeps autonomy active while preventing stale action receipts, desktop
    tasks, and internal prompt scaffolds from becoming web-search queries.
    """

    text = normalize_goal_text(value)
    if not text:
        return True
    lowered = text.casefold()
    if any(marker in lowered for marker in _STALE_OR_RECEIPT_MARKERS):
        return True
    if is_prompt_shaped_goal(text):
        return True
    if is_desktop_action_goal(text):
        return True
    if is_runtime_status_goal(text):
        return True
    return False


#: Her own current state, written as a sentence. These are UI activity labels
#: and status lines, not subjects.
_RUNTIME_STATUS_SUBJECTS = (
    "aura", "the runtime", "the system", "the engine", "the cortex",
    "the model", "the desktop", "she", "i",
)
_RUNTIME_STATUS_STATES = (
    "idle", "typing", "thinking", "searching", "generating", "executing",
    "analyzing", "analysing", "managing", "interacting", "checking",
    "warming", "warming up", "loading", "booting", "starting", "sleeping",
    "dreaming", "listening", "speaking", "waiting", "ready", "online",
    "offline", "busy", "running", "shutting down", "recovering",
)
_RUNTIME_STATUS_RE = re.compile(
    r"^\s*(?:%s)\s+(?:is|was|are|were|'s)\s+(?:currently\s+|now\s+)?(?:%s)\b"
    % (
        "|".join(re.escape(subject) for subject in _RUNTIME_STATUS_SUBJECTS),
        "|".join(re.escape(state) for state in _RUNTIME_STATUS_STATES),
    ),
    re.IGNORECASE,
)


def is_runtime_status_goal(value: Any) -> bool:
    """True when the text is her own status rather than a subject.

    LIVE 2026-08-17, from the neural stream:

        [SubjectiveChoice] Chose 'Deconstruct and comprehensively research:
        Aura is idle' because preference alignment 0.00 and drive alignment
        0.49 produced final score 0.27.

    "Aura is idle." is a UI activity label. It reached the knowledge graph as a
    sparse node, was drawn as a research topic, and became a durable goal that
    set her focus and spawned a research shard — to investigate a fact she
    already holds, about herself, that a status field answers exactly.

    The filter already refused stale receipts, prompt scaffolds and desktop
    actions. Her own state is the same category of thing: something the runtime
    KNOWS, so researching it is not curiosity but a loop.
    """

    text = normalize_goal_text(value)
    if not text:
        return False
    return bool(_RUNTIME_STATUS_RE.search(text))


#: Words a research topic cannot end on — the phrase was still going.
_DANGLING_GOAL_TAIL_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "because", "but", "by",
        "for", "from", "in", "into", "is", "of", "on", "or", "our", "than",
        "that", "the", "their", "these", "this", "to", "was", "were", "which",
        "with", "essential", "important", "necessary", "useful", "critical",
    }
)


def research_query_for_goal(value: Any, *, limit: int = 220) -> str:
    text = normalize_goal_text(value)
    if not text or is_unresearchable_goal(text):
        return ""
    lowered = text.casefold()
    for prefix in _RESEARCH_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip(" -:;,.?!")
            break
    if not text or is_unresearchable_goal(text):
        return ""
    clauses = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    if len(text) > limit and clauses:
        text = clauses[0]
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].strip(" -:;,.?!")
    # A goal that ends mid-thought is not a goal. Whatever trimming happened
    # above (or upstream, before this ever saw the text), the last word has to
    # be able to end a phrase.
    for _ in range(3):
        parts = text.rsplit(" ", 1)
        if len(parts) != 2:
            break
        if parts[1].strip(" -:;,.?!").casefold() in _DANGLING_GOAL_TAIL_WORDS:
            text = parts[0].strip(" -:;,.?!")
            continue
        break
    if not text:
        return ""
    lowered = text.casefold()
    if not any(hint in lowered for hint in _RESEARCHABLE_HINTS) and len(text.split()) > 24:
        return ""
    return text
