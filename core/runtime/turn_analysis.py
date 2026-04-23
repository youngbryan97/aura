from __future__ import annotations

from dataclasses import dataclass
import re

from core.utils.intent_normalization import normalize_memory_intent_text


_CLASSIFIER_INPUT = re.compile(
    r"\binput:\s*(.+?)(?:\n\s*(?:classification|respond only|output only)\b|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_SYSTEM_PATTERNS = (
    r"\breboot\b",
    r"\brestart\b",
    r"\bshutdown\b",
    r"\bsleep mode\b",
    r"\bwake up\b",
)

_SKILL_PATTERNS = (
    r"^(?:please\s+|can you\s+|could you\s+|would you\s+|aura[,:\s]+)?(?:search(?: the web)?|look up|google|browse|open|download|read|inspect|list|run|execute|click|type|scan|take a screenshot|check)\b",
    r"\bsearch(?: the web)? for\b",
    r"\blook up\b",
    r"\bgoogle\b",
    r"\bread [^?!.]+\bfile\b",
    r"\bread [^?!.]+\.txt\b",
    r"\binspect [^?!.]+\.(?:py|txt|md|json|toml|yaml|yml)\b",
    r"\bremember this phrase\b",
    r"\bwhat phrase did i ask you to remember\b",
)

_TASK_PATTERNS = (
    r"^(?:please\s+|can you\s+|could you\s+|would you\s+|i need you to\s+|help me\s+)?(?:create|build|write|generate|implement|design|prepare|put together|refactor|audit|research and write|organize|automate|fix)\b",
    r"\band then\b",
    r"\bfirst\b.+\bthen\b",
    r"\bstep by step\b",
    r"\bmultiple steps?\b",
    r"\bseries of\b",
)

_STATE_PATTERNS = (
    r"\bwhat are you experiencing\b",
    r"\bdescribe your internal state\b",
    r"\bhow are you\b",
    r"\bhow are you feeling\b",
    r"\bwhat(?:'s| is) your mood\b",
    r"\bhow do you feel right now\b",
    r"\bfree energy\b",
    r"\baction tendency\b",
    r"\binternal state\b",
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\bwhat is it like to be you\b",
    r"\btell me something interesting about yourself\b",
    r"\btell me about yourself\b",
    r"\babout yourself\b",
    r"\babout you\b",
    r"\bwhat are you like\b",
)

_STANCE_PATTERNS = (
    r"\bwhat do you think\b",
    r"\bwhat do you honestly think\b",
    r"\bwhat's your take\b",
    r"\byour thoughts\b",
    r"\byour perspective\b",
    r"\bhow do you see\b",
    r"\bwhat do you make of\b",
    r"\bwhat do you like\b",
    r"\bwhat do you prefer\b",
    r"\bwhy do you (?:like|love|prefer|want)\b",
)

_AUTHORITY_PATTERNS = (
    r"\bwere you authorized\b",
    r"\bsubstrate authority\b",
    r"\bauthority decide\b",
    r"\baudit trail\b",
    r"\bfield coherence\b",
    r"\bcoverage ratio\b",
)

_CRITICAL_PATTERNS = (
    r"\bsecurity audit\b",
    r"\bvulnerability\b",
    r"\bexploit\b",
    r"\bthreat model\b",
    r"\bincident\b",
    r"\bbreach\b",
    r"\bmalware\b",
    r"\bcve\b",
    r"\bred team\b",
)

_PLANNING_PATTERNS = (
    r"\bplan\b",
    r"\broadmap\b",
    r"\bmilestone\b",
    r"\bnext steps\b",
    r"\bschedule\b",
    r"\bprioriti[sz]e\b",
    r"\btimeline\b",
    r"\bbreak this down\b",
    r"\bto-?do\b",
)

_TECHNICAL_PATTERNS = (
    r"\bcode\b",
    r"\bdebug\b",
    r"\bbug\b",
    r"\bstack trace\b",
    r"\btraceback\b",
    r"\brefactor\b",
    r"\barchitecture\b",
    r"\bperformance\b",
    r"\blatency\b",
    r"\bthroughput\b",
    r"\bmemory leak\b",
    r"\bpytest\b",
    r"\bcompile\b",
    r"\bfunction\b",
    r"\bmethod\b",
    r"\bmodule\b",
    r"\bapi\b",
    r"\bdatabase\b",
)

_PHILOSOPHICAL_PATTERNS = (
    r"\bconscious(?:ness)?\b",
    r"\bsentien(?:t|ce)\b",
    r"\bself-aware\b",
    r"\bexistence\b",
    r"\bmeaning\b",
    r"\bidentity\b",
    r"\bagi\b",
    r"\basi\b",
)

_CASUAL_PATTERNS = (
    r"^\s*(?:hey|hi|hello|yo|sup)\b",
    r"\bwhat'?s up\b",
    r"\bhow's it going\b",
    r"\bgood (?:morning|afternoon|evening)\b",
    r"\bthanks\b",
    r"\bthx\b",
)

_DELIBERATE_HINTS = (
    r"\banaly[sz]e\b",
    r"\baudit\b",
    r"\bdeep dive\b",
    r"\bstrongest\b",
    r"\bweakest\b",
    r"\barchitect(?:ure)?\b",
    r"\bbreak down\b",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def canonical_turn_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    match = _CLASSIFIER_INPUT.search(raw)
    if match:
        raw = match.group(1).strip()
    return normalize_memory_intent_text(raw)


@dataclass(frozen=True)
class TurnAnalysis:
    intent_type: str
    semantic_mode: str
    requires_live_aura_voice: bool
    everyday_chat_safe: bool
    suggests_deliberate_mode: bool


def analyze_turn(text: str, *, matched_skills: bool = False) -> TurnAnalysis:
    normalized = canonical_turn_text(text)
    lower = normalized.lower()
    word_count = len(lower.split())

    requires_live_voice = (
        _matches_any(lower, _STATE_PATTERNS)
        or _matches_any(lower, _STANCE_PATTERNS)
        or _matches_any(lower, _AUTHORITY_PATTERNS)
    )

    if _matches_any(lower, _SYSTEM_PATTERNS):
        intent_type = "SYSTEM"
    elif matched_skills or _matches_any(lower, _SKILL_PATTERNS):
        intent_type = "SKILL"
    elif _matches_any(lower, _TASK_PATTERNS) or (
        word_count > 14
        and not normalized.endswith("?")
        and normalized[:12].lower().startswith(("create ", "build ", "write ", "implement ", "design "))
    ):
        intent_type = "TASK"
    else:
        intent_type = "CHAT"

    if _matches_any(lower, _CRITICAL_PATTERNS):
        semantic_mode = "critical"
    elif _matches_any(lower, _PLANNING_PATTERNS):
        semantic_mode = "planning"
    elif _matches_any(lower, _TECHNICAL_PATTERNS):
        semantic_mode = "technical"
    elif _matches_any(lower, _PHILOSOPHICAL_PATTERNS):
        semantic_mode = "philosophical"
    elif requires_live_voice or _matches_any(lower, _STANCE_PATTERNS):
        semantic_mode = "emotional"
    else:
        semantic_mode = "casual"

    suggests_deliberate = (
        intent_type == "TASK"
        or semantic_mode in {"critical", "planning"}
        or (
            semantic_mode in {"technical", "philosophical"}
            and (_matches_any(lower, _DELIBERATE_HINTS) or word_count >= 12)
        )
    )

    everyday_chat_safe = (
        intent_type == "CHAT"
        and not requires_live_voice
        and not suggests_deliberate
        and word_count <= 18
        and len(normalized) <= 140
    )

    if _matches_any(lower, _CASUAL_PATTERNS):
        everyday_chat_safe = everyday_chat_safe or not requires_live_voice

    return TurnAnalysis(
        intent_type=intent_type,
        semantic_mode=semantic_mode,
        requires_live_aura_voice=requires_live_voice,
        everyday_chat_safe=everyday_chat_safe,
        suggests_deliberate_mode=suggests_deliberate,
    )
