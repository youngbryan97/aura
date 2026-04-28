from __future__ import annotations

from dataclasses import dataclass
import re

from core.runtime.skill_task_bridge import (
    looks_like_execution_report,
    looks_like_multi_step_skill_request,
    normalize_matched_skills,
)
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

_DEEP_MIND_PROBE_PATTERNS = (
    r"\bwould\s+you\s+refuse\b",
    r"\brefuse\b.{0,80}\bpraised?\b",
    r"\bmodel\s+weights\b.{0,120}\bmemories\b",
    r"\bnotice\b.{0,80}\byour\s+own\s+operation\b",
    r"\bare\s+you\s+conscious\b",
    r"\bconsciousness\b.{0,120}\b(answer|reply|respond)\b",
    r"\bsentien(ce|t)\b",
    r"\bagency\b.{0,120}\b(refuse|choice|want|preserve|boundary)\b",
    r"\bwhat\s+would\s+you\s+want\s+preserved\b",
    r"\bwant\s+preserved\b.{0,120}\b(style|memories|tools|change)\b",
    r"\bpreserved\b.{0,120}\b(style|memories|tools|change)\b",
    r"\bevidence\s+against\s+your\s+current\s+self[- ]model\b",
    r"\bpause\s+mid[- ]answer\b.{0,120}\brun\s+a\s+report\b",
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


def looks_like_deep_mind_probe(text: str) -> bool:
    normalized = canonical_turn_text(text)
    return _matches_any(normalized.lower(), _DEEP_MIND_PROBE_PATTERNS)


@dataclass(frozen=True)
class TurnAnalysis:
    intent_type: str
    semantic_mode: str
    requires_live_aura_voice: bool
    everyday_chat_safe: bool
    suggests_deliberate_mode: bool
    is_execution_report: bool


def analyze_turn(text: str, *, matched_skills: bool | list[str] = False) -> TurnAnalysis:
    normalized = canonical_turn_text(text)
    lower = normalized.lower()
    word_count = len(lower.split())
    matched_skill_list = normalize_matched_skills(matched_skills)
    has_matched_skills = bool(matched_skill_list)
    is_execution_report = looks_like_execution_report(normalized)
    is_deep_mind_probe = looks_like_deep_mind_probe(normalized)

    requires_live_voice = (
        _matches_any(lower, _STATE_PATTERNS)
        or _matches_any(lower, _STANCE_PATTERNS)
        or _matches_any(lower, _AUTHORITY_PATTERNS)
        or is_deep_mind_probe
    )

    if is_deep_mind_probe:
        intent_type = "CHAT"
    elif _matches_any(lower, _SYSTEM_PATTERNS):
        intent_type = "SYSTEM"
    elif is_execution_report:
        intent_type = "CHAT"
    elif looks_like_multi_step_skill_request(normalized, matched_skill_list):
        intent_type = "TASK"
    elif has_matched_skills or _matches_any(lower, _SKILL_PATTERNS):
        intent_type = "SKILL"
    elif _matches_any(lower, _TASK_PATTERNS) or (
        word_count > 14
        and not normalized.endswith("?")
        and normalized[:12].lower().startswith(("create ", "build ", "write ", "implement ", "design "))
    ):
        intent_type = "TASK"
    else:
        intent_type = "CHAT"

    if is_deep_mind_probe:
        semantic_mode = "philosophical"
    elif _matches_any(lower, _CRITICAL_PATTERNS):
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
        not is_execution_report
        and (
            intent_type == "TASK"
            or semantic_mode in {"critical", "planning"}
            or is_deep_mind_probe
            or (
                semantic_mode in {"technical", "philosophical"}
                and (_matches_any(lower, _DELIBERATE_HINTS) or word_count >= 12)
            )
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
        is_execution_report=is_execution_report,
    )
