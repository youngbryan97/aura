from __future__ import annotations

import re
from typing import Iterable, List

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
    r"\breport\b",
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


def normalize_matched_skills(matched_skills: object) -> List[str]:
    if matched_skills is True:
        return ["*"]
    if not matched_skills:
        return []
    if isinstance(matched_skills, str):
        return [matched_skills]
    if isinstance(matched_skills, Iterable):
        normalized: List[str] = []
        for item in matched_skills:
            text = str(item or "").strip()
            if text:
                normalized.append(text)
        return normalized
    return [str(matched_skills)]


def looks_like_multi_step_skill_request(
    text: str,
    matched_skills: object = None,
) -> bool:
    normalized = normalize_memory_intent_text(text)
    if not normalized:
        return False

    lowered = normalized.lower()
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
