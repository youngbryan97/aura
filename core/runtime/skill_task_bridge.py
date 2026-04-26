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


def looks_like_multi_step_skill_request(
    text: str,
    matched_skills: object = None,
) -> bool:
    normalized = normalize_memory_intent_text(text)
    if not normalized:
        return False
    if looks_like_execution_report(normalized):
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
