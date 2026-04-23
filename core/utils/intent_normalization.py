from __future__ import annotations

import re

from core.utils.text_metrics import fuzzy_match_ratio

_WORD_RE = re.compile(r"\b[a-z']+\b", re.IGNORECASE)
_MEMORY_VERB_TARGETS = ("remember", "recall")


def _looks_like_memory_verb(word: str, target: str) -> bool:
    lowered = str(word or "").strip().lower()
    if not lowered:
        return False
    if lowered == target:
        return True
    if len(lowered) < 5 or abs(len(lowered) - len(target)) > 2:
        return False
    if lowered[0] != target[0]:
        return False
    threshold = 0.84 if target == "remember" else 0.8
    return fuzzy_match_ratio(lowered, target) >= threshold


def normalize_memory_intent_text(text: str) -> str:
    """Normalize common memory-verb typos without rewriting unrelated text."""
    lowered = " ".join(str(text or "").split()).lower()
    if not lowered:
        return ""

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0).lower()
        for target in _MEMORY_VERB_TARGETS:
            if _looks_like_memory_verb(token, target):
                return target
        return token

    return _WORD_RE.sub(_replace, lowered)
