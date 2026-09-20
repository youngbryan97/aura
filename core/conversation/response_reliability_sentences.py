"""Cutting a reply into sentences, and shaping them to a requested count.

Lifted whole out of `response_reliability`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import re
from typing import Any

from core.conversation.word_markers import names_any


def _split_sentences(text: str) -> list[str]:
    from .response_reliability import (
        _LIST_LINE_RE,
        _SENTENCE_SPLIT_RE,
        normalize_user_facing_format,
    )

    text = normalize_user_facing_format(text)
    lines: list[str] = []
    for line in text.splitlines():
        match = _LIST_LINE_RE.match(line)
        if match:
            body = str(match.group("body") or "").strip()
            if body:
                lines.append(body)
            continue
        if line.strip():
            lines.append(line.strip())
    if lines:
        text = " ".join(lines)
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text)]
    return [sentence for sentence in sentences if sentence]


def _finish_sentence_fragment(fragment: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s*)", "", str(fragment or "")).strip()
    cleaned = cleaned.strip(" \t\r\n,;:")
    if not cleaned:
        return ""
    lower = cleaned.lower()
    replacements = (
        ("ensuring that ", "That ensures that "),
        ("which ", "That "),
        ("and ", ""),
        ("but ", "But "),
        ("so ", "So "),
        ("because ", "That matters because "),
    )
    for prefix, replacement in replacements:
        if lower.startswith(prefix):
            cleaned = f"{replacement}{cleaned[len(prefix):]}".strip()
            break
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned and cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def _split_long_sentence_once(sentence: str) -> list[str]:
    from .response_reliability import (
        _word_count,
    )

    cleaned = _finish_sentence_fragment(sentence)
    if _word_count(cleaned) < 14:
        return [cleaned] if cleaned else []
    split_specs = (
        (r",\s+ensuring that\s+", "ensuring that "),
        (r",\s+which\s+", "which "),
        (r";\s+", ""),
        (r":\s+", ""),
        (r"\s+so that\s+", "so that "),
        (r"\s+because\s+", "because "),
    )
    for pattern, right_prefix in split_specs:
        matches = list(re.finditer(pattern, cleaned, re.IGNORECASE))
        for match in reversed(matches):
            left = cleaned[: match.start()]
            right = f"{right_prefix}{cleaned[match.end():]}"
            left_done = _finish_sentence_fragment(left)
            right_done = _finish_sentence_fragment(right)
            if _word_count(left_done) >= 5 and _word_count(right_done) >= 4:
                return [left_done, right_done]
    marker = ", and "
    idx = cleaned.lower().rfind(marker)
    if idx > 0:
        left_done = _finish_sentence_fragment(cleaned[:idx])
        right_done = _finish_sentence_fragment(cleaned[idx + len(marker):])
        if _word_count(left_done) >= 7 and _word_count(right_done) >= 5:
            return [left_done, right_done]
    return [cleaned]


def _expand_sentence_candidates(sentences: list[str], count: int) -> list[str]:
    from .response_reliability import (
        _word_count,
    )

    expanded = [_finish_sentence_fragment(sentence) for sentence in sentences]
    expanded = [sentence for sentence in expanded if sentence]
    while len(expanded) < count:
        split_index = max(
            range(len(expanded)),
            key=lambda idx: _word_count(expanded[idx]),
            default=-1,
        )
        if split_index < 0 or _word_count(expanded[split_index]) < 14:
            break
        split = _split_long_sentence_once(expanded[split_index])
        if len(split) <= 1:
            break
        expanded = expanded[:split_index] + split + expanded[split_index + 1 :]
    return expanded


def _pad_sentence_candidates(sentences: list[str], count: int) -> list[str]:
    """Return only sentences supported by the draft being repaired.

    A deterministic shape repair may split or reformat existing content. It
    cannot create the missing semantic predicates. The old implementation
    padded an undersized answer with sentences whose sole purpose was making
    the count pass; that converted a measured shortfall into a false success.
    Leaving the list short keeps ``missing_requested_sentence_count`` live so
    the caller can regenerate or serve an explicitly recorded partial result.
    """

    del count
    return list(sentences)


def _paragraphize_sentences(sentences: list[str], count: int) -> str:
    if count <= 1 or len(sentences) < count:
        return " ".join(sentences).strip()
    paragraphs: list[str] = []
    for idx in range(count):
        start = round(idx * len(sentences) / count)
        end = round((idx + 1) * len(sentences) / count)
        block = " ".join(sentences[start:end]).strip()
        if block:
            paragraphs.append(block)
    return "\n\n".join(paragraphs)


def _listify_sentences(sentences: list[str], count: int) -> str:
    if count <= 1 or len(sentences) < count:
        return " ".join(sentences).strip()
    return "\n".join(f"- {sentence}" for sentence in sentences[:count])


def _number_sentences(sentences: list[str], count: int) -> str:
    sentences = _expand_sentence_candidates(sentences, count)
    if count <= 1 or len(sentences) < count:
        return " ".join(sentences).strip()
    numbered: list[str] = []
    for idx, sentence in enumerate(sentences[:count], start=1):
        cleaned = _finish_sentence_fragment(sentence)
        if not cleaned:
            continue
        numbered.append(f"{idx}. {cleaned}")
    return "\n".join(numbered)


def _default_followup_question(user_message: Any) -> str:
    from .response_reliability import (
        _normalize,
    )

    user_norm = _normalize(user_message)
    if names_any(user_norm, ("live path", "desktop path", "validate", "probe", "runtime")):
        return "What should I validate next on this same live path?"
    if any(marker in user_norm for marker in ("project", "next hour", "focus", "work on", "spend")):
        return "Which outcome would make the next hour feel most useful?"
    if names_any(user_norm, ("demo", "show me", "open", "write", "search")):
        return "Which part should I do first so the whole chain stays visible and verifiable?"
    return "What outcome would make this most useful for you right now?"
