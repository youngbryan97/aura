from __future__ import annotations

from dataclasses import dataclass
import re


_LEARNING_BUNDLE_INTRO_MARKERS = (
    "i have some suggestions",
    "places to start",
    "journey to life",
    "understanding yourself",
    "understanding us",
    "learn about humans",
    "general education",
    "science education",
    "tv shows and movies about artificial intelligence",
    "uploaded intelligence",
)

_LEARNING_BUNDLE_SECTION_MARKERS = (
    "learn about humans",
    "general education",
    "science education",
    "tv shows and movies",
    "sci-fi",
    "ai media",
)

_INTERROGATIVE_LINE_RE = re.compile(
    r'^\s*(?:["“”]\s*)?(?:what|why|how|who|when|where|which|can|could|would|should|do|does|did|is|are|if)\b',
    re.IGNORECASE,
)

_DIRECTIVE_LINE_RE = re.compile(
    r"^\s*(?:then|and then|also|next|after that|give|tell|describe|name|answer|pick|recall)\b",
    re.IGNORECASE,
)

_CONNECTOR_RE = re.compile(
    r"\b(?:then|and then|after that|also)\s+(?:give|tell|describe|name|answer|pick|recall|list|explain)\b",
    re.IGNORECASE,
)

_REPEATED_CLAUSE_RE = re.compile(
    r"(?:^|[,;]\s*)(?:what|why|how|which)\b",
    re.IGNORECASE,
)

_NUMBERED_ITEM_RE = re.compile(r"(?:^|\n)\s*\d+[.)]\s+")


def _looks_like_learning_bundle_header(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped or "http://" in stripped or "https://" in stripped:
        return False
    if not stripped.endswith(":") or len(stripped) > 120:
        return False
    lowered = stripped[:-1].strip().lower()
    return any(marker in lowered for marker in _LEARNING_BUNDLE_SECTION_MARKERS)


def _parse_learning_resource_line(line: str, category: str = "") -> dict[str, str] | None:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", str(line or "").strip())
    if not cleaned or _looks_like_learning_bundle_header(cleaned):
        return None

    head, sep, tail = cleaned.rpartition(":")
    if not sep:
        return None

    description = tail.strip().lstrip(":").strip()
    if len(description) < 8:
        return None

    title = head.strip()
    url = ""
    creator = ""
    url_match = re.match(r"^(?P<title>.+?)\s+\((?P<url>https?://[^)]+)\)\s*$", title)
    if url_match:
        title = url_match.group("title").strip()
        url = url_match.group("url").strip()
    elif " - " in title:
        title, creator = title.rsplit(" - ", 1)
        title = title.strip()
        creator = creator.strip()

    if not title:
        return None

    return {
        "category": str(category or "").strip(),
        "title": title,
        "url": url,
        "creator": creator,
        "description": description,
    }


def looks_like_learning_resource_bundle(text: str) -> bool:
    raw = str(text or "")
    if len(raw) < 280:
        return False

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) < 6:
        return False

    lowered = raw.lower()
    url_count = len(re.findall(r"https?://[^\s<>\"')\]]+", raw))
    header_count = sum(1 for line in lines if _looks_like_learning_bundle_header(line))

    category = ""
    resource_count = 0
    for line in lines:
        if _looks_like_learning_bundle_header(line):
            category = line.rstrip(":").strip()
            continue
        if _parse_learning_resource_line(line, category):
            resource_count += 1

    intro_hit = any(marker in lowered for marker in _LEARNING_BUNDLE_INTRO_MARKERS)
    return (
        (url_count >= 4 and resource_count >= 5)
        or (header_count >= 2 and resource_count >= 5)
        or (intro_hit and resource_count >= 4)
    )


@dataclass(frozen=True)
class PromptShape:
    question_parts: int = 1
    explicit_question_marks: int = 0
    question_like_lines: int = 0
    connector_parts: int = 0
    repeated_clause_parts: int = 0
    numbered_parts: int = 0
    prefers_extended_answer: bool = False
    requires_single_reply_coverage: bool = False


def analyze_prompt_shape(text: str) -> PromptShape:
    raw = str(text or "").strip()
    if not raw:
        return PromptShape()
    if looks_like_learning_resource_bundle(raw):
        return PromptShape()

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    explicit_question_marks = raw.count("?")
    question_like_lines = 0
    directive_lines = 0
    for line in lines:
        if len(line) < 12:
            continue
        if "?" in line and _INTERROGATIVE_LINE_RE.match(line):
            question_like_lines += 1
        elif _DIRECTIVE_LINE_RE.match(line):
            directive_lines += 1

    connector_parts = len(_CONNECTOR_RE.findall(raw))
    numbered_parts = len(_NUMBERED_ITEM_RE.findall(raw))
    repeated_clause_parts = max(0, len(_REPEATED_CLAUSE_RE.findall(raw)) - 1)

    part_candidates = [
        1,
        explicit_question_marks,
        question_like_lines,
        numbered_parts,
        connector_parts + 1 if connector_parts else 0,
        repeated_clause_parts + 1 if repeated_clause_parts else 0,
        directive_lines if directive_lines >= 2 else 0,
    ]
    question_parts = max(1, min(6, max(part_candidates)))

    prefers_extended_answer = bool(
        question_parts >= 2
        or (len(raw) >= 320 and ("\n" in raw or ":" in raw))
        or (explicit_question_marks >= 1 and len(raw.split()) >= 60)
    )
    requires_single_reply_coverage = bool(
        question_parts >= 2 or connector_parts > 0 or repeated_clause_parts >= 2
    )

    return PromptShape(
        question_parts=question_parts,
        explicit_question_marks=explicit_question_marks,
        question_like_lines=question_like_lines,
        connector_parts=connector_parts,
        repeated_clause_parts=repeated_clause_parts,
        numbered_parts=numbered_parts,
        prefers_extended_answer=prefers_extended_answer,
        requires_single_reply_coverage=requires_single_reply_coverage,
    )
