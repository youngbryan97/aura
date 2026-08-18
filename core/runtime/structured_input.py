from __future__ import annotations

import re
from dataclasses import dataclass

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
    r"^\s*(?:then|and then|also|next|after that|give|tell|describe|name|answer|pick|"
    r"recall|compare|contrast|choose|explain|verify|evaluate|trace|test)\b",
    re.IGNORECASE,
)

_COORDINATED_DIRECTIVE_RE = re.compile(
    r"(?:^|[,;:]\s*(?:and\s+)?|[.!?]\s+|\b(?:and|then|also|next|finally)\s+)"
    r"(?:please\s+)?(?P<directive>"
    r"answer|analyze|build|calculate|choose|compare|contrast|create|debug|define|"
    r"derive|describe|design|diagnose|discuss|download|enumerate|evaluate|explain|"
    r"export|find|fix|give|identify|implement|inspect|justify|list|name|open|outline|"
    r"plan|prove|provide|read|recommend|remember|report|review|save|select|set|show|"
    r"state|summarize|tell|test|trace|validate|verify|write"
    r")\b",
    re.IGNORECASE | re.MULTILINE,
)

_CONNECTOR_RE = re.compile(
    r"\b(?:then|and then|after that|also)\s+(?:give|tell|describe|name|answer|pick|"
    r"recall|list|compare|contrast|choose|explain|verify|evaluate|trace|test)\b",
    re.IGNORECASE,
)

_REPEATED_CLAUSE_RE = re.compile(
    r"(?:^|[,;]\s*)(?:what|why|how|which)\b",
    re.IGNORECASE,
)

_NUMBERED_ITEM_RE = re.compile(r"(?:^|\n)\s*\d+[.)]\s+")
_INLINE_NUMBERED_ITEM_RE = re.compile(
    r"(?<!\w)\((?P<number>[1-9]|1[0-2])\)\s+"
)

# --- Supplied material -------------------------------------------------------
#
# A turn can CARRY THE THING IT IS ASKING ABOUT. When the user pastes a note,
# quotes a block, or attaches content, the answer is already in the message and
# no external evidence exists that could improve it.
#
# Measured live 2026-08-10: "i'm pasting a note a colleague sent me, just
# summarise it for me: --- BEGIN NOTE --- ... --- END NOTE ---" was classified
# as a grounded follow-up (the word "summarise" plus a prior turn's web_search
# evidence), the WHOLE message was handed to the search engine as the query,
# and the reply was a product page for an online summarising tool. The note
# itself was never read.
#
# Two things go wrong when material is present and unrecognised, and both are
# fixed by locating the material:
#   1. Words INSIDE the material manufacture search triggers that the user
#      never asked for — a colleague writing "the latest policy version" is
#      not a request to look up policy versions.
#   2. The referent of "it"/"this" is the pasted block, not a previous web
#      fetch, so follow-up grounding is reading the wrong antecedent.
#
# Detection is structural first (paired fences, blockquotes) and only falls
# back to prose announcements when those name supplied content, so that "here's
# the question: what's the latest release?" stays a search.

_SUPPLIED_MATERIAL_FENCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    for pattern in (
        # --- BEGIN NOTE --- ... --- END NOTE ---  /  [BEGIN] ... [END]
        r"^[ \t]*[-=*_\[]{2,}[ \t]*BEGIN\b[^\n]*\n(?P<body>.*?)\n[ \t]*[-=*_\[]{2,}[ \t]*END\b[^\n]*$",
        # ```lang ... ```
        r"^[ \t]*```[^\n]*\n(?P<body>.*?)\n[ \t]*```[ \t]*$",
        # \"\"\" ... \"\"\"
        r"\"\"\"(?P<body>.*?)\"\"\"",
        # <<< ... >>>
        r"<<<(?P<body>.*?)>>>",
    )
)

_BLOCKQUOTE_RUN_RE = re.compile(r"(?:^[ \t]*>[ \t]?[^\n]*(?:\n|$))+", re.MULTILINE)

#: Nouns that name content the user is HANDING OVER rather than asking about.
_SUPPLIED_MATERIAL_CONTENT_NOUNS = (
    "note", "notes", "text", "message", "email", "e-mail", "mail", "letter",
    "memo", "paragraph", "passage", "excerpt", "snippet", "transcript",
    "draft", "copy", "quote", "quotation", "blurb", "abstract", "review",
    "comment", "post", "thread", "article", "entry", "content", "wording",
    "writeup", "write-up", "description", "brief", "blurb", "bio", "readme",
    "log", "logs", "output", "error", "traceback", "stacktrace", "snippet",
)

_SUPPLIED_MATERIAL_NOUN_ALTERNATION = "|".join(
    re.escape(noun) for noun in sorted(set(_SUPPLIED_MATERIAL_CONTENT_NOUNS), key=len, reverse=True)
)

_SUPPLIED_MATERIAL_ANNOUNCEMENT_RE = re.compile(
    # A paste/attach/forward verb is self-announcing.
    r"\b(?:pasting|pasted|paste|attaching|attached|forwarding|forwarded|"
    r"copied|copying|quoting)\b"
    # Otherwise the phrase has to name the content being handed over.
    r"|\b(?:here(?:'s|s| is| are)|below (?:is|are)|this is|these are|"
    r"the following|following is|i got|i received|they sent me|he sent me|"
    r"she sent me|someone sent me)\b"
    rf"[^\n:]{{0,60}}?\b(?:{_SUPPLIED_MATERIAL_NOUN_ALTERNATION})\b",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")

#: A block has to be more than a pointer or a stray word to count as material.
_MIN_SUPPLIED_MATERIAL_WORDS = 3


def _is_substantive_material(body: str) -> bool:
    """A block counts only if it carries prose, not just a link."""
    stripped = _URL_RE.sub(" ", str(body or "")).strip()
    if not stripped:
        # A bare URL is a POINTER to content, not content. Fetching is the
        # right lane for those, so they must not read as supplied material.
        return False
    return len(stripped.split()) >= _MIN_SUPPLIED_MATERIAL_WORDS


@dataclass(frozen=True)
class SuppliedMaterial:
    """Content the user carried into the turn, split from their instruction."""

    #: Each pasted/quoted/attached block, in the order it appeared.
    blocks: tuple[str, ...] = ()
    #: The message with those blocks removed — what the user is ASKING.
    instruction_text: str = ""

    @property
    def has_material(self) -> bool:
        return bool(self.blocks)


def extract_supplied_material(text: str) -> SuppliedMaterial:
    """Split a turn into the material it carries and the instruction about it."""
    raw = str(text or "")
    if not raw.strip():
        return SuppliedMaterial(blocks=(), instruction_text="")

    blocks: list[str] = []
    remainder = raw

    def _consume(pattern: re.Pattern[str]) -> None:
        nonlocal remainder
        kept: list[str] = []
        cursor = 0
        for match in pattern.finditer(remainder):
            try:
                body = match.group("body")
            except IndexError:
                body = match.group(0)
            if not _is_substantive_material(body):
                continue
            blocks.append(body.strip())
            kept.append(remainder[cursor:match.start()])
            cursor = match.end()
        if cursor:
            kept.append(remainder[cursor:])
            remainder = "\n".join(part for part in kept if part.strip())

    for fence in _SUPPLIED_MATERIAL_FENCE_PATTERNS:
        _consume(fence)
    _consume(_BLOCKQUOTE_RUN_RE)

    if not blocks:
        announcement = _SUPPLIED_MATERIAL_ANNOUNCEMENT_RE.search(remainder)
        if announcement is not None:
            # The material begins at the first clause break after the
            # announcement; an announcement with nothing after it is just talk.
            tail_offset = announcement.end()
            separator = re.search(r"[:\n]", remainder[tail_offset:])
            if separator is not None:
                split_at = tail_offset + separator.end()
                body = remainder[split_at:]
                if _is_substantive_material(body):
                    blocks.append(body.strip())
                    remainder = remainder[:split_at]

    return SuppliedMaterial(
        blocks=tuple(blocks),
        instruction_text=" ".join(remainder.split()).strip(),
    )


def carries_supplied_material(text: str) -> bool:
    """True when the turn hands over the content it is asking about."""
    return extract_supplied_material(text).has_material


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
    imperative_parts: int = 0
    prefers_extended_answer: bool = False
    requires_single_reply_coverage: bool = False
    #: The actual text of each ask, not just how many there were.
    #:
    #: The count alone can shape a prompt ("3 parts detected") and size a
    #: voice budget, and it cannot check whether a reply covered them —
    #: checking needs to know WHAT was asked. Retained so
    #: validate_dialogue_response can hold the answer against the question.
    question_segments: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, int | bool | tuple[str, ...]]:
        return {
            "question_parts": self.question_parts,
            "explicit_question_marks": self.explicit_question_marks,
            "question_like_lines": self.question_like_lines,
            "connector_parts": self.connector_parts,
            "repeated_clause_parts": self.repeated_clause_parts,
            "numbered_parts": self.numbered_parts,
            "imperative_parts": self.imperative_parts,
            "prefers_extended_answer": self.prefers_extended_answer,
            "requires_single_reply_coverage": self.requires_single_reply_coverage,
            "question_segments": self.question_segments,
        }


def answer_surface_token_floor(text: str) -> int:
    """Minimum decode capacity needed to answer the visible request once.

    This is a structural capacity calculation, not an instruction to the
    model.  A five-part request cannot fit through the same answer surface as
    a closed question, regardless of sampling style or affective state.  The
    floor is intentionally coarse and rounded to allocator-sized blocks; EOS
    may still finish early, so granting capacity does not force verbosity.
    """

    shape = analyze_prompt_shape(text)
    obligations = max(
        1,
        int(shape.question_parts),
        int(shape.numbered_parts),
        int(shape.imperative_parts),
        len(shape.question_segments),
    )
    if not (
        shape.prefers_extended_answer
        or shape.requires_single_reply_coverage
        or obligations >= 2
    ):
        return 256

    # Reserve capacity by required work, not merely by clause count. A worked
    # example and executable pseudocode each need substantially more surface
    # than a one-line definition. Cardinality, paired comparisons and a named
    # alternative add independently verifiable content. This is admission
    # accounting only: semantic EOS can still return as soon as the work is
    # complete.
    lowered = str(text or "").lower()
    required = 256 + (192 * obligations)
    if re.search(r"\b(?:pseudo\s*code|code|algorithm|procedure)\b", lowered):
        required += 384
    if re.search(r"\b(?:worked|concrete|step[- ]by[- ]step)\s+example\b", lowered):
        required += 384
    if re.search(
        r"\b(?:at\s+least|minimum(?:\s+of)?|no\s+fewer\s+than)\s+"
        r"(?:[a-z-]+|\d+)\b",
        lowered,
    ):
        required += 128
    if re.search(r"\b(?:both|each\s+of|compare|contrast)\b", lowered):
        required += 128
    if re.search(r"\b(?:correct|proper|recommended)\s+alternative\b", lowered):
        required += 128
    if shape.prefers_extended_answer and obligations == 1:
        required = max(required, 512)
    block = 128
    return min(4096, max(384, ((required + block - 1) // block) * block))


#: Splits an utterance into the units a person would count as separate asks:
#: sentence enders, and the line breaks / numbered items that carry a list.
_ASK_SPLIT_RE = re.compile(r"(?<=[.?!])\s+|\n+")


def _inline_numbered_segments(text: str) -> tuple[str, ...]:
    """Return a contiguous ``(1) ... (N) ...`` obligation list.

    Parenthesized numbers also occur in prose and mathematics, so a single
    marker is not structural evidence.  A list is admitted only when it starts
    at one and advances without gaps.  This lets chat messages carry compact
    inline checklists without turning ``f(2)`` or a citation such as ``(3)``
    into a multi-part request.
    """

    raw = str(text or "")
    matches = tuple(_INLINE_NUMBERED_ITEM_RE.finditer(raw))
    if len(matches) < 2:
        return ()
    numbers = tuple(int(match.group("number")) for match in matches)
    if numbers != tuple(range(1, len(matches) + 1)):
        return ()

    segments: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        segment = raw[match.end() : end].strip(" \t\r\n,;:.!")
        if not segment:
            return ()
        segments.append(segment)
    return tuple(segments)


def _coordinated_directive_segments(text: str) -> tuple[str, ...]:
    """Split one sentence that carries several independently requested acts.

    People normally write ``explain X, give Y, state Z, and name W`` rather
    than numbering those obligations.  Counting the directive verbs already
    made that sentence *look* multipart, but the actual obligation text was
    discarded, leaving the completion verifier with nothing to check.  Keep
    each verb phrase as a first-class segment.  A single directive remains a
    normal sentence so noun coordination (``compare X and Y``) is untouched.
    """

    raw = str(text or "").strip()
    matches = tuple(_COORDINATED_DIRECTIVE_RE.finditer(raw))
    if len(matches) < 2:
        return ()

    segments: list[str] = []
    for index, match in enumerate(matches):
        start = match.start("directive")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        segment = raw[start:end].strip(" \t\r\n,;:.!?")
        if not segment:
            return ()
        segments.append(segment)
    return tuple(segments)


def _question_segments(text: str) -> tuple[str, ...]:
    """The individual asks in an utterance, as text.

    LIVE DEFECT, 2026-08-10. "give me one concrete example of a preposition
    doing more work than it should. and separately — do you actually enjoy
    that, or is 'interesting' a word you reach for because it's safe?" She
    answered the example and said nothing whatever about enjoyment. The same
    failure was called out earlier in the same session — "you dodged half of
    it. I asked two things and you answered one."

    The runtime already KNEW it was compound: question_parts was computed,
    the prompt was told "this prompt contains multiple asks (2 detected)",
    and the voice budget was widened for it. Nothing ever checked the reply
    against it, because the count was all that survived analysis. Keeping the
    segments is what makes coverage checkable at all.

    An ask is a SENTENCE that either ends in a question mark or opens with a
    directive verb. Sentences, not lines, because everything else here counts
    per line and that is what missed the case above: it arrived as one line,
    so _INTERROGATIVE_LINE_RE — which requires the LINE to begin with what,
    why, do, is — never matched, the line began with "give", and a two-part
    utterance scored one part. Anyone typing in a chat box writes several
    sentences on one line constantly.
    """
    raw = str(text or "").strip()
    if not raw:
        return ()
    segments = [part.strip() for part in _ASK_SPLIT_RE.split(raw) if part.strip()]
    sentence_asks: list[str] = []
    for part in segments:
        coordinated = _coordinated_directive_segments(part)
        if coordinated:
            sentence_asks.extend(coordinated)
        elif part.endswith("?") or _DIRECTIVE_LINE_RE.match(part):
            sentence_asks.append(part)
    inline_obligations = _inline_numbered_segments(raw)
    if not inline_obligations:
        return tuple(sentence_asks)

    # The sentence containing the inline list is a container for the same
    # obligations, not an additional request. Keep unrelated asks before it,
    # then expose each numbered requirement independently to coverage checks.
    first_marker = _INLINE_NUMBERED_ITEM_RE.search(raw)
    prefix = raw[: first_marker.start()] if first_marker is not None else ""
    prefix_asks = tuple(
        part
        for part in _question_segments(prefix)
        if part not in inline_obligations
    )
    return (*prefix_asks, *inline_obligations)


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
    inline_numbered_segments = _inline_numbered_segments(raw)
    numbered_parts = max(
        len(_NUMBERED_ITEM_RE.findall(raw)),
        len(inline_numbered_segments),
    )
    repeated_clause_parts = max(0, len(_REPEATED_CLAUSE_RE.findall(raw)) - 1)
    imperative_parts = len(_COORDINATED_DIRECTIVE_RE.findall(raw))

    ask_segments = _question_segments(raw)

    part_candidates = [
        1,
        # Sentence-level asks. Every other candidate below counts per LINE or
        # per verb list, and a chat box is one line: "give me an example of X.
        # and separately — do you enjoy it?" scored 1 part, so the prompt was
        # never told it was compound and the reply dropped half of it.
        len(ask_segments),
        explicit_question_marks,
        question_like_lines,
        numbered_parts,
        connector_parts + 1 if connector_parts else 0,
        repeated_clause_parts + 1 if repeated_clause_parts else 0,
        directive_lines if directive_lines >= 2 else 0,
        imperative_parts if imperative_parts >= 2 else 0,
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
        question_segments=ask_segments,
        question_parts=question_parts,
        explicit_question_marks=explicit_question_marks,
        question_like_lines=question_like_lines,
        connector_parts=connector_parts,
        repeated_clause_parts=repeated_clause_parts,
        numbered_parts=numbered_parts,
        imperative_parts=imperative_parts,
        prefers_extended_answer=prefers_extended_answer,
        requires_single_reply_coverage=requires_single_reply_coverage,
    )
