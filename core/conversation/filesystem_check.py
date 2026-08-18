"""Count the files, do not guess how many there are.

LIVE 2026-08-17: "count the .py files in core/introspection and tell me the
number" was answered "There are 3.py files in the core/introspection
directory." There are ten. Asked again with "use your tools and give me the
exact number", she answered "There are 3 Python files" — and the log for that
turn shows no tool ran at all. The number came out of the model both times.

This is the same shape as the arithmetic case that arithmetic_check already
solves. The runtime can compute the answer exactly, in microseconds, and a
generated guess competes with a fact that was available the whole time. The
precedent there is worth copying exactly: compute it, then RE-ANSWER from the
value rather than appending a correction underneath a wrong sentence.

Deliberately narrow:

  * counting only. Nothing here reads file contents, writes, or deletes; the
    worst outcome of a bad parse is a count of a directory the person did not
    mean, which is visible in the answer.
  * paths must resolve inside the repo or her state root. "How many files are
    in /etc" is not a question this answers.
  * a path that does not exist returns a MISSING result rather than zero,
    because "there are 0 files" and "that directory is not there" are
    different answers and only one of them is honest.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "FileRead",
    "FilesystemCount",
    "requested_file_read",
    "requested_filesystem_count",
]

#: "how many python files in core/introspection", "count the .py files in X"
_COUNT_RE = re.compile(
    r"\b(?:how\s+many|count(?:\s+the)?|number\s+of)\s+"
    r"(?P<kind>[.\w+]*?)\s*"
    r"(?:files?|scripts?|modules?)\s+"
    # Anything short may sit between the noun and the preposition — "files ARE
    # in", "files LIVE in", "files SIT inside", "files do we have in". Pinning
    # this to a fixed word list is how "how many python files live in
    # core/introspection?" went unparsed and got answered "I don't have file
    # system access", while the same question with "are in" was answered
    # exactly. The preposition is the anchor; what precedes it is filler.
    r"(?:\w+\s+){0,3}?(?:in|inside|under|within)\s+"
    r"(?:the\s+)?(?P<path>[\w./\-]+)",
    re.IGNORECASE,
)

#: "how many test files do you have", "how many tests are there"
#:
#: A question with no "in <path>" clause at all. The counting pattern above
#: needs a preposition and a path, so this shape reached nothing and she was
#: left to guess — measured live 2026-08-18, against a real answer of 2444.
#:
#: It is a fair question with a determinate answer, and the place is implied by
#: the kind: her tests live in tests/. Naming that here is what makes the
#: question answerable rather than ambiguous.
_OWNED_KIND_RE = re.compile(
    r"\b(?:how\s+many|count(?:\s+the)?|number\s+of)\s+"
    r"(?P<owned>[a-z]{3,})\s*"
    r"(?:files?|scripts?|modules?|suites?)?"
    r"(?:\s+(?:do|does|have|has|are|is)\b|\s*\?|\s*$)",
    re.IGNORECASE,
)


def _home_for_kind(kind: str) -> Path | None:
    """The directory a kind of file lives in, found rather than declared.

    A table mapping "test" to "tests" and "doc" to "docs" answers exactly the
    two questions someone thought of, and is one word behind every other one —
    "how many benchmarks do you have", "how many demos", "how many configs".
    The repository already states where things live, by having directories with
    those names, so the answer is looked up on disk instead of asserted here.

    Singular and plural are both tried because English asks either way.
    """
    word = str(kind or "").strip().lower()
    if not word:
        return None
    seen: set[str] = set()
    for name in (word, f"{word}s", word.rstrip("s")):
        if not name or name in seen:
            continue
        seen.add(name)
        found = _resolve(name)
        if found is not None and found.is_dir():
            return found
    return None


def _dominant_suffix(directory: Path) -> str:
    """The extension that directory is actually made of, or "" if mixed.

    Derived from the contents rather than declared, so "how many tests do you
    have" counts .py in tests/ and would count .ts in a TypeScript project
    without anyone editing a table. A directory with no clear majority counts
    every file, which is the honest reading of "how many X do you have".
    """
    try:
        suffixes = [item.suffix for item in directory.iterdir() if item.is_file() and item.suffix]
    except OSError:
        return ""
    if not suffixes:
        return ""
    top, count = Counter(suffixes).most_common(1)[0]
    return top if count * 2 > len(suffixes) else ""

#: Words that name a language/extension rather than a real suffix.
_KIND_SUFFIXES = {
    "python": ".py",
    "py": ".py",
    ".py": ".py",
    "json": ".json",
    ".json": ".json",
    "markdown": ".md",
    "md": ".md",
    ".md": ".md",
    "text": ".txt",
    "txt": ".txt",
    ".txt": ".txt",
    "yaml": ".yaml",
    "yml": ".yml",
    "toml": ".toml",
    "rust": ".rs",
    "swift": ".swift",
    "shell": ".sh",
    "sh": ".sh",
}


@dataclass(frozen=True, slots=True)
class FilesystemCount:
    """A count the runtime took, or a named reason it could not."""

    path: str
    suffix: str
    count: int
    exists: bool
    names: tuple[str, ...] = ()

    @property
    def answerable(self) -> bool:
        return self.exists


#: Words that stand in for a place instead of naming one.
#:
#: LIVE DEFECT, 2026-08-18. "count how many files are in your own source tree"
#: captured "your" as the path and answered "There is no directory at
#: /Users/bryan/.aura/live-source/your, so there is nothing to count there."
#:
#: The path group is `[\w./\-]+`, which a pronoun satisfies perfectly, so
#: nothing downstream could tell the difference: a question about her own code
#: became a confident fact about a directory nobody had mentioned. "my
#: downloads" and "our repo" fail identically.
#:
#: A pronoun is a REFERENCE. Either it resolves to somewhere real or the
#: question is not one this can answer; inventing a directory from it is the
#: one outcome that must not happen.
_PRONOUN_PLACEHOLDERS = frozenset(
    {
        "your", "yours", "my", "mine", "our", "ours", "its", "their", "theirs",
        "his", "her", "hers", "this", "that", "these", "those", "the", "a", "an",
    }
)

#: What she means by her own code, and where it actually is.
#:
#: "your source tree" IS answerable — she has one, and it is the root this
#: module already trusts. Declining it would trade a wrong answer for a
#: needless one.
_OWN_SOURCE_RE = re.compile(
    r"\b(?:your|its|her|his|their|the)\s+(?:own\s+)?"
    r"(?:source(?:\s+(?:tree|code|dir(?:ectory)?|repo(?:sitory)?))?"
    r"|code\s?base|repo(?:sitory)?)\b",
    re.IGNORECASE,
)


def _allowed_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        roots.append(Path(__file__).resolve().parents[2])
    except (IndexError, OSError):
        pass
    try:
        from core.runtime.state_ownership import state_root

        roots.append(Path(str(state_root())).resolve())
    except (ImportError, RuntimeError, OSError, ValueError):
        pass
    return [r for r in roots if r.exists()]


def _resolve(path_text: str) -> Path | None:
    """Resolve a mentioned path inside an allowed root, or None."""

    candidate = str(path_text or "").strip().strip(".,;:'\"").rstrip("/")
    if not candidate:
        return None
    # A pronoun names no directory, and the tail of this function invents one
    # from whatever it is handed so the caller can report it missing. Saying
    # "/.../your does not exist" is a confident statement about a path the
    # person never mentioned.
    if candidate.lower() in _PRONOUN_PLACEHOLDERS:
        return None
    # An absolute path silently escapes the root check, because pathlib treats
    # `root / "/etc"` as `/etc`. "How many files are in /etc" is not a question
    # this answers, and it must not become one by accident.
    if candidate.startswith("/") or candidate.startswith("~") or ".." in candidate:
        return None
    for root in _allowed_roots():
        for probe in (root / candidate, Path(candidate)):
            try:
                resolved = probe.resolve()
            except (OSError, ValueError, RuntimeError):
                continue
            if not str(resolved).startswith(str(root)):
                continue
            if resolved.is_dir():
                return resolved
    # Name a directory that does not exist, so the caller can say so rather
    # than answering zero.
    for root in _allowed_roots():
        try:
            return (root / candidate).resolve()
        except (OSError, ValueError, RuntimeError):
            continue
    return None


def requested_filesystem_count(user_message: Any) -> FilesystemCount | None:
    """The count this message asks for, taken from disk, or None."""

    text = " ".join(str(user_message or "").split())
    if not text:
        return None
    match = _COUNT_RE.search(text)
    if not match:
        owned = _OWNED_KIND_RE.search(text)
        if owned:
            home = _home_for_kind(owned.group("owned"))
            if home is not None:
                return _count_in(home, _dominant_suffix(home))
        return None
    kind = (match.group("kind") or "").strip().lower()
    suffix = _KIND_SUFFIXES.get(kind, "")
    if kind and not suffix:
        # An unrecognised qualifier ("how many test files") would silently
        # become "all files", which answers a different question.
        return None
    target = _resolve(match.group("path"))
    if target is None and _OWN_SOURCE_RE.search(text):
        roots = _allowed_roots()
        target = roots[0] if roots else None
    if target is None:
        return None
    return _count_in(target, suffix)


def _count_in(target: Path | None, suffix: str) -> FilesystemCount | None:
    """Count files of ``suffix`` directly inside ``target``."""
    if target is None:
        return None
    if not target.is_dir():
        return FilesystemCount(
            path=str(target), suffix=suffix, count=0, exists=False
        )
    try:
        entries = sorted(
            item.name
            for item in target.iterdir()
            if item.is_file() and (not suffix or item.name.endswith(suffix))
        )
    except OSError:
        return None
    return FilesystemCount(
        path=str(target),
        suffix=suffix,
        count=len(entries),
        exists=True,
        names=tuple(entries),
    )


#: "read CONTRIBUTING.md and tell me...", "open core/config.py", "what does
#: docs/WRITING_RULES.md say about X"
_READ_RE = re.compile(
    # The verb varies more than a fixed list survives: "read X", "open X",
    # "what does X say", "check X", "grab X". Anchor on the FILENAME — a token
    # with a real extension inside her roots is the unambiguous part — and let
    # a short verb phrase precede it.
    r"\b(?:read|open|look\s+at|check|show\s+me|grab|pull\s+up|cat|"
    r"what(?:'s| is| does)|tell\s+me\s+(?:what|about))\b[^.?!]{0,40}?"
    r"(?:the\s+)?(?:file\s+)?(?P<path>[\w./\-]+\.[A-Za-z0-9]{1,6})\b",
    re.IGNORECASE,
)

#: Enough of a file to answer a question about it without pasting a codebase
#: into a chat reply.
READ_CHAR_BUDGET = 4000


@dataclass(frozen=True, slots=True)
class FileRead:
    """Text actually read off the disk, or a named reason it was not."""

    path: str
    text: str
    exists: bool
    truncated: bool = False
    #: The topic the question asked about, and how often the WHOLE file uses
    #: it. A file that mentions a subject once, in a path, does not discuss it.
    topic: str = ""
    topic_mentions: int = 0

    @property
    def barely_covers_topic(self) -> bool:
        return bool(self.topic) and self.topic_mentions <= 1


#: Anything shaped like a path with a real extension. No verb, no vocabulary.
#:
#: The predecessor demanded one of read/open/check/show me/grab/cat within 40
#: characters of the filename, so "there's a file at X" — and every other way
#: a person mentions a file in passing — matched nothing. A token that
#: RESOLVES to a real file inside her roots is evidence on its own; whether
#: the sentence around it sounded like an instruction is not the question.
_PATH_TOKEN_RE = re.compile(r"(?:[~/]|\b)[\w./\-]*[\w\-]\.[A-Za-z0-9]{1,6}\b")


def _named_paths(text: str) -> list[str]:
    """Every path-shaped token in the message, in the order written."""
    seen: set[str] = set()
    found: list[str] = []
    for raw in _PATH_TOKEN_RE.findall(str(text or "")):
        candidate = raw.strip().strip(".,;:'\"")
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        found.append(candidate)
    return found


def requested_file_read(user_message: Any) -> FileRead | None:
    """Read a file this message names, or None when it names none.

    LIVE 2026-08-17: "read the file CONTRIBUTING.md and tell me the first rule
    it states" was answered "I don't have a clean grounded answer on that yet."
    The file is in the repo root and she has five skills that can read it. The
    capability was never the problem; nothing executed.

    LIVE 2026-08-11, the same defect one phrasing over: "there's a file at
    /Users/bryan/.aura/live-source/CLAUDE.md. how many times does the word
    'degradation' show up in it?" She answered "I didn't open the file. I
    estimated based on a pattern match against recent modifications and
    environmental factors in my degradation model. The number is 0." The real
    count is 3. Two independent reasons it could not have worked:

      * the trigger required a VERB from a fixed list — read, open, check,
        show me, grab, cat — within 40 characters of the filename. "there's a
        file at X" contains none of them. This is the fourth thing in this
        codebase to ask "does the phrasing look like a request?" instead of
        "does this message reference something real", and each one fails on
        the first wording its author did not picture;

      * an ABSOLUTE path was rejected outright, so the most explicit way a
        person can name a file was the one form that could never be read.
        Containment was already enforced by resolving against her roots,
        which is the check that actually matters; refusing every "/" as well
        was blocking the legitimate case to catch the illegitimate one.

    What decides now is whether the message names a path that RESOLVES to a
    real file inside her roots. That is unambiguous evidence and needs no
    vocabulary to recognise.
    """

    text = " ".join(str(user_message or "").split())
    if not text:
        return None
    missing: str | None = None
    for candidate in _named_paths(text):
        if ".." in candidate:
            # Refused on the token itself. Containment below is the real
            # guard, but a token that is trying to escape is not worth
            # resolving at all.
            continue
        for root in _allowed_roots():
            try:
                # Absolute paths resolve as themselves, bare names against
                # each root; both then face the identical containment check.
                raw = Path(candidate).expanduser()
                target = (raw if raw.is_absolute() else (root / candidate)).resolve()
            except (OSError, ValueError, RuntimeError):
                continue
            if not str(target).startswith(str(root)):
                continue
            if not target.is_file():
                continue
            try:
                body = target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            topic, mentions = _topic_coverage(body, text, filename=candidate)
            return FileRead(
                path=str(target),
                text=_relevant_span(body, text, filename=candidate),
                exists=True,
                truncated=len(body) > READ_CHAR_BUDGET,
                topic=topic,
                topic_mentions=mentions,
            )
        # Named something file-shaped that is not there. Remember the first
        # one so a missing file is REPORTED rather than silently ignored,
        # which is how "I estimated" became an acceptable answer.
        if missing is None:
            missing = candidate
    if missing is not None:
        return FileRead(path=missing, text="", exists=False)
    return None


#: Words that say nothing about WHICH part of a file is wanted.
_SPAN_STOPWORDS = frozenset(
    """
    a about an and are as at be but by can could do does file for from get has
    have how i if in into is it its me my of on or say says show tell that the
    their them then there these they this to two under up us use was what when
    where which who why will with within would you your read open check look
    max sentences sentence words line lines please just only more most
    """.split()
)


def _relevant_span(body: str, question: str, *, filename: str = "") -> str:
    """The part of the file the question is about, not simply its opening.

    LIVE 2026-08-17: "what does ARCHITECTURE.md say about layering?" returned
    the first 4,000 characters of a 200KB spec. The word "layering" first
    appears at line 2263, far outside that window, so she answered from the
    document's introduction and described the wrong thing with confidence.

    Reading the head of a file answers "what is this file", which is a
    different question from "what does it say about X". When the question names
    a topic, the window is centred where the file actually discusses it.
    """

    if len(body) <= READ_CHAR_BUDGET:
        return body
    # The FILENAME is not a search term. "what does ARCHITECTURE.md say about
    # layering" carries the word "architecture", which occurs throughout an
    # architecture spec, so the filename outvoted the actual topic and the
    # window landed on a path list that merely name-drops check_layering.py.
    # The name identifies the file; it says nothing about which part is wanted.
    name_terms = {
        w
        for w in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", str(filename or "").lower())
    }
    terms = [
        w
        for w in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", str(question or "").lower())
        if w not in _SPAN_STOPWORDS and w not in name_terms
    ]
    if not terms:
        return body[:READ_CHAR_BUDGET]

    lowered = body.lower()
    # Score fixed windows by how many DISTINCT asked-about terms they contain,
    # tie-broken by total mentions, so a passage that discusses the topic beats
    # one that name-drops it once.
    window = READ_CHAR_BUDGET
    step = max(1, window // 4)
    best_start, best_distinct, best_hits = 0, 0, 0
    for start in range(0, max(1, len(body) - window + 1), step):
        chunk = lowered[start : start + window]
        distinct = sum(1 for t in terms if t in chunk)
        if distinct == 0:
            continue
        hits = sum(chunk.count(t) for t in terms)
        if (distinct, hits) > (best_distinct, best_hits):
            best_start, best_distinct, best_hits = start, distinct, hits
    if best_distinct == 0:
        return body[:READ_CHAR_BUDGET]
    # Start at a line boundary so the excerpt does not open mid-sentence.
    boundary = body.rfind("\n", max(0, best_start - 200), best_start + 1)
    start = boundary + 1 if boundary != -1 else best_start
    return body[start : start + window]


def _topic_coverage(body: str, question: str, *, filename: str = "") -> tuple[str, int]:
    """The topic asked about and how often the whole file uses it.

    LIVE 2026-08-17: asked what ARCHITECTURE.md says about layering, she
    described "a shaping constraint in the foreground". The file uses the word
    exactly once, inside the path `tools/check_layering.py`. Handed a passage
    containing the term, she wrote a description of a subject the document does
    not cover.

    Counting is the difference between "here is the part about X" and "this
    file barely mentions X". Both are useful; only one of them is true here,
    and the reader cannot tell them apart from an excerpt alone.
    """

    name_terms = {
        w for w in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", str(filename or "").lower())
    }
    terms = [
        w
        for w in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", str(question or "").lower())
        if w not in _SPAN_STOPWORDS and w not in name_terms
    ]
    if not terms:
        return "", 0
    lowered = str(body or "").lower()
    # The rarest asked-about term is the most specific one, and specificity is
    # what "about X" means.
    scored = sorted(((lowered.count(t), t) for t in terms), key=lambda item: item[0])
    count, term = scored[0]
    return term, count
