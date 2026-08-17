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
        return None
    kind = (match.group("kind") or "").strip().lower()
    suffix = _KIND_SUFFIXES.get(kind, "")
    if kind and not suffix:
        # An unrecognised qualifier ("how many test files") would silently
        # become "all files", which answers a different question.
        return None
    target = _resolve(match.group("path"))
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


def requested_file_read(user_message: Any) -> FileRead | None:
    """Read the file this message names, or None when it names none.

    LIVE 2026-08-17: "read the file CONTRIBUTING.md and tell me the first rule
    it states" was answered "I don't have a clean grounded answer on that yet."
    The file is in the repo root and she has five skills that can read it. The
    capability was never the problem; nothing executed.
    """

    text = " ".join(str(user_message or "").split())
    if not text:
        return None
    match = _READ_RE.search(text)
    if not match:
        return None
    candidate = match.group("path").strip().strip(".,;:'\"")
    if candidate.startswith("/") or candidate.startswith("~") or ".." in candidate:
        return None
    for root in _allowed_roots():
        try:
            target = (root / candidate).resolve()
        except (OSError, ValueError, RuntimeError):
            continue
        if not str(target).startswith(str(root)):
            continue
        if not target.is_file():
            continue
        try:
            body = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        return FileRead(
            path=str(target),
            text=body[:READ_CHAR_BUDGET],
            exists=True,
            truncated=len(body) > READ_CHAR_BUDGET,
        )
    return FileRead(path=candidate, text="", exists=False)
