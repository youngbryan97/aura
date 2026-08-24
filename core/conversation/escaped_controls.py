"""Classify and repair literal whitespace escapes on a rendered text surface.

Backslash escapes in prose are serialization residue. The same bytes inside
code or mathematical notation are authored syntax. One parser owns that
distinction so validation and repair cannot disagree about the same draft.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

_PROTECTED_MARKUP_RE = re.compile(
    r"```[\s\S]*?(?:```|\Z)"
    r"|~~~[\s\S]*?(?:~~~|\Z)"
    r"|`[^`\n]*`"
    r"|\$\$[\s\S]*?(?:\$\$|\Z)"
    r"|(?<!\\)\$(?:\\.|[^$\n])+?(?<!\\)\$"
    r"|\\\([\s\S]*?(?:\\\)|\Z)"
    r"|\\\[[\s\S]*?(?:\\\]|\Z)",
)
_LITERAL_WHITESPACE_ESCAPE_RE = re.compile(r"(?<!\\)\\(?:r\\n|[nrt])")


def _unprotected_markup_segments(text: str) -> Iterator[tuple[int, int]]:
    """Yield prose spans outside fenced/inline code and mathematical spans."""

    cursor = 0
    for match in _PROTECTED_MARKUP_RE.finditer(text):
        if match.start() > cursor:
            yield cursor, match.start()
        cursor = match.end()
    if cursor < len(text):
        yield cursor, len(text)


def has_escaped_whitespace_artifact(value: object) -> bool:
    """Return whether prose contains a literal CR/LF/tab escape.

    The answer is syntax-bound. A ``\\text`` command inside ``$...$`` is not
    guessed from a command-name allowlist, and a ``\\n`` inside source code is
    not treated as damaged prose.
    """

    text = str(value or "")
    return any(
        _LITERAL_WHITESPACE_ESCAPE_RE.search(text, start, end) is not None
        for start, end in _unprotected_markup_segments(text)
    )


def repair_escaped_whitespace_artifacts(value: object) -> str | None:
    """Repair prose escapes while returning protected markup byte-for-byte."""

    text = str(value or "")
    if not text:
        return None
    pieces: list[str] = []
    cursor = 0
    changed = False
    for match in _PROTECTED_MARKUP_RE.finditer(text):
        prose = text[cursor : match.start()]
        repaired = _repair_prose(prose)
        changed = changed or repaired != prose
        pieces.extend((repaired, match.group(0)))
        cursor = match.end()
    prose = text[cursor:]
    repaired = _repair_prose(prose)
    changed = changed or repaired != prose
    pieces.append(repaired)
    if not changed:
        return None
    return re.sub(r"\n{3,}", "\n\n", "".join(pieces))


def _repair_prose(text: str) -> str:
    return _LITERAL_WHITESPACE_ESCAPE_RE.sub(
        lambda match: "\t" if match.group(0) == r"\t" else "\n",
        text,
    )
