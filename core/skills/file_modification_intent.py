"""Adding to a file is not writing a file.

Live 2026-08-18: `append a line saying "line two" to aura-test-note.txt on my
desktop` derived one step —

    write_text_file  {"path": ..., "content": "line two\n", "overwrite": true}

— which replaces the file. The turn also said "the file now contains both
lines". Had the step dispatched, the earlier line would have been gone and the
receipt would still have read as a success, because the write really did
happen and really did verify: the wrong file content, confirmed.

The general shape is that a request naming an existing file can mean either
REPLACE or MODIFY, and only the verb says which. A planner that reads the path
and the payload but not the verb collapses the two, and it collapses them in
the destructive direction.

So the verb is recognised here, once, with the phrasings it claims and the
neighbours it disclaims, and the modes it returns are the modes the executor
implements — nothing else can be planned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["MODIFICATION_MODES", "FileModification", "requested_file_modification"]

#: What `write_text_file` can actually carry out. A mode named here that the
#: executor does not implement would plan a silent no-op.
MODIFICATION_MODES = ("append", "prepend")

# "to the end of", "at the bottom", "onto" — the position, when it is stated.
_END = r"(?:end|bottom|last\s+line|tail)"
_START = r"(?:start|beginning|top|first\s+line|head)"

_APPEND_RE = re.compile(
    r"\b(?:append|add|attach|tack|stick|put|write|insert|include)\b"
    r"(?:(?!\bto\s+the\s+" + _START + r"\b).){0,80}?"
    r"\b(?:to|onto|at|into|in|on)\s+(?:the\s+)?(?:" + _END + r"\b|(?=\S*\.\w))",
    re.IGNORECASE | re.DOTALL,
)
_APPEND_PLAIN_RE = re.compile(r"\bappend(?:ing|ed|s)?\b", re.IGNORECASE)
_PREPEND_RE = re.compile(
    r"\b(?:prepend|add|insert|put|write)\b.{0,80}?"
    r"\b(?:to|at|into)\s+(?:the\s+)?" + _START + r"\b",
    re.IGNORECASE | re.DOTALL,
)
_PREPEND_PLAIN_RE = re.compile(r"\bprepend(?:ing|ed|s)?\b", re.IGNORECASE)

# A request that says it wants a NEW file means replace, whatever else it says.
_CREATES_RE = re.compile(
    r"\b(?:create|make|generate|start|save|draft)\s+(?:a|an|the|me\s+a)?\s*"
    r"(?:new\s+)?(?:text\s+|markdown\s+|plain\s+)?(?:file|note|document|doc|md)\b",
    re.IGNORECASE,
)
# ...and one that says REPLACE means replace even though "write ... to X" would
# otherwise read as an addition.
_REPLACES_RE = re.compile(
    r"\b(?:overwrite|replace|rewrite|wipe|clear|empty|truncate|reset)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class FileModification:
    """How an existing file was asked to change."""

    mode: str

    def __post_init__(self) -> None:
        if self.mode not in MODIFICATION_MODES:
            raise ValueError(f"unimplemented modification mode: {self.mode!r}")


def requested_file_modification(text: str) -> FileModification | None:
    """Return the modification a request asks for, or None to replace.

    None is the safe default only because the caller pairs it with the
    existing create/overwrite path; it means "this does not ADD to a file",
    not "this file has no prior content".
    """
    probe = str(text or "")
    if not probe.strip():
        return None
    if _REPLACES_RE.search(probe) or _CREATES_RE.search(probe):
        return None
    if _PREPEND_PLAIN_RE.search(probe) or _PREPEND_RE.search(probe):
        return FileModification(mode="prepend")
    if _APPEND_PLAIN_RE.search(probe) or _APPEND_RE.search(probe):
        return FileModification(mode="append")
    return None
