"""Which filesystem paths a message names.

LIVE, 2026-08-25: "Something weird is happening in a little project of mine at
/private/tmp/.../invoice-tools. There's no error and no failing test, but the
second invoice comes out with the first one's lines in it." The reply was a
capability inventory — "79 registered entries; 79 entries explicitly marked
available" — and a declaration that no tool would be run.

The path was captured as `invoice-tools.` with the sentence's full stop on the
end, so it did not resolve, so nothing pointed at anything real, so no tool was
selected, so the model was asked to answer from nothing and its attempts were
exhausted into a canned catalogue.

A path can contain a dot and will not end with one in prose. Six regexes in
this tree said otherwise, each written where it was needed. This is the one
that is read from now on, so the next punctuation mark is fixed once.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "first_existing_path",
    "named_paths",
    "points_at_a_file",
    "trim_sentence_punctuation",
]

#: A path-looking span. Deliberately generous: what it captures is trimmed and
#: then tested against the disk, so a false capture costs a stat call.
_PATH_SPAN = re.compile(r"(?<![\w])(~?/[\w.\-~/]*[\w.\-~])")

#: Marks that end a sentence or bracket a phrase, never a path in prose.
_TRAILING = ".,;:!?)]}>\"'`"

#: The same marks where they open rather than close.
_LEADING = "([{<\"'`"


def trim_sentence_punctuation(candidate: object) -> str:
    """A candidate with the prose stripped off both ends."""
    text = str(candidate or "").strip()
    while text and text[0] in _LEADING:
        text = text[1:]
    while text and text[-1] in _TRAILING:
        text = text[:-1]
    return text


def named_paths(text: object) -> tuple[str, ...]:
    """Every path this message names, longest first, punctuation removed.

    Both forms are returned when they differ — the raw capture and the trimmed
    one — because a file really can be called `notes.` and the disk is the
    thing that settles it.
    """
    found: list[str] = []
    for raw in _PATH_SPAN.findall(str(text or "")):
        for candidate in (raw, trim_sentence_punctuation(raw)):
            if candidate and candidate not in found:
                found.append(candidate)
    found.sort(key=len, reverse=True)
    return tuple(found)


def first_existing_path(text: object) -> Path | None:
    """The first named path that is actually on this disk."""
    for candidate in named_paths(text):
        try:
            resolved = Path(candidate).expanduser()
            if resolved.exists():
                return resolved
        except (OSError, ValueError):
            continue
    return None


#: A file being pointed AT, rather than the word for one appearing.
#:
#: The role is what makes it a reference: an article or a possessive in front
#: of the noun, a preposition placing something in it, or a verb that acts on
#: one. "the file", "my config", "in the log", "read the document".
#:
#: Without the role it is just a word. `core/conversation/filesystem_check.py`
#: matched the bare noun, so "what file format do you use?" and "my source of
#: truth" both said an unresolved dotted token nearby was a filename. That is
#: the failure this repository has recorded a hundred and thirteen times and
#: formed into a constraint: a token is not a decision.
_A_FILE_NOUN = (
    r"file|path|document|spreadsheet|workbook|script|source|readme|"
    r"configuration|config|log"
)
_POINTS_AT_ONE = re.compile(
    r"(?:"
    # Determined or possessed, with room for an adjective or two.
    r"\b(?:the|a|an|this|that|these|those|my|your|our|its|his|her|their)\s+"
    r"(?:[\w-]+\s+){0,2}"
    # Or placed somewhere.
    r"|\b(?:in|at|from|into|inside|within)\s+(?:the\s+|a\s+|an\s+|my\s+|your\s+)?"
    r"(?:[\w-]+\s+){0,1}"
    # Or acted on.
    r"|\b(?:read|open|write|edit|check|show|load|save|delete|create|find|"
    r"list|inspect|parse)\s+(?:the\s+|a\s+|an\s+|my\s+|your\s+)?"
    r"(?:[\w-]+\s+){0,1}"
    r")"
    rf"(?:{_A_FILE_NOUN})s?\b",
    re.IGNORECASE,
)


def points_at_a_file(text: object) -> bool:
    """Whether this message points at a file, by a path or by role.

    True when it names an actual path, and true when it refers to one the way
    people do — "the file", "my config", "read the document". False when a
    file word merely occurs, which is what "what file format do you use?" is.
    """

    body = str(text or "")
    if not body.strip():
        return False
    if named_paths(body):
        return True
    return bool(_POINTS_AT_ONE.search(body))
