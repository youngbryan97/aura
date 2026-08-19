"""Markers are words, not substrings.

The same defect has been found and fixed one site at a time, repeatedly:

    "in your own words"      launched Microsoft Word
    "notes.txt"              opened the Notes app
    "the latest Claude model" opened a browser conversation with Claude
    "i dont know what to do"  classified as a real-time news query
    "how do you distinguish"  classified as a practical GUI diagnostic

Each was a keyword list tested with `marker in text`. Containment does not
know where words begin, so any marker that is a fragment of an ordinary word
will eventually meet that word — and the shorter and more useful the marker,
the sooner. Widening the list afterwards never helps, because the next
collision is a different word.

`names_any` is the same test done on word boundaries. Stems still work the way
their authors intended: "run" matches "running", because a word that BEGINS
with the marker is that marker inflected. What stops matching is the marker
buried mid-word, where it belongs to something else entirely.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

__all__ = ["names_any", "names_marker", "stem_fold", "which_markers"]


@lru_cache(maxsize=4096)
def _pattern(marker: str) -> re.Pattern[str] | None:
    text = str(marker or "").strip().lower()
    if not text:
        return None
    # A marker may be a phrase ("have a conversation") or carry punctuation
    # ("can "). Anchor only the ends that are word characters, so a marker
    # written with its own spacing keeps meaning what it meant.
    lead = r"\b" if text[0].isalnum() else ""
    # Trailing inflection is the stem doing its job: "run" claims "running".
    trail = r"\w*\b" if text[-1].isalnum() else ""
    return re.compile(lead + re.escape(text) + trail)


def names_marker(text: str, marker: str) -> bool:
    """Does `text` use `marker` as a word, rather than inside one?"""
    pattern = _pattern(marker)
    if pattern is None:
        return False
    return bool(pattern.search(str(text or "").lower()))


def names_any(text: str, markers: Iterable[str]) -> bool:
    probe = str(text or "").lower()
    return any(
        (pattern := _pattern(marker)) is not None and pattern.search(probe)
        for marker in markers
    )


def which_markers(text: str, markers: Iterable[str]) -> list[str]:
    """The markers actually used, for a message that has to name its reason."""
    probe = str(text or "").lower()
    return [
        marker
        for marker in markers
        if (pattern := _pattern(marker)) is not None and pattern.search(probe)
    ]


@lru_cache(maxsize=8192)
def stem_fold(word: str) -> str:
    """One key for a word and its inflections, for matching word against word.

    `names_any` handles inflection by anchoring the start of a marker and
    letting the end run, which works when one side is a known stem. When both
    sides are ordinary prose — a question against a capability description —
    neither is the stem, and "reverse a string" misses "reversing ... a given
    string" over the last three letters.

    The rules are deliberately shallow. Anything deeper starts merging words
    that mean different things, and a false capability match is worse than a
    missed one.
    """
    text = re.sub(r"[^a-z]", "", str(word or "").lower())
    if len(text) <= 3:
        return text
    for suffix in ("ing", "ed"):
        if text.endswith(suffix) and len(text) - len(suffix) >= 3:
            text = text[: -len(suffix)]
            # running -> runn -> run
            if len(text) > 3 and text[-1] == text[-2] and text[-1] not in "aeiou":
                text = text[:-1]
            break
    else:
        if text.endswith("ies") and len(text) > 4:
            text = text[:-3] + "y"
        elif text.endswith("es") and len(text) > 4:
            text = text[:-2]
        elif text.endswith("s") and not text.endswith("ss") and len(text) > 3:
            text = text[:-1]
    # reverse -> revers, so that reversing -> revers meets it.
    if text.endswith("e") and len(text) > 3:
        text = text[:-1]
    return text
