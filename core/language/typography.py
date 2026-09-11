"""Fold the punctuation a model writes into the punctuation patterns expect.

LIVE, 2026-08-22. A check that strikes a refuted claim about her own uptime
did not fire on "I’ve been up since 0600", because the pattern was written
with an ASCII apostrophe and the reply carried the typographic one. The rule
was right, the text was right, and they could not meet.

This is not one pattern's problem. Anything in this tree that matches
contractions, quotes or dashes against model output has the same blind spot,
and the fix belongs in one place rather than in each of them.
"""

from __future__ import annotations

__all__ = ["fold_typography", "quotation_spans"]

#: Characters a language model writes where a pattern expects ASCII.
_FOLD = {
    "‘": "'",  # left single quote
    "’": "'",  # right single quote, the one that breaks contractions
    "‚": "'",
    "‛": "'",
    "ʼ": "'",  # modifier letter apostrophe
    "′": "'",  # prime
    "“": '"',
    "”": '"',
    "„": '"',
    "″": '"',
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # minus sign
    "…": "...",
    " ": " ",  # non-breaking space
    " ": " ",  # narrow no-break space
    " ": " ",  # thin space
}

_TABLE = str.maketrans({key: value for key, value in _FOLD.items()})


def fold_typography(text: object) -> str:
    """The same text with typographic punctuation written the plain way."""
    return str(text or "").translate(_TABLE)


def quotation_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Balanced outer quotations as half-open offsets, including delimiters.

    Apostrophes inside words and escaped delimiters are not quote boundaries.
    Unclosed quotations supply no span; callers must not hide text behind one.
    """
    pairs = {'"': '"', "'": "'", "\u201c": "\u201d", "\u2018": "\u2019"}
    stack: list[tuple[str, int]] = []
    spans: list[tuple[int, int]] = []
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", "\u2019"} and index > 0 and text[index - 1].isalnum():
            if index + 1 < len(text) and text[index + 1].isalnum():
                continue
            if not stack or stack[-1][0] != char:
                continue
        if stack and char == stack[-1][0]:
            _, start = stack.pop()
            if not stack:
                spans.append((start, index + 1))
        elif char in pairs:
            stack.append((pairs[char], index))
    return tuple(spans)
