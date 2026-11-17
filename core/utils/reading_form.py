"""Text as a reader of words sees it: typographic punctuation folded to the ASCII its patterns are written in.

Her model writes typographic punctuation: "can’t" with U+2019, quotations in U+201C and U+201D. The readers that
judge her replies are written against ASCII, "can'?t", and a pattern written that way does not match the word she
wrote. LIVE 17 November 2026, 09:16, a private window in front: her reply opened "I can’t read your screen right
now", and her reply check took it for a claim to have read the screen, because its pattern for an admission that she
could not read it knew only the ASCII apostrophe.

The fold is one character for one character, so an offset found in the reading form is the same offset in the text
as written, and a reader that cuts or quotes a span can still cut it from the original. Nothing else changes: no
case, no width, no dashes, no spacing other than the non-breaking kinds.
"""

from __future__ import annotations

from typing import Any

__all__ = ["as_read"]

_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "ʼ": "'", "′": "'", "＇": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "＂": '"',
    " ": " ", " ": " ", " ": " ",
})


def as_read(text: Any) -> str:
    """``text`` with typographic apostrophes, quotation marks and non-breaking spaces as their ASCII forms; the same
    length as ``str(text)``."""
    return str(text or "").translate(_FOLD)
