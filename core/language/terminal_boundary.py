"""Mechanical sentence-boundary evidence for generated user-facing text."""

from __future__ import annotations

from typing import Any

_TERMINAL_PUNCTUATION = frozenset(".!?…")
_CLOSING_WRAPPERS = frozenset(('"', "'", "”", "’", ")", "]", "}"))


def terminal_content(text: Any) -> str:
    """Return the visible text with terminal closing syntax unwrapped."""

    body = str(text or "").rstrip()
    while body and body[-1] in _CLOSING_WRAPPERS:
        body = body[:-1].rstrip()
    return body


def has_terminal_sentence_boundary(text: Any) -> bool:
    """Return whether the visible tail contains actual terminal punctuation.

    Closing syntax is not itself a sentence boundary. In particular, a model
    ending on the quoted word ``“closest”`` has closed a quotation but has not
    finished the surrounding sentence. Unwrap closing quotes or brackets, then
    require punctuation beneath them. This is deliberately stricter than an
    EOS judgement: callers use it to decide whether they may stop decoding
    before EOS or badge a clipped draft as physically complete.
    """

    body = terminal_content(text)
    if not body:
        return False
    return body[-1] in _TERMINAL_PUNCTUATION


__all__ = ["has_terminal_sentence_boundary", "terminal_content"]
