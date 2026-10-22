"""core/voice/duplex/clause_chunker.py — Cutting text into speakable pieces.

Perceived latency is dominated by *time to first audio*, not by total
synthesis time. So the first chunk is cut as short as prosody allows — a
three-word opener starts the reply ~150 ms after the text arrives, and by
the time it finishes playing the rest is already synthesised.

Later chunks are longer on purpose: a TTS model's prosody comes from the
context inside the chunk, so cutting everything into three-word fragments
makes the whole reply sound choppy and breathless.

The rule set matters more than it looks. Splitting on "." alone breaks
"Dr. Chen" and "3.5 seconds" into separate utterances, each with its own
sentence-final intonation, which is instantly audible as wrong.
"""
from __future__ import annotations

import re

# Abbreviations whose trailing period is not a sentence end.
_ABBREVIATIONS = frozenset(
    """
    mr mrs ms dr prof sr jr st rev hon gen col sgt lt capt
    inc ltd co corp dept est fig vs etc al eg ie approx
    jan feb mar apr jun jul aug sep sept oct nov dec
    mon tue tues wed thu thurs fri sat sun
    a.m p.m am pm no vol pp ch sec min hr
    """.split()
)

# Strongest to weakest break points.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_CLAUSE_BREAK = re.compile(r"(?<=[;:—–])\s+|\s+(?=—|–)")
_COMMA_BREAK = re.compile(r"(?<=,)\s+")


def _is_false_sentence_end(text: str, index: int) -> bool:
    """True when the period at ``index`` is an abbreviation or a decimal."""
    if index <= 0 or index >= len(text):
        return False
    if text[index - 1] != ".":
        return False

    # Decimal number: digit '.' digit
    if index < len(text) and text[index - 2 : index - 1].isdigit():
        after = text[index : index + 1]
        if after.isdigit():
            return True

    # Trailing word before the period
    prefix = text[: index - 1]
    match = re.search(r"([A-Za-z.]+)$", prefix)
    if not match:
        return False
    word = match.group(1).lower().strip(".")
    if word in _ABBREVIATIONS:
        return True
    # Single initial, e.g. "J. R. R."
    return len(word) == 1


def completed_sentence_prefix(text: str, *, first_only: bool = False) -> str:
    """Keep only sentence boundaries confirmed by following whitespace."""
    end = 0
    for match in _SENTENCE_END.finditer(text):
        if _is_false_sentence_end(text, match.start()):
            continue
        # An enumerator starts a list item; its period does not finish one.
        sentence = text[end:match.start()].strip()
        if re.fullmatch(r"\d+\.", sentence):
            continue
        end = match.end()
        if first_only:
            break
    return text[:end]


def _split_at(pattern: re.Pattern[str], text: str) -> list[str]:
    parts: list[str] = []
    last = 0
    for m in pattern.finditer(text):
        idx = m.start()
        if pattern is _SENTENCE_END and _is_false_sentence_end(text, idx):
            continue
        piece = text[last : m.end()].strip()
        if piece:
            parts.append(piece)
        last = m.end()
    tail = text[last:].strip()
    if tail:
        parts.append(tail)
    return parts or ([text.strip()] if text.strip() else [])


def _hard_wrap(text: str, limit: int) -> list[str]:
    """Last resort: break on whitespace, never mid-word."""
    words = text.split()
    out: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > limit:
            out.append(current)
            current = word
        else:
            current = candidate
    if current:
        out.append(current)
    return out


def split_for_speech(text: str, *, max_chars: int) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars``, best break first."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    for sentence in _split_at(_SENTENCE_END, text):
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue
        for clause in _split_at(_CLAUSE_BREAK, sentence):
            if len(clause) <= max_chars:
                chunks.append(clause)
                continue
            for comma_part in _split_at(_COMMA_BREAK, clause):
                if len(comma_part) <= max_chars:
                    chunks.append(comma_part)
                else:
                    chunks.extend(_hard_wrap(comma_part, max_chars))

    return [c for c in (p.strip() for p in chunks) if c]


def first_chunk(text: str, *, max_chars: int) -> tuple[str, str]:
    """Cut the shortest sane opener off the front. Returns (head, rest).

    Prefers a real boundary inside the budget; falls back to a word break.
    A one- or two-word head is allowed only when it is a genuine sentence
    ("Yes." / "Not quite.") — otherwise the reply starts on a fragment and
    sounds like a stutter.
    """
    text = (text or "").strip()
    if not text:
        return "", ""
    if len(text) <= max_chars:
        return text, ""

    window = text[: max_chars + 1]
    best = -1
    # Prefer a sentence end, then a clause break, then a comma.
    for pattern in (_SENTENCE_END, _CLAUSE_BREAK, _COMMA_BREAK):
        for m in pattern.finditer(window):
            if pattern is _SENTENCE_END and _is_false_sentence_end(window, m.start()):
                continue
            best = max(best, m.end())
        if best > 0:
            break

    if best <= 0:
        cut = window.rfind(" ")
        best = cut if cut > 0 else max_chars

    head = text[:best].strip()
    rest = text[best:].strip()
    if len(head.split()) < 2 and rest:
        # Too short to carry prosody on its own; take the next break too.
        extra_head, extra_rest = first_chunk(rest, max_chars=max_chars)
        return f"{head} {extra_head}".strip(), extra_rest
    return head, rest


class StreamingChunker:
    """Accumulates streamed text and yields chunks as soon as they are safe.

    "Safe" means the chunk ends at a real boundary, so a chunk is never
    emitted while the next token could still change its punctuation. The
    first chunk uses a smaller budget to get audio out fast.
    """

    def __init__(self, *, first_max_chars: int, max_chars: int) -> None:
        self._first_max = first_max_chars
        self._max = max_chars
        self._buffer = ""
        self._emitted_first = False

    def push(self, text: str) -> list[str]:
        """Add streamed text; return any chunks that are now complete."""
        if not text:
            return []
        # The retained remainder is stripped when a chunk is cut, so a naive
        # append fuses the last word of one token onto the first word of the
        # next ("median." + "Most" -> "median.Most"), which the TTS then
        # pronounces as a single mangled word.
        if (
            self._buffer
            and not self._buffer[-1].isspace()
            and not text[0].isspace()
        ):
            self._buffer += " "
        self._buffer += text
        out: list[str] = []
        # Every successful split shortens the buffer by at least one
        # character, so its initial length is a strict upper bound. Keeping
        # the bound explicit prevents a malformed splitter from pinning the
        # voice event loop.
        max_splits = max(1, len(self._buffer))
        for _ in range(max_splits):
            budget = self._max if self._emitted_first else self._first_max
            if not self._emitted_first:
                opener = completed_sentence_prefix(self._buffer, first_only=True)
                if opener and len(opener.strip()) <= budget:
                    out.append(opener.strip())
                    self._buffer = self._buffer[len(opener):]
                    self._emitted_first = True
                    continue
            if len(self._buffer) <= budget:
                break
            prior_length = len(self._buffer)
            head, rest = first_chunk(self._buffer, max_chars=budget)
            if not head:
                break
            if len(rest) >= prior_length:
                raise RuntimeError("speech chunker failed to make forward progress")
            out.append(head)
            self._buffer = rest
            self._emitted_first = True
        else:
            raise RuntimeError("speech chunker exceeded its bounded split count")
        return out

    def flush(self) -> list[str]:
        """Emit whatever remains at end of stream."""
        remaining = self._buffer.strip()
        self._buffer = ""
        if not remaining:
            return []
        budget = self._max if self._emitted_first else self._first_max
        self._emitted_first = True
        return split_for_speech(remaining, max_chars=budget)
