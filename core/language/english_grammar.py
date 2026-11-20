"""The dependency parse of English that her readers of grammar share.

Who is to act in a sentence (what_she_is_asked_to_do), and which words only introduced an address that was taken out
(core/intent/opaque_spans.py), are read from the same parse: one model, loaded once, warmed at boot with her other
readers so no turn pays for the load.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.EnglishGrammar")

__all__ = ["PARSER", "parse", "prepositions_left_without_object", "prewarm_english_grammar"]

#: The parse that reads it. The small English model read "Beat Stockfish at chess for me" as a name ("Beat
#: Stockfish"); the medium one reads it as an imperative, and got all 39 sentences of the asked-or-talk test set right.
PARSER = "en_core_web_md"
_SMALLER = "en_core_web_sm"
_LOCK = checked_lock("english_grammar.parser")
_HELD: list[Any] = []


def _parser() -> Any | None:
    with _LOCK:
        if not _HELD:
            loaded = None
            for name in (PARSER, _SMALLER):
                try:
                    import spacy

                    loaded = spacy.load(name, disable=["ner"])
                    break
                except (ImportError, OSError) as why:
                    logger.info("English is not parsed with %s (%s)", name, why)
            _HELD.append(loaded)
        return _HELD[0]


@lru_cache(maxsize=256)
def _parsed(text: str) -> Any | None:
    parse_with = _parser()
    if parse_with is None:
        return None
    try:
        return parse_with(text)
    except (ValueError, RuntimeError) as why:
        logger.debug("not parsed (%s)", why)
        return None


def parse(text: str) -> Any | None:
    """The parse of ``text``, or None when there is no parser. A reader must not change the parse it is given: the
    same text gives the same object back."""
    words = str(text or "")
    return _parsed(words) if words.strip() else None


def prepositions_left_without_object(text: str, gaps: tuple[int, ...]) -> list[tuple[int, int]]:
    """The character spans of prepositions in ``text`` that end right before one of ``gaps`` (offsets where an address
    was taken out) and are left with no object: "how are you doing? at" once "/private/tmp/..." is gone. A word that
    only introduced an address means nothing without it, and a reader of meaning reads it as part of the sentence.
    [] when there is no parser."""
    doc = parse(text)
    if doc is None:
        return []
    spans: list[tuple[int, int]] = []
    for gap in gaps:
        before = [t for t in doc if t.idx + len(t.text) <= gap]
        if not before:
            continue
        word = before[-1]
        end = word.idx + len(word.text)
        if text[end:gap].strip():
            continue
        if word.pos_ == "ADP" and not any(c.dep_ in ("pobj", "pcomp", "obj") for c in word.children):
            spans.append((word.idx, end))
    return spans


def prewarm_english_grammar() -> bool:
    """Load the parser now, off the event loop, so the first turn read for its grammar does not pay for it."""
    return _parser() is not None
