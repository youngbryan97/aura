"""An answer asked for in a typed shape is judged by the parser of that shape.

The gate's integrity reader is a reader of prose: a list of forty similar
items reads to it as a loop, and an answer cut off by its budget as an
unfinished sentence. A typed answer has its own parser, which keeps the
finished items of one cut off. LIVE 2026-10-05 a list of checks for four
features was refused as "low_lexical_diversity_loop, truncated_tail" after
418 seconds, and nothing reached the parser that would have kept twenty.

Nor is its length the body's to shorten. Under strain the gate shortens
answers, which suits a reply in words; an object or a piece of code cut short
does not parse, and is asked for again at full cost. LIVE 2026-10-05 after a
memory alarm a part of a program asked for at 2048 tokens went out at 429.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["judged_by_its_parser", "keeps_its_length"]

#: What a reader of prose objects to that the parser of a shape settles itself.
_THE_PARSER_SETTLES = frozenset({
    "low_lexical_diversity_loop", "repetitive_phrase_loop", "low_information_loop", "truncated_tail",
    # "\n" inside a JSON string is how JSON writes a line break, not a leak.
    "escaped_control_artifact",
})


def judged_by_its_parser(asked: Mapping[str, Any], reasons: Iterable[str]) -> bool:
    """Whether this answer was asked for in a typed shape and objected to only for what that shape's parser settles."""
    return _typed(asked) and set(reasons) <= _THE_PARSER_SETTLES


def _typed(asked: Mapping[str, Any]) -> bool:
    return asked.get("schema") is not None or str(asked.get("output_shape") or "") == "json_object"


def keeps_its_length(asked: Mapping[str, Any], max_tokens: int) -> int:
    """The length a typed answer's caller asked for, whatever the moment made of it; any other answer's as it is."""
    try:
        wanted = int(asked.get("max_tokens") or 0)
    except (TypeError, ValueError, OverflowError):
        return max_tokens
    return wanted if _typed(asked) and wanted > 0 else max_tokens
