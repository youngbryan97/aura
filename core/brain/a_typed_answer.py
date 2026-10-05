"""An answer asked for in a typed shape is judged by the parser of that shape.

The gate's integrity reader is a reader of prose: a list of forty similar
items reads to it as a loop, and an answer cut off by its budget as an
unfinished sentence. A typed answer has its own parser, which keeps the
finished items of one cut off. LIVE 2026-10-05 a list of checks for four
features was refused as "low_lexical_diversity_loop, truncated_tail" after
418 seconds, and nothing reached the parser that would have kept twenty.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["judged_by_its_parser"]

#: What a reader of prose objects to that the parser of a shape settles itself.
_THE_PARSER_SETTLES = frozenset({"low_lexical_diversity_loop", "repetitive_phrase_loop", "low_information_loop", "truncated_tail"})


def judged_by_its_parser(asked: Mapping[str, Any], reasons: Iterable[str]) -> bool:
    """Whether this answer was asked for in a typed shape and objected to only for what that shape's parser settles."""
    typed = asked.get("schema") is not None or str(asked.get("output_shape") or "") == "json_object"
    return typed and set(reasons) <= _THE_PARSER_SETTLES
