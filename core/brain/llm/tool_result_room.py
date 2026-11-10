"""What one tool result may take of her model's reading, and how a long one is cut so it still reads.

A result longer than its room used to be re-encoded whole as a string inside a "preview" field. That kept it valid
JSON and escaped it a second time, so a file she read came back as ``\\n`` and ``\\\"`` and stopped mid-line: on 9 and
10 November 2026, 80 of the 161 calls of her goal "Find something productive to work on" came back that way, and none
of her 14 goals reached an edit. A long result is now cut inside its longest values, at a line end, with a marker that
says how much is gone, so it stays a result she can read.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

#: The characters one tool result may take of her model's reading: what core/brain/llm/mlx_client.py
#: (_serialize_tool_result_for_model, _truncate_tool_result) bounds each one to.
TOOL_RESULT_CHARS = 4000

NOTE = "Result exceeded the context budget; the longest values are cut where marked."
_OPENING_TAG = re.compile(r"\s*<tool_call>")


def past_every_opening_tag(text: str, start: int) -> int:
    """Where a call begins after its opening tag, when the tag was written more than once.

    Her model wrote ``<tool_call> <tool_call> <function=read_source>`` on 10 November 2026, and the call was read as
    prose because the grammar looked for the function right after the first tag.
    """
    while True:
        more = _OPENING_TAG.match(text, start)
        if more is None:
            return start
        start = more.end()


def _encoded(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _strings(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        return [found for key, item in value.items() for found in _strings(item, (*path, key))]
    if isinstance(value, list):
        return [found for index, item in enumerate(value) for found in _strings(item, (*path, index))]
    return []


def _set(value: Any, path: tuple[Any, ...], new: str) -> Any:
    if not path:
        return new
    head, rest = path[0], path[1:]
    if isinstance(value, dict):
        return {**value, head: _set(value[head], rest, new)}
    copied = list(value)
    copied[head] = _set(copied[head], rest, new)
    return copied


def _cut(text: str, keep: int) -> str:
    """The first ``keep`` characters, back to the last line end when one is in the second half, and the marker."""
    kept = text[: max(0, keep)]
    if "\n" in kept[len(kept) // 2:]:
        kept = kept[: kept.rfind("\n") + 1]
    return f"{kept}[... {len(text) - len(kept)} more characters]"


def cut_the_longest_values(parsed: Any, *, original_chars: int, limit: int) -> str | None:
    """``parsed`` with its longest strings cut until it fits ``limit``, or None when no value can take the cut."""
    if not isinstance(parsed, dict):
        return None
    shaped: Any = {**parsed, "truncated": True, "original_chars": original_chars, "note": NOTE}
    for _ in range(64):
        encoded = _encoded(shaped)
        excess = len(encoded) - limit
        if excess <= 0:
            return encoded
        found = _strings(shaped)
        if not found:
            return None
        path, longest = max(found, key=lambda item: len(_encoded(item[1])))
        width = len(_encoded(longest))
        if path in {("note",)} or width <= excess:
            return None
        # Escaping makes a character cost more than one in the result; cut by what it costs here.
        per_char = width / max(1, len(longest))
        keep = len(longest) - math.ceil((excess + 40) / per_char)
        if keep <= 0:
            return None
        shaped = _set(shaped, path, _cut(longest, keep))
    return None
