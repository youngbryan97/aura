"""What a line in her neural feed says, in words anyone can read.

Bryan, 16 November 2026: "neural feed messages should be plain english and easily readable. a lot of them are coming in
as code lines. anyone looking at aura should be able to understand what they're reading/what is happening." The feed's
lines are her log records. Each comes from one call in her code, a logger and a format string, and the sentence for it is
kept in ``config/plain_feed.json`` under that logger and format string, with ``{0}``, ``{1}`` where the record's values go.
Her server puts the sentence on the card's face; the record's own line stays behind it for whoever wants the detail.

A format string is matched as written ("%s" style), or as an f-string's text with its values taken out. A line nobody has
written a sentence for gets none, and the card shows it as before. ``tools/plain_feed.py`` measures how much of a captured
feed has a face and lists the lines most worth one.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["pattern_of", "plain_face", "plain_face_of"]

logger = logging.getLogger("Aura.PlainFeed")

_CATALOGUE = Path(__file__).resolve().parents[2] / "config" / "plain_feed.json"
_PERCENT = re.compile(r"%(?:\([^)]*\))?[-#0 +]*(?:\d+|\*)?(?:\.(?:\d+|\*))?[sdrifxXeEgGc]|%%")
_EXACT: dict[tuple[str, str], str] | None = None
_WITH_VALUES: dict[str, list[tuple[re.Pattern, str]]] = {}


def pattern_of(says: str, kind: str) -> re.Pattern:
    """The lines a format string writes, as a pattern with one group for each value."""
    if kind == "f":
        pieces = re.split(r"(?<!\{)\{\}(?!\})", says)
        literal = [re.escape(piece.replace("{{", "{").replace("}}", "}")) for piece in pieces]
        return re.compile("(.*?)".join(literal), re.DOTALL)
    out, last = [], 0
    for match in _PERCENT.finditer(says):
        out.append(re.escape(says[last:match.start()]))
        out.append("%" if match.group() == "%%" else "(.*?)")
        last = match.end()
    out.append(re.escape(says[last:]))
    return re.compile("".join(out), re.DOTALL)


def _load() -> dict[tuple[str, str], str]:
    global _EXACT
    if _EXACT is None:
        exact: dict[tuple[str, str], str] = {}
        try:
            rows = json.loads(_CATALOGUE.read_text(encoding="utf-8")).get("lines", [])
        except (OSError, ValueError) as exc:
            logger.debug("no plain faces for the feed: %s", exc)
            rows = []
        for row in rows:
            name, says, plain = str(row.get("logger", "")), str(row.get("says", "")), str(row.get("plain", ""))
            if not (name and says and plain):
                continue
            if row.get("kind") == "f":
                _WITH_VALUES.setdefault(name, []).append((pattern_of(says, "f"), plain))
            else:
                exact[(name, says)] = plain
        _EXACT = exact
    return _EXACT


_AN_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[_:.\-]|->)[A-Za-z0-9_:.\->]*")
_WORDS_RUN_TOGETHER = re.compile(r"(?:[A-Z][a-z0-9]+){2,}")


def _readable(value: Any) -> Any:
    """A value as a person reads it: "drive_social" as "drive social", "OtherAgentStateEstimator" as "other agent
    state estimator", "person:bryan" as "bryan", a list as its items. Numbers are left for the sentence to format."""
    if isinstance(value, (list, tuple, set, frozenset)) and all(isinstance(item, str) for item in value):
        return ", ".join(str(_readable(item)) for item in value)
    if not isinstance(value, str):
        return value
    text = value.removeprefix("person:")
    if _WORDS_RUN_TOGETHER.fullmatch(text):
        return " ".join(re.findall(r"[A-Z][a-z0-9]+", text)).lower()
    if _AN_IDENTIFIER.fullmatch(text):
        return text.replace("->", " → ").replace("_", " ")
    return text


def plain_face(name: str, msg: Any, args: Any) -> str:
    """The plain sentence for a record from logger ``name`` with format string ``msg`` and values ``args``; "" if none."""
    exact = _load()
    says = str(msg)
    plain = exact.get((name, says))
    values: Any = args
    if plain is None:
        for pattern, written in _WITH_VALUES.get(name, ()):
            match = pattern.fullmatch(says)
            if match:
                plain, values = written, match.groups()
                break
    if plain is None:
        return ""
    try:
        if isinstance(values, Mapping):
            return plain.format_map({key: _readable(value) for key, value in values.items()})
        values = values if isinstance(values, tuple) else (values,) if values else ()
        return plain.format(*(_readable(value) for value in values))
    except (IndexError, KeyError, ValueError, TypeError, AttributeError):
        return ""


def plain_face_of(record: logging.LogRecord) -> str:
    """The plain sentence for a log record: the one its call gave (``extra={"plain": ...}``), else the catalogue's."""
    given = getattr(record, "plain", None)
    if isinstance(given, str) and given.strip():
        return given.strip()
    return plain_face(record.name, record.msg, record.args)
