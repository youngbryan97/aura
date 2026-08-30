"""Keeping the meanings she induced across a restart.

She can induce the meaning of a kind of rule nobody wrote and have the
interpreter run it. It lived in process memory, so it died with the process,
and a node saved with that kind came back as a name with nothing behind it.

Kept here rather than in the runtime, beside the thing it keeps. A package that
holds what a rule means is where knowledge of what a rule means belongs;
core/agency keeps the properties she invented the same way, and neither has to
reach across for the other.

What is kept is the RECIPE, never a pickled object: which two places a value is
read from and what is done with the pair. It reconstructs exactly, a person can
read it, and it cannot execute anything that was not already in the space she
searches.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.runtime.errors import record_degradation

__all__ = ["forget_everything", "keep", "recall"]

logger = logging.getLogger("Aura.WhatSheGaveMeaning")

_KEPT_AT = Path.home() / ".aura" / "state" / "meanings_she_induced.json"

#: A bound on the file. What she has given meaning to is a handful of recipes.
_MOST_KEPT = 200_000


def keep() -> bool:
    """Write down every kind of rule she has worked out the meaning of."""
    from core.cognition.an_invented_kind import KINDS

    body = {
        kind: {
            "where_from": meaning.where_from,
            "and_from": meaning.and_from,
            "what_of_it": meaning.what_of_it,
            "held_back": float(meaning.held_back),
            "from_examples": int(meaning.from_examples),
        }
        for kind, meaning in KINDS.items()
    }
    if not body:
        return False
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        written = json.dumps(body)
        if len(written) > _MOST_KEPT:
            logger.info("what she gave meaning to is too big to keep (%d)", len(written))
            return False
        get_file_write_gateway().ensure_directory(
            _KEPT_AT.parent, source="what_she_gave_meaning"
        )
        with local_internal_governed_scope(
            "what_she_gave_meaning.keep", domain="state_mutation"
        ):
            get_file_write_gateway().write_text(
                _KEPT_AT, written, source="what_she_gave_meaning"
            )
        logger.info("kept %d meaning(s) she worked out", len(body))
        return True
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "what_she_gave_meaning", exc, severity="info",
            action="keep the meanings she induced",
        )
        return False


def recall() -> int:
    """Put back the meanings she induced. Returns how many came back."""
    try:
        held = json.loads(_KEPT_AT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(held, dict):
        return 0
    from core.cognition.an_invented_kind import KINDS, Induced

    back = 0
    for kind, row in held.items():
        if not isinstance(row, dict):
            continue
        try:
            KINDS[str(kind)] = Induced(
                where_from=str(row["where_from"]),
                and_from=str(row["and_from"]),
                what_of_it=str(row["what_of_it"]),
                held_back=float(row.get("held_back") or 0.0),
                from_examples=int(row.get("from_examples") or 0),
            )
        except (KeyError, TypeError, ValueError):
            continue
        back += 1
    if back:
        logger.info("she remembered %d meaning(s) she had worked out", back)
    return back


def forget_everything() -> bool:
    """Drop the lot. What was induced on evidence can be dropped."""
    try:
        from core.runtime.file_write_gateway import get_file_write_gateway

        get_file_write_gateway().delete_file(_KEPT_AT, source="what_she_gave_meaning")
        return True
    except (OSError, RuntimeError, AttributeError):
        return False
