"""Keeping what she worked out for herself across a restart.

She can now compose a property of a situation nobody wrote, prove it plays
better and judge by it; and she can induce the meaning of a kind of rule
nobody wrote and have the interpreter run it. Both lived in process memory, so
both died when the process did. A mind that reinvents the same thing every
morning has not learned it.

That mattered more than it sounds. The point of the invention was never one
good measure — it was that what she works out is persistent, reusable,
composable and transferable. Three of those four fail if it does not survive a
restart, and the fourth is only interesting because of them.

What is kept is the RECIPE, never a pickled object: a measure is where it
looks, what it takes and how it combines; a meaning is which two places a
value is read from and what is done with the pair. Both reconstruct exactly,
both are readable by a person, and neither can execute anything that was not
already in the space she searches.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pathlib import Path

from core.runtime.errors import record_degradation

__all__ = ["forget_everything", "keep", "recall"]

logger = logging.getLogger("Aura.WhatSheInvented")

#: Where it lives. Beside what she learned about particular worlds, because it
#: is the same kind of thing: something she found out and should not have to
#: find out again.
_KEPT_AT = Path.home() / ".aura" / "state" / "what_she_invented.json"

#: A bound on the file. What she invented is a handful of recipes; anything
#: larger is a runaway rather than a mind that has learned a great deal.
_MOST_KEPT = 200_000


def keep() -> bool:
    """Write down every property and every meaning she has worked out."""
    try:
        from core.agency.how_good_is_this import AS_GOOD_A_GUESS_AS_ANY, INVENTED, ON_TRIAL
        from core.cognition.an_invented_kind import KINDS
    except ImportError:
        return False
    body = {
        "measures": [
            {
                "at": measure.at,
                "of": measure.of,
                "summed": measure.summed,
                "the_other_way_up": bool(measure.the_other_way_up),
                "worth": float(AS_GOOD_A_GUESS_AS_ANY.get(name, 0.0)),
                "on_trial": name in ON_TRIAL,
                "trial": dict(ON_TRIAL.get(name) or {}),
            }
            for name, measure in INVENTED.items()
            if hasattr(measure, "at")
        ],
        "meanings": {
            kind: {
                "where_from": meaning.where_from,
                "and_from": meaning.and_from,
                "what_of_it": meaning.what_of_it,
                "held_back": float(meaning.held_back),
                "from_examples": int(meaning.from_examples),
            }
            for kind, meaning in KINDS.items()
        },
    }
    if not body["measures"] and not body["meanings"]:
        return False
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        written = json.dumps(body)
        if len(written) > _MOST_KEPT:
            logger.info("what she invented is too big to keep (%d)", len(written))
            return False
        get_file_write_gateway().ensure_directory(
            _KEPT_AT.parent, source="what_she_invented"
        )
        with local_internal_governed_scope(
            "what_she_invented.keep", domain="state_mutation"
        ):
            get_file_write_gateway().write_text(
                _KEPT_AT, written, source="what_she_invented"
            )
        logger.info(
            "kept %d propert(ies) and %d meaning(s) she worked out",
            len(body["measures"]), len(body["meanings"]),
        )
        return True
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "what_she_invented", exc, severity="info", action="keep what she invented"
        )
        return False


def recall() -> dict[str, int]:
    """Put back what she worked out before. Returns how much came back.

    A property that was mid-trial when the process ended comes back mid-trial,
    with the observations it had already gathered — because a trial interrupted
    by a restart is not a trial that failed, and starting it again from nought
    would mean a property could never be judged on a machine that reboots.
    """
    back = {"measures": 0, "meanings": 0}
    try:
        held = json.loads(_KEPT_AT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return back
    if not isinstance(held, dict):
        return back
    try:
        from core.agency.how_good_is_this import ON_TRIAL, promote
        from core.agency.inventing_a_measure import Measure
        from core.cognition.an_invented_kind import KINDS, Induced
    except ImportError:
        return back

    for row in held.get("measures") or ():
        if not isinstance(row, dict):
            continue
        try:
            measure = Measure(
                at=str(row["at"]),
                of=str(row["of"]),
                summed=str(row["summed"]),
                the_other_way_up=bool(row.get("the_other_way_up")),
            )
        except (KeyError, TypeError, ValueError):
            continue
        name = promote(measure, float(row.get("worth") or 0.0))
        if not name:
            continue
        back["measures"] += 1
        if row.get("on_trial") and isinstance(row.get("trial"), dict):
            ON_TRIAL[name] = dict(row["trial"])

    for kind, row in (held.get("meanings") or {}).items():
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
        back["meanings"] += 1

    if any(back.values()):
        logger.info(
            "she remembered %d propert(ies) and %d meaning(s) she had worked out",
            back["measures"], back["meanings"],
        )
    return back


def forget_everything() -> bool:
    """Drop the lot. What was worked out on evidence can be dropped."""
    try:
        from core.runtime.file_write_gateway import get_file_write_gateway

        get_file_write_gateway().delete_file(_KEPT_AT, source="what_she_invented")
        return True
    except (OSError, RuntimeError, AttributeError):
        return False
