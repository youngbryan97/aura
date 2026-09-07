"""Keeping the properties she worked out for herself across a restart.

She can compose a property of a situation nobody wrote, prove it plays better
and judge by it — and it lived in process memory, so it died when the process
did. A mind that reinvents the same property every morning has not learned it.

Kept here rather than in the runtime, beside the thing it keeps. A package that
describes how she judges a situation is where knowledge of how she judges a
situation belongs; core/cognition keeps its own meanings the same way, and
neither has to reach across for the other.

That mattered more than it sounds. The point of the invention was never one
good measure — it was that what she works out is persistent, reusable,
composable and transferable. Three of those four fail if it does not survive a
restart, and the fourth is only interesting because of them.

What is kept is the RECIPE, never a pickled object: where it looks, what it
takes and how it combines. It reconstructs exactly, a person can read it, and
it cannot execute anything that was not already in the space she searches.
"""

from __future__ import annotations

import json
import asyncio
import logging

from pathlib import Path

from core.runtime.errors import record_degradation

__all__ = ["forget_everything", "keep", "recall"]

logger = logging.getLogger("Aura.PropertiesSheInvented")

#: Where it lives. Beside what she learned about particular worlds, because it
#: is the same kind of thing: something she found out and should not have to
#: find out again.
#:
#:
#: Set to send it somewhere else. Left alone in the live runtime; a test that
#: wants its own file names one here.
_KEPT_AT: Path | None = None


def _kept_at() -> Path:
    """Where it goes.

    Asked for on each call rather than fixed at import: a path resolved once
    aims a test run's writes at the live instance, the ownership guard then
    refuses them, and the persistence is never actually exercised.
    """
    if _KEPT_AT is not None:
        return _KEPT_AT
    from core.runtime.state_ownership import state_root

    return state_root() / "state" / "properties_she_invented.json"

#: A bound on the file. What she invented is a handful of recipes; anything
#: larger is a runaway rather than a mind that has learned a great deal.
_MOST_KEPT = 200_000

#: What shape the file is in. Bumped when a reader could no longer make sense
#: of an older one; the minor number moves when a field is added and an older
#: reader can still use what it understands.
_THE_SHAPE = (1, 0)


def _adopt_an_older_file(data: dict, major: int, minor: int) -> dict:
    """Read a file written before this store had a version on it.

    Version 0 is the bare object the old code wrote: the same fields, with no
    envelope. Nothing to change, so this returns it — the hook exists so that
    adopting a versioned store does not throw away what was already there.
    """
    return data


def _the_store():
    from core.persistence.a_versioned_store import AVersionedStore

    return AVersionedStore(
        _kept_at(),
        major=_THE_SHAPE[0],
        minor=_THE_SHAPE[1],
        migrate=_adopt_an_older_file,
        source="what_she_invented",
    )


def keep() -> bool:
    """Write down every property and every meaning she has worked out."""
    from core.agency.how_good_is_this import AS_GOOD_A_GUESS_AS_ANY, INVENTED, ON_TRIAL

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
    }
    if not body["measures"]:
        return False
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        written = json.dumps(body)
        if len(written) > _MOST_KEPT:
            logger.info("what she invented is too big to keep (%d)", len(written))
            return False
        def _write() -> None:
            with local_internal_governed_scope(
                "what_she_invented.keep", domain="state_mutation"
            ):
                get_file_write_gateway().ensure_directory(
                    _kept_at().parent, source="what_she_invented"
                )
                _the_store().save(body)

        try:
            asyncio.get_running_loop().create_task(asyncio.to_thread(_write))
        except RuntimeError:
            _write()
        logger.info("kept %d propert(ies) she worked out", len(body["measures"]))
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
    back = {"measures": 0}
    from core.persistence.a_versioned_store import CannotRead

    try:
        kept = _the_store().load()
    except CannotRead as exc:
        # It used to return an empty dict here, so a corrupt file was
        # indistinguishable from a fresh install — and the next keep wrote
        # over it, taking the evidence with it. The store puts the bad copy
        # aside; this says out loud that it happened.
        record_degradation(
            "what_she_invented",
            exc,
            severity="warning",
            action=f"put the unreadable file aside at {exc.put_aside_at}",
        )
        return back
    except (OSError, ValueError):
        return back
    if kept is None:
        return back
    held = kept.data
    if not isinstance(held, dict):
        return back
    from core.agency.how_good_is_this import ON_TRIAL, promote
    from core.agency.inventing_a_measure import Measure

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

    if any(back.values()):
        logger.info(
            "she remembered %d propert(ies) she had worked out", back["measures"]
        )
    return back


def forget_everything() -> bool:
    """Drop the lot. What was worked out on evidence can be dropped."""
    try:
        from core.runtime.file_write_gateway import get_file_write_gateway

        get_file_write_gateway().delete_file(_kept_at(), source="what_she_invented")
        return True
    except (OSError, RuntimeError, AttributeError):
        return False
