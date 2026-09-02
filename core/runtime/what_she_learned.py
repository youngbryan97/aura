"""What she worked out about a thing, kept for the next time she is in it.

Everything she learns about a world she is acting in — which part of it answers
to her, how it moves when she pushes it, which lines held — has been dying with
the process. So the fortieth run started exactly as ignorant as the first, and
experience was something she had during a run rather than something she had.

Kept per world rather than in general, because that is the honest scope: how a
game board moves is not how a spreadsheet moves, and a rule that held on one
thing is a guess about the next. What is remembered here is remembered about
the named thing it was learned in.

Nothing is trusted on the strength of having been written down. What comes
back is a starting point that has to keep earning its place against what she
sees now, which is why the counts come back discounted rather than whole.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

__all__ = ["forget", "named", "recall", "remember", "TRUST_CARRIED_OVER"]

logger = logging.getLogger("Aura.WhatSheLearned")

#: Where it is kept. One file per thing she has acted in.
_KEPT_IN = Path.home() / ".aura" / "state" / "worlds"

#: How much of what she knew comes back. Under one on purpose: something she
#: worked out yesterday is evidence about today and not a fact about it, and a
#: handful of things that disagree should be able to overturn it.
TRUST_CARRIED_OVER = 0.5

#: How much can be kept about any one thing. A record that grows without a
#: bound is a record nobody reads.
_MOST_KEPT = 60_000


def named(*parts: str) -> str:
    """A name for the thing she is acting in, from whatever identifies it.

    Whatever the caller has: an app, a page, an address. Made into something
    that can be a filename without pretending two different things are one.
    """
    said = " ".join(str(part or "").strip().lower() for part in parts if str(part or "").strip())
    cleaned = re.sub(r"[^a-z0-9]+", "-", said).strip("-")
    return cleaned[:80] or "somewhere"


def remember(world: str, what: dict[str, Any]) -> bool:
    """Keep what she worked out about this thing."""
    key = named(world)
    if not what:
        return False
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        # Its own bookkeeping, under a name a caller cannot mean.
        #
        # This was `{"world": key, **what}`, so a caller with something of its
        # own called "world" — and the pursuit has one, the model of what the
        # world does on its own — silently replaced the record's name with it,
        # or had its own replaced, depending which way round the merge went.
        # Both happened: one world file on this machine has the model where
        # the name should be and another has the name where the model should
        # be. Whichever way a collision resolves, one of the two is lost.
        body = json.dumps({"_kept_for": key, **what})
        if len(body) > _MOST_KEPT:
            logger.info("what she learned about %r is too big to keep (%d)", key, len(body))
            return False
        get_file_write_gateway().ensure_directory(_KEPT_IN, source="what_she_learned")
        with local_internal_governed_scope(
            "what_she_learned.remember",
            domain="state_mutation",
            constraints={"world": key},
        ):
            get_file_write_gateway().write_text(
                _KEPT_IN / f"{key}.json", body, source="what_she_learned"
            )
        return True
    except Exception as exc:  # noqa: BLE001 - remembering is never the task
        record_degradation(
            "what_she_learned", exc, severity="info", action="carried on without remembering"
        )
        return False


def recall(world: str) -> dict[str, Any]:
    """What she worked out about this thing last time, if she has been here."""
    key = named(world)
    try:
        held = json.loads((_KEPT_IN / f"{key}.json").read_text())
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(held, dict):
        return {}
    # Older records kept the name under "world", which is also what the
    # pursuit calls the model of what the world does on its own. A name there
    # is not a model, and every reader of it already refuses a string, so it
    # is left alone rather than guessed at.
    held.pop("_kept_for", None)
    return held


def forget(world: str) -> bool:
    """Drop what she knew about a thing, for a caller that has reason to."""
    from core.runtime.file_write_gateway import get_file_write_gateway

    try:
        get_file_write_gateway().delete_file(
            _KEPT_IN / f"{named(world)}.json", source="what_she_learned"
        )
        return True
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
