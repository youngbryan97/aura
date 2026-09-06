"""Two resources at once, in an order nobody has to think about.

:mod:`core.runtime.who_gets_it_next` hands out one resource at a time, which
is enough until something needs two. Then the order matters: a task that takes
the screen and then the model lane, running beside one that takes the model
lane and then the screen, is the deadlock everyone has read about and nobody
notices writing.

The fix is old and small — always acquire in one global order — and the part
that gets skipped is everything around it. A caller that asks for the same
resource twice must not wait for itself. A caller that asks for nothing must
get a context, not an error. A partial acquisition that times out must give
back exactly what it took, in reverse, without leaving one held.

So: sorted acquisition, deduplicated, released in reverse, and the release
happens whatever went wrong on the way. The tests for this are properties
rather than examples, because the failure only appears in the orderings nobody
wrote a test for: every permutation of every subset acquires in the same
sequence, and every failure point leaves nothing held.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from core.runtime.who_gets_it_next import THE_RESOURCES, claim

logger = logging.getLogger("Aura.ClaimingMoreThanOne")

__all__ = [
    "THE_ORDER",
    "in_order",
    "claim_all",
    "how_the_multi_claims_have_gone",
    "forget_everything",
]

#: The one global order. Alphabetical rather than by importance: importance is
#: an argument and alphabetical is not, and the only property that matters is
#: that everyone picks the same one.
THE_ORDER: tuple[str, ...] = tuple(sorted(THE_RESOURCES))


def in_order(wanted: Iterable[str]) -> tuple[str, ...]:
    """The resources to take, deduplicated, in the one global order.

    Asking twice for the same resource is not two claims. A caller that
    assembles its list from two places will do it, and waiting for a lock it
    already holds is a hang with no message.
    """
    asked = [str(one) for one in wanted]
    unknown = [one for one in asked if one not in THE_RESOURCES]
    if unknown:
        raise KeyError(
            f"no such resource: {sorted(set(unknown))}; "
            f"the ones there are: {list(THE_ORDER)}"
        )
    return tuple(name for name in THE_ORDER if name in set(asked))


_HISTORY: list[dict[str, Any]] = []
_LOCK = threading.Lock()
_KEEP = 200


@asynccontextmanager
async def claim_all(
    wanted: Sequence[str],
    by: str,
    *,
    context: Any = None,
    seconds: float = 0.0,
):
    """Hold all of ``wanted`` at once, or none of them.

    Acquired in :data:`THE_ORDER`, released in reverse, and the release runs
    however the block ended — including a partial acquisition that timed out
    part way, which is the case that leaves one resource held forever if it is
    handled anywhere but here.
    """
    order = in_order(wanted)
    # Every one of them checked before any of them is taken. Refusing half way
    # through is how a caller ends up holding the screen because the model
    # lane was never its to ask for.
    elsewhere = [
        name
        for name in order
        if THE_RESOURCES[name].get("granted") != "here"
    ]
    if elsewhere:
        raise ValueError(
            f"{sorted(elsewhere)} are granted elsewhere, not here; "
            "nothing was taken"
        )
    held: list[Any] = []
    entered: list[str] = []
    trouble = ""
    try:
        for name in order:
            holder = claim(name, by, context=context, seconds=seconds)
            await holder.__aenter__()
            held.append(holder)
            entered.append(name)
        yield tuple(entered)
    except BaseException as exc:  # noqa: BLE001 - re-raised after the giving back
        trouble = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # In reverse, and every one of them attempted: a failure giving one
        # back must not strand the others.
        gave_back: list[str] = []
        for name, holder in reversed(list(zip(entered, held))):
            try:
                await holder.__aexit__(None, None, None)
                gave_back.append(name)
            except BaseException as exc:  # noqa: BLE001
                logger.warning("releasing %s for %s failed: %s", name, by, exc)
        with _LOCK:
            _HISTORY.append(
                {
                    "by": by,
                    "asked": list(wanted),
                    "order": list(order),
                    "took": list(entered),
                    "gave_back": list(reversed(gave_back)),
                    "all_of_them": len(entered) == len(order),
                    "trouble": trouble,
                }
            )
            del _HISTORY[:-_KEEP]


def how_the_multi_claims_have_gone() -> dict[str, Any]:
    """For the health report: whether anything was ever left holding one."""
    with _LOCK:
        rows = list(_HISTORY)
    stranded = [r for r in rows if len(r["gave_back"]) != len(r["took"])]
    return {
        "the_order": list(THE_ORDER),
        "claims": len(rows),
        "partial": sum(1 for r in rows if not r["all_of_them"]),
        "left_holding_something": len(stranded),
        "stranded": stranded[:5],
        "recent": rows[-5:],
    }


def forget_everything() -> None:
    with _LOCK:
        _HISTORY.clear()
