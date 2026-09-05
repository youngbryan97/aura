"""core/interiority/hydrate.py — giving the ledger something to be about.

An appraisal is relational or it is a classifier, and the relation comes
from the ledger: what she is holding, what she promised, what she is
trying to do. On a booting runtime that ledger starts empty, so the
first live turns produced exactly what an empty ledger should — every
faculty declining, correctly, for want of anything at stake. The layer
ran and had nothing to run on.

This fills it from what the runtime already knows. Her active goals are
hers and carry no third-party consent question, so they are read
directly from the goal engine and become goals in the ledger with their
own priorities.

Bonds are deliberately not hydrated from the relationship graph. That
graph is consent-gated per node and reading it wholesale to populate an
affect substrate is the kind of thing that should be asked for rather
than assumed. Bonds arrive the way they should: from events that name a
subject, through the ledger write the faculties already emit.

Hydration is best-effort and bounded. A runtime that cannot reach its
goal engine gets an empty ledger and a layer that declines, which is the
honest failure and the one it already handles.
"""

from __future__ import annotations

import logging
from typing import Any

from core.interiority.params import ParamKind, declare
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Interiority.Hydrate")

_MAX_GOALS = declare(
    "interiority.hydrate.max_goals",
    24,
    unit="goals",
    basis=(
        "How many active goals become ledger stakes at boot. Set at twice "
        "the goal engine's own hot-path limit of twelve, because that covers "
        "what a relevance check might touch without turning every appraisal "
        "into a scan of a long list."
    ),
    kind=ParamKind.DERIVED,
    sensitivity=(
        "Too few and events about real commitments read as irrelevant; too "
        "many and relevance is high for everything, which is the same as "
        "being high for nothing."
    ),
    lower=1.0,
    upper=256.0,
    owner="core/interiority/hydrate.py",
)


def hydrate_goals(service: Any) -> dict[str, Any]:
    """Copy the runtime's active goals into the ledger. Never raises."""
    try:
        from core.container import ServiceContainer

        engine = ServiceContainer.get("goal_engine", default=None)
        if engine is None or not hasattr(engine, "get_active_goals"):
            return {"hydrated": 0, "reason": "no goal engine registered"}

        cap = int(_MAX_GOALS.value)
        # Bound against what the ledger already holds, not against this
        # call. A flag on the service object was not enough: the live
        # instance still reached seventy-six, because the boot path can
        # reach a second copy of this module and a second singleton with
        # it, and a guard that lives on one object cannot see the other.
        # The ledger is the thing being bounded, so it is what the bound
        # is measured against.
        already = service.ledger.counts().get("goals", 0)
        room = max(0, cap - already)
        if room == 0:
            return {"hydrated": 0, "reason": f"ledger already holds {already} goals"}

        goals = engine.get_active_goals(limit=room)
        added = 0
        for item in goals or []:
            if not isinstance(item, dict):
                continue
            name = str(
                item.get("goal_id") or item.get("id") or item.get("description") or ""
            )[:120]
            if not name:
                continue
            try:
                priority = float(item.get("priority") or 0.0)
            except (TypeError, ValueError):
                priority = 0.0
            # A goal with alternatives is one she can adjust to; the engine
            # does not model substitutes, so none are claimed rather than
            # invented. That keeps loss of a plan distinguishable from loss
            # of something with no replacement.
            if added >= room:
                break
            service.ledger.goal(name, max(0.0, min(1.0, priority)), substitutes=0)
            added += 1
        return {"hydrated": added, "available": len(goals or [])}
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "interiority.hydrate", exc, action="ledger not hydrated from goals"
        )
        return {"hydrated": 0, "error": type(exc).__name__}


def hydrate(service: Any) -> dict[str, Any]:
    """Everything the ledger can honestly be given at boot.

    Idempotent, because the boot path registers the derived engines more
    than once. Each pass returned a different slice of the goal snapshot,
    so a declared cap of twenty-four produced seventy-six goals in the
    ledger on the live instance — a bound that bounded nothing.
    """
    if getattr(service, "_hydrated", False):
        return {"skipped": "already hydrated", "ledger": service.ledger.counts()}
    report = {"goals": hydrate_goals(service)}
    try:
        service._hydrated = True
    except (AttributeError, TypeError):
        pass
    counts = service.ledger.counts()
    report["ledger"] = counts
    if counts.get("goals", 0) == 0:
        # Worth saying. An empty ledger is a layer that will decline
        # everything, which looks like calm and is having nothing at stake.
        logger.info(
            "interiority ledger hydrated with no goals; every relevance check "
            "will read zero until something is held"
        )
    return report


__all__ = ["hydrate", "hydrate_goals"]
