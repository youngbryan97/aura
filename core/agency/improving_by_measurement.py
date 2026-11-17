"""A change of hers proposed for a cost her own running measures, kept only when the same measurement says it helped
(Master Todo F16, part 3).

Her collector timing (core/runtime/collector_pauses.py) records each collection a line of hers asks for: when, how long
it held its thread, of which generation, and what it collected. LIVE 15 to 17 November 2026, the line in her resource
governor that asks for a full collection at its moderate and aggressive memory tiers asked for 482 of them, a median
of 232 ms each and 127 s in all, freeing a median of 586 objects. A full collection holds the interpreter lock, so her
event loop stands still for each one wherever it runs. Her collector runs its own full passes by its own rule
(about every 15 s once her boot objects are frozen, R06), so what an explicit request adds is the young garbage of
the moment, which a young collection (generations 0 and 1) frees at a small part of the cost.

So her loop finds the line that asks for the full collections holding her the longest, by her own record; proposes
that it ask for a young collection there, computed on the call itself; puts the change through her own gates and
loads it into the process she is running; and later compares the same measurement after the change with before it.
The change is kept when each collection the line asks for holds her less and the line asks no more often than it did
(its memory tier fires no more often); otherwise it is undone in her running process and taken back out on origin.
Nothing here generates text: the change is made on the syntax tree, and her gates and her record judge it.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ImprovingByMeasurement")

__all__ = ["MeasuredCost", "improve_by_measurement", "judged", "the_costliest_asked_for_collection",
           "young_collection_in_place_of"]

#: What the change is, in her record's words.
YOUNG_COLLECTION = "measured: a young collection in place of a full one"


@dataclass(frozen=True)
class MeasuredCost:
    """A cost her own running measured at a line of hers."""

    site: str  # "core/resource/resource_governor.py:481"
    measured: dict[str, Any]

    @property
    def path(self) -> str:
        return self.site.rsplit(":", 1)[0]

    @property
    def line(self) -> int:
        return int(self.site.rsplit(":", 1)[1])

    def in_words(self) -> str:
        m = self.measured
        return (f"{m['collections']} full collections asked for at {self.site}, a median of "
                f"{m['median_s'] * 1000:.0f} ms each and {m['total_s']:.1f} s in all, freeing a median of "
                f"{m['median_collected']} objects")


def the_costliest_asked_for_collection(since: float, until: float, *, budget_s: float | None = None,
                                       sites: dict[str, dict[str, Any]] | None = None) -> MeasuredCost | None:
    """The line of hers whose full collections held her the longest between ``since`` and ``until``, when each held
    her past her loop's budget at the median; None when no line did."""
    if sites is None:
        from core.runtime.collector_pauses import asked_for_collections

        sites = asked_for_collections(since, until)
    if budget_s is None:
        from core.runtime.lockdep import LOOP_BLOCKING_HOLD_S

        budget_s = LOOP_BLOCKING_HOLD_S
    full = {site: m for site, m in sites.items()
            if 2 in m.get("generations", ()) and m.get("median_s", 0.0) > budget_s and ":" in site}
    if not full:
        return None
    site = max(full, key=lambda s: full[s]["total_s"])
    return MeasuredCost(site, {**full[site], "since": since, "until": until})


def young_collection_in_place_of(source: str, line: int) -> str | None:
    """``source`` with the bare ``gc.collect()`` on ``line`` asking for generations 0 and 1 only; None when that line
    holds no such call."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and node.lineno == line and not node.args and not node.keywords
             and isinstance(node.func, ast.Attribute) and node.func.attr == "collect"
             and isinstance(node.func.value, ast.Name) and node.func.value.id == "gc"]
    if len(calls) != 1:
        return None
    rows = source.splitlines(keepends=True)
    row = rows[line - 1]
    changed, count = re.subn(r"\bgc\.collect\(\s*\)", "gc.collect(1)", row, count=1)
    if count != 1:
        return None
    rows[line - 1] = changed
    return "".join(rows)


def judged(before: dict[str, Any], after: dict[str, Any], *, before_s: float, after_s: float) -> tuple[bool, str]:
    """Whether the change helped, by the same measurement before and after it, and why in words.

    It helped when each collection the line asks for held her less at the median, and the line asked no more often per
    hour: asked more often, its memory tier is firing more, and less is being freed."""
    rate_before = before["collections"] / max(before_s, 1.0)
    rate_after = after["collections"] / max(after_s, 1.0)
    held_less = after["median_s"] < before["median_s"]
    no_more_often = rate_after <= rate_before
    said = (f"each collection held her {after['median_s'] * 1000:.0f} ms at the median against "
            f"{before['median_s'] * 1000:.0f} ms before, freeing a median of {after['median_collected']} objects "
            f"against {before['median_collected']}; asked {rate_after * 3600:.1f} times an hour against "
            f"{rate_before * 3600:.1f}")
    return held_less and no_more_often, said


def _module_of(path: str) -> str:
    return path[:-3].replace("/", ".") if path.endswith(".py") else ""


async def _written(workspace: Path, path: str, text: str) -> None:
    """A file of her workspace written through her write gateway."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope("improving_by_measurement.change", domain="file_write",
                                       receipt_prefix="her-improvement",
                                       constraints={"artifact": "aura.her_improvement.v1", "operation": "replace"}):
        await get_file_write_gateway().write_text_async(workspace / path, text, source="improving_by_measurement")


def _measured_worse_before(record: Any) -> bool:
    """Whether this change was made before and her measurement found it no better: it is not made again."""
    return any(offer.get("origin") == YOUNG_COLLECTION and (offer.get("judged") or {}).get("kept") is False
               for offer in [*(record.offers_before or []), record.offer or {}])


async def _propose(cost: MeasuredCost) -> dict[str, Any]:
    """The change for ``cost``, drafted in a workspace of hers, taken by her own yes, and loaded."""
    from core.agency import her_goal_at_work as goals
    from core.agency.mending_a_fault_of_hers import _governed_run

    goal = f"Lower what the full collections asked for at {cost.site} cost her"
    slug = goals.slug_of(goal)
    record = goals.load(slug) or goals.GoalRecord(goal=goal, slug=slug)
    if record.status in (goals.OFFERED, goals.MERGED):
        return {"slug": slug, "status": record.status, "why": "already changed or offered"}
    if _measured_worse_before(record):
        return {"slug": slug, "status": record.status, "why": "measured no better when it was made before"}
    if record.status == goals.STALLED and record.stalled_on == goals.source_head():
        return {"slug": slug, "status": record.status, "why": "tried on this source"}
    record.status = goals.WORKING
    workspace = Path(await asyncio.to_thread(goals.ensure_workspace, record))
    await goals.save(record)
    before = (workspace / cost.path).read_text(encoding="utf-8")
    after = young_collection_in_place_of(before, cost.line)
    if after is None:
        record.status, record.stalled_on = goals.STALLED, goals.source_head()
        record.said.append({"words": f"Not changed: {cost.site} no longer asks for a full collection with gc.collect().",
                            "at": time.time()})
        await goals.save(record)
        return {"slug": slug, "status": goals.STALLED, "why": "the line no longer asks for one"}
    await _written(workspace, cost.path, after)
    message = (f"Ask for a young collection at {cost.site}: the full ones cost her more than they freed\n\n"
               f"Measured by her own running: {cost.in_words()}. Her collector runs its own full passes; a young "
               f"collection frees what is young now.\n\nDrafted by Aura toward her goal: {goal}")
    code, out = await _governed_run(["git", "add", "--", cost.path], cwd=workspace, env=None, timeout=60.0,
                                    effect="commit")
    if code == 0:
        code, out = await _governed_run(
            ["git", "-c", "core.fsmonitor=false", "-c", "user.name=Aura", "-c", "user.email=aura@localhost",
             "commit", "-q", "-m", message], cwd=workspace, env=None, timeout=60.0, effect="commit")
    if code != 0:
        await _written(workspace, cost.path, before)
        record.status, record.stalled_on = goals.STALLED, goals.source_head()
        record.said.append({"words": f"My change could not be committed: {out.strip()[:200]}", "at": time.time()})
        await goals.save(record)
        return {"slug": slug, "status": goals.STALLED, "why": "not committed"}
    _, head = await _governed_run(["git", "rev-parse", "HEAD"], cwd=workspace, env=None, timeout=30.0,
                                  effect="compute")
    if (record.offer or {}).get("commit"):
        record.offers_before = [*record.offers_before, dict(record.offer)]
    record.offer = {"summary": message.splitlines()[0], "commit": head.strip(), "at": time.time(),
                    "files": [cost.path], "origin": YOUNG_COLLECTION, "measured_before": cost.measured,
                    "site": cost.site}
    record.status = goals.OFFERED
    await goals.save(record)

    def take_and_load() -> dict[str, Any]:
        taken = goals.take_it_herself(slug)
        if taken.get("ok"):
            taken["loaded"] = _loaded(cost.path, before, after, slug)
            goals.note_on_offer(slug, loaded={**taken["loaded"], "at": time.time()})
        return taken

    try:
        taken = await asyncio.to_thread(take_and_load)
    except (OSError, RuntimeError, ValueError) as exc:
        record_degradation("improving_by_measurement", exc, severity="warning",
                           action="left her change offered; her own yes did not run")
        taken = {"ok": False, "why": f"her own yes did not run: {type(exc).__name__}"}
    return {"slug": slug, "status": "merged" if taken.get("ok") else goals.OFFERED, "commit": head.strip(),
            "loaded": taken.get("loaded") or {}, "taken": taken, "why": cost.in_words()}


def _loaded(path: str, before: str, after: str, slug: str) -> dict[str, Any]:
    from core.governance_context import local_internal_governed_scope
    from core.self_modification.into_the_running_process import load_into_this_process

    with local_internal_governed_scope("improving_by_measurement.load", domain="state_mutation",
                                       receipt_prefix="her-improvement-load",
                                       constraints={"artifact": "aura.her_improvement_load.v1", "operation": "load"}):
        loaded = load_into_this_process(_module_of(path), before, after, goal=slug)
    return {"functions": [swap.qualname for swap in loaded.swaps], "waits": list(loaded.waits)}


async def _judge(record: Any, now: float) -> dict[str, Any] | None:
    """Her landed change judged by the same measurement after it as before it, once as much has run after as before;
    None while it has not."""
    from core.agency import her_goal_at_work as goals
    from core.runtime.collector_pauses import asked_for_collections

    offer = record.offer or {}
    before = offer.get("measured_before") or {}
    loaded_at = float((offer.get("loaded") or {}).get("at") or 0.0)
    if not before or not loaded_at or not (offer.get("loaded") or {}).get("functions"):
        return None
    before_s = float(before["until"]) - float(before["since"])
    after = asked_for_collections(loaded_at, now).get(offer.get("site", ""), {})
    after_s = now - loaded_at
    if after_s < before_s and after.get("collections", 0) < before["collections"]:
        return None  # not yet as much after as before
    if not after:
        # As long after as before and not asked for once: the line's tier never fired, and nothing can be compared.
        goals.note_on_offer(record.slug, judged={"at": now, "kept": True, "why": "not asked for once since"})
        return {"slug": record.slug, "status": "kept", "why": "not asked for once since"}
    kept, said = judged(before, after, before_s=before_s, after_s=after_s)
    goals.note_on_offer(record.slug, judged={"at": now, "kept": kept, "why": said, "measured_after": after})
    if kept:
        logger.info("🩹 Her change at %s is kept: %s", offer.get("site"), said)
        return {"slug": record.slug, "status": "kept", "why": said}
    from core.self_modification.into_the_running_process import loads

    for loaded in loads():
        if loaded.goal == record.slug:
            loaded.undo()
    taken_back = await asyncio.to_thread(goals.take_it_back, record.slug, f"it did not measure better: {said}")
    logger.warning("🩹 Her change at %s did not measure better and was taken back: %s", offer.get("site"), said)
    return {"slug": record.slug, "status": "taken back" if taken_back.get("ok") else "not taken back", "why": said}


def _her_start() -> float:
    """When the process she is running started: her collector timing measures from about then."""
    import os

    import psutil

    return float(psutil.Process(os.getpid()).create_time())


async def improve_by_measurement(*, since: float | None = None, now: float | None = None) -> list[dict[str, Any]]:
    """One step of her measured improvement: a landed change judged once enough has run since it, or else a change
    proposed for the costliest measured cost since ``since`` (her start). At most one a cycle."""
    from core.agency import her_goal_at_work as goals

    now = time.time() if now is None else now
    since = _her_start() if since is None else since
    try:
        for record in goals.all_records():
            offer = record.offer or {}
            if (record.status == goals.MERGED and offer.get("origin") == YOUNG_COLLECTION and not offer.get("judged")
                    and not offer.get("reverted")):
                outcome = await _judge(record, now)
                return [outcome] if outcome else []
        cost = the_costliest_asked_for_collection(since, now)
        if cost is None:
            return []
        return [await _propose(cost)]
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        record_degradation("improving_by_measurement", exc, severity="warning",
                           action="her measured improvement skipped this cycle")
        return []
