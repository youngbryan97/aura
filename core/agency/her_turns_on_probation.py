"""How long her turns take and how much memory she holds, before a change of hers landed and since (Master Todo F16).

A change on probation is settled by the outside benchmarks and by her own real calls of what it changed. Neither sees a
change that answers the same and costs more: a mend that makes every turn wait on a slower path, or one that keeps
what it should let go. This keeps a row for each turn she finishes, and a reading of her memory each time it does, in
her state. When a change has been on probation long enough to be measured, her turns after it landed are compared with
her turns before it in the same running, and how fast her memory grows the same way.

A turn is measured by its seconds outside her model, the time her own code spends; how long she decodes depends on
what she was asked. Turns are compared by the Mann-Whitney test, one-sided, at the 0.05 the outside bar uses: it asks
whether the turns after cost more than the turns before by more than chance would make them, and assumes nothing of
their spread. Memory
grows over any running, so its level after a landing is higher whatever landed; what is compared is how fast it grows,
the slope of her readings against time before and after, by a one-sided test on the difference of the two slopes. Only
turns
in the same running are compared, because a restart brings other code with it, and only turns that called her model,
because a reply the code assembled says nothing about her. A finding is said and kept on her record; the change is
taken back only when the cost has grown an order of magnitude, since one long turn after a landing can reach
significance against many short ones.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.Agency.TurnsOnProbation")

__all__ = ["ALPHA", "ORDER_OF_MAGNITUDE", "costlier_since", "keep_her_recent_turns", "where_her_turns_are_kept"]

#: The chance of a difference this lopsided by luck that counts as one: the convention the outside bar judges by.
ALPHA = 0.05
#: How many times the cost before a change it must have grown to be taken back for it: the measure by which the outside
#: check refuses a slowdown rather than doubting it (outside_bar/verify_change.py, ORDER_OF_MAGNITUDE).
ORDER_OF_MAGNITUDE = 10

#: This running of hers: a turn of another running is never compared with one of this.
_THIS_RUNNING = f"{os.getpid()}:{int(time.time())}"


def where_her_turns_are_kept() -> Path:
    from core.runtime.state_ownership import state_root

    return state_root() / "her_turns" / "turns.jsonl"


def _rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except (ImportError, OSError, AttributeError):
        return 0.0


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def keep_her_recent_turns(*, path: Path | None = None, receipts: list[dict[str, Any]] | None = None) -> int:
    """Append a row for each turn she finished that is not kept yet, and one reading of her memory; how many turns were
    added. Blocking: run it off her event loop."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway
    from core.verify.turn_receipt import recent_receipts

    path = path or where_her_turns_are_kept()
    kept = {str(row.get("turn")) for row in _read(path) if row.get("turn")}
    finished = [one for one in (recent_receipts(limit=64) if receipts is None else receipts)
                if one.get("finished_at") and one.get("duration_s") is not None and str(one.get("turn_id")) not in kept]
    rows = [{"turn": str(one.get("turn_id")), "at": float(one["finished_at"]), "seconds": float(one["duration_s"]),
             "outside_model": max(0.0, float(one["duration_s"]) - sum(float((one.get("latency_s") or {}).get(name) or 0.0)
                                                                      for name in ("prefill", "decode"))),
             "model": bool(one.get("model_generation")), "running": _THIS_RUNNING} for one in finished]
    rows.append({"memory_mb": round(_rss_mb(), 1), "at": time.time(), "running": _THIS_RUNNING})
    with local_internal_governed_scope("her_turns_on_probation.keep", domain="state_mutation",
                                       receipt_prefix="her-turns",
                                       constraints={"artifact": "aura.her_turns.v1", "operation": "append"}):
        # The append makes the folder it writes into (core/runtime/atomic_writer.atomic_append_text).
        get_file_write_gateway().append_text(path, "".join(json.dumps(row) + "\n" for row in rows),
                                             source="her_turns_on_probation")
    return len(finished)


def _greater(after: list[float], before: list[float]) -> float:
    """The one-sided Mann-Whitney chance that ``after`` runs this much above ``before`` by luck; 1.0 when it cannot be
    told."""
    if not after or not before:
        return 1.0
    from scipy.stats import mannwhitneyu

    try:
        return float(mannwhitneyu(after, before, alternative="greater", method="auto").pvalue)
    except ValueError:
        return 1.0


def _grows_faster(after: list[tuple[float, float]], before: list[tuple[float, float]]
                  ) -> tuple[float, float, float, float]:
    """(slope after, slope before, the one-sided chance the first exceeds the second by luck, the standard error of the
    slope before), in MB an hour, from (time, MB) readings; the chance is 1.0 when either side has too few to fit a
    line with an error (three)."""
    from scipy.stats import linregress, norm

    if len(after) < 3 or len(before) < 3:
        return 0.0, 0.0, 1.0, 0.0
    try:
        fitted = [linregress([at / 3600.0 for at, _ in side], [mb for _, mb in side]) for side in (after, before)]
    except ValueError:
        return 0.0, 0.0, 1.0, 0.0
    (now, was) = fitted
    spread = (now.stderr ** 2 + was.stderr ** 2) ** 0.5
    if not spread > 0.0:
        return float(now.slope), float(was.slope), 1.0, float(was.stderr)
    return float(now.slope), float(was.slope), float(norm.sf((now.slope - was.slope) / spread)), float(was.stderr)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def costlier_since(landed_at: float, *, path: Path | None = None, running: str | None = None,
                   measured_slower: bool = False) -> dict[str, Any]:
    """How her turns or her memory have grown since a change landed at ``landed_at``, in this running of hers:
    ``said`` in words where it is beyond what chance would make it, "" otherwise; ``beyond_doubt`` when it is also an
    order of magnitude, the measure the outside check refuses a slowdown by.

    A turn is measured by its seconds outside her model, the time her own code spends, which is what a change of hers
    can change: how long she decodes depends on what she was asked. A finding short of an order of magnitude is said and
    kept, not acted on, since one long question after a landing can reach significance against many short ones.

    ``measured_slower`` says the outside check timed the changed functions slower on the same inputs before and after
    it. A rise in her turns beyond chance then agrees with a controlled measure, and it is beyond doubt at any size: a
    change four times slower survived probation for ever while only ten times counted (round nine of the red team).
    Turns that called her model and turns her code answered alone are compared each with their own kind, since the
    second take a fraction of the first and a change to a path that never calls her model shows only among them."""
    rows = [row for row in _read(path or where_her_turns_are_kept()) if row.get("running") == (running or _THIS_RUNNING)]
    said, beyond_doubt = [], False
    for called, kind in ((True, "her turns"), (False, "her turns that did not call her model")):
        turns = [row for row in rows if bool(row.get("model")) is called and "outside_model" in row]
        before = [float(row["outside_model"]) for row in turns if float(row["at"]) < landed_at]
        after = [float(row["outside_model"]) for row in turns if float(row["at"]) >= landed_at]
        chance = _greater(after, before)
        if chance < ALPHA:
            now, was = _median(after), _median(before)
            said.append(f"{kind} spend {now:.1f} s outside her model at the median since it landed, and spent {was:.1f} s "
                        f"before ({len(after)} turns against {len(before)}, a chance of {chance:.3f})")
            beyond_doubt = beyond_doubt or measured_slower or now >= ORDER_OF_MAGNITUDE * max(was, 1e-9)
    memory = [(float(row["at"]), float(row["memory_mb"])) for row in rows
              if "memory_mb" in row and float(row["memory_mb"]) > 0.0]
    now, was, chance, spread = _grows_faster([one for one in memory if one[0] >= landed_at],
                                             [one for one in memory if one[0] < landed_at])
    if chance < ALPHA:
        said.append(f"her memory grows {now:.0f} MB an hour since it landed, and grew {was:.0f} MB an hour before "
                    f"(a chance of {chance:.3f})")
        beyond_doubt = beyond_doubt or now >= ORDER_OF_MAGNITUDE * max(was, spread, 1e-9)
    return {"said": "; ".join(said), "beyond_doubt": beyond_doubt}
