#!/usr/bin/env python3
"""Retire the open goals in her goal store that are requests her code wrote for a model.

Components asked her model for text through the cognitive engine, and the engine took each request as the
turn's objective: the swarm's role table until 15 September, self-play's adversary until 18 October, and others
before them. Those producers are fixed. What they left is still open in her goal store, and her initiative
arbiter keeps choosing it: on 19 October "[SWARM PROTOCOL: You are 'The Architect'..." won 25 times in three
hours and failed to plan every time ("No JSON array in response").

A goal is retired only when its objective opens with one of the templates below, and only when her source
holds that template or once held it (git, now or in its history). The status becomes "abandoned" through the
goal engine's own update, with the template and the commit that shows it was code. The store is copied first.

Usage:
    retire_requests_held_as_goals.py [--db PATH] [--backup-dir DIR] [--apply]
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Openings of requests written for a model by her code, each found among her goals, and the pattern the code
#: wrote it with (several were f-strings).
TEMPLATES = {
    "[SWARM PROTOCOL: You are '": r"\[SWARM PROTOCOL: You are '",
    "[SYSTEM ROLE: ": r"\[SYSTEM ROLE: ",
    "Simulate 3 plausible outcomes of this action": r"Simulate .{0,8} plausible outcomes of this action",
    "Predict how self will react if I take this action": r"Predict how .{0,30}will react",
}
OPEN = ("queued", "in_progress", "blocked", "paused")


@functools.cache
def where_her_code_held(pattern: str) -> str:
    """The first commit whose change to her code added or removed the pattern, or "" when none did."""
    found = subprocess.run(["git", "-c", "core.fsmonitor=false", "-C", str(ROOT), "log", f"-G{pattern}", "--format=%h",
                            "--", "core", "skills", "interface"], capture_output=True, text=True, check=False)
    commits = found.stdout.split()
    return commits[-1] if commits else ""


def requests_held_as_goals(db: Path) -> list[dict]:
    """Open goals whose objective opens with a template, with the template and where her code held it."""
    held = {t: where_her_code_held(pattern) for t, pattern in TEMPLATES.items()}
    unproven = [t for t, commit in held.items() if not commit]
    if unproven:
        raise SystemExit(f"not found in her code or its history, so not retired: {unproven}")
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            f"SELECT id, status, source, objective FROM goals WHERE status IN ({', '.join('?' for _ in OPEN)})",
            OPEN).fetchall()
    finally:
        connection.close()
    found = []
    for goal_id, status, source, objective in rows:
        text = str(objective or "").lstrip()
        template = next((t for t in TEMPLATES if text.startswith(t)), "")
        if template:
            found.append({"id": goal_id, "status": status, "source": source, "template": template,
                          "commit": held[template], "objective": text[:120]})
    return found


async def retire(db: Path, found: list[dict]) -> int:
    from core.goals.goal_engine import GoalEngine, GoalStatus
    from core.governance_context import local_internal_governed_scope

    engine = GoalEngine(db_path=str(db))
    retired = 0
    with local_internal_governed_scope("retire_requests_held_as_goals"):
        for goal in found:
            updated = await engine.update_goal_status(
                goal["id"], status=GoalStatus.ABANDONED.value,
                error=f"a request her code wrote for a model, not a goal: {goal['template']!r} (code at {goal['commit']})",
                metadata={"retired_by": "tools/retire_requests_held_as_goals.py", "retired_at": time.time()})
            retired += updated is not None
    engine.close()
    return retired


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    from core.goals.goal_engine import GoalEngine

    db = (args.db or GoalEngine._default_db_path()).expanduser().resolve(strict=True)
    found = requests_held_as_goals(db)
    for goal in found:
        print(json.dumps(goal))
    print(f"{len(found)} open goals in {db} are requests her code wrote for a model", flush=True)
    if not args.apply or not found:
        return 0
    if args.backup_dir is None:
        raise SystemExit("--apply needs --backup-dir: the store is copied before anything changes")
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    copy = args.backup_dir / f"{db.name}.before-retire-{time.strftime('%Y%m%d-%H%M%S')}"
    source, target = sqlite3.connect(f"file:{db}?mode=ro", uri=True), sqlite3.connect(copy)
    try:
        source.backup(target)
    finally:
        source.close()
        target.close()
    print(f"copied to {copy}")
    print(f"retired {asyncio.run(retire(db, found))} of {len(found)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
