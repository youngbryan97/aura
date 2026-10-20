"""Requests her code wrote for a model are retired from her goal store; her own goals are left as they are."""
from __future__ import annotations

import asyncio
import importlib.util
import sqlite3
import time
from pathlib import Path

from core.goals.goal_engine import GoalEngine

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "retire_requests_held_as_goals.py"
_spec = importlib.util.spec_from_file_location("retire_requests_held_as_goals", _TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

SWARM = "[SWARM PROTOCOL: You are 'The Architect'. Design the high-level structure to solve the problem.]"


def _store(tmp_path: Path) -> Path:
    db = tmp_path / "goal_lifecycle.db"
    GoalEngine(db_path=str(db)).close()
    now = time.time()
    with sqlite3.connect(db) as connection:
        for goal_id, objective, status in (("a", SWARM, "in_progress"), ("b", SWARM, "completed"),
                                           ("c", "Repair pong so the paddle tracks the ball", "in_progress")):
            connection.execute("INSERT INTO goals (id, name, objective, status, horizon, created_at, updated_at) "
                               "VALUES (?, ?, ?, ?, 'short_term', ?, ?)", (goal_id, objective[:40], objective,
                                                                           status, now, now))
    return db


def test_every_template_is_shown_to_have_been_her_code():
    assert all(tool.where_her_code_held(pattern) for pattern in tool.TEMPLATES.values())


def test_only_open_goals_that_open_with_a_template_are_found_and_retired(tmp_path):
    db = _store(tmp_path)
    found = tool.requests_held_as_goals(db)
    assert [goal["id"] for goal in found] == ["a"]
    assert asyncio.run(tool.retire(db, found)) == 1
    with sqlite3.connect(db) as connection:
        statuses = dict(connection.execute("SELECT id, status FROM goals").fetchall())
        error = connection.execute("SELECT error FROM goals WHERE id = 'a'").fetchone()[0]
    assert statuses == {"a": "abandoned", "b": "completed", "c": "in_progress"}
    assert "not a goal" in error and found[0]["commit"] in error


def test_tasks_in_her_commitments_that_are_requests_move_out_whole(tmp_path):
    import json

    store = tmp_path / "task_commitment_state.json"
    sandbox = "The following python code failed with an error in the sandbox:\n\nCODE:\nprint(x)\n\nERROR:\nNameError"
    tasks = [{"task_id": "s", "objective": sandbox, "status": "deferred"},
             {"task_id": "w", "objective": SWARM, "status": "interrupted"},
             {"task_id": "h", "objective": "Repair pong so the paddle tracks the ball", "status": "deferred"}]
    store.write_text(json.dumps({"active_tasks": tasks, "updated_at": 1.0}))

    found = tool.requests_held_as_tasks(store)
    retired = tool.retire_tasks(store, found)

    assert [task["task_id"] for task in json.loads(store.read_text())["active_tasks"]] == ["h"]
    moved = json.loads(retired.read_text())
    assert [task["task_id"] for task in moved] == ["s", "w"]
    assert all("not a task" in task["retired_because"] and task["objective"] for task in moved)
