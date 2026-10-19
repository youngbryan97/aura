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
