"""The order goals are served in comes from an index, not from a sort of the table.

`ORDER BY CASE ... END, priority DESC, updated_at DESC` cannot use an index,
because the rank is computed per row, so every read scanned the whole table and
built a temporary B-tree under the write lock. At 3,622 goals on the live store
that is 12.3 ms on an idle read-only handle with nothing competing; on the event
loop, under that lock, with the WAL writer active, it is where the loop was in 13
of 120 stall dumps read on 29 September.

The rank is stored as a generated column and indexed. This holds the plan rather
than a duration, because a timing threshold on a loaded host is what produced
the stall reports in the first place.
"""

from __future__ import annotations

import sqlite3

import pytest

from core.goals.goal_engine import _SCHEMA
from core.goals.goal_serving_order import (
    _SERVING_RANK,
    _add_the_serving_order,
)

pytestmark = pytest.mark.unit

BY_RANK = "SELECT * FROM goals ORDER BY status_rank, priority DESC, updated_at DESC LIMIT 100"
BY_EXPRESSION = (
    f"SELECT * FROM goals ORDER BY {_SERVING_RANK}, priority DESC, updated_at DESC LIMIT 100"
)
STATUSES = ("in_progress", "queued", "blocked", "paused", "done")


@pytest.fixture
def store():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(goals)")]
    kinds = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(goals)")}
    for i in range(400):
        values = []
        for name in columns:
            if name == "id":
                values.append(f"g{i}")
            elif name == "status":
                values.append(STATUSES[i % len(STATUSES)])
            elif name == "priority":
                values.append((i % 9) / 10.0)
            elif name in ("created_at", "updated_at"):
                values.append(float(i))
            elif kinds[name].upper().startswith(("REAL", "INT")):
                values.append(0)
            else:
                values.append("")
        conn.execute(
            f"INSERT INTO goals ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
            values,
        )
    conn.commit()
    return conn


def _plan(conn, query: str) -> str:
    return " | ".join(str(row[-1]) for row in conn.execute("EXPLAIN QUERY PLAN " + query))


def test_without_the_stored_rank_the_read_sorts_the_table(store):
    plan = _plan(store, BY_EXPRESSION)
    assert "TEMP B-TREE" in plan, plan


def test_with_it_the_index_serves_the_order(store):
    _add_the_serving_order(store)
    plan = _plan(store, BY_RANK)
    assert "idx_goal_serving_order" in plan, plan
    assert "TEMP B-TREE" not in plan, plan


def test_the_rows_and_their_order_are_the_same_either_way(store):
    _add_the_serving_order(store)
    assert [row["id"] for row in store.execute(BY_RANK)] == [
        row["id"] for row in store.execute(BY_EXPRESSION)
    ]


def test_the_migration_is_safe_to_run_twice(store):
    _add_the_serving_order(store)
    _add_the_serving_order(store)
    assert "idx_goal_serving_order" in _plan(store, BY_RANK)


def test_a_store_without_the_column_still_reads(store):
    # What `serving_order` falls back to when generated columns are absent.
    assert [row["id"] for row in store.execute(BY_EXPRESSION)]
    with pytest.raises(sqlite3.OperationalError):
        store.execute(BY_RANK)


@pytest.mark.parametrize("ranked", [True, False])
def test_the_engine_asks_the_store_which_order_it_can_use(store, ranked):
    """The engine's read is served by the index where the store has the rank, and still reads where it does not."""
    import threading

    from core.goals.goal_engine import GoalEngine

    if ranked:
        _add_the_serving_order(store)
    asked: list[str] = []
    store.set_trace_callback(asked.append)
    engine = GoalEngine.__new__(GoalEngine)
    engine._conn, engine._lock, engine._has_serving_rank = store, threading.RLock(), None
    served = [record.id for record in engine._fetch_records(limit=100)]
    read = next(query for query in asked if query.startswith("SELECT * FROM goals"))
    assert served == [row["id"] for row in store.execute(BY_EXPRESSION)]
    if ranked:
        # Asked by reading the store's columns the way generated ones are listed: it was asked with table_info,
        # which leaves them out, so every read sorted the table (15 November 2026).
        assert "CASE WHEN status" not in read and "TEMP B-TREE" not in _plan(store, read), read


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
