"""The order her goals are served in, and the stored rank that lets an index serve it."""
from __future__ import annotations

import sqlite3

#: The order goals are served in, as one expression. It ranks by what she is
#: doing now before what is waiting, then by priority and recency.
_SERVING_RANK = (
    "CASE WHEN status = 'in_progress' THEN 0 WHEN status = 'queued' THEN 1 "
    "WHEN status = 'blocked' THEN 2 WHEN status = 'paused' THEN 3 ELSE 4 END"
)

#: That rank stored, so an index can serve the order instead of a sort.
#:
#: `ORDER BY CASE ... END, priority DESC, updated_at DESC` cannot use an index,
#: because the rank is computed per row, so every read scanned the whole table
#: and built a temporary B-tree. At 3,622 goals on the live store that is
#: 12.3 ms on an idle read-only handle with nothing competing; on the loop, under
#: the write lock, with the WAL writer active, it is where the loop was in 13 of
#: 120 stall dumps read on 29 September (`_fetch_records`, lines 934, 936 and
#: 1051 of the frames). With the rank stored and indexed the plan loses its
#: temporary B-tree and the same hundred rows come back in the same order in
#: 0.89 ms.
_SERVING_ORDER_MIGRATION = (
    f"ALTER TABLE goals ADD COLUMN status_rank INTEGER "
    f"GENERATED ALWAYS AS ({_SERVING_RANK}) VIRTUAL",
    "CREATE INDEX IF NOT EXISTS idx_goal_serving_order "
    "ON goals(status_rank, priority DESC, updated_at DESC)",
)


def _add_the_serving_order(conn: sqlite3.Connection) -> None:
    """Give an existing store the stored rank and its index.

    The schema is `CREATE TABLE IF NOT EXISTS`, so a store that already exists
    never sees a new column. A generated column costs no bytes and needs no
    backfill, which is why the migration is an ALTER rather than a rebuild.
    """
    for statement in _SERVING_ORDER_MIGRATION:
        try:
            conn.execute(statement)
        # not a failure: the column is already there, or this SQLite is older
        # than generated columns (3.31). `_fetch_records` reads the plan and
        # falls back to the expression either way.
        except sqlite3.OperationalError:
            continue


def serving_order(conn: sqlite3.Connection | None, known: bool | None) -> tuple[str, bool]:
    """`status_rank` where the store has it, and the expression where it does not.

    The stored rank is what lets an index serve the order. A store on a
    SQLite without generated columns has no such column, and reading it
    would raise rather than sort — so the expression stays as the fallback
    and both give the same rows in the same order. `known` is the answer
    from the last call, handed back so the table is read once.
    """
    if known is None:
        known = False
        if conn is not None:
            try:
                # table_xinfo, because table_info leaves generated columns out: asked that way the rank was never
                # found, and every read sorted the whole table from the day the index was added (found 15 November
                # 2026 by the test that replaced reading this function's source with running it). A SQLite older
                # than the pragma answers an unknown one with no rows, and the expression serves.
                known = any(
                    row[1] == "status_rank"
                    for row in conn.execute("PRAGMA table_xinfo(goals)")
                )
            # not a failure: an unreadable store sorts by the expression.
            except sqlite3.Error:
                known = False
    return ("status_rank" if known else _SERVING_RANK), known
