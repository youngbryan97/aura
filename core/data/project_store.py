"""SQLite-backed project and strategic task persistence."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.ProjectStore")

PROJECT_STATUSES = frozenset({"active", "completed", "archived", "failed"})
TASK_STATUSES = frozenset({"pending", "in_progress", "completed", "failed", "archived"})


@dataclass(frozen=True)
class StrategicTask:
    id: str
    project_id: str
    description: str
    status: str = "pending"
    parent_id: str | None = None
    priority: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    goal: str
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())


class ProjectStore:
    """Durable store for Aura's strategic projects and task graph."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a caller-controlled transaction with fail-closed rollback."""
        conn = self._get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        finally:
            exc_type = sys.exc_info()[0]
            if conn.in_transaction:
                if exc_type is None:
                    conn.commit()
                else:
                    conn.rollback()
            conn.close()

    def _init_db(self) -> None:
        """Initialize or migrate the project/task schema."""
        with self.transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    parent_id TEXT,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority INTEGER NOT NULL DEFAULT 1,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY (parent_id) REFERENCES tasks(id) ON DELETE SET NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_projects_status_created "
                "ON projects(status, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_project_status_priority "
                "ON tasks(project_id, status, priority DESC, created_at)"
            )
        logger.debug("ProjectStore initialized at %s", self.db_path)

    def create_project(
        self,
        name: str,
        goal: str,
        metadata: dict[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> Project:
        now = time.time()
        project = Project(
            id=str(uuid.uuid4()),
            name=_non_empty(name, "project name"),
            goal=_non_empty(goal, "project goal"),
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )

        def _insert(c: sqlite3.Connection) -> Project:
            c.execute(
                """
                INSERT INTO projects (id, name, goal, status, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project.id,
                    project.name,
                    project.goal,
                    project.status,
                    _encode_metadata(project.metadata),
                    project.created_at,
                    project.updated_at,
                ),
            )
            return project

        return self._write(_insert, conn=conn)

    def add_task(
        self,
        project_id: str,
        description: str,
        parent_id: str | None = None,
        priority: int = 1,
        metadata: dict[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> StrategicTask:
        now = time.time()
        task = StrategicTask(
            id=str(uuid.uuid4()),
            project_id=_non_empty(project_id, "project_id"),
            parent_id=parent_id,
            description=_non_empty(description, "task description"),
            priority=int(priority),
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )

        def _insert(c: sqlite3.Connection) -> StrategicTask:
            c.execute(
                """
                INSERT INTO tasks (
                    id, project_id, parent_id, description, status,
                    priority, metadata, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.project_id,
                    task.parent_id,
                    task.description,
                    task.status,
                    task.priority,
                    _encode_metadata(task.metadata),
                    task.created_at,
                    task.updated_at,
                ),
            )
            return task

        return self._write(_insert, conn=conn)

    def get_project(self, project_id: str) -> Project | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?",
                (_non_empty(project_id, "project_id"),),
            ).fetchone()
        return _project_from_row(row) if row else None

    def get_active_projects(self) -> list[Project]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM projects WHERE status = 'active' ORDER BY created_at ASC"
            ).fetchall()
        return [_project_from_row(row) for row in rows]

    def get_tasks_for_project(self, project_id: str) -> list[StrategicTask]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tasks
                WHERE project_id = ?
                ORDER BY priority DESC, created_at ASC
                """,
                (_non_empty(project_id, "project_id"),),
            ).fetchall()
        return [_task_from_row(row) for row in rows]

    def get_next_strategic_task(self) -> StrategicTask | None:
        """Return the highest-priority pending task across active projects."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT t.* FROM tasks t
                JOIN projects p ON t.project_id = p.id
                WHERE p.status = 'active' AND t.status = 'pending'
                ORDER BY t.priority DESC, t.created_at ASC
                LIMIT 1
                """
            ).fetchone()
        return _task_from_row(row) if row else None

    def update_task_status(
        self,
        task_id: str,
        status: str,
        metadata_update: dict[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> bool:
        _validate_status(status, TASK_STATUSES, "task")

        def _update(c: sqlite3.Connection) -> bool:
            row = c.execute(
                "SELECT metadata FROM tasks WHERE id = ?",
                (_non_empty(task_id, "task_id"),),
            ).fetchone()
            if row is None:
                return False
            metadata = _decode_metadata(row["metadata"])
            if metadata_update:
                metadata.update(metadata_update)
            cursor = c.execute(
                """
                UPDATE tasks
                SET status = ?, metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, _encode_metadata(metadata), time.time(), task_id),
            )
            return cursor.rowcount == 1

        return self._write(_update, conn=conn)

    def update_project_status(
        self,
        project_id: str,
        status: str,
        conn: sqlite3.Connection | None = None,
    ) -> bool:
        _validate_status(status, PROJECT_STATUSES, "project")

        def _update(c: sqlite3.Connection) -> bool:
            cursor = c.execute(
                """
                UPDATE projects
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, time.time(), _non_empty(project_id, "project_id")),
            )
            return cursor.rowcount == 1

        return self._write(_update, conn=conn)

    def _write(
        self,
        operation: Callable[[sqlite3.Connection], Any],
        *,
        conn: sqlite3.Connection | None,
    ) -> Any:
        if conn is not None:
            return operation(conn)
        with self.transaction() as managed:
            return operation(managed)


def _project_from_row(row: sqlite3.Row) -> Project:
    return Project(
        id=str(row["id"]),
        name=str(row["name"]),
        goal=str(row["goal"]),
        status=str(row["status"]),
        metadata=_decode_metadata(row["metadata"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _task_from_row(row: sqlite3.Row) -> StrategicTask:
    return StrategicTask(
        id=str(row["id"]),
        project_id=str(row["project_id"]),
        parent_id=str(row["parent_id"]) if row["parent_id"] is not None else None,
        description=str(row["description"]),
        status=str(row["status"]),
        priority=int(row["priority"]),
        metadata=_decode_metadata(row["metadata"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _encode_metadata(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def _decode_metadata(raw: Any) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    data = json.loads(str(raw))
    if not isinstance(data, dict):
        raise ValueError("project metadata must decode to an object")
    return data


def _non_empty(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _validate_status(status: str, allowed: frozenset[str], label: str) -> None:
    if status not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"invalid {label} status {status!r}; expected one of: {allowed_values}")


__all__ = [
    "PROJECT_STATUSES",
    "TASK_STATUSES",
    "Project",
    "ProjectStore",
    "StrategicTask",
]
