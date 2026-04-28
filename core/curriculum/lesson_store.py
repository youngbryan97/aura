"""SQLite-backed store of lessons learned across curriculum iterations."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Lesson:
    lesson_id: str
    task_id: str
    iteration: int
    belief: str
    modality: str
    strategy: str
    success: bool
    brier: Optional[float]
    summary: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LessonStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = (
            Path(db_path)
            if db_path is not None
            else Path.home() / ".aura" / "data" / "curriculum_lessons.db"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lessons (
                    lesson_id  TEXT PRIMARY KEY,
                    task_id    TEXT NOT NULL,
                    iteration  INTEGER NOT NULL,
                    belief     TEXT NOT NULL,
                    modality   TEXT NOT NULL,
                    strategy   TEXT NOT NULL,
                    success    INTEGER NOT NULL,
                    brier      REAL,
                    summary    TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lessons_belief ON lessons(belief);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lessons_iteration ON lessons(iteration);"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def append(self, lesson: Lesson) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO lessons(
                    lesson_id, task_id, iteration, belief, modality, strategy,
                    success, brier, summary, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    lesson.lesson_id,
                    lesson.task_id,
                    lesson.iteration,
                    lesson.belief,
                    lesson.modality,
                    lesson.strategy,
                    1 if lesson.success else 0,
                    lesson.brier,
                    lesson.summary,
                    json.dumps(lesson.metadata, default=str),
                    lesson.created_at,
                ),
            )

    def for_belief(self, belief: str) -> List[Lesson]:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM lessons WHERE belief = ? ORDER BY iteration ASC;",
                (belief,),
            ).fetchall()
        return [self._row_to_lesson(r) for r in rows]

    def all(self) -> List[Lesson]:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM lessons ORDER BY iteration ASC, created_at ASC;"
            ).fetchall()
        return [self._row_to_lesson(r) for r in rows]

    def _row_to_lesson(self, row: sqlite3.Row) -> Lesson:
        return Lesson(
            lesson_id=row["lesson_id"],
            task_id=row["task_id"],
            iteration=int(row["iteration"]),
            belief=row["belief"],
            modality=row["modality"],
            strategy=row["strategy"],
            success=bool(row["success"]),
            brier=row["brier"],
            summary=row["summary"] or "",
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=float(row["created_at"]),
        )
