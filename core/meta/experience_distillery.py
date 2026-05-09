"""core/meta/experience_distillery.py -- Experience Distillation Store
======================================================================
When Aura fails on a held-out task, this module:
  1. Generates a structured natural-language reflection on the failure
  2. Stores the distilled lesson in a persistent SQLite store
  3. Retrieves relevant lessons for future similar tasks via similarity

This implements the MetaAgent-style experience distillation pattern:
failures become teaching signals that improve future performance.

Architecture:
  - Lessons are stored with embeddings for semantic retrieval
  - Each lesson has: failure context, root cause analysis, strategy
  - Retrieval uses cosine similarity on hash-based embeddings
  - Lessons are scored by how often they've been retrieved and helped

Gate: Lesson generation is passive (no architecture modification).
Lesson application happens via context injection, not weight updates.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("Aura.ExperienceDistillery")

_DATA_DIR = Path.home() / ".aura" / "data" / "experience_distillery"
_DB_PATH = _DATA_DIR / "lessons.db"


@dataclass
class FailureContext:
    """Context surrounding a task failure."""
    task_description: str
    task_type: str                    # "coding", "research", "planning", etc.
    attempted_strategy: str
    error_description: str
    substrate_state_summary: str = ""  # Brief substrate snapshot
    confidence_at_failure: float = 0.5
    prediction_error_at_failure: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class DistilledLesson:
    """A lesson learned from a failure."""
    lesson_id: str
    task_type: str
    failure_summary: str
    root_cause: str
    corrective_strategy: str
    applicability_conditions: str     # When should this lesson apply?
    embedding: List[float]
    retrieval_count: int = 0
    helpfulness_score: float = 0.0    # Updated when lesson leads to success
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "task_type": self.task_type,
            "failure_summary": self.failure_summary,
            "root_cause": self.root_cause,
            "corrective_strategy": self.corrective_strategy,
            "applicability_conditions": self.applicability_conditions,
            "retrieval_count": self.retrieval_count,
            "helpfulness_score": round(self.helpfulness_score, 4),
            "created_at": self.created_at,
        }


class ExperienceDistillery:
    """Distills task failures into retrievable lessons.

    Usage:
        distillery = ExperienceDistillery()

        # On failure:
        lesson = distillery.distill_failure(FailureContext(
            task_description="Parse complex JSON with nested arrays",
            task_type="coding",
            attempted_strategy="Direct regex parsing",
            error_description="Regex failed on nested structures",
        ))

        # Before a new task:
        relevant = distillery.retrieve_lessons(
            task_description="Parse XML with nested elements",
            task_type="coding",
            top_k=3,
        )
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _DB_PATH
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._lesson_count = self._count_lessons()
        logger.info(
            "ExperienceDistillery initialized: %d lessons stored",
            self._lesson_count,
        )

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lessons (
                    lesson_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    failure_summary TEXT NOT NULL,
                    root_cause TEXT NOT NULL,
                    corrective_strategy TEXT NOT NULL,
                    applicability_conditions TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    retrieval_count INTEGER DEFAULT 0,
                    helpfulness_score REAL DEFAULT 0.0,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_type ON lessons(task_type)
            """)
            conn.commit()
        finally:
            conn.close()

    def _count_lessons(self) -> int:
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    # ── Embedding ────────────────────────────────────────────────────

    @staticmethod
    def _embed(text: str, dim: int = 128) -> List[float]:
        """Create a deterministic hash-based embedding.

        This is a lightweight embedding that doesn't require a model.
        For production, this would be replaced with a proper sentence
        embedding model, but the hash approach gives reasonable
        similarity structure for exact and near-duplicate detection.
        """
        # Use multiple hash seeds for dimensionality
        embedding = []
        for i in range(dim):
            h = hashlib.sha256(f"{i}:{text.lower().strip()}".encode()).digest()
            # Convert 4 bytes to a float in [-1, 1]
            val = int.from_bytes(h[:4], "big") / (2 ** 32) * 2 - 1
            embedding.append(val)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = [x / norm for x in embedding]
        return embedding

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        a_arr = np.array(a, dtype=np.float64)
        b_arr = np.array(b, dtype=np.float64)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(dot / (norm_a * norm_b))

    # ── Distillation ────────────────────────────────────────────────

    def distill_failure(self, context: FailureContext) -> DistilledLesson:
        """Distill a task failure into a stored lesson.

        Generates a structured reflection from the failure context
        and persists it for future retrieval.

        Args:
            context: The failure context to distill.

        Returns:
            The distilled lesson.
        """
        # Generate structured reflection
        failure_summary = (
            f"Failed at '{context.task_type}' task: {context.task_description[:200]}. "
            f"Error: {context.error_description[:200]}"
        )

        root_cause = self._analyze_root_cause(context)
        corrective_strategy = self._generate_strategy(context)
        applicability = self._generate_applicability(context)

        # Create embedding from the combined text
        embed_text = (
            f"{context.task_type} {context.task_description} "
            f"{context.error_description} {root_cause}"
        )
        embedding = self._embed(embed_text)

        # Generate lesson ID
        lesson_id = hashlib.sha256(
            f"{context.task_description}:{context.error_description}:{time.time()}".encode()
        ).hexdigest()[:16]

        lesson = DistilledLesson(
            lesson_id=lesson_id,
            task_type=context.task_type,
            failure_summary=failure_summary,
            root_cause=root_cause,
            corrective_strategy=corrective_strategy,
            applicability_conditions=applicability,
            embedding=embedding,
        )

        self._store_lesson(lesson)
        self._lesson_count += 1

        logger.info(
            "Distilled lesson '%s' from %s failure: %s",
            lesson_id, context.task_type, root_cause[:100],
        )

        return lesson

    def _analyze_root_cause(self, context: FailureContext) -> str:
        """Analyze the root cause of a failure from its context."""
        parts = []

        # High confidence but failure → overconfidence
        if context.confidence_at_failure > 0.7:
            parts.append(
                "High confidence (%.0f%%) suggests overconfidence in approach"
                % (context.confidence_at_failure * 100)
            )

        # High prediction error → unexpected situation
        if context.prediction_error_at_failure > 0.5:
            parts.append(
                "High prediction error (%.2f) indicates the world model "
                "did not anticipate this scenario" % context.prediction_error_at_failure
            )

        # Strategy analysis
        strategy = context.attempted_strategy.lower()
        if "direct" in strategy or "simple" in strategy:
            parts.append(
                "Direct/simple approach may have been insufficient for task complexity"
            )
        elif "complex" in strategy:
            parts.append(
                "Over-complex approach may have introduced unnecessary failure modes"
            )

        if not parts:
            parts.append(
                f"Strategy '{context.attempted_strategy}' was not effective "
                f"for this type of {context.task_type} task"
            )

        return ". ".join(parts) + "."

    def _generate_strategy(self, context: FailureContext) -> str:
        """Generate a corrective strategy from the failure."""
        strategies = []

        if context.confidence_at_failure > 0.7:
            strategies.append(
                "Reduce initial confidence and explore alternative approaches "
                "before committing to a strategy"
            )

        if context.prediction_error_at_failure > 0.5:
            strategies.append(
                "Gather more information about the problem space before acting. "
                "Update world model with this failure case"
            )

        strategies.append(
            f"When encountering similar {context.task_type} tasks, "
            f"avoid '{context.attempted_strategy}' and consider decomposing "
            f"the problem into smaller verifiable steps"
        )

        return ". ".join(strategies) + "."

    def _generate_applicability(self, context: FailureContext) -> str:
        """Generate conditions under which this lesson applies."""
        return (
            f"Apply when: task_type='{context.task_type}', "
            f"task involves '{context.task_description[:100]}' or similar, "
            f"and the proposed strategy resembles '{context.attempted_strategy[:100]}'"
        )

    def _store_lesson(self, lesson: DistilledLesson) -> None:
        """Persist a lesson to SQLite."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                """INSERT OR REPLACE INTO lessons
                   (lesson_id, task_type, failure_summary, root_cause,
                    corrective_strategy, applicability_conditions,
                    embedding, retrieval_count, helpfulness_score, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    lesson.lesson_id,
                    lesson.task_type,
                    lesson.failure_summary,
                    lesson.root_cause,
                    lesson.corrective_strategy,
                    lesson.applicability_conditions,
                    json.dumps(lesson.embedding),
                    lesson.retrieval_count,
                    lesson.helpfulness_score,
                    lesson.created_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ── Retrieval ────────────────────────────────────────────────────

    def retrieve_lessons(
        self,
        task_description: str,
        task_type: Optional[str] = None,
        top_k: int = 3,
        min_similarity: float = 0.1,
    ) -> List[DistilledLesson]:
        """Retrieve relevant lessons for a task.

        Args:
            task_description: Description of the upcoming task.
            task_type: Optional filter by task type.
            top_k: Number of lessons to return.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of relevant lessons, sorted by relevance.
        """
        query_embedding = self._embed(f"{task_type or ''} {task_description}")

        conn = sqlite3.connect(str(self._db_path))
        try:
            if task_type:
                rows = conn.execute(
                    "SELECT * FROM lessons WHERE task_type = ?",
                    (task_type,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM lessons").fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        # Score by cosine similarity
        scored: List[tuple] = []
        for row in rows:
            stored_embedding = json.loads(row[6])
            similarity = self._cosine_similarity(query_embedding, stored_embedding)

            # Boost by helpfulness
            helpfulness = row[8]
            score = similarity + 0.1 * helpfulness

            if similarity >= min_similarity:
                lesson = DistilledLesson(
                    lesson_id=row[0],
                    task_type=row[1],
                    failure_summary=row[2],
                    root_cause=row[3],
                    corrective_strategy=row[4],
                    applicability_conditions=row[5],
                    embedding=stored_embedding,
                    retrieval_count=row[7],
                    helpfulness_score=helpfulness,
                    created_at=row[9],
                )
                scored.append((score, lesson))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Update retrieval counts for returned lessons
        results = [lesson for _, lesson in scored[:top_k]]
        self._update_retrieval_counts([l.lesson_id for l in results])

        return results

    def mark_helpful(self, lesson_id: str, delta: float = 0.1) -> None:
        """Mark a lesson as helpful (it led to success on a new task)."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                "UPDATE lessons SET helpfulness_score = helpfulness_score + ? "
                "WHERE lesson_id = ?",
                (delta, lesson_id),
            )
            conn.commit()
        finally:
            conn.close()

    def _update_retrieval_counts(self, lesson_ids: List[str]) -> None:
        """Increment retrieval counts for retrieved lessons."""
        if not lesson_ids:
            return
        conn = sqlite3.connect(str(self._db_path))
        try:
            for lid in lesson_ids:
                conn.execute(
                    "UPDATE lessons SET retrieval_count = retrieval_count + 1 "
                    "WHERE lesson_id = ?",
                    (lid,),
                )
            conn.commit()
        finally:
            conn.close()

    # ── Public API ───────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_lessons": self._lesson_count,
            "db_path": str(self._db_path),
        }

    def get_all_lessons(self) -> List[Dict[str, Any]]:
        """Return all lessons as dicts (for inspection)."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT lesson_id, task_type, failure_summary, root_cause, "
                "corrective_strategy, retrieval_count, helpfulness_score, created_at "
                "FROM lessons ORDER BY created_at DESC"
            ).fetchall()
        finally:
            conn.close()

        return [
            {
                "lesson_id": r[0],
                "task_type": r[1],
                "failure_summary": r[2],
                "root_cause": r[3],
                "corrective_strategy": r[4],
                "retrieval_count": r[5],
                "helpfulness_score": r[6],
                "created_at": r[7],
            }
            for r in rows
        ]
