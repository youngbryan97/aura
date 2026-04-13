from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.container import ServiceContainer

logger = logging.getLogger("Aura.GoalEngine")


class GoalStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class GoalHorizon(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


TERMINAL_GOAL_STATUSES = frozenset(
    {
        GoalStatus.COMPLETED.value,
        GoalStatus.FAILED.value,
        GoalStatus.ABANDONED.value,
    }
)

ACTIVE_GOAL_STATUSES = frozenset(
    {
        GoalStatus.QUEUED.value,
        GoalStatus.IN_PROGRESS.value,
        GoalStatus.BLOCKED.value,
        GoalStatus.PAUSED.value,
    }
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    objective TEXT NOT NULL,
    status TEXT NOT NULL,
    horizon TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'goal_engine',
    priority REAL NOT NULL DEFAULT 0.5,
    progress REAL NOT NULL DEFAULT 0.0,
    quick_win INTEGER NOT NULL DEFAULT 0,
    attention_policy TEXT NOT NULL DEFAULT 'sustained',
    steps_done INTEGER NOT NULL DEFAULT 0,
    steps_total INTEGER NOT NULL DEFAULT 0,
    success_criteria TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    required_tools_json TEXT NOT NULL DEFAULT '[]',
    required_skills_json TEXT NOT NULL DEFAULT '[]',
    evidence_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    project_id TEXT NOT NULL DEFAULT '',
    parent_goal_id TEXT NOT NULL DEFAULT '',
    plan_id TEXT NOT NULL DEFAULT '',
    task_id TEXT NOT NULL DEFAULT '',
    intention_id TEXT NOT NULL DEFAULT '',
    commitment_id TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    last_progress_at REAL
);

CREATE INDEX IF NOT EXISTS idx_goal_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_goal_horizon ON goals(horizon);
CREATE INDEX IF NOT EXISTS idx_goal_updated_at ON goals(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_goal_plan_id ON goals(plan_id);
CREATE INDEX IF NOT EXISTS idx_goal_task_id ON goals(task_id);
CREATE INDEX IF NOT EXISTS idx_goal_project_id ON goals(project_id);
"""


@dataclass
class GoalRecord:
    id: str
    name: str
    objective: str
    status: str = GoalStatus.QUEUED.value
    horizon: str = GoalHorizon.SHORT_TERM.value
    source: str = "goal_engine"
    priority: float = 0.5
    progress: float = 0.0
    quick_win: bool = False
    attention_policy: str = "sustained"
    steps_done: int = 0
    steps_total: int = 0
    success_criteria: str = ""
    summary: str = ""
    error: str = ""
    required_tools: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    project_id: str = ""
    parent_goal_id: str = ""
    plan_id: str = ""
    task_id: str = ""
    intention_id: str = ""
    commitment_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    last_progress_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["quick_win"] = bool(self.quick_win)
        payload["priority"] = round(float(self.priority or 0.0), 4)
        payload["progress"] = round(float(self.progress or 0.0), 4)
        payload["display_status"] = str(self.status or "").replace("_", " ").title()
        payload["display_horizon"] = str(self.horizon or "").replace("_", " ").title()
        payload["is_terminal"] = payload["status"] in TERMINAL_GOAL_STATUSES
        return payload


class GoalEngine:
    """
    Canonical durable goal lifecycle manager.

    Responsibilities:
      - Persist short-horizon execution goals durably across restarts.
      - Mirror active/completed state into cognition without losing history.
      - Merge long-horizon project / planner / commitment state into one snapshot.
      - Expose a single, bounded context block for cognition and executive layers.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = Path(db_path) if db_path else self._default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._state_repo = None
        self._gbm = None
        self._initialize()
        logger.info("GoalEngine initialized with durable store at %s", self._db_path)

    @staticmethod
    def _default_db_path() -> Path:
        try:
            from core.config import config

            return config.paths.data_dir / "goals" / "goal_lifecycle.db"
        except Exception:
            return Path.home() / ".aura" / "data" / "goals" / "goal_lifecycle.db"

    def _initialize(self) -> None:
        try:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except Exception as exc:
            logger.error("GoalEngine initialization failed: %s", exc)
            self._conn = None

    @property
    def state_repo(self):
        if self._state_repo is None:
            self._state_repo = ServiceContainer.get("state_repo", default=None)
        return self._state_repo

    @property
    def gbm(self):
        if self._gbm is None:
            self._gbm = ServiceContainer.get("goal_belief_manager", default=None)
        return self._gbm

    @staticmethod
    def _now() -> float:
        return time.time()

    @staticmethod
    def _coerce_horizon(value: Any, *, default: str = GoalHorizon.SHORT_TERM.value) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_")
        if normalized in {"short", "short_term", "operational", "tactical_short"}:
            return GoalHorizon.SHORT_TERM.value
        if normalized in {"long", "long_term", "strategic", "tactical", "project"}:
            return GoalHorizon.LONG_TERM.value
        return default

    @staticmethod
    def _coerce_status(value: Any, *, default: str = GoalStatus.QUEUED.value) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_")
        mapping = {
            "active": GoalStatus.IN_PROGRESS.value,
            "approved": GoalStatus.QUEUED.value,
            "blocked": GoalStatus.BLOCKED.value,
            "broken": GoalStatus.FAILED.value,
            "completed": GoalStatus.COMPLETED.value,
            "complete": GoalStatus.COMPLETED.value,
            "deferred": GoalStatus.PAUSED.value,
            "failed": GoalStatus.FAILED.value,
            "fulfilled": GoalStatus.COMPLETED.value,
            "in_progress": GoalStatus.IN_PROGRESS.value,
            "interrupted": GoalStatus.BLOCKED.value,
            "intended": GoalStatus.QUEUED.value,
            "partial": GoalStatus.IN_PROGRESS.value,
            "paused": GoalStatus.PAUSED.value,
            "pending": GoalStatus.QUEUED.value,
            "planned": GoalStatus.QUEUED.value,
            "queued": GoalStatus.QUEUED.value,
            "rejected": GoalStatus.ABANDONED.value,
            "running": GoalStatus.IN_PROGRESS.value,
            "started": GoalStatus.IN_PROGRESS.value,
            "succeeded": GoalStatus.COMPLETED.value,
            "waiting_for_approval": GoalStatus.BLOCKED.value,
        }
        return mapping.get(normalized, normalized or default)

    @staticmethod
    def _normalize_strings(values: Iterable[Any]) -> List[str]:
        seen: set[str] = set()
        normalized: List[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
        return normalized

    def _write_record(self, record: GoalRecord) -> GoalRecord:
        if self._conn is None:
            return record
        payload = record.to_dict()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO goals (
                    id, name, objective, status, horizon, source, priority, progress,
                    quick_win, attention_policy, steps_done, steps_total,
                    success_criteria, summary, error, required_tools_json,
                    required_skills_json, evidence_json, metadata_json, project_id,
                    parent_goal_id, plan_id, task_id, intention_id, commitment_id,
                    created_at, updated_at, started_at, completed_at, last_progress_at
                ) VALUES (
                    :id, :name, :objective, :status, :horizon, :source, :priority, :progress,
                    :quick_win, :attention_policy, :steps_done, :steps_total,
                    :success_criteria, :summary, :error, :required_tools_json,
                    :required_skills_json, :evidence_json, :metadata_json, :project_id,
                    :parent_goal_id, :plan_id, :task_id, :intention_id, :commitment_id,
                    :created_at, :updated_at, :started_at, :completed_at, :last_progress_at
                )
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    objective=excluded.objective,
                    status=excluded.status,
                    horizon=excluded.horizon,
                    source=excluded.source,
                    priority=excluded.priority,
                    progress=excluded.progress,
                    quick_win=excluded.quick_win,
                    attention_policy=excluded.attention_policy,
                    steps_done=excluded.steps_done,
                    steps_total=excluded.steps_total,
                    success_criteria=excluded.success_criteria,
                    summary=excluded.summary,
                    error=excluded.error,
                    required_tools_json=excluded.required_tools_json,
                    required_skills_json=excluded.required_skills_json,
                    evidence_json=excluded.evidence_json,
                    metadata_json=excluded.metadata_json,
                    project_id=excluded.project_id,
                    parent_goal_id=excluded.parent_goal_id,
                    plan_id=excluded.plan_id,
                    task_id=excluded.task_id,
                    intention_id=excluded.intention_id,
                    commitment_id=excluded.commitment_id,
                    updated_at=excluded.updated_at,
                    started_at=COALESCE(goals.started_at, excluded.started_at),
                    completed_at=excluded.completed_at,
                    last_progress_at=excluded.last_progress_at
                """,
                {
                    "id": record.id,
                    "name": record.name,
                    "objective": record.objective,
                    "status": record.status,
                    "horizon": record.horizon,
                    "source": record.source,
                    "priority": float(record.priority or 0.0),
                    "progress": float(record.progress or 0.0),
                    "quick_win": 1 if record.quick_win else 0,
                    "attention_policy": record.attention_policy,
                    "steps_done": int(record.steps_done or 0),
                    "steps_total": int(record.steps_total or 0),
                    "success_criteria": record.success_criteria,
                    "summary": record.summary,
                    "error": record.error,
                    "required_tools_json": json.dumps(record.required_tools),
                    "required_skills_json": json.dumps(record.required_skills),
                    "evidence_json": json.dumps(record.evidence),
                    "metadata_json": json.dumps(record.metadata),
                    "project_id": record.project_id,
                    "parent_goal_id": record.parent_goal_id,
                    "plan_id": record.plan_id,
                    "task_id": record.task_id,
                    "intention_id": record.intention_id,
                    "commitment_id": record.commitment_id,
                    "created_at": float(record.created_at or self._now()),
                    "updated_at": float(record.updated_at or self._now()),
                    "started_at": record.started_at,
                    "completed_at": record.completed_at,
                    "last_progress_at": record.last_progress_at,
                },
            )
            self._conn.commit()
        return record

    def _row_to_record(self, row: sqlite3.Row) -> GoalRecord:
        return GoalRecord(
            id=str(row["id"]),
            name=str(row["name"]),
            objective=str(row["objective"]),
            status=str(row["status"]),
            horizon=str(row["horizon"]),
            source=str(row["source"]),
            priority=float(row["priority"] or 0.0),
            progress=float(row["progress"] or 0.0),
            quick_win=bool(row["quick_win"]),
            attention_policy=str(row["attention_policy"] or "sustained"),
            steps_done=int(row["steps_done"] or 0),
            steps_total=int(row["steps_total"] or 0),
            success_criteria=str(row["success_criteria"] or ""),
            summary=str(row["summary"] or ""),
            error=str(row["error"] or ""),
            required_tools=list(json.loads(row["required_tools_json"] or "[]")),
            required_skills=list(json.loads(row["required_skills_json"] or "[]")),
            evidence=list(json.loads(row["evidence_json"] or "[]")),
            metadata=dict(json.loads(row["metadata_json"] or "{}")),
            project_id=str(row["project_id"] or ""),
            parent_goal_id=str(row["parent_goal_id"] or ""),
            plan_id=str(row["plan_id"] or ""),
            task_id=str(row["task_id"] or ""),
            intention_id=str(row["intention_id"] or ""),
            commitment_id=str(row["commitment_id"] or ""),
            created_at=float(row["created_at"] or self._now()),
            updated_at=float(row["updated_at"] or self._now()),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            last_progress_at=row["last_progress_at"],
        )

    def _fetch_records(self, *, statuses: Optional[Iterable[str]] = None, limit: int = 100) -> List[GoalRecord]:
        if self._conn is None:
            return []
        self._conn.row_factory = sqlite3.Row
        query = "SELECT * FROM goals"
        params: List[Any] = []
        if statuses:
            normalized = [self._coerce_status(status) for status in statuses]
            placeholders = ", ".join("?" for _ in normalized)
            query += f" WHERE status IN ({placeholders})"
            params.extend(normalized)
        query += " ORDER BY CASE WHEN status = 'in_progress' THEN 0 WHEN status = 'queued' THEN 1 WHEN status = 'blocked' THEN 2 WHEN status = 'paused' THEN 3 ELSE 4 END, priority DESC, updated_at DESC LIMIT ?"
        params.append(max(1, int(limit or 100)))
        with self._lock:
            cursor = self._conn.execute(query, params)
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def _find_existing_record(
        self,
        *,
        goal_id: str = "",
        plan_id: str = "",
        task_id: str = "",
        project_id: str = "",
        intention_id: str = "",
        commitment_id: str = "",
    ) -> Optional[GoalRecord]:
        if self._conn is None:
            return None
        self._conn.row_factory = sqlite3.Row
        clauses = []
        params: List[Any] = []
        for field_name, value in (
            ("id", goal_id),
            ("plan_id", plan_id),
            ("task_id", task_id),
            ("project_id", project_id),
            ("intention_id", intention_id),
            ("commitment_id", commitment_id),
        ):
            if value:
                clauses.append(f"{field_name} = ?")
                params.append(str(value))
        if not clauses:
            return None
        query = f"SELECT * FROM goals WHERE {' OR '.join(clauses)} ORDER BY updated_at DESC LIMIT 1"
        with self._lock:
            row = self._conn.execute(query, params).fetchone()
        return self._row_to_record(row) if row else None

    def _upsert_goal(
        self,
        *,
        goal_id: Optional[str] = None,
        name: str,
        objective: str,
        status: Any,
        horizon: Any = GoalHorizon.SHORT_TERM.value,
        source: str = "goal_engine",
        priority: float = 0.5,
        progress: Optional[float] = None,
        quick_win: bool = False,
        attention_policy: str = "sustained",
        steps_done: int = 0,
        steps_total: int = 0,
        success_criteria: str = "",
        summary: str = "",
        error: str = "",
        required_tools: Optional[Iterable[Any]] = None,
        required_skills: Optional[Iterable[Any]] = None,
        evidence: Optional[Iterable[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        project_id: str = "",
        parent_goal_id: str = "",
        plan_id: str = "",
        task_id: str = "",
        intention_id: str = "",
        commitment_id: str = "",
        created_at: Optional[float] = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
    ) -> GoalRecord:
        now = self._now()
        normalized_status = self._coerce_status(status)
        existing = self._find_existing_record(
            goal_id=str(goal_id or ""),
            plan_id=str(plan_id or ""),
            task_id=str(task_id or ""),
            project_id=str(project_id or ""),
            intention_id=str(intention_id or ""),
            commitment_id=str(commitment_id or ""),
        )

        if progress is None:
            if steps_total > 0:
                progress = max(0.0, min(1.0, float(steps_done) / float(steps_total)))
            elif existing is not None:
                progress = existing.progress
            else:
                progress = 1.0 if normalized_status == GoalStatus.COMPLETED.value else 0.0

        record = GoalRecord(
            id=str(goal_id or getattr(existing, "id", "") or uuid.uuid4().hex[:12]),
            name=str(name or getattr(existing, "name", "") or objective)[:160],
            objective=str(objective or getattr(existing, "objective", "") or name)[:4000],
            status=normalized_status,
            horizon=self._coerce_horizon(horizon, default=getattr(existing, "horizon", GoalHorizon.SHORT_TERM.value)),
            source=str(source or getattr(existing, "source", "goal_engine")),
            priority=float(priority if priority is not None else getattr(existing, "priority", 0.5)),
            progress=max(0.0, min(1.0, float(progress or 0.0))),
            quick_win=bool(quick_win or getattr(existing, "quick_win", False)),
            attention_policy=str(attention_policy or getattr(existing, "attention_policy", "sustained")),
            steps_done=max(int(steps_done or 0), int(getattr(existing, "steps_done", 0) or 0) if normalized_status in ACTIVE_GOAL_STATUSES else int(steps_done or 0)),
            steps_total=max(int(steps_total or 0), int(getattr(existing, "steps_total", 0) or 0)),
            success_criteria=str(success_criteria or getattr(existing, "success_criteria", "")),
            summary=str(summary or getattr(existing, "summary", ""))[:2400],
            error=str(error or getattr(existing, "error", ""))[:2400],
            required_tools=self._normalize_strings(required_tools or getattr(existing, "required_tools", [])),
            required_skills=self._normalize_strings(required_skills or getattr(existing, "required_skills", [])),
            evidence=self._normalize_strings(evidence or getattr(existing, "evidence", []))[:8],
            metadata=dict(getattr(existing, "metadata", {}) or {}) | dict(metadata or {}),
            project_id=str(project_id or getattr(existing, "project_id", "")),
            parent_goal_id=str(parent_goal_id or getattr(existing, "parent_goal_id", "")),
            plan_id=str(plan_id or getattr(existing, "plan_id", "")),
            task_id=str(task_id or getattr(existing, "task_id", "")),
            intention_id=str(intention_id or getattr(existing, "intention_id", "")),
            commitment_id=str(commitment_id or getattr(existing, "commitment_id", "")),
            created_at=float(created_at or getattr(existing, "created_at", now)),
            updated_at=now,
            started_at=started_at if started_at is not None else (
                getattr(existing, "started_at", None)
                or (now if normalized_status == GoalStatus.IN_PROGRESS.value else None)
            ),
            completed_at=completed_at if completed_at is not None else (
                now if normalized_status in TERMINAL_GOAL_STATUSES else None
            ),
            last_progress_at=now if normalized_status in ACTIVE_GOAL_STATUSES or progress else getattr(existing, "last_progress_at", None),
        )
        if record.status == GoalStatus.COMPLETED.value:
            record.progress = 1.0
            record.completed_at = record.completed_at or now
        if record.status == GoalStatus.FAILED.value and not record.error:
            record.error = "Execution failed."
        return self._write_record(record)

    async def add_goal(
        self,
        name: str,
        objective: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        objective_text = str(objective or name or "").strip()
        goal_name = str(name or objective_text or "Goal").strip()
        record = await asyncio.to_thread(
            self._upsert_goal,
            name=goal_name,
            objective=objective_text,
            status=kwargs.get("status", GoalStatus.QUEUED.value),
            horizon=kwargs.get("horizon", GoalHorizon.SHORT_TERM.value),
            source=kwargs.get("source", "goal_engine"),
            priority=float(kwargs.get("priority", 0.5) or 0.5),
            quick_win=bool(kwargs.get("quick_win", False)),
            attention_policy=str(kwargs.get("attention_policy", "sustained")),
            success_criteria=str(kwargs.get("success_criteria", "")),
            required_tools=kwargs.get("required_tools"),
            required_skills=kwargs.get("required_skills"),
            evidence=kwargs.get("evidence"),
            metadata=kwargs.get("metadata"),
            project_id=str(kwargs.get("project_id", "")),
            parent_goal_id=str(kwargs.get("parent_goal_id", "")),
            commitment_id=str(kwargs.get("commitment_id", "")),
            intention_id=str(kwargs.get("intention_id", "")),
            task_id=str(kwargs.get("task_id", "")),
            plan_id=str(kwargs.get("plan_id", "")),
        )
        if self.gbm and objective_text:
            try:
                self.gbm.reinforce_goal(objective_text, "Direct goal registration.")
            except Exception as exc:
                logger.debug("Goal belief reinforcement skipped: %s", exc)
        self._sync_state_view()
        return record.to_dict()

    async def track_dispatch(
        self,
        objective: str,
        *,
        task_id: str,
        source: str,
        commitment_id: Optional[str] = None,
        priority: float = 0.75,
        horizon: str = GoalHorizon.SHORT_TERM.value,
        quick_win: bool = False,
    ) -> Dict[str, Any]:
        record = await asyncio.to_thread(
            self._upsert_goal,
            name=objective[:140],
            objective=objective,
            status=GoalStatus.IN_PROGRESS.value,
            horizon=horizon,
            source=source,
            priority=priority,
            task_id=task_id,
            commitment_id=str(commitment_id or ""),
            quick_win=quick_win,
            attention_policy="interruptible" if quick_win else "sustained",
            metadata={"dispatch_source": source},
        )
        self._sync_state_view()
        return record.to_dict()

    async def update_task_lifecycle(
        self,
        *,
        task_id: str,
        status: Any,
        summary: str = "",
        error: str = "",
        evidence: Optional[Iterable[Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        existing = await asyncio.to_thread(self._find_existing_record, task_id=task_id)
        if existing is None:
            return None
        record = await asyncio.to_thread(
            self._upsert_goal,
            goal_id=existing.id,
            name=existing.name,
            objective=existing.objective,
            status=status,
            horizon=existing.horizon,
            source=existing.source,
            priority=existing.priority,
            progress=1.0 if self._coerce_status(status) == GoalStatus.COMPLETED.value else existing.progress,
            quick_win=existing.quick_win,
            attention_policy=existing.attention_policy,
            steps_done=existing.steps_done,
            steps_total=existing.steps_total,
            success_criteria=existing.success_criteria,
            summary=summary or existing.summary,
            error=error or existing.error,
            required_tools=existing.required_tools,
            required_skills=existing.required_skills,
            evidence=evidence or existing.evidence,
            metadata=existing.metadata,
            project_id=existing.project_id,
            parent_goal_id=existing.parent_goal_id,
            plan_id=existing.plan_id,
            task_id=existing.task_id,
            intention_id=existing.intention_id,
            commitment_id=existing.commitment_id,
            created_at=existing.created_at,
            started_at=existing.started_at,
        )
        self._sync_state_view()
        return record.to_dict()

    def get_goal(self, goal_id: str) -> Optional[Dict[str, Any]]:
        record = self._find_existing_record(goal_id=str(goal_id or ""))
        return record.to_dict() if record is not None else None

    async def update_goal_status(
        self,
        goal_id: str,
        *,
        status: Any,
        summary: str = "",
        error: str = "",
        progress: Optional[float] = None,
        evidence: Optional[Iterable[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        existing = await asyncio.to_thread(self._find_existing_record, goal_id=str(goal_id or ""))
        if existing is None:
            return None
        normalized_status = self._coerce_status(status)
        record = await asyncio.to_thread(
            self._upsert_goal,
            goal_id=existing.id,
            name=existing.name,
            objective=existing.objective,
            status=normalized_status,
            horizon=existing.horizon,
            source=existing.source,
            priority=existing.priority,
            progress=(
                1.0
                if normalized_status == GoalStatus.COMPLETED.value
                else progress
                if progress is not None
                else existing.progress
            ),
            quick_win=existing.quick_win,
            attention_policy=existing.attention_policy,
            steps_done=existing.steps_total if normalized_status == GoalStatus.COMPLETED.value and existing.steps_total else existing.steps_done,
            steps_total=existing.steps_total,
            success_criteria=existing.success_criteria,
            summary=summary or existing.summary,
            error=error or existing.error,
            required_tools=existing.required_tools,
            required_skills=existing.required_skills,
            evidence=evidence or existing.evidence,
            metadata=dict(existing.metadata or {}) | dict(metadata or {}),
            project_id=existing.project_id,
            parent_goal_id=existing.parent_goal_id,
            plan_id=existing.plan_id,
            task_id=existing.task_id,
            intention_id=existing.intention_id,
            commitment_id=existing.commitment_id,
            created_at=existing.created_at,
            started_at=existing.started_at,
        )
        self._sync_state_view()
        return record.to_dict()

    async def evaluate_goals(self) -> None:
        state = getattr(self.state_repo, "_current", None)
        if state is None:
            return
        recent_messages = [
            str(message.get("content", "") or "")
            for message in list(getattr(state.cognition, "working_memory", []) or [])[-12:]
            if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant"
        ]
        active = await asyncio.to_thread(self._fetch_records, statuses=ACTIVE_GOAL_STATUSES, limit=50)
        for goal in active:
            if not goal.objective:
                continue
            if not self._shows_goal_progress(goal.objective, recent_messages):
                continue
            next_progress = min(1.0, max(goal.progress, goal.progress + 0.2))
            await asyncio.to_thread(
                self._upsert_goal,
                goal_id=goal.id,
                name=goal.name,
                objective=goal.objective,
                status=GoalStatus.COMPLETED.value if next_progress >= 1.0 else GoalStatus.IN_PROGRESS.value,
                horizon=goal.horizon,
                source=goal.source,
                priority=goal.priority,
                progress=next_progress,
                quick_win=goal.quick_win,
                attention_policy=goal.attention_policy,
                steps_done=goal.steps_done,
                steps_total=goal.steps_total,
                success_criteria=goal.success_criteria,
                summary=goal.summary,
                error=goal.error,
                required_tools=goal.required_tools,
                required_skills=goal.required_skills,
                evidence=goal.evidence,
                metadata=goal.metadata,
                project_id=goal.project_id,
                parent_goal_id=goal.parent_goal_id,
                plan_id=goal.plan_id,
                task_id=goal.task_id,
                intention_id=goal.intention_id,
                commitment_id=goal.commitment_id,
                created_at=goal.created_at,
                started_at=goal.started_at,
            )
        self._sync_state_view()

    def sync_task_plan(self, plan: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        plan_context = dict(getattr(plan, "context", {}) or {})
        plan_context.update(dict(context or {}))
        tools = self._normalize_strings(
            getattr(step, "tool", "")
            for step in list(getattr(plan, "steps", []) or [])
        )
        steps_total = len(list(getattr(plan, "steps", []) or []))
        steps_done = len(list(getattr(plan, "succeeded_steps", []) or []))
        quick_win = bool(plan_context.get("quick_win")) or steps_total <= 2
        attention_policy = str(plan_context.get("attention_policy") or ("interruptible" if quick_win else "sustained"))
        record = self._upsert_goal(
            goal_id=str(plan_context.get("goal_id") or ""),
            name=str(getattr(plan, "goal", "") or "Goal")[:160],
            objective=str(getattr(plan, "goal", "") or ""),
            status=str(getattr(plan, "status", GoalStatus.QUEUED.value) or GoalStatus.QUEUED.value),
            horizon=plan_context.get("horizon", GoalHorizon.SHORT_TERM.value),
            source=str(plan_context.get("source", "task_engine")),
            priority=float(plan_context.get("priority", 0.75) or 0.75),
            progress=float((steps_done / steps_total) if steps_total else (1.0 if getattr(plan, "status", "") == "succeeded" else 0.0)),
            quick_win=quick_win,
            attention_policy=attention_policy,
            steps_done=steps_done,
            steps_total=steps_total,
            success_criteria=str(plan_context.get("success_criteria", "")),
            summary=str(getattr(plan, "final_result", "") or plan_context.get("summary", "")),
            error=str(plan_context.get("error", "")),
            required_tools=tools,
            required_skills=plan_context.get("required_skills"),
            evidence=plan_context.get("evidence"),
            metadata={
                "trace_id": getattr(plan, "trace_id", ""),
                "requires_approval": bool(getattr(plan, "requires_approval", False)),
                **dict(plan_context.get("metadata", {}) or {}),
            },
            project_id=str(plan_context.get("project_id", "")),
            parent_goal_id=str(plan_context.get("parent_goal_id", "")),
            plan_id=str(getattr(plan, "plan_id", "") or ""),
            task_id=str(plan_context.get("task_id", "")),
            intention_id=str(plan_context.get("intention_id", "")),
            commitment_id=str(plan_context.get("commitment_id", "")),
            started_at=(self._now() if self._coerce_status(getattr(plan, "status", "")) == GoalStatus.IN_PROGRESS.value else None),
        )
        self._sync_state_view()
        return record.to_dict()

    def get_active_goals(self, limit: int = 12, *, include_external: bool = True) -> List[Dict[str, Any]]:
        snapshot = self.build_snapshot(limit=limit, include_external=include_external)
        return [
            item
            for item in snapshot["items"]
            if str(item.get("status", "")) in ACTIVE_GOAL_STATUSES
        ][: max(1, int(limit or 12))]

    async def get_active_goals_async(self, limit: int = 12, *, include_external: bool = True) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.get_active_goals, limit, include_external=include_external)

    def get_completed_goals(self, limit: int = 20, *, include_external: bool = True) -> List[Dict[str, Any]]:
        snapshot = self.build_snapshot(limit=limit * 3, include_external=include_external)
        completed = [
            item
            for item in snapshot["items"]
            if str(item.get("status", "")) == GoalStatus.COMPLETED.value
        ]
        completed.sort(key=lambda item: float(item.get("completed_at") or item.get("updated_at") or 0.0), reverse=True)
        return completed[: max(1, int(limit or 20))]

    def build_snapshot(self, limit: int = 30, *, include_external: bool = True) -> Dict[str, Any]:
        internal_items = [record.to_dict() for record in self._fetch_records(limit=max(40, int(limit or 30) * 2))]
        items = list(internal_items)
        if include_external:
            items.extend(self._external_goal_items())

        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in self._sort_items(items):
            item_id = str(item.get("id", "") or f"{item.get('source','goal')}::{item.get('objective') or item.get('name')}")
            if item_id in seen:
                continue
            seen.add(item_id)
            deduped.append(item)

        active = [item for item in deduped if str(item.get("status", "")) in ACTIVE_GOAL_STATUSES]
        completed = [item for item in deduped if str(item.get("status", "")) == GoalStatus.COMPLETED.value]
        failed = [item for item in deduped if str(item.get("status", "")) == GoalStatus.FAILED.value]
        blocked = [item for item in deduped if str(item.get("status", "")) == GoalStatus.BLOCKED.value]

        summary = {
            "active_count": len(active),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "blocked_count": len(blocked),
            "short_term_count": sum(1 for item in deduped if item.get("horizon") == GoalHorizon.SHORT_TERM.value and item.get("status") in ACTIVE_GOAL_STATUSES),
            "long_term_count": sum(1 for item in deduped if item.get("horizon") == GoalHorizon.LONG_TERM.value and item.get("status") in ACTIVE_GOAL_STATUSES),
            "queued_count": sum(1 for item in deduped if item.get("status") == GoalStatus.QUEUED.value),
            "in_progress_count": sum(1 for item in deduped if item.get("status") == GoalStatus.IN_PROGRESS.value),
        }
        return {
            "items": deduped[: max(1, int(limit or 30))],
            "summary": summary,
        }

    def get_context_block(self, objective: str = "", limit: int = 6) -> str:
        snapshot = self.build_snapshot(limit=max(12, limit * 2), include_external=True)
        objective_tokens = set(self._normalize_tokens(objective))

        def _relevance(item: Dict[str, Any]) -> float:
            if not objective_tokens:
                return 0.0
            item_tokens = set(
                self._normalize_tokens(
                    f"{item.get('objective') or ''} {item.get('name') or ''} {item.get('summary') or ''}"
                )
            )
            if not item_tokens:
                return 0.0
            return len(objective_tokens & item_tokens) / max(1, len(objective_tokens))

        ranked_items = sorted(
            snapshot["items"],
            key=lambda item: (
                _relevance(item),
                1.0 if str(item.get("status", "")) in ACTIVE_GOAL_STATUSES else 0.0,
                float(item.get("priority", 0.0) or 0.0),
                float(item.get("updated_at") or item.get("created_at") or 0.0),
            ),
            reverse=True,
        )
        active = [item for item in ranked_items if item.get("status") in ACTIVE_GOAL_STATUSES]
        completed = [item for item in ranked_items if item.get("status") == GoalStatus.COMPLETED.value]
        blocked = [item for item in ranked_items if item.get("status") == GoalStatus.BLOCKED.value]
        short_term = [item for item in active if item.get("horizon") == GoalHorizon.SHORT_TERM.value]
        long_term = [item for item in active if item.get("horizon") == GoalHorizon.LONG_TERM.value]
        if not active and not completed:
            return ""
        lines = ["## GOAL EXECUTION STATE"]
        if short_term:
            lines.append("Immediate execution:")
            for item in short_term[: max(1, int(limit or 6))]:
                tools = ", ".join((item.get("required_tools") or [])[:3]) or "none"
                progress = int(round(float(item.get("progress", 0.0) or 0.0) * 100))
                steps_total = int(item.get("steps_total", 0) or 0)
                steps_done = int(item.get("steps_done", 0) or 0)
                step_progress = f" | steps={steps_done}/{steps_total}" if steps_total > 0 else ""
                lines.append(
                    f"- [{str(item.get('status', '')).upper()}] p={float(item.get('priority', 0.0) or 0.0):.2f}"
                    f" {str(item.get('objective') or item.get('name') or '')[:140]}"
                    f" | progress={progress}%{step_progress} | tools={tools}"
                )
        if long_term:
            lines.append("Long-horizon anchors:")
            for item in long_term[:2]:
                progress = int(round(float(item.get("progress", 0.0) or 0.0) * 100))
                source = str(item.get("source", "") or "goal_engine")
                lines.append(
                    f"- [{str(item.get('status', '')).upper()}] {str(item.get('objective') or item.get('name') or '')[:140]}"
                    f" | progress={progress}% | source={source}"
                )
        if blocked:
            lines.append("Recovery pressure:")
            for item in blocked[:2]:
                detail = str(item.get("error") or item.get("summary") or "Needs an explicit resume or unblock step.")[:160]
                lines.append(
                    f"- {str(item.get('objective') or item.get('name') or '')[:140]} | {detail}"
                )
        if completed:
            lines.append("Recently completed:")
            for item in completed[:3]:
                lines.append(f"- {str(item.get('objective') or item.get('name') or '')[:140]}")
        lines.append(
            "Follow through on queued and in-progress goals unless the user explicitly reprioritizes. Prefer quick interruptible wins only when they are truly short and then return to sustained work."
        )
        return "\n".join(lines)

    def _sort_items(self, items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        status_order = {
            GoalStatus.IN_PROGRESS.value: 0,
            GoalStatus.QUEUED.value: 1,
            GoalStatus.BLOCKED.value: 2,
            GoalStatus.PAUSED.value: 3,
            GoalStatus.COMPLETED.value: 4,
            GoalStatus.FAILED.value: 5,
            GoalStatus.ABANDONED.value: 6,
        }

        def _key(item: Dict[str, Any]) -> tuple[Any, ...]:
            status = str(item.get("status", "") or "")
            horizon = str(item.get("horizon", "") or GoalHorizon.SHORT_TERM.value)
            quick_win = 0 if bool(item.get("quick_win", False)) and status in ACTIVE_GOAL_STATUSES else 1
            horizon_rank = 0 if horizon == GoalHorizon.SHORT_TERM.value else 1
            if status == GoalStatus.COMPLETED.value:
                horizon_rank = 2
            updated = float(item.get("updated_at") or item.get("created_at") or 0.0)
            completed = float(item.get("completed_at") or 0.0)
            priority = float(item.get("priority") or 0.0)
            return (
                status_order.get(status, 9),
                quick_win,
                horizon_rank,
                -priority,
                -(completed or updated),
            )

        return sorted(list(items), key=_key)

    def _sync_state_view(self, limit: int = 6) -> None:
        try:
            state = getattr(self.state_repo, "_current", None)
            if state is None:
                return
            cognition = getattr(state, "cognition", None)
            if cognition is None:
                return
            active = self.get_active_goals(limit=limit, include_external=True)
            cognition.active_goals = [
                {
                    "id": item.get("id"),
                    "goal": item.get("objective") or item.get("name"),
                    "description": item.get("objective") or item.get("name"),
                    "status": item.get("status"),
                    "horizon": item.get("horizon"),
                    "priority": item.get("priority"),
                    "steps_done": item.get("steps_done"),
                    "steps_total": item.get("steps_total"),
                    "plan_id": item.get("plan_id"),
                    "task_id": item.get("task_id"),
                    "source": item.get("source"),
                }
                for item in active[:limit]
            ]
            if active and not getattr(cognition, "current_objective", None):
                cognition.current_objective = str(active[0].get("objective") or active[0].get("name") or "")
        except Exception as exc:
            logger.debug("GoalEngine state sync skipped: %s", exc)

    def _external_goal_items(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        items.extend(self._hierarchical_goal_items())
        items.extend(self._strategic_project_items())
        items.extend(self._commitment_items())
        items.extend(self._intention_items())
        return items

    def _hierarchical_goal_items(self) -> List[Dict[str, Any]]:
        try:
            planner = ServiceContainer.get("hierarchical_planner", default=None)
            goals = list(getattr(planner, "_goals", {}).values()) if planner is not None else []
        except Exception:
            goals = []
        items: List[Dict[str, Any]] = []
        for goal in goals:
            level = str(getattr(getattr(goal, "level", None), "value", getattr(goal, "level", "")) or "")
            horizon = GoalHorizon.LONG_TERM.value if level in {"strategic", "tactical"} else GoalHorizon.SHORT_TERM.value
            status = self._coerce_status(getattr(getattr(goal, "status", None), "value", getattr(goal, "status", "")))
            items.append(
                {
                    "id": f"hierarchical:{getattr(goal, 'id', '')}",
                    "name": getattr(goal, "title", "Hierarchical goal"),
                    "objective": getattr(goal, "description", "") or getattr(goal, "title", ""),
                    "status": status,
                    "horizon": horizon,
                    "source": "hierarchical_planner",
                    "priority": 0.95 if level == "strategic" else 0.75 if level == "tactical" else 0.55,
                    "progress": float(getattr(goal, "progress", 0.0) or 0.0),
                    "quick_win": horizon == GoalHorizon.SHORT_TERM.value and float(getattr(goal, "progress", 0.0) or 0.0) < 1.0,
                    "attention_policy": "sustained",
                    "steps_done": 0,
                    "steps_total": max(0, len(getattr(goal, "child_ids", []) or [])),
                    "success_criteria": str(getattr(goal, "success_criteria", "") or ""),
                    "summary": "",
                    "error": "",
                    "required_tools": [],
                    "required_skills": [],
                    "evidence": list(getattr(goal, "notes", []) or [])[-3:],
                    "metadata": {"level": level},
                    "project_id": "",
                    "parent_goal_id": str(getattr(goal, "parent_id", "") or ""),
                    "plan_id": "",
                    "task_id": "",
                    "intention_id": "",
                    "commitment_id": "",
                    "created_at": float(getattr(goal, "created_at", 0.0) or 0.0),
                    "updated_at": float(getattr(goal, "updated_at", 0.0) or 0.0),
                    "started_at": float(getattr(goal, "created_at", 0.0) or 0.0),
                    "completed_at": float(getattr(goal, "updated_at", 0.0) or 0.0) if status == GoalStatus.COMPLETED.value else None,
                    "last_progress_at": float(getattr(goal, "updated_at", 0.0) or 0.0),
                    "display_status": status.replace("_", " ").title(),
                    "display_horizon": horizon.replace("_", " ").title(),
                    "is_terminal": status in TERMINAL_GOAL_STATUSES,
                }
            )
        return items

    def _strategic_project_items(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        try:
            planner = ServiceContainer.get("strategic_planner", default=None)
            store = getattr(planner, "store", None)
            if store is None:
                return items
            with store._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT id, name, goal, status, metadata, created_at, updated_at FROM projects ORDER BY updated_at DESC LIMIT 40"
                )
                project_rows = list(cursor.fetchall())
                task_rows = conn.execute(
                    "SELECT id, project_id, description, status, priority, metadata, created_at, updated_at FROM tasks ORDER BY updated_at DESC LIMIT 120"
                ).fetchall()
            task_map: Dict[str, List[Any]] = {}
            for row in task_rows:
                task_map.setdefault(str(row[1]), []).append(row)
            for row in project_rows:
                project_id = str(row[0])
                tasks = task_map.get(project_id, [])
                active_tasks = [task for task in tasks if self._coerce_status(task[3]) in ACTIVE_GOAL_STATUSES]
                completed_tasks = [task for task in tasks if self._coerce_status(task[3]) == GoalStatus.COMPLETED.value]
                total_tasks = len(tasks)
                progress = (len(completed_tasks) / total_tasks) if total_tasks else 0.0
                status = self._coerce_status(row[3], default=GoalStatus.QUEUED.value)
                if status in ACTIVE_GOAL_STATUSES and active_tasks:
                    status = GoalStatus.IN_PROGRESS.value
                elif status in ACTIVE_GOAL_STATUSES and not active_tasks and completed_tasks:
                    status = GoalStatus.QUEUED.value
                items.append(
                    {
                        "id": f"project:{project_id}",
                        "name": str(row[1] or "Strategic project"),
                        "objective": str(row[2] or row[1] or ""),
                        "status": status,
                        "horizon": GoalHorizon.LONG_TERM.value,
                        "source": "strategic_planner",
                        "priority": 0.85,
                        "progress": progress,
                        "quick_win": False,
                        "attention_policy": "sustained",
                        "steps_done": len(completed_tasks),
                        "steps_total": total_tasks,
                        "success_criteria": "",
                        "summary": "",
                        "error": "",
                        "required_tools": [],
                        "required_skills": [],
                        "evidence": [str(task[2]) for task in active_tasks[:3]],
                        "metadata": {"project_name": str(row[1] or "")},
                        "project_id": project_id,
                        "parent_goal_id": "",
                        "plan_id": "",
                        "task_id": "",
                        "intention_id": "",
                        "commitment_id": "",
                        "created_at": float(row[5] or 0.0),
                        "updated_at": float(row[6] or 0.0),
                        "started_at": float(row[5] or 0.0),
                        "completed_at": float(row[6] or 0.0) if status == GoalStatus.COMPLETED.value else None,
                        "last_progress_at": float(row[6] or 0.0),
                        "display_status": status.replace("_", " ").title(),
                        "display_horizon": "Long Term",
                        "is_terminal": status in TERMINAL_GOAL_STATUSES,
                    }
                )
        except Exception as exc:
            logger.debug("Strategic project snapshot skipped: %s", exc)
        return items

    def _commitment_items(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        try:
            from core.agency.commitment_engine import get_commitment_engine

            engine = get_commitment_engine()
            commitments = list(getattr(engine, "_commitments", {}).values())
        except Exception:
            commitments = []
        for commitment in commitments:
            hours_remaining = float(getattr(commitment, "hours_remaining", lambda: 24.0)() or 24.0)
            priority = 0.9 if hours_remaining < 6 else 0.75 if hours_remaining < 24 else 0.55
            status = self._coerce_status(
                getattr(getattr(commitment, "status", None), "value", getattr(commitment, "status", "")),
                default=GoalStatus.QUEUED.value,
            )
            items.append(
                {
                    "id": f"commitment:{getattr(commitment, 'id', '')}",
                    "name": str(getattr(commitment, "description", "Commitment") or "Commitment"),
                    "objective": str(getattr(commitment, "outcome", "") or getattr(commitment, "description", "")),
                    "status": status,
                    "horizon": GoalHorizon.LONG_TERM.value,
                    "source": "commitment_engine",
                    "priority": priority,
                    "progress": float(getattr(commitment, "progress", 0.0) or 0.0),
                    "quick_win": False,
                    "attention_policy": "sustained",
                    "steps_done": 0,
                    "steps_total": 0,
                    "success_criteria": str(getattr(commitment, "outcome", "") or ""),
                    "summary": "",
                    "error": "",
                    "required_tools": [],
                    "required_skills": [],
                    "evidence": list(getattr(commitment, "notes", []) or [])[-3:],
                    "metadata": {"hours_remaining": hours_remaining},
                    "project_id": "",
                    "parent_goal_id": "",
                    "plan_id": "",
                    "task_id": "",
                    "intention_id": "",
                    "commitment_id": str(getattr(commitment, "id", "") or ""),
                    "created_at": float(getattr(commitment, "created_at", 0.0) or 0.0),
                    "updated_at": float(getattr(commitment, "last_checkin", 0.0) or getattr(commitment, "created_at", 0.0) or 0.0),
                    "started_at": float(getattr(commitment, "created_at", 0.0) or 0.0),
                    "completed_at": float(self._now()) if status == GoalStatus.COMPLETED.value else None,
                    "last_progress_at": float(getattr(commitment, "last_checkin", 0.0) or getattr(commitment, "created_at", 0.0) or 0.0),
                    "display_status": status.replace("_", " ").title(),
                    "display_horizon": "Long Term",
                    "is_terminal": status in TERMINAL_GOAL_STATUSES,
                }
            )
        return items

    def _intention_items(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        try:
            loop = ServiceContainer.get("intention_loop", default=None)
            active = list(getattr(loop, "_active_intentions", {}).values()) if loop is not None else []
            completed = list(getattr(loop, "_completed_intentions", []))[:10] if loop is not None else []
        except Exception:
            active = []
            completed = []
        for intention in list(active) + list(completed):
            status = self._coerce_status(
                getattr(getattr(intention, "status", None), "value", getattr(intention, "status", "")),
                default=GoalStatus.QUEUED.value,
            )
            items.append(
                {
                    "id": f"intention:{getattr(intention, 'id', '')}",
                    "name": str(getattr(intention, "intention", "Intention") or "Intention"),
                    "objective": str(getattr(intention, "expected_outcome", "") or getattr(intention, "intention", "")),
                    "status": status,
                    "horizon": GoalHorizon.SHORT_TERM.value,
                    "source": "intention_loop",
                    "priority": 0.7,
                    "progress": 1.0 if status == GoalStatus.COMPLETED.value else (0.5 if status == GoalStatus.IN_PROGRESS.value else 0.15),
                    "quick_win": False,
                    "attention_policy": "sustained",
                    "steps_done": len(getattr(intention, "actions_taken", []) or []),
                    "steps_total": len(getattr(intention, "plan", []) or []) if getattr(intention, "plan", None) else 0,
                    "success_criteria": str(getattr(intention, "expected_outcome", "") or ""),
                    "summary": str(getattr(intention, "observation", "") or getattr(intention, "actual_outcome", "") or ""),
                    "error": "",
                    "required_tools": [],
                    "required_skills": [],
                    "evidence": [str(getattr(intention, "observation", "") or "")] if getattr(intention, "observation", None) else [],
                    "metadata": {"drive": str(getattr(intention, "drive", "") or "")},
                    "project_id": "",
                    "parent_goal_id": "",
                    "plan_id": "",
                    "task_id": "",
                    "intention_id": str(getattr(intention, "id", "") or ""),
                    "commitment_id": "",
                    "created_at": float(getattr(intention, "intended_at", 0.0) or 0.0),
                    "updated_at": float(getattr(intention, "completed_at", 0.0) or getattr(intention, "intended_at", 0.0) or 0.0),
                    "started_at": float(getattr(intention, "intended_at", 0.0) or 0.0),
                    "completed_at": float(getattr(intention, "completed_at", 0.0) or 0.0) if status == GoalStatus.COMPLETED.value else None,
                    "last_progress_at": float(getattr(intention, "completed_at", 0.0) or getattr(intention, "intended_at", 0.0) or 0.0),
                    "display_status": status.replace("_", " ").title(),
                    "display_horizon": "Short Term",
                    "is_terminal": status in TERMINAL_GOAL_STATUSES,
                }
            )
        return items

    @staticmethod
    def _normalize_tokens(text: str) -> List[str]:
        stopwords = {
            "a", "an", "and", "are", "for", "i", "in", "is", "it", "of", "on",
            "or", "the", "to", "we", "why",
        }
        tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
        return [token for token in tokens if token not in stopwords]

    def _shows_goal_progress(self, objective: str, messages: List[str]) -> bool:
        objective_tokens = self._normalize_tokens(objective)
        if not objective_tokens:
            return False

        objective_phrase = " ".join(objective_tokens)
        objective_bigrams = {
            tuple(objective_tokens[i : i + 2])
            for i in range(max(0, len(objective_tokens) - 1))
        }

        for message in messages:
            message_tokens = self._normalize_tokens(message)
            if not message_tokens:
                continue

            message_token_set = set(message_tokens)
            coverage = len(set(objective_tokens) & message_token_set) / max(len(set(objective_tokens)), 1)
            if len(objective_tokens) <= 2 and coverage == 1.0:
                return True

            normalized_message = " ".join(message_tokens)
            if objective_phrase and objective_phrase in normalized_message:
                return True

            if objective_bigrams:
                message_bigrams = {
                    tuple(message_tokens[i : i + 2])
                    for i in range(max(0, len(message_tokens) - 1))
                }
                bigram_coverage = len(objective_bigrams & message_bigrams) / max(len(objective_bigrams), 1)
                if coverage >= 0.75 and bigram_coverage >= 0.5:
                    return True

        return False
