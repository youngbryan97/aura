"""Formal 11-state task lifecycle with transition guarantees.

The legacy ``GoalStatus`` enum exposes 7 statuses without a transition
table, which means callers can move a goal from ``COMPLETED`` back to
``IN_PROGRESS`` and the engine accepts it.  That's fine for the
existing free-form coordinator but cannot anchor long-horizon autonomy:
without an enforced lifecycle, "task complete" has no operational
meaning.

This module adds an explicit lifecycle alongside ``GoalStatus``.  Every
state declares:

    * who owns the next move (the ``owner_role``),
    * what evidence must accompany an exit transition,
    * which states are reachable next (closed transition table),
    * whether the state is terminal (no outgoing transitions),
    * a default rollback path used when an exit transition fails.

The 11 states match the ones requested in the AGI/enterprise gap list:

    proposed -> accepted -> planned -> in_progress -> testing
                                          |             |
                                          v             v
    blocked        waiting_for_user      deferred     completed
       \\_______________|________________/  \\
                       v                     +----> failed
                  (any active)               |
                                             +----> abandoned

The wrapper ``TaskLifecycleManager`` plugs into the existing
``GoalEngine`` so legacy callers continue to work; new code that wants
strict lifecycle enforcement goes through ``transition()``.

A ``migrate_legacy_status_db`` helper rewrites pre-existing rows in
``goal_lifecycle.db`` into the new vocabulary in a single idempotent
pass, recording the prior status in ``metadata.legacy_status``.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


class GoalState(str, Enum):
    """Formal lifecycle states.  String values are stable wire format."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    WAITING_FOR_USER = "waiting_for_user"
    TESTING = "testing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"
    DEFERRED = "deferred"


TERMINAL_STATES: FrozenSet[GoalState] = frozenset(
    {GoalState.COMPLETED, GoalState.FAILED, GoalState.ABANDONED}
)

ACTIVE_STATES: FrozenSet[GoalState] = frozenset(
    {
        GoalState.IN_PROGRESS,
        GoalState.TESTING,
    }
)

IDLE_STATES: FrozenSet[GoalState] = frozenset(
    {
        GoalState.PROPOSED,
        GoalState.ACCEPTED,
        GoalState.PLANNED,
        GoalState.BLOCKED,
        GoalState.WAITING_FOR_USER,
        GoalState.DEFERRED,
    }
)


class IllegalTransitionError(RuntimeError):
    """Raised when a (from, to) state pair is not allowed."""

    def __init__(self, *, from_state: GoalState, to_state: GoalState, reason: str = ""):
        msg = f"illegal transition: {from_state.value} -> {to_state.value}"
        if reason:
            msg = f"{msg} ({reason})"
        super().__init__(msg)
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason


class MissingEvidenceError(RuntimeError):
    """Raised when a transition requires evidence that the caller did not supply."""

    def __init__(self, *, from_state: GoalState, to_state: GoalState, missing: List[str]):
        super().__init__(
            f"transition {from_state.value} -> {to_state.value} requires "
            f"evidence: {', '.join(missing)}"
        )
        self.from_state = from_state
        self.to_state = to_state
        self.missing = list(missing)


@dataclass(frozen=True)
class StatePolicy:
    """Per-state metadata: who owns the move, what's needed to exit, etc."""

    state: GoalState
    owner_role: str
    description: str
    requires_owner: bool = True
    requires_deadline: bool = False
    rollback_to: Optional[GoalState] = None
    # Names of evidence keys that must be present in the transition's
    # ``evidence`` dict to *leave* this state for any reachable target.
    exit_evidence: Tuple[str, ...] = ()


# Per-state policies.  Owners are role names, not user IDs; an autonomy
# layer maps roles to actual operators.
STATE_POLICIES: Dict[GoalState, StatePolicy] = {
    GoalState.PROPOSED: StatePolicy(
        state=GoalState.PROPOSED,
        owner_role="reviewer",
        description="Goal exists in the ledger but has not been accepted.",
        requires_owner=True,
        rollback_to=GoalState.PROPOSED,
        exit_evidence=("acceptance_reason",),
    ),
    GoalState.ACCEPTED: StatePolicy(
        state=GoalState.ACCEPTED,
        owner_role="planner",
        description="Goal accepted; planning has not yet begun.",
        rollback_to=GoalState.PROPOSED,
        exit_evidence=("plan_id",),
    ),
    GoalState.PLANNED: StatePolicy(
        state=GoalState.PLANNED,
        owner_role="executor",
        description="Plan exists; execution has not begun.",
        requires_deadline=True,
        rollback_to=GoalState.ACCEPTED,
    ),
    GoalState.IN_PROGRESS: StatePolicy(
        state=GoalState.IN_PROGRESS,
        owner_role="executor",
        description="Goal is being actively executed.",
        requires_deadline=True,
        rollback_to=GoalState.PLANNED,
        exit_evidence=("progress_summary",),
    ),
    GoalState.BLOCKED: StatePolicy(
        state=GoalState.BLOCKED,
        owner_role="executor",
        description="Execution paused on an internal dependency.",
        rollback_to=GoalState.IN_PROGRESS,
        exit_evidence=("blocker",),
    ),
    GoalState.WAITING_FOR_USER: StatePolicy(
        state=GoalState.WAITING_FOR_USER,
        owner_role="user",
        description="Execution paused awaiting an explicit user response.",
        rollback_to=GoalState.IN_PROGRESS,
        exit_evidence=("user_question",),
    ),
    GoalState.TESTING: StatePolicy(
        state=GoalState.TESTING,
        owner_role="verifier",
        description="Implementation done; outcome under verification.",
        requires_deadline=True,
        rollback_to=GoalState.IN_PROGRESS,
        exit_evidence=("verification_result",),
    ),
    GoalState.COMPLETED: StatePolicy(
        state=GoalState.COMPLETED,
        owner_role="archivist",
        description="Terminal success.",
        requires_owner=False,
        requires_deadline=False,
    ),
    GoalState.FAILED: StatePolicy(
        state=GoalState.FAILED,
        owner_role="archivist",
        description="Terminal failure.",
        requires_owner=False,
        requires_deadline=False,
    ),
    GoalState.ABANDONED: StatePolicy(
        state=GoalState.ABANDONED,
        owner_role="archivist",
        description="Terminal cancellation by owner or governance.",
        requires_owner=False,
        requires_deadline=False,
    ),
    GoalState.DEFERRED: StatePolicy(
        state=GoalState.DEFERRED,
        owner_role="planner",
        description="Goal deliberately parked; may be resumed later.",
        requires_owner=True,
        rollback_to=GoalState.ACCEPTED,
        exit_evidence=("resume_reason",),
    ),
}


# Closed transition table.  Each entry is the set of states reachable
# from a given state.  Terminal states have empty out-sets.  Any state
# may also self-transition (e.g. a progress update that does not change
# state); self-transitions are *not* listed here and are handled
# separately by ``allow_self_transition``.
ALLOWED_TRANSITIONS: Dict[GoalState, FrozenSet[GoalState]] = {
    GoalState.PROPOSED: frozenset(
        {GoalState.ACCEPTED, GoalState.ABANDONED}
    ),
    GoalState.ACCEPTED: frozenset(
        {GoalState.PLANNED, GoalState.DEFERRED, GoalState.ABANDONED}
    ),
    GoalState.PLANNED: frozenset(
        {GoalState.IN_PROGRESS, GoalState.DEFERRED, GoalState.ABANDONED}
    ),
    GoalState.IN_PROGRESS: frozenset(
        {
            GoalState.BLOCKED,
            GoalState.WAITING_FOR_USER,
            GoalState.TESTING,
            GoalState.DEFERRED,
            GoalState.FAILED,
            GoalState.ABANDONED,
        }
    ),
    GoalState.BLOCKED: frozenset(
        {GoalState.IN_PROGRESS, GoalState.WAITING_FOR_USER, GoalState.ABANDONED, GoalState.FAILED}
    ),
    GoalState.WAITING_FOR_USER: frozenset(
        {GoalState.IN_PROGRESS, GoalState.BLOCKED, GoalState.ABANDONED, GoalState.FAILED}
    ),
    GoalState.TESTING: frozenset(
        {GoalState.IN_PROGRESS, GoalState.COMPLETED, GoalState.FAILED}
    ),
    GoalState.DEFERRED: frozenset(
        {GoalState.PLANNED, GoalState.IN_PROGRESS, GoalState.ABANDONED}
    ),
    GoalState.COMPLETED: frozenset(),
    GoalState.FAILED: frozenset(),
    GoalState.ABANDONED: frozenset(),
}


def reachable_states(state: GoalState) -> FrozenSet[GoalState]:
    """Return the set of states reachable from ``state`` in one transition."""
    return ALLOWED_TRANSITIONS[state]


def is_terminal(state: GoalState) -> bool:
    return state in TERMINAL_STATES


def is_active(state: GoalState) -> bool:
    return state in ACTIVE_STATES


def is_idle(state: GoalState) -> bool:
    return state in IDLE_STATES


def coerce_state(value: Any) -> GoalState:
    """Best-effort coercion of legacy strings or enum members to ``GoalState``.

    Maps the historical ``GoalStatus`` vocabulary onto the new lifecycle
    so that records written before the migration are still routable.
    """
    if isinstance(value, GoalState):
        return value
    text = str(value or "").strip().lower().replace("-", "_")
    direct = {state.value: state for state in GoalState}
    if text in direct:
        return direct[text]
    legacy_map = {
        # historical GoalStatus values
        "queued": GoalState.ACCEPTED,
        "paused": GoalState.DEFERRED,
        "active": GoalState.IN_PROGRESS,
        "running": GoalState.IN_PROGRESS,
        "in_progress": GoalState.IN_PROGRESS,
        "completed": GoalState.COMPLETED,
        "complete": GoalState.COMPLETED,
        "succeeded": GoalState.COMPLETED,
        "failed": GoalState.FAILED,
        "broken": GoalState.FAILED,
        "abandoned": GoalState.ABANDONED,
        "rejected": GoalState.ABANDONED,
        # other observed strings
        "blocked": GoalState.BLOCKED,
        "waiting_for_user": GoalState.WAITING_FOR_USER,
        "testing": GoalState.TESTING,
        "deferred": GoalState.DEFERRED,
        "proposed": GoalState.PROPOSED,
        "accepted": GoalState.ACCEPTED,
        "planned": GoalState.PLANNED,
        "intended": GoalState.ACCEPTED,
        "approved": GoalState.ACCEPTED,
        "pending": GoalState.PROPOSED,
    }
    if text in legacy_map:
        return legacy_map[text]
    raise ValueError(f"cannot coerce {value!r} to GoalState")


@dataclass
class TransitionRequest:
    """Concrete attempt to move from one state to another.

    Carries the evidence and operational metadata the lifecycle requires
    to record a forensic trail of why the move happened.
    """

    goal_id: str
    from_state: GoalState
    to_state: GoalState
    actor: str
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[float] = None
    requested_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionResult:
    """Outcome of a successful transition."""

    goal_id: str
    from_state: GoalState
    to_state: GoalState
    actor: str
    reason: str
    occurred_at: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    rollback_to: Optional[GoalState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "from": self.from_state.value,
            "to": self.to_state.value,
            "actor": self.actor,
            "reason": self.reason,
            "occurred_at": self.occurred_at,
            "evidence": dict(self.evidence),
            "rollback_to": self.rollback_to.value if self.rollback_to else None,
            "metadata": dict(self.metadata),
        }


def validate_transition(req: TransitionRequest) -> None:
    """Raise on illegal/incomplete transitions; return None on success."""
    if req.from_state == req.to_state:
        return  # self-transition is implicitly allowed
    allowed = ALLOWED_TRANSITIONS[req.from_state]
    if req.to_state not in allowed:
        raise IllegalTransitionError(
            from_state=req.from_state,
            to_state=req.to_state,
            reason=(
                "terminal state has no outgoing transitions"
                if is_terminal(req.from_state)
                else f"reachable: {sorted(s.value for s in allowed)}"
            ),
        )
    policy = STATE_POLICIES[req.from_state]
    missing: List[str] = [k for k in policy.exit_evidence if k not in req.evidence]
    if missing:
        raise MissingEvidenceError(
            from_state=req.from_state,
            to_state=req.to_state,
            missing=missing,
        )
    target_policy = STATE_POLICIES[req.to_state]
    if target_policy.requires_owner and not req.actor:
        raise IllegalTransitionError(
            from_state=req.from_state,
            to_state=req.to_state,
            reason=f"target state {req.to_state.value} requires an actor",
        )
    if target_policy.requires_deadline and req.deadline is None:
        raise IllegalTransitionError(
            from_state=req.from_state,
            to_state=req.to_state,
            reason=f"target state {req.to_state.value} requires a deadline",
        )


def apply_transition(req: TransitionRequest) -> TransitionResult:
    """Validate and produce a ``TransitionResult``.

    Persistence is the caller's responsibility (so the lifecycle module
    stays storage-agnostic).  Use ``TaskLifecycleManager.transition`` to
    persist into the existing ``goal_lifecycle.db``.
    """
    validate_transition(req)
    rollback = STATE_POLICIES[req.to_state].rollback_to
    return TransitionResult(
        goal_id=req.goal_id,
        from_state=req.from_state,
        to_state=req.to_state,
        actor=req.actor,
        reason=req.reason,
        occurred_at=req.requested_at,
        evidence=dict(req.evidence),
        rollback_to=rollback,
        metadata=dict(req.metadata),
    )


# ---------------------------------------------------------------------------
# DB-bound manager + migration
# ---------------------------------------------------------------------------
class TaskLifecycleManager:
    """SQLite-backed lifecycle wrapper around the existing goals table.

    * ``get_state(goal_id)`` reads the live state from the canonical
      ``goals.status`` column, coercing legacy values.
    * ``transition(req)`` validates the move, writes the new state, and
      persists the transition into a sibling ``goal_transitions`` table
      so the full trajectory is auditable.
    * ``history(goal_id)`` returns the recorded transitions in order.
    """

    TRANSITIONS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS goal_transitions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        goal_id TEXT NOT NULL,
        from_state TEXT NOT NULL,
        to_state TEXT NOT NULL,
        actor TEXT NOT NULL,
        reason TEXT NOT NULL DEFAULT '',
        occurred_at REAL NOT NULL,
        evidence_json TEXT NOT NULL DEFAULT '{}',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        rollback_to TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_transitions_goal ON goal_transitions(goal_id);
    CREATE INDEX IF NOT EXISTS idx_transitions_time ON goal_transitions(occurred_at);
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            # The goals table is owned by GoalEngine; mirror just the
            # subset we need for cases where this manager is used
            # standalone (e.g. unit tests).
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL DEFAULT '',
                    objective TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    horizon TEXT NOT NULL DEFAULT 'short_term',
                    source TEXT NOT NULL DEFAULT 'lifecycle',
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
                """
            )
            conn.executescript(self.TRANSITIONS_SCHEMA)

    # ------------------------------------------------------------------
    # standalone create (test/standalone usage)
    # ------------------------------------------------------------------
    def create(
        self,
        *,
        goal_id: str,
        name: str = "",
        objective: str = "",
        state: GoalState = GoalState.PROPOSED,
        actor: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if state in TERMINAL_STATES:
            raise IllegalTransitionError(
                from_state=GoalState.PROPOSED,
                to_state=state,
                reason="cannot create a goal in a terminal state",
            )
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO goals(
                    id, name, objective, status, source, created_at, updated_at,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING;
                """,
                (
                    goal_id,
                    name or objective or goal_id,
                    objective,
                    state.value,
                    actor or "lifecycle",
                    now,
                    now,
                    json.dumps(metadata or {}, default=str),
                ),
            )

    # ------------------------------------------------------------------
    # state read / write
    # ------------------------------------------------------------------
    def get_state(self, goal_id: str) -> Optional[GoalState]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM goals WHERE id = ?", (goal_id,)
            ).fetchone()
        if row is None:
            return None
        return coerce_state(row["status"])

    def transition(self, req: TransitionRequest) -> TransitionResult:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM goals WHERE id = ?", (req.goal_id,)
            ).fetchone()
            if row is None:
                raise IllegalTransitionError(
                    from_state=req.from_state,
                    to_state=req.to_state,
                    reason=f"goal {req.goal_id!r} does not exist",
                )
            persisted = coerce_state(row["status"])
            if persisted != req.from_state:
                raise IllegalTransitionError(
                    from_state=req.from_state,
                    to_state=req.to_state,
                    reason=(
                        f"persisted state is {persisted.value!r}, not "
                        f"{req.from_state.value!r}"
                    ),
                )
            result = apply_transition(req)
            now = result.occurred_at
            conn.execute(
                """
                UPDATE goals SET
                    status = ?,
                    updated_at = ?,
                    last_progress_at = ?,
                    started_at = COALESCE(started_at, ?),
                    completed_at = ?
                WHERE id = ?;
                """,
                (
                    result.to_state.value,
                    now,
                    now,
                    now if result.to_state in ACTIVE_STATES else None,
                    now if is_terminal(result.to_state) else None,
                    req.goal_id,
                ),
            )
            conn.execute(
                """
                INSERT INTO goal_transitions(
                    goal_id, from_state, to_state, actor, reason,
                    occurred_at, evidence_json, metadata_json, rollback_to
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    req.goal_id,
                    result.from_state.value,
                    result.to_state.value,
                    result.actor,
                    result.reason,
                    result.occurred_at,
                    json.dumps(result.evidence, default=str),
                    json.dumps(result.metadata, default=str),
                    result.rollback_to.value if result.rollback_to else None,
                ),
            )
        return result

    def history(self, goal_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM goal_transitions
                WHERE goal_id = ?
                ORDER BY occurred_at ASC, id ASC;
                """,
                (goal_id,),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "goal_id": row["goal_id"],
                    "from": row["from_state"],
                    "to": row["to_state"],
                    "actor": row["actor"],
                    "reason": row["reason"],
                    "occurred_at": float(row["occurred_at"]),
                    "evidence": json.loads(row["evidence_json"] or "{}"),
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "rollback_to": row["rollback_to"],
                }
            )
        return out


def migrate_legacy_status_db(db_path: Path) -> Dict[str, int]:
    """Idempotent migration of legacy ``goals.status`` values.

    For every row whose ``status`` is not already in ``GoalState``,
    map it through ``coerce_state`` and rewrite the column.  The prior
    status is captured in ``metadata_json.legacy_status`` so we can
    audit the rewrite later without losing the original signal.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return {"scanned": 0, "rewritten": 0, "skipped": 0, "unrecognized": 0}

    valid = {state.value for state in GoalState}
    stats = {"scanned": 0, "rewritten": 0, "skipped": 0, "unrecognized": 0}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, status, metadata_json FROM goals"
        ).fetchall()
        for row in rows:
            stats["scanned"] += 1
            current = str(row["status"] or "").strip().lower()
            if current in valid:
                stats["skipped"] += 1
                continue
            try:
                new_state = coerce_state(current)
            except ValueError:
                stats["unrecognized"] += 1
                continue
            try:
                meta = json.loads(row["metadata_json"] or "{}")
                if not isinstance(meta, dict):
                    meta = {}
            except json.JSONDecodeError:
                meta = {}
            meta.setdefault("legacy_status", current)
            meta["lifecycle_migrated_at"] = time.time()
            conn.execute(
                "UPDATE goals SET status = ?, metadata_json = ? WHERE id = ?;",
                (new_state.value, json.dumps(meta, default=str), row["id"]),
            )
            stats["rewritten"] += 1
        conn.commit()
    finally:
        conn.close()
    return stats
