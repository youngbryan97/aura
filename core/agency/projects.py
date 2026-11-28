"""core/agency/projects.py

Self-Originated Project Ledger
==============================
Long-horizon work that Aura starts on her own initiative, pursues across
sessions, revises in light of new evidence, and either completes or
explicitly abandons. Every transition is receipted; the project's lineage
is reconstructable from the durable JSONL ledger.

Each project records:

  - origin                 (drive that created it)
  - thesis                 (what it's trying to learn / build / change)
  - acceptance_criteria    (concrete, falsifiable)
  - milestones             (ordered list with completion timestamps)
  - revisions              (reasoned scope/criteria changes)
  - related_actions        (action receipt IDs)
  - permission_requests    (Will/Authority decisions blocking forward motion)
  - artifacts              (files, beliefs, tools created or used)
  - reflections            (post-completion lesson, durable in narrative arc)
  - lifecycle              (proposed → approved → active → blocked →
                            completed → abandoned)

The ledger is intentionally evidence-based: a project that claims completion
without artifacts and acceptance-criteria checks is rejected by
``mark_completed()``.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.runtime.lockdep import checked_lock
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Projects")


_PROJECT_DIR = state_root() / "data" / "projects"
_PROJECT_DIR.mkdir(parents=True, exist_ok=True)
_LEDGER_PATH = _PROJECT_DIR / "ledger.jsonl"


class Lifecycle(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class Milestone:
    id: str
    description: str
    completed_at: Optional[float] = None
    artifacts: List[str] = field(default_factory=list)
    #: What sort of act it is, by the part of her that does it (core/agency/her_objectives.py); "" where none was named.
    kind: str = ""
    #: The milestones it waits on, by id.
    after: List[str] = field(default_factory=list)
    #: What the part of her that does it knows it by: an inquiry, a request of hers, a goal of hers.
    key: str = ""
    tries: int = 0
    #: Passes in a row that came to nothing.
    fruitless: int = 0
    #: What each pass came to, as it was seen, the latest last.
    came: List[str] = field(default_factory=list)
    #: Why it was set aside, where it was.
    set_aside: str = ""


@dataclass
class Revision:
    when: float
    reason: str
    diff: Dict[str, Any]


@dataclass
class PermissionRequest:
    when: float
    action: str
    domain: str
    decision: str  # approved | denied | deferred
    receipt_id: Optional[str]
    rationale: str


@dataclass
class Project:
    project_id: str
    origin_drive: str
    thesis: str
    acceptance_criteria: List[str]
    started_at: float = field(default_factory=lambda: time.time())
    lifecycle: Lifecycle = Lifecycle.PROPOSED
    milestones: List[Milestone] = field(default_factory=list)
    revisions: List[Revision] = field(default_factory=list)
    related_actions: List[str] = field(default_factory=list)
    permission_requests: List[PermissionRequest] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    reflections: List[str] = field(default_factory=list)
    last_touched_at: float = field(default_factory=lambda: time.time())
    why: str = ""
    #: What would tell her it is not worth going on with, said when she set out, and what she would do instead.
    not_worth_it_if: str = ""
    otherwise: List[str] = field(default_factory=list)
    #: What she wrote of it as she thought it, before it was written in this shape.
    formed_from: str = ""
    #: What she wrote in this shape that her own words did not bear out, with how far they did.
    left_out: List[Dict[str, Any]] = field(default_factory=list)
    #: Why it is to be thought over before it is gone on with; "" where it is not.
    to_think_over: str = ""

    def is_completable(self) -> bool:
        if not self.acceptance_criteria:
            return False
        return all(m.completed_at is not None or m.set_aside for m in self.milestones) and bool(self.artifacts)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["lifecycle"] = self.lifecycle.value
        return d


class ProjectLedger:
    """Durable, append-only project ledger with per-project state cache.

    Read uses a fresh tail-scan of the JSONL file so the ledger remains
    the source of truth even if the cache is cold. Writes are atomic
    append-then-fsync.
    """

    def __init__(self, path: Path = _LEDGER_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = checked_lock("agency.projects", reentrant=True)
        self._cache: Dict[str, Project] = {}
        self._load()

    # -------- mutation --------

    def propose(
        self,
        *,
        origin_drive: str,
        thesis: str,
        acceptance_criteria: List[str],
        milestones: Optional[List[Any]] = None,
        **about: Any,
    ) -> Project:
        project_id = f"PRJ-{uuid.uuid4().hex[:10]}"
        proj = Project(
            project_id=project_id,
            origin_drive=origin_drive,
            thesis=thesis,
            acceptance_criteria=list(acceptance_criteria),
            milestones=[m if isinstance(m, Milestone) else Milestone(id=f"MS-{i}", description=m) for i, m in enumerate(milestones or [])],
            **about,
        )
        self._record_event(proj, "proposed", {})
        return proj

    def transition(self, project_id: str, lifecycle: Lifecycle, *, reason: str = "") -> Project:
        with self._lock:
            proj = self._cache[project_id]
            proj.lifecycle = lifecycle
            proj.last_touched_at = time.time()
            self._record_event(proj, "transition", {"to": lifecycle.value, "reason": reason})
            return proj

    def revise(self, project_id: str, *, reason: str, diff: Dict[str, Any]) -> Project:
        with self._lock:
            proj = self._cache[project_id]
            proj.revisions.append(Revision(when=time.time(), reason=reason, diff=diff))
            for k, v in diff.items():
                if hasattr(proj, k):
                    setattr(proj, k, v)
            proj.last_touched_at = time.time()
            self._record_event(proj, "revise", {"reason": reason, "diff": diff})
            return proj

    def attach_action(self, project_id: str, action_receipt_id: str) -> None:
        with self._lock:
            proj = self._cache[project_id]
            proj.related_actions.append(action_receipt_id)
            proj.last_touched_at = time.time()
            self._record_event(proj, "attach_action", {"action": action_receipt_id})

    def attach_artifact(self, project_id: str, artifact_id: str) -> None:
        with self._lock:
            proj = self._cache[project_id]
            proj.artifacts.append(artifact_id)
            proj.last_touched_at = time.time()
            self._record_event(proj, "attach_artifact", {"artifact": artifact_id})

    def complete_milestone(self, project_id: str, milestone_id: str, *, artifacts: Optional[List[str]] = None) -> None:
        with self._lock:
            proj = self._cache[project_id]
            for m in proj.milestones:
                if m.id == milestone_id:
                    m.completed_at = time.time()
                    if artifacts:
                        m.artifacts.extend(artifacts)
                        proj.artifacts.extend(artifacts)
                    break
            proj.last_touched_at = time.time()
            self._record_event(proj, "complete_milestone", {"milestone": milestone_id})

    def milestone_changed(self, project_id: str, milestone_id: str, **changes: Any) -> Milestone:
        """A milestone's record changed: a pass at it, what it came to, what does it, or why it was set aside."""
        with self._lock:
            proj = self._cache[project_id]
            milestone = next(m for m in proj.milestones if m.id == milestone_id)
            for name, value in changes.items():
                setattr(milestone, name, value)
            proj.last_touched_at = time.time()
            self._record_event(proj, "milestone", {"milestone": milestone_id, "changed": sorted(changes)})
            return milestone

    def record_permission(self, project_id: str, *, action: str, domain: str, decision: str, receipt_id: Optional[str], rationale: str) -> None:
        with self._lock:
            proj = self._cache[project_id]
            proj.permission_requests.append(PermissionRequest(
                when=time.time(), action=action, domain=domain,
                decision=decision, receipt_id=receipt_id, rationale=rationale,
            ))
            proj.last_touched_at = time.time()
            self._record_event(proj, "permission", {"action": action, "decision": decision, "receipt": receipt_id})

    def mark_completed(self, project_id: str, *, reflection: str) -> Project:
        with self._lock:
            proj = self._cache[project_id]
            if not proj.is_completable():
                raise ValueError(
                    f"Project {project_id} is not completable: missing acceptance criteria, milestones, or artifacts"
                )
            proj.reflections.append(reflection)
            return self.transition(project_id, Lifecycle.COMPLETED, reason=reflection)

    def abandon(self, project_id: str, *, reason: str) -> Project:
        return self.transition(project_id, Lifecycle.ABANDONED, reason=reason)

    # -------- query --------

    def get(self, project_id: str) -> Optional[Project]:
        with self._lock:
            return self._cache.get(project_id)

    def active(self) -> List[Project]:
        with self._lock:
            return [p for p in self._cache.values() if p.lifecycle in (Lifecycle.APPROVED, Lifecycle.ACTIVE)]

    def all(self) -> List[Project]:
        with self._lock:
            return list(self._cache.values())

    # -------- persistence --------

    def _record_event(self, proj: Project, event: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._cache[proj.project_id] = proj
            line = json.dumps({
                "when": time.time(),
                "event": event,
                "project_id": proj.project_id,
                "snapshot": proj.to_dict(),
                "payload": payload,
            }, default=str)
            from core.runtime.file_write_gateway import get_file_write_gateway

            get_file_write_gateway().append_text(
                self.path,
                line + "\n",
                encoding="utf-8",
                source="projects.record_event",
            )

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    snap = rec.get("snapshot") or {}
                    if not snap.get("project_id"):
                        continue
                    self._cache[snap["project_id"]] = self._project_from_snap(snap)
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation('projects', exc)
            logger.warning("project ledger load failed: %s", exc)

    @staticmethod
    def _project_from_snap(snap: Dict[str, Any]) -> Project:
        ms = [Milestone(**m) for m in snap.get("milestones", []) if isinstance(m, dict)]
        revs = [Revision(**r) for r in snap.get("revisions", []) if isinstance(r, dict)]
        perms = [PermissionRequest(**pr) for pr in snap.get("permission_requests", []) if isinstance(pr, dict)]
        return Project(
            project_id=snap["project_id"],
            origin_drive=snap.get("origin_drive", ""),
            thesis=snap.get("thesis", ""),
            acceptance_criteria=list(snap.get("acceptance_criteria", []) or []),
            started_at=float(snap.get("started_at", time.time())),
            lifecycle=Lifecycle(snap.get("lifecycle", "proposed")),
            milestones=ms,
            revisions=revs,
            related_actions=list(snap.get("related_actions", []) or []),
            permission_requests=perms,
            artifacts=list(snap.get("artifacts", []) or []),
            reflections=list(snap.get("reflections", []) or []),
            last_touched_at=float(snap.get("last_touched_at", time.time())),
            why=str(snap.get("why") or ""),
            not_worth_it_if=str(snap.get("not_worth_it_if") or ""),
            otherwise=list(snap.get("otherwise", []) or []),
            formed_from=str(snap.get("formed_from") or ""),
            left_out=list(snap.get("left_out", []) or []),
            to_think_over=str(snap.get("to_think_over") or ""),
        )


_LEDGER: Optional[ProjectLedger] = None


def get_ledger() -> ProjectLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ProjectLedger()
    return _LEDGER


__all__ = [
    "Lifecycle",
    "Milestone",
    "Project",
    "ProjectLedger",
    "get_ledger",
]
