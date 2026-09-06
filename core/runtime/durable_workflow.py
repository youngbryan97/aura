"""DurableWorkflowEngine — resumable multi-step task runtime.

Each long task runs as a `Workflow` with an immutable plan (a list of
`WorkflowStep`s). After every committed step, the engine writes a
durable checkpoint via the canonical AtomicWriter so the workflow can
be resumed on the next process from the exact step it failed/paused at,
without re-running already-committed side effects.

Key contracts:

* Each step has a unique ``step_id`` and an ``apply`` callable.
* ``apply`` receives the prior outputs and must be idempotent — the
  engine guarantees it is invoked at-most-once per ``step_id`` across
  resumes by checking the checkpoint first.
* A step may flag ``human_approval=True``; the engine pauses there and
  records a checkpoint with status PAUSED_FOR_APPROVAL. It stays paused
  until someone calls ``approve()`` or ``deny()`` — a *recorded decision*,
  not merely a re-entry into ``resume()``.
* If a step fails and ``rollback`` is provided, the engine runs rollback
  and marks the workflow FAILED. A ``RetryPolicy`` on the step is honoured
  first; rollback happens only once the retries are exhausted.

Two defects fixed here, both of the same shape — a documented capability with
no code behind it:

**The approval pause was a one-way door.** This docstring used to say a paused
workflow ran "until ``resume()`` is called", and ``resume()`` simply called
``run()`` again — which reached the same ``human_approval`` step and paused
again, forever. There was nowhere to record that a human had said yes. Every
test asserted only that it *paused* and that the pause was *discoverable*;
none asserted it could ever proceed, so a workflow that asked permission could
never get it. Approvals are now durable state on the checkpoint, and a denial
cancels rather than deadlocks.

**Every checkpoint fsynced on the event loop.** ``run()`` is async and called
the synchronous ``store.save()`` once per step. That is the exact pattern that
froze the live loop for 20 minutes; the async-write ratchet missed it because
the blocking write sat one call level down inside ``WorkflowStore``. The engine
now uses the async lane, and the ratchet has been taught to follow same-file
method calls.

Beyond the fixes, the engine keeps a **revision history** rather than only the
latest state, which is what makes replay and forking possible: you can load the
workflow as it stood at revision 3 and branch a new run from there without
disturbing the original.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import read_json_envelope
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.DurableWorkflow")

_WORKFLOW_STEP_ERRORS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

#: Reserved key in ``outputs`` carrying approval payloads, so a step that runs
#: only because a human said yes can see *what* they said.
APPROVALS_KEY = "__approvals__"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED_FOR_APPROVAL = "paused_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


def _became(checkpoint: "WorkflowCheckpoint", wanted: WorkflowStatus) -> Any:
    """Move a checkpoint's status through the declared table.

    Returns the change rather than a boolean, so a caller can tell a move that
    is not legal from one that lost a race — they need opposite responses.
    Applies the new status only when the table allowed it.
    """
    from core.runtime.what_a_status_may_become import the_workflow_statuses

    was = checkpoint.status
    change = the_workflow_statuses().change(was, was, wanted)
    if change.applied:
        checkpoint.status = wanted
    else:
        logger.warning(
            "workflow %s stayed %s: %s",
            getattr(checkpoint, "workflow_id", "?"), was, change.why,
        )
    return change


@dataclass(frozen=True)
class RetryPolicy:
    """How many times to re-attempt a step, and how long to wait between.

    Defaults to a single attempt: retrying is a decision with side effects
    (``apply`` runs again), so it is opt-in per step rather than ambient.
    """

    max_attempts: int = 1
    backoff_seconds: float = 0.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1; backoff may not shrink")

    def delay_before(self, attempt: int) -> float:
        """Seconds to wait before ``attempt`` (1-based). Zero before the first."""
        if attempt <= 1 or self.backoff_seconds == 0:
            return 0.0
        return self.backoff_seconds * (self.multiplier ** (attempt - 2))


@dataclass
class WorkflowStep:
    step_id: str
    name: str
    apply: Callable[[dict[str, Any]], Any | Awaitable[Any]]
    rollback: Callable[[dict[str, Any]], None | Awaitable[None]] | None = None
    human_approval: bool = False
    receipt_id: str | None = None
    retry: RetryPolicy | None = None


@dataclass(frozen=True)
class ApprovalDecision:
    """A human's recorded answer to a step that asked permission.

    Durable, attributed, and timestamped. An approval that lives only in a
    caller's memory is indistinguishable from one that never happened, which is
    how the pause became permanent.
    """

    step_id: str
    granted: bool
    approver: str
    note: str = ""
    value: Any = None
    decided_at: float = field(default_factory=time.time)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowCheckpoint:
    workflow_id: str
    objective: str
    status: WorkflowStatus
    completed_steps: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    failed_step: str | None = None
    failure_reason: str | None = None
    paused_at_step: str | None = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    #: step_id -> ApprovalDecision payload. The state that makes a pause
    #: escapable.
    approvals: dict[str, Any] = field(default_factory=dict)
    #: Monotonic version, bumped on every save. Identifies a point in history.
    revision: int = 0
    #: Set when this workflow was branched off another.
    forked_from: str | None = None
    forked_at_revision: int | None = None

    def decision_for(self, step_id: str) -> ApprovalDecision | None:
        payload = self.approvals.get(step_id)
        if not payload:
            return None
        return ApprovalDecision(**payload)


#: Statuses that describe work still owed. PAUSED_FOR_APPROVAL counts: the
#: workflow is waiting on a human, not finished, and a restart must not lose it.
_RESUMABLE_STATUSES = frozenset({
    WorkflowStatus.PENDING,
    WorkflowStatus.RUNNING,
    WorkflowStatus.PAUSED_FOR_APPROVAL,
})


class WorkflowStore:
    """Atomic-writer-backed checkpoint store with a revision history."""

    SCHEMA_VERSION = 2

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else (state_root() / "workflows")
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------

    def _latest_path(self, workflow_id: str) -> Path:
        return self.root / f"{workflow_id}.json"

    def _history_dir(self, workflow_id: str) -> Path:
        return self.root / "history" / workflow_id

    def _payload(self, checkpoint: WorkflowCheckpoint) -> dict[str, Any]:
        payload = asdict(checkpoint)
        payload["status"] = checkpoint.status.value
        return payload

    # -- writes ------------------------------------------------------------

    def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Synchronous save. Callers on the event loop want ``async_save``."""
        checkpoint.revision += 1
        payload = self._payload(checkpoint)
        gateway = get_file_write_gateway()
        gateway.write_json(
            self._latest_path(checkpoint.workflow_id),
            payload,
            schema_version=self.SCHEMA_VERSION,
            schema_name="workflow_checkpoint",
            source="runtime.durable_workflow.save",
        )
        history = self._history_dir(checkpoint.workflow_id)
        gateway.ensure_directory(history, source="runtime.durable_workflow.save")
        gateway.write_json(
            history / f"{checkpoint.revision:06d}.json",
            payload,
            schema_version=self.SCHEMA_VERSION,
            schema_name="workflow_checkpoint",
            source="runtime.durable_workflow.save",
        )

    async def async_save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save without an fsync on the calling loop.

        The engine's step loop runs here. A synchronous fsync per step is the
        pattern that wedged the live runtime.
        """
        checkpoint.revision += 1
        payload = self._payload(checkpoint)
        gateway = get_file_write_gateway()
        await gateway.write_json_async(
            self._latest_path(checkpoint.workflow_id),
            payload,
            schema_version=self.SCHEMA_VERSION,
            schema_name="workflow_checkpoint",
            source="runtime.durable_workflow.async_save",
        )
        history = self._history_dir(checkpoint.workflow_id)
        await gateway.ensure_directory_async(
            history, source="runtime.durable_workflow.async_save"
        )
        await gateway.write_json_async(
            history / f"{checkpoint.revision:06d}.json",
            payload,
            schema_version=self.SCHEMA_VERSION,
            schema_name="workflow_checkpoint",
            source="runtime.durable_workflow.async_save",
        )

    # -- reads -------------------------------------------------------------

    @staticmethod
    def _hydrate(payload: dict[str, Any]) -> WorkflowCheckpoint:
        payload = dict(payload)
        payload["status"] = WorkflowStatus(payload.get("status", "pending"))
        # Tolerate checkpoints written by an older schema: unknown-to-us keys
        # are dropped, absent-from-them keys take their defaults.
        known = {f for f in WorkflowCheckpoint.__dataclass_fields__}
        return WorkflowCheckpoint(**{k: v for k, v in payload.items() if k in known})

    def unfinished(self) -> list[WorkflowCheckpoint]:
        """Every workflow that was interrupted rather than completed.

        This is what made resume() unreachable in practice: the store could
        save and load BY ID, but nothing could discover which workflows were
        still owed work — and knowing the id is exactly what a crash destroys.
        Recovery has to start from "what was I doing?", not from a caller
        remembering.

        Corrupt or unreadable checkpoints are skipped rather than aborting the
        scan: one bad file must not hide every other resumable workflow.
        """
        out: list[WorkflowCheckpoint] = []
        try:
            paths = sorted(self.root.glob("*.json"))
        except OSError:
            return out
        for path in paths:
            try:
                env = read_json_envelope(path)
                checkpoint = self._hydrate(env.get("payload") or {})
            except (OSError, ValueError, TypeError, KeyError):
                continue
            if checkpoint.status in _RESUMABLE_STATUSES:
                out.append(checkpoint)
        return out

    def load(self, workflow_id: str) -> WorkflowCheckpoint | None:
        path = self._latest_path(workflow_id)
        if not path.exists():
            return None
        env = read_json_envelope(path)
        return self._hydrate(env.get("payload") or {})

    def history(self, workflow_id: str) -> list[WorkflowCheckpoint]:
        """Every recorded revision, oldest first.

        Keeping only the latest state makes a workflow un-replayable: you can
        see where it ended up but never how, and you cannot branch from a point
        before the decision you now regret.
        """
        directory = self._history_dir(workflow_id)
        if not directory.is_dir():
            return []
        out: list[WorkflowCheckpoint] = []
        try:
            paths = sorted(directory.glob("*.json"))
        except OSError:
            return out
        for path in paths:
            try:
                env = read_json_envelope(path)
                out.append(self._hydrate(env.get("payload") or {}))
            except (OSError, ValueError, TypeError, KeyError):
                continue
        return out

    def load_revision(self, workflow_id: str, revision: int) -> WorkflowCheckpoint | None:
        path = self._history_dir(workflow_id) / f"{revision:06d}.json"
        if not path.exists():
            return None
        env = read_json_envelope(path)
        return self._hydrate(env.get("payload") or {})


class DurableWorkflowEngine:
    def __init__(self, *, store: WorkflowStore | None = None):
        self.store = store or WorkflowStore()

    # -- execution ---------------------------------------------------------

    async def _run_step(
        self, step: WorkflowStep, outputs: dict[str, Any]
    ) -> Any:
        """Invoke ``apply``, honouring the step's retry policy."""
        policy = step.retry or RetryPolicy()
        last_exc: BaseException | None = None
        for attempt in range(1, policy.max_attempts + 1):
            delay = policy.delay_before(attempt)
            if delay:
                await asyncio.sleep(delay)
            try:
                result = step.apply(outputs)
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except _WORKFLOW_STEP_ERRORS as exc:
                last_exc = exc
                if attempt < policy.max_attempts:
                    logger.warning(
                        "Workflow step %s failed on attempt %d/%d: %s",
                        step.step_id, attempt, policy.max_attempts, exc,
                    )
                    continue
                raise
        raise last_exc  # pragma: no cover - loop always returns or raises

    async def run(
        self,
        objective: str,
        steps: list[WorkflowStep],
        *,
        workflow_id: str | None = None,
    ) -> WorkflowCheckpoint:
        workflow_id = workflow_id or f"wf-{uuid.uuid4()}"
        checkpoint = self.store.load(workflow_id)
        if checkpoint is None:
            checkpoint = WorkflowCheckpoint(
                workflow_id=workflow_id,
                objective=objective,
                status=WorkflowStatus.RUNNING,
            )
        else:
            # A resume that finds the workflow already finished must not put
            # it back to running. Nothing declared which moves were legal, so
            # nothing could refuse this one.
            resumed = _became(checkpoint, WorkflowStatus.RUNNING)
            if not resumed:
                logger.info(
                    "workflow %s was not resumed: %s", workflow_id, resumed.why
                )
                return checkpoint
        await self.store.async_save(checkpoint)

        for step in steps:
            if step.step_id in checkpoint.completed_steps:
                continue  # Idempotent: skip already-committed

            if step.human_approval:
                decision = checkpoint.decision_for(step.step_id)
                if decision is None:
                    _became(checkpoint, WorkflowStatus.PAUSED_FOR_APPROVAL)
                    checkpoint.paused_at_step = step.step_id
                    checkpoint.updated_at = time.time()
                    await self.store.async_save(checkpoint)
                    return checkpoint
                if not decision.granted:
                    # A refusal is an answer. Cancelling is the honest outcome;
                    # looping back to ask again would be pestering, and leaving
                    # it paused would be the deadlock this replaced.
                    _became(checkpoint, WorkflowStatus.CANCELED)
                    checkpoint.paused_at_step = None
                    checkpoint.failure_reason = (
                        f"{step.step_id} denied by {decision.approver}"
                        + (f": {decision.note}" if decision.note else "")
                    )
                    checkpoint.updated_at = time.time()
                    await self.store.async_save(checkpoint)
                    return checkpoint
                # Granted: make the decision visible to the step that asked.
                approvals = checkpoint.outputs.setdefault(APPROVALS_KEY, {})
                approvals[step.step_id] = decision.to_payload()
                checkpoint.paused_at_step = None

            try:
                result = await self._run_step(step, checkpoint.outputs)
                checkpoint.outputs[step.step_id] = result
                checkpoint.completed_steps.append(step.step_id)
                checkpoint.updated_at = time.time()
                await self.store.async_save(checkpoint)
            except _WORKFLOW_STEP_ERRORS as exc:
                checkpoint.failed_step = step.step_id
                checkpoint.failure_reason = repr(exc)
                _became(checkpoint, WorkflowStatus.FAILED)
                checkpoint.updated_at = time.time()
                await self.store.async_save(checkpoint)
                if step.rollback is not None:
                    try:
                        rb = step.rollback(checkpoint.outputs)
                        if asyncio.iscoroutine(rb):
                            await rb
                    except _WORKFLOW_STEP_ERRORS as rb_exc:
                        logger.error(
                            "Workflow %s rollback for %s failed: %s",
                            workflow_id, step.step_id, rb_exc,
                        )
                return checkpoint

        _became(checkpoint, WorkflowStatus.COMPLETED)
        checkpoint.updated_at = time.time()
        await self.store.async_save(checkpoint)
        return checkpoint

    async def resume(
        self,
        workflow_id: str,
        steps: list[WorkflowStep],
    ) -> WorkflowCheckpoint:
        return await self.run(objective=workflow_id, steps=steps, workflow_id=workflow_id)

    # -- approvals ---------------------------------------------------------

    async def _record_decision(
        self, workflow_id: str, decision: ApprovalDecision
    ) -> WorkflowCheckpoint:
        checkpoint = self.store.load(workflow_id)
        if checkpoint is None:
            raise LookupError(f"no such workflow: {workflow_id}")
        checkpoint.approvals[decision.step_id] = decision.to_payload()
        checkpoint.updated_at = time.time()
        await self.store.async_save(checkpoint)
        return checkpoint

    async def approve(
        self,
        workflow_id: str,
        step_id: str,
        *,
        approver: str,
        value: Any = None,
        note: str = "",
    ) -> WorkflowCheckpoint:
        """Record a human's yes. The next ``resume()`` runs the step.

        ``value`` is carried to the step through ``outputs[APPROVALS_KEY]`` —
        an approval that can only say yes cannot say *yes, but with this
        budget*, and that is most of what approvals are for.
        """
        return await self._record_decision(
            workflow_id,
            ApprovalDecision(
                step_id=step_id, granted=True, approver=approver,
                note=note, value=value,
            ),
        )

    async def deny(
        self, workflow_id: str, step_id: str, *, approver: str, note: str = ""
    ) -> WorkflowCheckpoint:
        """Record a human's no. The next ``resume()`` cancels the workflow."""
        return await self._record_decision(
            workflow_id,
            ApprovalDecision(
                step_id=step_id, granted=False, approver=approver, note=note
            ),
        )

    def pending_approvals(self) -> list[tuple[str, str]]:
        """``(workflow_id, step_id)`` for everything waiting on a human.

        Without this, a paused workflow is only findable by someone who already
        knows it exists — which is the same reason ``unfinished()`` had to be
        written.
        """
        return [
            (c.workflow_id, c.paused_at_step)
            for c in self.store.unfinished()
            if c.status is WorkflowStatus.PAUSED_FOR_APPROVAL and c.paused_at_step
        ]

    # -- time travel -------------------------------------------------------

    async def fork(
        self,
        workflow_id: str,
        *,
        at_revision: int,
        new_workflow_id: str | None = None,
    ) -> WorkflowCheckpoint:
        """Branch a new workflow from a historical revision.

        The original is untouched. This is how you retry a run from before the
        decision that ruined it without losing the evidence of what it did —
        the same reason a bisect keeps the bad commit.
        """
        source = self.store.load_revision(workflow_id, at_revision)
        if source is None:
            raise LookupError(
                f"no revision {at_revision} for {workflow_id}; "
                f"have {[c.revision for c in self.store.history(workflow_id)]}"
            )
        forked = WorkflowCheckpoint(
            workflow_id=new_workflow_id or f"{workflow_id}-fork-{uuid.uuid4().hex[:8]}",
            objective=source.objective,
            status=WorkflowStatus.PENDING,
            completed_steps=list(source.completed_steps),
            outputs=dict(source.outputs),
            approvals=dict(source.approvals),
            forked_from=workflow_id,
            forked_at_revision=at_revision,
            revision=0,
        )
        await self.store.async_save(forked)
        return forked
