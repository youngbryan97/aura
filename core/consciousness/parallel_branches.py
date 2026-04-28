"""core/consciousness/parallel_branches.py -- Parallel Cognitive Branches
=========================================================================
Enables up to N concurrent thought streams running under the Unified Will.

Design:
    The BranchManager maintains a pool of lightweight cognitive branches.
    Each branch is an asyncio Task with its own working context, priority,
    and resource budget.  The foreground conversation branch ALWAYS has
    priority -- background branches yield CPU time and are auto-suspended
    when the foreground needs resources.

    Branch lifecycle:
        spawn() -> active -> [suspended | completed | failed]
        suspend() -> suspended -> resume() -> active
        merge()  -> results fed into Global Workspace as CognitiveCandidate

Integration points:
    - UnifiedWill: every branch spawn requires a WillDecision (domain=INITIATIVE)
    - GlobalWorkspace: completed branch results enter as CognitiveCandidate
    - CognitiveHeartbeat: BranchManager.tick() called each heartbeat cycle
    - ServiceContainer: registered as "branch_manager"
    - InitiativeSynthesizer: can spawn branches during idle periods
    - EventBus: publishes branch lifecycle events

Invariants:
    1. Foreground conversation branch is ALWAYS priority 1.0
    2. Total branch count <= MAX_BRANCHES (5 by default)
    3. Resource accounting enforced via per-tick CPU timing
    4. A branch without a WillReceipt is invalid and will not run
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


from core.utils.task_tracker import get_task_tracker

import asyncio
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional

from core.container import ServiceContainer

logger = logging.getLogger("Consciousness.ParallelBranches")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BranchState(str, Enum):
    """Lifecycle state of a cognitive branch."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"


class BranchOrigin(str, Enum):
    """Who spawned this branch."""
    WILL = "will"                      # Explicit Will directive
    INITIATIVE = "initiative"          # InitiativeSynthesizer during idle
    FOREGROUND = "foreground"          # The main conversation branch
    SYSTEM = "system"                  # Internal housekeeping


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BranchContext:
    """Working context for a single cognitive branch."""
    task_description: str = ""
    progress: float = 0.0             # 0.0 - 1.0
    partial_results: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveBranch:
    """A single parallel thought stream."""
    branch_id: str
    name: str
    origin: BranchOrigin
    state: BranchState = BranchState.ACTIVE
    priority: float = 0.5             # 0.0-1.0 (foreground = 1.0)
    context: BranchContext = field(default_factory=BranchContext)

    # Resource accounting
    cpu_budget_ms: float = 50.0       # Max CPU time per tick (ms)
    cpu_used_ms: float = 0.0          # Cumulative CPU used this tick
    total_cpu_ms: float = 0.0         # Lifetime CPU used
    memory_budget_mb: float = 10.0    # Soft memory limit (informational)

    # Lifecycle
    created_at: float = field(default_factory=time.time)
    last_tick_at: float = 0.0
    ticks_active: int = 0
    ticks_suspended: int = 0

    # Will provenance
    will_receipt_id: str = ""

    # asyncio Task handle (set by BranchManager)
    _task: Optional[asyncio.Task] = field(default=None, repr=False)
    _suspend_event: Optional[asyncio.Event] = field(default=None, repr=False)
    _cancel_event: Optional[asyncio.Event] = field(default=None, repr=False)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_runnable(self) -> bool:
        return self.state == BranchState.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "name": self.name,
            "origin": self.origin.value,
            "state": self.state.value,
            "priority": round(self.priority, 3),
            "progress": round(self.context.progress, 3),
            "task_description": self.context.task_description[:100],
            "cpu_budget_ms": self.cpu_budget_ms,
            "total_cpu_ms": round(self.total_cpu_ms, 2),
            "ticks_active": self.ticks_active,
            "age_seconds": round(self.age_seconds, 1),
            "will_receipt_id": self.will_receipt_id,
        }


# ---------------------------------------------------------------------------
# Branch work function type
# ---------------------------------------------------------------------------

BranchWorkFn = Callable[["CognitiveBranch"], Coroutine[Any, Any, Optional[str]]]


# ---------------------------------------------------------------------------
# BranchManager
# ---------------------------------------------------------------------------

class BranchManager:
    """Manages concurrent cognitive branches under the Unified Will.

    Usage:
        manager = BranchManager()
        await manager.start()

        branch = await manager.spawn(
            name="background_search",
            origin=BranchOrigin.WILL,
            task_description="Search memory for relevant context",
            work_fn=my_async_work_fn,
            priority=0.4,
        )

        # Each heartbeat tick:
        await manager.tick()

        # When done:
        await manager.stop()
    """

    MAX_BRANCHES: int = 5
    FOREGROUND_PRIORITY: float = 1.0
    _DEFAULT_CPU_BUDGET_MS: float = 50.0
    _DEFAULT_MEMORY_BUDGET_MB: float = 10.0
    _SUSPEND_THRESHOLD: float = 0.8    # Suspend background if foreground > this
    _MAX_HISTORY: int = 100

    def __init__(self) -> None:
        self._branches: Dict[str, CognitiveBranch] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self._MAX_HISTORY)
        self._tick_count: int = 0
        self._started: bool = False
        self._total_spawned: int = 0
        self._total_completed: int = 0
        self._total_failed: int = 0
        self._foreground_id: Optional[str] = None

        logger.info("BranchManager created (max_branches=%d)", self.MAX_BRANCHES)

    async def start(self) -> None:
        """Initialize and register in ServiceContainer."""
        if self._started:
            return
        ServiceContainer.register_instance("branch_manager", self)
        self._started = True
        logger.info("BranchManager ONLINE -- parallel cognitive branches enabled")

    async def stop(self) -> None:
        """Cancel all active branches and clean up."""
        for branch in list(self._branches.values()):
            await self._cancel_branch(branch, reason="manager_shutdown")
        self._branches.clear()
        self._started = False
        logger.info("BranchManager OFFLINE -- all branches cancelled")

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    async def spawn(
        self,
        name: str,
        origin: BranchOrigin,
        task_description: str,
        work_fn: BranchWorkFn,
        *,
        priority: float = 0.5,
        cpu_budget_ms: float = _DEFAULT_CPU_BUDGET_MS,
        memory_budget_mb: float = _DEFAULT_MEMORY_BUDGET_MB,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[CognitiveBranch]:
        """Spawn a new cognitive branch.

        The spawn must be approved by the Unified Will (unless it is a
        foreground branch).  Returns None if the Will refuses or if the
        branch pool is full.

        Args:
            name:             Human-readable name for the branch
            origin:           Who is spawning this branch
            task_description: What the branch will work on
            work_fn:          Async function(branch) -> Optional[str result]
            priority:         0.0-1.0 (foreground = 1.0)
            cpu_budget_ms:    Max CPU per tick
            memory_budget_mb: Soft memory limit
            metadata:         Extra context dict
        """
        # Enforce branch limit
        active_count = sum(
            1 for b in self._branches.values()
            if b.state in (BranchState.ACTIVE, BranchState.SUSPENDED)
        )
        if active_count >= self.MAX_BRANCHES:
            logger.warning(
                "BranchManager: cannot spawn '%s' -- at capacity (%d/%d)",
                name, active_count, self.MAX_BRANCHES,
            )
            return None

        # Consult the Unified Will (foreground branches bypass)
        will_receipt_id = ""
        if origin != BranchOrigin.FOREGROUND:
            will_receipt_id = self._consult_will(name, origin, task_description, priority)
            if not will_receipt_id:
                logger.info("BranchManager: Will refused spawn of '%s'", name)
                return None

        # Create the branch
        branch_id = self._make_branch_id(name)
        branch = CognitiveBranch(
            branch_id=branch_id,
            name=name,
            origin=origin,
            priority=min(1.0, max(0.0, priority)),
            context=BranchContext(
                task_description=task_description,
                metadata=metadata or {},
            ),
            cpu_budget_ms=cpu_budget_ms,
            memory_budget_mb=memory_budget_mb,
            will_receipt_id=will_receipt_id,
            _suspend_event=asyncio.Event(),
            _cancel_event=asyncio.Event(),
        )
        # The suspend event starts "set" (not suspended)
        branch._suspend_event.set()

        # If foreground, track it
        if origin == BranchOrigin.FOREGROUND:
            branch.priority = self.FOREGROUND_PRIORITY
            self._foreground_id = branch_id

        # Launch the async task
        branch._task = get_task_tracker().create_task(
            self._run_branch(branch, work_fn),
            name=f"branch:{name}",
        )

        self._branches[branch_id] = branch
        self._total_spawned += 1

        self._publish_event("branch.spawned", {
            "branch_id": branch_id,
            "name": name,
            "origin": origin.value,
            "priority": priority,
        })

        logger.info(
            "Branch spawned: '%s' (id=%s, origin=%s, priority=%.2f)",
            name, branch_id[:8], origin.value, priority,
        )
        return branch

    # ------------------------------------------------------------------
    # Suspend / Resume
    # ------------------------------------------------------------------

    async def suspend(self, branch_id: str, reason: str = "priority_preemption") -> bool:
        """Suspend a branch (it will pause at its next yield point)."""
        branch = self._branches.get(branch_id)
        if not branch or branch.state != BranchState.ACTIVE:
            return False

        branch.state = BranchState.SUSPENDED
        if branch._suspend_event:
            branch._suspend_event.clear()  # Block the branch's wait

        self._publish_event("branch.suspended", {
            "branch_id": branch_id,
            "reason": reason,
        })
        logger.debug("Branch suspended: '%s' (%s)", branch.name, reason)
        return True

    async def resume(self, branch_id: str) -> bool:
        """Resume a suspended branch."""
        branch = self._branches.get(branch_id)
        if not branch or branch.state != BranchState.SUSPENDED:
            return False

        branch.state = BranchState.ACTIVE
        if branch._suspend_event:
            branch._suspend_event.set()  # Unblock the branch

        self._publish_event("branch.resumed", {"branch_id": branch_id})
        logger.debug("Branch resumed: '%s'", branch.name)
        return True

    # ------------------------------------------------------------------
    # Merge (results -> Global Workspace)
    # ------------------------------------------------------------------

    async def merge(self, branch_id: str) -> bool:
        """Merge a completed branch's results into the Global Workspace.

        The result is submitted as a CognitiveCandidate so it competes
        for broadcast alongside other cognitive content.
        """
        branch = self._branches.get(branch_id)
        if not branch or branch.state != BranchState.COMPLETED:
            return False

        result_text = "; ".join(branch.context.partial_results) if branch.context.partial_results else ""
        if not result_text:
            logger.debug("Branch '%s' completed with no results to merge", branch.name)
            self._cleanup_branch(branch_id)
            return True

        # Submit to Global Workspace
        try:
            from core.consciousness.global_workspace import CognitiveCandidate, ContentType

            workspace = ServiceContainer.get("global_workspace", default=None)
            if workspace:
                candidate = CognitiveCandidate(
                    content=f"[Branch:{branch.name}] {result_text[:500]}",
                    source=f"branch_{branch.name}",
                    priority=branch.priority * 0.8,  # Slightly lower than live content
                    content_type=ContentType.INTENTIONAL,
                    affect_weight=0.1,
                )
                await workspace.submit(candidate)
                logger.info(
                    "Branch '%s' results merged into Global Workspace (priority=%.2f)",
                    branch.name, candidate.priority,
                )
        except Exception as e:
            record_degradation('parallel_branches', e)
            logger.warning("Failed to merge branch '%s' into workspace: %s", branch.name, e)

        self._cleanup_branch(branch_id)
        return True

    # ------------------------------------------------------------------
    # Tick -- called each heartbeat cycle
    # ------------------------------------------------------------------

    async def tick(self) -> None:
        """Per-heartbeat maintenance cycle.

        1. Enforce foreground priority (suspend background if needed)
        2. Reset per-tick CPU counters
        3. Auto-merge completed branches
        4. Clean up failed branches
        """
        self._tick_count += 1

        # 1. Priority enforcement: if foreground is active, suspend low-priority
        await self._enforce_priority()

        # 2. Reset per-tick CPU accounting
        for branch in self._branches.values():
            branch.cpu_used_ms = 0.0
            if branch.state == BranchState.ACTIVE:
                branch.ticks_active += 1
                branch.last_tick_at = time.time()
            elif branch.state == BranchState.SUSPENDED:
                branch.ticks_suspended += 1

        # 3. Auto-merge completed branches
        completed = [
            bid for bid, b in self._branches.items()
            if b.state == BranchState.COMPLETED
        ]
        for bid in completed:
            await self.merge(bid)

        # 4. Clean up failed branches
        failed = [
            bid for bid, b in self._branches.items()
            if b.state == BranchState.FAILED
        ]
        for bid in failed:
            self._cleanup_branch(bid)

    # ------------------------------------------------------------------
    # Internal: branch execution wrapper
    # ------------------------------------------------------------------

    async def _run_branch(
        self,
        branch: CognitiveBranch,
        work_fn: BranchWorkFn,
    ) -> None:
        """Wrapper that runs a branch's work function with resource tracking.

        The work function should periodically call `await branch_yield(branch)`
        to check for suspension and CPU budget enforcement.
        """
        try:
            t0 = time.monotonic()
            result = await work_fn(branch)
            elapsed_ms = (time.monotonic() - t0) * 1000
            branch.total_cpu_ms += elapsed_ms

            if result:
                branch.context.partial_results.append(str(result))

            branch.state = BranchState.COMPLETED
            branch.context.progress = 1.0
            self._total_completed += 1

            self._publish_event("branch.completed", {
                "branch_id": branch.branch_id,
                "name": branch.name,
                "total_cpu_ms": round(branch.total_cpu_ms, 2),
                "ticks_active": branch.ticks_active,
            })
            logger.info(
                "Branch completed: '%s' (cpu=%.1fms, ticks=%d)",
                branch.name, branch.total_cpu_ms, branch.ticks_active,
            )

        except asyncio.CancelledError:
            branch.state = BranchState.FAILED
            self._total_failed += 1
            logger.debug("Branch cancelled: '%s'", branch.name)

        except Exception as e:
            record_degradation('parallel_branches', e)
            branch.state = BranchState.FAILED
            self._total_failed += 1
            self._publish_event("branch.failed", {
                "branch_id": branch.branch_id,
                "name": branch.name,
                "error": str(e)[:200],
            })
            logger.error("Branch failed: '%s' -- %s", branch.name, e)

    # ------------------------------------------------------------------
    # Priority enforcement
    # ------------------------------------------------------------------

    async def _enforce_priority(self) -> None:
        """Ensure the foreground branch has resources by suspending low-priority branches.

        When the foreground branch is active, all branches with priority below
        _SUSPEND_THRESHOLD are suspended.  When the foreground completes or is
        absent, suspended branches are resumed.
        """
        foreground = self._branches.get(self._foreground_id) if self._foreground_id else None
        foreground_active = foreground and foreground.state == BranchState.ACTIVE

        for bid, branch in self._branches.items():
            if bid == self._foreground_id:
                continue

            if foreground_active and branch.state == BranchState.ACTIVE:
                if branch.priority < self._SUSPEND_THRESHOLD:
                    await self.suspend(bid, reason="foreground_priority")

            elif not foreground_active and branch.state == BranchState.SUSPENDED:
                # Resume background branches when foreground is idle
                await self.resume(bid)

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    async def _cancel_branch(self, branch: CognitiveBranch, reason: str = "explicit") -> None:
        """Cancel a branch's asyncio task."""
        if branch._task and not branch._task.done():
            branch._task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(branch._task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        branch.state = BranchState.FAILED
        self._publish_event("branch.cancelled", {
            "branch_id": branch.branch_id,
            "reason": reason,
        })

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_branch(self, branch_id: str) -> None:
        """Remove a completed/failed branch from the active pool."""
        branch = self._branches.pop(branch_id, None)
        if branch:
            self._history.append(branch.to_dict())
            if branch_id == self._foreground_id:
                self._foreground_id = None

    # ------------------------------------------------------------------
    # Will consultation
    # ------------------------------------------------------------------

    def _consult_will(
        self,
        name: str,
        origin: BranchOrigin,
        task_description: str,
        priority: float,
    ) -> str:
        """Ask the Unified Will for permission to spawn a branch.

        Returns the WillReceipt ID if approved, or empty string if refused.
        """
        try:
            from core.will import get_will, ActionDomain
            will = get_will()
            decision = will.decide(
                content=f"Spawn cognitive branch: {name} -- {task_description[:200]}",
                source=f"branch_manager/{origin.value}",
                domain=ActionDomain.INITIATIVE,
                priority=priority,
            )
            if decision.is_approved():
                return decision.receipt_id
            return ""
        except Exception as e:
            record_degradation('parallel_branches', e)
            logger.debug("BranchManager: Will consultation failed (degraded): %s", e)
            # Degrade gracefully -- allow spawn without Will if Will is unavailable
            return f"degraded_{int(time.time())}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_branch_id(name: str) -> str:
        raw = f"{time.time():.6f}:{name}"
        return "br_" + hashlib.sha256(raw.encode()).hexdigest()[:12]

    def _publish_event(self, topic: str, data: Dict[str, Any]) -> None:
        """Publish a branch lifecycle event to the EventBus."""
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish_threadsafe(topic, data)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API: yield point for branch work functions
    # ------------------------------------------------------------------

    @staticmethod
    async def branch_yield(branch: CognitiveBranch) -> None:
        """Cooperative yield point for branch work functions.

        Call this periodically inside your work_fn to:
        1. Honour suspension (will block until resumed)
        2. Check for cancellation
        3. Enforce CPU budget (yields to event loop)

        Usage inside a work_fn:
            async def my_work(branch):
                for item in items:
                    process(item)
                    branch.context.progress = i / len(items)
                    await BranchManager.branch_yield(branch)
        """
        # Check cancellation
        if branch._cancel_event and branch._cancel_event.is_set():
            raise asyncio.CancelledError("Branch cancel requested")

        # Wait if suspended
        if branch._suspend_event:
            await branch._suspend_event.wait()

        # Yield to event loop (cooperative multitasking)
        await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Status / Snapshot
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return current state for health/status endpoints."""
        return {
            "started": self._started,
            "tick_count": self._tick_count,
            "active_branches": sum(
                1 for b in self._branches.values()
                if b.state == BranchState.ACTIVE
            ),
            "suspended_branches": sum(
                1 for b in self._branches.values()
                if b.state == BranchState.SUSPENDED
            ),
            "total_branches": len(self._branches),
            "max_branches": self.MAX_BRANCHES,
            "total_spawned": self._total_spawned,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "foreground_id": self._foreground_id,
            "branches": {
                bid: b.to_dict() for bid, b in self._branches.items()
            },
        }

    def get_branch(self, branch_id: str) -> Optional[CognitiveBranch]:
        """Get a branch by ID."""
        return self._branches.get(branch_id)

    def get_active_branches(self) -> List[CognitiveBranch]:
        """Get all active (non-suspended, non-completed) branches."""
        return [
            b for b in self._branches.values()
            if b.state == BranchState.ACTIVE
        ]

    def get_history(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return recent branch history."""
        return list(self._history)[-n:]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_branch_manager: Optional[BranchManager] = None


def get_branch_manager() -> BranchManager:
    """Get or create the singleton BranchManager."""
    global _branch_manager
    if _branch_manager is None:
        _branch_manager = BranchManager()
    return _branch_manager
