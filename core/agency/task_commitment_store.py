"""Where a commitment is written down, and how it is read back.

Every entry has three shapes: the synchronous write, the async one, and the
state mutation both of them go through. Keeping the trio together is what
stops a caller taking the lock twice, and it is why the split exists at all.
The pruning of terminal tasks is here too, because a ledger nobody prunes is a
ledger that stops being readable.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any, Dict

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.executors import run_durable_receipt_io, run_durable_receipt_io_sync


class _KeepsTheTaskLedger:
    """Lifted whole from TaskCommitmentVerifier; see task_commitment_verifier.py."""

    def _prune_terminal_tasks(self, max_age: float = 300.0) -> None:
        """Remove completed/failed/cancelled tasks older than max_age seconds."""
        snapshot = self._prune_terminal_tasks_state(max_age=max_age)
        if snapshot is not None:
            self._persist_task_snapshot_sync(*snapshot)

    async def _prune_terminal_tasks_async(self, max_age: float = 300.0) -> None:
        snapshot = self._prune_terminal_tasks_state(max_age=max_age)
        if snapshot is not None:
            await self._persist_task_snapshot_async(*snapshot)

    def _prune_terminal_tasks_state(
        self,
        *,
        max_age: float,
    ) -> tuple[int, dict[str, Any]] | None:
        now = time.time()

        def _cleanup_deadline(entry: Dict[str, Any]) -> float:
            explicit = float(entry.get("cleanup_at", 0.0) or 0.0)
            if explicit > 0.0:
                return explicit
            terminal_at = float(
                entry.get("completed_at")
                or entry.get("updated_at")
                or entry.get("started_at")
                or 0.0
            )
            return terminal_at + max_age if terminal_at > 0.0 else now + 1.0

        with self._lock:
            to_remove = [
                tid
                for tid, entry in self._active_tasks.items()
                if str(entry.get("status", "") or "") in {"completed", "failed", "cancelled"}
                and now >= _cleanup_deadline(entry)
            ]
            for tid in to_remove:
                self._active_tasks.pop(tid, None)
            if to_remove:
                self._updated_at = now
                return self._snapshot_task_state_locked()
        return None

    def _store_task_entry(self, task_id: str, entry: Dict[str, Any]) -> None:
        snapshot = self._store_task_entry_state(task_id, entry)
        self._persist_task_snapshot_sync(*snapshot)

    async def _store_task_entry_async(
        self,
        task_id: str,
        entry: Dict[str, Any],
    ) -> None:
        snapshot = self._store_task_entry_state(task_id, entry)
        await self._persist_task_snapshot_async(*snapshot)

    def _store_task_entry_state(
        self,
        task_id: str,
        entry: Dict[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        with self._lock:
            payload = dict(entry)
            payload.setdefault("task_id", task_id)
            payload.setdefault("updated_at", time.time())
            self._active_tasks[task_id] = payload
            self._updated_at = time.time()
            return self._snapshot_task_state_locked()

    def _update_task_entry(self, task_id: str, **updates: Any) -> None:
        snapshot = self._update_task_entry_state(task_id, **updates)
        if snapshot is not None:
            self._persist_task_snapshot_sync(*snapshot)

    async def _update_task_entry_async(self, task_id: str, **updates: Any) -> None:
        snapshot = self._update_task_entry_state(task_id, **updates)
        if snapshot is not None:
            await self._persist_task_snapshot_async(*snapshot)

    def _update_task_entry_state(
        self,
        task_id: str,
        **updates: Any,
    ) -> tuple[int, dict[str, Any]] | None:
        with self._lock:
            entry = self._active_tasks.get(task_id)
            if not entry:
                return None
            entry.update(updates)
            entry["updated_at"] = time.time()
            self._updated_at = time.time()
            return self._snapshot_task_state_locked()

    def _snapshot_task_state_locked(self) -> tuple[int, dict[str, Any]]:
        self._state_generation += 1
        payload = {
            "updated_at": self._updated_at or time.time(),
            "active_tasks": [
                deepcopy(entry)
                for entry in self._active_tasks.values()
                if not self._is_evaluation_task(entry)
            ],
        }
        return self._state_generation, payload

    def _persist_task_snapshot_sync(
        self,
        generation: int,
        payload: dict[str, Any],
    ) -> None:
        run_durable_receipt_io_sync(
            self._commit_task_snapshot,
            generation,
            payload,
            timeout_s=10.0,
            label="task_commitment.persist_snapshot.sync",
        )

    def _commit_task_snapshot(
        self,
        generation: int,
        payload: dict[str, Any],
    ) -> bool:
        """Commit one snapshot without allowing an older generation to win."""
        if generation <= self._persisted_generation:
            return False
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "task_commitment_verifier.persist_snapshot",
            domain="state_mutation",
            receipt_prefix="task-commitment-state",
            constraints={
                "artifact": "aura.task_commitment_state.v1",
                "generation": generation,
                "task_count": len(payload.get("active_tasks") or []),
                "operation": "replace",
            },
        ):
            get_file_write_gateway().write_text(
                self.persist_path,
                json.dumps(payload, indent=2, sort_keys=True),
                source="task_commitment_verifier.persist_snapshot",
            )
        self._persisted_generation = generation
        return True

    async def _persist_task_snapshot_async(
        self,
        generation: int,
        payload: dict[str, Any],
    ) -> None:
        await run_durable_receipt_io(
            self._commit_task_snapshot,
            generation,
            payload,
            timeout_s=10.0,
            label="task_commitment.persist_snapshot",
        )

    def _load(self) -> None:
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .task_commitment_verifier import (
            _RUNNING_TASK_STATUSES,
            logger,
        )

        try:
            if not self.persist_path.exists():
                return
            raw = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            record_degradation('task_commitment_verifier', exc)
            logger.debug("TaskCommitmentVerifier: state load skipped: %s", exc)
            return

        now = time.time()
        loaded: Dict[str, Dict[str, Any]] = {}
        needs_save = False
        for item in list(raw.get("active_tasks") or []):
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id", "") or "").strip()
            if not task_id:
                continue
            entry = dict(item)
            if self._is_evaluation_task(entry):
                needs_save = True
                continue
            status = str(entry.get("status", "") or "")
            if status in _RUNNING_TASK_STATUSES:
                entry["status"] = "interrupted"
                entry.setdefault(
                    "summary",
                    "The last run was interrupted before it could finish, so it needs an explicit resume.",
                )
                entry["interrupted_at"] = now
                entry["updated_at"] = now
                needs_save = True
            loaded[task_id] = entry

        snapshot: tuple[int, dict[str, Any]] | None = None
        with self._lock:
            self._active_tasks = loaded
            self._updated_at = float(raw.get("updated_at") or now)
            if needs_save:
                self._updated_at = now
                snapshot = self._snapshot_task_state_locked()
        pruned_snapshot = self._prune_terminal_tasks_state(max_age=86400.0)
        if pruned_snapshot is not None:
            snapshot = pruned_snapshot
        if snapshot is not None:
            self._persist_task_snapshot_sync(*snapshot)
