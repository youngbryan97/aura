"""How the goal store is brought back into agreement with the task engine.

Two disagreements are repaired: a record whose plan the engine no longer
holds, and two active records for one objective.

Lifted whole out of `goal_engine`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations


class _GoalReconciliationMixin:
    """Lifted whole out of GoalEngine; see goal_engine.py."""

    def _active_task_engine_plan_ids(self) -> set[str] | None:
        from .goal_engine import (
            FallbackClassification,
            ServiceContainer,
            _record_goal_degradation,
            logger,
        )

        try:
            task_engine = ServiceContainer.get("task_engine", default=None)
        # not a failure: no service here, so the caller falls back to its own default.
        except (ImportError, AttributeError, RuntimeError):
            task_engine = None

        if task_engine is None or not hasattr(task_engine, "get_active_plans"):
            return None

        try:
            snapshot = list(task_engine.get_active_plans() or [])
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_goal_degradation(
                exc,
                severity="warning",
                action="skipped task-engine active plan reconciliation for this cycle",
                classification=FallbackClassification.SILENT_LOSS_OF_CAPABILITY,
                extra={"phase": "active_task_engine_plan_ids"},
            )
            logger.debug("GoalEngine task-engine plan snapshot skipped: %s", exc)
            return None

        active_ids: set[str] = set()
        for item in snapshot:
            if not isinstance(item, dict):
                continue
            for key in ("plan_id", "task_id"):
                value = str(item.get(key, "") or "").strip()
                if value:
                    active_ids.add(value)
        return active_ids

    def _reconcile_stale_task_engine_records(self) -> None:
        from .goal_engine import (
            ACTIVE_GOAL_STATUSES,
            GoalRecord,
            GoalStatus,
            asdict,
        )

        now = self._now()
        active_plan_ids = self._active_task_engine_plan_ids()
        if active_plan_ids is None:
            return
        active_records = self._fetch_records(statuses=ACTIVE_GOAL_STATUSES, limit=500)

        for record in active_records:
            if str(record.source or "") != "task_engine":
                continue
            plan_id = str(record.plan_id or record.task_id or "").strip()
            if not plan_id or plan_id in active_plan_ids:
                continue
            age_s = max(0.0, now - float(record.updated_at or record.created_at or now))
            if age_s < 15.0:
                continue

            metadata = dict(record.metadata or {})
            if metadata.get("reconciled_stale_plan"):
                continue
            metadata["reconciled_stale_plan"] = True
            metadata["reconciled_at"] = now
            metadata["last_known_plan_id"] = plan_id

            summary = str(record.summary or "")
            if not summary:
                summary = "Interrupted before completion. No active task-engine plan still owns this goal."
            error = str(record.error or "")
            if not error:
                error = "Task engine plan interrupted or lost ownership before completion."

            repaired_payload = asdict(record)
            repaired_payload.update(
                {
                    "status": GoalStatus.BLOCKED.value,
                    "summary": summary[:2400],
                    "error": error[:2400],
                    "metadata": metadata,
                    "updated_at": now,
                    "last_progress_at": now,
                }
            )
            repaired = GoalRecord(**repaired_payload)
            self._write_record(repaired)

    def _reconcile_duplicate_active_records(self) -> None:
        from .goal_engine import (
            ACTIVE_GOAL_STATUSES,
            GoalRecord,
            GoalStatus,
            asdict,
        )

        now = self._now()
        active_records = self._fetch_records(statuses=ACTIVE_GOAL_STATUSES, limit=500)
        grouped: dict[tuple[str, str, str], list[GoalRecord]] = {}

        for record in active_records:
            signature = self._goal_signature(record.objective or record.name)
            if not signature:
                continue
            key = (str(record.source or ""), str(record.horizon or ""), signature)
            grouped.setdefault(key, []).append(record)

        for (source, _horizon, _signature), records in grouped.items():
            if len(records) <= 1:
                continue
            ordered = sorted(
                records,
                key=lambda item: (
                    float(item.updated_at or item.created_at or 0.0),
                    float(item.priority or 0.0),
                ),
                reverse=True,
            )
            canonical = ordered[0]
            for stale in ordered[1:]:
                metadata = dict(stale.metadata or {})
                if metadata.get("duplicate_reconciled_to") == canonical.id:
                    continue
                metadata["duplicate_reconciled_to"] = canonical.id
                metadata["reconciled_at"] = now
                summary = str(stale.summary or "") or "Superseded by a newer goal record for the same objective."
                error = str(stale.error or "")
                next_status = (
                    GoalStatus.ABANDONED.value
                    if source == "executive_authority"
                    else GoalStatus.BLOCKED.value
                )
                repaired_payload = asdict(stale)
                repaired_payload.update(
                    {
                        "status": next_status,
                        "summary": summary[:2400],
                        "error": error[:2400],
                        "metadata": metadata,
                        "updated_at": now,
                        "last_progress_at": now,
                    }
                )
                repaired = GoalRecord(**repaired_payload)
                self._write_record(repaired)

    def _maybe_reconcile_runtime_records(self) -> None:
        from .goal_engine import (
            FallbackClassification,
            _record_goal_degradation,
            logger,
        )

        if self._conn is None or self._reconciling:
            return
        now = self._now()
        if (now - float(self._last_reconcile_at or 0.0)) < self._reconcile_interval_s:
            return

        self._reconciling = True
        try:
            self._reconcile_stale_task_engine_records()
            self._reconcile_duplicate_active_records()
            self._last_reconcile_at = now
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_goal_degradation(
                exc,
                severity="degraded",
                action="left existing goal records unchanged after reconciliation failure",
                classification=FallbackClassification.SILENT_LOSS_OF_CAPABILITY,
                extra={"phase": "runtime_reconcile"},
            )
            logger.debug("GoalEngine reconciliation skipped: %s", exc)
        finally:
            self._reconciling = False

