"""The lane controller's steps that hold its file lock, which `prepare` and `cancel` hand to a thread.

Each takes the interprocess file lock, and marking a reservation ready writes the state with an fsync. LIVE 14
November 2026, lockdep caught that write on her event loop thread, reached from ``prepare``, and every task waited on
the disk for it. They are methods of ``ModelLaneController``, kept here so that class stays within its size.

The module-level names they use (the file lock, the lane budget, the states) are read from ``model_lane_control`` when
they run, so a test that replaces one there replaces it here too.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.runtime.model_lane_control import LaneOwnerObservation, LaneTransactionDecision

__all__ = ["LockedLaneSteps"]


class LockedLaneSteps:
    """``ModelLaneController``'s locked steps that run off the event loop."""

    def _mark_ready_sync(
        self: Any,
        decision: LaneTransactionDecision,
        observations: list[LaneOwnerObservation],
        evicted_any: bool,
    ) -> LaneTransactionDecision | None:
        """The reservation marked ready when it still fits, or None when the capacity changed under it."""
        from core.runtime import model_lane_control as lanes

        with self._thread_lock, lanes.interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            now = self._clock()
            self._prune_locked(state, now)
            self._sync_observations_locked(state, observations, now)
            record = state["reservations"].get(decision.request_id)
            if not isinstance(record, dict):
                raise lanes.ModelLaneControlError("lane_reservation_missing_before_ready")
            self._assert_fence(record, decision)
            committed, reserved = self._capacity_totals(
                state,
                exclude_request_id=decision.request_id,
            )
            observer = self.resource_observer
            budget_gb = lanes.lane_budget_gb(observer=observer)
            provenance = observer.provenance
            capacity_request_gb = float(record.get("capacity_request_gb") or decision.request_gb)
            if committed + reserved + capacity_request_gb > budget_gb:
                return None
            record["state"] = lanes.LaneTransactionState.READY.value
            record["reason"] = "required_evictions_reclaimed" if evicted_any else "capacity_reserved"
            record["committed_gb"] = committed
            record["reserved_gb"] = reserved
            record["budget_gb"] = budget_gb
            record["observation_source"] = provenance.source.value
            record["observation_scenario_id"] = provenance.scenario_id
            record["resource_observation_available"] = bool(
                observer.memory().available
            )
            record["ready_at"] = now
            self._append_event(
                state,
                "reservation_ready",
                at=now,
                transaction_id=decision.transaction_id,
                fencing_token=decision.fencing_token,
            )
            self._save_locked(state)
            ready: LaneTransactionDecision = self._record_to_decision(record)
            return ready

    def _compensation_pending_sync(self: Any, request_id: str) -> bool:
        """Whether a reservation still owes an owner its compensation."""
        from core.runtime import model_lane_control as lanes

        with self._thread_lock, lanes.interprocess_file_lock(self.lock_path):
            pending_record = self._load_locked()["reservations"].get(request_id)
            return bool(
                isinstance(pending_record, Mapping)
                and pending_record.get("compensation_pending_owner_ids")
            )
