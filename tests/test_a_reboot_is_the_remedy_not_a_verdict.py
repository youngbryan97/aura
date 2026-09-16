"""A worker rebooted for a wedge or a missed cancel comes back cold, not failed.

LIVE, 2026-09-15 23:55Z: the reflex worker missed one soft-cancel
acknowledgement (12s, host at load 30). The deferred verdict rebooted it
with mark_failed=True. The lane then sat in "failed" with that reason; the
router refuses a failed lane before it calls generate, so nothing could
spawn a new worker, and only a runtime probe clears "failed" — which never
fires for a cancel. Four hours, 359 "Circuit OPEN for Reflex" lines, no
reflex lane.
"""
from __future__ import annotations

import pytest

from core.brain.llm import mlx_client as mlx


class _Lane:
    _SOFT_CANCEL_PRESERVABLE_REASONS = mlx.MLXLocalClient._SOFT_CANCEL_PRESERVABLE_REASONS

    def __init__(self, ack: bool) -> None:
        self._ack = ack
        self.reboots: list[tuple[str, bool]] = []
        self.events: list[str] = []
        self.model_path = "/models/Qwen2.5-1.5B-Instruct-4bit"

    async def _soft_cancel_acknowledged(self) -> bool:
        return self._ack

    async def reboot_worker(self, reason: str = "manual_reboot", mark_failed: bool = False):
        self.reboots.append((reason, mark_failed))

    def _record_degraded_event(self, kind, **_):
        self.events.append(kind)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deferred",
    [
        "cancelled_worker_not_acknowledged",
        "worker_died_during_generation",
        "first_token_sla_exceeded",
        "token_progress_stalled",
        "heartbeat_stalled_during_generation",
        "cancelled_unhealthy",
        "generation_timeout_unhealthy",
    ],
)
async def test_an_unrecoverable_verdict_reboots_into_cold_not_failed(deferred):
    lane = _Lane(ack=False)
    await mlx.MLXLocalClient._resolve_deferred_reboot(lane, deferred)
    assert lane.reboots == [(deferred, False)]


@pytest.mark.asyncio
async def test_a_recoverable_verdict_still_preserves_an_acknowledged_worker():
    lane = _Lane(ack=True)
    reason = next(iter(mlx.MLXLocalClient._SOFT_CANCEL_PRESERVABLE_REASONS))
    await mlx.MLXLocalClient._resolve_deferred_reboot(lane, f"recoverable_{reason}")
    assert lane.reboots == []
    assert "warm_lane_preserved_after_soft_cancel" in lane.events


@pytest.mark.asyncio
async def test_a_recoverable_verdict_without_acknowledgement_reboots_cold():
    lane = _Lane(ack=False)
    reason = next(iter(mlx.MLXLocalClient._SOFT_CANCEL_PRESERVABLE_REASONS))
    await mlx.MLXLocalClient._resolve_deferred_reboot(lane, f"recoverable_{reason}")
    assert lane.reboots == [(reason, False)]
