"""A turn that ran out of budget must not be reported as a wedged worker.

Live 2026-08-03, two consecutive lines in the desktop log:

    🛑 [MLX] First-token HARD CEILING exceeded (livelocked: heartbeats but
       zero tokens) for Qwen2.5-7B-Instruct-4bit (18.4s elapsed, sla=8.0s,
       hard=16.8s).
    ⏱️ [MLX] Cortex ran past this turn's deadline (18.4s elapsed, budget
       16.8s) but is healthy (heartbeat 0.7s ago, livelock ceiling 20.0s).
       Cancelling the request and KEEPING the warm lane.

18.4s was under the 20.0s livelock ceiling, and the heartbeat was 0.7s old.
Nothing was livelocked. `hard_first_token_ceiling` is
min(livelock_ceiling, the caller's deadline), so exceeding it usually means
the TURN ran out of time — which says nothing about the worker's health.

The branch that decides what to DO already tested the two apart correctly and
kept the warm 20GB lane. Only the line a person reads did not, and it was
logged at error severity, so the incident machinery went hunting a fault that
never happened.
"""
from __future__ import annotations

import re

import pytest

from core.brain.llm.mlx_client import MLXLocalClient
from tests.source_contract import class_with_its_bases


@pytest.fixture(scope="module")
def source() -> str:
    return class_with_its_bases(MLXLocalClient)


@pytest.fixture(scope="module")
def ceiling_block(source: str) -> str:
    start = source.index("elapsed_without_token = time.time() - request_started_at")
    end = source.index("_deferred_reboot_reason", start)
    return source[start:end]


class TestTheMessageMatchesTheVerdict:
    def test_livelock_is_only_claimed_when_the_livelock_ceiling_fired(self, ceiling_block):
        assert "livelocked = elapsed_without_token > livelock_ceiling" in ceiling_block
        livelock_at = ceiling_block.index("livelocked = elapsed_without_token")
        claim_at = ceiling_block.index("LIVELOCK")
        assert livelock_at < claim_at, (
            "the verdict must be computed before the line that announces it"
        )

    def test_the_old_conflation_is_gone(self, source):
        assert "HARD CEILING exceeded (livelocked: heartbeats but zero tokens)" not in source, (
            "this wording fired on a healthy worker that merely missed a deadline"
        )

    def test_a_deadline_overrun_says_the_worker_is_not_wedged(self, ceiling_block):
        assert "The worker is not wedged" in ceiling_block

    def test_the_deadline_branch_is_not_an_error(self, ceiling_block):
        deadline = ceiling_block[ceiling_block.index("elif elapsed_without_token >"):]
        head = deadline[: deadline.index("else:")]
        assert "logger.warning" in head
        assert "logger.error" not in head, (
            "expected backpressure is recorded below error — CLAUDE.md"
        )

    def test_only_a_real_livelock_is_recorded_as_an_error(self, ceiling_block):
        assert 'severity="error" if livelocked else "warning"' in ceiling_block


class TestTheDecisionItselfIsUnchanged:
    """The branch that keeps or recycles the warm lane must not have moved."""

    def test_a_healthy_worker_past_its_deadline_keeps_the_lane(self, source):
        assert "Cancelling the request and KEEPING the warm lane." in source
        assert "first_token_deadline_exceeded_worker_healthy" in source

    def test_a_real_livelock_still_recycles(self, source):
        assert 'self._deferred_reboot_reason = "recoverable_first_token_sla_exceeded"' in source

    def test_a_silent_worker_still_reboots(self, source):
        assert "heartbeat_age > 30.0" in source
        assert 'self._deferred_reboot_reason = "first_token_sla_exceeded"' in source

    def test_the_verdict_is_computed_once(self, source):
        assignments = re.findall(r"livelocked\s*=\s*elapsed_without_token\s*>", source)
        assert len(assignments) == 1, (
            "two copies of this verdict can disagree; that is the defect being fixed"
        )
