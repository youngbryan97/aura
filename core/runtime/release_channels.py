"""Release channels (canary / dev / beta / stable / lts).

Per the audit, each channel has different gates. Stable promotion
requires conformance + abuse + migration + rollback proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional


@dataclass(frozen=True)
class ChannelPolicy:
    name: str
    promotion_threshold_days: int
    update_frequency_hours: int
    crash_rate_threshold: float
    receipt_coverage_threshold: float
    requires_abuse_pass: bool
    requires_conformance_pass: bool
    requires_migration_pass: bool
    requires_rollback_pass: bool
    max_memory_slope_mb_per_hour: float


CHANNELS: Dict[str, ChannelPolicy] = {
    "nightly": ChannelPolicy(
        name="nightly",
        promotion_threshold_days=0,
        update_frequency_hours=24,
        crash_rate_threshold=1.0,
        receipt_coverage_threshold=0.0,
        requires_abuse_pass=False,
        requires_conformance_pass=False,
        requires_migration_pass=False,
        requires_rollback_pass=False,
        max_memory_slope_mb_per_hour=100.0,
    ),
    "dev": ChannelPolicy(
        name="dev",
        promotion_threshold_days=1,
        update_frequency_hours=24,
        crash_rate_threshold=0.05,
        receipt_coverage_threshold=0.95,
        requires_abuse_pass=False,
        requires_conformance_pass=True,
        requires_migration_pass=False,
        requires_rollback_pass=False,
        max_memory_slope_mb_per_hour=50.0,
    ),
    "beta": ChannelPolicy(
        name="beta",
        promotion_threshold_days=3,
        update_frequency_hours=48,
        crash_rate_threshold=0.005,
        receipt_coverage_threshold=0.99,
        requires_abuse_pass=True,
        requires_conformance_pass=True,
        requires_migration_pass=True,
        requires_rollback_pass=False,
        max_memory_slope_mb_per_hour=20.0,
    ),
    "stable": ChannelPolicy(
        name="stable",
        promotion_threshold_days=7,
        update_frequency_hours=168,
        crash_rate_threshold=0.001,
        receipt_coverage_threshold=1.0,
        requires_abuse_pass=True,
        requires_conformance_pass=True,
        requires_migration_pass=True,
        requires_rollback_pass=True,
        max_memory_slope_mb_per_hour=10.0,
    ),
    "lts": ChannelPolicy(
        name="lts",
        promotion_threshold_days=14,
        update_frequency_hours=720,
        crash_rate_threshold=0.0005,
        receipt_coverage_threshold=1.0,
        requires_abuse_pass=True,
        requires_conformance_pass=True,
        requires_migration_pass=True,
        requires_rollback_pass=True,
        max_memory_slope_mb_per_hour=5.0,
    ),
}


@dataclass
class ReleaseSubmission:
    target_channel: str
    crash_rate: float
    receipt_coverage: float
    abuse_pass: bool
    conformance_pass: bool
    migration_pass: bool
    rollback_pass: bool
    memory_slope_mb_per_hour: float


@dataclass
class ReleaseGateResult:
    accepted: bool
    failed_gates: List[str] = field(default_factory=list)


def evaluate_release(submission: ReleaseSubmission) -> ReleaseGateResult:
    policy = CHANNELS.get(submission.target_channel)
    if policy is None:
        return ReleaseGateResult(False, ["unknown_channel"])
    failures: List[str] = []
    if submission.crash_rate > policy.crash_rate_threshold:
        failures.append("crash_rate_above_threshold")
    if submission.receipt_coverage < policy.receipt_coverage_threshold:
        failures.append("receipt_coverage_below_threshold")
    if policy.requires_abuse_pass and not submission.abuse_pass:
        failures.append("abuse_gauntlet_required")
    if policy.requires_conformance_pass and not submission.conformance_pass:
        failures.append("conformance_required")
    if policy.requires_migration_pass and not submission.migration_pass:
        failures.append("migration_required")
    if policy.requires_rollback_pass and not submission.rollback_pass:
        failures.append("rollback_required")
    if submission.memory_slope_mb_per_hour > policy.max_memory_slope_mb_per_hour:
        failures.append("memory_slope_above_threshold")
    return ReleaseGateResult(accepted=not failures, failed_gates=failures)
