"""Canonical source-promotion policy for self-modification.

Normal Aura runtime may diagnose, draft, stage, and validate repairs. Writing a
validated patch back into the live source tree is a separate promotion action
and must only happen in an operator-controlled repair-lab or supervised flow.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

RUNTIME_SELF_MODIFICATION_ENV = "AURA_ALLOW_RUNTIME_SELF_MODIFICATION"
AUTONOMOUS_PATCH_PROMOTION_ENV = "AURA_ALLOW_AUTONOMOUS_PATCH_PROMOTION"
REPAIR_LAB_SOURCE_PROMOTION_ENV = "AURA_ALLOW_REPAIR_LAB_SOURCE_PROMOTION"
SUPERVISED_SELF_MODIFICATION_ENV = "AURA_ALLOW_SUPERVISED_SELF_MODIFICATION"
SAFE_AUTONOMOUS_REPAIR_ENV = "AURA_ENABLE_SAFE_AUTO_REPAIR"


@dataclass(frozen=True)
class PromotionPolicyDecision:
    allowed: bool
    reason: str
    required_env: tuple[str, ...]


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def autonomous_source_promotion_decision() -> PromotionPolicyDecision:
    required = (
        RUNTIME_SELF_MODIFICATION_ENV,
        AUTONOMOUS_PATCH_PROMOTION_ENV,
        REPAIR_LAB_SOURCE_PROMOTION_ENV,
    )
    missing = tuple(name for name in required if not env_flag(name, False))
    if missing:
        return PromotionPolicyDecision(
            allowed=False,
            reason=(
                "autonomous source promotion requires an explicit repair-lab "
                f"profile with {', '.join(required)}=1"
            ),
            required_env=required,
        )
    return PromotionPolicyDecision(
        allowed=True,
        reason="autonomous repair-lab source promotion enabled",
        required_env=required,
    )


def supervised_source_promotion_decision() -> PromotionPolicyDecision:
    required = (SUPERVISED_SELF_MODIFICATION_ENV,)
    if not env_flag(SUPERVISED_SELF_MODIFICATION_ENV, False):
        return PromotionPolicyDecision(
            allowed=False,
            reason=f"{SUPERVISED_SELF_MODIFICATION_ENV}=1 is required for supervised source promotion",
            required_env=required,
        )
    return PromotionPolicyDecision(
        allowed=True,
        reason="supervised source promotion enabled",
        required_env=required,
    )


def safe_autonomous_repair_decision() -> PromotionPolicyDecision:
    """Return whether low-risk, validator-backed repair promotion is live.

    This is deliberately narrower than the repair-lab source-promotion profile.
    It only authorizes callers that have already classified the target as a
    safe auto-apply mutation tier and are still running the normal quarantine,
    harness, architecture, rollback, and git gates. Operators can turn it off
    with ``AURA_ENABLE_SAFE_AUTO_REPAIR=0`` when they want proposal-only mode.
    """

    required = (SAFE_AUTONOMOUS_REPAIR_ENV,)
    if not env_flag(SAFE_AUTONOMOUS_REPAIR_ENV, True):
        return PromotionPolicyDecision(
            allowed=False,
            reason=f"{SAFE_AUTONOMOUS_REPAIR_ENV}=1 is required for safe autonomous repair",
            required_env=required,
        )
    return PromotionPolicyDecision(
        allowed=True,
        reason="safe autonomous repair enabled",
        required_env=required,
    )


def source_promotion_decision(*, supervised: bool) -> PromotionPolicyDecision:
    if supervised:
        return supervised_source_promotion_decision()
    return autonomous_source_promotion_decision()


__all__ = [
    "AUTONOMOUS_PATCH_PROMOTION_ENV",
    "PromotionPolicyDecision",
    "REPAIR_LAB_SOURCE_PROMOTION_ENV",
    "RUNTIME_SELF_MODIFICATION_ENV",
    "SUPERVISED_SELF_MODIFICATION_ENV",
    "SAFE_AUTONOMOUS_REPAIR_ENV",
    "autonomous_source_promotion_decision",
    "env_flag",
    "safe_autonomous_repair_decision",
    "source_promotion_decision",
    "supervised_source_promotion_decision",
]
