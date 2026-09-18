"""core/runtime/mode.py -- Canonical Runtime Mode Enforcement
================================================================

Aura supports exactly five strict runtime modes, plus backward compatibility for research/safe.
Every module that needs to check "am I in production?" or "is research enabled?"
must use these helpers. No ad-hoc os.environ checks.

Modes:
    production  — Default. Research features disabled. Fail-closed. Unsigned
                  skills do not load. Self-modification disabled.
    live        — Strictly live mode. No simulation or mocking. Fail-closed.
    simulated   — Sandboxed environment for testing. No real side effects.
                  All tool calls are mocked.
    test        — Sandboxed environment specifically for pytest and unit tests.
    dev         — Full access. Self-modification allowed (sandboxed). Debug
                  endpoints active. Hot-reload enabled.
    research    — Research features enabled. Self-repair allowed (sandboxed).
    safe        — Emergency lockdown. All autonomous behavior disabled. No tools.

Invariant:
    os.environ["AURA_MODE"] is the single source of truth.
    If unset, mode is "production".
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Runtime.Mode")


class AuraMode(StrEnum):
    """The canonical runtime modes."""
    PRODUCTION = "production"
    LIVE = "live"
    SIMULATED = "simulated"
    TEST = "test"
    DEV = "dev"
    RESEARCH = "research"
    SAFE = "safe"


_VALID_MODES = frozenset(m.value for m in AuraMode)

# Mode Manifests defining exact capability sets
MODE_MANIFESTS: dict[AuraMode, dict[str, Any]] = {
    AuraMode.PRODUCTION: {
        "runtime_mode": "production",
        "llm_backend": "primary_instruct",
        "tools_live": True,
        "network_live": True,
        "filesystem_live": True,
        "computer_use_live": False,
        "worlds_simulated": False,
        "worlds_external": True,
        "allows_research": False,
        # Tier-gated, not forbidden. The mutation constitution is the real
        # constraint here — Tier 0 and Tier 1 paths auto-apply after
        # targeted tests, behavioural contracts, shadow validation and a
        # rollback snapshot, and sealed paths are refused from inside a
        # running Aura whatever this says. Declaring False was a
        # declaration nobody checked: the engine never consulted this
        # manifest, so the table said one thing for eleven modes and the
        # runtime did another. Now that admit_mutation reads it, it has
        # to be true.
        "allows_self_modification": True,
        "allows_autonomous": True,
        "allows_tools": True,
        "allows_unsigned_skills": False,
        "fail_closed_on_degradation": True,
        "max_autonomy_level": 2,
    },
    AuraMode.LIVE: {
        "runtime_mode": "live",
        "llm_backend": "primary_instruct",
        "tools_live": True,
        "network_live": True,
        "filesystem_live": True,
        "computer_use_live": False,
        "worlds_simulated": False,
        "worlds_external": True,
        "allows_research": False,
        # Tier-gated, not forbidden. The mutation constitution is the real
        # constraint here — Tier 0 and Tier 1 paths auto-apply after
        # targeted tests, behavioural contracts, shadow validation and a
        # rollback snapshot, and sealed paths are refused from inside a
        # running Aura whatever this says. Declaring False was a
        # declaration nobody checked: the engine never consulted this
        # manifest, so the table said one thing for eleven modes and the
        # runtime did another. Now that admit_mutation reads it, it has
        # to be true.
        "allows_self_modification": True,
        "allows_autonomous": True,
        "allows_tools": True,
        "allows_unsigned_skills": False,
        "fail_closed_on_degradation": True,
        "max_autonomy_level": 2,
    },
    AuraMode.DEV: {
        "runtime_mode": "dev",
        "llm_backend": "mlx_client",
        "tools_live": True,
        "network_live": True,
        "filesystem_live": True,
        "computer_use_live": True,
        "worlds_simulated": True,
        "worlds_external": False,
        "allows_research": True,
        "allows_self_modification": True,
        "allows_autonomous": True,
        "allows_tools": True,
        "allows_unsigned_skills": True,
        "fail_closed_on_degradation": False,
        "max_autonomy_level": 5,
    },
    AuraMode.SIMULATED: {
        "runtime_mode": "simulated",
        "llm_backend": "mock_client",
        "tools_live": False,
        "network_live": False,
        "filesystem_live": False,
        "computer_use_live": False,
        "worlds_simulated": True,
        "worlds_external": False,
        "allows_research": True,
        "allows_self_modification": False,
        "allows_autonomous": True,
        "allows_tools": True,
        "allows_unsigned_skills": True,
        "fail_closed_on_degradation": False,
        "max_autonomy_level": 3,
    },
    AuraMode.TEST: {
        "runtime_mode": "test",
        "llm_backend": "mock_client",
        "tools_live": False,
        "network_live": False,
        "filesystem_live": False,
        "computer_use_live": False,
        "worlds_simulated": True,
        "worlds_external": False,
        "allows_research": True,
        "allows_self_modification": False,
        "allows_autonomous": False,
        "allows_tools": True,
        "allows_unsigned_skills": True,
        "fail_closed_on_degradation": False,
        "max_autonomy_level": 1,
    },
    AuraMode.RESEARCH: {
        "runtime_mode": "research",
        "llm_backend": "primary_instruct",
        "tools_live": True,
        "network_live": True,
        "filesystem_live": True,
        "computer_use_live": False,
        "worlds_simulated": True,
        "worlds_external": False,
        "allows_research": True,
        "allows_self_modification": False,
        "allows_autonomous": True,
        "allows_tools": True,
        "allows_unsigned_skills": True,
        "fail_closed_on_degradation": False,
        "max_autonomy_level": 4,
    },
    AuraMode.SAFE: {
        "runtime_mode": "safe",
        "llm_backend": "mock_client",
        "tools_live": False,
        "network_live": False,
        "filesystem_live": False,
        "computer_use_live": False,
        "worlds_simulated": True,
        "worlds_external": False,
        "allows_research": False,
        "allows_self_modification": False,
        "allows_autonomous": False,
        "allows_tools": False,
        "allows_unsigned_skills": False,
        "fail_closed_on_degradation": True,
        "max_autonomy_level": 0,
    },
}


def get_mode() -> AuraMode:
    """Return the current runtime mode. Default is production."""
    raw = os.environ.get("AURA_MODE", "production").strip().lower()
    # Backward compatibility mappings
    if raw == "simulation":
        raw = "simulated"
    if raw not in _VALID_MODES:
        logger.warning(
            "Unknown AURA_MODE=%r; falling back to 'production'. Valid modes: %s",
            raw,
            ", ".join(sorted(_VALID_MODES)),
        )
        return AuraMode.PRODUCTION
    return AuraMode(raw)


def get_active_manifest() -> dict[str, Any]:
    """Return the capability manifest for the current runtime mode."""
    mode = get_mode()
    return MODE_MANIFESTS.get(mode, MODE_MANIFESTS[AuraMode.PRODUCTION])


def is_production() -> bool:
    """True if running in production mode (default)."""
    return get_mode() == AuraMode.PRODUCTION


def is_research() -> bool:
    """True if running in research or dev or test mode."""
    return get_mode() in (AuraMode.RESEARCH, AuraMode.DEV, AuraMode.TEST)


def is_dev() -> bool:
    """True if running in developer mode."""
    return get_mode() == AuraMode.DEV


def is_simulation() -> bool:
    """True if running in simulation mode (all side effects mocked)."""
    return get_mode() == AuraMode.SIMULATED


def is_safe() -> bool:
    """True if running in safe mode (emergency lockdown)."""
    return get_mode() == AuraMode.SAFE


# ── Governance strictness: the single cross-reference ──────────────────────
# Historically four independent switches decided "am I hardened?" and none
# referenced the others: contracts.py (AURA_CONTRACTS_ENFORCE), Will
# (AURA_STRICT_WILL / AURA_GOVERNANCE_MODE), this module (AURA_MODE), and
# governance_context (AURA_GOVERNANCE_MODE). An operator could set
# AURA_MODE=production believing the runtime was hardened while Will's
# default-deny stayed asleep. This resolver is the one place that reads every
# signal, so consumers cross-reference through it, and it flags the case where
# the capability mode claims production but governance is not actually hard.
#
# Design note (deliberate): AURA_MODE is the *capability* manifest (what side
# effects are live). Governance hardening (Will default-deny, contract
# fail-closed) is a *separate, explicit* switch so that turning on production
# capabilities does not silently flip default-deny across every consumer — the
# offline suite runs at the production default and must not become fail-closed
# implicitly. Consistency is surfaced loudly at startup instead.
@dataclass(frozen=True)
class GovernanceStrictness:
    strict_will: bool           # Will default-deny active
    enforce_contracts: bool     # DbC violations fail closed
    governance_production: bool # governance_context raises on violations
    mode_claims_production: bool
    hardened: bool              # the load-bearing gate (Will default-deny)
    consistent: bool            # not (mode claims production but unhardened)
    advisory: str = ""


def governance_strictness() -> GovernanceStrictness:
    """Resolve every governance-hardening signal in one place."""
    gov_mode = os.environ.get("AURA_GOVERNANCE_MODE", "").strip().lower()
    strict_will = (
        os.environ.get("AURA_STRICT_WILL") == "1" or gov_mode in {"production", "strict"}
    )
    enforce_contracts = os.environ.get("AURA_CONTRACTS_ENFORCE", "0") == "1"
    governance_production = gov_mode == "production"
    mode_claims_production = get_mode() in (AuraMode.PRODUCTION, AuraMode.LIVE)
    hardened = strict_will
    consistent = (not mode_claims_production) or hardened
    advisory = ""
    if not consistent:
        advisory = (
            "AURA_MODE claims production/live but governance is NOT hardened: "
            "Will default-deny is off. Set AURA_GOVERNANCE_MODE=production to "
            "harden Will and AURA_CONTRACTS_ENFORCE=1 to fail contracts closed."
        )
    return GovernanceStrictness(
        strict_will=strict_will,
        enforce_contracts=enforce_contracts,
        governance_production=governance_production,
        mode_claims_production=mode_claims_production,
        hardened=hardened,
        consistent=consistent,
        advisory=advisory,
    )


def strict_will_active() -> bool:
    """Will default-deny gate. Canonical definition, read by core.governance.will."""
    return governance_strictness().strict_will


def contracts_enforced() -> bool:
    """DbC fail-closed gate. Canonical definition, read by core.resilience.contracts."""
    return governance_strictness().enforce_contracts


def governance_production_active() -> bool:
    """governance_context hard-raise gate."""
    return governance_strictness().governance_production


def governance_hardened() -> bool:
    """True when the runtime's governance is actually fail-closed."""
    return governance_strictness().hardened


def allows_self_modification() -> bool:
    """True if the current mode allows self-modification."""
    return get_active_manifest()["allows_self_modification"]


def allows_research_features() -> bool:
    """True if the current mode allows research/experimental features."""
    return get_active_manifest()["allows_research"]


def allows_autonomous_behavior() -> bool:
    """True if the current mode allows autonomous background behavior."""
    return get_active_manifest()["allows_autonomous"]


def allows_tool_execution() -> bool:
    """True if the current mode allows tool/skill execution."""
    return get_active_manifest()["allows_tools"]


def allows_unsigned_skills() -> bool:
    """True if the current mode allows unsigned/unmanifested skills."""
    return get_active_manifest()["allows_unsigned_skills"]


def max_autonomy_level() -> int:
    """Return the maximum autonomy level for the current mode."""
    return get_active_manifest()["max_autonomy_level"]


def enforce_production_gate(feature_name: str) -> None:
    """Raise RuntimeError if a research/dev feature is used in production mode."""
    mode = get_mode()
    if mode in (AuraMode.PRODUCTION, AuraMode.LIVE):
        raise RuntimeError(
            f"Feature '{feature_name}' is not available in production or live mode. "
            f"Set AURA_MODE=research or AURA_MODE=dev to enable it."
        )


def mode_context() -> dict[str, Any]:
    """Return a dict describing the current mode for logging/diagnostics."""
    manifest = get_active_manifest()
    ctx = {
        "mode": manifest["runtime_mode"],
        "allows_research": manifest["allows_research"],
        "allows_self_modification": manifest["allows_self_modification"],
        "allows_autonomous": manifest["allows_autonomous"],
        "allows_tools": manifest["allows_tools"],
        "allows_unsigned_skills": manifest["allows_unsigned_skills"],
        "max_autonomy_level": manifest["max_autonomy_level"],
        "llm_backend": manifest["llm_backend"],
        "tools_live": manifest["tools_live"],
        "network_live": manifest["network_live"],
        "filesystem_live": manifest["filesystem_live"],
        "computer_use_live": manifest["computer_use_live"],
        "worlds_simulated": manifest["worlds_simulated"],
        "worlds_external": manifest["worlds_external"],
    }
    return ctx


def validate_mode_at_startup() -> None:
    """Log the current mode and validate configuration consistency."""
    mode = get_mode()
    ctx = mode_context()
    logger.info("🔧 Runtime mode: %s", mode.value)
    for key, value in ctx.items():
        if key != "mode":
            logger.debug("  %s: %s", key, value)

    # Validate environment variable consistency
    autonomy_env = os.environ.get("AURA_AUTONOMY_LEVEL")
    if autonomy_env is not None:
        try:
            level = int(autonomy_env)
            if level > max_autonomy_level():
                logger.warning(
                    "AURA_AUTONOMY_LEVEL=%d exceeds max for mode %s (%d); clamping.",
                    level,
                    mode.value,
                    max_autonomy_level(),
                )
                os.environ["AURA_AUTONOMY_LEVEL"] = str(max_autonomy_level())
        except ValueError:
            logger.warning("Invalid AURA_AUTONOMY_LEVEL=%r; ignoring.", autonomy_env)

    # Governance-hardening consistency: close the "I set AURA_MODE=production
    # so I must be hardened" trap by surfacing the disagreement loudly.
    strictness = governance_strictness()
    if not strictness.consistent:
        logger.warning("⚠️  GOVERNANCE NOT HARDENED: %s", strictness.advisory)
        # In a headless proof/offline run the split is intentional: the suite
        # exercises production capabilities at the production default and must
        # NOT become fail-closed implicitly (see governance_strictness above).
        # Surface it loudly, but don't count it as a runtime degradation there.
        _proof_offline = False
        try:
            from core.runtime.proof_policy import proof_headless_run

            _proof_offline = proof_headless_run()
        except (ImportError, RuntimeError, AttributeError):
            _proof_offline = False
        if not _proof_offline:
            try:
                from core.runtime.errors import record_degradation

                record_degradation(
                    "runtime.mode",
                    RuntimeError(strictness.advisory),
                    severity="warning",
                    action="ran with production capabilities but non-hardened governance",
                )
            except (ImportError, AttributeError, RuntimeError):
                pass
    else:
        logger.info(
            "Governance: strict_will=%s enforce_contracts=%s (hardened=%s)",
            strictness.strict_will,
            strictness.enforce_contracts,
            strictness.hardened,
        )

    if mode == AuraMode.SAFE:
        logger.warning("🔒 SAFE MODE ACTIVE: All autonomous behavior disabled.")
