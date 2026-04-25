"""Canonical Aura ServiceManifest.

Codifies the runtime ownership rules that the audits demand:

- exactly one runtime owner
- exactly one model owner
- exactly one memory write authority
- exactly one state mutation authority
- exactly one event bus mode
- exactly one canonical substrate service
- exactly one output gate
- exactly one governance authority
- exactly one autonomy initiator

The manifest does not start services; it records who is allowed to own
each role and provides verification helpers used by the boot path and the
conformance harness.

Critical roles must have exactly one resolved owner before READY. Optional
roles may be unbound. Late critical registrations are forbidden once the
registry is locked, which is enforced by the existing ServiceContainer
lock_registration() gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

# ---------------------------------------------------------------------------
# Manifest data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceRole:
    """A single runtime role with declared ownership properties."""

    name: str
    description: str
    critical: bool
    canonical_owner: str
    aliases: FrozenSet[str] = field(default_factory=frozenset)
    duplicate_owner_is_fatal: bool = True


# Canonical roles that the audits identified as load-bearing for runtime
# invariants. The names match service registry keys / aliases used in the
# rest of the codebase.
SERVICE_MANIFEST: Dict[str, ServiceRole] = {
    "runtime": ServiceRole(
        name="runtime",
        description="Single canonical AuraRuntime / orchestrator owner",
        critical=True,
        canonical_owner="orchestrator",
        aliases=frozenset({"orchestrator", "aura_runtime"}),
    ),
    "model": ServiceRole(
        name="model",
        description="Single model runtime owner (mlx_client / model_runtime)",
        critical=True,
        canonical_owner="model_runtime",
        aliases=frozenset({"model_runtime", "mlx_client", "cognitive_engine"}),
    ),
    "memory_writer": ServiceRole(
        name="memory_writer",
        description="Single MemoryWriteGateway / memory authority",
        critical=True,
        canonical_owner="memory_write_gateway",
        aliases=frozenset({"memory_write_gateway", "memory_facade"}),
    ),
    "state_writer": ServiceRole(
        name="state_writer",
        description="Single StateGateway / StateRepository owner",
        critical=True,
        canonical_owner="state_repository",
        aliases=frozenset({"state_repository", "state_vault", "state_gateway"}),
    ),
    "event_bus": ServiceRole(
        name="event_bus",
        description="Single AuraEventBus owner",
        critical=True,
        canonical_owner="event_bus",
        aliases=frozenset({"event_bus", "aura_event_bus"}),
    ),
    "actor_bus": ServiceRole(
        name="actor_bus",
        description="Single ActorBus owner",
        critical=True,
        canonical_owner="actor_bus",
    ),
    "output_gate": ServiceRole(
        name="output_gate",
        description="Single OutputGateway owner",
        critical=True,
        canonical_owner="output_gate",
    ),
    "governance": ServiceRole(
        name="governance",
        description="Single Unified Will / governance authority",
        critical=True,
        canonical_owner="unified_will",
        aliases=frozenset({"unified_will", "will", "governance"}),
    ),
    "autonomy": ServiceRole(
        name="autonomy",
        description="Single autonomous-initiative owner",
        critical=False,
        canonical_owner="metabolic_coordinator",
        aliases=frozenset({"metabolic_coordinator", "autonomy_engine"}),
    ),
    "substrate": ServiceRole(
        name="substrate",
        description="Single canonical substrate service",
        critical=False,
        canonical_owner="liquid_substrate",
    ),
    "task_supervisor": ServiceRole(
        name="task_supervisor",
        description="Single TaskTracker that owns background tasks",
        critical=True,
        canonical_owner="task_tracker",
    ),
    "shutdown_coordinator": ServiceRole(
        name="shutdown_coordinator",
        description="Single ShutdownCoordinator owning teardown ordering",
        critical=True,
        canonical_owner="shutdown_coordinator",
    ),
}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestViolation:
    role: str
    reason: str
    severity: str  # "critical" or "warning"


def verify_manifest(
    registered: Dict[str, object],
    *,
    strict: bool = False,
    manifest: Optional[Dict[str, ServiceRole]] = None,
) -> List[ManifestViolation]:
    """Check the manifest against a snapshot of the live service registry.

    Args:
        registered: Mapping from service name to instance.
        strict: When True, optional services with duplicate authority
            are also reported as critical.
        manifest: Override the manifest (used by tests).

    Returns:
        List of violations. Empty list means the manifest is satisfied.
    """
    if manifest is None:
        manifest = SERVICE_MANIFEST
    violations: List[ManifestViolation] = []
    for role_name, role in manifest.items():
        candidates: Set[str] = set()
        if role.canonical_owner in registered:
            candidates.add(role.canonical_owner)
        for alias in role.aliases:
            if alias in registered:
                candidates.add(alias)
        if not candidates:
            if role.critical:
                violations.append(
                    ManifestViolation(
                        role=role_name,
                        reason=f"no canonical owner registered (expected '{role.canonical_owner}')",
                        severity="critical",
                    )
                )
            continue
        owners = {id(registered[name]) for name in candidates}
        if len(owners) > 1:
            severity = "critical" if (role.critical or strict) and role.duplicate_owner_is_fatal else "warning"
            violations.append(
                ManifestViolation(
                    role=role_name,
                    reason=(
                        "multiple distinct owners registered: "
                        + ", ".join(sorted(candidates))
                    ),
                    severity=severity,
                )
            )
    return violations


def critical_violations(violations: List[ManifestViolation]) -> List[ManifestViolation]:
    return [v for v in violations if v.severity == "critical"]


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def required_role_names() -> Set[str]:
    return {role.name for role in SERVICE_MANIFEST.values() if role.critical}


def all_role_names() -> Set[str]:
    return set(SERVICE_MANIFEST.keys())
