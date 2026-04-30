"""Typed failures for the Autonomous Architecture Governor."""
from __future__ import annotations


class ArchitectError(RuntimeError):
    """Base class for ASA failures."""


class ArchitectConfigError(ArchitectError):
    """Configuration is invalid or unsafe."""


class GraphBuildError(ArchitectError):
    """The architecture graph could not be built."""


class ProtectedSurfaceError(ArchitectError):
    """A proposed mutation touches a protected or sealed surface."""


class ShadowWorkspaceError(ArchitectError):
    """A shadow workspace could not be prepared or executed."""


class GhostBootError(ArchitectError):
    """A ghost boot or shadow proof command failed."""


class ProofError(ArchitectError):
    """Proof obligations failed or could not be verified."""


class RollbackError(ArchitectError):
    """Rollback packet creation or restoration failed."""


class PromotionError(ArchitectError):
    """A candidate could not be promoted."""


class MonitorError(ArchitectError):
    """Post-promotion monitoring failed."""
