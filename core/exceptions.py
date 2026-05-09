"""Aura Zenith Unified Exception Hierarchy.
"""

from typing import Any


class AuraError(Exception):
    """Base exception for all Aura-specific errors."""
    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}

class LLMError(AuraError):
    """Raised when an LLM provider fails (MLX, Gemini, OpenAI)."""
    pass  # no-op: intentional

class NetworkError(AuraError):
    """Raised when external connectivity fails."""
    pass  # no-op: intentional

class AuraTimeoutError(AuraError):
    """Raised when an operation exceeds its time limit."""
    pass  # no-op: intentional

class CircuitOpenError(AuraError):
    """Raised when a circuit breaker blocks a request."""
    pass  # no-op: intentional

class AuraMemoryError(AuraError):
    """Raised when resource constraints (OOM) are hit."""
    pass  # no-op: intentional

class CognitiveError(AuraError):
    pass  # no-op: intentional

class SensesError(AuraError):
    pass  # no-op: intentional

class OrchestratorError(AuraError):
    pass  # no-op: intentional

class CapabilityError(AuraError):
    pass  # no-op: intentional

class InfrastructureError(AuraError):
    pass  # no-op: intentional

class SecurityError(AuraError):
    """Raised when a sandbox or permission guard is breached."""
    pass  # no-op: intentional

class SecurityConfigError(SecurityError):
    """Raised when the system configuration is insecure (e.g. public API without token)."""
    pass  # no-op: intentional

class CircularDependencyError(AuraError):
    """Raised when the ServiceContainer detects a dependency loop."""
    pass  # no-op: intentional

class ContainerError(AuraError):
    """General container failure."""
    pass  # no-op: intentional

class LifecycleError(ContainerError):
    """Service failed during on_start or on_stop."""
    pass  # no-op: intentional

class ServiceNotFoundError(ContainerError):
    """Requested service is not registered."""
    pass  # no-op: intentional

class CriticalServiceMissingError(ContainerError):
    """A required service was not found in the container."""
    pass  # no-op: intentional


# ── Governance & Safety Exception Hierarchy ──────────────────────────────
# These MUST NEVER be caught by broad `except Exception` in critical paths.
# The audit identified ~5,500 broad handlers; governance paths now use these
# typed exceptions which propagate unambiguously.

class GovernanceError(SecurityError):
    """Authority gate or constitutional governance violation.

    Raised when a consequential action is blocked by the Will,
    Constitution, or AuthorityGateway. Never swallow this silently.
    """
    pass  # no-op: intentional


class SubstrateMutationError(GovernanceError):
    """Substrate mutation rejected by authority gate or blocked by fail-closed policy."""
    pass  # no-op: intentional


class AuthorityUnavailableError(GovernanceError):
    """Authority service is unavailable and the operation requires live authorization.

    Fail-closed: this is NOT a degraded-pass condition. The operation must
    wait or abort.
    """
    pass  # no-op: intentional


class ConstitutionIntegrityError(GovernanceError):
    """Constitutional artifact tampering detected via cryptographic seal mismatch.

    Raised when a constitutional file (canonical_self.json, constitutional_core.json,
    etc.) has been modified externally outside the authorized pipeline. The system
    should form a CONSTITUTION_MODIFIED_EXTERNALLY scar and enter safe mode.
    """
    pass  # no-op: intentional


class ProofPathDegradedError(GovernanceError):
    """A proof obligation or verification path has degraded.

    Raised when the behavioral proof pipeline cannot complete its checks
    due to missing dependencies, unavailable test infrastructure, etc.
    """
    pass  # no-op: intentional


class SkillExecutionError(CapabilityError):
    """A skill failed to execute through the CapabilityEngine."""
    def __init__(self, skill_name: str, operation: str, cause: Exception | None = None):
        self.skill_name = skill_name
        self.operation = operation
        self.cause = cause
        super().__init__(
            f"Skill {skill_name}.{operation} failed: {cause}",
            context={"skill": skill_name, "operation": operation},
        )


class MemoryWriteError(AuraError):
    """Memory write operation failed."""
    pass  # no-op: intentional

