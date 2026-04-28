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
