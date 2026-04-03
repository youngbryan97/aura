"""Aura Zenith Unified Exception Hierarchy.
"""

class AuraError(Exception):
    """Base exception for all Aura-specific errors."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.context = context or {}

class LLMError(AuraError):
    """Raised when an LLM provider fails (MLX, Gemini, OpenAI)."""
    pass

class NetworkError(AuraError):
    """Raised when external connectivity fails."""
    pass

class AuraTimeoutError(AuraError):
    """Raised when an operation exceeds its time limit."""
    pass

class CircuitOpenError(AuraError):
    """Raised when a circuit breaker blocks a request."""
    pass

class AuraMemoryError(AuraError):
    """Raised when resource constraints (OOM) are hit."""
    pass

class CognitiveError(AuraError):
    pass

class SensesError(AuraError):
    pass

class OrchestratorError(AuraError):
    pass

class CapabilityError(AuraError):
    pass

class InfrastructureError(AuraError):
    pass

class SecurityError(AuraError):
    """Raised when a sandbox or permission guard is breached."""
    pass

class SecurityConfigError(SecurityError):
    """Raised when the system configuration is insecure (e.g. public API without token)."""
    pass

class CircularDependencyError(AuraError):
    """Raised when the ServiceContainer detects a dependency loop."""
    pass

class ContainerError(AuraError):
    """General container failure."""
    pass

class LifecycleError(ContainerError):
    """Service failed during on_start or on_stop."""
    pass

class ServiceNotFoundError(ContainerError):
    """Requested service is not registered."""
    pass

class CriticalServiceMissingError(ContainerError):
    """A required service was not found in the container."""
    pass