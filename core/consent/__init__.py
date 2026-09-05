"""User Consent Workflow - Explicit approval for sensitive operations."""

from core.consent.workflow import (
    ConsentWorkflow,
    SensitivityLevel,
    get_consent_workflow,
    OPERATION_SENSITIVITY,
)

__all__ = [
    "ConsentWorkflow",
    "SensitivityLevel",
    "get_consent_workflow",
    "OPERATION_SENSITIVITY",
]
