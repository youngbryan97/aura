"""User Consent Workflow - Explicit approval for sensitive operations."""

from core.consent.workflow import (
    OPERATION_SENSITIVITY,
    ConsentWorkflow,
    SensitivityLevel,
    get_consent_workflow,
)

__all__ = [
    "ConsentWorkflow",
    "SensitivityLevel",
    "get_consent_workflow",
    "OPERATION_SENSITIVITY",
]
