"""
Explicit Consent Workflow (v2026.5.1)

Provides user-facing consent prompts for sensitive operations while
maintaining security and control boundaries.

Sensitive operations requiring consent:
- Sensitive file operations (write to system dirs)
- Network/API calls
- State mutations
- Memory writes
- Background task execution
- System commands
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class SensitivityLevel(str, Enum):
    """Sensitivity classification for operations."""
    LOW = "low"              # Non-critical, informational
    MEDIUM = "medium"        # Requires disclosure
    HIGH = "high"            # Requires explicit approval
    CRITICAL = "critical"    # Requires explicit user approval


# Operation classification map
OPERATION_SENSITIVITY: Dict[str, SensitivityLevel] = {
    # File operations
    "write_system_file": SensitivityLevel.CRITICAL,
    "delete_file": SensitivityLevel.HIGH,
    "write_file_downloads": SensitivityLevel.MEDIUM,
    "write_file_home": SensitivityLevel.MEDIUM,
    "read_system_file": SensitivityLevel.HIGH,
    
    # Network
    "http_request": SensitivityLevel.MEDIUM,
    "api_call": SensitivityLevel.MEDIUM,
    "web_search": SensitivityLevel.LOW,
    "sovereign_browser": SensitivityLevel.LOW,
    
    # System
    "execute_command": SensitivityLevel.CRITICAL,
    "modify_config": SensitivityLevel.CRITICAL,
    
    # Memory
    "write_memory": SensitivityLevel.MEDIUM,
    "erase_memory": SensitivityLevel.HIGH,
    
    # Background tasks
    "spawn_background_task": SensitivityLevel.MEDIUM,
    "long_running_task": SensitivityLevel.MEDIUM,
}


class ConsentWorkflow:
    """
    Manages user consent for sensitive operations.
    
    Features:
    - Operation sensitivity classification
    - User-facing consent prompts
    - Approved operations tracking
    - Denial logging
    - Consent memorization (don't ask twice for same operation)
    """
    
    def __init__(self):
        self.approved_operations: Dict[str, bool] = {}
        self.user_consent_callback: Optional[Callable] = None
        self.auto_approve_callback: Optional[Callable] = None
        
    def set_user_consent_handler(self, handler: Callable[[str, Dict[str, Any]], asyncio.Future]):
        """
        Set a callback to handle user consent requests.
        
        Handler should return a Future that resolves to True (approve) or False (deny).
        """
        self.user_consent_callback = handler
    
    def set_auto_approve_handler(self, handler: Callable[[str], bool]):
        """
        Set a callback to determine if an operation can be auto-approved.
        
        This allows dev mode or trusted workflows to auto-approve certain operations.
        """
        self.auto_approve_callback = handler
    
    async def check_consent(self, operation: str, details: Dict[str, Any] = None) -> bool:
        """
        Check if an operation requires and has consent.
        
        Returns: True if approved, False if denied or needs approval.
        """
        details = details or {}
        sensitivity = OPERATION_SENSITIVITY.get(operation, SensitivityLevel.LOW)
        
        # Check cache
        cache_key = f"{operation}:{details.get('path', details.get('url', ''))}"
        if cache_key in self.approved_operations:
            return self.approved_operations[cache_key]
        
        # Low sensitivity operations don't need consent
        if sensitivity == SensitivityLevel.LOW:
            return True
        
        # Check auto-approve
        if self.auto_approve_callback and self.auto_approve_callback(operation):
            logger.info("✓ Auto-approved: %s", operation)
            self.approved_operations[cache_key] = True
            return True
        
        # For high/critical, require explicit user consent
        if sensitivity in {SensitivityLevel.HIGH, SensitivityLevel.CRITICAL}:
            if not self.user_consent_callback:
                logger.warning("⚠️  Blocking %s (no consent handler)", operation)
                return False
            
            logger.warning("🔐 Requesting user consent for: %s", operation)
            approved = await self.user_consent_callback(operation, details)
            
            self.approved_operations[cache_key] = approved
            if approved:
                logger.info("✓ User approved: %s", operation)
            else:
                logger.warning("✗ User denied: %s", operation)
            
            return approved
        
        # Medium sensitivity: log and proceed (log for audit trail)
        logger.info("📋 Medium sensitivity operation: %s (details: %s)", operation, details)
        self.approved_operations[cache_key] = True
        return True


# Global workflow instance
_consent_workflow: Optional[ConsentWorkflow] = None


def get_consent_workflow() -> ConsentWorkflow:
    """Get or create the global consent workflow instance."""
    global _consent_workflow
    if _consent_workflow is None:
        _consent_workflow = ConsentWorkflow()
    return _consent_workflow
