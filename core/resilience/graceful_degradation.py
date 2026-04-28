# core/graceful_degradation.py
"""Graceful Degradation System
Tracks component health and adjusts capabilities when components fail.
Allows Aura to continue operating even when some subsystems are unavailable.
"""
from core.runtime.errors import record_degradation
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set

logger = logging.getLogger("Aura.GracefulDegradation")


class ComponentStatus(Enum):
    """Component health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class ComponentState:
    """State of a single component"""

    name: str
    status: ComponentStatus = ComponentStatus.HEALTHY
    last_error: Optional[str] = None
    failure_count: int = 0
    last_check: datetime = field(default_factory=datetime.now)
    recovery_attempts: int = 0
    fallback_active: bool = False


class GracefulDegradationManager:
    """Manages component health and enables graceful degradation.
    
    When a component fails:
    1. Records the failure
    2. Activates fallback if available
    3. Adjusts system capabilities
    4. Periodically attempts recovery
    """
    
    # Component categories and their criticality
    CRITICAL_COMPONENTS = {"cognitive_engine", "memory", "skill_router"}
    IMPORTANT_COMPONENTS = {"llm_client", "planner", "executor"}
    OPTIONAL_COMPONENTS = {"vision", "hearing", "speech", "browser"}
    
    def __init__(self):
        self.components: Dict[str, ComponentState] = {}
        self.fallbacks: Dict[str, Callable] = {}
        self.capability_adjustments: Dict[str, bool] = {}
        self._failed_components: Set[str] = set()
        
    def register_component(self, name: str, 
                          fallback: Optional[Callable] = None,
                          initial_status: ComponentStatus = ComponentStatus.HEALTHY):
        """Register a component for health tracking"""
        self.components[name] = ComponentState(
            name=name,
            status=initial_status
        )
        if fallback:
            self.fallbacks[name] = fallback
        logger.debug("Registered component: %s", name)
    
    def report_failure(self, component: str, error: str):
        """Report a component failure"""
        if component not in self.components:
            self.register_component(component, initial_status=ComponentStatus.FAILED)
        
        state = self.components[component]
        state.status = ComponentStatus.FAILED
        state.last_error = error
        state.failure_count += 1
        state.last_check = datetime.now()
        
        self._failed_components.add(component)
        
        # Log appropriately based on criticality
        if component in self.CRITICAL_COMPONENTS:
            logger.critical("CRITICAL component FAILED: %s - %s", component, error)
        elif component in self.IMPORTANT_COMPONENTS:
            logger.error("Important component FAILED: %s - %s", component, error)
        else:
            logger.warning("Optional component FAILED: %s - %s", component, error)
        
        # Activate fallback if available
        if component in self.fallbacks:
            try:
                self.fallbacks[component]()
                state.fallback_active = True
                state.status = ComponentStatus.DEGRADED
                logger.info("Activated fallback for %s", component)
            except Exception as e:
                record_degradation('graceful_degradation', e)
                logger.error("Fallback for %s also failed: %s", component, e)
        
        # Adjust capabilities
        self._adjust_capabilities()
    
    def report_recovery(self, component: str):
        """Report that a component has recovered"""
        if component in self.components:
            state = self.components[component]
            state.status = ComponentStatus.HEALTHY
            state.fallback_active = False
            state.recovery_attempts = 0
            self._failed_components.discard(component)
            logger.info("Component recovered: %s", component)
            self._adjust_capabilities()
    
    def _adjust_capabilities(self):
        """Adjust system capabilities based on component health"""
        # Determine what's available
        self.capability_adjustments = {
            "can_reason": self._is_available("cognitive_engine"),
            "can_remember": self._is_available("memory"),
            "can_execute_skills": self._is_available("skill_router"),
            "can_call_llm": self._is_available("llm_client"),
            "can_plan": self._is_available("planner"),
            "can_see": self._is_available("vision"),
            "can_hear": self._is_available("hearing"),
            "can_speak": self._is_available("speech"),
            "can_browse": self._is_available("browser"),
        }
        
        # Log capability summary
        disabled = [k for k, v in self.capability_adjustments.items() if not v]
        if disabled:
            logger.warning("Capabilities disabled: %s", ', '.join(disabled))
    
    def _is_available(self, component: str) -> bool:
        """Check if a component is available (healthy or degraded)"""
        if component not in self.components:
            return True  # Assume available if not tracked
        return self.components[component].status in [
            ComponentStatus.HEALTHY, 
            ComponentStatus.DEGRADED
        ]
    
    def can(self, capability: str) -> bool:
        """Check if a capability is available"""
        return self.capability_adjustments.get(capability, True)
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of system health"""
        return {
            "healthy_count": sum(1 for c in self.components.values() 
                                if c.status == ComponentStatus.HEALTHY),
            "degraded_count": sum(1 for c in self.components.values() 
                                 if c.status == ComponentStatus.DEGRADED),
            "failed_count": sum(1 for c in self.components.values() 
                               if c.status == ComponentStatus.FAILED),
            "failed_components": list(self._failed_components),
            "capabilities": self.capability_adjustments,
            "overall_status": self._get_overall_status()
        }
    
    def _get_overall_status(self) -> str:
        """Determine overall system status"""
        if any(c in self._failed_components for c in self.CRITICAL_COMPONENTS):
            return "CRITICAL"
        elif any(c in self._failed_components for c in self.IMPORTANT_COMPONENTS):
            return "DEGRADED"
        elif self._failed_components:
            return "IMPAIRED"
        else:
            return "HEALTHY"


# Singleton
_degradation_manager: Optional[GracefulDegradationManager] = None


def get_degradation_manager() -> GracefulDegradationManager:
    """Get or create the global degradation manager"""
    global _degradation_manager
    if _degradation_manager is None:
        _degradation_manager = GracefulDegradationManager()
    return _degradation_manager


def safe_init(factory: Callable, component_name: str, 
              fallback: Any = None) -> Any:
    """Safe initialization wrapper with automatic degradation tracking.
    
    Usage:
        client = safe_init(lambda: LocalBrain(), "llm_client", fallback=MockClient())
    """
    manager = get_degradation_manager()
    try:
        result = factory()
        manager.register_component(component_name, initial_status=ComponentStatus.HEALTHY)
        return result
    except Exception as e:
        record_degradation('graceful_degradation', e)
        manager.report_failure(component_name, str(e))
        logger.warning("Using fallback for %s: %s", component_name, type(fallback).__name__)
        return fallback


def require_capability(capability: str):
    """Decorator that checks capability before executing.
    If capability unavailable, returns graceful error dict.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_degradation_manager()
            if not manager.can(capability):
                return {
                    "ok": False,
                    "error": "capability_unavailable",
                    "capability": capability,
                    "message": f"Required capability '{capability}' is not currently available"
                }
            return func(*args, **kwargs)
        return wrapper
    return decorator