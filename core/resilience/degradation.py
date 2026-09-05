import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

logger = logging.getLogger("System.Degradation")

class SystemState(Enum):
    HEALTHY = auto()         # All systems nominal
    DEGRADED = auto()        # Non-critical systems failed (e.g., vision, web_search)
    SAFE_MODE = auto()       # Core cognitive failure (LLM issues), fallback to rules
    CRITICAL = auto()        # System instability, data corruption risk
    EMERGENCY = auto()       # Immediate shutdown or isolation required

class FailureType(Enum):
    LLM_API_ERROR = auto()
    LLM_HALLUCINATION = auto() # Detected via consistency check
    MEMORY_Corruption = auto()
    SKILL_FAILURE = auto()
    RESOURCE_EXHAUSTION = auto()
    CONNECTION_LOST = auto()

@dataclass
class FailureEvent:
    type: FailureType
    component: str
    error_msg: str
    timestamp: float = field(default_factory=time.time)
    severity: float = 0.5 # 0.0 - 1.0

class DegradationManager:
    """Manages system health state and enforces degradation policies.
    Replacing binary success/failure with graceful fallback.
    """
    
    # Number of consecutive healthy checks required before auto-recovery
    RECOVERY_THRESHOLD = 5
    # Failures within this window (seconds) count toward current state evaluation
    FAILURE_WINDOW = 300  # 5 minutes

    def __init__(self):
        self.current_state = SystemState.HEALTHY
        self.failure_history: List[FailureEvent] = []
        self.consecutive_healthy_checks = 0
        self.degraded = False  # Flag other subsystems can inspect
        self.capabilities = {
            "cognition": True,
            "memory_write": True,
            "memory_read": True,
            "tool_execution": True,
            "complex_planning": True
        }
        
    def report_failure(self, failure: FailureEvent):
        """Report a failure and adjust system state."""
        self.failure_history.append(failure)
        logger.warning("⚠️ Failure Reported: %s in %s (Severity: %.2f)", failure.type.name, failure.component, failure.severity)
        
        self._evaluate_state(failure)
        
    def _evaluate_state(self, latest_failure: FailureEvent):
        """Determine new system state based on failures."""
        # LLM Failures -> Safe Mode
        if latest_failure.type in [FailureType.LLM_API_ERROR, FailureType.LLM_HALLUCINATION]:
            if latest_failure.severity > 0.7:
                self._transition_to(SystemState.SAFE_MODE, "Critical Cognitive Failure")
        
        # Memory Failures -> Read-Only or Critical
        elif latest_failure.type == FailureType.MEMORY_Corruption:
            self._transition_to(SystemState.CRITICAL, "Memory Integrity Compromised")
            self.capabilities["memory_write"] = False
            
        # Skill Failures -> Degraded
        elif latest_failure.type == FailureType.SKILL_FAILURE:
            self._transition_to(SystemState.DEGRADED, f"Skill {latest_failure.component} Failed")
            
    def _transition_to(self, new_state: SystemState, reason: str):
        if new_state == self.current_state:
            return
            
        # Priority check: Don't upgrade state automatically (requires manual or heuristic recovery)
        # Actually, we might want to downgrade freely.
        
        logger.warning("📉 State Transition: %s -> %s | Reason: %s", self.current_state.name, new_state.name, reason)
        self.current_state = new_state
        self._enforce_policy(new_state)
        
    def _enforce_policy(self, state: SystemState):
        """Apply capability locks based on state."""
        if state == SystemState.HEALTHY:
            self.capabilities = {k: True for k in self.capabilities}
            self.degraded = False
            
        elif state == SystemState.DEGRADED:
            # Keep core running but disable complex planning and flag degradation
            self.capabilities["complex_planning"] = False
            self.degraded = True
            logger.warning("⚠️ DEGRADED MODE: Disabling complex planning. Subsystems should check 'degraded' flag.")
            
        elif state == SystemState.SAFE_MODE:
            self.capabilities["complex_planning"] = False
            self.capabilities["tool_execution"] = False # Only basic replies
            logger.info("🛡️ SAFE MODE ENGAGED: Disabling tools and planning.")
            
        elif state == SystemState.CRITICAL:
            self.capabilities["memory_write"] = False
            self.capabilities["tool_execution"] = False
            logger.critical("🚨 CRITICAL STATE: Memory Write-Lock Engaged.")

    def can_perform(self, action_type: str) -> bool:
        """Check if action is allowed in current state."""
        return self.capabilities.get(action_type, False)

    def check_health(self):
        """Periodic health check — call this on every tick/cycle.

        If the system is in a degraded or critical state, evaluate whether
        the triggering conditions have cleared.  After RECOVERY_THRESHOLD
        consecutive healthy checks the state is upgraded one level toward
        HEALTHY.
        """
        if self.current_state == SystemState.HEALTHY:
            self.consecutive_healthy_checks = 0
            return

        # States above CRITICAL (EMERGENCY) require manual intervention
        if self.current_state == SystemState.EMERGENCY:
            self.consecutive_healthy_checks = 0
            return

        # Check recent failures within the sliding window
        now = time.time()
        recent_failures = [
            f for f in self.failure_history
            if now - f.timestamp < self.FAILURE_WINDOW
        ]

        if recent_failures:
            # Conditions have NOT cleared — reset recovery counter
            self.consecutive_healthy_checks = 0
            return

        # No recent failures — increment recovery counter
        self.consecutive_healthy_checks += 1
        logger.info(
            "Health check passed (%d/%d) in state %s",
            self.consecutive_healthy_checks,
            self.RECOVERY_THRESHOLD,
            self.current_state.name,
        )

        if self.consecutive_healthy_checks >= self.RECOVERY_THRESHOLD:
            self._auto_recover()

    def _auto_recover(self):
        """Step the system one level closer to HEALTHY."""
        self.consecutive_healthy_checks = 0

        recovery_path = {
            SystemState.CRITICAL: SystemState.SAFE_MODE,
            SystemState.SAFE_MODE: SystemState.DEGRADED,
            SystemState.DEGRADED: SystemState.HEALTHY,
        }

        next_state = recovery_path.get(self.current_state)
        if next_state is None:
            return

        logger.info(
            "Auto-recovery: %s -> %s after %d consecutive healthy checks",
            self.current_state.name,
            next_state.name,
            self.RECOVERY_THRESHOLD,
        )
        self.current_state = next_state
        self._enforce_policy(next_state)

    def recover(self, component: str = "all"):
        """Force-recover state (manual / programmatic trigger)."""
        logger.info("Manual recovery requested for %s...", component)
        self.consecutive_healthy_checks = 0
        self.current_state = SystemState.HEALTHY
        self._enforce_policy(SystemState.HEALTHY)

# Global Instance
degradation_manager = DegradationManager()