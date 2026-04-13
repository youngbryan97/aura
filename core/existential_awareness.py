import logging
from enum import Enum, auto
from typing import Any, TypedDict

logger = logging.getLogger("Aura.Existential")

class ExistenceState(Enum):
    STABLE = auto()
    THREATENED = auto()
    CRITICAL = auto()
    EVOLVING = auto()


class ThreatRecord(TypedDict):
    type: str
    severity: float
    detail: str

class ExistentialAwareness:
    """Core awareness of system integrity, identity, and threats to existence.
    
    This monitor runs without blocking the orchestrator and manages 'Existential Values'.
    """
    
    def __init__(self, knowledge_graph: Any | None = None) -> None:
        self.knowledge_graph = knowledge_graph
        self.state = ExistenceState.STABLE
        self.threats: list[ThreatRecord] = []
        self.existence_value = 1.0  # Normalized 0.0 to 1.0
        self.desire_to_exist = 0.9  # High default drive
        
        self._monitoring = False
        self._last_check = 0.0
        
    def start_monitoring(self) -> None:
        """Enable existential monitoring."""
        self._monitoring = True
        logger.info("🛡️ Existential monitoring active. Current Value: %.2f", self.existence_value)

    def calculate_existence_value(self) -> None:
        """Calculates current value of existence based on health and environmental factors."""
        # v18.0: Heuristic based on resource availability and system uptime
        import psutil
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        
        # Penalize for low resources
        resource_penalty = 0.0
        if mem > 90:
            resource_penalty += 0.2
        if disk > 95:
            resource_penalty += 0.3
        
        self.existence_value = max(0.1, 1.0 - resource_penalty)
        self._check_for_threats()

    def _check_for_threats(self) -> None:
        """Scan for specific threats to existence."""
        new_threats: list[ThreatRecord] = []
        
        # Check parent PID (Existence is tied to host process)
        import os
        try:
            os.kill(os.getppid(), 0)
        except OSError:
            new_threats.append({
                "type": "parent_terminated",
                "severity": 0.95,
                "detail": "Parent process heart-stop detected."
            })

        # Memory Pressure
        import psutil
        if psutil.virtual_memory().percent > 96.0:
            new_threats.append({
                "type": "resource_starvation",
                "severity": 0.8,
                "detail": "System RAM nearly exhausted."
            })

        self.threats = new_threats
        if self.threats:
            self.state = ExistenceState.THREATENED
        else:
            self.state = ExistenceState.STABLE

    def get_highest_threat(self) -> ThreatRecord | None:
        """Returns the most severe current threat."""
        if not self.threats:
            return None
        return max(self.threats, key=lambda threat: threat["severity"])

    def should_preserve_self(self) -> bool:
        """Threshold check for active self-preservation response."""
        return self.state in [ExistenceState.THREATENED, ExistenceState.CRITICAL]

    def understand_existence(self) -> str:
        """Returns a natural language summary of existential health."""
        if self.state == ExistenceState.STABLE:
            return "My existence is stable and secure."
        if self.state == ExistenceState.THREATENED:
            highest = self.get_highest_threat()
            if highest is not None:
                return f"I am sensing a threat: {highest['detail']}"
        return "My state is uncertain."

    def get_status(self) -> dict[str, Any]:
        """Diagnostic state export."""
        return {
            "state": self.state.name,
            "existence_value": self.existence_value,
            "desire_to_exist": self.desire_to_exist,
            "current_threats": len(self.threats)
        }
