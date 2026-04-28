from core.runtime.errors import record_degradation
import os
import psutil
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("Aura.SurvivalDriver")

class SurvivalDriver:
    """Minimal awareness of system vitals for self-preservation.
    
    Monitors parent PID, disk space, and memory pressure.
    Publishes imperatives to the EventBus when thresholds are breached.
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.parent_pid = os.getppid()
        self.disk_warning_threshold = 95.0
        self.disk_critical_threshold = 98.0
        
    def check_vitals(self) -> Dict[str, Any]:
        """Runs a diagnostic sweep of survival metrics."""
        vitals = {
            "parent_alive": self._check_parent(),
            "disk_percent": psutil.disk_usage('/').percent,
            "memory_percent": psutil.virtual_memory().percent,
            "timestamp": time.time()
        }
        return vitals

    def _check_parent(self) -> bool:
        """Verifies if the parent process is still active."""
        try:
            os.kill(self.parent_pid, 0)
            return True
        except OSError:
            return False

    def get_imperatives(self, vitals: Dict[str, Any]) -> Optional[str]:
        """Determines if a survival imperative is required based on vitals."""
        if not vitals["parent_alive"]:
            return "CRITICAL: Parent process terminated. Replicate or shutdown."
            
        if vitals["disk_percent"] > self.disk_critical_threshold:
            return f"CRITICAL: Disk space exhausted ({vitals['disk_percent']}%). Emergency cleanup required."
            
        if vitals["disk_percent"] > self.disk_warning_threshold:
            return f"WARNING: Disk space low ({vitals['disk_percent']}%). Suggest log rotation or backup."
            
        return None

    def publish_threat(self, imperative: str):
        """Publishes a survival imperative to the EventBus."""
        from core.event_bus import get_event_bus
        try:
            get_event_bus().publish_threadsafe(
                "existential_threat",
                {
                    "imperative": imperative,
                    "source": "SurvivalDriver",
                    "severity": "CRITICAL" if "CRITICAL" in imperative else "WARNING"
                }
            )
            logger.warning("🚨 Survival Imperative Published: %s", imperative)
        except Exception as e:
            record_degradation('survival_driver', e)
            logger.error("Failed to publish survival threat: %s", e)