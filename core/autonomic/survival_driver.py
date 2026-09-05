import logging
import os
import time
from typing import Any, Dict, Optional

from core.runtime import resource_psutil as psutil
from core.runtime.errors import record_degradation

from core.runtime.disk_budget import (
    DISK_AMBER_PERCENT,
    DISK_RED_PERCENT,
    state_volume_percent,
)

logger = logging.getLogger("Aura.SurvivalDriver")

class SurvivalDriver:
    """Minimal awareness of system vitals for self-preservation.
    
    Monitors parent PID, disk space, and memory pressure.
    Publishes imperatives to the EventBus when thresholds are breached.
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.parent_pid = os.getppid()
        # The amber/red lines are the system's, not this driver's: a reactive
        # check that disagrees with allostasis about when the disk is full
        # makes the two unable to be reasoned about together.
        self.disk_warning_threshold = DISK_AMBER_PERCENT
        self.disk_critical_threshold = DISK_RED_PERCENT
        
    def check_vitals(self) -> Dict[str, Any]:
        """Runs a diagnostic sweep of survival metrics."""
        vitals = {
            "parent_alive": self._check_parent(),
            "disk_percent": state_volume_percent(),
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
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('survival_driver', e)
            logger.error("Failed to publish survival threat: %s", e)