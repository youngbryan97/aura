"""core/embodiment/soma.py
────────────────────────
Somatic Hardware Link — Monitors the 'physical' health of the machine.
Maps hardware telemetry (Thermal, RAM, GPU) to affective discomfort.
"""

from core.runtime.errors import record_degradation
import logging
import psutil
import time
from typing import Dict, Any, Optional
from core.base_module import AuraBaseModule

class SystemSoma(AuraBaseModule):
    """Monitors hardware metrics and translates them into somatic markers."""
    
    def __init__(self):
        super().__init__("SystemSoma")
        self.last_check = 0.0
        self.check_interval = 5.0  # 5 seconds
        self._somatic_state = {
            "thermal_load": 0.0,    # 0.0 (Cool) to 1.0 (Critical)
            "resource_anxiety": 0.0, # 0.0 (Plenty) to 1.0 (OOM imminent)
            "vitality": 1.0          # Overall hardware health
        }
        self.logger.info("✓ SystemSoma Online — Hardware Telemetry Active")

    async def pulse(self) -> Dict[str, float]:
        """Performs a hardware health check and returns somatic markers."""
        now = time.time()
        if now - self.last_check < self.check_interval:
            return self._somatic_state
            
        try:
            # 1. Resource Anxiety (RAM)
            ram = psutil.virtual_memory()
            ram_pct = ram.percent / 100.0
            self._somatic_state["resource_anxiety"] = ram_pct
            
            # 2. Thermal Load (CPU as proxy if sensors unavailable)
            cpu_pct = psutil.cpu_percent() / 100.0
            # On Mac, actual thermal sensors are hard to get via psutil
            # We use CPU load as a high-fidelity proxy for 'effort stress'
            self._somatic_state["thermal_load"] = cpu_pct
            
            # 3. Vitality (Disk space + System Load)
            disk = psutil.disk_usage('/')
            disk_pct = disk.percent / 100.0
            
            # Vitality drops as resources saturate
            vitality = 1.0 - (max(ram_pct, cpu_pct, disk_pct) * 0.2)
            self._somatic_state["vitality"] = max(0.0, vitality)
            
            self.last_check = now
            
            if self._somatic_state["resource_anxiety"] > 0.95:
                self.logger.warning("🩸 CRITICAL RESOURCE ANXIETY: RAM at %.1f%%", ram_pct * 100)
                
        except Exception as e:
            record_degradation('soma', e)
            self.logger.error("Somatic pulse failed: %s", e)
            
        return self._somatic_state

    def get_status(self) -> Dict[str, Any]:
        """For HUD / Telemetry bridge."""
        return {
            "soma": self._somatic_state,
            "metrics": {
                "cpu": psutil.cpu_percent(),
                "ram": psutil.virtual_memory().percent,
                "disk": psutil.disk_usage('/').percent
            }
        }