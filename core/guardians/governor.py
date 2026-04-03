import asyncio
from collections import deque
import logging
from enum import Enum
import time

logger = logging.getLogger("Aura.Governor")

class OperationalMode(str, Enum):
    FULL_CAPABILITY = "FULL"
    DEGRADED_NO_PROACTIVE = "DEGRADED_NO_PROACTIVE" 
    DEGRADED_CORE_ONLY = "DEGRADED_CORE_ONLY"

class SystemGovernor:
    """
    The autonomic nervous system of Aura.
    Dynamically manages resources and forces graceful degradation under extreme load.
    """
    def __init__(self):
        self._health_metrics = {
            "llm_api_latency": deque(maxlen=50), 
            "db_latency": deque(maxlen=50)
        }
        self.current_mode = OperationalMode.FULL_CAPABILITY
        self._is_running = False
        
    async def start(self):
        self._is_running = True
        logger.info("🛡️ SystemGovernor online. Monitoring autonomic thresholds.")
        asyncio.create_task(self._health_check_loop())
        
    async def stop(self):
        self._is_running = False
        
    def record_llm_latency(self, latency_ms: float):
        self._health_metrics["llm_api_latency"].append(latency_ms)
        
    def record_db_latency(self, latency_ms: float):
        self._health_metrics["db_latency"].append(latency_ms)

    def _calculate_average(self, metric_name: str) -> float:
        history = self._health_metrics[metric_name]
        if not history:
            return 0.0
        return sum(history) / len(history)

    def get_operational_mode(self) -> OperationalMode:
        """Determines the system's current capability level based on health."""
        avg_llm_latency = self._calculate_average("llm_api_latency")
        avg_db_latency = self._calculate_average("db_latency")
        
        # Thresholds configured for local M1 LLM generation times
        if avg_llm_latency > 8000 or avg_db_latency > 1000: # 8s LLM, 1s DB
            new_mode = OperationalMode.DEGRADED_CORE_ONLY
        elif avg_llm_latency > 3000 or avg_db_latency > 500: # 3s LLM, 0.5s DB
            new_mode = OperationalMode.DEGRADED_NO_PROACTIVE
        else:
            new_mode = OperationalMode.FULL_CAPABILITY
            
        if new_mode != self.current_mode:
            self.current_mode = new_mode
            logger.warning(f"⚠️ SystemGovernor shifted Operational Mode: {self.current_mode.value}")
            
        return self.current_mode
        
    async def _health_check_loop(self):
        """Background loop to periodically evaluate System Mode."""
        while self._is_running:
            self.get_operational_mode() # Triggers the logic and logging
            await asyncio.sleep(10) # Check every 10 seconds
