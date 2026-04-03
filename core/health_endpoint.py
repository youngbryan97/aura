import asyncio
import logging
import json
import time
from typing import Dict, Any
from .container import get_container

logger = logging.getLogger("Aura.Health")

class HealthAggregator:
    """Unified health aggregator for Aura components.
    
    Provides a machine-readable report of system health, dependencies,
    and performance metrics.
    """
    
    @classmethod
    async def get_report(cls) -> Dict[str, Any]:
        """Collect health data from the ServiceContainer and system state."""
        container = get_container()
        base_report = container.get_health_report()
        
        # Add system-level metrics
        report = {
            "timestamp": time.time(),
            "aura_status": base_report.get("status", "unknown"),
            "uptime": base_report.get("uptime_seconds", 0),
            "services": base_report.get("services", {}),
            "system": cls._get_system_metrics()
        }
        
        return report

    @classmethod
    def _get_system_metrics(cls) -> Dict[str, Any]:
        """Low-level resource stats."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        return {
            "memory_usage_mb": process.memory_info().rss / (1024 * 1024),
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads(),
            "pid": os.getpid()
        }

async def check_health_json() -> str:
    """Helper for web/terminal health checks."""
    report = await HealthAggregator.get_report()
    return json.dumps(report, indent=2)
