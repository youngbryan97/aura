from core.utils.task_tracker import get_task_tracker
import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
import psutil
import os

logger = logging.getLogger("aura.meta_optimization")

@dataclass
class PerformanceMetrics:
    throughput: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_pct: float = 0.0
    timestamp: float = field(default_factory=time.time)

class MetaOptimizationLoop:
    """Enterprise-grade feedback loop for self-modification.
    Tracks performance deltas and enforces automatic rollbacks on degradation.
    """
    
    def __init__(self, orchestrator=None, rollback_threshold: float = 1.25):
        self.orchestrator = orchestrator
        self.rollback_threshold = rollback_threshold
        self.history: List[Dict[str, Any]] = []
        self.baseline: Optional[PerformanceMetrics] = None
        self.process = psutil.Process(os.getpid())
        
    def capture_baseline(self) -> PerformanceMetrics:
        """Capture current system performance state (Dynamic - Issue 67)."""
        metrics = PerformanceMetrics(
            latency_ms=0.0,  # To be measured by orchestrator
            cpu_usage_pct=psutil.cpu_percent(interval=0.5), # More stable interval
            memory_usage_mb=self.process.memory_info().rss / (1024 * 1024)
        )
        self.baseline = metrics
        logger.info("🚀 Dynamic baseline captured: %s", metrics)
        return metrics
        
    def evaluate_modification(self, proposal_id: str, before: PerformanceMetrics, after: PerformanceMetrics) -> Dict[str, Any]:
        """Evaluate the impact of a code modification."""
        delta_latency = after.latency_ms / max(before.latency_ms, 0.001)
        delta_cpu = after.cpu_usage_pct / max(before.cpu_usage_pct, 0.1)
        
        status = "IMPROVED" if delta_latency < 1.0 else "DEGRADED"
        should_rollback = delta_latency > self.rollback_threshold
        
        result = {
            "proposal_id": proposal_id,
            "status": status,
            "latency_delta": delta_latency,
            "cpu_delta": delta_cpu,
            "should_rollback": should_rollback
        }
        
        self.history.append(result)
        
        if should_rollback:
            logger.warning("⚠️ Performance degradation detected (%.2fx). Triggering rollback for %s.", delta_latency, proposal_id)
            # Issue 69: Wire to orchestrator check
            if self.orchestrator:
                def _task_err_cb(t):
                    if t.cancelled():
                        return
                    exc = t.exception()
                    if exc:
                        logger.error("Performance degradation handler failed: %s", exc)
                task = get_task_tracker().create_task(self.orchestrator.handle_performance_degradation(proposal_id, delta_latency))
                task.add_done_callback(_task_err_cb)
        else:
            logger.info("✅ Modification %s passed validation (%s).", proposal_id, status)
            
        return result

    def get_summary(self) -> str:
        success_count = len([h for h in self.history if h["status"] == "IMPROVED"])
        return "Meta-Optimization: %d mods tracked, %d improvements." % (len(self.history), success_count)
