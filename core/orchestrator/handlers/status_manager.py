import logging
import time
import asyncio
from typing import Any, Dict
from core.config import config
from core.container import ServiceContainer
from core.runtime.organism_status import get_organism_status

logger = logging.getLogger(__name__)

def capture_and_log(e, meta):
    logger.error("Error in status manager: %s | Meta: %s", e, meta)

class StatusManagerMixin:
    """Mixin for status reporting and telemetry emission."""
    
    # Type hints for attributes provided by RobustOrchestrator
    status: Any
    start_time: float
    stats: Dict[str, Any]
    message_queue: Any
    reply_queue: Any
    liquid_state: Any
    _integrity_monitor: Any
    acceleration_factor: float
    singularity_threshold: bool

    def get_status(self) -> dict[str, Any]:
        """Provides a comprehensive status report using cached data where possible."""
        # [STABILITY] Recursion Guard
        if getattr(self, "_in_status_call", False):
            return {"status": "recursive_depth_guard", "healthy": False}
        self._in_status_call = True
        
        try:
            if not hasattr(self, "_cached_status") or self._cached_status is None:
                self._cached_status = {
                    "status": "operational",
                    "uptime": 0.0,
                    "stats": {},
                    "message_queue_size": 0,
                    "reply_queue_size": 0,
                    "initialized": getattr(self.status, "initialized", False),
                    "running": getattr(self.status, "running", False),
                    "cycle_count": getattr(self.status, "cycle_count", 0),
                    "healthy": True
                }

            self._cached_status["uptime"] = time.time() - self.start_time
            self._cached_status["stats"] = self.stats.copy()
            self._cached_status["message_queue_size"] = self.message_queue.qsize() if hasattr(self, "message_queue") else 0
            self._cached_status["reply_queue_size"] = self.reply_queue.qsize() if hasattr(self, "reply_queue") else 0
            
            status_report = self._cached_status.copy()
            status_report["config"] = config.model_dump() if hasattr(config, "model_dump") else {}

            if hasattr(self, "status") and self.status:
                if not isinstance(self.status, type) and hasattr(self.status, "model_dump"):
                    try:
                        # Health check before reporting
                        if hasattr(self, "health_check"):
                            self.health_check()
                        m_dump = self.status.model_dump()
                        status_report.update(m_dump)
                        status_report["status"] = m_dump 
                        for key in ("initialized", "running", "cycle_count", "is_processing", "mode", "skills_loaded"):
                            if key not in status_report["status"]:
                                status_report["status"][key] = getattr(self.status, key, True if key in ("initialized", "running") else (0 if key == "cycle_count" else (False if key == "is_processing" else (0 if key == "skills_loaded" else "neutral"))))
                        
                        status_report["initialized"] = status_report["status"]["initialized"]
                        status_report["cycle_count"] = getattr(self.status, "cycle_count", status_report["status"].get("cycle_count", 0))
                    except Exception as e:
                        capture_and_log(e, {'module': __name__})
                else:
                    status_report["running"] = bool(getattr(self.status, "running", True))
                    status_report["initialized"] = bool(getattr(self.status, "initialized", True))
                    status_report["cycle_count"] = int(getattr(self.status, "cycle_count", 0))
                    raw_sk = getattr(self.status, "skills_loaded", 0)
                    status_report["status"] = {
                        "running": status_report["running"],
                        "initialized": status_report["initialized"],
                        "cycle_count": status_report["cycle_count"],
                        "is_processing": bool(getattr(self.status, "is_processing", False)),
                        "mode": getattr(self.status, "mode", "neutral"),
                        "skills_loaded": raw_sk
                    }
                    if hasattr(self, "health_check"):
                        status_report["healthy"] = self.health_check()

            try:
                evidence = ServiceContainer.get("consciousness_evidence", default=None)
                if evidence and hasattr(evidence, "snapshot"):
                    status_report["consciousness_evidence"] = evidence.snapshot()
            except Exception as exc:
                logger.debug("Consciousness evidence unavailable for status: %s", exc)

            try:
                executive_closure = ServiceContainer.get("executive_closure", default=None)
                if executive_closure and hasattr(executive_closure, "get_status"):
                    status_report["executive_closure"] = executive_closure.get_status()
            except Exception as exc:
                logger.debug("Executive closure unavailable for status: %s", exc)

            try:
                executive_authority = ServiceContainer.get("executive_authority", default=None)
                if executive_authority and hasattr(executive_authority, "get_status"):
                    status_report["executive_authority"] = executive_authority.get_status()
            except Exception as exc:
                logger.debug("Executive authority unavailable for status: %s", exc)

            try:
                status_report["organism"] = get_organism_status(self)
            except Exception as exc:
                logger.debug("Organism status unavailable for status report: %s", exc)

            return status_report

        finally:
            self._in_status_call = False

    def _emit_telemetry_pulse(self):
        """Emit real-time liquid state telemetry."""
        try:
            ls = getattr(self, "liquid_state", None)
            if ls:
                ls_status = ls.get_status()
                monitor_stats = self._integrity_monitor.get_stats() if hasattr(self, '_integrity_monitor') else {}
                
                if hasattr(self, "_publish_telemetry"):
                    self._publish_telemetry({
                        "energy": ls_status.get("energy", 80),
                        "curiosity": ls_status.get("curiosity", 50),
                        "frustration": ls_status.get("frustration", 0),
                        "confidence": ls_status.get("focus", 50),
                        "mood": ls_status.get("mood", "NEUTRAL"),
                        "acceleration_factor": getattr(self.status, "acceleration_factor", 1.0),
                        "singularity_active": getattr(self.status, "singularity_threshold", 0.0),
                        "cpu_percent": monitor_stats.get("cpu_percent", 0),
                        "memory_mb": monitor_stats.get("memory_mb", 0),
                        "link_thickness": 5.0
                    })
        except Exception as exc:
            logger.error("Telemetry pulse failure: %s", exc)
            if hasattr(self, "_recover_from_stall"):
                from core.utils.task_tracker import get_task_tracker
                get_task_tracker().track(self._recover_from_stall())

    def _emit_telemetry(self, flow: str, text: str):
        """Helper to send updates to Thought Stream UI."""
        try:
            from ..thought_stream import get_emitter
            cycle = self.status.cycle_count if hasattr(self, 'status') else 0
            get_emitter().emit(flow, text, level="info", category="Cognition", cycle=cycle)
        except Exception as e:
            logger.debug("Telemetry emit failed: %s", e)
