from core.runtime.errors import record_degradation
import json
import logging
import os
import time
from typing import Any, Dict, List
from core.config import config

logger = logging.getLogger("Meta.CognitiveTrace")

class CognitiveTrace:
    """Records deep traces of Aura's reasoning process for auditing and debugging.
    """

    def __init__(self, trace_id: str = None):
        self.trace_id = trace_id or str(int(time.time()))
        self.steps: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.log_dir = str(config.paths.home_dir / "traces")
        os.makedirs(self.log_dir, exist_ok=True)

    def record_step(self, step_type: str, content: Any, metadata: Dict[str, Any] = None):
        """Record a single step in the reasoning chain."""
        self.steps.append({
            "type": step_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time() - self.start_time
        })

    def save(self):
        """Save the trace to disk."""
        filename = f"trace_{self.trace_id}.json"
        path = os.path.join(self.log_dir, filename)
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            # Trace persistence is internal audit bookkeeping; callers arrive
            # from arbitrary contexts (including bare threads), so the scope
            # is established here rather than assumed. Without it the live
            # runtime refused every save as a governance violation.
            with local_internal_governed_scope(
                "cognitive_trace.save",
                receipt_prefix="cognitive-trace-save",
            ):
                get_file_write_gateway().write_text(
                    path,
                    json.dumps({
                        "id": self.trace_id,
                        "duration": time.time() - self.start_time,
                        "steps": self.steps
                    }, indent=2),
                    source="cognitive_trace.save",
                )
            logger.info("Cognitive Trace saved: %s", path)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('cognitive_trace', e)
            logger.error("Failed to save trace: %s", e)

# Global instance for easy access if needed, or initialized per turn
