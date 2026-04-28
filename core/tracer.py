from core.runtime.errors import record_degradation
import json
import logging
import os
import time
import uuid
from typing import Any

logger = logging.getLogger("Kernel.Tracer")

class Tracer:
    def __init__(self, trace_file: str = "autonomy_engine/data/traces.jsonl") -> None:
        self.trace_file = trace_file
        os.makedirs(os.path.dirname(self.trace_file), exist_ok=True)
        self.current_trace: dict[str, Any] | None = None

    def start_trace(self, goal: dict[str, Any]) -> None:
        """Starts a new trace session."""
        self.current_trace = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "goal": goal,
            "steps": [],
            "outcome": None,
            "latency": 0
        }
        
    def log_step(self, step_type: str, content: Any) -> None:
        """Logs a reasoning or execution step."""
        if self.current_trace:
            self.current_trace["steps"].append({
                "type": step_type,
                "t": time.time(),
                "content": content
            })

    def end_trace(self, outcome: Any) -> None:
        """Finalizes the trace and writes to disk."""
        if self.current_trace:
            self.current_trace["outcome"] = outcome
            self.current_trace["latency"] = time.time() - self.current_trace["timestamp"]
            
            try:
                with open(self.trace_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(self.current_trace) + "\n")
            except Exception as e:
                record_degradation('tracer', e)
                logger.error("Failed to write trace: %s", e)
            
            self.current_trace = None
