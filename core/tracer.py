import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List

logger = logging.getLogger("Kernel.Tracer")

class Tracer:
    def __init__(self, trace_file="autonomy_engine/data/traces.jsonl"):
        self.trace_file = trace_file
        os.makedirs(os.path.dirname(self.trace_file), exist_ok=True)
        self.current_trace = None

    def start_trace(self, goal: Dict[str, Any]):
        """Starts a new trace session."""
        self.current_trace = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "goal": goal,
            "steps": [],
            "outcome": None,
            "latency": 0
        }
        
    def log_step(self, step_type: str, content: Any):
        """Logs a reasoning or execution step."""
        if self.current_trace:
            self.current_trace["steps"].append({
                "type": step_type,
                "t": time.time(),
                "content": content
            })

    def end_trace(self, outcome: Any):
        """Finalizes the trace and writes to disk."""
        if self.current_trace:
            self.current_trace["outcome"] = outcome
            self.current_trace["latency"] = time.time() - self.current_trace["timestamp"]
            
            try:
                with open(self.trace_file, "a") as f:
                    f.write(json.dumps(self.current_trace) + "\n")
            except Exception as e:
                logger.error("Failed to write trace: %s", e)
            
            self.current_trace = None