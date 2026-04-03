import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ThoughtTracer:
    """Deep Observability System.
    Records structured traces of the cognitive process for debugging and auditing.
    """
    
    def __init__(self, log_dir: str = "data/traces"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_trace_file = self.log_dir / f"trace_{int(time.time())}.jsonl"
        self.logger = logging.getLogger("System.ThoughtTracer")
        
    def log_cycle(self, 
                  objective: str, 
                  context: Dict[str, Any], 
                  thought: Dict[str, Any], 
                  outcome: Optional[str] = None):
        """Log a complete cognitive cycle.
        """
        entry = {
            "timestamp": time.time(),
            "iso_time": datetime.now().isoformat(),
            "objective": objective,
            "context_summary": list(context.keys()), # Avoid dumping huge context
            "thought": thought,
            "outcome": outcome
        }
        
        try:
            with open(self.current_trace_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.logger.error("Failed to write trace: %s", e)

    def log_event(self, event_type: str, details: Dict[str, Any]):
        """Log a discrete event (e.g., tool usage, state change)."""
        entry = {
            "timestamp": time.time(),
            "type": event_type,
            "details": details
        }
        try:
            with open(self.current_trace_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.logger.error("Failed to write event trace: %s", e)

# Global instance
tracer = ThoughtTracer()