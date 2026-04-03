"""research/state_observatory.py

Advanced Research Metrology for Aura.
Logs internal cognitive metrics over time to correlate internal states 
(phi, coherence, affect) with task performance and reasoning quality.
"""

import time
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, List

logger = logging.getLogger("Aura.Research.Observatory")

class StateObservatory:
    """
    Logs and instruments internal metrics from UnifiedState and QualiaEngine.
    Used to study the relationship between consciousness proxies and task performance.
    """
    
    def __init__(self, registry, max_history: int = 5000, log_dir: Optional[Path] = None):
        self.registry = registry
        self.history = deque(maxlen=max_history)
        self.log_dir = log_dir or Path.home() / ".aura" / "research"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self.log_dir / "state_metrics.jsonl"
        
        logger.info("🔭 State Observatory initialized. Logging to %s", self.current_log_file)

    def snapshot(self, event_type: str, context_metadata: Optional[Dict[str, Any]] = None):
        """Captures a snapshot of the current unified state and correlates it with an event."""
        state = self.registry.get_state()
        
        record = {
            "time": time.time(),
            "event": event_type,
            "metrics": {
                "phi": getattr(state, "phi", 0.0),
                "coherence": getattr(state, "coherence", 1.0),
                "curiosity": getattr(state, "curiosity", 0.0),
                "loop_depth": getattr(state, "loop_depth", 0.0),
                "salience": getattr(state, "salience", 0.0),
                "valence": getattr(state, "valence", 0.0),
                "arousal": getattr(state, "arousal", 0.0),
                "energy": getattr(state, "energy", 1.0)
            },
            "metadata": context_metadata or {}
        }
        
        self.history.append(record)
        self._persist_record(record)
        logger.debug("📸 State Snapshot captured for event: %s", event_type)

    def _persist_record(self, record: Dict[str, Any]):
        """Append record to the JSONL log file."""
        try:
            with open(self.current_log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error("Failed to persist research record: %s", e)

    def export_history(self, path: Optional[Path] = None) -> Path:
        """Export the full history to a specific file."""
        target = path or self.log_dir / f"export_{int(time.time())}.json"
        with open(target, "w") as f:
            json.dump(list(self.history), f, indent=2)
        return target

    def get_recent_metrics(self, n: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the last N snapshots."""
        return list(self.history)[-n:]

# Singleton / Global access
_observatory = None

def get_observatory(registry=None) -> StateObservatory:
    global _observatory
    if _observatory is None:
        if registry is None:
            from core.state_registry import get_registry
            registry = get_registry()
        _observatory = StateObservatory(registry)
    return _observatory
