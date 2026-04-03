"""Dead Letter Queue (DLQ) Service
Captures and analyzes failed cognitive cycles.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("Resilience.DLQ")

class DeadLetterQueue:
    """Service to handle failed thought payloads and system blocks."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        from core.config import config
        self.storage_path = storage_path or config.paths.data_dir / "dlq.jsonl"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory failure patterns
        self.failure_counts: Dict[str, int] = {}
        self.last_failure: Optional[Dict[str, Any]] = None

    def capture_failure(
        self, 
        message: str, 
        context: Dict[str, Any], 
        error: Exception, 
        source: str = "orchestrator"
    ):
        """Log a failed cognitive payload for later analysis."""
        entry = {
            "timestamp": time.time(),
            "source": source,
            "error": str(error),
            "error_type": type(error).__name__,
            "message_snippet": message[:200] if isinstance(message, str) else "non-string-payload",
            "context_keys": list(context.keys()) if context else [],
        }
        
        # Update patterns
        err_key = entry["error_type"]
        self.failure_counts[err_key] = self.failure_counts.get(err_key, 0) + 1
        self.last_failure = entry

        # Patch 11: Robust Atomic Write
        try:
            import os
            # Use encoding for cross-platform reliability
            with open(self.storage_path, "a", encoding="utf-8") as f:
                # On many OSs/filesystems, a single write call <= PIPE_BUF is atomic
                # for appends. But for JSONL, we ensure each entry is a single line.
                line = json.dumps(entry) + "\n"
                f.write(line)
                # Ensure it's flushed to disk physically
                f.flush()
                os.fsync(f.fileno())
            logger.info("💀 DLQ: Captured cognitive failure (%s)", err_key)
        except Exception as e:
            logger.error("Failed to write to DLQ: %s", e)

    def get_failure_report(self) -> Dict[str, Any]:
        """Get summary of recent failures."""
        return {
            "total_captured": sum(self.failure_counts.values()),
            "pattern_distribution": self.failure_counts,
            "last_failure": self.last_failure
        }

    def clear(self):
        """Reset the DLQ stats."""
        self.failure_counts = {}
        self.last_failure = None