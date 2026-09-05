"""Dead Letter Queue (DLQ) Service
Captures and analyzes failed cognitive cycles.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Resilience.DLQ")

class DeadLetterQueue:
    """Service to handle failed thought payloads and system blocks."""
    
    def __init__(self, storage_path: Path | None = None):
        from core.config import config
        self.storage_path = storage_path or config.paths.data_dir / "dlq.jsonl"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory failure patterns
        self.failure_counts: dict[str, int] = {}
        self.last_failure: dict[str, Any] | None = None

    def capture_failure(
        self, 
        message: str, 
        context: dict[str, Any], 
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
            get_file_write_gateway().append_text(
                self.storage_path,
                json.dumps(entry) + "\n",
                source="resilience.dlq.failure",
            )
            logger.info("💀 DLQ: Captured cognitive failure (%s)", err_key)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('dlq_service', e)
            logger.error("Failed to write to DLQ: %s", e)

    def get_failure_report(self) -> dict[str, Any]:
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
