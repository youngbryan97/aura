"""core/state_manager.py
Atomic snapshots and state restoration.
"""
from core.runtime.errors import record_degradation
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Core.StateManager")

class StateManager:
    def __init__(self, persist_dir: Optional[Path] = None):
        from core.config import config
        self.persist_dir = persist_dir or (config.paths.data_dir / "state")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_file = self.persist_dir / "latest_snapshot.json"
        self.checkpoints: List[Dict] = []

    def create_snapshot(self, orchestrator: Any) -> Dict:
        """Captures the current state of the system into a serializable dict.
        """
        snapshot = {
            "timestamp": time.time(),
            "cycle_count": getattr(orchestrator, "cycle_count", 0),
            "state": str(getattr(orchestrator, "state", "UNKNOWN")),
            "skills": list(getattr(orchestrator, "skills", {}).keys()),
            "memory_stats": self._get_memory_stats(orchestrator),

        }
        
        try:
            with open(self.snapshot_file, "w") as f:
                json.dump(snapshot, f, indent=2)
            logger.debug("StateManager: Snapshot saved.")
            return snapshot
        except Exception as e:
            record_degradation('state_manager', e)
            logger.error("StateManager: Failed to save snapshot: %s", e)
            return {}

    def _get_memory_stats(self, orchestrator: Any) -> Dict[str, Any]:
        """Safely extract memory stats from the orchestrator's memory subsystem."""
        try:
            memory = getattr(orchestrator, "memory", None)
            if memory is None:
                return {"status": "unavailable"}
            # Try common memory interfaces
            if hasattr(memory, "get_stats"):
                return memory.get_stats()
            if hasattr(memory, "count"):
                return {"items": memory.count(), "status": "ok"}
            # Fall back to checking for known sub-stores
            stats: Dict[str, Any] = {"status": "ok"}
            for store_name in ("episodic", "semantic", "vector"):
                store = getattr(memory, store_name, None)
                if store is not None:
                    count = getattr(store, "count", lambda: "attached")()
                    stats[store_name] = count
            return stats if len(stats) > 1 else {"status": "no_stores"}
        except Exception as e:
            record_degradation('state_manager', e)
            logger.debug("Could not read memory stats: %s", e)
            return {"status": "error", "detail": str(e)}

    def push_checkpoint(self, state: Dict):
        self.checkpoints.append(state)
        if len(self.checkpoints) > 10:
            self.checkpoints.pop(0)

    def rollback(self) -> Optional[Dict]:
        if not self.checkpoints:
            logger.warning("StateManager: No checkpoints available for rollback.")
            return None
        
        last_state = self.checkpoints.pop()
        logger.info("StateManager: Rolling back to checkpoint from %s", last_state.get('timestamp'))
        return last_state

