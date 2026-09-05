"""core/memory/episode_store.py
Episode store driving autobiographical disk persistence.
"""
import json
import logging
from pathlib import Path
from typing import Any

from core.config import get_config
from core.memory.life_event import LifeEvent
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Memory.EpisodeStore")

_EPISODE_STORE_ERRORS = (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError)


class EpisodeStore:
    """Handles disk writing and querying of structured life events."""

    def __init__(self):
        cfg = get_config()
        self.db_path = Path(cfg.paths.memory_dir) / "autobiography.jsonl"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def save_event(self, event: LifeEvent) -> None:
        """Append event transaction to autobiographical log file (off-loop)."""
        try:
            await get_file_write_gateway().append_text_async(
                self.db_path,
                json.dumps(event.to_dict(), sort_keys=True) + "\n",
                source="memory.episode_store",
            )
        except _EPISODE_STORE_ERRORS as e:
            record_degradation("memory.episode_store.save", e)
            logger.error("Failed to persist life event: %s", e)

    async def load_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Loads the most recent N events from disk storage."""
        if not self.db_path.exists():
            return []
        
        events = []
        try:
            with self.db_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line.strip()))
        except _EPISODE_STORE_ERRORS as e:
            record_degradation("memory.episode_store.load", e)
            logger.error("Failed to read autobiographical logs: %s", e)

        return events[-limit:]
