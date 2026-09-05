"""core/workspace/thought_lifecycle.py
Thought Lifecycle tracker managing compilation, update, and eviction of thought nodes.
"""
from typing import Dict, Any, List
import time


class ThoughtLifecycle:
    """Manages active thought nodes inside the workspace."""

    def __init__(self):
        self._thoughts: List[Dict[str, Any]] = []

    def spawn_thought(self, node_id: str, content: str) -> None:
        self._thoughts.append({
            "id": node_id,
            "content": content,
            "spawned_at": time.time(),
            "last_updated": time.time()
        })

    def update_thought(self, node_id: str, content: str) -> None:
        for t in self._thoughts:
            if t["id"] == node_id:
                t["content"] = content
                t["last_updated"] = time.time()

    def evict_stale_thoughts(self, max_age_s: float = 300.0) -> None:
        cutoff = time.time() - max_age_s
        self._thoughts = [t for t in self._thoughts if t["last_updated"] > cutoff]

    def list_active_thoughts(self) -> List[Dict[str, Any]]:
        return self._thoughts
