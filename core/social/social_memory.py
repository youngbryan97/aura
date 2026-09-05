"""core/social/social_memory.py
Social memory store logging interactions.
"""
import time
from typing import Any


class SocialMemoryStore:
    """Logs human-agent interaction transcripts and sentiments."""

    def __init__(self):
        self._history: list[dict[str, Any]] = []

    def record_interaction(self, speaker: str, utterance: str, sentiment: float = 0.5) -> None:
        self._history.append({
            "timestamp": time.time(),
            "speaker": speaker,
            "utterance": utterance,
            "sentiment": sentiment
        })

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._history[-limit:]
