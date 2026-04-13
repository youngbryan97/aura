"""core/senses/perceptual_buffer.py — The Continuous Sensorium

A thread-safe, ring-buffered registry for Aura's real-time sensory metadata.
Acts as the intermediate layer between raw hardware perception and cognitive agency.
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

logger = logging.getLogger("Aura.Senses.PerceptualBuffer")

@dataclass
class SensoryMoment:
    source: str
    content: Any
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

class PerceptualBuffer:
    """A rolling buffer of Aura's sensory experiences."""
    
    def __init__(self, maxsize: int = 100) -> None:
        self._buffer: deque[SensoryMoment] = deque(maxlen=maxsize)
        self._lock = Lock()
        logger.info("📡 Perceptual Buffer initialized (capacity: %d moments)", maxsize)

    def append(self, source: str, content: Any, metadata: dict[str, Any] | None = None) -> None:
        """Add a new sensory moment to the buffer."""
        moment = SensoryMoment(source=source, content=content, metadata=metadata or {})
        with self._lock:
            self._buffer.append(moment)

    def get_recent(self, seconds: float = 60.0) -> list[SensoryMoment]:
        """Retrieve moments from the last N seconds."""
        now = time.time()
        start_time = now - seconds
        with self._lock:
            return [m for m in self._buffer if m.timestamp > start_time]

    def get_summary(self, seconds: float = 60.0) -> str:
        """Generates a text summary of the recent sensorium for the Agency."""
        moments = self.get_recent(seconds)
        if not moments:
            return "No recent sensory input."
            
        summary_lines = []
        by_source: dict[str, list[SensoryMoment]] = {}
        for m in moments:
            if m.source not in by_source:
                by_source[m.source] = []
            by_source[m.source].append(m)
            
        for source, items in by_source.items():
            last_item = items[-1]
            count = len(items)
            time_ago = time.time() - last_item.timestamp
            summary_lines.append(f"- {source.upper()}: {last_item.content} (detected {count} times, latest {time_ago:.1f}s ago)")
            
        return "\n".join(summary_lines)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
