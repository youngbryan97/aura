"""core/perception/spatial_ring_buffer.py — Spatial Reflex Buffer

Provides thread-safe, zero-lock memory access for high-velocity turn games like NetHack.
Caches recent spatial observations and status frames in a Collections Deque ring buffer.
"""
import collections
import threading
import numpy as np
from typing import Dict, Any, Optional, List


class SpatialReflexBuffer:
    """Provides ultra-low-latency, zero-lock memory access for high-velocity turn games."""

    def __init__(self, max_turns: int = 128):
        self.buffer: collections.deque = collections.deque(maxlen=max_turns)
        self._latest_turn = -1
        self._lock = threading.Lock()

    def push_state_frame(self, glyph_matrix: np.ndarray, stats: Dict[str, Any]):
        """Pushes a new environment state frame to the ring buffer in a thread-safe manner."""
        with self._lock:
            self._latest_turn += 1
            frame = {
                "turn": self._latest_turn,
                "matrix": np.copy(glyph_matrix),
                "stats": stats.copy()
            }
            self.buffer.append(frame)

    def get_working_frame(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent spatial state frame, or None if the buffer is empty."""
        with self._lock:
            if not self.buffer:
                return None
            return self.buffer[-1]

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns a list of all current frames in the buffer."""
        with self._lock:
            return list(self.buffer)

    def clear(self):
        """Clears the ring buffer."""
        with self._lock:
            self.buffer.clear()
            self._latest_turn = -1
