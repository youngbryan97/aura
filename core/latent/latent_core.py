import threading
from typing import Any, Dict, List

import numpy as np


class LatentCore:
    def __init__(self, dim: int = 2048):
        self._dim = dim
        # Initialize with random state to ensure uniqueness per instance
        self._vec = np.random.randn(dim).astype(np.float32)
        self._lock = threading.Lock()
        self._update_counter = 0

    def update(self, sensory_vector: np.ndarray, lr: float = 0.05):
        """Update latent state with simple gradient-free update: moving average.
        This represents the continuous 'stream of consciousness' at a mathematical level.
        """
        if sensory_vector is None:
            return

        # Ensure input matches dimension or project/truncate
        # Ideally, projection would happen before calling update, but we handle robustly here
        if sensory_vector.shape[0] != self._dim:
             # Very naive projection for robustness: resize or pad
             if sensory_vector.shape[0] > self._dim:
                 sensory_vector = sensory_vector[:self._dim]
             else:
                 sensory_vector = np.pad(sensory_vector, (0, self._dim - sensory_vector.shape[0]))

        with self._lock:
            # EMA Update: internal_state = (1-lr)*old + lr*input
            self._vec = (1 - lr) * self._vec + lr * sensory_vector
            self._update_counter += 1
            
            # Audit-52: Optimize normalization frequency (every 10 updates or high norm)
            if self._update_counter % 10 == 0:
                norm = np.linalg.norm(self._vec)
                if norm > 0 and (abs(norm - 1.0) > 0.05 or self._update_counter % 100 == 0):
                    self._vec = self._vec / norm

    def get_summary(self) -> Dict[str, Any]:
        """Returns only non-sensitive diagnostics (metadata), NEVER the raw vector.
        This summary is what is allowed to be logged/serialized.
        """
        with self._lock:
            norm = float(np.linalg.norm(self._vec))
            # Top-k indices act as a 'fingerprint' of the current state without revealing content
            topk_idx = list(np.argsort(-np.abs(self._vec))[:5].astype(int))
            
        return {
            "dim": self._dim, 
            "norm": round(norm, 4), 
            "topk_idx": topk_idx,
            "status": "active"
        }