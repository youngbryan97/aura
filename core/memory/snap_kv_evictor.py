"""
core/memory/snap_kv_evictor.py
==============================
CORTANA'S SNAPKV EVICTOR

Implements attention-based KV cache management.
Derived from the SnapKV research (efficient KV cache compression).
When memory pressure is high, Cortana identifies the most important 
attention heads and evicts the rest, preserving cognitive coherence 
while reducing physical memory footprint.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("Aura.SnapKV")

class SnapKVEvictor:
    """
    Manages KV cache health by evicting low-importance tokens.

    Core Memory Lock:
    Tokens/entries tagged via mark_immutable() are permanently excluded from
    eviction passes.  Use this for memories anchored to high-arousal/high-valence
    events or explicit user alignment corrections — losing these would cause
    Identity Drift where Aura retains behavioural heuristics but loses the
    episodic context that grounds them.
    """

    def __init__(self, memory_limit_gb: float = 24.0):  # 64GB system — 32B model needs KV headroom
        self.limit = memory_limit_gb
        self.token_importance_map: Dict[int, float] = {}
        self._immutable_indices: set = set()  # Core Memory Lock — never evicted
        logger.info("🧠 SnapKVEvictor initialized. Limit: %.1f GB", self.limit)

    def mark_immutable(self, token_indices: List[int]) -> None:
        """Lock token indices against eviction.

        Call this for memories associated with:
        - Affective spikes (high valence × high arousal)
        - Explicit user alignment corrections
        - Foundational identity-anchoring interactions

        These entries are excluded from all LRU/importance-based eviction passes,
        preventing the Identity Drift that accumulates when Aura loses the episodic
        context behind her behavioural patterns.
        """
        before = len(self._immutable_indices)
        self._immutable_indices.update(token_indices)
        added = len(self._immutable_indices) - before
        if added:
            logger.info("🔒 SnapKV: Locked %d token(s) as Core Memory (total locked: %d).",
                        added, len(self._immutable_indices))

    def unmark_immutable(self, token_indices: List[int]) -> None:
        """Release a Core Memory lock (requires explicit call — never done automatically)."""
        self._immutable_indices.difference_update(token_indices)

    def is_protected(self, token_index: int) -> bool:
        """Return True if this token index is locked against eviction."""
        return token_index in self._immutable_indices

    def calculate_eviction_targets(
        self, 
        attention_scores: np.ndarray, 
        tokens: List[str], 
        target_reduction: float = 0.2
    ) -> List[int]:
        """
        Identifies which tokens to evict based on attention importance.
        score shape: (num_layers, num_heads, seq_len, seq_len)
        """
        # Average attention across layers and heads
        avg_attention = np.mean(attention_scores, axis=(0, 1))
        # Sum attention received by each token (as a destination)
        token_importance = np.sum(avg_attention, axis=0)
        
        # Identify indices of tokens to remove
        num_to_evict = int(len(tokens) * target_reduction)
        candidate_indices = np.argsort(token_importance)

        # Core Memory Lock: skip any token index that is marked immutable
        evict_indices = [
            int(i) for i in candidate_indices
            if int(i) not in self._immutable_indices
        ][:num_to_evict]

        protected_skipped = num_to_evict - len(evict_indices)
        if protected_skipped:
            logger.info(
                "🔒 SnapKV: Skipped %d protected Core Memory token(s) during eviction pass.",
                protected_skipped,
            )

        logger.info("🧠 Cortana: Evicting %d tokens from cache to maintain health", len(evict_indices))
        return evict_indices

    def check_memory_pressure(self, current_gb: float) -> bool:
        """Determines if eviction is urgent."""
        return current_gb > (self.limit * 0.85)

    def get_compressed_context(self, context: str, reduction_factor: float = 0.3) -> str:
        """Simplified textual context compression (surrogate for actual KV eviction)."""
        lines = context.split('\n')
        if len(lines) < 10: return context
        
        # Keep first 2 and last 5 lines, prune the middle
        mid_idx = int(len(lines) * reduction_factor)
        compressed = lines[:2] + ["... [Cortana: Context Compressed] ..."] + lines[mid_idx + 2:]
        return '\n'.join(compressed)
