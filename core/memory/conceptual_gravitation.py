"""Conceptual Gravitation: related memories drift closer in embedding space.

Inspired by C.O.R.E.'s gravitational mechanism where related concepts have
their base vectors subtly modified to become more similar over time.

When two memories are frequently co-accessed (recalled in the same
conversation or within a short time window), their embeddings are nudged
toward each other. Over many dream cycles, this creates clusters of
semantically related memories that are faster to recall associatively.

This is NOT just cosine similarity — it physically modifies the stored
embeddings, creating emergent structure that wasn't in the original encoding.

Algorithm:
  1. Track co-access pairs: when memories A and B are both recalled in
     the same conversation turn, record (A, B) as a co-access event.
  2. During dream consolidation, for each co-access pair:
     - Compute direction: d = normalize(emb_B - emb_A)
     - Nudge both toward midpoint: emb_A += alpha * d, emb_B -= alpha * d
     - Alpha decays with distance (far memories attract less)
  3. Re-normalize embeddings after nudging to maintain unit sphere.
"""
from core.runtime.errors import record_degradation
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("Aura.ConceptualGravitation")

# Gravitational constants
GRAVITATION_ALPHA = 0.02       # Base nudge strength per co-access
ALPHA_DECAY_RATE = 0.8         # Decay per distance unit (far = weaker pull)
MAX_NUDGE_PER_CYCLE = 0.05    # Cap total displacement per dream cycle
MIN_COACCESSES = 2             # Minimum co-accesses before gravity applies
RECENCY_HALF_LIFE_S = 86400   # Co-accesses older than 1 day count less


class ConceptualGravitationEngine:
    """Modifies memory embeddings so related memories cluster over time."""

    def __init__(self):
        # co_access_counts[(id_a, id_b)] = [(timestamp, strength), ...]
        self._co_accesses: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
        self._current_turn_recalls: Set[str] = set()
        self._total_nudges = 0
        self._last_consolidation = 0.0

    def record_recall(self, memory_id: str):
        """Record that a memory was recalled in the current turn."""
        self._current_turn_recalls.add(memory_id)

    def end_turn(self):
        """Mark end of conversation turn. Record co-access pairs."""
        recalled = list(self._current_turn_recalls)
        now = time.time()

        # Every pair of co-recalled memories gets a co-access event
        for i in range(len(recalled)):
            for j in range(i + 1, len(recalled)):
                pair = tuple(sorted([recalled[i], recalled[j]]))
                self._co_accesses[pair].append((now, 1.0))

        self._current_turn_recalls.clear()

    def consolidate(self, memory_store) -> Dict[str, int]:
        """Apply gravitational nudges during dream consolidation.

        Args:
            memory_store: Object with get_embedding(id) -> np.ndarray
                         and set_embedding(id, np.ndarray) methods.

        Returns:
            Stats dict with nudge counts.
        """
        now = time.time()
        nudged = 0
        pairs_processed = 0

        for pair, events in list(self._co_accesses.items()):
            # Weight recent co-accesses more heavily
            total_weight = 0.0
            for ts, strength in events:
                age_s = now - ts
                recency = 0.5 ** (age_s / RECENCY_HALF_LIFE_S)
                total_weight += strength * recency

            if total_weight < MIN_COACCESSES:
                continue

            id_a, id_b = pair
            try:
                emb_a = memory_store.get_embedding(id_a)
                emb_b = memory_store.get_embedding(id_b)
            except Exception:
                continue

            if emb_a is None or emb_b is None:
                continue

            # Compute gravitational nudge
            diff = emb_b - emb_a
            distance = np.linalg.norm(diff)
            if distance < 1e-8:
                continue

            direction = diff / distance

            # Attraction strength: stronger for closer memories, capped
            alpha = GRAVITATION_ALPHA * total_weight * (ALPHA_DECAY_RATE ** distance)
            alpha = min(alpha, MAX_NUDGE_PER_CYCLE)

            # Nudge toward midpoint
            emb_a_new = emb_a + alpha * direction
            emb_b_new = emb_b - alpha * direction

            # Re-normalize to unit sphere
            norm_a = np.linalg.norm(emb_a_new)
            norm_b = np.linalg.norm(emb_b_new)
            if norm_a > 1e-8:
                emb_a_new = emb_a_new / norm_a
            if norm_b > 1e-8:
                emb_b_new = emb_b_new / norm_b

            try:
                memory_store.set_embedding(id_a, emb_a_new)
                memory_store.set_embedding(id_b, emb_b_new)
                nudged += 1
            except Exception as e:
                record_degradation('conceptual_gravitation', e)
                logger.debug("Gravitation nudge failed for %s: %s", pair, e)

            pairs_processed += 1

        # Prune old co-access events (older than 7 days)
        cutoff = now - 7 * 86400
        for pair in list(self._co_accesses.keys()):
            self._co_accesses[pair] = [
                (ts, s) for ts, s in self._co_accesses[pair] if ts > cutoff
            ]
            if not self._co_accesses[pair]:
                del self._co_accesses[pair]

        self._total_nudges += nudged
        self._last_consolidation = now

        logger.info(
            "Conceptual gravitation: %d pairs nudged, %d processed, %d total historical",
            nudged, pairs_processed, self._total_nudges,
        )
        return {"nudged": nudged, "pairs_processed": pairs_processed, "total_historical": self._total_nudges}

    def get_status(self) -> Dict:
        return {
            "active_pairs": len(self._co_accesses),
            "total_nudges": self._total_nudges,
            "last_consolidation": self._last_consolidation,
        }


_instance: Optional[ConceptualGravitationEngine] = None


def get_gravitation_engine() -> ConceptualGravitationEngine:
    global _instance
    if _instance is None:
        _instance = ConceptualGravitationEngine()
    return _instance
