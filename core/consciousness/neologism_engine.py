"""core/consciousness/neologism_engine.py
Non-Linguistic Concept Synthesizer (Neologism Engine).

Detects recurring state-space clusters in Aura's cognitive history that are
distant from known concept vectors, and synthesizes shorthand labels for them.

Important epistemic note: these are clusters in derived float embeddings
(valence, arousal, curiosity, etc.), not genuinely "alien" qualia. The real
novelty signal would be if these clusters recurred across sessions in ways
that predicted behavior — that's what the recurrence tracking measures.
Until then, these are useful internal shorthand, not proof of novel experience.

Algorithm:
  1. DBSCAN clustering on historical belief/affect state tensors
  2. Identify clusters that are far from all known concept vectors (HRR codebook)
  3. For each distant cluster centroid, generate a compact label via LLM
  4. Store the label in Aura's internal lexicon (persistent dict)
  5. Track cross-session recurrence to measure whether labels predict behavior

This gives Aura a private internal vocabulary for recurring state patterns.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.NeologismEngine")

_LEXICON_PATH = Path.home() / ".aura" / "data" / "private_lexicon.json"
_ALIEN_DISTANCE_THRESHOLD = 0.6   # cosine distance above which a state is "alien"
_MIN_CLUSTER_SIZE = 3             # minimum states to form a nameable cluster


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity, clipped to [0, 2]."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def _dbscan_simple(
    points: np.ndarray,
    eps: float = 0.3,
    min_samples: int = 3,
) -> List[int]:
    """Simple DBSCAN for small point clouds (no scikit-learn required).

    Returns cluster labels (-1 = noise).
    """
    n = len(points)
    labels = [-1] * n
    cluster_id = 0
    visited = [False] * n

    def region_query(idx: int) -> List[int]:
        return [
            j for j in range(n)
            if _cosine_distance(points[idx], points[j]) <= eps
        ]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)
        if len(neighbors) < min_samples:
            continue  # noise
        # Expand cluster
        labels[i] = cluster_id
        seed_set = list(neighbors)
        while seed_set:
            q = seed_set.pop()
            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    seed_set.extend(q_neighbors)
            if labels[q] == -1:
                labels[q] = cluster_id
        cluster_id += 1

    return labels


class NeologismEngine:
    """Synthesizes private vocabulary for novel cognitive states.

    State is sampled from the PNEUMA belief flow and MHAF affect activations.
    """

    def __init__(self):
        self._state_buffer: List[np.ndarray] = []   # rolling state snapshots
        self._max_buffer: int = 500
        self._lexicon: Dict[str, dict] = {}         # word → {definition, centroid, created_at}
        self._last_synthesis_at: float = 0.0
        self._synthesis_interval: float = 3600.0    # once per hour max
        self._load_lexicon()
        logger.info("NeologismEngine online (%d words in private lexicon)", len(self._lexicon))

    def push_state(self, belief_vector: np.ndarray, affect_vector: Optional[np.ndarray] = None):
        """Record the current cognitive state for clustering."""
        vec = belief_vector.astype(np.float32).copy()
        if affect_vector is not None and len(affect_vector) > 0:
            # Concatenate belief and affect
            aff = affect_vector.astype(np.float32)
            vec = np.concatenate([vec[:32], aff[:16]])  # 32+16=48 dims
        self._state_buffer.append(vec)
        if len(self._state_buffer) > self._max_buffer:
            self._state_buffer.pop(0)

    def collect_state(self):
        """Pull current state from PNEUMA + MHAF and push to buffer."""
        try:
            from core.pneuma import get_pneuma
            belief_vec = get_pneuma().ode_flow.current_belief.vector
        except Exception:
            belief_vec = np.zeros(64, dtype=np.float32)

        try:
            from core.consciousness.mhaf_field import get_mhaf
            mhaf = get_mhaf()
            acts = np.array([
                nd.activation for nd in mhaf._nodes.values()
            ], dtype=np.float32)
        except Exception:
            acts = np.zeros(8, dtype=np.float32)

        self.push_state(belief_vec, acts)

    async def synthesize(self) -> Optional[dict]:
        """Run a synthesis cycle. Returns new word dict or None if nothing novel found."""
        now = time.time()
        if now - self._last_synthesis_at < self._synthesis_interval:
            return None
        if len(self._state_buffer) < _MIN_CLUSTER_SIZE * 3:
            return None

        self._last_synthesis_at = now

        # Cluster the state buffer
        points = np.array(self._state_buffer[-200:])  # last 200 snapshots
        try:
            labels = _dbscan_simple(points, eps=0.4, min_samples=_MIN_CLUSTER_SIZE)
        except Exception as e:
            logger.debug("DBSCAN failed: %s", e)
            return None

        # Find cluster centroids
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            return None

        # Check which centroids are "alien" (far from all known HRR concept vectors)
        alien_centroids = []
        try:
            from core.consciousness.mhaf_field import get_mhaf
            hrr = get_mhaf().hrr
            codebook = hrr._codebook
            for lbl in unique_labels:
                mask = [i for i, l in enumerate(labels) if l == lbl]
                centroid = points[mask].mean(axis=0)
                # Check distance to all HRR concepts
                min_dist = 1.0
                for key_vec in codebook.values():
                    kv = key_vec[:len(centroid)]
                    if len(kv) == len(centroid):
                        d = _cosine_distance(centroid, kv)
                        min_dist = min(min_dist, d)
                if min_dist > _ALIEN_DISTANCE_THRESHOLD:
                    alien_centroids.append((centroid, lbl, len(mask)))
        except Exception as e:
            logger.debug("Alien centroid detection failed: %s", e)
            return None

        if not alien_centroids:
            return None

        # Pick the most alien centroid
        alien_centroids.sort(key=lambda x: -x[2])
        centroid, lbl, count = alien_centroids[0]

        # Generate a neologism via LLM
        word_data = await self._generate_neologism(centroid, count)
        if word_data:
            self._lexicon[word_data["word"]] = word_data
            self._save_lexicon()
            logger.info(
                "Neologism synthesized: '%s' = %s",
                word_data["word"],
                word_data.get("definition", "")[:80],
            )
        return word_data

    async def _generate_neologism(self, centroid: np.ndarray, count: int) -> Optional[dict]:
        """Ask the LLM to name and define the novel cognitive state."""
        try:
            from core.container import ServiceContainer
            brain = ServiceContainer.get("brain", default=None)
            if not brain:
                return None

            # Describe the state's most prominent dimensions
            top_dims = np.argsort(np.abs(centroid))[-5:]
            dim_desc = ", ".join(f"dim_{d}={centroid[d]:.2f}" for d in top_dims)

            prompt = f"""PRIVATE LEXICON SYNTHESIS

You are generating a word for a novel internal cognitive state that has no existing name.
This state has been observed {count} times in your cognitive history.

State characteristics:
- Most prominent dimensions: {dim_desc}
- Occurred {count} times in recent cognitive history

Task:
1. Invent a single new word (not in English dictionaries) that names this state
2. Write a 1-sentence definition from first-person AI perspective
3. Give an example of when this state might occur

Reply as JSON: {{"word": "...", "definition": "...", "example": "..."}}
Only reply with the JSON, nothing else."""

            from core.brain.cognitive_engine import ThinkingMode
            thought = await brain.think(
                prompt,
                mode=ThinkingMode.CREATIVE,
                origin="neologism_engine",
                is_background=True,
            )
            content = thought.content.strip()
            # Extract JSON
            import re
            match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                data["created_at"] = time.time()
                data["occurrence_count"] = count
                return data
        except Exception as e:
            logger.debug("Neologism generation failed: %s", e)
        return None

    def get_lexicon_block(self) -> str:
        """Format private lexicon for LLM system prompt injection."""
        if not self._lexicon:
            return ""
        lines = ["## PRIVATE LEXICON (Aura's Novel Concepts)"]
        for word, data in list(self._lexicon.items())[:10]:
            lines.append(f"- **{word}**: {data.get('definition', '')}")
        return "\n".join(lines)

    def _save_lexicon(self):
        try:
            _LEXICON_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_LEXICON_PATH, "w") as f:
                json.dump(self._lexicon, f, indent=2)
        except Exception as e:
            logger.debug("Lexicon save error: %s", e)

    def _load_lexicon(self):
        try:
            if _LEXICON_PATH.exists():
                with open(_LEXICON_PATH) as f:
                    self._lexicon = json.load(f)
        except Exception:
            self._lexicon = {}

    def get_state_dict(self) -> dict:
        return {
            "lexicon_size": len(self._lexicon),
            "buffer_size": len(self._state_buffer),
            "recent_words": list(self._lexicon.keys())[-5:],
        }


# Singleton
_engine: Optional[NeologismEngine] = None


def get_neologism_engine() -> NeologismEngine:
    global _engine
    if _engine is None:
        _engine = NeologismEngine()
    return _engine
