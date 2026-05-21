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

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.errors import FallbackClassification, Severity, record_degradation

logger = logging.getLogger("Consciousness.NeologismEngine")

_LEXICON_PATH = Path.home() / ".aura" / "data" / "private_lexicon.json"
_ALIEN_DISTANCE_THRESHOLD = 0.6  # cosine distance above which a state is "alien"
_MIN_CLUSTER_SIZE = 3  # minimum states to form a nameable cluster
_STATE_VECTOR_DIMS = 48
_RECOVERABLE_NEOLOGISM_ERRORS = (
    AttributeError,
    ImportError,
    json.JSONDecodeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_WORD_RE = re.compile(r"[^a-zA-Z-]+")


def _record_neologism_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        record_degradation(
            "neologism_engine",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError:
        record_degradation("neologism_engine", error)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity, clipped to [0, 2]."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float32).ravel(), copy=False)
    b = np.nan_to_num(np.asarray(b, dtype=np.float32).ravel(), copy=False)
    if a.size == 0 or b.size == 0:
        return 1.0
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    similarity = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.clip(1.0 - similarity, 0.0, 2.0))


def _dbscan_simple(
    points: np.ndarray,
    eps: float = 0.3,
    min_samples: int = 3,
) -> list[int]:
    """Simple DBSCAN for small point clouds (no scikit-learn required).

    Returns cluster labels (-1 = noise).
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("DBSCAN requires a non-empty 2D point cloud")
    points = np.nan_to_num(points, copy=False)
    n = len(points)
    labels = [-1] * n
    cluster_id = 0
    visited = [False] * n

    def region_query(idx: int) -> list[int]:
        return [j for j in range(n) if _cosine_distance(points[idx], points[j]) <= eps]

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
        self._state_buffer: list[np.ndarray] = []  # rolling state snapshots
        self._max_buffer: int = 500
        self._lexicon: dict[str, dict[str, Any]] = {}  # word → {definition, centroid, created_at}
        self._last_synthesis_at: float = 0.0
        self._synthesis_interval: float = 3600.0  # once per hour max
        self._load_lexicon()
        logger.info("NeologismEngine online (%d words in private lexicon)", len(self._lexicon))

    def _coerce_vector(self, raw: Any, *, dims: int, label: str) -> np.ndarray:
        try:
            vector = np.asarray(raw, dtype=np.float32).ravel()
        except (TypeError, ValueError) as exc:
            _record_neologism_degradation(
                exc,
                severity="warning",
                action="used zero vector for malformed neologism state component",
                extra={"component": label, "dims": dims},
            )
            return np.zeros(dims, dtype=np.float32)
        if vector.size == 0:
            _record_neologism_degradation(
                ValueError("empty neologism state vector"),
                severity="warning",
                action="used zero vector for empty neologism state component",
                extra={"component": label, "dims": dims},
            )
            return np.zeros(dims, dtype=np.float32)
        if not np.isfinite(vector).all():
            _record_neologism_degradation(
                ValueError("non-finite neologism state vector"),
                severity="warning",
                action="replaced non-finite neologism state values",
                extra={"component": label},
            )
            vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
        if vector.size < dims:
            vector = np.pad(vector, (0, dims - vector.size), mode="constant")
        return vector[:dims].astype(np.float32, copy=False)

    def _compose_state_vector(
        self,
        belief_vector: Any,
        affect_vector: Any | None = None,
    ) -> np.ndarray:
        belief = self._coerce_vector(belief_vector, dims=32, label="belief")
        if affect_vector is None:
            affect = np.zeros(16, dtype=np.float32)
        else:
            affect = self._coerce_vector(affect_vector, dims=16, label="affect")
        return np.concatenate([belief, affect]).astype(np.float32, copy=False)

    def _coerce_state_snapshot(self, raw: Any) -> np.ndarray:
        return self._coerce_vector(raw, dims=_STATE_VECTOR_DIMS, label="state_snapshot")

    def push_state(self, belief_vector: np.ndarray, affect_vector: np.ndarray | None = None):
        """Record the current cognitive state for clustering."""
        vec = self._compose_state_vector(belief_vector, affect_vector)
        self._state_buffer.append(vec)
        if len(self._state_buffer) > self._max_buffer:
            self._state_buffer.pop(0)

    def collect_state(self):
        """Pull current state from PNEUMA + MHAF and push to buffer."""
        try:
            from core.pneuma import get_pneuma

            belief_vec = get_pneuma().ode_flow.current_belief.vector
        except _RECOVERABLE_NEOLOGISM_ERRORS as exc:
            _record_neologism_degradation(
                exc,
                severity="warning",
                action="used zero belief vector during neologism state collection",
            )
            belief_vec = np.zeros(64, dtype=np.float32)

        try:
            from core.consciousness.mhaf_field import get_mhaf

            mhaf = get_mhaf()
            acts = np.array([nd.activation for nd in mhaf._nodes.values()], dtype=np.float32)
        except _RECOVERABLE_NEOLOGISM_ERRORS as exc:
            _record_neologism_degradation(
                exc,
                severity="warning",
                action="used zero affect vector during neologism state collection",
            )
            acts = np.zeros(8, dtype=np.float32)

        self.push_state(belief_vec, acts)

    async def synthesize(self) -> dict[str, Any] | None:
        """Run a synthesis cycle. Returns new word dict or None if nothing novel found."""
        now = time.time()
        if now - self._last_synthesis_at < self._synthesis_interval:
            return None
        if len(self._state_buffer) < _MIN_CLUSTER_SIZE * 3:
            return None

        self._last_synthesis_at = now

        # Cluster the state buffer
        points = np.vstack([self._coerce_state_snapshot(vec) for vec in self._state_buffer[-200:]])
        try:
            labels = _dbscan_simple(points, eps=0.4, min_samples=_MIN_CLUSTER_SIZE)
        except _RECOVERABLE_NEOLOGISM_ERRORS as e:
            _record_neologism_degradation(
                e,
                action="skipped neologism synthesis after clustering failure",
            )
            logger.debug("DBSCAN failed: %s", e)
            return None

        # Find cluster centroids
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            return None

        # Check which centroids are "alien" (far from all known HRR concept vectors)
        alien_centroids = []
        codebook_values: list[np.ndarray] = []
        try:
            from core.consciousness.mhaf_field import get_mhaf

            hrr = get_mhaf().hrr
            codebook = getattr(hrr, "_codebook", {})
            if isinstance(codebook, dict):
                codebook_values = [
                    np.asarray(value, dtype=np.float32) for value in codebook.values()
                ]
        except _RECOVERABLE_NEOLOGISM_ERRORS as e:
            _record_neologism_degradation(
                e,
                severity="warning",
                action="used recurrence-only novelty gate without HRR codebook",
            )
            logger.debug("Alien centroid detection degraded: %s", e)

        for lbl in unique_labels:
            mask = [i for i, label in enumerate(labels) if label == lbl]
            centroid = points[mask].mean(axis=0)
            min_dist = 1.0
            for key_vec in codebook_values:
                d = _cosine_distance(centroid, key_vec)
                min_dist = min(min_dist, d)
            if min_dist > _ALIEN_DISTANCE_THRESHOLD:
                alien_centroids.append((centroid, lbl, len(mask), min_dist))

        if not alien_centroids:
            return None

        # Pick the most alien centroid
        alien_centroids.sort(key=lambda x: (-x[2], -x[3]))
        centroid, lbl, count, _min_dist = alien_centroids[0]

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

    def _fallback_neologism(
        self,
        centroid: np.ndarray,
        count: int,
        *,
        reason: str,
    ) -> dict[str, Any]:
        centroid = self._coerce_state_snapshot(centroid)
        digest = hashlib.sha1(centroid.tobytes()).hexdigest()[:8]
        top_dims = np.argsort(np.abs(centroid))[-3:][::-1]
        dim_desc = ", ".join(f"dim_{int(dim)}={centroid[dim]:+.2f}" for dim in top_dims)
        return {
            "word": f"velm{digest}",
            "definition": (
                "A recurring internal state cluster marked by "
                f"{dim_desc}; named deterministically because LLM naming was unavailable."
            ),
            "example": "May recur when similar belief and affect vectors reappear across cognition.",
            "created_at": time.time(),
            "occurrence_count": int(max(0, count)),
            "source": "deterministic_fallback",
            "fallback_reason": reason,
            "centroid_fingerprint": digest,
        }

    def _sanitize_word_data(
        self,
        data: dict[str, Any],
        centroid: np.ndarray,
        count: int,
    ) -> dict[str, Any] | None:
        raw_word = str(data.get("word", "")).strip().lower()
        word = _WORD_RE.sub("", raw_word)[:32].strip("-")
        if len(word) < 3:
            return None
        definition = str(data.get("definition", "")).strip()
        example = str(data.get("example", "")).strip()
        if not definition or not example:
            return None
        sanitized = {
            "word": word,
            "definition": definition[:500],
            "example": example[:500],
            "created_at": time.time(),
            "occurrence_count": int(max(0, count)),
            "source": data.get("source", "llm"),
        }
        sanitized["centroid_fingerprint"] = hashlib.sha1(
            self._coerce_state_snapshot(centroid).tobytes()
        ).hexdigest()[:8]
        return sanitized

    def _extract_json_object(self, content: Any) -> dict[str, Any] | None:
        text = str(content or "").strip()
        if not text:
            return None
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        parsed = json.loads(match.group())
        return parsed if isinstance(parsed, dict) else None

    async def _generate_neologism(self, centroid: np.ndarray, count: int) -> dict[str, Any] | None:
        """Ask the LLM to name and define the novel cognitive state."""
        try:
            from core.container import ServiceContainer

            brain = ServiceContainer.get("brain", default=None)
            if not brain:
                _record_neologism_degradation(
                    RuntimeError("brain service unavailable"),
                    severity="warning",
                    action="used deterministic neologism fallback without brain service",
                )
                return self._fallback_neologism(centroid, count, reason="brain_unavailable")

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
            data = self._extract_json_object(getattr(thought, "content", ""))
            if data is None:
                _record_neologism_degradation(
                    ValueError("LLM returned no JSON object for neologism"),
                    severity="warning",
                    action="used deterministic neologism fallback after invalid LLM payload",
                )
                return self._fallback_neologism(centroid, count, reason="invalid_llm_payload")
            sanitized = self._sanitize_word_data(data, centroid, count)
            if sanitized is None:
                _record_neologism_degradation(
                    ValueError("LLM neologism JSON failed validation"),
                    severity="warning",
                    action="used deterministic neologism fallback after invalid LLM fields",
                )
                return self._fallback_neologism(centroid, count, reason="invalid_llm_fields")
            return sanitized
        except _RECOVERABLE_NEOLOGISM_ERRORS as e:
            _record_neologism_degradation(
                e,
                action="used deterministic neologism fallback after generation failure",
            )
            logger.debug("Neologism generation failed: %s", e)
            return self._fallback_neologism(centroid, count, reason=type(e).__name__)

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
            tmp_path = _LEXICON_PATH.with_suffix(_LEXICON_PATH.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._lexicon, f, indent=2)
            tmp_path.replace(_LEXICON_PATH)
        except _RECOVERABLE_NEOLOGISM_ERRORS as e:
            _record_neologism_degradation(
                e,
                action="kept in-memory private lexicon after save failure",
            )
            logger.debug("Lexicon save error: %s", e)

    def _load_lexicon(self):
        try:
            if _LEXICON_PATH.exists():
                with open(_LEXICON_PATH, encoding="utf-8") as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict):
                    raise ValueError("private lexicon must be a JSON object")
                self._lexicon = {
                    str(word): data for word, data in loaded.items() if isinstance(data, dict)
                }
        except _RECOVERABLE_NEOLOGISM_ERRORS as exc:
            _record_neologism_degradation(
                exc,
                severity="warning",
                action="started with empty private lexicon after load failure",
            )
            self._lexicon = {}

    def get_state_dict(self) -> dict:
        return {
            "lexicon_size": len(self._lexicon),
            "buffer_size": len(self._state_buffer),
            "recent_words": list(self._lexicon.keys())[-5:],
        }


# Singleton
_engine: NeologismEngine | None = None


def get_neologism_engine() -> NeologismEngine:
    global _engine
    if _engine is None:
        _engine = NeologismEngine()
    return _engine
