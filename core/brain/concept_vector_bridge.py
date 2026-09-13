"""Latent concept routing between internal cognitive services.

The bridge prefers Aura's shared local embedding engine. When that provider is
not yet admitted during boot, it uses a deterministic lexical projection so a
concept remains stable across processes and can be upgraded after the provider
comes online.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service, register_runtime_service

logger = logging.getLogger("Aura.Cryptolalia")

@dataclass
class LatentThought:
    """A raw, un-decoded semantic vector representing a concept."""
    id: str
    source_node: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())


class ConceptVectorBridge:
    """The central hub for latent telepathy between nodes."""

    name = "concept_vector_bridge"
    VECTOR_DIM = 384

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.active_streams: dict[str, list[LatentThought]] = {}
        self._concept_cache: dict[str, list[float]] = {}
        self._concept_sources: dict[str, str] = {}
        self._fallback_notice_emitted = False
        
    async def transmit(
        self,
        source: str,
        target: str,
        semantic_vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Send a raw vector payload to another node.
        """
        thought_id = f"latent_{int(time.time() * 1000)}"
        thought = LatentThought(
            id=thought_id,
            source_node=source,
            vector=semantic_vector,
            metadata=metadata or {}
        )
        
        if target not in self.active_streams:
            self.active_streams[target] = []
            
        self.active_streams[target].append(thought)
        logger.debug("🌌 [Cryptolalia] %s -> %s (Vector dim: %s)", source, target, len(semantic_vector))
        
        # Fire event for the decoder or monitoring
        event_bus = get_runtime_service("event_bus", default=None)
        if event_bus:
            await event_bus.publish("cryptolalia_transmission", {
                "source": source,
                "target": target,
                "thought_id": thought_id
            })
            
        return thought_id

    async def receive(self, target: str, consume: bool = True) -> list[LatentThought]:
        """
        Fetch pending latent thoughts for a given node.
        """
        if target not in self.active_streams:
            return []
            
        thoughts = self.active_streams[target]
        if consume:
            self.active_streams[target] = []
        return thoughts

    @classmethod
    def _deterministic_lexical_vector(cls, text: str) -> list[float]:
        """Return a stable, normalized token/character feature projection."""

        normalized = " ".join(str(text or "").casefold().split())
        vector = np.zeros(cls.VECTOR_DIM, dtype=np.float32)
        tokens = re.findall(r"[\w']+", normalized, flags=re.UNICODE)
        features: list[tuple[str, float]] = [(f"w:{token}", 1.0) for token in tokens]
        padded = f"  {normalized}  "
        features.extend(
            (f"c:{padded[index:index + 3]}", 0.25)
            for index in range(max(0, len(padded) - 2))
        )
        if not features:
            features = [("empty", 1.0)]
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % cls.VECTOR_DIM
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign * weight
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector.tolist()

    @staticmethod
    def _coerce_provider_vector(value: Any) -> list[float]:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("embedding provider returned an empty or non-finite vector")
        return vector.tolist()

    async def _provider_vector(self, text: str) -> list[float] | None:
        vector_memory = get_runtime_service("vector_memory_engine", default=None)
        embedder = getattr(vector_memory, "embedder", None)
        embed = getattr(embedder, "embed", None)
        if callable(embed):
            try:
                return self._coerce_provider_vector(
                    await asyncio.to_thread(embed, text)
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation("concept_vector_bridge", exc)
                logger.warning("Shared embedding provider failed; using bounded fallback: %s", exc)

        cognition = get_runtime_service("cognitive_engine", default=None)
        generate = getattr(getattr(cognition, "client", None), "generate_embedding", None)
        if callable(generate):
            try:
                result = generate(text)
                if hasattr(result, "__await__"):
                    result = await result
                return self._coerce_provider_vector(result)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation("concept_vector_bridge", exc)
                logger.warning("Cognitive embedding provider failed; using bounded fallback: %s", exc)
        return None

    async def generate_concept_vector(self, text_concept: str) -> list[float]:
        """Convert text into a provider-backed or deterministic concept vector."""

        concept = str(text_concept or "").strip()
        cached = self._concept_cache.get(concept)
        if cached is not None and self._concept_sources.get(concept) == "provider":
            return cached

        provider_vector = await self._provider_vector(concept)
        if provider_vector is not None:
            self._concept_cache[concept] = provider_vector
            self._concept_sources[concept] = "provider"
            return provider_vector
        if cached is not None:
            return cached

        vector = self._deterministic_lexical_vector(concept)
        self._concept_cache[concept] = vector
        self._concept_sources[concept] = "deterministic_lexical_fallback"
        if not self._fallback_notice_emitted:
            logger.info(
                "Embedding provider not yet available; using stable lexical concept projection."
            )
            self._fallback_notice_emitted = True
        return vector

def register_concept_bridge(orchestrator=None):
    bridge = ConceptVectorBridge(orchestrator)
    register_runtime_service(
        "concept_bridge",
        bridge,
        owner="core/brain/concept_vector_bridge.py",
        registered_by="register_concept_bridge",
    )
    return bridge
