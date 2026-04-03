"""Memory Consolidator — Long-Term Potentiation

Scans the vector store for near-duplicate memories (cosine sim > threshold)
and merges them into a single, strengthened memory.

Prevents memory bloat and mimics biological long-term potentiation.
Runs as the final maintenance step in the Dreamer sleep cycle.
"""
import logging
import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Kernel.MemoryConsolidator")


@dataclass
class ConsolidationReport:
    memories_scanned: int = 0
    duplicates_merged: int = 0
    clusters_found: int = 0
    errors: List[str] = field(default_factory=list)
    duration_s: float = 0.0

    def __str__(self) -> str:
        return (
            f"Consolidation: scanned={self.memories_scanned}, "
            f"merged={self.duplicates_merged}, "
            f"clusters={self.clusters_found} ({self.duration_s:.1f}s)"
        )


class MemoryConsolidator:
    """Merges near-duplicate memory vectors.
    Keeps the richest (longest content) and deletes duplicates.
    Works with both ChromaDB-backed and JSON-fallback vector stores.
    """

    def __init__(
        self,
        vector_memory: Any = None,
        similarity_threshold: float = 0.97,
        batch_size: int = 100,
    ):
        self.vector_memory = vector_memory
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size

    async def consolidate(self) -> ConsolidationReport:
        """Scan for near-duplicate memories and merge them."""
        report = ConsolidationReport()
        t0 = time.monotonic()
        logger.info("🧠 Memory consolidation starting...")

        if not self.vector_memory:
            logger.warning("No vector memory available — skipping consolidation.")
            report.errors.append("no_vector_memory")
            report.duration_s = time.monotonic() - t0
            return report

        try:
            memories = self._fetch_memories()
            report.memories_scanned = len(memories)
            if len(memories) < 2:
                logger.info("🧠 Not enough memories to consolidate.")
                report.duration_s = time.monotonic() - t0
                return report

            clusters = self._find_duplicate_clusters(memories)
            report.clusters_found = len(clusters)
            for cluster in clusters:
                self._merge_cluster(cluster, report)
        except Exception as exc:
            msg = f"Consolidation error: {exc}"
            logger.error(msg, exc_info=True)
            report.errors.append(msg)

        report.duration_s = time.monotonic() - t0
        logger.info("🧠 %s", report)

        try:
            from core.thought_stream import get_emitter
            get_emitter().emit("Memory Consolidation 🧠", str(report), level="info")
        except Exception as exc:
            logger.debug("Suppressed thought-stream emit: %s", exc)

        return report

    # ------------------------------------------------------------------
    def _fetch_memories(self) -> List[Dict]:
        try:
            if hasattr(self.vector_memory, "_collection") and self.vector_memory._collection:
                result = self.vector_memory._collection.get(
                    limit=self.batch_size,
                    include=["documents", "metadatas", "embeddings"],
                )
                memories = []
                ids = result.get("ids", [])
                docs = result.get("documents", [])
                metas = result.get("metadatas", [])
                embeddings = result.get("embeddings", [])
                for i, mid in enumerate(ids):
                    memories.append({
                        "id": mid,
                        "content": docs[i] if i < len(docs) else "",
                        "metadata": metas[i] if i < len(metas) else {},
                        "embedding": embeddings[i] if embeddings and i < len(embeddings) else None,
                    })
                return memories
            if hasattr(self.vector_memory, "_memories"):
                # Use slice explicitly to satisfy linter
                all_items = list(self.vector_memory._memories.items())
                items = all_items[0:self.batch_size]
                return [
                    {"id": mid, "content": data.get("content", ""), "metadata": data.get("metadata", {})}
                    for mid, data in items
                ]
        except Exception as exc:
            logger.warning("Failed to fetch memories: %s", exc)
        return []

    def _find_duplicate_clusters(self, memories: List[Dict]) -> List[List[Dict]]:
        """Find duplicate memory clusters using vectorized cosine similarity (OPT-03)."""
        # Batch embed all memories if embeddings are pre-fetched
        embeddings = [m.get("embedding") for m in memories]
        if all(e is not None for e in embeddings) and len(embeddings) > 1:
            try:
                import numpy as np
                emb_matrix = np.array(embeddings, dtype=np.float32)
                # Normalize for cosine similarity calculation
                norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                normalized = emb_matrix / (norms + 1e-8)
                sim_matrix = normalized @ normalized.T  # Cosine similarity matrix
                
                merged_ids: set = set()
                clusters: List[List[Dict]] = []
                for i, mem in enumerate(memories):
                    if mem["id"] in merged_ids:
                        continue
                    cluster = [mem]
                    for j in range(i + 1, len(memories)):
                        if memories[j]["id"] not in merged_ids and sim_matrix[i, j] >= self.similarity_threshold:
                            cluster.append(memories[j])
                            merged_ids.add(memories[j]["id"])
                    if len(cluster) > 1:
                        merged_ids.add(mem["id"])
                        clusters.append(cluster)
                return clusters
            except Exception as e:
                logger.debug("Vectorized similarity check failed: %s", e)

        # Fallback to existing search_similar path (O(n) search calls)
        merged_ids: set = set()
        clusters: List[List[Dict]] = []
        for mem in memories:
            if mem["id"] in merged_ids:
                continue
            cluster = [mem]
            content = mem.get("content", "")
            if not content:
                continue
            try:
                similar = self.vector_memory.search_similar(content, limit=5)
                for result in similar:
                    result_id = result.get("id", "")
                    if result_id and result_id != mem["id"] and result_id not in merged_ids:
                        score = result.get("score", result.get("similarity", 0))
                        if score >= self.similarity_threshold:
                            cluster.append(result)
                            merged_ids.add(result_id)
            except Exception as exc:
                logger.debug("Similarity search failed for %s: %s", mem["id"], exc)
            if len(cluster) > 1:
                merged_ids.add(mem["id"])
                clusters.append(cluster)
        return clusters

    def _merge_cluster(self, cluster: List[Dict], report: ConsolidationReport) -> None:
        """Merge a cluster of memories into a single strengthened one."""
        cluster.sort(key=lambda m: len(m.get("content", "")), reverse=True)
        winner = cluster[0]
        winner_id = winner.get("id")
        
        # Strengthen the winner's metadata (Long-Term Potentiation)
        combined_importance = max(
            (m.get("metadata", {}).get("importance", 0.5) for m in cluster),
            default=0.5
        )
        if hasattr(self.vector_memory, "_collection") and self.vector_memory._collection:
            try:
                # Up-voted metadata for potentiation
                new_meta = {
                    **winner.get("metadata", {}),
                    "importance": min(1.0, combined_importance * 1.1),
                    "merged_count": (winner.get("metadata", {}).get("merged_count", 0) or 0) + len(cluster) - 1
                }
                self.vector_memory._collection.update(
                    ids=[winner_id],
                    metadatas=[new_meta]
                )
            except Exception as _e:
                logger.debug('Ignored Exception in memory_management.py: %s', _e)

        for loser in cluster[1:]:
            loser_id = loser.get("id")
            if not loser_id:
                continue
            try:
                if hasattr(self.vector_memory, "_collection") and self.vector_memory._collection:
                    self.vector_memory._collection.delete(ids=[loser_id])
                elif hasattr(self.vector_memory, "_memories") and loser_id in self.vector_memory._memories:
                    del self.vector_memory._memories[loser_id]
                    if hasattr(self.vector_memory, "_save_fallback"):
                        self.vector_memory._save_fallback()
                report.duplicates_merged += 1
                logger.debug("Merged memory %s into %s", loser_id[:12], winner_id[:12])
            except Exception as exc:
                report.errors.append(f"merge delete {loser_id}: {exc}")