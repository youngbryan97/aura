# core/memory/hybrid_store.py — drop-in safe starter
from core.runtime.errors import record_degradation
import asyncio
import logging
import json
import time
from pathlib import Path
from typing import List, Dict

from core.memory.retention_policy import hybrid_memory_retention_policy

logger = logging.getLogger("Aura.MemoryStore")

class HybridMemoryStore:
    """
    Zenith Audit Fix 2.2: Minimal safe Vector + Episodic Store.
    Prevents 'retrieve own past errors' loop via confidence filter.
    """
    def __init__(self, storage_dir: str = "data/memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.episodic_path = self.storage_dir / "episodic.jsonl"
        self._lock: asyncio.Lock | None = None  # Lazy-init to avoid event loop binding
        self.retention_policy = hybrid_memory_retention_policy()
        self.prune_threshold = self.retention_policy.max_items

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def store(self, content: str, metadata: Dict):
        """Store an entry in episodic memory with confidence level."""
        entry = {
            "timestamp": time.time(),
            "content": content,
            "confidence": metadata.get("confidence", 0.8),
            "source": metadata.get("source", "unknown"),
            "importance": metadata.get("importance", metadata.get("emotional_weight", 0.0)),
            "protected": bool(metadata.get("protected") or metadata.get("pinned")),
        }
        should_prune = False
        try:
            async with self._get_lock():
                # Unicode-safe write
                def _write():
                    with self.episodic_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(entry) + "\n")
                await asyncio.to_thread(_write)

            # Self-pruning periodic check. Run outside the write lock because
            # _prune_oldest owns that lock while rewriting the JSONL file.
            should_prune = await self._count_entries() > self.prune_threshold
            if should_prune:
                await self._prune_oldest()
        except (json.JSONDecodeError, TypeError, ValueError, OSError) as e:
            record_degradation('hybrid_store', e)
            logger.error("Memory store failed: %s", e)

    async def _count_entries(self) -> int:
        if not self.episodic_path.exists():
            return 0
        def _count():
            with self.episodic_path.open("r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        return await asyncio.to_thread(_count)

    @staticmethod
    def _entry_score(entry: Dict, index: int, total: int) -> float:
        if entry.get("protected") or entry.get("pinned"):
            return float("inf")
        confidence = float(entry.get("confidence", 0.0) or 0.0)
        importance = float(entry.get("importance", entry.get("emotional_weight", 0.0)) or 0.0)
        recency = index / max(1, total - 1)
        return (confidence * 0.50) + (importance * 0.35) + (recency * 0.15)

    async def _prune_oldest(self):
        """Prunes memory under the adaptive threshold while preserving salient entries."""
        logger.info(
            "HybridStore: retention cap reached. Pruning with %s policy.",
            self.retention_policy.basis,
        )
        async with self._get_lock():
            def _prune():
                with self.episodic_path.open("r", encoding="utf-8") as f:
                    lines = [json.loads(line) for line in f if line.strip()]

                if len(lines) <= self.retention_policy.max_items:
                    return

                keep_count = self.retention_policy.keep_count(len(lines))
                recent_floor = min(len(lines), max(100, int(keep_count * 0.10)))
                forced_indices = set(range(max(0, len(lines) - recent_floor), len(lines)))
                scored = [
                    (idx, self._entry_score(entry, idx, len(lines)))
                    for idx, entry in enumerate(lines)
                ]
                protected_indices = {idx for idx, score in scored if score == float("inf")}
                keep_indices = forced_indices | protected_indices
                slots = max(0, keep_count - len(keep_indices))
                for idx, _score in sorted(scored, key=lambda item: item[1], reverse=True):
                    if slots <= 0:
                        break
                    if idx in keep_indices:
                        continue
                    keep_indices.add(idx)
                    slots -= 1

                if len(keep_indices) > keep_count:
                    keep_indices = {
                        idx
                        for idx, _score in sorted(
                            ((idx, score) for idx, score in scored if idx in keep_indices),
                            key=lambda item: item[1],
                            reverse=True,
                        )[:keep_count]
                    }

                sorted_kept = [entry for idx, entry in enumerate(lines) if idx in keep_indices]
                with self.episodic_path.open("w", encoding="utf-8") as f:
                    for entry in sorted_kept:
                        f.write(json.dumps(entry) + "\n")
            await asyncio.to_thread(_prune)

    async def retrieve(self, query: str, top_k=5, min_confidence=0.6) -> List[Dict]:
        """
        Simple retrieval with 'Blood-Brain Barrier' filter.
        Prevents retrieving own past errors.
        """
        if not self.episodic_path.exists():
            return []
            
        def _search():
            with self.episodic_path.open("r", encoding="utf-8") as f:
                results = []
                for line in f:
                    entry = json.loads(line)
                    # Filter by minimum confidence
                    if entry.get('confidence', 0) < min_confidence:
                        continue
                    # Primitive keyword search (provisional reference for FAISS)
                    if any(word.lower() in entry['content'].lower() for word in query.split()):
                        results.append(entry)
                
                return sorted(results, key=lambda x: x['timestamp'], reverse=True)[:top_k]
        
        return await asyncio.to_thread(_search)

# Singleton helper
_store = None
def get_hybrid_store() -> HybridMemoryStore:
    global _store
    if _store is None:
        _store = HybridMemoryStore()
    return _store
