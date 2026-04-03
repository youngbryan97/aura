# core/memory/hybrid_store.py — drop-in safe starter
import numpy as np
import asyncio
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

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
        self.prune_threshold = 2000  # entries

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
            "source": metadata.get("source", "unknown")
        }
        async with self._get_lock():
            try:
                # Unicode-safe write
                def _write():
                    with self.episodic_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(entry) + "\n")
                await asyncio.to_thread(_write)
                
                # Self-pruning periodic check
                if await self._count_entries() > self.prune_threshold:
                    await self._prune_oldest()
            except Exception as e:
                logger.error("Memory store failed: %s", e)

    async def _count_entries(self) -> int:
        if not self.episodic_path.exists():
            return 0
        def _count():
            with self.episodic_path.open("r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        return await asyncio.to_thread(_count)

    async def _prune_oldest(self):
        """Prunes the memory to stay under threshold, keeping high confidence items."""
        logger.info("HybridStore: Threshold reached. Pruning memory.")
        async with self._get_lock():
            def _prune():
                with self.episodic_path.open("r", encoding="utf-8") as f:
                    lines = [json.loads(line) for line in f if line.strip()]
                
                # Confidence filter: prefer keeping items with confidence > 0.8
                # but always keep the last 500
                recent = lines[-500:]
                high_conf = [l for l in lines[:-500] if l.get('confidence', 0) > 0.85]
                
                sorted_kept = sorted(recent + high_conf, key=lambda x: x['timestamp'])
                
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
                    # Primitive keyword search (placeholder for FAISS)
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
