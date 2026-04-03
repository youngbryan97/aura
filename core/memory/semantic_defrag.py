"""
core/memory/semantic_defrag.py
──────────────────────────────
Implements "Semantic Sleep" for vector memory consolidation.
Finds dense clusters of similar memories and merges them into unified concepts.
"""
import logging
import time
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any
from core.container import ServiceContainer

logger = logging.getLogger("Aura.Memory.Defrag")

class SemanticDefragmenter:
    FILE_REFERENCE_RE = re.compile(
        r"(?<![\w/.-])((?:[\w.-]+/)+[\w.-]+\.(?:py|tsx?|jsx?|json|md|ya?ml|toml|sh|go|rs|java|c|cpp|h))(?:[:#]\d+)?"
    )

    def __init__(self, collection_name: str = "aura_memories"):
        self.collection_name = collection_name
        self.repo_root = Path(__file__).resolve().parents[2]

    def _get_id(self, item: Dict[str, Any]) -> str:
        metadata = dict(item.get("metadata") or {})
        return str(item.get("id") or item.get("memory_id") or metadata.get("id") or "")

    def _collect_file_references(self, docs: List[str]) -> List[str]:
        refs = set()
        for doc in docs:
            for match in self.FILE_REFERENCE_RE.finditer(doc or ""):
                refs.add(match.group(1))
        return sorted(refs)

    def _build_resolution_context(self, docs: List[str]) -> str:
        refs = self._collect_file_references(docs)
        if not refs:
            return ""

        existing: List[str] = []
        missing: List[str] = []
        for ref in refs[:6]:
            candidate = Path(ref)
            if not candidate.is_absolute():
                candidate = (self.repo_root / candidate).resolve()
            if candidate.exists():
                existing.append(ref)
            else:
                missing.append(ref)

        notes: List[str] = []
        if existing:
            notes.append("Live file check found existing references: " + ", ".join(existing))
        if missing:
            notes.append("Some remembered file references no longer exist: " + ", ".join(missing))
        if len(existing) + len(missing) > 1:
            notes.append("If the memories disagree on technical facts, preserve uncertainty rather than inventing a single confident file claim.")
        return "\n".join(notes)

    async def run_defrag_cycle(self):
        """
        Scans the vector database, finds clusters, and consolidates them.
        """
        logger.info("🌙 SEMANTIC SLEEP: Starting defragmentation cycle for '%s'...", self.collection_name)
        
        memory = ServiceContainer.get("vector_memory", default=None)
        if not memory or memory._fallback_mode:
            logger.warning("Semantic Defrag: Vector memory unavailable or in fallback mode. Skipping.")
            return

        try:
            # 1. Fetch a micro-batch of recent memories
            try:
                results = memory._collection.get(include=["documents", "metadatas"], limit=50)
            except TypeError:
                # Fallback if limit is not supported by older ChromaDB get()
                results = memory._collection.get(include=["documents", "metadatas"])
                
            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            
            # Slice to micro-batch if DB didn't support limit natively
            if len(ids) > 50:
                ids = ids[:50]
                docs = docs[:50]
                metas = metas[:50]
                
            if len(ids) < 10:
                logger.debug("Semantic Defrag: Not enough memories in micro-batch to justify defrag. Sleeping.")
                return

            # 2. Find high-similarity clusters in this micro-batch
            
            to_merge = [] # List of (doc_ids, shared_topic)
            
            checked = set()
            for i in range(len(ids)):
                if ids[i] in checked: continue
                
                # Exclude already consolidated concepts
                if metas[i].get("type") == "consolidated_concept":
                    checked.add(ids[i])
                    continue
                    
                # Find very similar docs to this one
                similars = memory.search_similar(docs[i], limit=5)
                cluster = [ids[i]]
                cluster_docs = [docs[i]]
                
                for sim in similars:
                    sim_id = self._get_id(sim)
                    if sim_id == ids[i]: continue
                    # threshold: distance < 0.1 (very similar in cosine space)
                    if sim.get("distance", 1.0) < 0.1 and sim.get("metadata", {}).get("type") != "consolidated_concept":
                        cluster.append(sim_id)
                        cluster_docs.append(sim.get("content", ""))
                        checked.add(sim_id)
                
                if len(cluster) > 2:
                    to_merge.append((cluster, cluster_docs))
                
                checked.add(ids[i])

            if not to_merge:
                logger.info("Semantic Defrag: No fragmentation clusters detected.")
                return

            # 3. Consolidate via LLM
            llm = ServiceContainer.get("llm_router", default=None)
            for cluster_ids, cluster_docs in to_merge:
                logger.info("🧠 Consolidating cluster of %s memories...", len(cluster_ids))
                
                context_block = "\n".join([f"- {d}" for d in cluster_docs])
                resolution_context = self._build_resolution_context(cluster_docs)
                sum_prompt = (
                    "Synthesize the following fragmented memories into a single, dense, factual consolidated concept. "
                    "Preserve all unique details but remove internal redundancies. Keep it under 100 words. "
                    "If technical facts conflict, keep the uncertainty instead of inventing a false precise answer."
                )

                full_req = f"{sum_prompt}\n\nMEMORIES:\n{context_block}"
                if resolution_context:
                    full_req = f"{sum_prompt}\n\nLIVE CHECKS:\n{resolution_context}\n\nMEMORIES:\n{context_block}"
                from core.brain.types import ThinkingMode
                response = await llm.think(
                    full_req, 
                    system_prompt="Memory Consolidation Subsystem.",
                    mode=ThinkingMode.FAST
                )
                consolidated_content = response.strip()
                
                if consolidated_content:
                    # Add new consolidated memory
                    meta = {
                        "type": "consolidated_concept",
                        "original_count": len(cluster_ids),
                        "timestamp": time.time(),
                        "last_accessed": time.time(),
                        "valence": sum(m.get("valence", 0) for m in metas if m.get("id") in cluster_ids) / len(cluster_ids)
                    }
                    memory.add_memory(consolidated_content, metadata=meta)
                    
                    # Delete fragmented originals
                    memory._collection.delete(ids=cluster_ids)
                    logger.info("✅ Successfully merged %s memories into a single Concept.", len(cluster_ids))

        except Exception as e:
            logger.error("Semantic Defrag failed: %s", e)

async def start_defrag_scheduler():
    """Continuous background daemon that runs micro-batch defrags periodically."""
    defragger = SemanticDefragmenter()
    while True:
        # Sleep for a short interval (e.g., 5 minutes) to run continuous micro-batches
        await asyncio.sleep(5 * 60)
        await defragger.run_defrag_cycle()