"""
core/memory/semantic_defrag.py

Semantic sleep for vector memory consolidation.

This module is deliberately conservative: it only merges very tight clusters,
keeps every degradation receipt actionable, and never deletes source memories
unless a consolidated memory was written first.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.service_access import resolve_orchestrator

logger = logging.getLogger("Aura.Memory.Defrag")

DEFAULT_DEFRAG_INTERVAL_S = 5 * 60
MAX_DEFRAG_BATCH = 50
MIN_DEFRAG_BATCH = 10
MAX_CLUSTER_SIMILARS = 5
SIMILARITY_DISTANCE_THRESHOLD = 0.1
MAX_CONSOLIDATED_WORDS = 100
MAX_CONTEXT_DOC_CHARS = 800
MAX_CONTEXT_BLOCK_CHARS = 8_000

_SEMANTIC_DEFRAG_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_semantic_defrag_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "semantic_defrag",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation(
                "semantic_defrag",
                error,
                severity=severity,
                action=action,
            )
        except TypeError:
            logger.debug(
                "Semantic defrag degradation could not be recorded: %s",
                signature_exc,
            )


def _safe_text(value: object, *, max_chars: int = 4096) -> str:
    try:
        text = str(value if value is not None else "")
    except (RuntimeError, TypeError, ValueError):
        return ""
    return " ".join(text.replace("\x00", "").split())[:max_chars]


def _safe_float(value: object, *, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


class SemanticDefragmenter:
    FILE_REFERENCE_RE = re.compile(
        r"(?<![\w/.-])((?:[\w.-]+/)+[\w.-]+\.(?:py|tsx?|jsx?|json|md|ya?ml|toml|sh|go|rs|java|c|cpp|h))(?:[:#]\d+)?"
    )

    def __init__(
        self,
        collection_name: str = "aura_memories",
        *,
        interval_s: float = DEFAULT_DEFRAG_INTERVAL_S,
    ):
        self.collection_name = collection_name
        self.interval_s = max(1.0, float(interval_s))
        self.repo_root = Path(__file__).resolve().parents[2]
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self) -> bool:
        """Start managed semantic sleep scheduling on the active event loop."""
        if self._running and self._task is not None and not self._task.done():
            return True

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            _record_semantic_defrag_degradation(
                exc,
                action="semantic defrag scheduler not started because no event loop was running",
                severity="warning",
            )
            return False

        self._running = True
        try:
            from core.utils.task_tracker import get_task_tracker

            self._task = get_task_tracker().create_task(
                self._run_scheduler(),
                name="semantic_defrag.scheduler",
            )
        except _SEMANTIC_DEFRAG_ERRORS as exc:
            self._running = False
            _record_semantic_defrag_degradation(
                exc,
                action="semantic defrag scheduler not started because task tracker ownership failed",
                severity="warning",
            )
            return False
        logger.info("Semantic Defrag scheduler started for '%s'", self.collection_name)
        return True

    def stop(self) -> None:
        """Stop managed semantic sleep scheduling."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
        logger.info("Semantic Defrag scheduler stopped for '%s'", self.collection_name)

    async def _run_scheduler(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self.run_defrag_cycle()
            # not a failure: the wait ran out or the work was cancelled, which are the two
            # ways this bounded join ends without a result.
            except asyncio.CancelledError:
                return
            except _SEMANTIC_DEFRAG_ERRORS as exc:
                _record_semantic_defrag_degradation(
                    exc,
                    action="kept semantic defrag scheduler alive after one cycle failed",
                    severity="degraded",
                )

    def _get_id(self, item: dict[str, Any]) -> str:
        metadata = dict(item.get("metadata") or {})
        return _safe_text(item.get("id") or item.get("memory_id") or metadata.get("id") or metadata.get("created"))

    def _collect_file_references(self, docs: list[str]) -> list[str]:
        refs = set()
        for doc in docs:
            for match in self.FILE_REFERENCE_RE.finditer(doc or ""):
                refs.add(match.group(1))
        return sorted(refs)

    def _build_resolution_context(self, docs: list[str]) -> str:
        refs = self._collect_file_references(docs)
        if not refs:
            return ""

        existing: list[str] = []
        missing: list[str] = []
        for ref in refs[:6]:
            candidate = Path(ref)
            if not candidate.is_absolute():
                candidate = (self.repo_root / candidate).resolve()
            if candidate.exists():
                existing.append(ref)
            else:
                missing.append(ref)

        notes: list[str] = []
        if existing:
            notes.append("Live file check found existing references: " + ", ".join(existing))
        if missing:
            notes.append("Some remembered file references no longer exist: " + ", ".join(missing))
        if len(existing) + len(missing) > 1:
            notes.append("If the memories disagree on technical facts, preserve uncertainty rather than inventing a single confident file claim.")
        return "\n".join(notes)

    async def run_defrag_cycle(self) -> dict[str, Any]:
        """Scan vector memory for tight duplicate clusters and consolidate them."""
        logger.info("Semantic sleep: starting defragmentation cycle for '%s'", self.collection_name)

        block_reason = self._background_block_reason()
        if block_reason:
            logger.info("Semantic Defrag: deferred — %s.", block_reason)
            return {"status": "skipped", "reason": block_reason}

        memory = ServiceContainer.get("vector_memory", default=None)
        if not memory or getattr(memory, "_fallback_mode", False):
            logger.warning("Semantic Defrag: vector memory unavailable or in fallback mode; skipping.")
            return {"status": "skipped", "reason": "vector_memory_unavailable"}

        try:
            batch = await asyncio.to_thread(self._fetch_batch, memory)
            if len(batch) < MIN_DEFRAG_BATCH:
                logger.debug("Semantic Defrag: not enough memories in micro-batch to justify defrag.")
                return {"status": "skipped", "reason": "not_enough_memories", "count": len(batch)}

            clusters = await asyncio.to_thread(self._find_clusters, memory, batch)
            if not clusters:
                logger.info("Semantic Defrag: no fragmentation clusters detected.")
                return {"status": "completed", "clusters": 0, "merged": 0}

            llm = ServiceContainer.get("llm_router", default=None)
            merged = 0
            for cluster in clusters:
                if await self._consolidate_cluster(memory, llm, cluster):
                    merged += 1

            return {"status": "completed", "clusters": len(clusters), "merged": merged}
        except _SEMANTIC_DEFRAG_ERRORS as exc:
            _record_semantic_defrag_degradation(
                exc,
                action="aborted current semantic defrag cycle without deleting source memories",
                severity="degraded",
            )
            logger.error("Semantic Defrag failed: %s", exc)
            return {"status": "failed", "error": type(exc).__name__}

    def _background_block_reason(self) -> str:
        try:
            from core.runtime.background_policy import (
                MAINTENANCE_BACKGROUND_POLICY,
                background_activity_reason,
            )

            orchestrator = resolve_orchestrator()
            return background_activity_reason(
                orchestrator,
                profile=MAINTENANCE_BACKGROUND_POLICY,
                allow_no_user_anchor=True,
            )
        except _SEMANTIC_DEFRAG_ERRORS as exc:
            _record_semantic_defrag_degradation(
                exc,
                action="blocked semantic defrag because background-policy probe failed",
                severity="warning",
            )
            return "background_policy_unavailable"

    def _fetch_batch(self, memory: Any) -> list[dict[str, Any]]:
        collection = getattr(memory, "_collection", memory)
        get = getattr(collection, "get", None)
        if not callable(get):
            raise AttributeError("vector memory collection does not expose get()")

        try:
            results = get(include=["documents", "metadatas"], limit=MAX_DEFRAG_BATCH)
        except TypeError:
            results = get(include=["documents", "metadatas"])

        if not isinstance(results, dict):
            raise TypeError("vector memory collection returned a non-dict result")

        ids = list(results.get("ids") or [])
        docs = list(results.get("documents") or [])
        metas = list(results.get("metadatas") or [])
        usable = min(len(ids), len(docs), MAX_DEFRAG_BATCH)
        if usable < min(len(ids), len(docs)):
            _record_semantic_defrag_degradation(
                ValueError("vector memory batch exceeded semantic defrag limit"),
                action="trimmed semantic defrag batch to bounded micro-batch size",
                severity="warning",
                extra={"received_ids": len(ids), "received_docs": len(docs), "used": usable},
            )
        if len(metas) < usable:
            metas.extend({} for _ in range(usable - len(metas)))

        batch: list[dict[str, Any]] = []
        for index in range(usable):
            memory_id = _safe_text(ids[index], max_chars=128)
            content = _safe_text(docs[index], max_chars=MAX_CONTEXT_DOC_CHARS)
            metadata = metas[index] if isinstance(metas[index], dict) else {}
            if memory_id and content:
                batch.append({"id": memory_id, "content": content, "metadata": dict(metadata)})
        return batch

    def _find_clusters(self, memory: Any, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        search_similar = getattr(memory, "search_similar", None)
        if not callable(search_similar):
            raise AttributeError("vector memory does not expose search_similar()")

        metadata_by_id = {item["id"]: item["metadata"] for item in batch}
        docs_by_id = {item["id"]: item["content"] for item in batch}
        checked: set[str] = set()
        clusters: list[dict[str, Any]] = []

        for item in batch:
            memory_id = item["id"]
            if memory_id in checked:
                continue
            if item["metadata"].get("type") == "consolidated_concept":
                checked.add(memory_id)
                continue

            cluster_ids = [memory_id]
            cluster_docs = [item["content"]]
            try:
                similars = search_similar(item["content"], limit=MAX_CLUSTER_SIMILARS)
            except _SEMANTIC_DEFRAG_ERRORS as exc:
                _record_semantic_defrag_degradation(
                    exc,
                    action="skipped one semantic defrag seed after similarity search failed",
                    severity="warning",
                    extra={"memory_id": memory_id},
                )
                checked.add(memory_id)
                continue

            for similar in similars or []:
                if not isinstance(similar, dict):
                    continue
                similar_id = self._get_id(similar)
                if not similar_id or similar_id == memory_id or similar_id in checked:
                    continue
                similar_metadata = dict(similar.get("metadata") or {})
                if similar_metadata.get("type") == "consolidated_concept":
                    continue
                if _safe_float(similar.get("distance"), default=1.0) >= SIMILARITY_DISTANCE_THRESHOLD:
                    continue

                cluster_ids.append(similar_id)
                cluster_docs.append(
                    _safe_text(
                        similar.get("content") or docs_by_id.get(similar_id, ""),
                        max_chars=MAX_CONTEXT_DOC_CHARS,
                    )
                )
                checked.add(similar_id)
                metadata_by_id.setdefault(similar_id, similar_metadata)

            if len(cluster_ids) > 2:
                clusters.append(
                    {
                        "ids": cluster_ids,
                        "docs": [doc for doc in cluster_docs if doc],
                        "metadata_by_id": metadata_by_id,
                    }
                )
            checked.add(memory_id)

        return clusters

    async def _consolidate_cluster(self, memory: Any, llm: Any, cluster: dict[str, Any]) -> bool:
        cluster_ids = list(cluster["ids"])
        cluster_docs = list(cluster["docs"])
        logger.info("Consolidating cluster of %s memories", len(cluster_ids))

        consolidated_content = await self._llm_summary(llm, cluster_docs)
        merge_method = "llm"
        if not consolidated_content:
            consolidated_content = self._deterministic_summary(cluster_docs)
            merge_method = "deterministic"

        if not consolidated_content:
            _record_semantic_defrag_degradation(
                ValueError("empty semantic consolidation content"),
                action="left source memories untouched because no consolidated content was produced",
                severity="warning",
                extra={"cluster_size": len(cluster_ids)},
            )
            return False

        add_memory = getattr(memory, "add_memory", None)
        if not callable(add_memory):
            raise AttributeError("vector memory does not expose add_memory()")

        metadata = {
            "type": "consolidated_concept",
            "original_count": len(cluster_ids),
            "source_ids": cluster_ids,
            "timestamp": time.time(),
            "last_accessed": time.time(),
            "valence": self._cluster_valence(cluster_ids, cluster.get("metadata_by_id", {})),
            "merge_method": merge_method,
        }

        try:
            await asyncio.to_thread(add_memory, consolidated_content, metadata=metadata)
        except _SEMANTIC_DEFRAG_ERRORS as exc:
            _record_semantic_defrag_degradation(
                exc,
                action="left source memories untouched after consolidated memory write failed",
                severity="degraded",
                extra={"cluster_size": len(cluster_ids)},
            )
            return False

        delete = getattr(getattr(memory, "_collection", memory), "delete", None)
        if not callable(delete):
            _record_semantic_defrag_degradation(
                AttributeError("vector memory collection does not expose delete()"),
                action="wrote consolidated memory and kept source memories because delete is unavailable",
                severity="warning",
                extra={"cluster_size": len(cluster_ids)},
            )
            return True

        try:
            await asyncio.to_thread(delete, ids=cluster_ids)
            logger.info("Successfully merged %s memories into one concept", len(cluster_ids))
            return True
        except _SEMANTIC_DEFRAG_ERRORS as exc:
            _record_semantic_defrag_degradation(
                exc,
                action="wrote consolidated memory and kept source memories after delete failed",
                severity="warning",
                extra={"cluster_size": len(cluster_ids)},
            )
            return True

    async def _llm_summary(self, llm: Any, cluster_docs: list[str]) -> str:
        think = getattr(llm, "think", None)
        if not callable(think):
            return ""

        context_block = "\n".join(f"- {doc}" for doc in cluster_docs)[:MAX_CONTEXT_BLOCK_CHARS]
        resolution_context = self._build_resolution_context(cluster_docs)
        sum_prompt = (
            "Synthesize the following fragmented memories into a single, dense, factual consolidated concept. "
            "Preserve all unique details but remove internal redundancies. Keep it under 100 words. "
            "If technical facts conflict, keep the uncertainty instead of inventing a false precise answer."
        )
        full_request = f"{sum_prompt}\n\nMEMORIES:\n{context_block}"
        if resolution_context:
            full_request = f"{sum_prompt}\n\nLIVE CHECKS:\n{resolution_context}\n\nMEMORIES:\n{context_block}"

        try:
            from core.brain.types import ThinkingMode

            result = think(
                full_request,
                system_prompt="Memory Consolidation Subsystem.",
                mode=ThinkingMode.FAST,
            )
            response = await asyncio.wait_for(result, timeout=30.0) if asyncio.iscoroutine(result) else result
            return _safe_text(response, max_chars=1200)
        except _SEMANTIC_DEFRAG_ERRORS as exc:
            _record_semantic_defrag_degradation(
                exc,
                action="used deterministic consolidation after LLM summary failed",
                severity="warning",
                extra={"cluster_docs": len(cluster_docs)},
            )
            return ""

    def _deterministic_summary(self, cluster_docs: list[str]) -> str:
        words: list[str] = []
        seen: set[str] = set()
        for doc in cluster_docs:
            for word in _safe_text(doc, max_chars=MAX_CONTEXT_DOC_CHARS).split():
                normalized = word.lower().strip(".,;:()[]{}")
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                words.append(word)
                if len(words) >= MAX_CONSOLIDATED_WORDS:
                    return " ".join(words)
        return " ".join(words)

    def _cluster_valence(self, cluster_ids: list[str], metadata_by_id: dict[str, Any]) -> float:
        values: list[float] = []
        for memory_id in cluster_ids:
            metadata = metadata_by_id.get(memory_id)
            if isinstance(metadata, dict) and "valence" in metadata:
                values.append(_safe_float(metadata.get("valence"), default=0.0))
        return sum(values) / len(values) if values else 0.0


_defrag_running = True


async def start_defrag_scheduler() -> None:
    """Continuous background daemon that runs micro-batch defrags periodically."""
    defragger = SemanticDefragmenter()
    while _defrag_running:
        await asyncio.sleep(DEFAULT_DEFRAG_INTERVAL_S)
        await defragger.run_defrag_cycle()
