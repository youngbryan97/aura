"""Transactional, evidence-bound vector-memory consolidation.

The consolidator scans one complete storage namespace, forms deterministic
connected components under proved cosine embeddings, preserves every distinct
source text and provenance record, and commits each component with compare-and-
swap semantics. SQLite uses one database transaction. Chroma uses a durable
rollback journal plus a backend mutation lock and verified postconditions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Kernel.MemoryConsolidator")

_STORAGE_ERRORS = (
    OSError,
    ConnectionError,
    TimeoutError,
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
)
_SENSITIVE_CLASSES = {"credential", "credentials", "secret", "restricted", "legal_hold"}
_PRINCIPAL_KEYS = ("principal_id", "user_id", "owner_id", "tenant_id")
_NAMESPACE_KEYS = ("memory_namespace", "namespace")
_CLASS_KEYS = ("memory_class", "memory_type", "type", "node_type")


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    content: str
    metadata: dict[str, Any]
    embedding: tuple[float, ...]
    metric: str
    embedding_version: str
    scope: tuple[str, str, str]
    revision: str
    backend_revision: float | None = None


class ClusterMergeError(RuntimeError):
    def __init__(self, message: str, *, rolled_back: bool = False) -> None:
        super().__init__(message)
        self.rolled_back = rolled_back


@dataclass
class ConsolidationReport:
    memories_scanned: int = 0
    duplicates_merged: int = 0
    clusters_found: int = 0
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    pages_scanned: int = 0
    records_skipped: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    transactions_committed: int = 0
    transactions_rolled_back: int = 0
    scan_complete: bool = False
    backend: str = "unknown"
    receipts: list[dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Consolidation: scanned={self.memories_scanned}, "
            f"merged={self.duplicates_merged}, clusters={self.clusters_found}, "
            f"committed={self.transactions_committed}, complete={self.scan_complete} "
            f"({self.duration_s:.1f}s)"
        )


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


class MemoryConsolidator:
    """Consolidate duplicate memories without crossing evidence or owner boundaries."""

    JOURNAL_SCHEMA = "aura.memory.consolidation.rollback.v1"
    MAX_BATCH_SIZE = 10_000
    BLACK_HOLE_VECTOR_DIM = 384

    def __init__(
        self,
        vector_memory: Any = None,
        similarity_threshold: float = 0.97,
        batch_size: int = 100,
        *,
        principal_id: str | None = None,
        namespace: str | None = None,
        allow_unscoped: bool | None = None,
    ) -> None:
        threshold = float(similarity_threshold)
        if not math.isfinite(threshold) or not 0.5 <= threshold <= 1.0:
            raise ValueError("similarity_threshold must be finite and within [0.5, 1]")
        if isinstance(batch_size, bool) or not 1 <= int(batch_size) <= self.MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be within [1, {self.MAX_BATCH_SIZE}]")
        self.vector_memory = vector_memory
        self.similarity_threshold = threshold
        self.batch_size = int(batch_size)
        self.principal_id = str(principal_id or "").strip()
        self.namespace = str(namespace or "").strip()
        self.allow_unscoped = (
            bool(getattr(vector_memory, "single_principal_collection", False))
            if allow_unscoped is None
            else bool(allow_unscoped)
        )
        self._run_lock = checked_lock("memory_management")

    async def consolidate(self) -> ConsolidationReport:
        """Run blocking storage and similarity work outside the event loop."""
        report = await asyncio.to_thread(self._consolidate_sync)
        try:
            from core.thought_stream import get_emitter

            get_emitter().emit("Memory Consolidation", str(report), level="info")
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("memory_management", exc)
            logger.debug("Memory consolidation thought-stream emit skipped: %s", exc)
        return report

    def _consolidate_sync(self) -> ConsolidationReport:
        report = ConsolidationReport()
        started = time.monotonic()
        if self.vector_memory is None:
            report.errors.append("no_vector_memory")
            report.duration_s = time.monotonic() - started
            return report
        if not self._run_lock.acquire(blocking=False):
            report.errors.append("consolidation_already_running")
            report.duration_s = time.monotonic() - started
            return report

        try:
            lock = self._backend_lock()
            with lock:
                if not self._recover_pending_journal(report):
                    return report
                records = self._fetch_memories(report)
            report.memories_scanned = len(records) + report.records_skipped
            if not report.scan_complete or len(records) < 2:
                return report

            clusters = self._find_duplicate_clusters(records)
            report.clusters_found = len(clusters)
            for cluster in clusters:
                try:
                    receipt = self._merge_cluster(cluster)
                except ClusterMergeError as exc:
                    record_degradation("memory_management", exc)
                    if exc.rolled_back:
                        report.transactions_rolled_back += 1
                    report.errors.append(f"cluster_merge_failed:{type(exc).__name__}:{exc}")
                    continue
                except _STORAGE_ERRORS as exc:
                    record_degradation("memory_management", exc)
                    report.errors.append(f"cluster_merge_failed:{type(exc).__name__}:{exc}")
                    continue
                report.transactions_committed += 1
                report.duplicates_merged += len(receipt["loser_ids"])
                report.receipts.append(receipt)
        except _STORAGE_ERRORS as exc:
            record_degradation("memory_management", exc)
            report.errors.append(f"consolidation_failed:{type(exc).__name__}:{exc}")
        finally:
            self._run_lock.release()
            report.duration_s = time.monotonic() - started
            logger.info("%s", report)
        return report

    def _backend_lock(self) -> threading.RLock:
        lock = getattr(self.vector_memory, "_mutation_lock", None)
        if lock is None:
            lock = checked_lock("memory_management._backend_lock", reentrant=True)
            try:
                self.vector_memory._mutation_lock = lock
            except (AttributeError, TypeError):
                pass
        return lock

    def _backend_kind(self) -> str:
        declared = str(
            getattr(self.vector_memory, "memory_consolidation_backend", "") or ""
        ).strip()
        if declared == "black_hole_vault":
            memories = getattr(self.vector_memory, "memories", None)
            if isinstance(memories, list) and callable(
                getattr(self.vector_memory, "_save_vault", None)
            ):
                return declared
            return "unsupported"

        sqlite = getattr(self.vector_memory, "_sqlite_vectors", None)
        if sqlite is not None and self._has_methods(
            sqlite,
            "count",
            "iter_records",
            "merge_records_atomic",
        ):
            return "sqlite"
        collection = getattr(self.vector_memory, "_collection", None)
        if collection is not None and self._has_methods(
            collection,
            "count",
            "get",
            "update",
            "delete",
            "upsert",
        ):
            return "chroma"
        return "unsupported"

    @staticmethod
    def _has_methods(candidate: Any, *names: str) -> bool:
        return all(callable(getattr(candidate, name, None)) for name in names)

    def _fetch_memories(self, report: ConsolidationReport) -> list[MemoryRecord]:
        report.backend = self._backend_kind()
        if report.backend == "sqlite":
            raw, complete = self._fetch_sqlite(report)
        elif report.backend == "chroma":
            raw, complete = self._fetch_chroma(report)
        elif report.backend == "black_hole_vault":
            raw, complete = self._fetch_black_hole(report)
        else:
            report.errors.append("unsupported_memory_backend")
            report.scan_complete = False
            return []

        report.scan_complete = complete
        skipped: Counter[str] = Counter()
        records: list[MemoryRecord] = []
        for item in raw:
            try:
                records.append(self._validated_record(item, report.backend))
            except ValueError as exc:
                skipped[str(exc)] += 1
        report.records_skipped = sum(skipped.values())
        report.skipped_reasons = dict(sorted(skipped.items()))
        return sorted(records, key=lambda item: item.id)

    def _fetch_sqlite(self, report: ConsolidationReport) -> tuple[list[dict[str, Any]], bool]:
        store = self.vector_memory._sqlite_vectors
        collection = self._collection_name()
        expected = int(store.count(collection=collection))
        rows: list[dict[str, Any]] = []
        for row in store.iter_records(collection=collection, batch_size=self.batch_size):
            rows.append(dict(row))
            if len(rows) % self.batch_size == 1:
                report.pages_scanned += 1
        observed_after = int(store.count(collection=collection))
        complete = expected == observed_after == len(rows)
        if not complete:
            report.errors.append(
                f"scan_changed_during_read:before={expected}:read={len(rows)}:after={observed_after}"
            )
        return rows, complete

    def _fetch_chroma(self, report: ConsolidationReport) -> tuple[list[dict[str, Any]], bool]:
        collection = self.vector_memory._collection
        expected = int(collection.count())
        by_id: dict[str, dict[str, Any]] = {}
        offset = 0
        while offset < expected:
            result = collection.get(
                limit=min(self.batch_size, expected - offset),
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            ids = list(result.get("ids") or [])
            if not ids:
                break
            docs = list(result.get("documents") or [])
            metas = list(result.get("metadatas") or [])
            embeddings = result.get("embeddings")
            embeddings = list(embeddings) if embeddings is not None else []
            for index, record_id in enumerate(ids):
                by_id[str(record_id)] = {
                    "id": record_id,
                    "content": docs[index] if index < len(docs) else "",
                    "metadata": metas[index] if index < len(metas) else {},
                    "embedding": embeddings[index] if index < len(embeddings) else None,
                }
            offset += len(ids)
            report.pages_scanned += 1
        observed_after = int(collection.count())
        complete = expected == observed_after == len(by_id)
        if not complete:
            report.errors.append(
                f"scan_changed_or_incomplete:before={expected}:unique={len(by_id)}:after={observed_after}"
            )
        return list(by_id.values()), complete

    def _fetch_black_hole(
        self,
        report: ConsolidationReport,
    ) -> tuple[list[dict[str, Any]], bool]:
        ensure_ids = getattr(self.vector_memory, "_ensure_memory_ids", None)
        if callable(ensure_ids):
            ensure_ids(persist=True)
        memories = getattr(self.vector_memory, "memories", None)
        if not isinstance(memories, list):
            raise TypeError("black_hole_memories_not_a_list")
        expected = len(memories)
        snapshot = deepcopy(memories)
        raw: list[dict[str, Any]] = []
        memory_id = getattr(self.vector_memory, "_memory_id", None)
        for memory in snapshot:
            metadata = dict(memory.get("metadata") or {})
            metadata.setdefault("embedding_metric", "cosine")
            metadata.setdefault("embedding_version", "black-hole-tfhash384-v1")
            identifier = (
                str(memory_id(memory))
                if callable(memory_id)
                else str(memory.get("id") or memory.get("created") or "")
            )
            raw.append(
                {
                    "id": identifier,
                    "content": str(memory.get("text") or ""),
                    "metadata": metadata,
                    "embedding": self._black_hole_embedding(
                        str(memory.get("text") or ""),
                        memory.get("vec"),
                    ),
                }
            )
        report.pages_scanned = (
            math.ceil(expected / self.batch_size) if expected else 0
        )
        observed_after = len(getattr(self.vector_memory, "memories", []))
        complete = expected == observed_after == len(raw)
        if not complete:
            report.errors.append(
                f"scan_changed_during_read:before={expected}:read={len(raw)}:after={observed_after}"
            )
        return raw, complete

    @classmethod
    def _black_hole_embedding(
        cls,
        content: str,
        raw_vector: Any = None,
    ) -> list[float]:
        """Bound sparse lexical vectors without corpus-sized dense allocation."""
        if isinstance(raw_vector, Mapping):
            term_map: dict[str, float] = {}
            for term, raw_weight in raw_vector.items():
                try:
                    weight = float(raw_weight)
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(weight):
                    term_map[str(term)] = weight
        else:
            from core.memory.rag import compute_term_freq, tokenize

            term_map = compute_term_freq(tokenize(content))

        embedding = [0.0] * cls.BLACK_HOLE_VECTOR_DIM
        for term, weight in term_map.items():
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            encoded = int.from_bytes(digest, "big")
            slot = encoded % cls.BLACK_HOLE_VECTOR_DIM
            sign = -1.0 if encoded & (1 << 63) else 1.0
            embedding[slot] += sign * weight
        return embedding

    def _validated_record(self, raw: Mapping[str, Any], backend: str) -> MemoryRecord:
        record_id = str(raw.get("id") or "").strip()
        if not record_id:
            raise ValueError("missing_id")
        metadata_raw = raw.get("metadata")
        if not isinstance(metadata_raw, Mapping):
            raise ValueError("invalid_metadata")
        metadata = dict(metadata_raw)
        if metadata.get("legal_hold") is True or metadata.get("consolidation_allowed") is False:
            raise ValueError("governance_hold")
        memory_class = self._metadata_value(metadata, _CLASS_KEYS) or "general"
        sensitivity = str(metadata.get("sensitivity") or "").strip().casefold()
        if memory_class.casefold() in _SENSITIVE_CLASSES or sensitivity in _SENSITIVE_CLASSES:
            raise ValueError("sensitive_memory_class")

        collection = self._collection_name()
        record_namespace = self._metadata_value(metadata, _NAMESPACE_KEYS)
        if self.namespace and record_namespace and self.namespace != record_namespace:
            raise ValueError("namespace_scope_mismatch")
        namespace = self.namespace or record_namespace or collection
        record_principal = self._metadata_value(metadata, _PRINCIPAL_KEYS)
        if self.principal_id and record_principal and self.principal_id != record_principal:
            raise ValueError("principal_scope_mismatch")
        principal = self.principal_id or record_principal
        if not principal:
            if not self.allow_unscoped:
                raise ValueError("principal_scope_missing")
            principal = f"single-principal-collection:{collection}"
        scope = (principal, namespace, memory_class)

        metric = str(
            metadata.get("embedding_metric")
            or getattr(self.vector_memory, "embedding_metric", "")
            or self._collection_metric()
        ).strip().casefold()
        if metric != "cosine":
            raise ValueError("unproved_cosine_metric")
        version = str(
            metadata.get("embedding_version")
            or getattr(self.vector_memory, "embedding_version", "")
        ).strip()
        if not version:
            raise ValueError("embedding_version_missing")

        embedding = self._coerce_embedding(raw.get("embedding"), raw.get("dim"))
        content = str(raw.get("content") or "")
        revision = self._revision(record_id, content, metadata, embedding)
        backend_revision = raw.get("updated_at") if backend == "sqlite" else None
        return MemoryRecord(
            id=record_id,
            content=content,
            metadata=metadata,
            embedding=embedding,
            metric=metric,
            embedding_version=version,
            scope=scope,
            revision=revision,
            backend_revision=float(backend_revision) if backend_revision is not None else None,
        )

    @staticmethod
    def _metadata_value(metadata: Mapping[str, Any], keys: Sequence[str]) -> str:
        for key in keys:
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
        return ""

    def _collection_name(self) -> str:
        return str(getattr(self.vector_memory, "collection_name", "default") or "default")

    def _collection_metric(self) -> str:
        collection = getattr(self.vector_memory, "_collection", None)
        metadata = getattr(collection, "metadata", None)
        if isinstance(metadata, Mapping):
            return str(metadata.get("hnsw:space") or "")
        return ""

    @staticmethod
    def _coerce_embedding(value: Any, dim: Any = None) -> tuple[float, ...]:
        if isinstance(value, (bytes, bytearray, memoryview)):
            array = np.frombuffer(value, dtype=np.float32)
            if dim is not None and int(dim) != int(array.size):
                raise ValueError("embedding_dimension_mismatch")
        else:
            try:
                array = np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid_embedding") from exc
            if array.ndim != 1:
                raise ValueError("ragged_or_nondimensional_embedding")
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError("nonfinite_or_empty_embedding")
        norm = float(np.linalg.norm(array))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("zero_norm_embedding")
        return tuple(float(item) for item in array)

    @staticmethod
    def _revision(
        record_id: str,
        content: str,
        metadata: Mapping[str, Any],
        embedding: Sequence[float],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(record_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(metadata, sort_keys=True, default=str).encode("utf-8"))
        digest.update(np.asarray(embedding, dtype=np.float32).tobytes())
        return digest.hexdigest()

    def _find_duplicate_clusters(self, records: list[MemoryRecord]) -> list[list[MemoryRecord]]:
        grouped: dict[tuple[tuple[str, str, str], str, int], list[MemoryRecord]] = defaultdict(list)
        for record in records:
            grouped[(record.scope, record.embedding_version, len(record.embedding))].append(record)

        clusters: list[list[MemoryRecord]] = []
        block_size = min(self.batch_size, 512)
        for group in grouped.values():
            if len(group) < 2:
                continue
            matrix = np.asarray([record.embedding for record in group], dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1)
            normalized = matrix / norms[:, None]
            union = _UnionFind(len(group))
            for left_start in range(0, len(group), block_size):
                left_end = min(left_start + block_size, len(group))
                for right_start in range(left_start, len(group), block_size):
                    right_end = min(right_start + block_size, len(group))
                    similarities = (
                        normalized[left_start:left_end]
                        @ normalized[right_start:right_end].T
                    )
                    matches = np.argwhere(similarities >= self.similarity_threshold)
                    for left_offset, right_offset in matches:
                        left = left_start + int(left_offset)
                        right = right_start + int(right_offset)
                        if left < right:
                            union.union(left, right)
            components: dict[int, list[MemoryRecord]] = defaultdict(list)
            for index, record in enumerate(group):
                components[union.find(index)].append(record)
            clusters.extend(
                sorted(component, key=lambda item: item.id)
                for component in components.values()
                if len(component) > 1
            )
        return sorted(clusters, key=lambda cluster: tuple(item.id for item in cluster))

    def _merge_cluster(self, cluster: list[MemoryRecord]) -> dict[str, Any]:
        winner = max(cluster, key=self._winner_rank)
        losers = sorted((item for item in cluster if item.id != winner.id), key=lambda item: item.id)
        receipt_id = hashlib.sha256(
            "|".join(sorted(item.revision for item in cluster)).encode("ascii")
        ).hexdigest()
        merged_content = self._merged_content(winner, losers)
        merged_embedding = self._embed_merged_content(merged_content, winner)
        merged_metadata = self._merged_metadata(winner, losers, receipt_id)

        backend = self._backend_kind()
        with self._backend_lock():
            if backend == "sqlite":
                self._commit_sqlite(
                    winner,
                    losers,
                    merged_content,
                    merged_embedding,
                    merged_metadata,
                )
            elif backend == "chroma":
                self._commit_chroma(
                    winner,
                    losers,
                    merged_content,
                    merged_embedding,
                    merged_metadata,
                    receipt_id,
                )
            elif backend == "black_hole_vault":
                self._commit_black_hole(
                    winner,
                    losers,
                    merged_content,
                    merged_metadata,
                )
            else:
                raise RuntimeError("unsupported_memory_backend")

        return {
            "receipt_id": receipt_id,
            "backend": backend,
            "winner_id": winner.id,
            "loser_ids": [item.id for item in losers],
            "source_revisions": [item.revision for item in sorted(cluster, key=lambda item: item.id)],
            "content_sha256": hashlib.sha256(merged_content.encode("utf-8")).hexdigest(),
            "verified": True,
        }

    @staticmethod
    def _winner_rank(record: MemoryRecord) -> tuple[int, float, float, str]:
        importance = MemoryConsolidator._finite_number(record.metadata.get("importance"), 0.5)
        timestamp = MemoryConsolidator._finite_number(record.metadata.get("timestamp"), 0.0)
        return (len(record.content), importance, timestamp, record.id)

    @staticmethod
    def _finite_number(value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return number if math.isfinite(number) else default

    @staticmethod
    def _merged_content(winner: MemoryRecord, losers: list[MemoryRecord]) -> str:
        parts = [winner.content]
        seen = {" ".join(winner.content.split()).casefold()}
        for loser in losers:
            normalized = " ".join(loser.content.split()).casefold()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            parts.append(f"[Additional preserved context]\n{loser.content}")
        return "\n\n".join(parts)

    def _embed_merged_content(
        self,
        content: str,
        winner: MemoryRecord,
    ) -> tuple[float, ...]:
        if content == winner.content:
            return winner.embedding
        if self._backend_kind() == "black_hole_vault":
            return tuple(self._black_hole_embedding(content))
        if getattr(self.vector_memory, "_fallback_mode", False):
            from core.memory.vector_memory import AuraEmbeddingFunction

            embedded = AuraEmbeddingFunction()._pseudo_embed(content)
        else:
            embedder = getattr(self.vector_memory, "_embed_fn", None)
            if not callable(embedder):
                raise RuntimeError("merged_content_embedding_unavailable")
            result = embedder([content])
            if not isinstance(result, Sequence) or not result:
                raise RuntimeError("merged_content_embedding_failed")
            embedded = result[0]
        coerced = self._coerce_embedding(embedded)
        if len(coerced) != len(winner.embedding):
            raise RuntimeError("merged_content_embedding_dimension_changed")
        return coerced

    def _merged_metadata(
        self,
        winner: MemoryRecord,
        losers: list[MemoryRecord],
        receipt_id: str,
    ) -> dict[str, Any]:
        records = [winner, *losers]
        provenance = []
        for record in sorted(records, key=lambda item: item.id):
            source_metadata = deepcopy(record.metadata)
            prior_raw = source_metadata.pop("consolidation_provenance_json", "")
            prior: list[Any] = []
            if prior_raw:
                try:
                    parsed = json.loads(str(prior_raw))
                    if isinstance(parsed, list):
                        prior = parsed
                except (json.JSONDecodeError, TypeError, ValueError):
                    prior = [
                        {
                            "unparseable_prior_provenance_sha256": hashlib.sha256(
                                str(prior_raw).encode("utf-8")
                            ).hexdigest()
                        }
                    ]
            provenance.append(
                {
                    "id": record.id,
                    "content_sha256": hashlib.sha256(
                        record.content.encode("utf-8")
                    ).hexdigest(),
                    "metadata": source_metadata,
                    "metadata_sha256": hashlib.sha256(
                        json.dumps(record.metadata, sort_keys=True, default=str).encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    "revision": record.revision,
                    "prior_provenance": prior,
                }
            )
        importance = max(
            self._finite_number(record.metadata.get("importance"), 0.5)
            for record in records
        )
        previous_merged = int(max(0.0, self._finite_number(winner.metadata.get("merged_count"), 0.0)))
        metadata = deepcopy(winner.metadata)
        metadata.update(
            {
                "importance": min(1.0, importance * 1.1),
                "merged_count": previous_merged + len(losers),
                "consolidation_receipt_id": receipt_id,
                "consolidated_at": time.time(),
                "consolidation_provenance_json": json.dumps(
                    provenance, sort_keys=True, separators=(",", ":"), default=str
                ),
                "embedding_metric": winner.metric,
                "embedding_version": winner.embedding_version,
            }
        )
        return metadata

    def _commit_sqlite(
        self,
        winner: MemoryRecord,
        losers: list[MemoryRecord],
        content: str,
        embedding: tuple[float, ...],
        metadata: dict[str, Any],
    ) -> None:
        revisions = {item.id: item.backend_revision for item in [winner, *losers]}
        if any(value is None for value in revisions.values()):
            raise RuntimeError("sqlite_revision_missing")
        result = self.vector_memory._sqlite_vectors.merge_records_atomic(
            winner_id=winner.id,
            loser_ids=[item.id for item in losers],
            content=content,
            embedding=embedding,
            metadata=metadata,
            expected_updated_at={key: float(value) for key, value in revisions.items()},
            collection=self._collection_name(),
        )
        if result.get("deleted") != len(losers):
            raise RuntimeError("sqlite_merge_postcondition_failed")
        store = getattr(self.vector_memory, "_store", None)
        if isinstance(store, list):
            try:
                self.vector_memory._store = self.vector_memory._sqlite_vectors.list_records(
                    collection=self._collection_name()
                )
            except _STORAGE_ERRORS as exc:
                record_degradation(
                    "memory_management",
                    exc,
                    action="durable SQLite merge committed; in-memory cache refresh deferred",
                )
                logger.warning("SQLite memory cache refresh failed after commit: %s", exc)

    def _commit_chroma(
        self,
        winner: MemoryRecord,
        losers: list[MemoryRecord],
        content: str,
        embedding: tuple[float, ...],
        metadata: dict[str, Any],
        receipt_id: str,
    ) -> None:
        collection = self.vector_memory._collection
        originals = self._get_chroma_records([winner.id, *(item.id for item in losers)])
        expected = {item.id: item.revision for item in [winner, *losers]}
        if set(originals) != set(expected):
            raise RuntimeError("chroma_cluster_changed_before_merge")
        for record_id, record in originals.items():
            if record.revision != expected[record_id]:
                raise RuntimeError(f"chroma_revision_changed_before_merge:{record_id}")

        journal_path = self._journal_path()
        if journal_path is None:
            raise RuntimeError("chroma_rollback_journal_path_unavailable")
        self._write_journal(journal_path, receipt_id, list(originals.values()))
        try:
            collection.update(
                ids=[winner.id],
                documents=[content],
                metadatas=[metadata],
                embeddings=[list(embedding)],
            )
            strengthened = self._get_chroma_records([winner.id]).get(winner.id)
            if strengthened is None or strengthened.metadata.get("consolidation_receipt_id") != receipt_id:
                raise RuntimeError("chroma_winner_update_not_verified")
            collection.delete(ids=[item.id for item in losers])
            final = self._get_chroma_records([winner.id, *(item.id for item in losers)])
            if set(final) != {winner.id}:
                raise RuntimeError("chroma_delete_postcondition_failed")
            journal_path.unlink()
        except _STORAGE_ERRORS as exc:
            self._restore_chroma_records(list(originals.values()))
            restored = self._get_chroma_records(list(originals))
            if set(restored) != set(originals) or any(
                restored[key].revision != originals[key].revision for key in originals
            ):
                raise RuntimeError("chroma_rollback_verification_failed") from exc
            journal_path.unlink(missing_ok=True)
            raise ClusterMergeError(
                f"chroma transaction rolled back after {type(exc).__name__}: {exc}",
                rolled_back=True,
            ) from exc

    def _commit_black_hole(
        self,
        winner: MemoryRecord,
        losers: list[MemoryRecord],
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        vault = self.vector_memory
        memories = getattr(vault, "memories", None)
        memory_id = getattr(vault, "_memory_id", None)
        if not isinstance(memories, list) or not callable(memory_id):
            raise RuntimeError("black_hole_transaction_contract_missing")

        originals = deepcopy(memories)
        by_id = {str(memory_id(memory)): memory for memory in memories}
        expected = {item.id: item.revision for item in [winner, *losers]}
        if not set(expected).issubset(by_id):
            raise RuntimeError("black_hole_cluster_changed_before_merge")

        current_raw, complete = self._fetch_black_hole(ConsolidationReport())
        if not complete:
            raise RuntimeError("black_hole_cluster_changed_before_merge")
        current = {
            item.id: item
            for item in (
                self._validated_record(raw, "black_hole_vault")
                for raw in current_raw
            )
        }
        if any(
            current.get(record_id) is None
            or current[record_id].revision != revision
            for record_id, revision in expected.items()
        ):
            raise RuntimeError("black_hole_revision_changed_before_merge")

        loser_ids = {item.id for item in losers}
        replacement = deepcopy(by_id[winner.id])
        replacement.update(
            {
                "id": winner.id,
                "text": content,
                "metadata": metadata,
            }
        )
        from core.memory.rag import compute_term_freq, tokenize

        replacement["vec"] = compute_term_freq(tokenize(content))
        vault.memories = [
            replacement if str(memory_id(memory)) == winner.id else memory
            for memory in memories
            if str(memory_id(memory)) not in loser_ids
        ]
        vault._dirty = True
        try:
            vault._save_vault()
            final_ids = {str(memory_id(memory)) for memory in vault.memories}
            final_winner = next(
                memory
                for memory in vault.memories
                if str(memory_id(memory)) == winner.id
            )
            if loser_ids & final_ids or final_winner.get("metadata", {}).get(
                "consolidation_receipt_id"
            ) != metadata.get("consolidation_receipt_id"):
                raise RuntimeError("black_hole_merge_postcondition_failed")
        except _STORAGE_ERRORS as exc:
            vault.memories = originals
            vault._dirty = True
            try:
                vault._save_vault()
            except _STORAGE_ERRORS as rollback_exc:
                raise RuntimeError("black_hole_rollback_failed") from rollback_exc
            raise ClusterMergeError(
                f"black-hole transaction rolled back after {type(exc).__name__}: {exc}",
                rolled_back=True,
            ) from exc

    def _get_chroma_records(self, ids: Sequence[str]) -> dict[str, MemoryRecord]:
        result = self.vector_memory._collection.get(
            ids=list(ids),
            include=["documents", "metadatas", "embeddings"],
        )
        raw_ids = list(result.get("ids") or [])
        docs = list(result.get("documents") or [])
        metas = list(result.get("metadatas") or [])
        raw_embeddings = result.get("embeddings")
        embeddings = list(raw_embeddings) if raw_embeddings is not None else []
        records: dict[str, MemoryRecord] = {}
        for index, record_id in enumerate(raw_ids):
            raw = {
                "id": record_id,
                "content": docs[index] if index < len(docs) else "",
                "metadata": metas[index] if index < len(metas) else {},
                "embedding": embeddings[index] if index < len(embeddings) else None,
            }
            record = self._validated_record(raw, "chroma")
            records[record.id] = record
        return records

    def _journal_path(self) -> Path | None:
        root = getattr(self.vector_memory, "persist_directory", None)
        if root is None:
            return None
        return Path(root) / f".{self._collection_name()}.consolidation-rollback.json"

    def _write_journal(
        self,
        path: Path,
        receipt_id: str,
        records: list[MemoryRecord],
    ) -> None:
        payload = {
            "schema": self.JOURNAL_SCHEMA,
            "receipt_id": receipt_id,
            "collection": self._collection_name(),
            "records": [
                {
                    "id": record.id,
                    "content": record.content,
                    "metadata": record.metadata,
                    "embedding": list(record.embedding),
                    "revision": record.revision,
                }
                for record in records
            ],
        }
        atomic_write_text(
            path,
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
            durable=True,
            mode=0o600,
        )

    def _recover_pending_journal(self, report: ConsolidationReport) -> bool:
        if self._backend_kind() != "chroma":
            return True
        path = self._journal_path()
        if path is None or not path.exists():
            return True
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != self.JOURNAL_SCHEMA:
                raise ValueError("unknown consolidation journal schema")
            records = [self._validated_record(item, "chroma") for item in payload["records"]]
            self._restore_chroma_records(records)
            restored = self._get_chroma_records([item.id for item in records])
            if len(restored) != len(records) or any(
                restored[item.id].revision != item.revision for item in records
            ):
                raise RuntimeError("journal_recovery_verification_failed")
            path.unlink()
            report.transactions_rolled_back += 1
            report.receipts.append(
                {
                    "receipt_id": payload.get("receipt_id"),
                    "backend": "chroma",
                    "recovered": True,
                    "verified": True,
                }
            )
            return True
        except _STORAGE_ERRORS as exc:
            record_degradation("memory_management", exc, severity="critical")
            report.errors.append(f"rollback_journal_recovery_failed:{type(exc).__name__}:{exc}")
            return False

    def _restore_chroma_records(self, records: list[MemoryRecord]) -> None:
        self.vector_memory._collection.upsert(
            ids=[item.id for item in records],
            documents=[item.content for item in records],
            metadatas=[item.metadata for item in records],
            embeddings=[list(item.embedding) for item in records],
        )
