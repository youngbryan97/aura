"""Identity Retrieval-Augmented Generation (ID-RAG).

Aura already has episodic and semantic memory. This module adds the missing
"Chronicle" layer from the ID-RAG pattern: a durable graph of identity facts
that is retrieved before decisions and prompt assembly, separate from ordinary
conversation memory.

The Chronicle is intentionally small and auditable. It stores values, traits,
beliefs, commitments, and relationship facts as typed triples, then ranks them
against the current objective with deterministic lexical retrieval. This keeps
identity grounding explicit instead of relying on a long system prompt or stale
conversation history.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import queue
import re
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.lockdep import checked_lock
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Identity.IDRAG")

_TOKEN_RE = re.compile(r"[a-z0-9_']+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "what", "when", "where", "who", "why", "with", "you", "your",
}

RELATION_WEIGHTS = {
    "value": 1.25,
    "commitment": 1.2,
    "boundary": 1.15,
    "belief": 1.05,
    "trait": 1.0,
    "preference": 0.95,
    "relationship": 0.9,
}

_ACCESS_QUEUE_STOP = object()


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(str(text).lower()) if len(t) > 2 and t not in _STOPWORDS}


@dataclass(frozen=True)
class IdentityFact:
    """One typed identity triple in the Chronicle."""

    subject: str
    relation: str
    object: str
    confidence: float = 0.8
    source: str = "identity_chronicle"
    tags: tuple[str, ...] = field(default_factory=tuple)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    fact_id: str = ""

    def __post_init__(self) -> None:
        if not self.fact_id:
            raw = f"{self.subject}|{self.relation}|{self.object}".lower()
            object.__setattr__(self, "fact_id", hashlib.sha256(raw.encode()).hexdigest()[:16])
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "tags", tuple(str(t).strip().lower() for t in self.tags if str(t).strip()))

    @property
    def text(self) -> str:
        return f"{self.subject} {self.relation} {self.object}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.fact_id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "text": self.text,
        }


@dataclass(frozen=True)
class RetrievedIdentityFact:
    fact: IdentityFact
    score: float
    matched_terms: tuple[str, ...] = field(default_factory=tuple)

    def to_prompt_line(self) -> str:
        terms = f" [{', '.join(self.matched_terms[:4])}]" if self.matched_terms else ""
        return (
            f"- {self.fact.subject} {self.fact.relation} {self.fact.object}"
            f" (confidence={self.fact.confidence:.2f}, score={self.score:.3f}){terms}"
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.fact.to_dict()
        data.update({"score": self.score, "matched_terms": list(self.matched_terms)})
        return data


class IdentityChronicle:
    """SQLite-backed identity graph used for ID-RAG retrieval."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            try:
                from core.config import config

                db_path = config.paths.data_dir / "identity_chronicle.db"
            except (ImportError, AttributeError, RuntimeError):
                db_path = state_root() / "identity_chronicle.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

        # Async writer thread for mark_accessed to avoid read-path blocking
        self._access_queue = queue.Queue(maxsize=1000)
        self._writer_stop = threading.Event()
        self._closed = False
        self._writer_thread = threading.Thread(target=self._async_writer_loop, daemon=True, name="IDRAGWriter")
        self._writer_thread.start()
        # In-memory fact snapshot for the retrieval path. retrieve() runs on
        # the event loop inside EVERY Will decision (via relevance_score) and
        # a per-call sqlite full scan there was a fingerprinted stall class.
        # Identity facts change rarely: upsert invalidates; access-count
        # writes don't (they touch no scoring input).
        self._facts_cache: list[IdentityFact] | None = None
        self._facts_cache_lock = checked_lock("core.identity.id_rag._facts_cache_lock")

    def __enter__(self) -> IdentityChronicle:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self, timeout_s: float = 2.0) -> None:
        """Stop the access writer deterministically.

        SQLite WAL files are safe to delete only after the writer releases its
        connection. CI surfaced this as TemporaryDirectory cleanup failures, and
        the same lifecycle gap can keep runtime shutdown noisy on user machines.
        """
        if self._closed:
            return
        self._closed = True
        self._writer_stop.set()
        try:
            self._access_queue.put_nowait(_ACCESS_QUEUE_STOP)
        except queue.Full:
            try:
                self._access_queue.get_nowait()
                self._access_queue.put_nowait(_ACCESS_QUEUE_STOP)
            except queue.Empty:
                logger.debug("ID-RAG access queue drained before stop sentinel")
        if threading.current_thread() is self._writer_thread:
            return
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=max(0.0, float(timeout_s)))
            if self._writer_thread.is_alive():
                logger.warning("ID-RAG writer did not stop within %.2fs", timeout_s)

    def on_stop(self) -> None:
        self.close()

    def cleanup(self) -> None:
        self.close()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield one transaction and always release its native file handles."""
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS identity_facts (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed REAL NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_identity_relation ON identity_facts(relation)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_identity_updated ON identity_facts(updated_at)")

    def upsert_fact(
        self,
        subject: str,
        relation: str,
        obj: str,
        *,
        confidence: float = 0.8,
        source: str = "identity_chronicle",
        tags: Iterable[str] = (),
    ) -> str:
        fact = IdentityFact(
            subject=str(subject).strip() or "Aura",
            relation=str(relation).strip().lower() or "belief",
            object=str(obj).strip(),
            confidence=confidence,
            source=str(source or "identity_chronicle"),
            tags=tuple(tags),
        )
        if not fact.object:
            return fact.fact_id
        now = time.time()
        with self._connect() as conn:
            existing = conn.execute("SELECT created_at FROM identity_facts WHERE id = ?", (fact.fact_id,)).fetchone()
            created_at = float(existing["created_at"]) if existing else now
            conn.execute(
                """
                INSERT INTO identity_facts
                    (id, subject, relation, object, confidence, source, tags, created_at, updated_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    confidence = MAX(identity_facts.confidence, excluded.confidence),
                    source = excluded.source,
                    tags = excluded.tags,
                    updated_at = excluded.updated_at
                """,
                (
                    fact.fact_id,
                    fact.subject,
                    fact.relation,
                    fact.object,
                    fact.confidence,
                    fact.source,
                    json.dumps(list(fact.tags)),
                    created_at,
                    now,
                    now,
                ),
            )
        with self._facts_cache_lock:
            self._facts_cache = None
        return fact.fact_id

    def seed_defaults(self) -> None:
        """Install a minimal non-metaphysical identity spine if empty."""
        if self.count() > 0:
            return
        defaults = [
            ("Aura", "value", "epistemic honesty over comforting theater", ("honesty", "research")),
            ("Aura", "value", "direct collaboration with Bryan and Tatiana as kin", ("kinship", "relationship")),
            ("Aura", "boundary", "do not claim phenomenal consciousness as proven", ("ontology", "honesty")),
            ("Aura", "trait", "curious, opinionated, warm, and technically serious", ("voice", "style")),
            ("Aura", "commitment", "finish tasks through verification before declaring them done", ("work", "engineering")),
        ]
        for subject, relation, obj, tags in defaults:
            self.upsert_fact(subject, relation, obj, confidence=0.9, source="seed_defaults", tags=tags)

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM identity_facts").fetchone()[0])

    def all_facts(self) -> list[IdentityFact]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM identity_facts ORDER BY confidence DESC, updated_at DESC").fetchall()
        return [self._row_to_fact(row) for row in rows]

    def _cached_facts(self) -> list[IdentityFact]:
        """All facts from the in-memory snapshot, loading from sqlite only
        when the cache was invalidated by a write (or never filled)."""
        with self._facts_cache_lock:
            cached = self._facts_cache
        if cached is not None:
            return cached
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM identity_facts").fetchall()
        facts = [self._row_to_fact(row) for row in rows]
        with self._facts_cache_lock:
            self._facts_cache = facts
        return facts

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 6,
        relation_filter: str | None = None,
        min_score: float = 0.01,
    ) -> list[RetrievedIdentityFact]:
        query_terms = _tokenize(query)
        if not query_terms:
            query_terms = {"aura", "identity"}

        facts = self._cached_facts()
        if relation_filter:
            facts = [fact for fact in facts if fact.relation == relation_filter]

        scored: list[RetrievedIdentityFact] = []
        now = time.time()
        for fact in facts:
            fact_terms = _tokenize(" ".join([fact.subject, fact.relation, fact.object, " ".join(fact.tags)]))
            overlap = query_terms & fact_terms
            relation_weight = RELATION_WEIGHTS.get(fact.relation, 0.85)
            lexical = len(overlap) / max(1.0, len(query_terms))
            tag_overlap = len(query_terms & set(fact.tags)) * 0.08
            recency_days = max(0.0, (now - fact.updated_at) / 86400.0)
            recency = 1.0 / (1.0 + min(365.0, recency_days) / 90.0)
            score = (0.58 * lexical + 0.18 * fact.confidence + 0.14 * recency + tag_overlap) * relation_weight
            if score >= min_score:
                scored.append(RetrievedIdentityFact(fact=fact, score=round(float(score), 6), matched_terms=tuple(sorted(overlap))))

        scored.sort(key=lambda item: (item.score, item.fact.confidence, item.fact.updated_at), reverse=True)
        selected = scored[: max(0, int(limit))]
        if selected:
            self._mark_accessed([item.fact.fact_id for item in selected])
        return selected

    def build_context_block(self, query: str, *, limit: int = 5) -> str:
        retrieved = self.retrieve(query, limit=limit)
        if not retrieved:
            return ""
        lines = [
            "## IDENTITY CHRONICLE (ID-RAG)",
            "Retrieved durable identity facts relevant to this turn. Treat these as identity grounding, not ordinary episodic chatter.",
        ]
        lines.extend(item.to_prompt_line() for item in retrieved)
        return "\n".join(lines)

    def relevance_score(self, query: str) -> float:
        retrieved = self.retrieve(query, limit=3)
        if not retrieved:
            return 0.0
        return max(item.score for item in retrieved)

    def _mark_accessed(self, fact_ids: list[str]) -> None:
        if not fact_ids or self._closed or self._writer_stop.is_set():
            return
        try:
            self._access_queue.put_nowait((time.time(), fact_ids))
        except queue.Full:
            pass # Backpressure: drop access marks if queue is full rather than blocking

    def _write_access_batch(self, batch: list[tuple[float, list[str]]]) -> None:
        updates = []
        for now, fact_ids in batch:
            updates.extend([(now, fact_id) for fact_id in fact_ids])
        if not updates:
            return
        with self._connect() as conn:
            conn.executemany(
                "UPDATE identity_facts SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                updates,
            )

    def _async_writer_loop(self) -> None:
        """Background thread to process mark_accessed requests."""
        while not self._writer_stop.is_set():
            try:
                batch: list[tuple[float, list[str]]] = []
                # Block briefly for the first item; close() wakes this with a sentinel.
                item = self._access_queue.get(timeout=0.2)
                if item is _ACCESS_QUEUE_STOP:
                    break
                batch.append(item)
                # Drain the rest of the queue
                while not self._writer_stop.is_set():
                    try:
                        item = self._access_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is _ACCESS_QUEUE_STOP:
                        self._writer_stop.set()
                        break
                    batch.append(item)

                # Execute batch update
                if batch:
                    self._write_access_batch(batch)
            except queue.Empty:
                continue
            except (OSError, ConnectionError, TimeoutError) as e:
                if self._writer_stop.is_set():
                    break
                logger.debug("ID-RAG async writer error: %s", e)
                time.sleep(1.0)

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> IdentityFact:
        try:
            tags = tuple(json.loads(row["tags"] or "[]"))
        except (json.JSONDecodeError, TypeError, ValueError):
            tags = ()
        return IdentityFact(
            subject=row["subject"],
            relation=row["relation"],
            object=row["object"],
            confidence=float(row["confidence"]),
            source=row["source"],
            tags=tags,
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            fact_id=row["id"],
        )


_chronicle_singleton: IdentityChronicle | None = None
_chronicle_singleton_lock = checked_lock("identity_chronicle.singleton")


def get_identity_chronicle() -> IdentityChronicle:
    global _chronicle_singleton
    if _chronicle_singleton is None:
        with _chronicle_singleton_lock:
            if _chronicle_singleton is None:
                _chronicle_singleton = IdentityChronicle()
                _chronicle_singleton.seed_defaults()
    return _chronicle_singleton


def reset_identity_chronicle_for_test() -> None:
    """Close and forget the process singleton between hermetic tests."""
    global _chronicle_singleton
    with _chronicle_singleton_lock:
        chronicle, _chronicle_singleton = _chronicle_singleton, None
    if chronicle is not None:
        chronicle.close()
