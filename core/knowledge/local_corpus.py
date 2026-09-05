"""core/knowledge/local_corpus.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Local reference-knowledge substrate: offline corpora behind SQLite FTS5.

Why this exists: the serving model's weights cannot contain facts absent
from its training data, and a local system cannot call out to patch the
gap. This store gives the intentional-retrieval seam a REFERENCE lane —
grounded lookup over locally ingested corpora (Wikipedia, textbooks,
manuals) so base semantic gaps are answered by retrieval with provenance
instead of confabulation.

Design:
- SQLite FTS5 (BM25 ranking, porter stemming) — stdlib, offline, no model
  needed, scales to tens of GB on this hardware
- reads open short-lived read-only connections (URI mode=ro) so retrieval
  never contends with ingestion; writes serialize behind one lock
- FTS query strings are NEVER passed raw: user text is tokenized to plain
  terms (implicit AND, OR fallback) so FTS5 syntax cannot be injected
- honest misses: an empty result is an empty result; the caller's
  epistemic layer says "not in my corpus" rather than inventing

The bulk ingester lives in tools/knowledge_substrate/ingest_wikipedia.py.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Knowledge.LocalCorpus")

_SCHEMA_VERSION = 1

# Terms: words of 2+ chars, plus standalone single digits/letters —
# dropping them loses real discriminators ("Apollo 11" retrieval
# regression pinned by tests; "vitamin C" is the same class).
_TERM_RE = re.compile(r"[A-Za-z0-9_'-]{2,}|\b[A-Za-z0-9]\b")


def default_corpus_db_path() -> Path:
    override = os.environ.get("AURA_KNOWLEDGE_DB", "").strip()
    if override:
        return Path(override).expanduser()
    return state_root() / "knowledge" / "corpus.db"


#: The ceiling on any single corpus search. A backstop, not a budget.
#:
#: Generous on purpose: a legitimate any-term fallback over a 7M-page index is
#: genuinely slow, and an offline retrieval that a caller is willing to wait
#: for must not be cut off just because another caller is not. This only stops
#: a search running unboundedly.
SEARCH_DEADLINE_S = 5.0

#: What a caller answering a person in conversation should pass instead.
#:
#: A real topic resolves in 8-80ms. The any-term fallback on ordinary words
#: does not — "thanks, that helps" spent 3.0 SECONDS returning nothing —
#: so a turn that needed no reference at all was the slowest turn on the lane.
#: A conversation lane cannot spend longer looking for context than the answer
#: takes to produce, and the right ceiling is a property of the LANE rather
#: than of the index: an ingress pass building deep context legitimately waits
#: where a chat turn cannot.
CONVERSATION_SEARCH_DEADLINE_S = 0.25


@dataclass(frozen=True)
class CorpusHit:
    """One retrieval hit with provenance."""
    title: str
    snippet: str
    source: str
    rank: float          # BM25: lower is better
    doc_id: int

    def to_memory_dict(self) -> dict[str, Any]:
        """Shape consumed by the intentional-retrieval REFERENCE adapter."""
        return {
            "content": f"{self.title}: {self.snippet}",
            "metadata": {
                "store": "reference",
                "title": self.title,
                "source": self.source,
                "doc_id": self.doc_id,
                "provenance": "local_corpus",
            },
        }


class LocalCorpusStore:
    """FTS5-backed reference corpus. Thread-safe; writes serialized."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else default_corpus_db_path()
        self._write_lock = threading.Lock()
        self._ensure_schema()

    # ── schema ───────────────────────────────────────────────────────

    def _connect_rw(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _connect_ro(self) -> sqlite3.Connection:
        return sqlite3.connect(
            f"file:{self.db_path}?mode=ro", uri=True, timeout=10.0,
        )

    def _ensure_schema(self) -> None:
        if not self.db_path.parent.exists() and not self.db_path.exists():
            # Defer directory creation to first write; a read-only host
            # (tests probing absence) must not grow directories.
            return
        if not self.db_path.exists():
            return
        # Existing DB: nothing to do; schema created at first ingest.

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                ingested_at REAL NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                title, body, tokenize='porter unicode61'
            );
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )

    # ── ingestion ────────────────────────────────────────────────────

    def add_documents(self, docs: list[tuple[str, str, str]]) -> int:
        """Bulk-insert (title, body, source) rows in one transaction.

        Returns the number of rows written.
        """
        if not docs:
            return 0
        now = time.time()
        with self._write_lock:
            conn = self._connect_rw()
            try:
                self._create_schema(conn)
                cur = conn.cursor()
                written = 0
                for title, body, source in docs:
                    title = (title or "").strip()
                    body = (body or "").strip()
                    if not title or not body:
                        continue
                    cur.execute(
                        "INSERT INTO docs(title, source, ingested_at) VALUES(?,?,?)",
                        (title, source, now),
                    )
                    doc_id = cur.lastrowid
                    cur.execute(
                        "INSERT INTO docs_fts(rowid, title, body) VALUES(?,?,?)",
                        (doc_id, title, body),
                    )
                    written += 1
                conn.commit()
                return written
            finally:
                conn.close()

    def add_retained_document(
        self, title: str, body: str, *, artifact_id: str,
        source: str = "web_retained",
    ) -> bool:
        """Insert one verified retained-knowledge document, deduped by
        artifact ID — the corpus grows continuously from what Aura actually
        researches and verifies, not only from dump snapshots.

        Returns True if a new document was written.
        """
        artifact_key = f"retained:{artifact_id}"
        if self.get_meta(artifact_key):
            return False
        written = self.add_documents([(title, body, source)])
        if written:
            self.set_meta(artifact_key, "1")
        return bool(written)

    def set_meta(self, key: str, value: str) -> None:
        with self._write_lock:
            conn = self._connect_rw()
            try:
                self._create_schema(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)",
                    (key, value),
                )
                conn.commit()
            finally:
                conn.close()

    def get_meta(self, key: str, default: str = "") -> str:
        if not self.db_path.exists():
            return default
        try:
            conn = self._connect_ro()
        except sqlite3.OperationalError:
            return default
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
            return row[0] if row else default
        except sqlite3.OperationalError:
            return default
        finally:
            conn.close()

    # ── retrieval ────────────────────────────────────────────────────

    @staticmethod
    def _fts_query(text: str, *, any_term: bool = False) -> str:
        """Sanitize free text into an FTS5 match expression.

        Terms are extracted and double-quoted so FTS5 operators/syntax in
        the input cannot be injected; joined with implicit AND (or OR for
        the fallback pass).
        """
        terms = _TERM_RE.findall(text or "")[:16]
        if not terms:
            return ""
        quoted = [f'"{t}"' for t in terms]
        return " OR ".join(quoted) if any_term else " ".join(quoted)

    def search(
        self, query: str, limit: int = 5, *, deadline_s: float = SEARCH_DEADLINE_S
    ) -> list[CorpusHit]:
        """BM25 search; AND semantics with an OR fallback pass.

        Never raises on malformed input; missing/empty corpus returns [].

        Bounded in TIME, not just in rows. The any-term fallback exists so a
        query that shares no single document still returns something, and over
        a 7M-page index its posting lists for ordinary words are enormous:
        "thanks that helps" took 3.0 SECONDS to return nothing, while a real
        topic answers in 8-80ms. Any caller on a conversation lane pays that,
        so a turn that needed no reference at all was the slowest turn there
        was.

        The deadline is enforced with sqlite's own progress handler, which is
        the supported way to abandon a running statement. That bounds every
        caller rather than the one that noticed, and an abandoned search
        returns [] — the same as no match, which is what it is.
        """
        if not self.db_path.exists():
            return []
        match = self._fts_query(query)
        if not match:
            return []
        try:
            conn = self._connect_ro()
        except sqlite3.OperationalError:
            return []
        expires_at = time.monotonic() + max(0.01, float(deadline_s))
        # Checked every N VM instructions; small enough to be responsive, large
        # enough that the callback is not itself the cost.
        conn.set_progress_handler(lambda: 1 if time.monotonic() > expires_at else 0, 2000)
        try:
            rows = self._search_conn(conn, match, limit)
            if not rows and time.monotonic() < expires_at:
                fallback = self._fts_query(query, any_term=True)
                if fallback and fallback != match:
                    rows = self._search_conn(conn, fallback, limit)
            return rows
        finally:
            conn.set_progress_handler(None, 0)
            conn.close()

    def _search_conn(
        self, conn: sqlite3.Connection, match: str, limit: int,
    ) -> list[CorpusHit]:
        try:
            cursor = conn.execute(
                """
                SELECT f.rowid,
                       d.title,
                       d.source,
                       snippet(docs_fts, 1, '', '', ' … ', 40),
                       bm25(docs_fts)
                FROM docs_fts AS f
                JOIN docs AS d ON d.id = f.rowid
                WHERE docs_fts MATCH ?
                ORDER BY bm25(docs_fts)
                LIMIT ?
                """,
                (match, max(1, int(limit))),
            )
            return [
                CorpusHit(
                    doc_id=int(row[0]),
                    title=str(row[1]),
                    source=str(row[2]),
                    snippet=str(row[3]),
                    rank=float(row[4]),
                )
                for row in cursor.fetchall()
            ]
        except sqlite3.OperationalError as exc:
            # Missing FTS table (pre-ingest DB) or malformed match string
            # that survived sanitization — a miss, never a crash.
            logger.debug("Corpus search degraded to empty: %s", exc)
            return []

    # ── status ───────────────────────────────────────────────────────

    def document_count(self) -> int:
        if not self.db_path.exists():
            return 0
        try:
            conn = self._connect_ro()
        except sqlite3.OperationalError:
            return 0
        try:
            row = conn.execute("SELECT COUNT(*) FROM docs").fetchone()
            return int(row[0]) if row else 0
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()

    def has_documents(self) -> bool:
        """O(1) populated check. ``document_count()`` is a full table scan —
        at corpus scale it held the event loop for 5s when a hot path used
        it as a mere existence guard; this is the guard those paths want."""
        if not self.db_path.exists():
            return False
        try:
            conn = self._connect_ro()
        except sqlite3.OperationalError:
            return False
        try:
            return conn.execute("SELECT 1 FROM docs LIMIT 1").fetchone() is not None
        except sqlite3.OperationalError:
            return False
        finally:
            conn.close()

    def status(self) -> dict[str, Any]:
        exists = self.db_path.exists()
        return {
            "db_path": str(self.db_path),
            "exists": exists,
            "documents": self.document_count() if exists else 0,
            "size_mb": round(self.db_path.stat().st_size / 1_048_576, 1) if exists else 0.0,
            "sources": self.get_meta("sources", ""),
        }


# ── Module singleton ─────────────────────────────────────────────────

_store: LocalCorpusStore | None = None
_store_lock = threading.Lock()


def get_local_corpus_store(db_path: Path | str | None = None) -> LocalCorpusStore:
    """Process-wide corpus store; first caller binds the path."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = LocalCorpusStore(db_path)
    return _store
