"""Every paragraph of her corpus, found by itself rather than through its article's name.

Her careful lookup (core/knowledge/passages_about.py) reads the articles a
question names. An answer in an article the question does not name never
reaches her: asked which ship a captain sailed, the answer may sit only in the
ship's article. Here each paragraph is its own row, split exactly as the
careful lookup splits an article, with the article's title beside it so a
paragraph that says "he" still carries its subject. BM25 then ranks paragraphs
across the whole corpus.

The index is contentless FTS5: it holds terms, not text. A paragraph's rowid
is its article's id shifted left by ``POSITION_BITS``, plus its position among
the article's paragraphs, and its text is read back from corpus.db. So the
paragraphs can never disagree with the corpus they came from, and the text is
not stored twice.

Building reads corpus.db in rowid order and records the last article written,
so an interrupted build resumes where it stopped.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("Aura.PassageIndex")

#: A position fits in this many bits. Corpus bodies are cut at 60,000
#: characters on ingest and a paragraph has at least four words, so an article
#: has far fewer than 2**16 paragraphs; one that had more would be refused.
POSITION_BITS = 16


def default_passage_db_path() -> Path:
    override = os.environ.get("AURA_PASSAGE_DB", "").strip()
    if override:
        return Path(override).expanduser()
    from core.knowledge.local_corpus import default_corpus_db_path

    return default_corpus_db_path().with_name("passages.db")


#: How many paragraphs BM25 scores a second here, replaced by what each search measures. The first
#: value is the rate measured on this host on 15 October 2026 (0.74 million a second, while the index
#: was still being written).
_SCORING = {"rows_per_s": 740_000.0}


def _paragraphs_holding(conn: sqlite3.Connection, words: list[str]) -> dict[str, int]:
    """For each word, how many paragraphs hold it, by the index's own stemming; a word it does not hold is left out.

    Each word is tokenized by a scratch table with the index's tokenizer, so
    its stem is the index's, and the count is read from the index's
    vocabulary.
    """
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.asked USING fts5(word, tokenize='porter unicode61')")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.asked_stems USING fts5vocab(temp, asked, 'instance')")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.held USING fts5vocab(main, passages_fts, 'row')")
    conn.execute("DELETE FROM temp.asked")
    conn.executemany("INSERT INTO temp.asked(rowid, word) VALUES (?, ?)", list(enumerate(words, start=1)))
    stems = {int(doc): str(term) for term, doc in conn.execute("SELECT term, doc FROM temp.asked_stems")}
    counts: dict[str, int] = {}
    for rowid, word in enumerate(words, start=1):
        stem = stems.get(rowid)
        if stem is None:
            continue
        row = conn.execute("SELECT doc FROM temp.held WHERE term = ?", (stem,)).fetchone()
        if row:
            counts[word] = int(row[0])
    return counts


@dataclass(frozen=True)
class Passage:
    title: str
    paragraph: str
    doc_id: int
    position: int
    rank: float


def _schema(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5("
                 "title, text, content='', tokenize='porter unicode61')")


def _articles(corpus_db: Path, after: int) -> Iterator[tuple[int, str, str]]:
    conn = sqlite3.connect(f"file:{corpus_db}?mode=ro", uri=True, timeout=10.0)
    try:
        cursor = conn.execute("SELECT d.id, d.title, f.body FROM docs AS d JOIN docs_fts AS f ON f.rowid = d.id "
                              "WHERE d.id > ? ORDER BY d.id", (after,))
        yield from cursor
    finally:
        conn.close()


def build(corpus_db: Path, passage_db: Path, *, until: int | None = None, commit_every: int = 20_000,
          on_progress: Callable[[dict], None] | None = None) -> dict:
    """Index the paragraphs of every article after the last one indexed, up to article id ``until``.

    ``commit_every`` articles go in one transaction; the last article written
    is recorded in the same transaction, so a stop loses at most that many.
    """
    from core.knowledge.passages_about import _paragraphs

    passage_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(passage_db))
    # The index can be rebuilt from corpus.db, so a crash mid-build costs a rebuild, never the corpus.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _schema(conn)
    row = conn.execute("SELECT value FROM meta WHERE key = 'last_doc'").fetchone()
    after = int(row[0]) if row else 0
    started = time.monotonic()
    articles = paragraphs = refused = 0
    last = after
    pending: list[tuple[int, str, str]] = []

    def flush() -> None:
        with conn:
            conn.executemany("INSERT INTO passages_fts(rowid, title, text) VALUES (?, ?, ?)", pending)
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('last_doc', ?)", (str(last),))
        pending.clear()

    try:
        for doc_id, title, body in _articles(corpus_db, after):
            if until is not None and doc_id > until:
                break
            split = _paragraphs(body)
            if len(split) >= 1 << POSITION_BITS:
                refused += 1
            else:
                pending.extend(((doc_id << POSITION_BITS) | position, title, text)
                               for position, text in enumerate(split))
                paragraphs += len(split)
            articles += 1
            last = doc_id
            if articles % commit_every == 0:
                flush()
                if on_progress is not None:
                    on_progress({"articles": articles, "paragraphs": paragraphs, "last_doc": last,
                                 "seconds": round(time.monotonic() - started, 1)})
        flush()
    finally:
        conn.close()
    return {"articles": articles, "paragraphs": paragraphs, "refused": refused, "last_doc": last,
            "seconds": round(time.monotonic() - started, 1)}


def optimize(passage_db: Path) -> float:
    """Merge the index's segments into one, so a lookup reads one b-tree instead of every segment; seconds taken.

    A build leaves the index in many segments, and FTS5 consults each of
    them for every word looked up. Merging rewrites the whole index once.
    """
    started = time.monotonic()
    conn = sqlite3.connect(str(passage_db))
    try:
        with conn:
            conn.execute("INSERT INTO passages_fts(passages_fts) VALUES ('optimize')")
    finally:
        conn.close()
    return round(time.monotonic() - started, 1)


class PassageIndex:
    """BM25 over every paragraph; text read back from corpus.db."""

    def __init__(self, passage_db: Path | str | None = None, corpus_db: Path | str | None = None) -> None:
        from core.knowledge.local_corpus import default_corpus_db_path

        self.passage_db = Path(passage_db) if passage_db else default_passage_db_path()
        self.corpus_db = Path(corpus_db) if corpus_db else default_corpus_db_path()

    def available(self) -> bool:
        return self.passage_db.exists() and self.corpus_db.exists()

    def search(self, question: str, *, limit: int, deadline_s: float) -> list[Passage]:
        """The paragraphs whose words best match the question's rarer words, best first; [] past the deadline.

        BM25 scores every paragraph holding any word asked for, and a common
        word is in millions of them. So the words go in rarest first, while
        the paragraphs they reach can still be scored in the time left at the
        rate searches here have been scoring; a word too common to fit, and
        every commoner one, is left out.
        """
        from core.knowledge.passages_about import _terms

        terms = sorted(_terms(question))
        if not terms or not self.available():
            return []
        conn = sqlite3.connect(f"file:{self.passage_db}?mode=ro", uri=True, timeout=10.0)
        expires = time.monotonic() + max(0.01, float(deadline_s))
        conn.set_progress_handler(lambda: 1 if time.monotonic() > expires else 0, 2000)
        try:
            counts = _paragraphs_holding(conn, terms)
            budget = max(0.0, expires - time.monotonic()) * _SCORING["rows_per_s"]
            chosen, reached = [], 0
            for word, held in sorted(counts.items(), key=lambda item: item[1]):
                if reached + held > budget:
                    break
                chosen.append(word)
                reached += held
            if not chosen:
                return []
            match = " OR ".join('"' + w.replace('"', '""') + '"' for w in chosen)
            started = time.monotonic()
            rows = conn.execute("SELECT rowid, bm25(passages_fts) FROM passages_fts WHERE passages_fts MATCH ? "
                                "ORDER BY bm25(passages_fts) LIMIT ?", (match, max(1, int(limit)))).fetchall()
            took = time.monotonic() - started
            if reached and took > 0:
                _SCORING["rows_per_s"] = reached / took
        except sqlite3.OperationalError as exc:
            # Interrupted at the deadline, or no index yet: a miss, as the corpus store treats it.
            logger.debug("Passage search returned nothing: %s", exc)
            return []
        finally:
            conn.set_progress_handler(None, 0)
            conn.close()
        return self._read_back(rows)

    def _read_back(self, rows: list[tuple[int, float]]) -> list[Passage]:
        from core.knowledge.passages_about import _paragraphs

        mask = (1 << POSITION_BITS) - 1
        wanted = [(rowid >> POSITION_BITS, rowid & mask, rank) for rowid, rank in rows]
        conn = sqlite3.connect(f"file:{self.corpus_db}?mode=ro", uri=True, timeout=10.0)
        found: list[Passage] = []
        split: dict[int, tuple[str, list[str]]] = {}
        try:
            for doc_id, position, rank in wanted:
                if doc_id not in split:
                    row = conn.execute("SELECT d.title, f.body FROM docs AS d JOIN docs_fts AS f ON f.rowid = d.id "
                                       "WHERE d.id = ?", (doc_id,)).fetchone()
                    split[doc_id] = (str(row[0]), list(_paragraphs(row[1]))) if row else ("", [])
                title, paragraphs = split[doc_id]
                if position < len(paragraphs):
                    found.append(Passage(title, paragraphs[position], doc_id, position, float(rank)))
        finally:
            conn.close()
        return found


def fused(rankings: list[list[tuple[str, str]]], *, limit: int) -> list[tuple[str, str]]:
    """Reciprocal rank fusion (Cormack, Clarke and Buettcher 2009) of several best-first lists of (title, paragraph).

    Its constant 60 is the paper's, which it found robust across collections.
    """
    score: dict[tuple[str, str], float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            score[item] = score.get(item, 0.0) + 1.0 / (60 + rank)
    return sorted(score, key=lambda item: -score[item])[:max(1, limit)]


__all__ = ["POSITION_BITS", "Passage", "PassageIndex", "build", "default_passage_db_path", "fused", "optimize"]

