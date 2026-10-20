"""Every paragraph of her corpus, found by itself rather than through its article's name.

Her careful lookup (core/knowledge/passages_about.py) reads the articles a
question names. An answer in an article the question does not name never
reaches her: asked which ship a captain sailed, the answer may sit only in the
ship's article. Here each paragraph is its own document, split exactly as the
careful lookup splits an article, with the article's title beside it so a
paragraph that says "he" still carries its subject. BM25 then ranks paragraphs
across the whole corpus.

The index is Tantivy's (the Rust search library modelled on Lucene). Its
top-k search over a union of words skips every block of postings that cannot
reach the current k best (block-max WAND), so a question over a hundred
million paragraphs answers in milliseconds with every one of its words. The
SQLite FTS5 index this replaced scores every paragraph holding any word, and
took a median 3.3 s a question at 5 s allowed (15 October).

The index holds terms and one number per paragraph: its article's id shifted
left by ``POSITION_BITS``, plus its position among the article's paragraphs.
Its text is read back from corpus.db, so the paragraphs can never disagree
with the corpus they came from. Building reads corpus.db in id order and
records the last article committed, so an interrupted build resumes there.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.flags import FlagKind, declare

_FLAG_PASSAGE_INDEX = declare("AURA_PASSAGE_INDEX", kind=FlagKind.STRING, default='',
                              description="where her paragraph index is, when not the default place", owner="knowledge")

logger = logging.getLogger("Aura.PassageIndex")

#: A position fits in this many bits. Corpus bodies are cut at 60,000
#: characters on ingest and a paragraph has at least four words, so an article
#: has far fewer than 2**16 paragraphs; one that had more would be refused.
POSITION_BITS = 16
#: Where a build records the last article it committed.
_PROGRESS = "aura_progress.json"
#: The query parser's own syntax; a question's words are searched for, never parsed as operators.
_NOT_A_WORD = re.compile(r"[^\w\s]+", re.UNICODE)


def default_passage_index_path() -> Path:
    override = _FLAG_PASSAGE_INDEX.value().strip()
    if override:
        return Path(override).expanduser()
    from core.knowledge.local_corpus import default_corpus_db_path

    return default_corpus_db_path().with_name("paragraphs.tantivy")


@dataclass(frozen=True)
class Passage:
    title: str
    paragraph: str
    doc_id: int
    position: int
    rank: float


def _schema() -> Any:
    import tantivy

    built = tantivy.SchemaBuilder()
    built.add_text_field("title", stored=False, tokenizer_name="en_stem")
    built.add_text_field("text", stored=False, tokenizer_name="en_stem")
    built.add_unsigned_field("key", stored=True, indexed=False)
    return built.build()


def _articles(corpus_db: Path, after: int) -> Iterator[tuple[int, str, str]]:
    conn = sqlite3.connect(f"file:{corpus_db}?mode=ro", uri=True, timeout=10.0)
    try:
        cursor = conn.execute("SELECT d.id, d.title, f.body FROM docs AS d JOIN docs_fts AS f ON f.rowid = d.id "
                              "WHERE d.id > ? ORDER BY d.id", (after,))
        yield from cursor
    finally:
        conn.close()


def build(corpus_db: Path, index_dir: Path, *, until: int | None = None, commit_every: int = 20_000,
          threads: int = 4, heap_bytes: int = 1_000_000_000,
          on_progress: Callable[[dict], None] | None = None) -> dict:
    """Index the paragraphs of every article after the last one committed, up to article id ``until``.

    ``commit_every`` articles go in one commit, recorded after it, so a stop
    loses at most that many and a rerun writes none twice.
    """
    import tantivy

    from core.knowledge.passages_about import _paragraphs

    index_dir.mkdir(parents=True, exist_ok=True)
    progress = index_dir / _PROGRESS
    after = int(json.loads(progress.read_text(encoding="utf-8"))["last_doc"]) if progress.exists() else 0
    index = tantivy.Index(_schema(), path=str(index_dir))
    writer = index.writer(heap_size=heap_bytes, num_threads=threads)
    started = time.monotonic()
    articles = paragraphs = refused = 0
    last = after

    def commit() -> None:
        writer.commit()
        atomic_write_text(progress, json.dumps({"last_doc": last}))

    try:
        for doc_id, title, body in _articles(corpus_db, after):
            if until is not None and doc_id > until:
                break
            split = _paragraphs(body)
            if len(split) >= 1 << POSITION_BITS:
                refused += 1
            else:
                for position, text in enumerate(split):
                    writer.add_document(tantivy.Document(title=title, text=text,
                                                         key=(doc_id << POSITION_BITS) | position))
                paragraphs += len(split)
            articles += 1
            last = doc_id
            if articles % commit_every == 0:
                commit()
                if on_progress is not None:
                    on_progress({"articles": articles, "paragraphs": paragraphs, "last_doc": last,
                                 "seconds": round(time.monotonic() - started, 1)})
        commit()
        writer.wait_merging_threads()
    except BaseException:
        writer.rollback()
        raise
    return {"articles": articles, "paragraphs": paragraphs, "refused": refused, "last_doc": last,
            "seconds": round(time.monotonic() - started, 1)}


_OPEN: dict[str, Any] = {}


def _opened(index_dir: Path) -> Any:
    """One open index per directory for the process: opening maps its files, searching does not reopen them."""
    import tantivy

    key = str(index_dir)
    if key not in _OPEN:
        _OPEN[key] = tantivy.Index.open(key)
    return _OPEN[key]


class PassageIndex:
    """BM25 over every paragraph; text read back from corpus.db."""

    def __init__(self, index_dir: Path | str | None = None, corpus_db: Path | str | None = None) -> None:
        from core.knowledge.local_corpus import default_corpus_db_path

        self.index_dir = Path(index_dir) if index_dir else default_passage_index_path()
        self.corpus_db = Path(corpus_db) if corpus_db else default_corpus_db_path()

    def available(self) -> bool:
        try:
            import tantivy  # noqa: F401
        except ImportError:
            return False
        return (self.index_dir / _PROGRESS).exists() and self.corpus_db.exists()

    def search(self, question: str, *, limit: int, deadline_s: float | None = None) -> list[Passage]:
        """The paragraphs whose words best match the question's, best first; [] when it has none to search.

        ``deadline_s`` is accepted for the callers that pass one; a search
        takes milliseconds and is not cut short.
        """
        from core.knowledge.passages_about import _terms

        words = _NOT_A_WORD.sub(" ", " ".join(sorted(_terms(question)))).split()
        if not words or not self.available():
            return []
        try:
            index = _opened(self.index_dir)
            searcher = index.searcher()
            query = index.parse_query(" ".join(words), ["title", "text"])
            hits = searcher.search(query, max(1, int(limit))).hits
            rows = [(int(searcher.doc(address)["key"][0]), float(score)) for score, address in hits]
        except (ValueError, OSError, RuntimeError) as exc:
            # An index being rebuilt or a word the parser refuses: a miss, as the corpus store treats one.
            logger.debug("Passage search returned nothing: %s", exc)
            return []
        return self._read_back(rows)

    def articles_titled(self, name: str, around: list[str], *, limit: int) -> list[int]:
        """Ids of the articles whose title holds every word of ``name``, best first by their paragraphs' match to
        ``around`` (the words said with the name); [] when there are none.

        One search: the name's words are required in the title and the words
        around it rank the paragraphs, so "Kobe" said with "81 point game" puts
        the player's article before the city's.
        """
        named = _NOT_A_WORD.sub(" ", name).split()
        words = _NOT_A_WORD.sub(" ", " ".join(around)).split()
        if not named or not self.available():
            return []
        query_text = " ".join([*(f"+title:{w}" for w in named), *words])
        try:
            index = _opened(self.index_dir)
            searcher = index.searcher()
            hits = searcher.search(index.parse_query(query_text, ["title", "text"]), max(1, int(limit))).hits
            keys = [int(searcher.doc(address)["key"][0]) for _score, address in hits]
        except (ValueError, OSError, RuntimeError) as exc:
            logger.debug("Title search returned nothing: %s", exc)
            return []
        return list(dict.fromkeys(key >> POSITION_BITS for key in keys))

    def _read_back(self, rows: list[tuple[int, float]]) -> list[Passage]:
        from core.knowledge.passages_about import _paragraphs

        mask = (1 << POSITION_BITS) - 1
        conn = sqlite3.connect(f"file:{self.corpus_db}?mode=ro", uri=True, timeout=10.0)
        found: list[Passage] = []
        split: dict[int, tuple[str, list[str]]] = {}
        try:
            for key, score in rows:
                doc_id, position = key >> POSITION_BITS, key & mask
                if doc_id not in split:
                    row = conn.execute("SELECT d.title, f.body FROM docs AS d JOIN docs_fts AS f ON f.rowid = d.id "
                                       "WHERE d.id = ?", (doc_id,)).fetchone()
                    split[doc_id] = (str(row[0]), list(_paragraphs(row[1]))) if row else ("", [])
                title, paragraphs = split[doc_id]
                if position < len(paragraphs):
                    found.append(Passage(title, paragraphs[position], doc_id, position, score))
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


__all__ = ["POSITION_BITS", "Passage", "PassageIndex", "build", "default_passage_index_path", "fused"]
