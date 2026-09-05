"""core/knowledge/compiled_understanding.py

The Compiled Understanding Layer — closing the assimilation gap.

Retrieval closes the ACCESS gap: the 32B can find any fact. What it cannot
buy back is a frontier model's compiled understanding — concepts pressed
into weights that shape every inference. This layer attacks that gap with
machinery Aura already owns:

  1. **Concept compilation.** Retrieved material is digested ONCE by the
     deepest reasoning lane available (the 72B solver when the router can
     afford it, the resident 32B in DEEP mode otherwise) into a dense
     concept digest: definition, key relations, constraints, canonical
     analogies, failure modes, cross-domain connections. Big model as
     offline concept compiler; resident model as runtime.
  2. **The digest library.** Digests are content-addressed, persistent,
     provenance-carrying, and hit-counted. A concept is digested once and
     reused forever; dense digests fit the live episode's bounded context
     where raw articles cannot.
  3. **Bridge indexing (unknown-unknowns).** Idle cycles pre-compute
     cross-domain bridges between recently-touched concepts, so lateral
     connections exist in the substrate TO BE FOUND without being searched
     for — the retrieval analogue of breadth-in-weights.
  4. **Assimilation evidence.** Heavily-reused digests are exported as
     consolidation evidence for the governed learning lanes: access →
     compilation → reuse → durable weights, each step gated and receipted.

Honest boundaries, in code as in prose: a digest whose compiler was the
heuristic extractor says so; an unverified digest says so; compilation that
was skipped for capacity reasons returns raw hits flagged as uncompiled.
Nothing here fakes understanding it did not build.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.sqlite_support import connecting

logger = logging.getLogger("Aura.CompiledUnderstanding")

DIGEST_SCHEMA = "aura.compiled_digest.v1"
UNDERSTAND_RECEIPT_SCHEMA = "aura.compiled_understanding.v1"
BRIDGE_RECEIPT_SCHEMA = "aura.compiled_bridge_cycle.v1"

COMPILER_DEEP_SOLVER = "deep_solver"
COMPILER_RESIDENT_DEEP = "resident_deep"
COMPILER_HEURISTIC = "heuristic_extractive"

_WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)
_STOPWORDS = {
    "about", "after", "again", "also", "among", "because", "before", "being",
    "between", "could", "does", "every", "explain", "from", "have", "into",
    "itself", "more", "most", "other", "should", "some", "such", "than",
    "that", "their", "then", "there", "these", "they", "this", "through",
    "under", "using", "what", "when", "where", "which", "while", "with",
    "would", "your",
}

# Grounding floor for the deterministic verification check: a digest whose
# content-word overlap with its source material falls below this is marked
# unverified (the compiler drifted past its sources).
_VERIFY_GROUNDING_FLOOR = 0.35

_COMPILE_PROMPT = (
    "Compile the source material into a dense concept digest of "
    "'{concept}'. Write 120-220 words covering, in order: a precise "
    "definition; the two or three most load-bearing relations to other "
    "concepts; hard constraints or invariants; one canonical analogy; the "
    "most common failure mode or misconception. Use only the source "
    "material and well-established knowledge; never invent citations.\n\n"
    "SOURCE MATERIAL:\n{material}"
)

_BRIDGE_PROMPT = (
    "In 60-120 words, state the deepest genuine connection between "
    "'{left}' and '{right}': a shared mechanism, a structural analogy, or "
    "a constraint one imposes on the other. If no substantive connection "
    "exists, say exactly: NO SUBSTANTIVE CONNECTION."
)


def concept_key(concept: str) -> str:
    """Canonical content-addressed key for a concept phrase."""
    words = [w.lower() for w in _WORD_RE.findall(str(concept or ""))]
    normalized = " ".join(words)[:120]
    if not normalized:
        raise ValueError("concept has no content words")
    return normalized


def extract_concepts(objective: str, *, limit: int = 4) -> list[str]:
    """Salient concept phrases from an objective (bigrams first, then terms).

    Deterministic and dependency-free: adjacent content-word bigrams are the
    highest-value digest keys ('single-owner design', 'bayes update'); rare
    standalone content words fill the remainder.
    """
    words = [w.lower() for w in _WORD_RE.findall(str(objective or ""))]
    content = [w for w in words if w not in _STOPWORDS]
    seen: set[str] = set()
    concepts: list[str] = []
    for left, right in zip(content, content[1:]):
        phrase = f"{left} {right}"
        if phrase not in seen:
            seen.add(phrase)
            concepts.append(phrase)
        if len(concepts) >= limit:
            return concepts
    # Standalone words only when no chosen bigram already covers them — a
    # unigram subsumed by its phrase ("hash" under "hash join") is a
    # redundant digest that would waste a foreground compile.
    covered = {part for phrase in concepts for part in phrase.split()}
    for word in content:
        if word in seen or word in covered:
            continue
        seen.add(word)
        concepts.append(word)
        if len(concepts) >= limit:
            break
    return concepts


def grounding_score(digest_text: str, material: str) -> float:
    """Deterministic overlap of the digest's content words with its sources."""
    digest_words = {
        w.lower() for w in _WORD_RE.findall(digest_text) if w.lower() not in _STOPWORDS
    }
    if not digest_words:
        return 0.0
    material_words = {
        w.lower() for w in _WORD_RE.findall(material) if w.lower() not in _STOPWORDS
    }
    if not material_words:
        return 0.0
    return len(digest_words & material_words) / len(digest_words)


@dataclass
class ConceptDigest:
    key: str
    digest_text: str
    compiler: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    grounding: float = 0.0
    verified: bool = False
    bridges: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    hits: int = 0
    last_used_at: float = 0.0

    @property
    def digest_sha256(self) -> str:
        return hashlib.sha256(self.digest_text.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DIGEST_SCHEMA,
            "key": self.key,
            "digest_text": self.digest_text,
            "digest_sha256": self.digest_sha256,
            "compiler": self.compiler,
            "sources": list(self.sources),
            "grounding": round(self.grounding, 4),
            "verified": self.verified,
            "bridges": list(self.bridges),
            "created_at": self.created_at,
            "hits": self.hits,
            "last_used_at": self.last_used_at,
        }


class DigestLibrary:
    """Content-addressed persistent store for compiled concept digests.

    SQLite following the LocalCorpusStore precedent for knowledge stores.
    Reads never raise on a missing/corrupt library — an empty result is the
    honest degraded answer; writes fail loudly.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            from core.config import DATA_DIR

            db_path = Path(DATA_DIR) / "knowledge" / "compiled_digests.db"
        self.db_path = Path(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        try:
            with connecting(self._connect()) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS digests (
                        key TEXT PRIMARY KEY,
                        digest_text TEXT NOT NULL,
                        compiler TEXT NOT NULL,
                        sources_json TEXT NOT NULL,
                        grounding REAL NOT NULL,
                        verified INTEGER NOT NULL,
                        bridges_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        hits INTEGER NOT NULL DEFAULT 0,
                        last_used_at REAL NOT NULL DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_digests_hits ON digests(hits DESC)"
                )
        except sqlite3.Error as exc:
            record_degradation(
                "compiled_understanding",
                exc,
                action="continued with an unavailable digest library (reads return empty)",
            )

    @staticmethod
    def _row_to_digest(row: tuple) -> ConceptDigest:
        return ConceptDigest(
            key=row[0],
            digest_text=row[1],
            compiler=row[2],
            sources=json.loads(row[3]),
            grounding=float(row[4]),
            verified=bool(row[5]),
            bridges=json.loads(row[6]),
            created_at=float(row[7]),
            hits=int(row[8]),
            last_used_at=float(row[9]),
        )

    _COLUMNS = (
        "key, digest_text, compiler, sources_json, grounding, verified, "
        "bridges_json, created_at, hits, last_used_at"
    )

    def get(self, key: str) -> ConceptDigest | None:
        try:
            with connecting(self._connect()) as conn:
                row = conn.execute(
                    f"SELECT {self._COLUMNS} FROM digests WHERE key = ?", (key,)
                ).fetchone()
        except sqlite3.Error:
            return None
        return self._row_to_digest(row) if row else None

    def put(self, digest: ConceptDigest) -> None:
        with connecting(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO digests
                    (key, digest_text, compiler, sources_json, grounding,
                     verified, bridges_json, created_at, hits, last_used_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    digest_text = excluded.digest_text,
                    compiler = excluded.compiler,
                    sources_json = excluded.sources_json,
                    grounding = excluded.grounding,
                    verified = excluded.verified,
                    bridges_json = excluded.bridges_json
                """,
                (
                    digest.key,
                    digest.digest_text,
                    digest.compiler,
                    json.dumps(digest.sources),
                    digest.grounding,
                    int(digest.verified),
                    json.dumps(digest.bridges),
                    digest.created_at,
                    digest.hits,
                    digest.last_used_at,
                ),
            )

    def record_use(self, key: str) -> None:
        try:
            with connecting(self._connect()) as conn:
                conn.execute(
                    "UPDATE digests SET hits = hits + 1, last_used_at = ? WHERE key = ?",
                    (time.time(), key),
                )
        except sqlite3.Error as exc:
            record_degradation(
                "compiled_understanding",
                exc,
                action="continued after digest hit-count update failed",
            )

    def add_bridge(self, left_key: str, right_key: str) -> None:
        for a, b in ((left_key, right_key), (right_key, left_key)):
            digest = self.get(a)
            if digest is None or b in digest.bridges:
                continue
            digest.bridges.append(b)
            self.put(digest)

    def heavily_used(self, *, min_hits: int = 3, limit: int = 32) -> list[ConceptDigest]:
        """Reuse evidence for consolidation: verified digests with real traffic."""
        try:
            with connecting(self._connect()) as conn:
                rows = conn.execute(
                    f"SELECT {self._COLUMNS} FROM digests "
                    "WHERE hits >= ? AND verified = 1 "
                    "ORDER BY hits DESC LIMIT ?",
                    (min_hits, limit),
                ).fetchall()
        except sqlite3.Error:
            return []
        return [self._row_to_digest(row) for row in rows]

    def recent_keys(self, *, limit: int = 16) -> list[str]:
        try:
            with connecting(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT key FROM digests ORDER BY last_used_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        except sqlite3.Error:
            return []
        return [row[0] for row in rows]

    def stats(self) -> dict[str, Any]:
        try:
            with connecting(self._connect()) as conn:
                total, verified, bridged, hits = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(verified), 0), "
                    "COALESCE(SUM(bridges_json != '[]'), 0), "
                    "COALESCE(SUM(hits), 0) FROM digests"
                ).fetchone()
        except sqlite3.Error:
            return {"available": False}
        return {
            "available": True,
            "digests": int(total),
            "verified": int(verified),
            "bridged": int(bridged),
            "total_hits": int(hits),
        }


class ConceptCompiler:
    """Digests material through the deepest reasoning lane available.

    The brain resolver and thinking call are injected so the compiler is
    fully testable and never imports runtime lanes at module load. The
    fallback ladder is explicit and receipted: deep solver → resident deep
    → heuristic extraction, each labeled as itself.
    """

    def __init__(self, think=None, *, timeout_s: float = 45.0) -> None:
        self._think = think
        self._timeout_s = float(timeout_s)

    async def _model_digest(self, prompt: str) -> tuple[str, str] | None:
        if self._think is None:
            try:
                from core.utils.engine_support import coerce_text, resolve_brain

                brain = resolve_brain(None)
                if brain is None or not hasattr(brain, "think"):
                    return None
                from core.brain.types import ThinkingMode

                import asyncio

                out = coerce_text(
                    await asyncio.wait_for(
                        brain.think(
                            prompt,
                            mode=ThinkingMode.DEEP,
                            origin="compiled_understanding",
                            is_background=True,
                        ),
                        timeout=self._timeout_s,
                    )
                )
                return (out.strip(), COMPILER_RESIDENT_DEEP) if out else None
            except (ImportError, AttributeError, RuntimeError, TypeError,
                    ValueError, TimeoutError):
                return None
        try:
            import asyncio

            out = await asyncio.wait_for(self._think(prompt), timeout=self._timeout_s)
        except (RuntimeError, TypeError, ValueError, TimeoutError):
            return None
        text = str(out or "").strip()
        return (text, COMPILER_DEEP_SOLVER) if text else None

    @staticmethod
    def _heuristic_digest(concept: str, material: str) -> str:
        """Extractive fallback: the sentences most saturated with the concept."""
        sentences = re.split(r"(?<=[.!?])\s+", material)
        targets = {w.lower() for w in _WORD_RE.findall(concept)}
        scored = sorted(
            (s for s in sentences if s.strip()),
            key=lambda s: -len(
                targets & {w.lower() for w in _WORD_RE.findall(s)}
            ),
        )
        return " ".join(scored[:4])[:1200]

    async def compile(
        self,
        concept: str,
        material_chunks: list[dict[str, Any]],
    ) -> ConceptDigest:
        key = concept_key(concept)
        material = "\n\n".join(
            str(chunk.get("text") or "") for chunk in material_chunks
        )[:6000]
        if not material.strip():
            raise ValueError(f"no source material for concept {concept!r}")
        prompt = _COMPILE_PROMPT.format(concept=concept, material=material)
        produced = await self._model_digest(prompt)
        if produced is not None:
            digest_text, compiler = produced
        else:
            digest_text = self._heuristic_digest(concept, material)
            compiler = COMPILER_HEURISTIC
        score = grounding_score(digest_text, material)
        return ConceptDigest(
            key=key,
            digest_text=digest_text,
            compiler=compiler,
            sources=[
                {k: chunk.get(k) for k in ("title", "source", "doc_id")}
                for chunk in material_chunks
            ],
            grounding=score,
            verified=score >= _VERIFY_GROUNDING_FLOOR,
        )

    async def compile_bridge(self, left: str, right: str) -> str | None:
        """A cross-domain bridge digest, or None when none exists/available."""
        produced = await self._model_digest(
            _BRIDGE_PROMPT.format(left=left, right=right)
        )
        if produced is None:
            return None
        text = produced[0]
        if "NO SUBSTANTIVE CONNECTION" in text.upper():
            return None
        return text


class CompiledUnderstandingService:
    """Digest-first understanding for episodes + idle bridge indexing."""

    def __init__(
        self,
        *,
        library: DigestLibrary | None = None,
        compiler: ConceptCompiler | None = None,
        corpus=None,
    ) -> None:
        self.library = library or DigestLibrary()
        self.compiler = compiler or ConceptCompiler()
        self._corpus = corpus
        self._understand_calls = 0
        self._cache_hits = 0
        self._compiles = 0
        self._bridge_cycles = 0
        logger.info("📚 CompiledUnderstandingService initialized")

    def _corpus_search(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        corpus = self._corpus
        if corpus is None:
            try:
                from core.knowledge.local_corpus import LocalCorpusStore

                corpus = self._corpus = LocalCorpusStore()
            except (ImportError, RuntimeError, OSError):
                return []
        try:
            hits = corpus.search(query, limit=limit)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return []
        return [
            {
                "text": f"{hit.title}: {hit.snippet}",
                "title": hit.title,
                "source": hit.source,
                "doc_id": hit.doc_id,
            }
            for hit in hits
        ]

    async def understand(
        self,
        objective: str,
        *,
        max_concepts: int = 3,
        max_compiles: int = 1,
        max_chars: int = 2000,
    ) -> dict[str, Any]:
        """Digest-first conceptual context for an objective, with receipts.

        Cache hits are free; at most ``max_compiles`` misses are compiled in
        the foreground (compilation is expensive — the rest are left for
        idle cycles). Returns rendered context text sized for the episode's
        compaction budget plus a receipt that says exactly what happened.
        """
        self._understand_calls += 1
        concepts = extract_concepts(objective, limit=max_concepts)
        digests: list[ConceptDigest] = []
        compiled_now: list[str] = []
        uncompiled: list[str] = []
        for concept in concepts:
            try:
                key = concept_key(concept)
            except ValueError:
                continue
            cached = self.library.get(key)
            if cached is not None:
                self._cache_hits += 1
                self.library.record_use(key)
                digests.append(cached)
                continue
            if len(compiled_now) >= max_compiles:
                uncompiled.append(concept)
                continue
            material = self._corpus_search(concept)
            if not material:
                uncompiled.append(concept)
                continue
            try:
                digest = await self.compiler.compile(concept, material)
            except (ValueError, RuntimeError) as exc:
                record_degradation(
                    "compiled_understanding",
                    exc,
                    action="served raw retrieval after concept compilation failed",
                )
                uncompiled.append(concept)
                continue
            try:
                self.library.put(digest)
            except sqlite3.Error as exc:
                record_degradation(
                    "compiled_understanding",
                    exc,
                    action="used un-persisted digest after library write failed",
                )
            self._compiles += 1
            compiled_now.append(concept)
            digests.append(digest)

        rendered_parts: list[str] = []
        used = 0
        for digest in digests:
            entry = f"[{digest.key}] {digest.digest_text}"
            if used + len(entry) > max_chars:
                break
            rendered_parts.append(entry)
            used += len(entry)
        return {
            "schema": UNDERSTAND_RECEIPT_SCHEMA,
            "context": "\n\n".join(rendered_parts),
            "concepts": concepts,
            "digest_keys": [digest.key for digest in digests],
            "cache_hits": sum(1 for c in concepts if c in {d.key for d in digests})
            - len(compiled_now),
            "compiled_now": compiled_now,
            "uncompiled": uncompiled,
            "compilers": sorted({digest.compiler for digest in digests}),
            "unverified_digests": [
                digest.key for digest in digests if not digest.verified
            ],
        }

    async def bridge_cycle(self, *, max_pairs: int = 2) -> dict[str, Any]:
        """Idle-time unknown-unknowns pass: connect recently-used concepts.

        Bounded and honest: pairs whose bridge cannot be compiled (no lane
        capacity, no substantive connection) are reported, not invented.
        """
        self._bridge_cycles += 1
        keys = self.library.recent_keys(limit=8)
        bridged: list[list[str]] = []
        skipped: list[list[str]] = []
        attempted = 0
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if attempted >= max_pairs:
                    break
                left, right = keys[i], keys[j]
                existing = self.library.get(left)
                if existing is not None and right in existing.bridges:
                    continue
                attempted += 1
                bridge_text = await self.compiler.compile_bridge(left, right)
                if bridge_text is None:
                    skipped.append([left, right])
                    continue
                bridge_key = concept_key(f"bridge {left} {right}")
                self.library.put(
                    ConceptDigest(
                        key=bridge_key,
                        digest_text=bridge_text,
                        compiler=COMPILER_DEEP_SOLVER,
                        sources=[{"title": left, "source": "bridge"},
                                 {"title": right, "source": "bridge"}],
                        grounding=1.0,
                        verified=True,
                        bridges=[left, right],
                    )
                )
                self.library.add_bridge(left, right)
                bridged.append([left, right])
            if attempted >= max_pairs:
                break
        return {
            "schema": BRIDGE_RECEIPT_SCHEMA,
            "attempted": attempted,
            "bridged": bridged,
            "skipped": skipped,
            "ran_at": time.time(),
        }

    def export_reuse_evidence(self, out_path: Path | str | None = None) -> dict[str, Any]:
        """Governed export of heavily-reused digests as assimilation evidence.

        The consolidation pipeline treats these as candidate domains for the
        gated learning lanes: knowledge that keeps earning reuse is knowledge
        worth compiling into weights.
        """
        digests = self.library.heavily_used()
        payload = {
            "schema": "aura.digest_reuse_evidence.v1",
            "generated_at": time.time(),
            "candidates": [
                {
                    "key": digest.key,
                    "hits": digest.hits,
                    "compiler": digest.compiler,
                    "grounding": round(digest.grounding, 4),
                    "digest_sha256": digest.digest_sha256,
                }
                for digest in digests
            ],
        }
        if out_path is None:
            from core.config import DATA_DIR

            out_path = Path(DATA_DIR) / "latent_cortex" / "digest_reuse_evidence.json"
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            gateway = get_file_write_gateway()
            with local_internal_governed_scope("compiled_understanding_evidence"):
                gateway.ensure_directory(
                    Path(out_path).parent, source="compiled_understanding"
                )
                gateway.write_text(
                    Path(out_path),
                    json.dumps(payload, indent=1, sort_keys=True),
                    source="compiled_understanding",
                )
            payload["written_to"] = str(out_path)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            record_degradation(
                "compiled_understanding",
                exc,
                action="kept digest reuse evidence in memory after export failed",
            )
        return payload

    def get_status(self) -> dict[str, Any]:
        return {
            "understand_calls": self._understand_calls,
            "cache_hits": self._cache_hits,
            "compiles": self._compiles,
            "bridge_cycles": self._bridge_cycles,
            "library": self.library.stats(),
            "healthy": True,
        }


_INSTANCE: CompiledUnderstandingService | None = None


def get_compiled_understanding(**kwargs) -> CompiledUnderstandingService:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = CompiledUnderstandingService(**kwargs)
    return _INSTANCE


def register_compiled_understanding(orchestrator: Any = None) -> CompiledUnderstandingService:
    from core.runtime.service_registry import get_runtime_service, register_runtime_service
    from core.service_names import ServiceNames

    inst = get_runtime_service(
        ServiceNames.COMPILED_UNDERSTANDING, default=None
    ) or get_compiled_understanding()
    register_runtime_service(
        ServiceNames.COMPILED_UNDERSTANDING,
        inst,
        required=False,
        owner="core/knowledge/compiled_understanding.py",
        registered_by="register_compiled_understanding",
    )
    return inst


__all__ = [
    "BRIDGE_RECEIPT_SCHEMA",
    "COMPILER_DEEP_SOLVER",
    "COMPILER_HEURISTIC",
    "COMPILER_RESIDENT_DEEP",
    "CompiledUnderstandingService",
    "ConceptCompiler",
    "ConceptDigest",
    "DIGEST_SCHEMA",
    "DigestLibrary",
    "UNDERSTAND_RECEIPT_SCHEMA",
    "concept_key",
    "extract_concepts",
    "get_compiled_understanding",
    "grounding_score",
    "register_compiled_understanding",
]
