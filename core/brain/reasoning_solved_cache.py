"""Reasoning solved-cache — memoize verifier-clean derivations.

The expensive part of frontier-competitive local reasoning is the *search*:
generating N candidates, verifying, repairing, judging. That cost is paid once
per hard problem — but the same (or structurally identical) problem recurs. This
cache stores **only verifier-clean answers** keyed by the normalized problem so a
re-encounter is an O(1) lookup instead of a full amplifier run.

This is the lawful "free lunch": we do not break the cost-of-compute constraint,
we *amortize* it. The organism gets faster — and effectively smarter per unit
wall-clock — the longer it runs, because its store of solved derivations grows.

Soundness boundary (important):
  Only *source-independent, closed-form* task types are cached by default
  (``math``, ``code``, ``logic``). Answers that depend on mutable external state
  — ``repo_audit``/``architecture`` (the source tree changes) or ``factual``
  (the world changes) — are NOT cached, so a stale derivation can never be served
  as current truth. Override via ``cacheable_task_types`` only if you know the
  answer is genuinely stable.

Bounded by construction: a max entry count (LRU-ish eviction by last-hit/age) and
a TTL. No unbounded growth.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.ReasoningSolvedCache")

# Source-independent, verifier-clean task types are safe to memoize. Anything that
# depends on the live source tree or the changing world is excluded by default.
# CP126 2e31dd71 also flagged `code` as wrongly source-independent. It is
# kept, but ONLY because the key now carries a context fingerprint: a code
# answer derived under one interpreter no longer answers the same question
# asked under another. Repository questions are a different matter and are
# classified as repo_audit (not cacheable) before they ever reach here.
DEFAULT_CACHEABLE_TASK_TYPES: frozenset[str] = frozenset({"math", "code", "logic"})

# Bump to invalidate every existing key when the key definition changes.
# Old entries simply stop being found, which is the correct outcome: they
# answered a question posed under a definition we no longer use.
_CACHE_KEY_SCHEMA = 2

_DEFAULT_MAX_ENTRIES = 2000
_DEFAULT_TTL_S = 14 * 24 * 3600.0  # two weeks
# Verification is the real gate; we only reject empty/whitespace answers here. A
# correct math answer ("4") is legitimately short and must still be cacheable.
_MIN_ANSWER_CHARS = 1
_MIN_CONFIDENCE = 0.55

_WS_RE = re.compile(r"\s+")


def _normalize_objective(objective: str) -> str:
    """Whitespace/case-normalize so trivially-different phrasings collide."""
    return _WS_RE.sub(" ", str(objective or "").strip().lower())


def _context_fingerprint() -> str:
    """The environment an answer was derived in, and is only valid within.

    CP126 2e31dd71. The key was task_type plus the normalized objective, so
    the same words asked under a different interpreter, a different model or
    a changed verifier hit the SAME entry — replaying a stale or
    cross-context answer as verified truth. "What is the result of this
    code" does not have a context-free answer.

    Cheap and stable inputs only: this runs on every lookup, so it must not
    do I/O or import heavy modules.
    """
    parts = [
        f"py{sys.version_info.major}.{sys.version_info.minor}",
        f"schema{_CACHE_KEY_SCHEMA}",
    ]
    # Model identity, when the runtime has published one. Absent is a valid
    # answer and simply yields a distinct fingerprint from present.
    parts.append("model=" + str(os.environ.get("AURA_ACTIVE_MODEL_ID", "") or "-"))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _problem_key(objective: str, task_type: str) -> str:
    payload = (
        f"{str(task_type or '').strip().lower()}\n"
        f"{_context_fingerprint()}\n"
        f"{_normalize_objective(objective)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class SolvedEntry:
    answer: str
    confidence: float
    mode: str
    task_type: str
    verifiers_run: list[str] = field(default_factory=list)
    # Which required-evidence items the cached derivation actually saw. A
    # later request naming evidence absent from this list is asking a
    # different question and must not be served from here (CP126 236526e0).
    required_evidence: list[str] = field(default_factory=list)
    stored_at: float = field(default_factory=time.time)
    last_hit_at: float = field(default_factory=time.time)
    hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": round(float(self.confidence), 4),
            "mode": self.mode,
            "task_type": self.task_type,
            "verifiers_run": list(self.verifiers_run),
            "required_evidence": list(self.required_evidence),
            "stored_at": self.stored_at,
            "last_hit_at": self.last_hit_at,
            "hits": self.hits,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SolvedEntry":
        return cls(
            answer=str(data.get("answer", "")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            mode=str(data.get("mode", "")),
            task_type=str(data.get("task_type", "")),
            verifiers_run=list(data.get("verifiers_run", []) or []),
            # Absent on entries written before this field existed. Empty
            # means "covered nothing", so any evidence requirement misses —
            # the safe direction for a stale entry.
            required_evidence=list(data.get("required_evidence", []) or []),
            stored_at=float(data.get("stored_at", time.time()) or time.time()),
            last_hit_at=float(data.get("last_hit_at", time.time()) or time.time()),
            hits=int(data.get("hits", 0) or 0),
        )


class ReasoningSolvedCache:
    """Bounded, persistent, thread-safe store of verifier-clean derivations."""

    _CACHE_ERRORS = (OSError, ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError)

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        ttl_s: float = _DEFAULT_TTL_S,
        cacheable_task_types: frozenset[str] = DEFAULT_CACHEABLE_TASK_TYPES,
    ) -> None:
        self._path = Path(
            path
            or str(state_root() / "data/runtime/reasoning_solved_cache.json")
        )
        self._max_entries = max(16, int(max_entries))
        self._ttl_s = max(60.0, float(ttl_s))
        self._cacheable = frozenset(cacheable_task_types)
        self._lock = threading.RLock()
        self._entries: dict[str, SolvedEntry] = {}
        self._stats = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0, "skipped": 0}
        self._load()

    # ── public API ────────────────────────────────────────────────────────
    def is_cacheable(self, task_type: str) -> bool:
        return str(task_type or "").strip().lower() in self._cacheable

    def get(self, objective: str, task_type: str) -> SolvedEntry | None:
        if not self.is_cacheable(task_type):
            return None
        key = _problem_key(objective, task_type)
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if now - entry.stored_at > self._ttl_s:
                # Expired — drop it and miss.
                self._entries.pop(key, None)
                self._stats["misses"] += 1
                return None
            entry.hits += 1
            entry.last_hit_at = now
            self._stats["hits"] += 1
            return entry

    def put(
        self,
        objective: str,
        task_type: str,
        *,
        answer: str,
        confidence: float,
        mode: str,
        verifiers_run: list[str] | None = None,
        required_evidence: list[str] | None = None,
        verified: bool,
    ) -> bool:
        """Store a verifier-clean answer. Returns True if stored.

        Refuses to store: non-cacheable task types, unverified answers, low
        confidence, or trivially short answers — a wrong cached answer is worse
        than no cache, so the bar to enter is deliberately high.
        """
        if not verified:
            self._stats["skipped"] += 1
            return False
        if not self.is_cacheable(task_type):
            self._stats["skipped"] += 1
            return False
        clean = str(answer or "").strip()
        if len(clean) < _MIN_ANSWER_CHARS or float(confidence) < _MIN_CONFIDENCE:
            self._stats["skipped"] += 1
            return False
        key = _problem_key(objective, task_type)
        now = time.time()
        with self._lock:
            self._entries[key] = SolvedEntry(
                answer=clean,
                confidence=float(confidence),
                mode=str(mode or ""),
                task_type=str(task_type or "").strip().lower(),
                verifiers_run=list(verifiers_run or []),
                required_evidence=[
                    str(item).strip().lower()
                    for item in (required_evidence or [])
                    if str(item).strip()
                ],
                stored_at=now,
                last_hit_at=now,
                hits=0,
            )
            self._stats["stores"] += 1
            self._evict_if_needed()
            self._persist()
        return True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total) if total else 0.0
            return {**self._stats, "entries": len(self._entries), "hit_rate": round(hit_rate, 4)}

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._persist()

    # ── internals ─────────────────────────────────────────────────────────
    def _evict_if_needed(self) -> None:
        # Caller holds the lock. Drop expired first, then LRU by last_hit_at.
        now = time.time()
        expired = [k for k, e in self._entries.items() if now - e.stored_at > self._ttl_s]
        for k in expired:
            self._entries.pop(k, None)
            self._stats["evictions"] += 1
        while len(self._entries) > self._max_entries:
            victim = min(self._entries.items(), key=lambda kv: kv[1].last_hit_at)[0]
            self._entries.pop(victim, None)
            self._stats["evictions"] += 1

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            items = raw.get("entries", {}) if isinstance(raw, dict) else {}
            now = time.time()
            with self._lock:
                for key, data in items.items():
                    try:
                        entry = SolvedEntry.from_dict(data)
                    except self._CACHE_ERRORS:
                        continue
                    if now - entry.stored_at <= self._ttl_s:
                        self._entries[key] = entry
                self._evict_if_needed()
        except self._CACHE_ERRORS as exc:
            record_degradation("reasoning_solved_cache_load", exc)
            logger.debug("Solved-cache load failed (starting empty): %s", exc)

    def _persist(self) -> None:
        # Caller holds the lock.
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "saved_at": time.time(),
                "entries": {k: e.to_dict() for k, e in self._entries.items()},
            }
            fd, tmp = tempfile.mkstemp(
                prefix=".solved_cache_", suffix=".json", dir=str(self._path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                os.replace(tmp, self._path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except self._CACHE_ERRORS as exc:
            record_degradation("reasoning_solved_cache_persist", exc)
            logger.debug("Solved-cache persist failed (kept in memory): %s", exc)


_cache_singleton: ReasoningSolvedCache | None = None
_singleton_lock = threading.Lock()


def get_reasoning_solved_cache() -> ReasoningSolvedCache:
    """Process-wide solved-cache singleton."""
    global _cache_singleton
    if _cache_singleton is None:
        with _singleton_lock:
            if _cache_singleton is None:
                _cache_singleton = ReasoningSolvedCache()
    return _cache_singleton


def reset_reasoning_solved_cache() -> None:
    """Test hook: drop the singleton so the next call rebuilds it."""
    global _cache_singleton
    with _singleton_lock:
        _cache_singleton = None
